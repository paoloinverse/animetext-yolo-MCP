"""The main window: controls on the left, input preview above the output, scrollable.

Layout follows PLAN §3 Part 5: the whole central body sits in a `QScrollArea`, so a 3000 px
scan at 100 % zoom scrolls both ways instead of being squeezed into nothing. The window
talks to **`JobManager` only** (through `JobBridge`): it never calls an engine, never
starts a thread, and derives its busy state from the job signals rather than from
statements around a call (see `gui/workers.py`).

The output area (Part 6) is the annotated `ResultCanvas` beside the block table and the
JSON pane, all three synced on the block id. `set_result()` / `clear_result()` are the only
ways a result enters the window — the headless path, a Generate click and (in P7) an MCP
`run_*` call all end up in the same two methods.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMenu, QProgressBar, QPushButton, QScrollArea, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .. import APP_NAME, __version__
from ..core.annotation import (CONTAINMENT_MARK, FLAG_MARKS, flag_marks, flag_note,
                               flagged_count)
from ..core.config import (
    KNOWN_MODELS, AppConfig, changed_config_keys, load_config_file, restart_required,
    save_config_file,
)
from ..core.imaging import describe, image_from_bytes, load_image, pil_to_qimage
from ..core.jobs import JobManager
from ..core.model_store import ModelStore
from ..core.pipeline import Pipeline
from ..core.types import ErrorCode, ImageInfo, JobState, PipelineResult, StudioError
from ..engines.base import engines_badge_text, model_catalog, models_badge_text
from .canvas import ResultCanvas
from .controls import ControlPanel
from .json_pane import JsonPane
from .workers import JobBridge

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def _is_image_path(path: Union[str, Path]) -> bool:
    return Path(str(path)).suffix.lower() in IMAGE_SUFFIXES


class MainWindow(QMainWindow):
    """Interactive shell: pick an image, press Generate, watch the job."""

    result_ready = Signal(object)        # PipelineResult (P6 listens to draw it)
    #: (names, error) from the model-list worker. The thread must not touch a widget, so it
    #: reports through this and the window repaints on its own thread (P9.2).
    model_list_ready = Signal(list, str)

    #: The result-1.2 observations, spelled out in one place (P8.6) so the table header and
    #: the summary tooltip cannot drift from what `flag_marks` actually paints.
    _flags_legend = (
        f"{FLAG_MARKS['possible_truncation']}  possible_truncation \u2014 the text may stop at "
        "max_new_tokens\n"
        f"{CONTAINMENT_MARK}n  contained_in block n \u2014 a larger box holds this one\n"
        "Both are also in the JSON document; nothing here changes it."
    )

    def __init__(self, manager: JobManager, config: Optional[AppConfig] = None, *,
                 mock: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnimeTextStudioMainWindow")
        self._manager = manager
        #: True between `start_preload()` and its finish (P8.3) — guards double preloads.
        self._preloading = False
        self._config = config or AppConfig.default()
        self._mock = mock
        self._image = None
        self._info: Optional[ImageInfo] = None
        self._payload: tuple = ()
        self._result = None
        self._syncing = False           # re-entrancy guard for the three-way selection sync

        self.setWindowTitle(f"{APP_NAME} {__version__}" + ("  [mock]" if mock else ""))
        self.resize(1280, 860)          # the layout target from PLAN; splitters do the rest
        self.setAcceptDrops(True)
        self._bridge = JobBridge(manager, self)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.finished.connect(self._on_finished)
        self._bridge.failed.connect(self._on_failed)
        self._bridge.cancelled.connect(self._on_cancelled)
        self._bridge.warmup_done.connect(self._on_warmup_done)

        self._build_body()
        self._build_toolbar()
        self._build_statusbar()
        # P9.2: the listing worker reports through this signal — it must not touch a widget from
        # its own thread, and the window is the only thing that knows which widgets exist.
        self.model_list_ready.connect(self.on_model_list)

        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.paste_from_clipboard)
        self.panel.set_models(self._startup_catalog(), offline=bool(self._config.paths.offline))
        self.panel.apply_config(self._config)
        self._apply_view_settings()
        self._set_busy(False)
        self.refresh_engine_badge()
        self._say("Drop an image, press Ctrl+V, or use Browse. Then Generate.")

    # -- public API (tests now, the MCP bridge in P7) ------------------------

    @property
    def controls(self) -> ControlPanel:
        """The control panel (P7 sets/reads widgets through it)."""
        return self.panel

    @property
    def job_bridge(self) -> JobBridge:
        return self._bridge

    @property
    def job_manager(self) -> JobManager:
        return self._manager

    def refresh_engine_badge(self) -> None:
        """Repaint the engines badge from the engines themselves (P8).

        Called at the two moments the answer can change: when a job ends, and when an MCP
        `unload_engines` call empties them. The window never owns an engine — only the
        `JobManager` does — so it asks instead of remembering, and never raises: a badge
        must not be able to break a run.
        """
        try:
            detection = self._manager.detection.status()
            text = engines_badge_text(detection, self._manager.ocr.status())
            # Same status object, second question: the engines may be perfectly ready while the
            # list of variants they offer is a fallback (P9.2).
            models = models_badge_text(detection)
            detail = str(getattr(detection, "detail", "") or "")
        except Exception:                                          # noqa: BLE001
            text, models, detail = "engines: ?", "", ""
        self.panel.set_engine_status(text)
        self.panel.set_model_list_status(models, detail)

    @property
    def has_image(self) -> bool:
        return self._image is not None

    @property
    def is_mock(self) -> bool:
        return self._mock

    @property
    def canvas(self) -> ResultCanvas:
        """The annotated view (P7: `render_annotated_image` reads its style and export)."""
        return self._canvas

    @property
    def json_pane(self) -> JsonPane:
        """The JSON view; `copy_all()` is what the MCP document tools mirror."""
        return self._json_pane

    @property
    def current_result(self):
        return self._result

    @property
    def image(self):
        """The loaded PIL image (P7: the MCP layer reads the page through here)."""
        return self._image

    @property
    def image_info(self) -> Optional[ImageInfo]:
        """Where those pixels came from (`ImageInfo`, or None before the first load)."""
        return self._info

    @property
    def config(self) -> AppConfig:
        """The live config object. Write it with `set_config()` only."""
        return self._config

    # -- construction -------------------------------------------------------

    #: The listing thread, while it is running (P9.2). Class-level default so the window is
    #: correct before `__init__` reaches the toolbar.
    _listing_worker: Optional[threading.Thread] = None

    def _startup_catalog(self) -> list[dict]:
        """Model list from the on-disk bundle — no network at window startup (D27)."""
        return self._catalog_for(list(KNOWN_MODELS))

    def _catalog_for(self, names: list[str]) -> list[dict]:
        """The same catalog, for any list of names: which of them are on disk is the store's."""
        store = ModelStore(self._config.paths.models_dir, offline=self._config.paths.offline)
        bundled = set(store.bundled_detection_variants())
        sources = {n: ("bundled" if n in bundled else "hf") for n in names}
        return model_catalog(list(names), sources)

    def refresh_model_list(self) -> None:
        """Ask the repo for its variants again — a user action, never a startup one (P9.2).

        D27 keeps the network out of window construction, which is right, and leaves a window
        opened on a train holding the preset list forever. This is the missing half: one bounded
        listing (the engine already caps it at 10 s), on a worker thread, reported whichever way
        it goes. Nothing is downloaded and no file is replaced — only the *names* are re-read, and
        the fifth badge repaints from the answer.
        """
        if self._listing_worker is not None and self._listing_worker.is_alive():
            self.status_label.setText("model list: refresh already running")
            return
        self.refresh_models_button.setEnabled(False)
        self.status_label.setText("model list: asking the repo…")

        def work() -> None:
            try:
                names, error = list(self._manager.detection.refresh_models()), ""
            except Exception as exc:                                     # noqa: BLE001
                names, error = [], f"{type(exc).__name__}: {exc}"
            self.model_list_ready.emit(names, error)

        self._listing_worker = threading.Thread(target=work, daemon=True, name="model-list")
        self._listing_worker.start()

    def on_model_list(self, names: list[str], error: str) -> None:
        """Repaint the combo and the badge from a finished listing (runs on the UI thread)."""
        self._listing_worker = None
        self.refresh_models_button.setEnabled(True)
        status = None
        try:
            status = self._manager.detection.status()
        except Exception:                                                # noqa: BLE001
            pass
        if error:
            self.status_label.setText(f"model list: refresh failed — {error}")
        else:
            wanted = self.panel.model_combo.currentText()
            self.panel.set_models(self._catalog_for(names or list(KNOWN_MODELS)),
                                  offline=bool(self._config.paths.offline))
            index = self.panel.model_combo.findText(wanted)
            if index >= 0:                       # a refresh must not silently change the selection
                self.panel.model_combo.setCurrentIndex(index)
            self.status_label.setText(
                f"model list: {len(names)} variants" + ("" if names else " (none answered)"))
        self.refresh_engine_badge()              # the badge is the answer to what just happened
        if status is not None and not error and not getattr(status, "model_list_is_fallback", False):
            self.status_label.setText(self.status_label.text() + " from the repo")

    def _build_body(self) -> None:
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setCentralWidget(self.scroll)

        body = QWidget(self.scroll)
        self.scroll.setWidget(body)
        row = QHBoxLayout(body)

        self.panel = ControlPanel(self._config, body, mock=self._mock)
        self.panel.setMinimumWidth(300)
        self.panel.setMaximumWidth(380)
        self.panel.generate_requested.connect(self.run_now)
        self.panel.cancel_requested.connect(self.cancel)
        self.panel.export_requested.connect(self.export_config)
        self.panel.import_requested.connect(self.import_config)
        # The panel owns the `gui.*` widgets, the canvas draws them: one connection, so a
        # tick, a slider drag or an imported config all reach the canvas the same way.
        self.panel.view_changed.connect(self._apply_view_settings)
        row.addWidget(self.panel)

        self.splitter = QSplitter(Qt.Orientation.Vertical, body)
        row.addWidget(self.splitter, 1)
        self.splitter.addWidget(self._input_group())
        self.splitter.addWidget(self._output_group())
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)

    def _input_group(self) -> QWidget:
        box = QGroupBox("Input", self.splitter)
        layout = QVBoxLayout(box)
        self.image_scroll = QScrollArea(box)
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QLabel("no image", self.image_scroll)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(220)
        self.image_scroll.setWidget(self.image_label)
        layout.addWidget(self.image_scroll, 1)
        self.caption_label = QLabel("", box)
        self.caption_label.setObjectName("caption")
        layout.addWidget(self.caption_label)
        return box

    def _output_group(self) -> QWidget:
        """Summary line, then the canvas beside (block table over the JSON pane).

        Three views of one result, all keyed on the block id: the canvas is where the eye
        is, the table is where the count is, the JSON pane is what gets copied. Clicking
        any of them selects the block in the other two.
        """
        box = QGroupBox("Output", self.splitter)
        layout = QVBoxLayout(box)
        self.summary_label = QLabel("no result yet", box)
        layout.addWidget(self.summary_label)

        columns = QSplitter(Qt.Orientation.Horizontal, box)
        self._canvas = ResultCanvas(columns, gui_config=self._config.gui)
        columns.addWidget(self._canvas)

        right = QSplitter(Qt.Orientation.Vertical, columns)
        self.block_tree = QTreeWidget(right)
        # The 4th column carries the result-1.2 observations (P8.6): ⚠ = possible_truncation,
        # ⊂n = contained_in block n. The header spells them out because the glyphs alone are
        # a private alphabet, and the row tooltip says what to do about them.
        self.block_tree.setHeaderLabels(["#", "score", "text / error", "flags"])
        # `headerItem()`, not `header()`: QHeaderView.setToolTip(section) shadows the
        # two-argument QWidget form, so the per-column tooltip has to go on the header item.
        self.block_tree.headerItem().setToolTip(3, self._flags_legend)
        self.block_tree.setColumnWidth(3, 52)
        self.block_tree.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.block_tree.setRootIsDecorated(False)
        self.block_tree.itemSelectionChanged.connect(self._on_table_selection)
        # P9.3: the table is where a flag becomes an action. Right-click one row (or several) and
        # read that box again — the detector is not asked twice, and the rest of the page stands.
        self.block_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.block_tree.customContextMenuRequested.connect(self._block_menu)
        right.addWidget(self.block_tree)
        self._json_pane = JsonPane(right)
        right.addWidget(self._json_pane)
        right.setStretchFactor(0, 2)
        right.setStretchFactor(1, 3)

        columns.addWidget(right)
        columns.setStretchFactor(0, 3)
        columns.setStretchFactor(1, 2)
        layout.addWidget(columns, 1)

        self._canvas.block_selected.connect(self._on_canvas_selection)
        self._canvas.zoom_changed.connect(self.panel.set_view_zoom)
        self._canvas.annotated_saved.connect(
            lambda path: self._say(f"Annotated PNG written to {path}"))
        self._json_pane.copied.connect(self._say)
        self._json_pane.saved.connect(lambda path: self._say(f"Result JSON written to {path}"))
        return box

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("main")
        bar.setMovable(False)
        self.browse_button = QPushButton("Browse…", bar)
        self.browse_button.clicked.connect(self.browse)
        bar.addWidget(self.browse_button)
        self.run_button = QPushButton("Generate", bar)
        self.run_button.clicked.connect(self.run_now)
        bar.addWidget(self.run_button)
        self.cancel_tool_button = QPushButton("Cancel", bar)
        self.cancel_tool_button.clicked.connect(self.cancel)
        bar.addWidget(self.cancel_tool_button)
        bar.addSeparator()
        # Both targets take an optional `path`, and `clicked` carries a `checked` bool:
        # connecting them directly handed the toolbar's `False` to `path`, so Export wrote
        # a file named "False" and Import tried to read one (T9 — only a real click shows
        # this, never a test that passes a path).
        # Named (not a loop) so the MCP activity dock can flash exactly the button a tool
        # touched — `export_config` flashes Export, `import_config` flashes Import.
        self.export_button = QPushButton("Export config…", bar)
        self.export_button.clicked.connect(self._export_button_pressed)
        bar.addWidget(self.export_button)
        self.import_button = QPushButton("Import config…", bar)
        self.import_button.clicked.connect(self._import_button_pressed)
        bar.addWidget(self.import_button)
        bar.addSeparator()
        #: P9.2: the escape from the fifth badge. Startup stays network-free (D27), so this is
        #: where a user says "ask again" — and the only place the window reaches the repo at all.
        self.refresh_models_button = QPushButton("Refresh models", bar)
        self.refresh_models_button.clicked.connect(lambda: self.refresh_model_list())
        bar.addWidget(self.refresh_models_button)

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        self.status_label = QLabel("", bar)
        bar.addWidget(self.status_label, 1)
        self.stage_label = QLabel("", bar)
        bar.addPermanentWidget(self.stage_label)
        self.progress = QProgressBar(bar)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFixedWidth(220)
        bar.addPermanentWidget(self.progress)

    # -- image input --------------------------------------------------------

    def browse(self) -> None:
        start = self._config.paths.last_image_dir or str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self, "Open scan", start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*)")
        if path:
            self.load_path(path)

    def load_path(self, path: Union[str, Path], *, source: str = "file") -> bool:
        """Load a file. `source` records provenance in the result document
        (`"file"` for a human, `"mcp-path"` for a remote call — see IMAGE_SOURCES)."""
        try:
            image, info = load_image(path, source=source)
        except StudioError as exc:
            self._on_failed(exc)
            return False
        self._payload = ("path", str(path))
        self._config.paths.last_image_dir = str(Path(str(path)).parent)
        return self._accept_image(image, info)

    def load_bytes(self, data: bytes, *, file_name: Optional[str] = None,
                   source: str = "clipboard") -> bool:
        try:
            image, info = image_from_bytes(data, file_name=file_name, source=source)
        except StudioError as exc:
            self._on_failed(exc)
            return False
        self._payload = ("bytes", data, file_name)
        return self._accept_image(image, info)

    def paste_from_clipboard(self) -> bool:
        """Ctrl+V: a copied image, or a file whose URL is on the clipboard.

        Qt hands us whatever pixel format the source used; `qimage_to_pil` accepts only
        `Format_RGB888` (D31), so the conversion is *this* layer's job, at the paste site.
        """
        clipboard = QGuiApplication.clipboard()
        if clipboard is None:
            return False
        urls = clipboard.mimeData().urls() if clipboard.mimeData().hasUrls() else []
        for url in urls:
            if url.isLocalFile() and _is_image_path(url.toLocalFile()):
                return self.load_path(url.toLocalFile())
        from PySide6.QtGui import QImage
        from ..core.imaging import qimage_to_pil

        source = clipboard.image()
        if source.isNull():
            self._say("Clipboard has no image (copy a file or a screenshot instead).")
            return False
        if source.format() != QImage.Format.Format_RGB888:
            source = source.convertToFormat(QImage.Format.Format_RGB888)
        try:
            image = qimage_to_pil(source)
        except StudioError as exc:
            self._on_failed(exc)
            return False
        # `_accept_image` (not a hand-rolled preview update) is what sets `_image`: a paste
        # that only set `_info` would show one picture and run another.
        self._payload = ("pil", image)
        return self._accept_image(image, describe(image, source="clipboard", file_name=None))

    def _accept_image(self, image, info: ImageInfo) -> bool:
        self._image, self._info = image, info
        self.clear_result()          # boxes from the previous page must not follow this one
        self._show_preview()
        self._say(f"Loaded {info.file_name or info.source} "
                  f"({info.width}×{info.height} px)")
        self.panel.generate_button.setEnabled(True)
        return True

    def _show_preview(self) -> None:
        if self._image is None:
            return
        try:
            qimage = pil_to_qimage(self._image)
        except StudioError as exc:              # PySide6 missing: we are the GUI, so no
            self._on_failed(exc)
            return
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.setMinimumSize(pixmap.size())
        if self._info is not None:
            self.caption_label.setText(
                f"{self._info.file_name or self._info.source} · "
                f"{self._info.width}×{self._info.height} px · "
                f"sha256 {self._info.sha256[:12]}")

    # -- drag & drop --------------------------------------------------------

    def dragEnterEvent(self, event) -> None:                      # noqa: N802 (Qt API)
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() and _is_image_path(u.toLocalFile())
                                  for u in mime.urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:                           # noqa: N802 (Qt API)
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if url.isLocalFile() and _is_image_path(local):
                self.load_path(local)
                event.acceptProposedAction()
                return

    # -- running ------------------------------------------------------------

    def run_now(self) -> bool:
        if self._image is None:
            self._on_failed(StudioError(ErrorCode.NO_IMAGE, "Load an image first.",
                                        {"hint": "Browse, drop a file, or press Ctrl+V"}))
            return False
        config = self.panel.collect_into(self._config)
        self._config = config
        self._manager.pipeline = Pipeline(self._manager.detection, self._manager.ocr, config)
        kind = self._payload[0] if self._payload else "pil"
        if kind == "path":
            return self._bridge.start_file(self._payload[1])
        if kind == "bytes":
            return self._bridge.start_bytes(self._payload[1], file_name=self._payload[2],
                                            source="clipboard")
        # a pasted QImage: re-encode once so the job path is identical to a file/bytes run
        from ..core.imaging import to_png_bytes

        return self._bridge.start_bytes(to_png_bytes(self._image), file_name=None,
                                        source="clipboard")

    def cancel(self) -> bool:
        return self._bridge.cancel()

    # -- job signals --------------------------------------------------------

    def _on_progress(self, state) -> None:
        self.progress.setVisible(state.state in (JobState.RUNNING, JobState.CANCELLING))
        self.progress.setValue(int(round(state.percent)))
        self.stage_label.setText(state.stage)
        if state.message:
            self.status_label.setText(state.message)
        self._set_busy(state.state in (JobState.RUNNING, JobState.CANCELLING))

    def _on_finished(self, result: PipelineResult) -> None:
        self._set_busy(False)
        self.set_result(result)
        self.result_ready.emit(result)
        self.refresh_engine_badge()      # a run is what loads the engines (P8)
        self._say(f"{result.block_count} block(s) · detect {result.timing_ms.detect} ms · "
                  f"ocr {result.timing_ms.ocr} ms")

    def _on_failed(self, error: StudioError) -> None:
        self._set_busy(False)
        self.refresh_engine_badge()      # a failed load is exactly when the state matters
        self._say(f"{error.code.value}: {error.message}")

    def _on_cancelled(self) -> None:
        self._set_busy(False)
        self.refresh_engine_badge()
        self._say("Cancelled.")

    # -- engine warm-up (P8.3) ----------------------------------------------

    def start_preload(self, *, probe: bool = True) -> bool:
        """`--preload`: warm both engines now, on the job worker. Never blocks the window.

        It is a job, so the progress bar shows `warmup`, Cancel works, and a second caller gets
        `BUSY` instead of a second load. The window itself does no loading: `gui/app.py` decides
        whether to ask (and skips it under `--mock`, where there is nothing to warm).
        """
        if self._preloading:
            return False
        ok = self._bridge.start_warmup(probe=probe)
        if ok:
            self._preloading = True
            self._say("preload: warming engines…")
        return ok

    def _on_warmup_done(self, report) -> None:
        """Warm-up finished: say what actually happened, then let the badge prove it."""
        self._preloading = False
        self.refresh_engine_badge()      # the badge reads `status()`, which now knows it is warm
        self._say(report.summary() if report is not None else "warm-up finished")

    def _set_busy(self, busy: bool) -> None:
        self.panel.set_busy(busy)
        self.run_button.setEnabled(not busy)
        self.cancel_tool_button.setEnabled(bool(busy))
        self.browse_button.setEnabled(not busy)

    # -- result rendering ----------------------------------------------------

    def set_result(self, result: PipelineResult) -> None:
        """The single entry point for a finished run: table, canvas and JSON at once.

        P7 calls this too (an MCP `run_*` finishes here), which is why nothing else in the
        window reads `result` directly.
        """
        self._result = result
        flagged = flagged_count(result.blocks)
        summary = (f"{result.block_count} block(s) · {result.detection.model} · "
                   f"{sum(1 for b in result.blocks if b.text)} read")
        if flagged:
            # Said out loud because these are the blocks a run should not be believed on:
            # one may stop mid-sentence, the others are held inside a bigger box.
            summary += f" · {flagged} flagged"
            self.summary_label.setToolTip(self._flags_legend)
        else:
            self.summary_label.setToolTip("")
        self.summary_label.setText(summary)
        self.block_tree.clear()
        for block in result.blocks:
            shown = block.text if block.text else (
                block.error.get("code", "") if block.error else "")
            item = QTreeWidgetItem(
                self.block_tree,
                [str(block.id), f"{block.score:.2f}", shown[:120],
                 " ".join(flag_marks(block))])
            note = flag_note(block)
            if note:
                item.setToolTip(3, note)
                item.setToolTip(2, note)
        self._canvas.set_result(result, self._image)
        self._json_pane.set_result(result)

    def clear_result(self) -> None:
        self._result = None
        self.block_tree.clear()
        self._canvas.clear()
        self._json_pane.clear()
        self.summary_label.setText("no result yet")

    def clear_image(self) -> None:
        """Drop the loaded page too (the MCP `reset` door; mirrors `clear_result`)."""
        self._image = None
        self._info = None
        self._payload = ()
        self.image_label.clear()
        self.image_label.setMinimumSize(0, 0)
        self.caption_label.setText("no image")
        self.clear_result()
        # Generate stays enabled (as it is at startup): pressing it with no page is the
        # NO_IMAGE path the window already reports, not a dead button.

    # -- selection sync (canvas <-> table <-> JSON) --------------------------

    def _on_canvas_selection(self, block_id: int) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._select_row(block_id)
            self._json_pane.focus_block(block_id)
        finally:
            self._syncing = False
        block = next((b for b in (self._result.blocks if self._result else ())
                      if b.id == block_id), None)
        if block is not None:
            marks = flag_marks(block)
            self._say(f"Block #{block.id} · {block.score:.2f} · "
                      f"{block.text or (block.error or {}).get('code', 'no text')}"
                      + (f"  [{' '.join(marks)}]" if marks else ""))

    def selected_block_ids(self) -> list[int]:
        """The ids of the selected rows, sorted and de-duplicated (empty when none apply)."""
        ids: list[int] = []
        for item in self.block_tree.selectedItems():
            try:
                ids.append(int(item.text(0)))
            except ValueError:                    # a header or a hand-added row
                continue
        return sorted(set(ids))

    def _block_menu(self, pos) -> None:
        """The block table's context menu: read the selection again (P9.3).

        Nothing else lives here. A menu that offered "delete block" or "merge boxes" would make
        the window the author of its own result document, and the app's rule is that it reports
        what the engines said (D58/D60). Re-reading is the one action that stays inside that rule:
        same box, same engine, another look.
        """
        ids = self.selected_block_ids()
        if not ids or self.current_result is None:
            return
        menu = QMenu(self)
        action = menu.addAction("Re-read this block" if len(ids) == 1
                                else f"Re-read {len(ids)} blocks")
        action.setToolTip("Crop the box and run OCR again; the detector is not asked twice.\n"
                          "If the text was cut off, raise OCR › Max new tokens first.")
        if menu.exec(self.block_tree.viewport().mapToGlobal(pos)) is action:
            self.reocr_selected()

    def reocr_selected(self) -> bool:
        """Re-read the selected blocks of the shown result (P9.3).

        The same payload the page came from is handed back to the job, so a pasted image, a
        dropped file and a clipboard import all re-read exactly the pixels that were read before.
        """
        ids = self.selected_block_ids()
        result = self.current_result
        if not ids or result is None:
            return False
        kind = self._payload[0] if self._payload else "pil"
        if kind == "path":
            return self._bridge.start_reocr_file(self._payload[1], result, ids)
        from ..core.imaging import to_png_bytes
        if kind == "bytes":
            return self._bridge.start_reocr_bytes(self._payload[1], result, ids,
                                                  file_name=self._payload[2],
                                                  source="clipboard")
        return self._bridge.start_reocr_bytes(to_png_bytes(self._image), result, ids,
                                              file_name=None, source="clipboard")

    def _on_table_selection(self) -> None:
        if self._syncing:
            return
        items = self.block_tree.selectedItems()
        if not items:
            return
        try:
            block_id = int(items[0].text(0))
        except ValueError:                        # a header or a hand-added row
            return
        self._syncing = True
        try:
            self._canvas.select_block(block_id)
            self._json_pane.focus_block(block_id)
        finally:
            self._syncing = False

    def _select_row(self, block_id: int) -> None:
        for row in range(self.block_tree.topLevelItemCount()):
            item = self.block_tree.topLevelItem(row)
            if item is not None and item.text(0) == str(block_id):
                self.block_tree.setCurrentItem(item)
                return

    # -- view settings -------------------------------------------------------

    def set_config(self, config: AppConfig) -> None:
        """The single door a *config* enters the window by (P7; mirrors `set_result()`).

        `import_config()` and the MCP `set_config` / `import_config` tools both end here, so
        the window's own object, the widgets and the canvas can never disagree — and a later
        Generate (which rebuilds the pipeline from `collect_into(self._config)`) runs what the
        MCP client asked for, not what the window remembered.
        """
        self._config = config
        self.panel.set_models(self._startup_catalog(), offline=bool(config.paths.offline))
        self.panel.apply_config(config)
        self._apply_view_settings()

    def apply_view_config(self) -> None:
        """Public door for "the `gui.*` section moved, redraw the canvas" (P7)."""
        self._apply_view_settings()

    def _apply_view_settings(self) -> None:
        """Push the panel's `gui.*` widgets into the canvas (it is the only consumer)."""
        gui = self.panel.collect_into(self._config).gui
        self._canvas.set_overlay_visible(gui.overlay_visible)
        self._canvas.set_labels_visible(gui.labels_visible)
        self._canvas.set_fill_opacity(gui.fill_opacity)
        self._canvas.set_hover_opacity(gui.hover_opacity)
        self._canvas.set_zoom(mode=gui.zoom_mode, percent=gui.zoom_percent)

    # -- config -------------------------------------------------------------

    def _export_button_pressed(self) -> None:
        """Toolbar slot — see `_build_toolbar`: never let `clicked`'s bool become `path`."""
        self.export_config()

    def _import_button_pressed(self) -> None:
        self.import_config()

    def export_config(self, path: Optional[Union[str, Path]] = None) -> Optional[str]:
        target = path
        if target is None:
            target, _ = QFileDialog.getSaveFileName(self, "Export configuration",
                                                    "animetext.config.json",
                                                    "JSON (*.json)")
            if not target:
                return None
        try:
            written = save_config_file(self.panel.collect_into(self._config), target)
        except StudioError as exc:
            self._on_failed(exc)
            return None
        self._say(f"Configuration written to {written}")
        return str(written)

    def import_config(self, path: Optional[Union[str, Path]] = None) -> bool:
        source = path
        if source is None:
            source, _ = QFileDialog.getOpenFileName(self, "Import configuration", ".",
                                                    "JSON (*.json)")
            if not source:
                return False
        try:
            config = load_config_file(source)
        except StudioError as exc:
            self._on_failed(exc)
            return False
        before = self._config.to_dict()
        self.set_config(config)
        # P8.8: some keys are read once, at server start. The file's value is kept (an export
        # stays faithful), but the badge on the panel still describes the server that is
        # actually listening — so say so here instead of letting the two disagree in silence.
        waiting = restart_required(changed_config_keys(before, config.to_dict()))
        if waiting:
            self._say(f"Configuration imported from {source} — "
                      f"{', '.join(waiting)} take effect next start: the MCP server is still "
                      f"running with the settings it bound with")
        else:
            self._say(f"Configuration imported from {source}")
        return True

    # -- odds and ends ------------------------------------------------------

    def _say(self, message: str) -> None:
        self.status_label.setText(message)

    def set_closing_hook(self, hook) -> None:                       # P7 (the MCP service)
        """Register a callable to run first thing on close (closes the Qt bridge)."""
        self._closing_hook = hook

    def closeEvent(self, event) -> None:                          # noqa: N802 (Qt API)
        hook = getattr(self, "_closing_hook", None)
        if hook is not None:
            try:
                hook()
            except Exception:                                       # noqa: BLE001
                pass    # a dead bridge must not keep the window open
        self._bridge.shutdown()
        if self._manager.is_busy:
            self._manager.cancel()
            self._manager.wait(2.0)
        super().closeEvent(event)
