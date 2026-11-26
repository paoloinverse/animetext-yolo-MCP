"""Part 5 tests - the GUI shell, driven offscreen.

Everything here runs with `QT_QPA_PLATFORM=offscreen` and the mock (or stub) engines: the
point is the wiring - drop/paste/browse set the image, Generate runs the *real* job path
through `JobManager`, buttons follow the job, and config export/import round-trips through
the widgets. Pixel-perfect drawing is Part 6; here we only assert that a render happens.
"""
from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed (pip install animetext-studio[gui])")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path                                        # noqa: E402
from types import SimpleNamespace                               # noqa: E402

from PIL import Image                                           # noqa: E402

from animetext_studio.core import annotation as A              # noqa: E402
from animetext_studio.core.config import MODEL_PRESETS, AppConfig  # noqa: E402
from animetext_studio.core.jobs import JobManager               # noqa: E402
from animetext_studio.core.pipeline import Pipeline             # noqa: E402
from animetext_studio.core.types import (                       # noqa: E402
    BBox, Detection, ErrorCode, StudioError, TEXT_LABEL,
)
from animetext_studio.engines.base import EngineState            # noqa: E402
from animetext_studio.gui.canvas import ResultCanvas             # noqa: E402
from animetext_studio.gui.controls import ControlPanel          # noqa: E402
from animetext_studio.gui.main_window import MainWindow         # noqa: E402

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "sample_page.png")


def select_order(combo, value: str) -> None:
    """The reading-order combo shows friendly labels; the value lives in itemData."""
    index = combo.findData(value)
    assert index >= 0, f"{value!r} is not an offered reading order"
    combo.setCurrentIndex(index)


# ---------------------------------------------------------------------------
# Stub engines: the GUI must drive the same JobManager the CLI drives
# ---------------------------------------------------------------------------

class StubDetection:
    name = "detection"
    loaded = False          # flipped by detect(), cleared by unload() (P8 badge)

    def __init__(self, blocked: bool = False):
        self.detections = [Detection(BBox(10, 10, 200, 90), TEXT_LABEL, 0.9),
                           Detection(BBox(40, 120, 220, 200), TEXT_LABEL, 0.8)]
        self.release = threading.Event()
        self.entered = threading.Event()
        if not blocked:
            self.release.set()

    def status(self):
        # P8: the engines badge reads `state` + `loaded_models`, so the stubs carry them
        # too — a stub that cannot answer is a stub that hides a regression.
        return SimpleNamespace(backend="stub", device="cpu",
                               state=EngineState.READY if self.loaded else EngineState.UNLOADED,
                               loaded_models=("stub",) if self.loaded else ())

    def resolved_score_threshold(self, model_name, requested):
        return 0.05 if requested is None else float(requested)

    def unload(self):
        self.loaded = False

    def detect(self, image, model_name, **kwargs):
        self.loaded = True
        self.entered.set()
        cancel = kwargs.get("cancel_event")
        while not self.release.is_set():
            if cancel is not None and cancel.is_set():
                raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")
            self.release.wait(0.01)
        if cancel is not None and cancel.is_set():
            raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")
        return list(self.detections)


class StubOcr:
    name = "ocr"

    def __init__(self):
        self.unloaded = False
        self.loaded = False

    def status(self):
        return SimpleNamespace(device="cpu", backend="stub",
                               state=EngineState.READY if self.loaded else EngineState.UNLOADED,
                               loaded_models=("stub",) if self.loaded else ())

    def unload(self):
        self.unloaded = True
        self.loaded = False

    def ocr_crops(self, images, **kwargs):
        self.loaded = True
        return ["first", "second"][:len(images)]


def build_window(*, blocked: bool = False, mock: bool = False, config=None):
    from animetext_studio.core.pipeline import engines_from_config

    config = config or AppConfig.default()
    if mock:
        engines = engines_from_config(config, mock=True)
    else:
        engines = SimpleNamespace(detection=StubDetection(blocked=blocked), ocr=StubOcr())
    manager = JobManager(Pipeline(engines.detection, engines.ocr, config))
    return MainWindow(manager, config, mock=mock), manager


@pytest.fixture
def window(qtbot):
    win, _manager = build_window()
    qtbot.addWidget(win)
    win.show()
    return win


# ---------------------------------------------------------------------------
# Layout and controls
# ---------------------------------------------------------------------------

class TestWindow:
    def test_the_window_builds_and_lists_the_models(self, window):
        assert window.objectName() == "AnimeTextStudioMainWindow"
        assert window.controls.model_count == 5
        assert window.controls.current_model() == "yolo12m_animetext"
        assert not window.has_image
        assert "no image" in window.image_label.text()

    def test_the_controls_come_up_with_the_config(self, window):
        panel = window.controls
        assert panel.iou_spin.value() == pytest.approx(0.35)
        assert panel.score_spin.value() == pytest.approx(0.05)
        assert panel.device_combo.currentText() == "auto"
        assert panel.patches_combo.currentText() == "auto"
        assert panel.order_combo.currentData() == "manga_rtl"
        assert not panel.nfkc_check.isChecked()

    def test_the_mock_badge_only_shows_for_mock_engines(self, qtbot):
        mock_win, _ = build_window(mock=True)
        qtbot.addWidget(mock_win)
        assert mock_win.controls.mock_badge.text() == "MOCK ENGINES"
        assert mock_win.controls.mock_badge.isVisibleTo(mock_win)
        assert mock_win.is_mock is True
        assert "[mock]" in mock_win.windowTitle()

    def test_the_mcp_indicator_starts_off_and_can_be_flipped(self, window):
        """P7 drives this label; the shell must already have the slot for it."""
        assert window.controls.mcp_badge.text() == "MCP: off"
        window.controls.set_mcp_status("MCP :8765")
        assert window.controls.mcp_badge.text() == "MCP :8765"

    def test_the_central_widget_is_scrollable_both_ways(self, window):
        from PySide6.QtWidgets import QScrollArea

        assert isinstance(window.centralWidget(), QScrollArea)

    def test_a_screenshot_renders_non_blank(self, window, tmp_path):
        window.load_path(SAMPLE)
        pixmap = window.grab()
        assert pixmap.width() > 200 and pixmap.height() > 200
        image = pixmap.toImage()
        non_white = sum(1 for y in range(0, image.height(), 37)
                        for x in range(0, image.width(), 37)
                        if image.pixelColor(x, y).lightness() < 240)
        assert non_white > 20, "the window rendered blank"
        pixmap.save(str(tmp_path / "window.png"))


class TestImageInput:
    def test_browse_path_sets_preview_and_caption(self, window):
        assert window.load_path(SAMPLE)
        assert window.has_image
        assert window.caption_label.text().startswith("sample_page.png")
        assert "900" in window.caption_label.text() and "1300" in window.caption_label.text()
        assert not window.image_label.pixmap().isNull()

    def test_a_missing_file_reports_instead_of_raising(self, window):
        assert window.load_path("/nope/nothing.png") is False
        assert "IO_ERROR" in window.status_label.text()

    def test_paste_of_a_pasted_qimage_in_another_format_still_loads(self, window, monkeypatch):
        """D31: `qimage_to_pil` refuses non-RGB888, so the paste site converts."""
        from PySide6.QtGui import QImage

        grey = QImage(24, 16, QImage.Format.Format_Grayscale8)
        grey.fill(0)
        clipboard = SimpleNamespace(mimeData=lambda: SimpleNamespace(hasUrls=lambda: False,
                                                                    urls=lambda: []),
                                    image=lambda: grey)
        monkeypatch.setattr("animetext_studio.gui.main_window.QGuiApplication.clipboard",
                            staticmethod(lambda: clipboard))
        assert window.paste_from_clipboard() is True
        assert window.has_image
        assert window._image.size == (24, 16), "the preview and the job must be the same image"
        assert window._info.source == "clipboard"

    def test_paste_of_a_file_url_loads_that_file(self, window, monkeypatch):
        from PySide6.QtCore import QUrl

        url = QUrl.fromLocalFile(SAMPLE)
        mime = SimpleNamespace(hasUrls=lambda: True, urls=lambda: [url])
        clipboard = SimpleNamespace(mimeData=lambda: mime, image=lambda: None)
        monkeypatch.setattr("animetext_studio.gui.main_window.QGuiApplication.clipboard",
                            staticmethod(lambda: clipboard))
        assert window.paste_from_clipboard() is True
        assert window.caption_label.text().startswith("sample_page.png")

    def test_an_empty_clipboard_says_so(self, window, monkeypatch):
        from PySide6.QtGui import QImage

        clipboard = SimpleNamespace(
            mimeData=lambda: SimpleNamespace(hasUrls=lambda: False, urls=lambda: []),
            image=lambda: QImage())
        monkeypatch.setattr("animetext_studio.gui.main_window.QGuiApplication.clipboard",
                            staticmethod(lambda: clipboard))
        assert window.paste_from_clipboard() is False
        assert "no image" in window.status_label.text().lower()

    def test_drag_and_drop_of_an_image_file_loads_it(self, window):
        from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
        from PySide6.QtGui import QDragEnterEvent, QDropEvent

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("/nope.txt"), QUrl.fromLocalFile(SAMPLE)])
        enter = QDragEnterEvent(QPointF(20.0, 20.0), Qt.DropAction.CopyAction, mime,
                                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        window.dragEnterEvent(enter)
        assert enter.isAccepted()
        drop = QDropEvent(QPointF(20.0, 20.0), Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        window.dropEvent(drop)
        assert window.has_image

    def test_a_non_image_drag_is_refused(self, window):
        from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
        from PySide6.QtGui import QDragEnterEvent

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("/nope.txt")])
        enter = QDragEnterEvent(QPointF(5.0, 5.0), Qt.DropAction.CopyAction, mime,
                                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        window.dragEnterEvent(enter)
        assert not enter.isAccepted()


# ---------------------------------------------------------------------------
# Running a job
# ---------------------------------------------------------------------------

class TestRun:
    def test_generate_with_mock_engines_produces_a_result(self, qtbot):
        win, _ = build_window(mock=True)
        qtbot.addWidget(win)
        win.show()
        win.load_path(SAMPLE)
        with qtbot.waitSignal(win.result_ready, timeout=20000) as blocker:
            win.controls.generate_button.click()
        result = blocker.args[0]
        assert result.block_count >= 1
        assert win.current_result is result
        assert "block(s)" in win.summary_label.text()
        assert win.block_tree.topLevelItemCount() == result.block_count
        assert "NO_IMAGE" not in win.status_label.text()

    def test_generate_without_an_image_reports_no_image(self, window):
        window.run_now()
        assert "NO_IMAGE" in window.status_label.text()

    def test_buttons_follow_the_job(self, qtbot):
        win, manager = build_window(blocked=True)
        qtbot.addWidget(win)
        win.show()
        win.load_path(SAMPLE)
        win.run_now()
        qtbot.waitUntil(lambda: win._manager.detection.entered.is_set(), timeout=5000)
        assert not win.run_button.isEnabled()
        assert not win.controls.generate_button.isEnabled()
        assert win.controls.cancel_button.isEnabled()
        assert win.progress.isVisible()
        win._manager.detection.release.set()
        qtbot.waitUntil(lambda: win.run_button.isEnabled(), timeout=5000)
        assert not win.progress.isVisible()

    def test_cancel_from_the_panel_stops_the_job(self, qtbot):
        win, manager = build_window(blocked=True)
        qtbot.addWidget(win)
        win.show()
        win.load_path(SAMPLE)
        win.run_now()
        qtbot.waitUntil(lambda: win._manager.detection.entered.is_set(), timeout=5000)
        with qtbot.waitSignal(win.job_bridge.cancelled, timeout=5000):
            win.controls.cancel_button.click()
        assert "Cancelled" in win.status_label.text()
        assert win.run_button.isEnabled()

    def test_a_failing_engine_shows_the_error_code_not_a_crash(self, qtbot):
        from animetext_studio.core.types import JobState

        class Boom(StubDetection):
            def detect(self, image, model_name, **kwargs):
                raise StudioError(ErrorCode.MODEL_NOT_FOUND, "no weights here")

        engines = SimpleNamespace(detection=Boom(), ocr=StubOcr())
        config = AppConfig.default()
        win = MainWindow(JobManager(Pipeline(engines.detection, engines.ocr, config)),
                         config, mock=False)
        qtbot.addWidget(win)
        win.load_path(SAMPLE)
        win.run_now()
        qtbot.waitUntil(lambda: "MODEL_NOT_FOUND" in win.status_label.text(), timeout=5000)
        assert win.current_result is None
        assert win.job_manager.state is JobState.ERROR


class TestEngineBadge:
    """P8: engine state must be visible in the panel, not only in the activity dock."""

    def test_a_fresh_window_says_unloaded(self, window):
        assert window.controls.engines_badge.text() == "engines: unloaded"

    def test_running_the_job_loads_the_badge(self, qtbot):
        win, _manager = build_window()
        qtbot.addWidget(win)
        win.load_path(SAMPLE)
        with qtbot.waitSignal(win.job_bridge.finished, timeout=5000):
            win.run_now()
        assert win.controls.engines_badge.text() == "engines: det stub · ocr stub"

    def test_unload_engines_is_visible_on_the_panel(self, qtbot):
        """The whole point: an MCP `unload_engines` must change something you can see."""
        win, manager = build_window()
        qtbot.addWidget(win)
        win.load_path(SAMPLE)
        with qtbot.waitSignal(win.job_bridge.finished, timeout=5000):
            win.run_now()
        assert "stub" in win.controls.engines_badge.text()
        manager.unload_engines()
        win.refresh_engine_badge()
        assert win.controls.engines_badge.text() == "engines: unloaded"

    def test_a_broken_engine_status_cannot_break_the_window(self, qtbot):
        class NoStatus(SimpleNamespace):
            name = "detection"

            def status(self):
                raise RuntimeError("engine exploded")

        engines = SimpleNamespace(detection=NoStatus(), ocr=StubOcr())
        config = AppConfig.default()
        win = MainWindow(JobManager(Pipeline(engines.detection, engines.ocr, config)),
                         config, mock=False)
        qtbot.addWidget(win)
        win.refresh_engine_badge()
        assert win.controls.engines_badge.text() == "engines: ?"


# ---------------------------------------------------------------------------
# Configuration round-trip
# ---------------------------------------------------------------------------

class TestConfigWiring:
    def test_collect_into_reads_the_widgets(self, window):
        panel = window.controls
        panel.score_spin.setValue(0.42)
        panel.iou_slider.setValue(75)
        panel.device_combo.setCurrentText("cpu")
        panel.patches_combo.setCurrentText("128")
        panel.batch_spin.setValue(4)
        panel.tokens_spin.setValue(64)
        panel.padding_spin.setValue(6)
        panel.nfkc_check.setChecked(True)
        select_order(panel.order_combo, "top_to_bottom")
        config = panel.collect_into(AppConfig.default())
        assert config.detection.score_threshold == pytest.approx(0.42)
        assert config.detection.iou_threshold == pytest.approx(0.75)
        assert (config.ocr.device, config.ocr.max_patches) == ("cpu", 128)
        assert (config.ocr.batch_size, config.ocr.max_new_tokens) == (4, 64)
        assert config.ocr.crop_padding_px == 6
        assert config.ocr.normalize_nfkc is True
        assert config.pipeline.reading_order == "top_to_bottom", "combo data must survive"

    def test_the_recommended_threshold_button_uses_the_preset(self, window):
        panel = window.controls
        panel.model_combo.setCurrentText("yolo12s_animetext")
        panel.recommended_button.click()
        recommended = MODEL_PRESETS["yolo12s_animetext"]["recommended_score_threshold"]
        assert panel.score_spin.value() == pytest.approx(recommended, abs=0.011)

    def test_export_then_import_restores_the_widgets(self, window, tmp_path):
        panel = window.controls
        panel.score_spin.setValue(0.31)
        select_order(panel.order_combo, "score")
        target = tmp_path / "cfg.json"
        assert window.export_config(target) == str(target)
        panel.score_spin.setValue(0.05)
        select_order(panel.order_combo, "manga_rtl")
        assert window.import_config(target) is True
        assert panel.score_spin.value() == pytest.approx(0.31)
        assert panel.order_combo.currentData() == "score"

    def test_importing_garbage_reports_and_keeps_the_widgets(self, window, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        panel = window.controls
        panel.score_spin.setValue(0.22)
        assert window.import_config(bad) is False
        assert "INVALID" in window.status_label.text() or "JSON" in window.status_label.text()
        assert panel.score_spin.value() == pytest.approx(0.22)

    def test_offline_marks_hub_only_variants_unusable(self, qtbot):
        from animetext_studio.engines.base import model_catalog

        panel = ControlPanel(AppConfig.default())
        qtbot.addWidget(panel)
        catalog = model_catalog(["yolo12n_animetext", "yolo12x_animetext"],
                                {"yolo12n_animetext": "bundled"})
        panel.set_models(catalog, offline=True)
        from PySide6.QtCore import Qt

        enabled = Qt.ItemFlag.ItemIsEnabled
        model = panel.model_combo.model()
        flags = {panel.model_combo.itemText(i): bool(model.item(i).flags() & enabled)
                 for i in range(panel.model_combo.count())}
        assert flags["yolo12n_animetext"] is True
        assert flags["yolo12x_animetext"] is False, \
            "an unbundled variant must not be selectable offline"


# ---------------------------------------------------------------------------
# Part 6 - the annotated canvas
# ---------------------------------------------------------------------------

def make_result(blocks, *, width=400, height=300):
    """A result document around hand-made blocks (the mock pipeline cannot nest boxes)."""
    from animetext_studio.core.types import (
        DetectionInfo, ImageInfo, OcrInfo, PipelineResult, TimingMs,
    )

    info = ImageInfo(source="file", file_name="synthetic.png", width=width, height=height,
                     sha256="0" * 64)
    return PipelineResult(
        image=info,
        detection=DetectionInfo("repo", "model", 0.35, 0.05, True),
        ocr=OcrInfo("repo", "cpu", "auto", 16, 128, 2, False),
        timing_ms=TimingMs(1, 2, 3),
        blocks=tuple(blocks),
    )


def viewport_pixel(canvas, scene_x, scene_y):
    """The colour of a page pixel as it appears in the canvas viewport, right now.

    The view is centred on the point first: at 100 % a 900\u00d71300 page is larger than the
    viewport, and a probe that lands off-screen would silently read the wrong pixel.
    """
    from PySide6.QtCore import QPointF

    view = canvas.view
    view.centerOn(float(scene_x), float(scene_y))
    point = view.mapFromScene(QPointF(float(scene_x), float(scene_y)))
    image = view.viewport().grab().toImage()
    x = max(0, min(image.width() - 1, int(round(point.x()))))
    y = max(0, min(image.height() - 1, int(round(point.y()))))
    color = image.pixelColor(x, y)
    return (color.red(), color.green(), color.blue())


def near(color, target, tolerance=48):
    return all(abs(a - b) <= tolerance for a, b in zip(color, target))


@pytest.fixture
def finished(qtbot):
    """A window whose mock run has actually landed, plus the result it produced."""
    win, _manager = build_window(mock=True)
    qtbot.addWidget(win)
    win.show()
    win.load_path(SAMPLE)
    with qtbot.waitSignal(win.result_ready, timeout=30000) as blocker:
        win.controls.generate_button.click()
    return win, blocker.args[0]


class TestCanvas:
    def test_the_result_lands_on_the_canvas(self, finished):
        win, result = finished
        assert win.canvas.has_result
        assert win.canvas.block_ids() == [b.id for b in result.blocks]
        assert win.canvas.blocks == result.blocks

    def test_a_box_border_is_painted_in_the_blocks_own_colour(self, finished):
        win, result = finished
        win.canvas.zoom_actual()
        block = result.blocks[0]
        left_edge = viewport_pixel(win.canvas, block.bbox.x0,
                                   (block.bbox.y0 + block.bbox.y1) // 2)
        assert near(left_edge, A.rgb_of(block.color)), \
            f"expected {block.color} on the border, got {left_edge}"

    def test_the_fill_is_translucent_not_a_sticker(self, finished):
        win, result = finished
        win.canvas.zoom_actual()
        block = result.blocks[0]
        probe = ((block.bbox.x0 + block.bbox.x1) // 2, block.bbox.y0 + 3)
        with Image.open(SAMPLE) as original:
            page_colour = original.convert("RGB").getpixel(probe)
        centre = viewport_pixel(win.canvas, *probe)
        rgb = A.rgb_of(block.color)
        assert centre != page_colour, "the page should be tinted inside the box"
        assert not near(centre, rgb, tolerance=24), "the fill must let the page through"

    def test_unticking_show_boxes_removes_them_from_the_pixels(self, finished):
        win, result = finished
        win.canvas.zoom_actual()
        block = result.blocks[0]
        probe = ((block.bbox.x0 + block.bbox.x1) // 2, block.bbox.y0 + 3)
        with Image.open(SAMPLE) as original:
            page_colour = original.convert("RGB").getpixel(probe)
        assert viewport_pixel(win.canvas, *probe) != page_colour
        win.controls.overlay_check.setChecked(False)
        assert win.canvas.style_spec.overlay is False
        assert viewport_pixel(win.canvas, *probe) == page_colour, \
            "with the overlay off the canvas shows the page, nothing else"

    def test_unticking_show_labels_changes_the_picture(self, finished):
        win, _result = finished
        win.canvas.zoom_actual()
        with_labels = win.canvas.view.viewport().grab().toImage()
        win.controls.labels_check.setChecked(False)
        without = win.canvas.view.viewport().grab().toImage()
        assert win.canvas.style_spec.labels is False
        assert with_labels.bits() != without.bits()

    def test_the_fill_slider_reaches_the_canvas(self, finished):
        win, result = finished
        win.canvas.zoom_actual()
        block = result.blocks[0]
        probe = ((block.bbox.x0 + block.bbox.x1) // 2, block.bbox.y0 + 3)
        light = viewport_pixel(win.canvas, *probe)
        win.controls.opacity_spin.setValue(0.9)
        assert win.canvas.style_spec.fill_opacity == pytest.approx(0.9)
        assert viewport_pixel(win.canvas, *probe) != light

    def test_hovering_shows_the_full_text_and_leaving_hides_it(self, finished):
        win, result = finished
        block = result.blocks[1]
        win.canvas.set_hover(block.id)
        assert win.canvas.hover_panel.isVisible()
        assert block.text in win.canvas.hover_panel.text()
        assert f"#{block.id}" in win.canvas.hover_panel.text()
        assert "px" in win.canvas.hover_panel.text(), "the panel also names the box"
        win.canvas.set_hover(-1)
        assert not win.canvas.hover_panel.isVisible()

    def test_the_hover_panel_stays_inside_the_viewport(self, finished):
        win, result = finished
        canvas = win.canvas
        canvas.zoom_actual()
        canvas._cursor = SimpleNamespace(
            x=lambda: canvas.view.viewport().width() - 2,
            y=lambda: canvas.view.viewport().height() - 2)
        canvas.set_hover(result.blocks[0].id)
        panel = canvas.hover_panel
        area = canvas.view.viewport().rect()
        assert panel.x() >= 0 and panel.y() >= 0
        assert panel.x() + panel.width() <= area.right() + 1
        assert panel.y() + panel.height() <= area.bottom() + 1

    def test_a_click_selects_the_block_in_the_table_and_the_json(self, finished, qtbot):
        win, result = finished
        block = result.blocks[2]
        with qtbot.waitSignal(win.canvas.block_selected, timeout=2000) as blocker:
            win.canvas.click_at(block.id)
        assert blocker.args[0] == block.id
        assert win.canvas.selected_block_id == block.id
        assert win.block_tree.currentItem().text(0) == str(block.id)
        assert win.json_pane.focused_block_id == block.id
        assert f"#{block.id}" in win.status_label.text()

    def test_clicking_empty_space_clears_the_selection(self, finished):
        win, result = finished
        win.canvas.select_block(result.blocks[0].id)
        win.canvas.click_at(-1)
        assert win.canvas.selected_block_id == -1

    def test_selecting_from_the_table_selects_the_box_and_the_json(self, finished):
        win, result = finished
        block = result.blocks[1]
        row = next(win.block_tree.topLevelItem(i) for i
                   in range(win.block_tree.topLevelItemCount())
                   if win.block_tree.topLevelItem(i).text(0) == str(block.id))
        win.block_tree.setCurrentItem(row)
        assert win.canvas.selected_block_id == block.id
        assert win.json_pane.focused_block_id == block.id

    def test_selecting_an_id_that_is_not_on_the_page_is_refused(self, finished):
        win, _result = finished
        assert win.canvas.select_block(9999) is False

    def test_the_smallest_box_wins_a_click_when_boxes_nest(self, qtbot):
        from animetext_studio.core.types import BBox, TextBlock

        page = TextBlock(id=0, bbox=BBox(0, 0, 400, 300), label="speech", score=0.4,
                         text="page", reading_order=0, color="#808080")
        band = TextBlock(id=1, bbox=BBox(10, 10, 300, 60), label="speech", score=0.7,
                         text="band", reading_order=1, color="#E6194B")
        bubble = TextBlock(id=2, bbox=BBox(20, 20, 120, 55), label="speech", score=0.9,
                           text="bubble", reading_order=2, color="#3CB44B")
        canvas = ResultCanvas()
        qtbot.addWidget(canvas)
        canvas.resize(400, 300)
        canvas.show()
        canvas.set_result(make_result([page, band, bubble]), Image.new("RGB", (400, 300)))
        assert canvas.pick(30, 30) == 2
        assert canvas.pick(200, 30) == 1
        assert canvas.pick(350, 250) == 0
        assert canvas.pick(500, 500) == -1

    def test_a_new_image_never_keeps_the_old_boxes(self, finished):
        win, _result = finished
        assert win.canvas.has_result
        win.load_path(SAMPLE)
        assert not win.canvas.has_result
        assert win.summary_label.text() == "no result yet"

    def test_fit_and_actual_size_are_mirrored_in_the_panel(self, finished):
        win, _result = finished
        win.canvas.zoom_actual()
        assert win.controls.zoom_combo.currentData() == "percent"
        assert win.controls.zoom_spin.value() == 100
        win.canvas.zoom_fit()
        assert win.controls.zoom_combo.currentData() == "fit"

    def test_choosing_a_percentage_in_the_panel_switches_the_canvas(self, finished):
        win, _result = finished
        win.canvas.zoom_fit()
        win.controls.zoom_spin.setValue(250)
        assert win.canvas.zoom_mode == "percent"
        assert win.canvas.zoom_percent == 250
        assert win.canvas.current_scale() == pytest.approx(2.5, abs=0.01)

    def test_ctrl_wheel_walks_the_zoom_steps(self, finished):
        win, _result = finished
        win.canvas.set_zoom(mode="percent", percent=100)
        win.canvas.zoom_by(1)
        assert win.canvas.zoom_percent == 125
        win.canvas.zoom_by(-2)
        assert win.canvas.zoom_percent == 80
        win.canvas.zoom_by(-50)
        assert win.canvas.zoom_percent == 10, "clamped at the far end, never below"
        win.canvas.zoom_by(50)
        assert win.canvas.zoom_percent == 800

    def test_fit_reflows_when_the_view_is_resized(self, finished):
        win, _result = finished
        win.canvas.zoom_fit()
        before = win.canvas.current_scale()
        win.canvas.view.resize(200, 200)
        assert win.canvas.current_scale() != pytest.approx(before, abs=0.01)

    def test_save_annotated_writes_the_page_at_full_resolution(self, finished, tmp_path):
        win, result = finished
        target = tmp_path / "nested" / "annotated.png"
        written = win.canvas.save_annotated(target)
        assert written == str(target) and Path(target).exists()
        with Image.open(target) as saved:
            assert saved.size == (900, 1300), "the export is the page, not the viewport"
            plain = Image.open(SAMPLE).convert("RGB")
            assert saved.convert("RGB").tobytes() != plain.tobytes()

    def test_the_save_button_asks_where_to_put_it(self, finished, tmp_path, monkeypatch):
        """`clicked` carries a `checked` bool: it must not become the path argument."""
        target = tmp_path / "from-dialog.png"
        calls = []

        def fake_dialog(parent, title, suggested, filter_string):
            calls.append(suggested)
            return str(target), "PNG (*.png)"

        monkeypatch.setattr("animetext_studio.gui.canvas.QFileDialog.getSaveFileName",
                            staticmethod(fake_dialog))
        win, _result = finished
        win.canvas.save_button.click()
        assert calls and calls[0].endswith(".annotated.png")
        assert Path(target).exists()
        assert not Path("False").exists(), "the checked bool leaked into save_annotated(path)"

    def test_the_export_carries_the_selection(self, finished, tmp_path):
        """What the user has selected is part of the picture they save."""
        win, result = finished
        win.canvas.select_block(-1)
        plain = tmp_path / "plain.png"
        assert win.canvas.save_annotated(plain) == str(plain)
        win.canvas.select_block(result.blocks[0].id)
        lit = tmp_path / "lit.png"
        assert win.canvas.save_annotated(lit) == str(lit)
        assert Path(plain).read_bytes() != Path(lit).read_bytes()


class TestJsonPane:
    def test_the_pane_holds_the_document_byte_for_byte(self, finished):
        win, result = finished
        assert win.json_pane.text == result.to_json(indent=2)
        assert win.json_pane.editor.toPlainText() == result.to_json(indent=2)

    def test_the_pane_is_read_only(self, finished):
        assert finished[0].json_pane.editor.isReadOnly()

    def test_the_count_line_says_what_is_there(self, finished):
        win, result = finished
        assert f"{result.block_count} block(s)" in win.json_pane.count_label.text()

    def test_copy_all_puts_the_exact_document_on_the_clipboard(self, finished):
        win, result = finished
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is None:
            pytest.skip("no clipboard on this platform")
        copied = win.json_pane.copy_all()
        assert clipboard.text() == result.to_json(indent=2)
        assert copied == clipboard.text()
        assert "JSON copied" in win.status_label.text()

    def test_save_as_writes_readable_utf8(self, finished, tmp_path):
        win, result = finished
        target = tmp_path / "result.json"
        assert win.json_pane.save_as(target) == str(target)
        written = Path(target).read_text(encoding="utf-8")
        assert written == result.to_json(indent=2)
        assert any(ord(ch) > 0x2FFF for ch in written), "CJK must not be escaped"
        assert "\\u" not in written

    def test_blocks_are_indexed_by_line_and_focusable(self, finished):
        win, result = finished
        pane = win.json_pane
        for block in result.blocks:
            lines = pane.block_lines(block.id)
            assert lines is not None and lines[1] > lines[0]
        first = result.blocks[0].id
        assert pane.focus_block(first) is True
        document = pane.editor.document()
        start, end = pane.block_lines(first)
        cursor_line = pane.editor.textCursor().blockNumber()
        assert start <= cursor_line <= end
        assert pane.focus_block(4242) is False

    def test_copy_block_text_and_json(self, finished):
        win, result = finished
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is None:
            pytest.skip("no clipboard on this platform")
        block = result.blocks[1]
        assert win.json_pane.copy_block_text(block.id) == block.text
        assert clipboard.text() == block.text
        import json as _json

        win.json_pane.copy_block_json(block.id)
        assert _json.loads(clipboard.text())["id"] == block.id

    def test_the_copy_block_button_uses_the_current_selection(self, finished):
        win, result = finished
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard is None:
            pytest.skip("no clipboard on this platform")
        block = result.blocks[0]
        win.json_pane.focus_block(block.id)
        win.json_pane.copy_block_button.click()
        assert clipboard.text() == block.text

    def test_the_save_button_asks_where_to_put_it(self, finished, tmp_path, monkeypatch):
        target = tmp_path / "from-dialog.json"
        monkeypatch.setattr("animetext_studio.gui.json_pane.QFileDialog.getSaveFileName",
                            staticmethod(lambda *a, **k: (str(target), "JSON (*.json)")))
        finished[0].json_pane.save_button.click()
        assert Path(target).exists()
        assert not Path("False").exists()

    def test_clearing_removes_the_document(self, finished):
        win, _result = finished
        win.clear_result()
        assert win.json_pane.text == ""
        assert win.json_pane.block_lines(0) is None


class TestViewConfigRoundTrip:
    """`gui.*` is configuration, so it must survive export/import like every other knob."""

    def test_the_view_widgets_are_read_by_collect_into(self, window):
        panel = window.controls
        panel.overlay_check.setChecked(False)
        panel.labels_check.setChecked(False)
        panel.opacity_spin.setValue(0.4)
        panel.zoom_spin.setValue(250)
        config = panel.collect_into(AppConfig.default())
        assert config.gui.overlay_visible is False
        assert config.gui.labels_visible is False
        assert config.gui.fill_opacity == pytest.approx(0.4)
        assert (config.gui.zoom_mode, config.gui.zoom_percent) == ("percent", 250)

    def test_export_then_import_restores_the_view_and_the_canvas(self, window, tmp_path):
        panel = window.controls
        panel.overlay_check.setChecked(False)
        panel.opacity_spin.setValue(0.55)
        panel.zoom_spin.setValue(300)
        target = tmp_path / "cfg.json"
        assert window.export_config(target) == str(target)

        panel.overlay_check.setChecked(True)
        panel.opacity_spin.setValue(0.18)
        panel.zoom_combo.setCurrentIndex(panel.zoom_combo.findData("fit"))
        assert window.import_config(target) is True

        assert panel.overlay_check.isChecked() is False
        assert panel.opacity_spin.value() == pytest.approx(0.55)
        assert panel.zoom_combo.currentData() == "percent"
        assert panel.zoom_spin.value() == 300
        assert window.canvas.style_spec.overlay is False
        assert window.canvas.style_spec.fill_opacity == pytest.approx(0.55)
        assert window.canvas.zoom_percent == 300

    def test_the_default_config_keeps_the_documented_view_defaults(self, window):
        defaults = AppConfig.default().gui
        assert window.canvas.style_spec.fill_opacity == pytest.approx(defaults.fill_opacity)
        assert window.canvas.zoom_mode == defaults.zoom_mode == "fit"

    def test_a_result_draws_through_the_configured_view(self, finished, tmp_path):
        """The exported PNG obeys the toggles: what you see is what you save."""
        win, _result = finished
        win.controls.labels_check.setChecked(False)
        win.controls.overlay_check.setChecked(False)
        hidden = tmp_path / "no-overlay.png"
        win.canvas.save_annotated(hidden)
        with Image.open(hidden) as saved, Image.open(SAMPLE) as original:
            # PNG re-encoding is not byte-stable, so compare the pixels, not the files.
            assert saved.convert("RGB").tobytes() == original.convert("RGB").tobytes(), \
                "overlay and labels off must reproduce the untouched page"


# ---------------------------------------------------------------------------
# Rect conversion: our boxes are xyxy, Qt's QRect/QRectF are (x, y, w, h)
# ---------------------------------------------------------------------------

class TestQtRectConversion:
    """Both traps were live in T9: a 122x19 chip painted as 281x90 over the box below."""

    def test_qrectf_is_built_from_corners(self):
        from animetext_studio.gui.canvas import _qrectf

        rect = _qrectf(BBox(10, 20, 30, 50))
        assert (rect.left(), rect.top()) == (10.0, 20.0)
        assert rect.width() == 20.0 and rect.height() == 30.0

    def test_qrect_is_built_from_corners(self):
        from animetext_studio.gui.canvas import _qrect

        rect = _qrect((10, 20, 30, 50))
        assert rect.width() == 20 and rect.height() == 30
        assert (rect.left(), rect.top()) == (10, 20)


class TestCanvasExtras:
    """The two PLAN acceptance items the main classes don't cover verbatim."""

    def test_the_hover_panel_opacity_comes_from_the_config(self, qtbot):
        from animetext_studio.core.config import GuiConfig
        from animetext_studio.gui.canvas import ResultCanvas

        panel = ResultCanvas(gui_config=GuiConfig(hover_opacity=0.25))
        qtbot.addWidget(panel)
        assert "rgba(24, 24, 26, 64)" in panel.hover_panel.styleSheet()
        panel.set_hover_opacity(1.0)
        assert "rgba(24, 24, 26, 255)" in panel.hover_panel.styleSheet()

    def test_the_export_carries_every_block(self, finished, tmp_path):
        """PLAN: 'annotated export matches block count' — every block's own colour must
        be somewhere in the baked PNG (border or chip), for all 5 mock blocks."""
        win, result = finished
        target = tmp_path / "every-block.png"
        assert win.canvas.save_annotated(target) == str(target)
        with Image.open(target) as img:
            data = img.convert("RGB").tobytes()
        for block in result.blocks:
            assert bytes(A.rgb_of(block.color)) in data, \
                f"block #{block.id} ({block.color}) is missing from the export"


class TestModelsBadge:
    """P9.2: the fifth badge — usually absent, unmissable when the list is a fallback."""

    def test_a_window_that_never_asked_shows_nothing(self, window):
        assert window.controls.models_badge.text() == ""

    def test_the_badge_appears_with_the_reason_behind_it(self, window):
        window.panel.set_model_list_status("models: presets", "model list: offline")
        assert window.controls.models_badge.text() == "models: presets"
        assert "offline" in window.controls.models_badge.toolTip()

    def test_answering_empty_hides_it_rather_than_blanking_it(self, window):
        window.panel.set_model_list_status("models: presets", "x")
        window.panel.set_model_list_status("", "")
        assert window.controls.models_badge.text() == ""
        assert not window.controls.models_badge.isVisibleTo(window)

    def test_the_window_paints_it_from_the_engine_status(self, window, monkeypatch):
        from animetext_studio.engines.base import EngineState, EngineStatus
        monkeypatch.setattr(window.job_manager.detection, "status", lambda: EngineStatus(
            name="detection", state=EngineState.READY, loaded_models=("yolo11m",),
            backend="stub", device="cpu", detail="model list fallback (timeout)",
            inferences=1, model_list_is_fallback=True))
        window.refresh_engine_badge()
        assert window.controls.models_badge.text() == "models: presets"
        assert "timeout" in window.controls.models_badge.toolTip()
        assert window.controls.engines_badge.text() != "engines: ?"   # the badge never breaks a run


class TestModelListRefresh:
    """The badge's way out: a user action, off the UI thread, reported either way (D27)."""

    def test_refresh_replaces_the_list_and_keeps_the_selection(self, window, qtbot):
        window.job_manager.detection.refresh_models = lambda: ["yolo11m", "hub_only_variant"]
        window.panel.model_combo.setCurrentText("yolo11m")
        window.refresh_model_list()
        qtbot.waitUntil(
            lambda: window.panel.model_combo.findText("hub_only_variant") >= 0, timeout=5000)
        assert window.panel.model_combo.currentText() == "yolo11m"
        assert window.refresh_models_button.isEnabled()

    def test_a_failed_refresh_says_so_and_leaves_the_list_alone(self, window, qtbot):
        def boom():
            raise RuntimeError("no route to host")
        window.job_manager.detection.refresh_models = boom
        before = [window.panel.model_combo.itemText(i)
                  for i in range(window.panel.model_combo.count())]
        window.refresh_model_list()
        qtbot.waitUntil(lambda: "refresh failed" in window.status_label.text(), timeout=5000)
        after = [window.panel.model_combo.itemText(i)
                 for i in range(window.panel.model_combo.count())]
        assert after == before
        assert window.refresh_models_button.isEnabled()
        assert "no route to host" in window.status_label.text()

    def test_a_second_click_while_one_is_running_does_not_start_another(self, window, qtbot):
        calls = []
        release = threading.Event()

        def slow():
            calls.append(1)
            release.wait(timeout=5)
            return ["yolo11m"]
        window.job_manager.detection.refresh_models = slow
        window.refresh_model_list()
        qtbot.waitUntil(lambda: calls and window._listing_worker is not None, timeout=2000)
        window.refresh_model_list()                      # while the first is still in flight
        assert "already running" in window.status_label.text()
        release.set()


class TestReocrFromTheTable:
    """P9.3: the block table is where a flag becomes an action (right-click → Re-read)."""

    def _page(self, window, qtbot):
        assert window.load_path(SAMPLE)
        assert window.run_now()
        qtbot.waitUntil(lambda: window.current_result is not None, timeout=15000)
        return window.current_result

    def test_the_selection_is_read_from_the_rows(self, window, qtbot):
        self._page(window, qtbot)
        assert window.selected_block_ids() == []
        row = window.block_tree.topLevelItemCount() - 1
        window.block_tree.setCurrentItem(window.block_tree.topLevelItem(row))
        window.block_tree.clearSelection()
        window.block_tree.topLevelItem(row).setSelected(True)
        assert window.selected_block_ids() == [int(window.block_tree.topLevelItem(row).text(0))]

    def test_re_reading_a_block_replaces_the_document_with_a_reocr_one(self, window, qtbot):
        first = self._page(window, qtbot)
        row = window.block_tree.topLevelItemCount() - 1
        window.block_tree.setCurrentItem(window.block_tree.topLevelItem(row))
        window.block_tree.clearSelection()
        window.block_tree.topLevelItem(row).setSelected(True)
        assert window.reocr_selected() is True
        qtbot.waitUntil(lambda: window.current_result.stage == "reocr", timeout=15000)
        again = window.current_result
        assert again.block_count == first.block_count          # a whole document, not a patch
        assert [b.id for b in again.blocks] == [b.id for b in first.blocks]

    def test_no_selection_means_no_job(self, window, qtbot):
        self._page(window, qtbot)
        assert window.reocr_selected() is False
        assert window.job_manager.is_busy is False

    def test_the_menu_is_wired_to_the_table(self, window):
        from PySide6.QtCore import Qt
        assert window.block_tree.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

    def test_a_re_read_with_no_result_is_simply_false(self, window):
        """Nothing to re-read: no menu, no job, no error — the row list is empty anyway."""
        assert window.current_result is None
        assert window.reocr_selected() is False
