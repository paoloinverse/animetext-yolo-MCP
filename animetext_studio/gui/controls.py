"""The left control panel: every knob the pipeline takes, plus the run buttons.

Built so the window can be constructed and driven headlessly: no dialogs are opened from
inside a widget (Export/Import *emit* a request and the window decides), every value is a
real widget with a bounded range, and `apply_config()` / `collect_into()` are the only ways
state crosses the boundary — so a test can set the whole panel without touching pixels.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..core.config import AppConfig, MODEL_PRESETS
from ..core.pipeline import order_detections  # noqa: F401  (documents the modes live here)

#: kept in sync with `config.GuiConfig.zoom_percent` bounds (validated there too)
ZOOM_MIN_PERCENT = 10
ZOOM_MAX_PERCENT = 800

READING_ORDERS = ("manga_rtl", "top_to_bottom", "score")
READING_ORDER_LABELS = {
    "manga_rtl": "Manga (right→left)",
    "top_to_bottom": "Top→bottom, left→right",
    "score": "Most confident first",
}
DEVICES = ("auto", "cpu", "cuda", "cuda:0", "cuda:1")
#: The budgets the vendored processor implements; "auto" routes between them per crop.
#: 512 was missing until P8.2b — see `core.config.MAX_PATCHES_CHOICES`.
PATCH_BUDGETS = ("auto", "128", "192", "256", "384", "512")


def _link(slider: QSlider, spin: QDoubleSpinBox) -> None:
    """Keep a 0–100 slider and a 0.00–1.00 spin box in step (single source: the spin)."""
    slider.setValue(int(round(spin.value() * 100)))

    def from_slider(value: int) -> None:
        spin.setValue(value / 100.0)

    def from_spin(value: float) -> None:
        if int(round(value * 100)) != slider.value():
            slider.setValue(int(round(value * 100)))

    slider.valueChanged.connect(from_slider)
    spin.valueChanged.connect(from_spin)


class ControlPanel(QWidget):
    """Model + thresholds + OCR group + reading order + run controls."""

    generate_requested = Signal()
    cancel_requested = Signal()
    export_requested = Signal()
    import_requested = Signal()
    model_changed = Signal(str)
    #: any `gui.*` widget moved — the window pushes the new view settings to the canvas
    view_changed = Signal()

    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QWidget] = None,
                 *, mock: bool = False) -> None:
        super().__init__(parent)
        self._mock = mock
        self._catalog: list[dict[str, Any]] = []
        self._programmatic = False          # suppress model_changed while applying a config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._badges())
        layout.addWidget(self._detection_group())
        layout.addWidget(self._ocr_group())
        layout.addWidget(self._order_group())
        layout.addWidget(self._view_group())
        layout.addWidget(self._run_group())
        layout.addWidget(self._config_group())
        layout.addStretch(1)
        self.apply_config(config or AppConfig.default())

    # -- building blocks ----------------------------------------------------

    def _badges(self) -> QWidget:
        box = QWidget(self)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self.mock_badge = QLabel("MOCK ENGINES" if self._mock else "REAL ENGINES", box)
        self.offline_badge = QLabel("", box)
        #: P7 flips this to "MCP :8765"; until then it states the truth (not started).
        self.mcp_badge = QLabel("MCP: off", box)
        #: P8: engine state must be visible — `unload_engines` used to leave no trace on
        #: screen but a log row. `MainWindow.refresh_engine_badge()` owns the wording.
        self.engines_badge = QLabel("engines: —", box)
        #: P9.2: "models: presets" when the variant list came from the bundle + compiled presets
        #: rather than the repo. Empty otherwise, so it is never noise — it only appears when the
        #: list is a fallback, which is exactly when a user picks a model and should know.
        self.models_badge = QLabel("", box)
        for badge in (self.mock_badge, self.offline_badge, self.mcp_badge, self.engines_badge,
                      self.models_badge):
            badge.setObjectName("badge")
            row.addWidget(badge)
        row.addStretch(1)
        return box

    def set_mcp_status(self, text: str) -> None:
        """Called by the MCP server (P7) when it starts/stops."""
        self.mcp_badge.setText(text or "MCP: off")

    def set_engine_status(self, text: str) -> None:
        """Show the engine state (P8). Empty text restores the neutral placeholder."""
        self.engines_badge.setText(text or "engines: —")

    def set_model_list_status(self, text: str, detail: str = "") -> None:
        """Show "models: presets" while the repo listing did not answer (P9.2).

        Empty ``text`` hides the badge entirely — a badge that is always there becomes scenery,
        and scenery does not carry a warning. ``detail`` (the engine's own reason) goes in the
        tooltip, where it is available to whoever wants it and invisible to whoever does not.
        """
        self.models_badge.setText(text or "")
        self.models_badge.setToolTip(detail or "")
        self.models_badge.setVisible(bool(text))

    def _detection_group(self) -> QWidget:
        box = QGroupBox("Detection", self)
        form = QFormLayout(box)
        self.model_combo = QComboBox(box)
        self.model_combo.currentTextChanged.connect(self._on_model_text)
        form.addRow("Model", self.model_combo)

        self.iou_spin = QDoubleSpinBox(box)
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        self.iou_slider = QSlider(Qt.Orientation.Horizontal, box)
        self.iou_slider.setRange(0, 100)
        iou_row = QHBoxLayout()
        iou_row.addWidget(self.iou_slider, 1)
        iou_row.addWidget(self.iou_spin)
        _link(self.iou_slider, self.iou_spin)
        form.addRow("IoU", iou_row)

        self.score_spin = QDoubleSpinBox(box)
        self.score_spin.setRange(0.0, 1.0)
        self.score_spin.setSingleStep(0.01)
        self.score_spin.setDecimals(2)
        self.score_slider = QSlider(Qt.Orientation.Horizontal, box)
        self.score_slider.setRange(0, 100)
        score_row = QHBoxLayout()
        score_row.addWidget(self.score_slider, 1)
        score_row.addWidget(self.score_spin)
        _link(self.score_slider, self.score_spin)
        self.recommended_button = QPushButton("Recommended", box)
        self.recommended_button.setToolTip("Use the threshold the model was shipped with")
        self.recommended_button.clicked.connect(self._use_recommended_threshold)
        score_row.addWidget(self.recommended_button)
        form.addRow("Score", score_row)
        return box

    def _ocr_group(self) -> QWidget:
        box = QGroupBox("OCR", self)
        form = QFormLayout(box)
        self.device_combo = QComboBox(box)
        self.device_combo.addItems(DEVICES)
        self.device_combo.setEditable(True)          # cuda:2 on a box with more GPUs
        form.addRow("Device", self.device_combo)

        self.patches_combo = QComboBox(box)
        self.patches_combo.addItems(PATCH_BUDGETS)
        form.addRow("Max patches", self.patches_combo)

        self.batch_spin = QSpinBox(box)
        self.batch_spin.setRange(1, 64)
        form.addRow("Batch size", self.batch_spin)

        self.tokens_spin = QSpinBox(box)
        self.tokens_spin.setRange(16, 1024)
        self.tokens_spin.setSingleStep(16)
        form.addRow("Max new tokens", self.tokens_spin)

        self.padding_spin = QSpinBox(box)
        self.padding_spin.setRange(0, 16)
        form.addRow("Crop padding", self.padding_spin)

        self.nfkc_check = QCheckBox("Normalise text (NFKC)", box)
        form.addRow(self.nfkc_check)
        return box

    def _order_group(self) -> QWidget:
        box = QGroupBox("Layout", self)
        form = QFormLayout(box)
        self.order_combo = QComboBox(box)
        for mode in READING_ORDERS:
            self.order_combo.addItem(READING_ORDER_LABELS[mode], mode)
        form.addRow("Reading order", self.order_combo)
        return box

    def _view_group(self) -> QWidget:
        """Overlay/label/fill/zoom — the `gui.*` section, so it exports and imports like
        everything else. The canvas is the *consumer*; nothing here draws anything."""
        box = QGroupBox("View", self)
        form = QFormLayout(box)
        self.overlay_check = QCheckBox("Show boxes", box)
        self.overlay_check.toggled.connect(self._on_view_changed)
        self.labels_check = QCheckBox("Show labels", box)
        self.labels_check.toggled.connect(self._on_view_changed)
        form.addRow(self.overlay_check)
        form.addRow(self.labels_check)

        self.opacity_spin = QDoubleSpinBox(box)
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.02)
        self.opacity_spin.setDecimals(2)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, box)
        self.opacity_slider.setRange(0, 100)
        _link(self.opacity_slider, self.opacity_spin)
        self.opacity_spin.valueChanged.connect(self._on_view_changed)
        form.addRow("Fill", self.opacity_slider)

        self.zoom_combo = QComboBox(box)
        self.zoom_combo.addItem("Fit to view", "fit")
        self.zoom_combo.addItem("Actual size %", "percent")
        self.zoom_combo.currentIndexChanged.connect(self._on_view_changed)
        form.addRow("Zoom", self.zoom_combo)

        self.zoom_spin = QSpinBox(box)
        self.zoom_spin.setRange(ZOOM_MIN_PERCENT, ZOOM_MAX_PERCENT)
        self.zoom_spin.setSingleStep(25)
        self.zoom_spin.setSuffix(" %")
        self.zoom_spin.valueChanged.connect(self._on_zoom_spin_changed)
        form.addRow("Percent", self.zoom_spin)
        return box

    def _on_zoom_spin_changed(self, _value: int) -> None:
        """Typing a percentage *is* choosing percent mode — flip the combo with it."""
        if self._programmatic:
            return
        index = self.zoom_combo.findData("percent")
        if index >= 0 and self.zoom_combo.currentIndex() != index:
            self._programmatic = True
            try:
                self.zoom_combo.setCurrentIndex(index)
            finally:
                self._programmatic = False
        self.view_changed.emit()

    def set_view_zoom(self, mode: str, percent: int) -> None:
        """Mirror a zoom the canvas chose itself (its Fit/100% buttons, Ctrl+wheel).

        Written without firing `view_changed`: the canvas already has the value, and an
        echo back to it would be a loop with a float comparison at the end of it.
        """
        self._programmatic = True
        try:
            index = self.zoom_combo.findData(str(mode))
            if index >= 0:
                self.zoom_combo.setCurrentIndex(index)
            self.zoom_spin.setValue(int(percent))
        finally:
            self._programmatic = False

    def _on_view_changed(self, *_args) -> None:
        if not self._programmatic:
            self.view_changed.emit()

    def _run_group(self) -> QWidget:
        box = QGroupBox("Run", self)
        row = QHBoxLayout(box)
        self.generate_button = QPushButton("Generate", box)
        self.generate_button.setDefault(True)
        self.generate_button.clicked.connect(self.generate_requested)
        self.cancel_button = QPushButton("Cancel", box)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)
        row.addWidget(self.generate_button, 1)
        row.addWidget(self.cancel_button)
        return box

    def _config_group(self) -> QWidget:
        box = QGroupBox("Configuration", self)
        row = QHBoxLayout(box)
        self.export_button = QPushButton("Export…", box)
        self.export_button.clicked.connect(self.export_requested)
        self.import_button = QPushButton("Import…", box)
        self.import_button.clicked.connect(self.import_requested)
        row.addWidget(self.export_button)
        row.addWidget(self.import_button)
        row.addStretch(1)
        return box

    # -- model combo --------------------------------------------------------

    def set_models(self, catalog: Iterable[dict[str, Any]], *, offline: bool = False) -> None:
        """Fill the combo from `model_catalog()` records; offline hides what isn't bundled.

        Never fetches anything itself: the catalog comes from the caller (the window reads
        the on-disk store at startup, P7 can push a live Hub list later). Unbundled
        variants stay visible but disabled with a tooltip, so the combo does not silently
        change shape between online and offline runs.
        """
        self._programmatic = True
        self.model_combo.clear()
        self._catalog = list(catalog)
        for record in self._catalog:
            name = str(record.get("name"))
            bits = []
            if record.get("params_m"):
                bits.append(f"{record['params_m']} M params")
            if record.get("flops_g"):
                bits.append(f"{record['flops_g']} GFLOPs")
            if record.get("f1") is not None:
                bits.append(f"F1 {record['f1']}")
            if record.get("recommended_score_threshold") is not None:
                bits.append(f"recommended score {record['recommended_score_threshold']}")
            source = record.get("source", "hf")
            bits.append("weights: on-premises" if source == "bundled" else "weights: Hugging Face")
            item = self.model_combo.addItem(name)
            index = self.model_combo.count() - 1
            self.model_combo.setItemData(index, " | ".join(bits), Qt.ItemDataRole.ToolTipRole)
            if offline and source != "bundled":
                # QComboBox has no setItemEnabled: the view reads the model item's flags.
                self.model_combo.model().item(index).setEnabled(False)
                self.model_combo.setItemData(
                    index, "Not on premises; run scripts/fetch_models.py --variants "
                           f"{name}", Qt.ItemDataRole.ToolTipRole)
        self._programmatic = False

    def current_model(self) -> str:
        return self.model_combo.currentText()

    @property
    def model_count(self) -> int:
        return self.model_combo.count()

    def _on_model_text(self, text: str) -> None:
        if not self._programmatic and text:
            self.model_changed.emit(text)

    def _use_recommended_threshold(self) -> None:
        record = next((r for r in self._catalog
                       if r.get("name") == self.model_combo.currentText()), None)
        value = (record or {}).get("recommended_score_threshold")
        if value is None:
            preset = MODEL_PRESETS.get(self.model_combo.currentText(), {})
            value = preset.get("recommended_score_threshold")
        if value is not None:
            self.score_spin.setValue(float(value))

    # -- config boundary ----------------------------------------------------

    def apply_config(self, config: AppConfig) -> None:
        self._programmatic = True
        try:
            self.iou_spin.setValue(float(config.detection.iou_threshold))
            self.score_spin.setValue(float(config.detection.score_threshold))
            index = self.model_combo.findText(config.detection.model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            self.device_combo.setCurrentText(config.ocr.device)
            self.patches_combo.setCurrentText(str(config.ocr.max_patches))
            self.batch_spin.setValue(int(config.ocr.batch_size))
            self.tokens_spin.setValue(int(config.ocr.max_new_tokens))
            self.padding_spin.setValue(int(config.ocr.crop_padding_px))
            self.nfkc_check.setChecked(bool(config.ocr.normalize_nfkc))
            order_index = self.order_combo.findData(config.pipeline.reading_order)
            if order_index >= 0:
                self.order_combo.setCurrentIndex(order_index)
            self.overlay_check.setChecked(bool(config.gui.overlay_visible))
            self.labels_check.setChecked(bool(config.gui.labels_visible))
            self.opacity_spin.setValue(float(config.gui.fill_opacity))
            zoom_index = self.zoom_combo.findData(str(config.gui.zoom_mode))
            if zoom_index >= 0:
                self.zoom_combo.setCurrentIndex(zoom_index)
            self.zoom_spin.setValue(int(config.gui.zoom_percent))
        finally:
            self._programmatic = False
        self.set_offline(bool(config.paths.offline))

    def collect_into(self, config: AppConfig) -> AppConfig:
        """Copy the widget values onto a copy of `config` and return it."""
        out = AppConfig.from_dict(config.to_dict())
        out.detection.model = self.current_model()
        out.detection.iou_threshold = round(float(self.iou_spin.value()), 2)
        out.detection.score_threshold = round(float(self.score_spin.value()), 2)
        out.ocr.device = self.device_combo.currentText().strip() or "auto"
        text = self.patches_combo.currentText()
        out.ocr.max_patches = "auto" if text == "auto" else int(text)
        out.ocr.batch_size = int(self.batch_spin.value())
        out.ocr.max_new_tokens = int(self.tokens_spin.value())
        out.ocr.crop_padding_px = int(self.padding_spin.value())
        out.ocr.normalize_nfkc = bool(self.nfkc_check.isChecked())
        out.pipeline.reading_order = self.order_combo.currentData()
        out.gui.overlay_visible = bool(self.overlay_check.isChecked())
        out.gui.labels_visible = bool(self.labels_check.isChecked())
        out.gui.fill_opacity = round(float(self.opacity_spin.value()), 2)
        out.gui.zoom_mode = self.zoom_combo.currentData() or "fit"
        out.gui.zoom_percent = int(self.zoom_spin.value())
        return out

    # -- run-state helpers --------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        self.generate_button.setEnabled(not busy)
        self.cancel_button.setEnabled(bool(busy))

    def set_offline(self, offline: bool) -> None:
        self.offline_badge.setText("OFFLINE" if offline else "")
        self.offline_badge.setVisible(bool(offline))
