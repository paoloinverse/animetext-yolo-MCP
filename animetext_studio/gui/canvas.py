"""`ResultCanvas` — the interactive annotated view (PLAN Part 6).

A `QGraphicsView` showing a copy of the input image with the blocks drawn on top:
translucent fill + solid border + a label chip, hover to read the full text in a
floating panel, click to select. Zoom is fit / actual size / Ctrl+wheel, and
"Save annotated PNG" bakes the overlay at **full** resolution.

Three decisions worth stating:

* **The recipe is not re-implemented here.** Colours, chip text, chip placement, the
  nested-box draw order and click picking all come from `core/annotation.py`, which is
  also what produces the exported PNG (and, in P7, the MCP image tool). Only the brush
  differs.
* **The overlay is painted in `drawForeground`, not as one item per block.** The result is
  a list of rectangles, not something the user edits; item-per-block would mean z-order
  bookkeeping for nested boxes and a hover event per box, for no benefit.
* **Geometry in scene coordinates, sizes in device pixels.** Boxes are mapped through the
  view transform, then drawn with an identity transform, so borders and chips stay
  legible at 8 % fit-zoom of a 3000 px scan *and* at 800 %. The exported PNG is the same
  recipe at scale 1.0, where the natural (image-pixel) sizes apply.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Union

from PySide6.QtCore import QPoint, QPointF, QTimer, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap, QTransform)
from PySide6.QtWidgets import (
    QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ..core.annotation import (
    ChipOffer, OverlayStyle, border_width_for, chip_variants, draw_order, font_size_for,
    hover_text, pick_block, place_chips, rgb_of, save_annotated,
)
from ..core.imaging import pil_to_qimage
from ..core.types import PipelineResult, TextBlock

#: Zoom steps Ctrl+wheel walks through; 100 % is always reachable from either side. The
#: ends are exactly MIN/MAX_ZOOM_PERCENT (and the `gui.zoom_percent` config bounds), so no
#: step is unreachable and none is silently clamped.
ZOOM_STEPS = (0.10, 0.125, 0.17, 0.25, 0.33, 0.5, 0.67, 0.8, 1.0,
              1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
MIN_ZOOM_PERCENT = 10
MAX_ZOOM_PERCENT = 800
NO_BLOCK = -1


def _hover_style(hover_opacity: float) -> str:
    """Stylesheet for the floating text panel: dark, rounded, `hover_opacity` opaque."""
    alpha = max(0.0, min(1.0, float(hover_opacity)))
    return (
        "background-color: rgba(24, 24, 26, " + str(int(round(255 * alpha))) + ");"
        "color: #f6f6f6; padding: 6px 8px; border: 1px solid rgba(255,255,255,90);"
        "border-radius: 5px;"
    )


class _OverlayView(QGraphicsView):
    """The view half: drawing, zooming and picking. `ResultCanvas` is the widget around it."""

    def __init__(self, canvas: "ResultCanvas", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)                    # hover without a button held
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(38, 38, 40))
        self.setMinimumHeight(240)

    # -- painting -----------------------------------------------------------

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        self._canvas.paint_overlay(painter)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802 (Qt API)
        super().drawBackground(painter, rect)

    # -- interaction --------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:                  # noqa: N802
        """Ctrl+wheel zooms in discrete steps; plain wheel scrolls as usual."""
        if bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._canvas.zoom_by(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event) -> None:                              # noqa: N802
        super().resizeEvent(event)
        self._canvas.apply_zoom()

    def leaveEvent(self, event) -> None:                               # noqa: N802
        self._canvas.set_hover(NO_BLOCK)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event) -> None:                           # noqa: N802
        point = self.mapToScene(_event_pos(event))
        self._canvas.set_hover(self._canvas.pick(point.x(), point.y()))
        self._canvas.move_hover_panel(event)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:                          # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(_event_pos(event))
            self._canvas.click_at(self._canvas.pick(point.x(), point.y()))
        super().mousePressEvent(event)


def _event_pos(event: Any) -> Any:
    """Qt 6 moved `QMouseEvent.pos()` to `position()`; accept either."""
    position = getattr(event, "position", None)
    if callable(position):
        return position().toPoint()
    return event.pos()


class ResultCanvas(QWidget):
    """Annotated image view with its own command row."""

    block_selected = Signal(int)        # block id, or -1 for "clicked empty space"
    block_hovered = Signal(int)         # block id, or -1
    zoom_changed = Signal(str, int)     # ("fit" | "percent", percent)
    annotated_saved = Signal(str)       # path written

    def __init__(self, parent: Optional[QWidget] = None, *,
                 gui_config: Any = None) -> None:
        super().__init__(parent)
        self._blocks: list[TextBlock] = []
        self._result: Optional[PipelineResult] = None
        self._image: Any = None
        self._pixmap: Optional[QPixmap] = None
        self._hover_id: int = NO_BLOCK
        self._selected_id: int = NO_BLOCK
        #: Ids whose label found no room in the last paint (P8.6). Written by the painter,
        #: read by `hidden_labels`; never read *during* a paint, so it cannot feedback-loop.
        self._hidden_labels: tuple[int, ...] = ()
        #: The de-collision layout, reused across repaints (P8.6). It is a pure function of
        #: the blocks, the style and the view — hover is *not* an input, which is exactly why
        #: it may be cached: moving the mouse must not reshuffle anybody else's label, and a
        #: repaint per mouse move re-running it would cost ~9 ms on a 300-box page.
        self._chip_plan: Any = None
        self._chip_key: tuple = ()
        self._blocks_rev = 0              # bumped whenever `_blocks` is replaced
        self._cursor: Any = None        # last cursor position, viewport coordinates
        self._zoom_mode = "fit"
        self._zoom_percent = 100
        self._hover_opacity = 0.5
        self._style = OverlayStyle.from_config(gui_config)
        if gui_config is not None:
            self._zoom_mode = str(getattr(gui_config, "zoom_mode", "fit"))
            self._zoom_percent = int(getattr(gui_config, "zoom_percent", 100))
            self._hover_opacity = float(getattr(gui_config, "hover_opacity", 0.5))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.view = _OverlayView(self, self)
        self.scene = QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pixmap_item)

        self.fit_button = QPushButton("Fit", self)
        self.fit_button.setToolTip("Fit the page to the view")
        self.fit_button.clicked.connect(self.zoom_fit)
        self.actual_button = QPushButton("100%", self)
        self.actual_button.setToolTip("Actual size (Ctrl+wheel also zooms)")
        self.actual_button.clicked.connect(self.zoom_actual)
        self.save_button = QPushButton("Save annotated PNG…", self)
        self.save_button.setToolTip("Bake the boxes into a full-resolution PNG")
        self.save_button.clicked.connect(self._save_button_pressed)
        self.zoom_label = QLabel("no page", self)
        self.zoom_label.setObjectName("zoomLabel")
        strip = QHBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(6)
        for widget in (self.fit_button, self.actual_button, self.save_button):
            widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
            strip.addWidget(widget)
        strip.addWidget(self.zoom_label, 1)
        layout.addLayout(strip)
        layout.addWidget(self.view, 1)

        #: Floating hover panel: a child of the viewport, so it is clipped by it (and
        #: appears in `viewport().grab()` screenshots) instead of being a separate window.
        self.hover_panel = QLabel(self.view.viewport())
        self.hover_panel.setObjectName("hoverPanel")
        self.hover_panel.setWordWrap(True)
        self.hover_panel.setTextFormat(Qt.TextFormat.PlainText)
        self.hover_panel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.hover_panel.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hover_panel.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.hover_panel.setStyleSheet(_hover_style(self._hover_opacity))
        self.hover_panel.hide()

    # -- content ------------------------------------------------------------

    def set_result(self, result: Optional[PipelineResult],
                   image: Any = None) -> None:
        """Show `result`'s blocks over `image` (the window's own PIL copy)."""
        self._result = result
        self._blocks = list(result.blocks) if result is not None else []
        self._blocks_rev += 1             # invalidates the cached chip layout
        self._hover_id = NO_BLOCK
        self._selected_id = NO_BLOCK
        self.hover_panel.hide()
        if image is not None:
            self.set_image(image)
        else:
            self.view.viewport().update()

    def set_image(self, image: Any) -> None:
        """New page: replace the pixmap, and with it any boxes left over from the old one.

        The QImage is built *before* anything is mutated, so a `StudioError` here (no
        PySide6 runtime, an image we cannot convert) leaves the canvas showing the page it
        was showing, not half a new one.
        """
        pixmap = QPixmap.fromImage(pil_to_qimage(image))
        self._image = image
        self._pixmap = pixmap
        self._pixmap_item.setPixmap(self._pixmap)
        self.scene.setSceneRect(QRectF(self._pixmap.rect()))
        self.apply_zoom()
        self.view.viewport().update()

    def clear(self) -> None:
        self._result = None
        self._blocks = []
        self._blocks_rev += 1             # invalidates the cached chip layout
        self._hover_id = NO_BLOCK
        self._selected_id = NO_BLOCK
        self._hidden_labels = ()
        self.hover_panel.hide()
        self.view.viewport().update()

    @property
    def blocks(self) -> tuple[TextBlock, ...]:
        return tuple(self._blocks)

    @property
    def selected_block_id(self) -> int:
        return self._selected_id

    @property
    def hovered_block_id(self) -> int:
        return self._hover_id

    @property
    def hidden_labels(self) -> tuple[int, ...]:
        """Ids whose chip had no room in the last paint — the de-collision report (P8.6).

        Diagnostic, not a warning: those blocks keep their box, their row in the table, and
        their full text in the hover panel. Zooming in usually empties this list, because
        the room a chip needs is measured in screen pixels.
        """
        return self._hidden_labels

    @property
    def has_result(self) -> bool:
        return bool(self._blocks)

    def block_ids(self) -> list[int]:
        return [int(b.id) for b in self._blocks]

    def block_by_id(self, block_id: int) -> Optional[TextBlock]:
        return next((b for b in self._blocks if int(b.id) == int(block_id)), None)

    def pick(self, scene_x: float, scene_y: float) -> int:
        """Block id under a scene point, or -1 (smallest box wins: nested boxes, backlog 9)."""
        block = pick_block(self._blocks, scene_x, scene_y)
        return NO_BLOCK if block is None else int(block.id)

    # -- style --------------------------------------------------------------

    def set_overlay_visible(self, visible: bool) -> None:
        self._style = replace(self._style, overlay=bool(visible))
        self.view.viewport().update()

    def set_labels_visible(self, visible: bool) -> None:
        self._style = replace(self._style, labels=bool(visible))
        self.view.viewport().update()

    def set_fill_opacity(self, opacity: float) -> None:
        self._style = replace(self._style, fill_opacity=max(0.0, min(1.0, float(opacity))))
        self.view.viewport().update()

    def set_hover_opacity(self, opacity: float) -> None:
        self._hover_opacity = max(0.0, min(1.0, float(opacity)))
        self.hover_panel.setStyleSheet(_hover_style(self._hover_opacity))

    @property
    def style_spec(self) -> OverlayStyle:
        return self._style

    # -- hover / selection --------------------------------------------------

    def set_hover(self, block_id: int) -> None:
        block_id = int(block_id)
        if block_id == self._hover_id:
            return
        self._hover_id = block_id
        block = None if block_id == NO_BLOCK else self.block_by_id(block_id)
        if block is None:
            self.hover_panel.hide()
        else:
            self.hover_panel.setText(hover_text(block))
            self.hover_panel.show()
            self._place_panel()
        self.view.viewport().update()
        self.block_hovered.emit(self._hover_id)

    def flash_hover(self, block_id: int, duration_ms: int) -> None:
        """Hover `block_id` for `duration_ms` and then clear it (P7: `highlight_block`).

        The timer is a **child of the canvas**, not a `QTimer.singleShot` with a lambda: a
        singleShot keeps running after the window is destroyed and fires into a deleted C++
        object (`RuntimeError: ... QLabel already deleted`) if the user quits mid-flash.
        """
        self.set_hover(block_id)
        if duration_ms > 0:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.set_hover(-1))
            timer.start(max(50, int(duration_ms)))

    def move_hover_panel(self, event: Any) -> None:
        """Follow the cursor (`mouseMoveEvent`); the panel never leaves the viewport."""
        self._cursor = _event_pos(event)
        self._place_panel()

    def _place_panel(self) -> None:
        """Position the hover panel beside the cursor, flipping at the viewport edges.

        A selection made from the block table or the JSON pane has no cursor, so it lands
        at the top-left of the view rather than at whatever stale place the mouse left it.
        """
        if self._hover_id == NO_BLOCK or not self.hover_panel.isVisible():
            return
        area = self.view.viewport().rect()
        self.hover_panel.setMaximumWidth(max(160, int(area.width() * 0.55)))
        self.hover_panel.adjustSize()
        panel = self.hover_panel.size()
        cursor = self._cursor
        if cursor is None:
            self.hover_panel.move(max(0, min(12, area.right() - panel.width())),
                                  max(0, min(12, area.bottom() - panel.height())))
            return
        x, y = cursor.x() + 14, cursor.y() + 18
        if x + panel.width() > area.right():
            x = max(area.left(), cursor.x() - panel.width() - 12)
        if y + panel.height() > area.bottom():
            y = max(area.top(), cursor.y() - panel.height() - 12)
        self.hover_panel.move(x, y)

    def click_at(self, block_id: int) -> None:
        self.select_block(block_id, emit=True)

    def select_block(self, block_id: int, *, emit: bool = False) -> bool:
        """Select by id (-1 clears). Returns False when the id is not on this page."""
        block_id = int(block_id)
        if block_id != NO_BLOCK and self.block_by_id(block_id) is None:
            return False
        self._selected_id = block_id
        self.view.viewport().update()
        if emit:
            self.block_selected.emit(block_id)
        return True

    # -- zoom ---------------------------------------------------------------

    @property
    def zoom_mode(self) -> str:
        return self._zoom_mode

    @property
    def zoom_percent(self) -> int:
        return self._zoom_percent

    def set_zoom(self, mode: Optional[str] = None,
                 percent: Optional[int] = None) -> None:
        """Set the mode, the percentage, or both (a percentage *means* percent mode).

        Emits `zoom_changed` so the control panel mirrors a zoom the user triggered here
        (Fit, 100 %, Ctrl+wheel); the panel writes it back programmatically, so there is
        no echo loop — just one signal per user action.
        """
        before = (self._zoom_mode, self._zoom_percent)
        if mode is not None and str(mode) in ("fit", "percent"):
            self._zoom_mode = str(mode)
        if percent is not None:
            self._zoom_percent = max(MIN_ZOOM_PERCENT, min(MAX_ZOOM_PERCENT, int(percent)))
            if mode is None:
                self._zoom_mode = "percent"      # a bare percentage means "show me that"
        self.apply_zoom()
        if (self._zoom_mode, self._zoom_percent) != before:
            self.zoom_changed.emit(self._zoom_mode, self._zoom_percent)

    def zoom_fit(self) -> None:
        self.set_zoom(mode="fit")

    def zoom_actual(self) -> None:
        self.set_zoom(mode="percent", percent=100)

    def zoom_by(self, steps: int) -> None:
        """Ctrl+wheel: walk `ZOOM_STEPS` from the level nearest the current scale."""
        steps = int(steps)
        if steps == 0:
            return
        current = self._zoom_percent / 100.0
        ordered = sorted(set(ZOOM_STEPS))
        nearest = min(range(len(ordered)), key=lambda i: abs(ordered[i] - current))
        index = max(0, min(len(ordered) - 1, nearest + steps))
        self.set_zoom(mode="percent", percent=int(round(ordered[index] * 100)))

    def apply_zoom(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        if self._zoom_mode == "fit":
            size = self.view.viewport().size()
            if size.width() < 2 or size.height() < 2:
                return
            self.view.fitInView(self._pixmap_item,
                                Qt.AspectRatioMode.KeepAspectRatio)
        else:
            factor = self._zoom_percent / 100.0
            self.view.resetTransform()
            self.view.scale(factor, factor)
        self._update_zoom_label()

    def current_scale(self) -> float:
        transform = self.view.transform()
        return float(transform.m11()) if transform.m11() else 1.0

    def _update_zoom_label(self) -> None:
        shown = int(round(self.current_scale() * 100))
        self.zoom_label.setText(f"{shown}% ({self._zoom_mode})")

    # -- export -------------------------------------------------------------

    def _save_button_pressed(self) -> None:
        """Button slot. `clicked` carries a `checked` bool, so it must never reach
        `save_annotated(path=…)`: a raw `connect(self.save_annotated)` would write a file
        named "False" instead of opening the dialog (same trap as the P5 config buttons).
        """
        self.save_annotated()

    def save_annotated(self, path: Union[str, Path, None] = None) -> Optional[str]:
        """Bake the overlay at full resolution. Asks for a path when none is given."""
        if self._image is None:
            return None
        if path is None:
            suggested = "annotated.png"
            if self._result is not None and self._result.image.file_name:
                suggested = Path(self._result.image.file_name).stem + ".annotated.png"
            path, _ = QFileDialog.getSaveFileName(self, "Save annotated PNG", suggested,
                                                  "PNG (*.png)")
            if not path:
                return None
        written = save_annotated(self._image, self._blocks, path, self._style,
                                 highlight=None if self._hover_id == NO_BLOCK
                                 else self._hover_id,
                                 selected=None if self._selected_id == NO_BLOCK
                                 else self._selected_id)
        self.annotated_saved.emit(str(written))
        return str(written)

    # -- the painter (device pixels; geometry from the scene transform) ------

    def paint_overlay(self, painter: QPainter) -> None:
        if not self._blocks or not self._style.overlay:
            return
        if self._pixmap is None or self._pixmap.isNull():
            return
        transform = painter.worldTransform()          # QTransform, scene -> viewport
        viewport = QRectF(self.view.viewport().rect())
        scale = transform.m11() or 1.0
        page = (self._pixmap.width(), self._pixmap.height())
        # Chip and border sizes follow the zoom inside a legible band: the export's natural
        # size at 100 %, never so small that a fitted 3000 px scan has invisible labels,
        # and never so large that a fitted page is nothing but chips (T9 screenshot review).
        font = QFont(self.view.font())
        font.setPixelSize(min(20, max(8,
                                      int(round(font_size_for(page, self._style) * scale)))))
        metrics = QFontMetrics(font)
        pen_base = min(4.0, max(1.0, border_width_for(page, self._style) * scale))

        painter.save()
        painter.setTransform(QTransform())          # draw in viewport pixels
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # The font that *measured* the chips is the font that must draw them. Without this
        # line `drawText` used the widget's default font, so every label was wider than the
        # chip it was measured for and Qt clipped both ends — `#0·0.80·この攻撃で決まりだ！`
        # came out as `.80 この攻撃で決ま`. It also made the de-collision maths lie, since
        # `place_chips` reserves the space the *measured* text needs (P8.6).
        painter.setFont(font)
        drawn: list = []
        for block in draw_order(self._blocks):
            color = QColor(*rgb_of(getattr(block, "color", ""), fallback_index=int(block.id)))
            active = int(block.id) in (self._hover_id, self._selected_id)
            device = transform.mapRect(_qrectf(block.bbox))
            if not device.intersects(viewport):
                continue
            alpha = int(round(255 * self._style.fill_opacity
                              * (self._style.highlight_fill_boost if active else 1.0)))
            fill = QColor(color)
            fill.setAlpha(max(0, min(255, alpha)))
            painter.fillRect(device, fill)
            pen = QPen(color, pen_base * (2.0 if active else 1.0))
            painter.setPen(pen)
            painter.drawRect(device)
            drawn.append((block, color, device))
        # Second pass: every chip above every box, laid out by the shared rule
        # (`place_chips`) so the window and the exported PNG cannot disagree about where a
        # label sits (P8.6). Only the boxes that reached this viewport are offered, which is
        # why zooming in brings labels back: the room to lay them out is the room on screen.
        if self._style.labels:
            page = (int(viewport.width()), int(viewport.height()))
            colors = {int(block.id): color for block, color, _d in drawn}
            # Recompute only when something the layout depends on changed — the blocks, the
            # style, the zoom, the scroll, the viewport size or the font. Not on hover.
            key = (self._blocks_rev, self._style, _matrix_of(transform), page,
                   font.pixelSize())
            if self._chip_key != key:
                self._chip_plan = place_chips(
                    [self._offer(block, device, metrics, page)
                     for block, _c, device in drawn],
                    style=self._style,
                    image_size=page,
                )
                self._chip_key = key
                self._hidden_labels = self._chip_plan.hidden
            plan = self._chip_plan
            for chip in plan.paintable:
                self._paint_chip(painter, chip, colors[int(chip.block.id)])
            # The block under the cursor or already selected always keeps its label, painted
            # last so it covers whoever was in the way. The layout above ignores hover state,
            # so moving the mouse never makes other labels jump.
            for block, color, _d in drawn:
                if int(block.id) in (self._hover_id, self._selected_id) \
                        and not plan.painted(int(block.id)):
                    wanted = plan.wanted(int(block.id))
                    if wanted is not None:
                        self._paint_chip(painter, wanted, color)
        else:
            self._hidden_labels = ()
        painter.restore()

    def _offer(self, block: Any, device: QRectF, metrics: QFontMetrics,
               page: tuple[int, int]) -> ChipOffer:
        """Measure this block's chip texts with Qt, for the shared de-collision rule.

        The chip is sized for the *zoomed* box, so at 25 % a long sentence is measured
        against a chip the font can barely spell — which is exactly when the `#id` variant
        earns its keep.
        """
        box = (int(device.left()), int(device.top()), int(device.right()), int(device.bottom()))
        variants = []
        for text in chip_variants(block, max_chars=self._style.label_max_chars):
            variants.append((text, metrics.horizontalAdvance(text), metrics.height()))
        return ChipOffer(block=block, box=box, variants=tuple(variants))

    @staticmethod
    def _paint_chip(painter: QPainter, chip: Any, color: QColor) -> None:
        rect = _qrect(chip.rect)
        painter.fillRect(rect, color)
        painter.setPen(QColor(0, 0, 0) if color.lightnessF() > 0.6 else QColor(255, 255, 255))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, chip.text)


def _matrix_of(transform: QTransform) -> tuple[float, ...]:
    """A `QTransform` as a hashable tuple, for the cached chip layout's key.

    PySide6 has no `QTransform.matrix()`; the nine accessors are the portable way to say
    "the zoom and the scroll position", which is all the key needs (P8.6).
    """
    return (transform.m11(), transform.m12(), transform.m13(),
            transform.m21(), transform.m22(), transform.m23(),
            transform.m31(), transform.m32(), transform.m33())


def _rect(bbox: Any) -> tuple[float, float, float, float]:
    """`(x0, y0, x1, y1)` — the box we store everywhere, as floats.

    Kept as xyxy on purpose: `QRectF` takes **width and height**, so `QRectF(*_rect(...))`
    would paint a box whose right edge is `x0 + x1` (T9 — caught by the pixel probes, and
    the reason the canvas and the PNG disagreed until the rect is built with `_qrectf`).
    """
    return (float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1))


def _qrectf(bbox: Any) -> QRectF:
    """A stored bbox as a `QRectF`, converting xyxy into Qt's x, y, w, h."""
    x0, y0, x1, y1 = _rect(bbox)
    return QRectF(QPointF(x0, y0), QPointF(x1, y1))


def _qrect(rect: tuple[float, float, float, float]) -> QRect:
    """An xyxy rectangle as a `QRect`.

    `QRect(x, y, w, h)` and `QRect(QPoint(a), QPoint(b))` are the two constructors that
    look identical to a slip of the hand; the second is the corner-to-corner one, whose
    bottom-right is inclusive, hence the -1. (The first version of this passed xyxy into
    the first form and painted a chip the size of the box below it.)
    """
    return QRect(QPoint(int(round(rect[0])), int(round(rect[1]))),
                 QPoint(int(round(rect[2])) - 1, int(round(rect[3])) - 1))
