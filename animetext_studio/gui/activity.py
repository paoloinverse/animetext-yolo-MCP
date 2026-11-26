"""P7 - the MCP activity dock: the "visual confirmation" half of the bridge.

A remote call must be as visible as a mouse click. `ActivityLog` already carries one entry
per call; this module is the Qt end of that stream:

* a dock widget with the recent calls (time, tool, arguments, outcome, duration),
* the status-bar line for the call that just arrived,
* and a short **flash** on the widget the call actually touched - which is how you see a
  harness move your zoom, replace your page or apply a config while you are watching.

The dock listens to the *log*, not to the server, so it works identically for the HTTP and
the stdio transport, and it keeps its history even if the server is restarted.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QDockWidget, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..mcp.bridge import ActivityEvent, ActivityLog

__all__ = ["ActivityDock", "attach_activity_dock", "flash_widget", "widget_for"]

#: How long a flash lasts. Long enough to notice, short enough not to lie about state.
FLASH_MS = 700
_FLASH_COLOR = "#ffd166"
_FLASH_ERROR = "#ff6b6b"

#: tool -> the widget(s) that call changes. Keys are MainWindow attribute paths, resolved
#: lazily (`"controls.model_combo"`), so this module never has to know how the window is built.
TOOL_TARGETS: dict[str, tuple[str, ...]] = {
    "list_models": ("controls.model_combo",),
    "set_config": ("controls",),
    "import_config": ("controls", "import_button"),
    "export_config": ("export_button",),
    "load_image": ("image_label",),
    "detect": ("canvas",),
    "run_pipeline": ("canvas", "json_pane"),
    "cancel_job": ("progress",),
    "get_result": ("json_pane",),
    "save_result": ("json_pane",),
    "get_annotated_image": ("canvas",),
    "set_overlay_visible": ("canvas",),
    "set_zoom": ("canvas",),
    "highlight_block": ("canvas",),
    "reset": ("canvas", "image_label", "json_pane"),
    "unload_engines": ("controls",),
}


def widget_for(window: Any, path: str) -> Optional[QWidget]:
    """Resolve "controls.model_combo" against a window, tolerating a renamed widget."""
    current: Any = window
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, QWidget) else None


def flash_widget(widget: Optional[QWidget], *, ms: int = FLASH_MS,
                 color: str = _FLASH_COLOR) -> None:
    """Briefly tint **one** widget, then put its stylesheet back exactly as it was.

    Two details cost a screenshot each to learn:

    * the selector is `#objectName`, never `*` - a bare `*` rule inherits into every child,
      and flashing the control panel used to redden the whole left column;
    * a plain QWidget only paints a stylesheet background with `WA_StyledBackground`, which
      these widgets do not set, so it is switched on for the flash and restored after.
    """
    if widget is None:
        return
    original = widget.styleSheet()
    name = widget.objectName()
    if not name:
        name = f"mcpFlashTarget{id(widget):x}"
        widget.setObjectName(name)
    styled_before = bool(widget.testAttribute(Qt.WidgetAttribute.WA_StyledBackground))
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setProperty("mcpFlash", True)
    widget.setStyleSheet(f"{original}\n#{name} {{ background-color: {color}; }}")
    QTimer.singleShot(int(ms),
                      lambda: _end_flash(widget, original, styled_before))


def _end_flash(widget: QWidget, original: str, styled_before: bool) -> None:
    try:
        widget.setStyleSheet(original)
        widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, styled_before)
        widget.setProperty("mcpFlash", False)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
    except RuntimeError:                                          # widget died mid-flash
        pass


# ---------------------------------------------------------------------------
# The dock
# ---------------------------------------------------------------------------

class ActivityDock(QDockWidget):
    """Recent MCP calls, live.

    `entry` is a Qt signal on purpose: `ActivityLog` records from the MCP server thread, and
    a signal emission is the one documented way to hand that to the main thread (it is
    queued when the emitter is a different thread). Everything below therefore runs in the
    GUI thread and may touch widgets freely.
    """

    entry = Signal(object)

    COLUMNS = ("time", "tool", "outcome", "ms", "detail")

    def __init__(self, log: ActivityLog, window: Any = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__("MCP activity", parent)
        self.setObjectName("AnimeTextMcpActivityDock")
        self.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)
        self._log = log
        self._window = window
        self.calls = 0
        self.errors = 0
        self.setWidget(self._build())
        self.entry.connect(self._on_entry)
        self._log.add_listener(self.entry.emit)
        self.reload()

    # -- construction --------------------------------------------------------

    def _build(self) -> QWidget:
        box = QWidget(self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)

        self.table = QTableWidget(0, len(self.COLUMNS), box)
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(self.COLUMNS) - 1,
                                    QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.summary = QLabel("no MCP calls yet", box)
        self.summary.setObjectName("caption")
        row.addWidget(self.summary, 1)
        self.clear_button = QPushButton("Clear", box)
        self.clear_button.clicked.connect(self._clear_pressed)
        row.addWidget(self.clear_button)
        layout.addLayout(row)
        return box

    # -- the live part -------------------------------------------------------

    def _on_entry(self, event: ActivityEvent) -> None:
        self.calls += 1
        if not event.ok:
            self.errors += 1
        color = QColor("#1b1b1b") if event.ok else QColor(_FLASH_ERROR)
        detail = event.message or (event.args and _args_line(event.args)) or ""
        # the log keeps a full ISO-8601 stamp; the dock has a narrow column for it
        cells = [event.at.split("T")[-1].split("+")[0], event.tool, event.status,
                 str(event.duration_ms), detail]
        self.table.insertRow(0)
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setForeground(color)
            self.table.setItem(0, column, item)
        while self.table.rowCount() > self._log.summary()["capacity"]:
            self.table.removeRow(self.table.rowCount() - 1)
        self.summary.setText(f"{self.calls} call(s), {self.errors} error(s) - "
                             f"last: {event.tool} {event.status}")
        if self._window is not None:
            try:
                self._window._say(f"MCP {event.tool}: "
                                  f"{event.message or event.status}")
            except Exception:                                     # noqa: BLE001
                pass
            for path in TOOL_TARGETS.get(event.tool, ()):
                flash_widget(widget_for(self._window, path),
                             color=_FLASH_COLOR if event.ok else _FLASH_ERROR)

    def reload(self) -> None:
        """Re-fill the table from the log (attach-after-calls, tests, --gui-smoke)."""
        self.table.setRowCount(0)
        for event in reversed(self._log.events()):
            self._on_entry(event)

    def _clear_pressed(self) -> None:
        self.table.setRowCount(0)
        self._log.clear()
        self.summary.setText("no MCP calls yet")

    def last_detail(self) -> str:
        item = self.table.item(0, len(self.COLUMNS) - 1)
        return item.text() if item is not None else ""

    def detach(self) -> None:
        """Stop listening (window teardown): no signals into a dying widget."""
        self._log.remove_listener(self.entry.emit)


def _args_line(args: dict[str, Any]) -> str:
    inner = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
    return inner[:120]


def attach_activity_dock(window: Any, service: Any) -> ActivityDock:
    """Add the dock to `window` and point it at `service`'s activity log."""
    dock = ActivityDock(service.activity, window, window)
    window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    dock.setObjectName("AnimeTextMcpActivityDock")
    return dock
