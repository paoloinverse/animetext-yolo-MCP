"""`JsonPane` — the result document, readable and copyable (PLAN Part 6).

Pretty-printed, read-only, `ensure_ascii=False` so Japanese/Korean/Chinese text is legible
in the pane and in whatever the user pastes it into. `Copy all` puts **exactly**
`PipelineResult.to_json()` on the clipboard — byte for byte, not a re-serialisation, so
what the user copies is what the file and the MCP tool would return.

Blocks are indexed by line range, which is what makes "show me block 7" possible: the
canvas, the block table and this pane all sync on a block id, and the pane can scroll to
and highlight the lines that block occupies. The index is derived from the text we
actually printed (the indentation is measured, not assumed), so it cannot silently drift
if the schema gains a field.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMenu, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

from ..core.types import PipelineResult

#: Highlighting for the pane: keys dimmed, strings green, numbers blue, null grey.
_KEY_COLOR = "#7f8fa6"
_STRING_COLOR = "#2e8b57"
_NUMBER_COLOR = "#1d5fbf"
_NULL_COLOR = "#8a8a8a"


class _JsonHighlighter(QSyntaxHighlighter):
    """Small, dependency-free JSON colouring (read-only pane, so regexes are enough)."""

    def __init__(self, document) -> None:
        super().__init__(document)
        self._rules: list[tuple[re.Pattern, QTextCharFormat]] = []

        def add(pattern: str, color: str, *, bold: bool = False) -> None:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            fmt.setFontWeight(75 if bold else 50)
            self._rules.append((re.compile(pattern), fmt))

        add(r'"(?:\\.|[^"\\])*"(?=\s*:)', _KEY_COLOR, bold=True)
        add(r'(?<=:)\s*"(?:\\.|[^"\\])*"', _STRING_COLOR)
        add(r'(?<!\w)-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?=[,\s\]}])', _NUMBER_COLOR)
        add(r'\b(?:true|false|null)\b', _NULL_COLOR)

    def highlightBlock(self, text: str) -> None:                      # noqa: N802 (Qt API)
        for pattern, fmt in self._rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class JsonPane(QWidget):
    """Read-only JSON viewer with Copy all / Save As… and per-block copy."""

    copied = Signal(str)        # what was copied, for the status bar
    saved = Signal(str)         # path written

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._result: Optional[PipelineResult] = None
        self._text = ""
        self._ranges: dict[int, tuple[int, int]] = {}     # block id -> (first, last) line
        self._focused_id: Optional[int] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        strip = QHBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(6)
        self.title_label = QLabel("JSON", self)
        self.copy_button = _button("Copy all", self)
        self.copy_button.clicked.connect(self.copy_all)
        self.copy_block_button = _button("Copy block", self)
        self.copy_block_button.setToolTip("Copy the text of the selected block "
                                          "(right-click in the pane for its JSON)")
        self.copy_block_button.clicked.connect(self.copy_focused_block_text)
        self.save_button = _button("Save As…", self)
        self.save_button.clicked.connect(self._save_button_pressed)
        self.count_label = QLabel("", self)
        self.count_label.setObjectName("jsonCount")
        strip.addWidget(self.title_label)
        strip.addWidget(self.copy_button)
        strip.addWidget(self.copy_block_button)
        strip.addWidget(self.save_button)
        strip.addWidget(self.count_label, 1)
        layout.addLayout(strip)

        self.editor = QTextEdit(self)
        self.editor.setReadOnly(True)
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.editor.setObjectName("jsonEditor")
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        font.setPointSize(10)
        self.editor.setFont(font)
        self.editor.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.editor.customContextMenuRequested.connect(self._show_context_menu)
        self._highlighter = _JsonHighlighter(self.editor.document())
        layout.addWidget(self.editor, 1)

    # -- content ------------------------------------------------------------

    def set_result(self, result: Optional[PipelineResult]) -> None:
        """Show the document. `None` clears the pane (and any stale block index)."""
        self._result = result
        if result is None:
            self.clear()
            return
        self._text = result.to_json(indent=2)
        self._ranges = _index_blocks(self._text)
        self._focused_id = None
        self.editor.setPlainText(self._text)
        self.count_label.setText(f"{result.block_count} block(s) · "
                                 f"{len(self._text)} chars")

    def clear(self) -> None:
        self._result = None
        self._text = ""
        self._ranges = {}
        self._focused_id = None
        self.editor.clear()
        self.count_label.setText("")

    @property
    def text(self) -> str:
        return self._text

    @property
    def result(self) -> Optional[PipelineResult]:
        return self._result

    @property
    def focused_block_id(self) -> Optional[int]:
        return self._focused_id

    def block_lines(self, block_id: int) -> Optional[tuple[int, int]]:
        """First and last line (0-based, inclusive) of that block in the document."""
        return self._ranges.get(int(block_id))

    # -- focus / selection --------------------------------------------------

    def focus_block(self, block_id: int) -> bool:
        """Scroll to and highlight the lines of `block_id`. False if it is not here."""
        lines = self.block_lines(block_id)
        if lines is None:
            return False
        self._focused_id = int(block_id)
        document = self.editor.document()
        first, last = lines
        cursor = QTextCursor(document.findBlockByNumber(first))
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        end = document.findBlockByNumber(last)
        cursor.setPosition(end.position())
        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine,
                            QTextCursor.MoveMode.KeepAnchor)
        self.editor.setTextCursor(cursor)
        self.editor.ensureCursorVisible()
        return True

    # -- clipboard / disk ---------------------------------------------------

    def copy_all(self) -> str:
        """Clipboard := the document, byte for byte (never re-serialised)."""
        clipboard = QApplication.clipboard()
        payload = self._text
        if clipboard is not None:
            clipboard.setText(payload)
        self.copied.emit(f"JSON copied ({len(payload)} chars)")
        return payload

    def copy_block_text(self, block_id: int) -> Optional[str]:
        block = self._block(block_id)
        if block is None:
            return None
        payload = block.text or ""
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(payload)
        self.copied.emit(f"Block #{block.id} text copied ({len(payload)} chars)")
        return payload

    def copy_block_json(self, block_id: int) -> Optional[str]:
        block = self._block(block_id)
        if block is None:
            return None
        payload = json.dumps(block.to_dict(), indent=2, ensure_ascii=False)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(payload)
        self.copied.emit(f"Block #{block.id} JSON copied")
        return payload

    def copy_focused_block_text(self) -> Optional[str]:
        if self._focused_id is None:
            self.copied.emit("Select a block first (click its box or its row).")
            return None
        return self.copy_block_text(self._focused_id)

    def _save_button_pressed(self) -> None:
        """Button slot: `clicked`'s `checked` bool must not arrive as `path`."""
        self.save_as()

    def save_as(self, path: Union[str, Path, None] = None) -> Optional[str]:
        """Write the document as UTF-8 JSON (asks for a path when none is given)."""
        if not self._text:
            return None
        if path is None:
            suggested = "animetext-result.json"
            if self._result is not None and self._result.image.file_name:
                suggested = Path(self._result.image.file_name).stem + ".json"
            path, _ = QFileDialog.getSaveFileName(self, "Save result JSON", suggested,
                                                  "JSON (*.json)")
            if not path:
                return None
        target = Path(str(path))
        try:
            if target.parent and not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self._text, encoding="utf-8")
        except OSError as exc:
            self.copied.emit(f"Cannot write {target}: {exc}")
            return None
        self.saved.emit(str(target))
        return str(target)

    # -- context menu -------------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        block_id = self._block_at_line(self.editor.cursorForPosition(pos).blockNumber())
        if block_id is None:
            block_id = self._focused_id
        menu = QMenu(self)
        add = menu.addAction
        add("Copy all", self.copy_all)
        copy_text = add("Copy block text", lambda: self._copy_from_menu(block_id, "text"))
        copy_json = add("Copy block JSON", lambda: self._copy_from_menu(block_id, "json"))
        for action in (copy_text, copy_json):
            action.setEnabled(block_id is not None)
        menu.addSeparator()
        add("Save As…", self.save_as)
        menu.exec(self.editor.mapToGlobal(pos))

    def _copy_from_menu(self, block_id: Optional[int], kind: str) -> None:
        if block_id is None:
            return
        (self.copy_block_json if kind == "json" else self.copy_block_text)(block_id)

    def _block(self, block_id: int):
        if self._result is None:
            return None
        return next((b for b in self._result.blocks if int(b.id) == int(block_id)), None)

    def _block_at_line(self, line: int) -> Optional[int]:
        for block_id, (first, last) in self._ranges.items():
            if first <= line <= last:
                return block_id
        return None


def _button(text: str, parent: QWidget) -> QPushButton:
    """A flat button: this strip is a tool row, not a form."""
    button = QPushButton(text, parent)
    button.setFlat(True)
    return button


def _index_blocks(text: str) -> dict[int, tuple[int, int]]:
    """Map block id -> (first line, last line) in the printed document.

    The indentation is measured from the `"blocks": [` line rather than assumed, and a
    block's range runs from its opening brace to its closing one.
    """
    lines = text.split("\n")
    item_indent = None
    key_indent = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"blocks"'):
            key_indent = len(line) - len(line.lstrip())
            item_indent = key_indent + 2
            break
    if item_indent is None:
        return {}
    open_re = re.compile(r"^ {%d}\{$" % item_indent)
    close_re = re.compile(r"^ {%d}\},?$" % item_indent)
    field = key_indent + 4
    id_re = re.compile(r'^ {%d}"id":\s*(-?\d+),?$' % field)

    ranges: dict[int, tuple[int, int]] = {}
    started: Optional[int] = None
    for number, line in enumerate(lines):
        if started is None:
            if open_re.match(line):
                started = number
            continue
        if close_re.match(line):
            block_id = _id_between(lines, started, number, id_re)
            if block_id is not None:
                ranges[block_id] = (started, number)
            started = None
    return ranges


def _id_between(lines: list[str], start: int, end: int, id_re: re.Pattern) -> Optional[int]:
    for line in lines[start:end + 1]:
        match = id_re.match(line)
        if match:
            return int(match.group(1))
    return None
