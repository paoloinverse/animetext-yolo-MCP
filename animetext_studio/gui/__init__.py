"""GUI package (PySide6).

Nothing in `animetext_studio.core` or `animetext_studio.engines` imports Qt, and this
package's `__init__` does not either — importing `animetext_studio.gui` on a machine
without the `[gui]` extra must not explode, so each module imports Qt itself and
`qt_available()` tells callers (tests, `--headless` fallbacks) whether it is there.
"""
from __future__ import annotations

__all__ = ["qt_available", "QT_IMPORT_ERROR"]

QT_IMPORT_ERROR: str = ""


def qt_available() -> bool:
    """True when PySide6 (and its shiboken runtime) can actually be imported."""
    global QT_IMPORT_ERROR
    try:
        import PySide6.QtCore          # noqa: F401  (shiboken6 is loaded by PySide6 itself)
        return True
    except ImportError as exc:          # pragma: no cover - depends on the machine
        QT_IMPORT_ERROR = str(exc)
        return False
