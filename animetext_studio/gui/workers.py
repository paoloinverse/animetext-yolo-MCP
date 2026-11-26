"""Qt bridge over `JobManager` — signals in, signals out, no second thread.

`JobManager` already owns exactly one worker thread (D14), so the GUI must not start
another one: two threads running an ONNX session and a torch model is how you get a crash
that only happens when the user clicks twice. What the GUI needs is *marshalling*: the
manager calls its listener from the worker thread, and Qt widgets may only be touched from
the GUI thread. `JobBridge` is that marshalling layer — a `QObject` whose signals are
emitted from wherever the manager happens to call and delivered to widget slots through a
queued connection.

Contract the window can rely on:

* `progress(JobProgress)` — every state change, including the initial `queue` (D44).
* `finished(PipelineResult)` / `failed(StudioError)` / `cancelled()` — exactly one of these
  per job, after the last `progress`.
* Busy state is **derived** from those signals, never from statements around the call: the
  first event of a job can arrive before `start_file()` returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from PySide6.QtCore import QObject, Signal

from ..core.jobs import JobManager
from ..core.types import JobState


class JobBridge(QObject):
    """Queued-signal façade over a `JobManager` for use from Qt slots."""

    progress = Signal(object)        # JobProgress
    finished = Signal(object)        # PipelineResult
    failed = Signal(object)          # StudioError
    cancelled = Signal()
    #: Engine warm-up finished (P8.3). Separate from `finished` on purpose: a warm-up has no
    #: result document, and the window must not re-render (or re-announce) the previous page.
    warmup_done = Signal(object)     # WarmupReport

    def __init__(self, manager: JobManager, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._manager.add_listener(self._on_state)

    # -- manager callbacks (may be called from the worker thread) -----------

    def _on_state(self, state: Any) -> None:
        self.progress.emit(state)
        if state.state is JobState.DONE:
            if getattr(self._manager, "kind", "run") == "warmup":
                # `manager.result` is still the *previous* page here (a warm-up does not touch
                # it), so emitting `finished` would repaint and re-announce a run that did not
                # happen. Route to the warm-up slot instead.
                self.warmup_done.emit(getattr(self._manager, "warmup_report", None))
                return
            result = self._manager.result
            if result is not None:
                self.finished.emit(result)
        elif state.state is JobState.CANCELLED:
            self.cancelled.emit()
        elif state.state is JobState.ERROR:
            error = self._manager.error
            if error is not None:
                self.failed.emit(error)

    # -- actions ------------------------------------------------------------

    def start_file(self, path: Union[str, Path]) -> bool:
        """Start a job on a file. False (instead of raising) when the manager is busy."""
        from ..core.types import StudioError

        try:
            self._manager.start_file(str(path))
            return True
        except StudioError as exc:
            self.failed.emit(exc)          # BUSY is a UI state: the window disables Run
            return False

    def start_bytes(self, data: bytes, *, file_name: Optional[str] = None,
                    source: str = "clipboard") -> bool:
        from ..core.types import StudioError

        try:
            self._manager.start_bytes(data, file_name=file_name, source=source)
            return True
        except StudioError as exc:
            self.failed.emit(exc)
            return False

    def start_reocr_file(self, path: Union[str, Path], result: Any,
                         block_ids: list[int]) -> bool:
        """Re-read blocks of `result` from the same file (P9.3). False when busy or no result."""
        from ..core.types import StudioError

        try:
            self._manager.start_reocr_file(str(path), result, block_ids)
            return True
        except StudioError as exc:
            self.failed.emit(exc)
            return False

    def start_reocr_bytes(self, data: bytes, result: Any, block_ids: list[int], *,
                          file_name: Optional[str] = None,
                          source: str = "clipboard") -> bool:
        from ..core.types import StudioError

        try:
            self._manager.start_reocr_bytes(data, result, block_ids,
                                            file_name=file_name, source=source)
            return True
        except StudioError as exc:
            self.failed.emit(exc)
            return False

    def start_warmup(self, *, probe: bool = True) -> bool:
        """Warm the engines (P8.3). False (instead of raising) when a job is already running."""
        from ..core.types import StudioError

        try:
            self._manager.start_warmup(probe=probe)
            return True
        except StudioError as exc:
            self.failed.emit(exc)
            return False

    def cancel(self) -> bool:
        return self._manager.cancel()

    # -- introspection ------------------------------------------------------

    @property
    def manager(self) -> JobManager:
        return self._manager

    @property
    def busy(self) -> bool:
        return self._manager.is_busy

    def shutdown(self) -> None:
        """Detach the listener (called on window close) so no signal outlives the widgets."""
        self._manager.remove_listener(self._on_state)
