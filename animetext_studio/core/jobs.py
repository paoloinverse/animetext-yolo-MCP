r"""Part 4 — `JobManager`: the single-flight job runner behind the GUI and the MCP server.

One page at a time (D14): an ONNX inference session and a torch model are neither cheap
nor thread-safe, so the app runs exactly one pipeline job on exactly one worker thread and
every client — GUI button, `--headless --run`, MCP tool call — goes through this same
object. That is the point: what the GUI shows and what the MCP reports are the same state
machine, not two implementations of it.

State machine (PLAN §3 Part 4):

    idle -> running -> done
                   \-> error
                   \-> cancelling -> cancelled   (or -> done, if it finished anyway)

Notes on the deliberate choices:

* **Busy is an error, not a queue** (`StudioError(BUSY)`, exit 4). Queuing pages silently
  reorders a user's intent and hides the fact that the previous job is still holding the
  GPU; the caller decides what to do (GUI: disable the button; MCP: tell the client).
* **`cancel()` is a request.** It flips the event and moves the state to `cancelling`; the
  worker — which is the only thing that can stop — moves it to `cancelled`. If the job
  happens to finish first, the state becomes `done`: a finished page is not a cancelled one.
* **The latest successful result is cached** (`result`): that *is* the app's "current
  result" for the GUI pane and the MCP `result` resource.
* **Listener callbacks run outside the lock**, so a listener may call back into the
  manager (`status()`, `cancel()`) without deadlocking — GUI slots always do.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional, Union

from .pipeline import Pipeline
from .types import ErrorCode, JobProgress, JobState, PipelineResult, StudioError

__all__ = ["JobManager"]

ProgressListener = Callable[[JobProgress], None]

_BUSY_STATES = (JobState.RUNNING, JobState.CANCELLING)


class JobManager:
    """Runs one `Pipeline` at a time on a worker thread; thread-safe to interrogate."""

    def __init__(self, pipeline: Pipeline, *, on_progress: Optional[ProgressListener] = None,
                 thread_name: str = "animetext-job") -> None:
        self.pipeline = pipeline
        self._lock = threading.RLock()
        self._listeners: list[ProgressListener] = [on_progress] if on_progress else []
        self._thread_name = thread_name
        self._seq = 0
        self._job_id = ""
        #: "run" (produces a result document), "warmup" (P8.3: engines only, no result) or
        #: "reocr" (P9.3: an existing document's boxes, some blocks read again).
        #: Listeners need this because a finished warm-up must not be mistaken for a page.
        self._kind = "run"
        self._state = JobState.IDLE
        self._progress = JobProgress(job_id="", state=JobState.IDLE, stage="idle",
                                     percent=0.0, message="no job yet")
        self._cancel: Optional[threading.Event] = None
        self._worker: Optional[threading.Thread] = None
        self._result: Optional[PipelineResult] = None
        self._error: Optional[StudioError] = None
        #: The last `WarmupReport` (P8.3). Kept apart from `_result` on purpose: a warm-up is
        #: not a page, and the previous page must survive one.
        self._warmup: Optional[Any] = None

    # -- engine passthroughs (the manager owns the pair; GUI + MCP share it) --

    @property
    def detection(self) -> Any:
        return self.pipeline.detection

    @property
    def ocr(self) -> Any:
        return self.pipeline.ocr

    def unload_engines(self) -> None:
        """Free both engines (GUI shutdown / MCP `unload`). Waits for the running job."""
        self.wait()
        for engine in (self.ocr, self.detection):
            try:
                engine.unload()
            except Exception:                                  # noqa: BLE001
                pass    # freeing memory must never be an error

    # -- listeners ----------------------------------------------------------

    def add_listener(self, listener: ProgressListener) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: ProgressListener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # -- state --------------------------------------------------------------

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    @property
    def job_id(self) -> str:
        with self._lock:
            return self._job_id

    @property
    def progress(self) -> JobProgress:
        with self._lock:
            return self._progress

    @property
    def warmup_report(self) -> Any:
        """The last `WarmupReport`, or None if no warm-up has finished (P8.3)."""
        with self._lock:
            return self._warmup

    @property
    def kind(self) -> str:
        """"run" or "warmup" for the current (or last) job — see `start_warmup`."""
        with self._lock:
            return self._kind

    def start_warmup(self, *, probe: bool = True) -> JobProgress:
        """Warm both engines on the worker thread (`--preload`, P8.3).

        A job, not a side effect: it takes the single-flight lock (so a warm-up and a page can
        never load the same model twice at once), reports on the normal progress channel with
        stage `warmup`, and is cancellable. It stores **no result** — the previous page survives
        it, `has_result` does not change — and a warm-up that failed ends the job ERROR with the
        engine's own code rather than pretending to be ready.

        `probe=False` loads the weights without the throwaway inference.
        """
        from .warmup import warm_engines

        def work(progress: Any, cancel: threading.Event) -> Any:
            report = warm_engines(self.detection, self.ocr, self.pipeline.config,
                                  progress=progress, cancel_event=cancel, probe=probe)
            return report.raise_for_failures()

        return self._start(work, kind="warmup")

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._state in _BUSY_STATES

    @property
    def result(self) -> Optional[PipelineResult]:
        """The latest successful result (the app's "current result")."""
        with self._lock:
            return self._result

    @property
    def error(self) -> Optional[StudioError]:
        with self._lock:
            return self._error

    def clear_result(self) -> None:
        """Forget the cached result (MCP `reset`). Refuses while a job is running.

        The GUI clears its own view; this clears the *state* the MCP resources read, so
        `result://latest` cannot serve a page the user has already thrown away.
        """
        with self._lock:
            if self._state in _BUSY_STATES:
                raise StudioError(
                    ErrorCode.BUSY, "Cannot clear the result while a job is running.",
                    {"job_id": self._job_id, "state": self._state.value,
                     "hint": "cancel_job first"},
                )
            self._result = None
            self._error = None
            self._job_id = ""
            self._progress = JobProgress(job_id="", state=JobState.IDLE, stage="idle",
                                         percent=0.0, message="no job yet")

    def status(self) -> dict[str, Any]:
        """JSON-ready snapshot for the MCP `job` resource and the GUI status bar."""
        with self._lock:
            return {
                "job_id": self._job_id,
                "kind": self._kind,
                "state": self._state.value,
                "stage": self._progress.stage,
                "percent": self._progress.percent,
                "message": self._progress.message,
                "busy": self._state in _BUSY_STATES,
                "has_result": self._result is not None,
                "block_count": self._result.block_count if self._result else 0,
                "error": self._error.to_dict() if self._error else None,
            }

    # -- start / cancel / wait ---------------------------------------------

    def start(self, image: Any, **kwargs: Any) -> JobProgress:
        """Start a job on a PIL image. Raises `StudioError(BUSY)` if one is already running."""
        return self._start(lambda progress, cancel: self.pipeline.run(
            image, cancel_event=cancel, progress=progress, **kwargs))

    def start_file(self, path: Union[str, Any], **kwargs: Any) -> JobProgress:
        return self._start(lambda progress, cancel: self.pipeline.run_file(
            path, cancel_event=cancel, progress=progress, **kwargs))

    def start_bytes(self, data: bytes, **kwargs: Any) -> JobProgress:
        return self._start(lambda progress, cancel: self.pipeline.run_bytes(
            data, cancel_event=cancel, progress=progress, **kwargs))

    # -- re-read selected blocks (P9.3) ------------------------------------

    def start_reocr(self, image: Any, result: Any, block_ids: Any, **kwargs: Any) -> JobProgress:
        """Re-read the named blocks of `result` — a job like any other (P9.3).

        Single-flight against the same lock a page takes, so a re-read can never overlap a run on
        one ONNX session (D14); progress and cancellation work as usual. Its document replaces
        the previous one as `latest`: a re-read is not a side effect on the old document, it is a
        newer answer about the same page (D81).
        """
        return self._start(lambda progress, cancel: self.pipeline.reocr_blocks(
            image, result, block_ids, cancel_event=cancel, progress=progress, **kwargs),
            kind="reocr")

    def start_reocr_file(self, path: Union[str, Any], result: Any, block_ids: Any,
                         **kwargs: Any) -> JobProgress:
        return self._start(lambda progress, cancel: self.pipeline.reocr_blocks_file(
            path, result, block_ids, cancel_event=cancel, progress=progress, **kwargs),
            kind="reocr")

    def start_reocr_bytes(self, data: bytes, result: Any, block_ids: Any,
                          **kwargs: Any) -> JobProgress:
        return self._start(lambda progress, cancel: self.pipeline.reocr_blocks_bytes(
            data, result, block_ids, cancel_event=cancel, progress=progress, **kwargs),
            kind="reocr")

    def _start(self, work: Callable[[Any, threading.Event], Any], *,
               kind: str = "run") -> JobProgress:
        with self._lock:
            if self._state in _BUSY_STATES:
                raise StudioError(
                    ErrorCode.BUSY,
                    f"A job is already running ({self._state.value}).",
                    {"job_id": self._job_id, "state": self._state.value,
                     "hint": "cancel it, or wait for it to finish"},
                )
            self._seq += 1
            self._job_id = f"job-{self._seq}"
            self._kind = kind
            self._cancel = threading.Event()
            self._error = None
            self._state = JobState.RUNNING
            self._progress = JobProgress(job_id=self._job_id, state=JobState.RUNNING,
                                         stage="queue", percent=0.0, message="queued")
            cancel = self._cancel
            job_id = self._job_id
            self._worker = threading.Thread(target=self._run,
                                            args=(job_id, work, cancel, kind),
                                            name=f"{self._thread_name}-{job_id}", daemon=True)
            started, listeners = self._progress, list(self._listeners)
        # Announce *before* the worker runs (and outside the lock, because a GUI slot may
        # call straight back into us): "queue" is then always the first event a listener
        # sees, and a cancel issued from inside that callback still lands before inference.
        self._notify(listeners, started)
        self._worker.start()
        return started

    def cancel(self) -> bool:
        """Request cancellation. False when there is nothing to cancel."""
        with self._lock:
            if self._state not in _BUSY_STATES or self._cancel is None:
                return False
            self._cancel.set()
            moved = self._state is JobState.RUNNING     # a second cancel changes nothing
            if moved:
                self._state = JobState.CANCELLING
                self._progress = JobProgress(job_id=self._job_id, state=JobState.CANCELLING,
                                             stage=self._progress.stage,
                                             percent=self._progress.percent,
                                             message="cancelling…")
            snapshot, listeners = self._progress, list(self._listeners)
        if moved:
            self._notify(listeners, snapshot)
        return True

    def wait(self, timeout: Optional[float] = None) -> JobProgress:
        """Block until the current job settles (or `timeout` elapses). Never holds the lock."""
        with self._lock:
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout)
        return self.progress

    # -- headless convenience: start + wait + unwrap -----------------------

    def run_file(self, path: Union[str, Any], *, timeout: Optional[float] = None,
                 **kwargs: Any) -> PipelineResult:
        return self._await(self.start_file(path, **kwargs), timeout)

    def run_bytes(self, data: bytes, *, timeout: Optional[float] = None,
                  **kwargs: Any) -> PipelineResult:
        return self._await(self.start_bytes(data, **kwargs), timeout)

    def _await(self, started: JobProgress, timeout: Optional[float]) -> PipelineResult:
        settled = self.wait(timeout)
        if settled.state is JobState.ERROR:
            error = self.error
            raise error or StudioError(ErrorCode.INTERNAL, "Job failed without an error.")
        if settled.state is JobState.CANCELLED:
            raise StudioError(ErrorCode.CANCELLED, "Job cancelled.",
                              {"job_id": settled.job_id})
        result = self.result
        if result is None:
            raise StudioError(ErrorCode.INTERNAL, "Job finished without a result.",
                              {"state": settled.state.value})
        return result

    # -- worker -------------------------------------------------------------

    def _run(self, job_id: str, work: Callable[[Any, threading.Event], Any],
             cancel: threading.Event, kind: str = "run") -> None:
        def on_progress(stage: str, percent: float, message: str) -> None:
            self._update(job_id, stage, percent, message)

        try:
            result = work(on_progress, cancel)
        except StudioError as exc:
            self._finish(job_id, JobState.CANCELLED if exc.code is ErrorCode.CANCELLED
                         else JobState.ERROR, error=exc,
                         message=exc.message if exc.code is not ErrorCode.CANCELLED
                         else "cancelled")
        except Exception as exc:                              # noqa: BLE001
            wrapped = StudioError(ErrorCode.INTERNAL, f"Unexpected error: {exc}",
                                  {"reason": f"{type(exc).__name__}: {exc}"})
            self._finish(job_id, JobState.ERROR, error=wrapped, message=wrapped.message)
        else:
            if kind == "warmup":
                # No result to store: a warm-up is not a page. The report's own summary becomes
                # the status message, so "warm" is something the user reads, not infers.
                self._warmup = result
                summary = getattr(result, "summary", None)
                self._finish(job_id, JobState.DONE,
                             message=summary() if callable(summary) else "engines warm")
                return
            self._finish(job_id, JobState.DONE, result=result,
                         message=f"{result.block_count} block(s)")

    def _update(self, job_id: str, stage: str, percent: float, message: str) -> None:
        with self._lock:
            if job_id != self._job_id or self._state not in _BUSY_STATES:
                return                                        # stale or already settled
            self._progress = JobProgress(job_id=job_id, state=self._state, stage=stage,
                                         percent=percent, message=message)
            snapshot, listeners = self._progress, list(self._listeners)
        self._notify(listeners, snapshot)

    def _finish(self, job_id: str, state: JobState, *,
                result: Optional[PipelineResult] = None,
                error: Optional[StudioError] = None, message: str = "") -> None:
        with self._lock:
            if job_id != self._job_id:
                return
            self._state = state
            if result is not None:
                self._result = result
            self._error = error
            self._progress = JobProgress(job_id=job_id, state=state,
                                         stage=state.value, percent=100.0, message=message)
            snapshot, listeners = self._progress, list(self._listeners)
        self._notify(listeners, snapshot)

    def _notify(self, listeners: list[ProgressListener], snapshot: JobProgress) -> None:
        """Called without the lock held, with the state captured under it: listeners may
        re-enter the manager, and they must never see a newer state than the event was for
        (nor miss the event because the worker already moved on)."""
        for listener in listeners:
            try:
                listener(snapshot)
            except Exception:                                  # noqa: BLE001
                pass    # a broken GUI/MCP listener must not kill the job
