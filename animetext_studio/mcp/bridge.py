"""P7 - the bridge: where the MCP server thread and the Qt main thread meet.

Three things live here and nowhere else:

* **`Dispatcher`** - run a callable on *the thread that owns the app state* and hand back a
  `concurrent.futures.Future`. With a GUI that is the Qt main thread, reached by posting a
  custom `QEvent` to a receiver object that lives there (`postEvent` is the documented
  thread-safe way in from a foreign thread); headless it is the calling thread, inline.
  Everything the MCP tools do - read the config, touch the canvas, start a job - goes
  through `submit()`, so there is exactly one place where threads change.
* **`ActivityLog`** - one entry per tool call (time, tool, arguments, outcome, duration).
  It is the single source for the log dock, the status bar, the `log://mcp` resource and the
  demo transcript: the "visual confirmation" requirement is implemented once, here, and not
  per tool.
* **`redact_args`** - what a call looks like when it is logged: shapes and lengths, never
  pixels, base64 payloads or tokens.

Nothing here imports the `mcp` SDK or the pipeline: only `core.types` and, inside
`QtDispatcher`, PySide6. That is what makes the bridge testable without a display -
`InlineDispatcher` is the same contract without Qt.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent import futures as _futures
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Optional

from ..core.types import ErrorCode, StudioError

__all__ = [
    "ActivityEvent", "ActivityLog", "Dispatcher", "InlineDispatcher", "QtDispatcher",
    "redact_args", "redact_value", "SECRET_KEYS",
]

#: Argument keys whose values must never reach a log, a widget or the `log://mcp` resource.
SECRET_KEYS = frozenset({
    "auth_token", "token", "hf_token", "password", "secret",
    "base64_png_jpg", "base64", "data", "image_base64", "png",
})
#: Longest string value kept verbatim in an activity entry.
MAX_LOGGED_STRING = 120


# ---------------------------------------------------------------------------
# Argument redaction
# ---------------------------------------------------------------------------

def redact_value(key: Any, value: Any, *, depth: int = 0) -> Any:
    """A loggable shape for one argument value (see `redact_args`)."""
    name = str(key).lower()
    if name in SECRET_KEYS:
        if isinstance(value, (bytes, bytearray)):
            return f"<{len(value)} bytes>"
        if isinstance(value, str):
            return f"<{len(value)} chars>"
        return "<redacted>"
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        if len(value) > MAX_LOGGED_STRING:
            return value[:MAX_LOGGED_STRING] + f"... (+{len(value) - MAX_LOGGED_STRING} chars)"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        if depth >= 2:
            return f"<{len(value)} keys>"
        return {str(k): redact_value(k, v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if depth >= 2:
            return f"<{len(value)} items>"
        return [redact_value(f"[{i}]", item, depth=depth + 1)
                for i, item in enumerate(list(value)[:12])]
    return f"<{type(value).__name__}>"


def redact_args(args: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Summarise tool arguments for the activity log (keys kept, secrets/size only)."""
    if not args:
        return {}
    return {str(key): redact_value(key, value) for key, value in args.items()}


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivityEvent:
    """One MCP tool call, as the log dock and `log://mcp` see it."""

    index: int
    at: str                       # local ISO-8601, second resolution
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    duration_ms: int = 0
    code: str = ""                # ErrorCode value when ok is False
    message: str = ""             # one-line outcome (result summary or error message)
    job_id: str = ""
    transport: str = ""
    session: str = ""

    @property
    def status(self) -> str:
        return "ok" if self.ok else (self.code or "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "at": self.at,
            "tool": self.tool,
            "args": dict(self.args),
            "ok": self.ok,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "code": self.code,
            "message": self.message,
            "job_id": self.job_id,
            "transport": self.transport,
            "session": self.session,
        }

    def line(self) -> str:
        """The one-line form the status bar shows and the log table renders."""
        args = ""
        if self.args:
            inner = ", ".join(f"{k}={v}" for k, v in list(self.args.items())[:3])
            args = f"({inner})"
        tail = f" - {self.message}" if self.message else ""
        return f"{self.at} {self.tool}{args} {self.status} {self.duration_ms}ms{tail}"


class ActivityLog:
    """Bounded activity buffer with listeners (the GUI dock, the demo client, tests).

    Listeners are called from whatever thread recorded the event - the MCP server thread for
    tool calls - so a Qt listener must hop threads itself (a `Signal.emit` does exactly that).
    A listener that raises is ignored: the log is a display, never a failure mode.
    """

    def __init__(self, capacity: int = 200) -> None:
        self._capacity = max(1, int(capacity))
        self._events: Deque[ActivityEvent] = deque(maxlen=self._capacity)
        self._listeners: list[Callable[[ActivityEvent], None]] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.calls = 0
        self.errors = 0

    def set_capacity(self, capacity: int) -> int:
        """Resize the ring buffer (P8: `mcp.log_capacity`) and return the new capacity.

        Called when the config changes while the app runs, so the setting is live without a
        restart. Shrinking drops the *oldest* entries — `deque(iterable, maxlen=)` keeps the
        tail, which is the recent history the dock and `log://mcp` are for.
        """
        want = max(1, int(capacity))
        with self._lock:
            if want != self._capacity:
                self._events = deque(self._events, maxlen=want)
                self._capacity = want
            return self._capacity

    # -- writing ------------------------------------------------------------

    def record(self, tool: str, *, ok: bool = True, args: Optional[dict[str, Any]] = None,
               duration_ms: int = 0, code: str = "", message: str = "",
               job_id: str = "", transport: str = "", session: str = "",
               at: Optional[str] = None) -> ActivityEvent:
        with self._lock:
            self._seq += 1
            event = ActivityEvent(
                index=self._seq,
                at=at or datetime.now().astimezone().isoformat(timespec="seconds"),
                tool=tool,
                args=redact_args(args),
                ok=bool(ok),
                duration_ms=int(duration_ms),
                code=code or "",
                message=message or "",
                job_id=job_id or "",
                transport=transport or "",
                session=session or "",
            )
            self._events.append(event)
            self.calls += 1
            if not event.ok:
                self.errors += 1
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:                                       # noqa: BLE001
                pass    # a broken display must not break a tool call
        return event

    # -- reading ------------------------------------------------------------

    def events(self, limit: Optional[int] = None) -> list[ActivityEvent]:
        """Most recent first (a harness wants the tail)."""
        with self._lock:
            items = list(self._events)
        items.reverse()
        return items[:limit] if limit else items

    def entries(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events(limit)]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {"calls": self.calls, "errors": self.errors,
                    "capacity": self._capacity, "kept": len(self._events)}

    def last(self) -> Optional[ActivityEvent]:
        with self._lock:
            return self._events[-1] if self._events else None

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    # -- listeners ----------------------------------------------------------

    def add_listener(self, listener: Callable[[ActivityEvent], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[ActivityEvent], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class Dispatcher:
    """Runs callables on the app-state thread; `submit` hands back a `Future`."""

    mode = "inline"

    def submit(self, fn: Callable[[], Any], *, timeout: Optional[float] = None) -> Future:
        raise NotImplementedError

    def call(self, fn: Callable[[], Any], *, timeout: Optional[float] = None) -> Any:
        """`submit()` + wait. A timeout is a structured error, never a hung client."""
        future = self.submit(fn, timeout=timeout)
        try:
            return future.result(timeout)
        except _futures.TimeoutError as exc:      # an alias of the builtin on 3.11+
            raise self._timeout_error(timeout) from exc

    @staticmethod
    def _timeout_error(timeout: Optional[float]) -> StudioError:
        return StudioError(
            ErrorCode.INTERNAL,
            f"The application did not answer within {timeout} s.",
            {"hint": "the Qt main thread is busy or its event loop is not running",
             "timeout_s": timeout},
        )

    def describe(self) -> str:
        return self.mode


class InlineDispatcher(Dispatcher):
    """Headless: the caller's thread *is* the app thread, so run now and return a done future.

    This is what `--headless` (and every unit test without a window) uses, and the fallback
    when Qt is not installed - which keeps the tool layer free of `if gui:` branches.
    """

    mode = "inline"

    def __init__(self) -> None:
        self.calls = 0

    def submit(self, fn: Callable[[], Any], *, timeout: Optional[float] = None) -> Future:
        future: Future = Future()
        self.calls += 1
        try:
            future.set_result(fn())
        except BaseException as exc:                                 # noqa: BLE001
            future.set_exception(exc)
        return future


class _Call:
    """One queued invocation, carried to the Qt thread inside a `QEvent`."""

    __slots__ = ("future", "fn", "submitted_at")

    def __init__(self, future: Future, fn: Callable[[], Any]) -> None:
        self.future = future
        self.fn = fn
        self.submitted_at = time.perf_counter()

    def run(self) -> None:
        if not self.future.set_running_or_notify_cancel():
            return
        try:
            self.future.set_result(self.fn())
        except BaseException as exc:                                 # noqa: BLE001
            self.future.set_exception(exc)


class QtDispatcher(Dispatcher):
    """Queue invocations onto the Qt main thread (the only thread allowed to touch widgets).

    Construct it **on the main thread**: the receiver object adopts the thread that creates
    it. From any other thread `submit()` posts an event; `postEvent` is thread-safe and the
    receiver's `event()` handler runs the callable in the main thread and resolves the
    future. Called *from* the main thread it runs inline - a GUI slot calling into the
    service must not deadlock on its own queue.
    """

    mode = "qt-main-thread"

    def __init__(self, app: Any = None) -> None:
        from PySide6.QtCore import QCoreApplication, QEvent, QObject

        self._app = app or QCoreApplication.instance()
        if self._app is None:
            raise StudioError(
                ErrorCode.INTERNAL,
                "QtDispatcher needs a QCoreApplication to post events to.",
                {"hint": "build the service after the QApplication exists, "
                         "or use InlineDispatcher for headless runs"},
            )
        self._main_thread = self._app.thread()
        self._event_type = QEvent.Type(QEvent.registerEventType())
        self._pending: list[_Call] = []
        self._lock = threading.Lock()
        self.closed = False
        self.queued = 0
        self.ran = 0

        outer = self

        class _CallEvent(QEvent):
            def __init__(self, call: _Call) -> None:
                super().__init__(outer._event_type)
                self.call = call

        self._call_event = _CallEvent

        class _Receiver(QObject):
            """Lives in the main thread; runs whatever `_CallEvent` is posted to it."""

            def event(self, q_event: Any) -> bool:                   # noqa: D102
                call = getattr(q_event, "call", None)
                if not isinstance(call, _Call):
                    return False
                with outer._lock:
                    if call in outer._pending:
                        outer._pending.remove(call)
                if outer.closed:
                    if not call.future.done():
                        call.future.set_exception(StudioError(
                            ErrorCode.INTERNAL,
                            "The window closed before this call could run."))
                    return True
                outer.ran += 1
                call.run()
                return True

        self._receiver = _Receiver()
        self._receiver.setObjectName("animetextMcpBridge")

    # -- identity -----------------------------------------------------------

    def is_main_thread(self) -> bool:
        from PySide6.QtCore import QThread

        return QThread.currentThread() is self._main_thread

    def close(self) -> None:
        """Refuse further work (window closing) and fail whatever is still queued."""
        self.closed = True
        with self._lock:
            pending, self._pending = list(self._pending), []
        for call in pending:
            if not call.future.done():
                call.future.set_exception(StudioError(
                    ErrorCode.INTERNAL, "The window closed before this call could run."))

    # -- dispatch -----------------------------------------------------------

    def submit(self, fn: Callable[[], Any], *, timeout: Optional[float] = None) -> Future:
        if self.closed:
            raise StudioError(ErrorCode.INTERNAL,
                              "The GUI bridge is closed (window shut down).")
        future: Future = Future()
        call = _Call(future, fn)
        if self.is_main_thread():
            self.ran += 1
            call.run()
            return future
        with self._lock:
            self._pending.append(call)
        self.queued += 1
        self._app.postEvent(self._receiver, self._call_event(call))
        return future

    def pump(self, timeout: float = 2.0, *, future: Optional[Future] = None) -> bool:
        """Drain the Qt event queue from the main thread until `future` is done.

        Only for tests and `--gui-smoke`-style paths where no `app.exec()` is running;
        production code never needs it, because the event loop is live.
        """
        deadline = time.monotonic() + timeout
        while True:
            self._app.processEvents()
            if future is None or future.done() or time.monotonic() >= deadline:
                return future is None or future.done()
            time.sleep(0.002)

    def describe(self) -> str:
        return f"qt-main-thread (queued={self.queued}, ran={self.ran})"
