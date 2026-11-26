"""Part 7 — the bridge: activity log, arg redaction, and the two dispatchers.

The Qt dispatcher is tested the way it will actually be used: a *worker* thread submits a
callable and the main thread pumps events, which is the only way a queued invocation can
run. Everything here runs offscreen and needs no server.
"""
from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("mcp", reason="the MCP SDK is not installed (pip install 'animetext-studio[mcp]')")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from animetext_studio.core.types import ErrorCode, StudioError  # noqa: E402
from animetext_studio.mcp.bridge import (                        # noqa: E402
    ActivityEvent, ActivityLog, InlineDispatcher, QtDispatcher, redact_args,
)


# ---------------------------------------------------------------------------
# Activity log
# ---------------------------------------------------------------------------

def test_activity_log_numbers_and_keeps_order():
    log = ActivityLog(capacity=3)
    first = log.record("get_status")
    second = log.record("reset")
    assert (first.index, second.index) == (1, 2)
    assert [e.tool for e in log.events()] == ["reset", "get_status"]   # newest first
    assert log.summary() == {"calls": 2, "errors": 0, "capacity": 3, "kept": 2}


def test_activity_log_is_a_ring_buffer():
    log = ActivityLog(capacity=3)
    for i in range(5):
        log.record(f"tool_{i}")
    assert [e.tool for e in log.events()] == ["tool_4", "tool_3", "tool_2"]
    assert log.summary()["calls"] == 5 and log.summary()["kept"] == 3


def test_activity_log_listener_sees_every_entry_and_swallows_its_own_errors():
    log = ActivityLog()
    seen, broken = [], []
    log.add_listener(seen.append)
    log.add_listener(lambda event: 1 / 0)          # a broken listener must not break the rest
    log.record("get_config")
    assert [e.tool for e in seen] == ["get_config"]
    log.remove_listener(seen.append)               # same function -> removed
    log.record("reset")
    assert len(seen) == 1


def test_activity_event_line_and_dict():
    event = ActivityEvent(index=4, at="12:00:00", tool="set_zoom",
                          args={"mode": "fit"}, ok=False, code="INVALID_ARGUMENT",
                          message="bad mode", job_id="job-1", transport="streamable-http")
    line = event.line()
    assert line.startswith("12:00:00 set_zoom") and "INVALID_ARGUMENT" in line
    assert "bad mode" in line
    assert event.to_dict()["code"] == "INVALID_ARGUMENT"
    assert event.to_dict()["job_id"] == "job-1"


def test_redact_args_hides_pixels_and_secrets_but_keeps_shapes():
    redacted = redact_args({
        "base64_png_jpg": "A" * 500,
        "auth_token": "hunter2",
        "path": "/tmp/page.png",
        "long": "x" * 300,
        "mode": "fit",
        "nested": {"block_ids": [1, 2, 3], "deep": {"base64_png_jpg": "Q" * 90}},
    })
    assert redacted["base64_png_jpg"] == "<500 chars>"
    assert redacted["auth_token"] == "<7 chars>"
    assert redacted["path"] == "/tmp/page.png"
    assert "(+180 chars)" in redacted["long"] and len(redacted["long"]) < 140
    assert redacted["mode"] == "fit"
    assert redacted["nested"]["block_ids"] == [1, 2, 3]
    assert redacted["nested"]["deep"] == {"base64_png_jpg": "<90 chars>"}


def test_redact_args_counts_bytes_without_keeping_them():
    redacted = redact_args({"payload": b"0123456789"})
    assert redacted["payload"] == "<10 bytes>"


# ---------------------------------------------------------------------------
# Inline dispatcher (headless)
# ---------------------------------------------------------------------------

def test_inline_dispatcher_runs_now_and_propagates_errors():
    dispatcher = InlineDispatcher()
    assert dispatcher.call(lambda: 6 * 7) == 42
    with pytest.raises(StudioError) as excinfo:
        dispatcher.call(lambda: (_ for _ in ()).throw(StudioError(ErrorCode.NO_IMAGE, "nope")))
    assert excinfo.value.code is ErrorCode.NO_IMAGE
    assert dispatcher.describe() == "inline"


# ---------------------------------------------------------------------------
# Qt dispatcher
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def run_from_worker(dispatcher, fn, app, *, timeout=5.0):
    """Submit `fn` from another thread; pump the main loop until the future settles."""
    box: dict[str, object] = {}

    def worker():
        try:
            box["value"] = dispatcher.call(fn, timeout=timeout)
        except BaseException as exc:                          # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    thread.join(1.0)
    app.processEvents()
    return box


def test_qt_dispatcher_runs_the_callable_on_the_main_thread(qapp):
    # `QThread.currentThreadId` does not exist in PySide6, and what actually matters is
    # Python's thread identity anyway: Qt widget access is only legal on this thread.
    dispatcher = QtDispatcher(qapp)
    main_thread = threading.get_ident()
    box = run_from_worker(dispatcher, threading.get_ident, qapp)
    assert "error" not in box, box.get("error")
    assert box["value"] == main_thread, "the callable must not run on the caller's thread"
    assert "qt-main-thread" in dispatcher.describe()


def test_qt_dispatcher_carries_exceptions_back_to_the_caller(qapp):
    dispatcher = QtDispatcher(qapp)

    def boom():
        raise StudioError(ErrorCode.BUSY, "job-7 is running", {"job_id": "job-7"})

    box = run_from_worker(dispatcher, boom, qapp)
    error = box.get("error")
    assert isinstance(error, StudioError) and error.code is ErrorCode.BUSY
    assert error.details == {"job_id": "job-7"}


def test_qt_dispatcher_stays_out_of_its_own_thread(qapp):
    """From the main thread a queued call would deadlock — so it runs inline."""
    dispatcher = QtDispatcher(qapp)
    assert dispatcher.call(threading.get_ident) == threading.get_ident()


def test_qt_dispatcher_times_out_instead_of_hanging(qapp):
    """No event pump -> the caller gives up on a timeout instead of blocking forever."""
    from concurrent import futures

    dispatcher = QtDispatcher(qapp)
    box: dict[str, object] = {}

    def worker():
        box["future"] = dispatcher.submit(lambda: "late")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    for _ in range(20):                       # let the worker submit, but do NOT pump
        time.sleep(0.005)
    future = box["future"]
    with pytest.raises(futures.TimeoutError):
        future.result(0.05)
    for _ in range(50):                       # now pump: the queued call still runs
        qapp.processEvents()
        time.sleep(0.002)
        if future.done():
            break
    assert future.result(2.0) == "late"
    thread.join(2.0)


def test_closed_dispatcher_answers_with_a_structured_error(qapp):
    dispatcher = QtDispatcher(qapp)
    dispatcher.close()
    with pytest.raises(StudioError) as excinfo:
        dispatcher.call(lambda: 1)
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert "closed" in excinfo.value.message.lower()
