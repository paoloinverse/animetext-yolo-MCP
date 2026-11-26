"""Part 4 tests — `JobManager`: single-flight, cancellation, listeners, the result cache.

These are the guarantees the GUI (P5) and the MCP server (P7) both rest on, so they are
tested against the real threading, not a fake scheduler: a job in flight must refuse a
second one, cancel must actually stop the worker, and a listener that throws or re-enters
the manager must never take the job down with it.
"""
from __future__ import annotations

import threading
import time

import pytest
from PIL import Image

from animetext_studio.core.config import AppConfig
from animetext_studio.core.imaging import describe
from animetext_studio.core.jobs import JobManager
from animetext_studio.core.pipeline import Pipeline, engines_from_config
from animetext_studio.core.types import (
    BBox, Detection, ErrorCode, JobState, StudioError, TEXT_LABEL,
)

W, H = 300, 200


def image() -> Image.Image:
    base = Image.new("RGB", (W, H), "white")
    pixels = base.load()
    for x in range(W):
        for y in range(H):
            pixels[x, y] = (x % 256, y % 256, 128)
    return base


def info():
    return describe(image(), source="clipboard")


def det(x0, y0, x1, y1, score=0.8) -> Detection:
    return Detection(BBox(x0, y0, x1, y1), TEXT_LABEL, score)


class GatedDetection:
    """Holds the worker inside `detect()` until `release` is set; honours cancellation."""

    name = "detection"

    def __init__(self, detections=None, error=None, exception=None, blocked: bool = False):
        self.detections = list(detections if detections is not None
                               else [det(10, 10, 120, 90), det(150, 100, 280, 190)])
        self.error = error
        self.exception = exception
        self.release = threading.Event()
        if not blocked:
            self.release.set()          # normal stub: detect() returns at once
        self.entered = threading.Event()
        self.calls = 0
        self.unloaded = 0

    def status(self):
        from types import SimpleNamespace

        return SimpleNamespace(backend="stub", device="cpu", name=self.name,
                               state="ready", loaded_models=(), detail="")

    def unload(self):
        self.unloaded += 1

    def detect(self, image, model_name, **kwargs):
        self.calls += 1
        self.entered.set()
        cancel = kwargs.get("cancel_event")
        while not self.release.is_set():
            if cancel is not None and cancel.is_set():
                raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")
            time.sleep(0.002)
        if self.error is not None:
            raise self.error
        if self.exception is not None:
            raise self.exception
        return list(self.detections)


class QuietOcr:
    name = "ocr"

    def __init__(self, texts=("alpha", "beta"), error=None):
        self.texts = list(texts)
        self.error = error
        self.calls = 0
        self.unloaded = 0

    def status(self):
        from types import SimpleNamespace

        return SimpleNamespace(backend="stub", device="cpu", name=self.name,
                               state="ready", loaded_models=(), detail="")

    def unload(self):
        self.unloaded += 1

    def ocr_crops(self, images, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [self.texts[i] if i < len(self.texts) else "z" for i in range(len(images))]


def manager(detection=None, ocr=None, **kwargs) -> JobManager:
    pipeline = Pipeline(detection or GatedDetection(), ocr or QuietOcr(),
                        AppConfig.default())
    return JobManager(pipeline, **kwargs)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_a_job_runs_to_done_and_caches_its_result(self):
        job = manager()
        assert job.state is JobState.IDLE and job.result is None
        job.start(image(), info=info())
        settled = job.wait(timeout=5)
        assert settled.state is JobState.DONE
        assert job.result is not None and job.result.block_count == 2
        assert job.progress.percent == 100.0
        assert job.state is JobState.DONE and not job.is_busy

    def test_status_is_json_ready_and_describes_the_manager(self):
        job = manager()
        before = job.status()
        assert set(before) == {"job_id", "kind", "state", "stage", "percent", "message",
                               "busy", "has_result", "block_count", "error"}
        assert before["kind"] == "run"        # P8.3: "run" produces a document, "warmup" does not
        assert before["state"] == "idle" and before["busy"] is False
        job.start(image(), info=info())
        job.wait(timeout=5)
        after = job.status()
        assert after["has_result"] is True and after["block_count"] == 2
        assert after["error"] is None
        import json

        json.dumps(after)                              # must be serialisable for MCP

    def test_job_ids_increment_and_the_latest_result_wins(self):
        job = manager()
        job.start(image(), info=info())
        job.wait(timeout=5)
        first = job.result
        assert job.job_id == "job-1"
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert job.job_id == "job-2"
        assert job.result is not first
        assert job.result.block_count == first.block_count

    def test_wait_without_a_job_returns_immediately(self):
        job = manager()
        started = time.monotonic()
        assert job.wait(timeout=0.5).state is JobState.IDLE
        assert time.monotonic() - started < 0.4

    def test_start_bytes_and_start_file_use_the_same_path(self, tmp_path):
        from animetext_studio.core.imaging import to_png_bytes

        data = to_png_bytes(image())
        job = manager()
        job.start_bytes(data, file_name="page.png")
        assert job.wait(timeout=5).state is JobState.DONE
        assert job.result.image.source == "mcp-base64"

    def test_the_engines_are_reused_across_jobs_not_rebuilt(self):
        detection, ocr = GatedDetection(), QuietOcr()
        job = manager(detection, ocr)
        job.start(image(), info=info())
        job.wait(timeout=5)
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert detection.calls == 2 and ocr.calls == 2
        assert job.detection is detection and job.ocr is ocr

    def test_unload_engines_frees_both_after_waiting(self):
        detection, ocr = GatedDetection(), QuietOcr()
        job = manager(detection, ocr)
        job.start(image(), info=info())
        job.wait(timeout=5)
        job.unload_engines()
        assert detection.unloaded == 1 and ocr.unloaded == 1


# ---------------------------------------------------------------------------
# Busy: a second job is refused, not queued
# ---------------------------------------------------------------------------

class TestSingleFlight:
    def test_a_second_job_is_refused_while_one_is_running(self):
        detection = GatedDetection(blocked=True)
        job = manager(detection)
        job.start(image(), info=info())
        assert detection.entered.wait(2), "worker never reached detect()"
        with pytest.raises(StudioError) as ei:
            job.start(image(), info=info())
        assert ei.value.code is ErrorCode.BUSY
        assert ei.value.details["job_id"] == "job-1"
        assert ei.value.exit_code == 4
        detection.release.set()
        assert job.wait(timeout=5).state is JobState.DONE

    def test_a_refused_start_does_not_disturb_the_running_job(self):
        detection = GatedDetection(blocked=True)
        seen: list[JobState] = []
        job = manager(detection, on_progress=lambda p: seen.append(p.state))
        job.start(image(), info=info())
        detection.entered.wait(2)
        with pytest.raises(StudioError):
            job.start(image(), info=info())
        detection.release.set()
        job.wait(timeout=5)
        assert seen[-1] is JobState.DONE
        assert JobState.ERROR not in seen

    def test_a_new_job_is_accepted_once_the_previous_one_settles(self):
        job = manager()
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert job.start(image(), info=info()).state is JobState.RUNNING
        job.wait(timeout=5)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

class TestCancellation:
    def test_cancel_stops_the_worker_and_lands_on_cancelled(self):
        detection = GatedDetection(blocked=True)
        job = manager(detection)
        job.start(image(), info=info())
        detection.entered.wait(2)
        assert job.cancel() is True
        assert job.wait(timeout=5).state is JobState.CANCELLED
        assert job.result is None                       # a cancelled job has no result

    def test_cancelling_transitions_through_cancelling(self):
        detection = GatedDetection(blocked=True)
        states: list[JobState] = []
        job = manager(detection, on_progress=lambda p: states.append(p.state))
        job.start(image(), info=info())
        detection.entered.wait(2)
        job.cancel()
        job.wait(timeout=5)
        assert JobState.CANCELLING in states
        assert states[-1] is JobState.CANCELLED

    def test_cancel_is_false_when_nothing_is_running(self):
        job = manager()
        assert job.cancel() is False
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert job.cancel() is False

    def test_cancelling_twice_is_harmless(self):
        detection = GatedDetection(blocked=True)
        job = manager(detection)
        job.start(image(), info=info())
        detection.entered.wait(2)
        assert job.cancel() is True and job.cancel() is True
        detection.release.set()
        job.wait(timeout=5)
        assert job.state in (JobState.CANCELLED, JobState.DONE)

    def test_a_job_that_finishes_after_a_cancel_request_is_done_not_cancelled(self):
        detection = GatedDetection(blocked=True)
        job = manager(detection)
        job.start(image(), info=info())
        detection.entered.wait(2)
        detection.release.set()                 # let it finish, then cancel immediately
        job.wait(timeout=5)
        job.cancel()                            # too late: the job already settled
        assert job.state is JobState.DONE and job.result is not None

    def test_run_file_raises_the_cancellation(self):
        detection = GatedDetection(blocked=True)
        job = manager(detection)
        started = job.start(image(), info=info())

        def cancel_soon():
            detection.entered.wait(2)
            job.cancel()

        threading.Thread(target=cancel_soon, daemon=True).start()
        with pytest.raises(StudioError) as ei:
            job._await(started, 5)
        assert ei.value.code is ErrorCode.CANCELLED


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_a_domain_error_becomes_the_error_state_and_reraises(self):
        detection = GatedDetection(error=StudioError(ErrorCode.MODEL_NOT_FOUND, "gone"))
        job = manager(detection)
        job.start(image(), info=info())
        settled = job.wait(timeout=5)
        assert settled.state is JobState.ERROR
        assert job.error.code is ErrorCode.MODEL_NOT_FOUND
        assert job.status()["error"]["code"] == "MODEL_NOT_FOUND"
        assert job.result is None

    def test_run_file_unwraps_the_stored_error(self):
        detection = GatedDetection(error=StudioError(ErrorCode.ENGINE_LOAD_FAILED, "no torch"))
        job = manager(detection)
        with pytest.raises(StudioError) as ei:
            job._await(job.start(image(), info=info()), 5)
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED

    def test_an_unexpected_exception_is_wrapped_as_internal(self):
        detection = GatedDetection(exception=ValueError("boom"))
        job = manager(detection)
        job.start(image(), info=info())
        assert job.wait(timeout=5).state is JobState.ERROR
        assert job.error.code is ErrorCode.INTERNAL
        assert "boom" in job.error.details["reason"]

    def test_a_fatal_ocr_error_is_a_job_error(self):
        ocr = QuietOcr(error=StudioError(ErrorCode.ENGINE_LOAD_FAILED, "no model"))
        job = manager(GatedDetection(), ocr)
        job.start(image(), info=info())
        assert job.wait(timeout=5).state is JobState.ERROR
        assert job.error.code is ErrorCode.ENGINE_LOAD_FAILED

    def test_an_isolated_block_failure_is_still_a_successful_job(self):
        # One unreadable crop must not turn the page into an ERROR (D37): the pipeline
        # isolates it and the job completes, with that block carrying the error.
        class PoisonOcr(QuietOcr):
            def ocr_crops(self, images, **kwargs):
                self.calls += 1
                if len(images) > 1:
                    raise StudioError(ErrorCode.OCR_FAILED, "batch failed")
                if self.calls == 2:
                    raise StudioError(ErrorCode.OCR_FAILED, "this crop is unreadable")
                return ["ok"]

        job = manager(GatedDetection(), PoisonOcr())
        result = job._await(job.start(image(), info=info()), 5)
        assert job.state is JobState.DONE
        assert sum(1 for block in result.blocks if block.error) == 1
        assert sum(1 for block in result.blocks if block.text) == 1


# ---------------------------------------------------------------------------
# Listeners
# ---------------------------------------------------------------------------

class TestListeners:
    def test_a_listener_sees_progress_and_the_final_state(self):
        events = []
        job = manager(on_progress=lambda p: events.append(p))
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert events, "listener never called"
        assert events[0].job_id == "job-1"
        assert events[-1].state is JobState.DONE and events[-1].percent == 100.0
        percents = [e.percent for e in events]
        assert percents == sorted(percents)

    def test_listeners_can_be_added_and_removed(self):
        job = manager()
        seen: list[str] = []
        listener = lambda p: seen.append(p.stage)          # noqa: E731
        job.add_listener(listener)
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert seen
        count = len(seen)
        job.remove_listener(listener)
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert len(seen) == count

    def test_a_listener_that_throws_does_not_kill_the_job(self):
        def angry(progress):
            raise RuntimeError("GUI slot exploded")

        job = manager(on_progress=angry)
        job.start(image(), info=info())
        assert job.wait(timeout=5).state is JobState.DONE
        assert job.result is not None

    def test_a_listener_may_call_back_into_the_manager(self):
        seen: list[str] = []

        def nosy(progress):
            seen.append(job.status()["state"])      # re-entrant: must not deadlock

        job = manager(on_progress=nosy)
        job.start(image(), info=info())
        done = threading.Event()

        def wait_and_check():
            job.wait(timeout=5)
            done.set()

        thread = threading.Thread(target=wait_and_check, daemon=True)
        thread.start()
        thread.join(5)
        assert done.is_set(), "re-entrant listener deadlocked the manager"
        assert seen and seen[-1] == "done"

    def test_progress_events_carry_the_stage_names_the_gui_renders(self):
        events = []
        job = manager(on_progress=lambda p: events.append(p.stage))
        job.start(image(), info=info())
        job.wait(timeout=5)
        assert "queue" in events and "detect" in events and "ocr" in events
        assert events[-1] == "done"


# ---------------------------------------------------------------------------
# Real fakes end-to-end through the manager
# ---------------------------------------------------------------------------

class TestWithRealFakes:
    def test_a_mock_page_through_the_manager_is_valid_and_cached(self):
        engines = engines_from_config(AppConfig.default(), mock=True)
        job = JobManager(Pipeline(engines.detection, engines.ocr))
        result = job.run_file("examples/sample_page.png", timeout=30)
        assert result.block_count >= 1
        assert job.result is result
        assert job.status()["state"] == "done"

    def test_two_managers_do_not_share_state(self):
        engines_a = engines_from_config(AppConfig.default(), mock=True)
        engines_b = engines_from_config(AppConfig.default(), mock=True)
        a = JobManager(Pipeline(engines_a.detection, engines_a.ocr))
        b = JobManager(Pipeline(engines_b.detection, engines_b.ocr))
        a.run_file("examples/sample_page.png", timeout=30)
        assert a.result is not None and b.result is None
        assert b.status()["state"] == "idle"
