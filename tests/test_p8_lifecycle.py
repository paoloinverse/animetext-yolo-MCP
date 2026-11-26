"""P8.3 — warm-up, engine warmth, and memory release.

Three claims have to be provable rather than asserted in prose:

1. **`--preload` really warms the engines** — weights loaded *and* one inference run, because a
   loaded-but-unrun session on CUDA is not the same cost as a used one.
2. **A warm-up is a job and not a page** — it holds the single-flight lock, reports on the
   progress channel, is cancellable, and leaves the previous result document alone.
3. **`unload` gives the memory back** — measured, not hoped for.

Everything here is stub- or mock-engine; the two `-m models` tests are the only ones that touch
real weights, and they are the only ones that claim anything about ONNX Runtime's behaviour.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pytest
from PIL import Image

from animetext_studio.core.config import AppConfig
from animetext_studio.core.jobs import JobManager
from animetext_studio.core.memory import process_rss_mb, rss_available
from animetext_studio.core.pipeline import Pipeline
from animetext_studio.core.types import (
    ErrorCode, JobState, StudioError,
)
from animetext_studio.core.warmup import (DEFAULT_SWEEP_DEVICES, WARM_STAGE, WarmupOutcome,
                             WarmupReport, warm_devices, warm_engines)
from animetext_studio.engines.base import (
    BaseOcrEngine, EngineState, EngineStatus, FakeDetectionEngine, FakeOcrEngine,
)

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "sample_page.png")


# ---------------------------------------------------------------------------
# Recording stubs: the point is to see *what warm-up asked for*
# ---------------------------------------------------------------------------

class StubDetection:
    name = "detection"

    def __init__(self, *, load_error=None, detect_error=None, hold=None):
        self.load_error, self.detect_error, self.hold = load_error, detect_error, hold
        self.calls: list[tuple] = []
        self.state = EngineState.UNLOADED
        self.inferences = 0

    def status(self) -> EngineStatus:
        return EngineStatus(name="detection", state=self.state,
                            loaded_models=("stub",) if self.state is EngineState.READY else (),
                            backend="stub", device="cpu", detail="", inferences=self.inferences)

    def _hold(self) -> None:
        """Block until cancelled, announcing that we are inside the slow part.

        Real engines spend their whole first-call latency here (weights, session, CUDA context),
        and `detect()` may trigger that load itself — so both entry points honour the hold.
        """
        if self.hold is None:
            return
        self.hold["in_load"].set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not self.hold["cancel"].is_set():
            time.sleep(0.005)
        if self.hold["cancel"].is_set():
            raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")

    def load(self, model_name, progress=None) -> None:
        self.calls.append(("load", model_name))
        self._hold()
        if self.load_error is not None:
            raise self.load_error
        self.state = EngineState.READY

    def detect(self, image, model_name, **kwargs):
        self.calls.append(("detect", model_name, image.size))
        if self.state is not EngineState.READY:
            self._hold()
            self.state = EngineState.READY
        if self.detect_error is not None:
            raise self.detect_error
        self.inferences += 1
        return []


class StubOcr:
    name = "ocr"

    def __init__(self, *, load_error=None, ocr_error=None):
        self.load_error, self.ocr_error = load_error, ocr_error
        self.calls: list[tuple] = []
        self.state = EngineState.UNLOADED
        self.inferences = 0

    def status(self) -> EngineStatus:
        return EngineStatus(name="ocr", state=self.state,
                            loaded_models=("stub",) if self.state is EngineState.READY else (),
                            backend="stub", device="cpu", detail="", inferences=self.inferences)

    def load(self, progress=None) -> None:
        self.calls.append(("load",))
        if self.load_error is not None:
            raise self.load_error
        self.state = EngineState.READY

    def ocr_crops(self, images, **kwargs):
        self.calls.append(("ocr", tuple(i.size for i in images)))
        if self.ocr_error is not None:
            raise self.ocr_error
        self.inferences += 1
        return ["" for _ in images]


def manager(detection=None, ocr=None) -> JobManager:
    cfg = AppConfig.default()
    return JobManager(Pipeline(detection or FakeDetectionEngine(),
                               ocr or FakeOcrEngine(), cfg))


# ---------------------------------------------------------------------------
# warm_engines()
# ---------------------------------------------------------------------------

class TestWarmEngines:
    def test_it_loads_both_engines_and_runs_one_inference_through_each(self):
        det, ocr = StubDetection(), StubOcr()
        report = warm_engines(det, ocr, AppConfig.default())
        assert det.calls[0] == ("load", AppConfig.default().detection.model)
        assert [c[0] for c in det.calls] == ["load", "detect"]
        assert [c[0] for c in ocr.calls] == ["load", "ocr"]
        # The probe is a throwaway: tiny, and its output is discarded.
        assert det.calls[1][2] == (64, 64) and ocr.calls[1][1] == ((64, 64),)
        assert report.ok and report.probe
        assert [o.engine for o in report.outcomes] == ["detection", "ocr"]
        assert all(o.ok and o.state == "ready" for o in report.outcomes)

    def test_probe_false_loads_without_inferring_and_is_not_warm(self):
        det, ocr = StubDetection(), StubOcr()
        report = warm_engines(det, ocr, AppConfig.default(), probe=False)
        assert [c[0] for c in det.calls] == ["load"] and [c[0] for c in ocr.calls] == ["load"]
        assert det.state is EngineState.READY
        # The distinction P8.3 exists to make visible: loaded is not warm.
        assert det.status().warm is False and det.status().inferences == 0
        assert report.probe is False and report.ok

    def test_progress_is_reported_on_the_warmup_stage_only(self):
        seen: list[tuple[str, float]] = []
        warm_engines(StubDetection(), StubOcr(), AppConfig.default(),
                     progress=lambda stage, percent, msg: seen.append((stage, percent)))
        assert seen and all(stage == WARM_STAGE for stage, _ in seen)
        assert seen[0][1] == 0.0 and seen[-1][1] == 100.0

    def test_a_broken_detection_does_not_stop_the_ocr_warmup(self):
        boom = StudioError(ErrorCode.HF_AUTH_REQUIRED, "The repo is gated.")
        det, ocr = StubDetection(load_error=boom), StubOcr()
        report = warm_engines(det, ocr, AppConfig.default())
        assert report.ok is False
        by = {o.engine: o for o in report.outcomes}
        assert by["detection"].error["code"] == "HF_AUTH_REQUIRED"
        assert by["ocr"].ok is True                      # the other engine still came up
        with pytest.raises(StudioError) as ei:
            report.raise_for_failures()
        assert ei.value.code is ErrorCode.HF_AUTH_REQUIRED
        assert ei.value.details["engine"] == "detection"
        assert ei.value.details["warmup"]["outcomes"][1]["ok"] is True

    def test_a_clean_report_raises_nothing_and_summarises_itself(self):
        report = warm_engines(StubDetection(), StubOcr(), AppConfig.default())
        assert report.raise_for_failures() is report
        text = report.summary()
        assert text.startswith("warm-up: ") and "detection warm" in text and "ocr warm" in text
        broken = WarmupReport((report.outcomes[0],
                               report.outcomes[1].__class__(engine="ocr", ok=False)), 1)
        assert "ocr FAILED" in broken.summary()

    def test_cancellation_propagates_instead_of_becoming_a_report(self):
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(StudioError) as ei:
            warm_engines(StubDetection(), StubOcr(), AppConfig.default(), cancel_event=cancel)
        assert ei.value.code is ErrorCode.CANCELLED

    def test_no_engines_is_not_an_error(self):
        report = warm_engines(None, None, AppConfig.default())
        assert report.outcomes == () and report.ok and report.summary() == "warm-up: nothing to do"


# ---------------------------------------------------------------------------
# Warm-up as a job
# ---------------------------------------------------------------------------

class TestWarmupJob:
    def test_it_warms_and_reports_through_the_progress_channel(self):
        det, ocr = StubDetection(), StubOcr()
        mgr = manager(det, ocr)
        stages: list[str] = []
        mgr.add_listener(lambda p: stages.append(p.stage))
        mgr.start_warmup()
        done = mgr.wait()
        assert done.state is JobState.DONE
        assert "warmup" in stages and "detect" not in stages
        assert mgr.kind == "warmup" and mgr.status()["kind"] == "warmup"
        assert det.status().warm and ocr.status().warm
        assert mgr.warmup_report is not None and mgr.warmup_report.ok
        assert "warm-up:" in mgr.status()["message"]

    def test_it_produces_no_result_and_leaves_the_previous_page_alone(self):
        mgr = manager()
        mgr.start_file(SAMPLE)
        mgr.wait()
        first = mgr.result
        assert first is not None and mgr.status()["has_result"] is True
        mgr.start_warmup()
        assert mgr.wait().state is JobState.DONE
        assert mgr.result is first                       # the page survived its warm-up
        assert mgr.status()["has_result"] is True
        assert mgr.kind == "warmup"

    def test_single_flight_both_ways(self):
        hold = {"in_load": threading.Event(), "cancel": threading.Event()}
        mgr = manager(StubDetection(hold=hold), StubOcr())
        mgr.start_warmup()
        assert hold["in_load"].wait(5.0)
        with pytest.raises(StudioError) as ei:
            mgr.start_file(SAMPLE)
        assert ei.value.code is ErrorCode.BUSY
        hold["cancel"].set()
        mgr.wait()

        # …and the other way round: a warm-up must not start on top of a page. The run is held
        # open by a detection stub that blocks inside load(), which is the slowest thing a real
        # first run does — so the BUSY window is the one that matters in practice.
        hold_run = {"in_load": threading.Event(), "cancel": threading.Event()}
        mgr2 = manager(StubDetection(hold=hold_run), StubOcr())
        mgr2.start_file(SAMPLE)
        assert hold_run["in_load"].wait(5.0)
        with pytest.raises(StudioError) as ei2:
            mgr2.start_warmup()
        assert ei2.value.code is ErrorCode.BUSY
        hold_run["cancel"].set()
        mgr2.wait()

    def test_a_failed_warmup_ends_the_job_error_with_the_engines_own_code(self):
        boom = StudioError(ErrorCode.MODEL_NOT_FOUND, "Not on the Hub.")
        mgr = manager(StubDetection(load_error=boom), StubOcr())
        mgr.start_warmup()
        done = mgr.wait()
        assert done.state is JobState.ERROR
        assert mgr.error is not None and mgr.error.code is ErrorCode.MODEL_NOT_FOUND
        # The engine that did come up is still in the report the error carries.
        assert mgr.error.details["warmup"]["outcomes"][1]["ok"] is True

    def test_a_warmup_can_be_cancelled_mid_load(self):
        hold = {"in_load": threading.Event(), "cancel": threading.Event()}
        mgr = manager(StubDetection(hold=hold), StubOcr())
        mgr.start_warmup()
        assert hold["in_load"].wait(5.0)
        assert mgr.cancel() is True
        done = mgr.wait()
        assert done.state is JobState.CANCELLED
        assert mgr.warmup_report is None                 # nothing finished, nothing to report


# ---------------------------------------------------------------------------
# Warmth is part of the engine contract
# ---------------------------------------------------------------------------

class TestEngineWarmth:
    def test_inferences_count_up_and_reset_on_unload(self):
        det = FakeDetectionEngine()
        assert det.status().inferences == 0 and det.status().warm is False
        det.detect(Image.open(SAMPLE), "yolo12m_animetext")
        assert det.status().inferences == 1 and det.status().warm is True
        det.detect(Image.open(SAMPLE), "yolo12m_animetext")
        assert det.status().inferences == 2
        det.unload()
        assert det.status().inferences == 0 and det.status().warm is False
        assert det.status().state is EngineState.UNLOADED

    def test_ocr_counts_one_inference_per_call_not_per_crop(self):
        ocr = FakeOcrEngine()
        crops = [Image.new("RGB", (32, 32)) for _ in range(4)]
        ocr.ocr_crops(crops)
        ocr.ocr_crops(crops)
        assert ocr.status().inferences == 2

    def test_status_to_dict_carries_the_evidence(self):
        det = FakeDetectionEngine()
        det.load("yolo12m_animetext")
        cold = det.status().to_dict()
        assert cold["inferences"] == 0 and cold["warm"] is False
        det.detect(Image.open(SAMPLE), "yolo12m_animetext")
        warm = det.status().to_dict()
        assert warm["inferences"] == 1 and warm["warm"] is True

    def test_the_mcp_status_reports_loaded_warm_and_the_job_kind(self):
        from animetext_studio.mcp.service import StudioService

        cfg = AppConfig.default()
        mgr = manager(FakeDetectionEngine(), FakeOcrEngine())
        service = StudioService(mgr, cfg, mock=True)
        cold = service.snapshot()["engines"]
        assert [e["loaded"] for e in cold] == [False, False]   # P8.3: this was always false
        mgr.start_warmup()
        mgr.wait()
        snap = service.snapshot()
        assert [e["loaded"] for e in snap["engines"]] == [True, True]
        assert [e["warm"] for e in snap["engines"]] == [True, True]
        assert [e["state"] for e in snap["engines"]] == ["ready", "ready"]
        assert snap["job"]["kind"] == "warmup"
        # `unload_engines` is visible in the same fields, which is what makes the RSS claim
        # checkable from outside the process.
        service.unload()
        assert [e["warm"] for e in service.snapshot()["engines"]] == [False, False]


class TestWarmEnginesTool:
    """The MCP surface needs the counterpart of `unload_engines` (P8.3)."""

    def service(self):
        from animetext_studio.core.jobs import JobManager
        from animetext_studio.core.pipeline import Pipeline
        from animetext_studio.engines.base import FakeDetectionEngine, FakeOcrEngine
        from animetext_studio.mcp.service import StudioService

        cfg = AppConfig.default()
        mgr = JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), cfg))
        return StudioService(mgr, cfg, mock=True), mgr

    def test_it_answers_with_a_job_handle_not_a_wait(self):
        service, mgr = self.service()
        out = service.warm_engines()
        assert out["kind"] == "warmup" and out["job_id"] and out["busy"] is True
        mgr.wait()
        assert mgr.kind == "warmup" and mgr.warmup_report is not None

    def test_a_second_call_while_it_runs_is_busy(self):
        service, mgr = self.service()
        service.warm_engines()
        mgr.wait()
        # jobs are fast with fakes, so prove the guard at the manager level, where it lives
        mgr.start_warmup()
        with pytest.raises(StudioError) as ei:
            service.warm_engines()
        assert ei.value.code is ErrorCode.BUSY
        mgr.wait()

    def test_probe_false_from_cold_loads_without_making_it_warm(self):
        service, mgr = self.service()
        service.warm_engines(probe=False)
        mgr.wait()
        engines = {e["name"]: e for e in service.snapshot()["engines"]}
        assert engines["detection"]["loaded"] is True       # weights are in
        assert engines["detection"]["warm"] is False        # but nothing has been inferred
        assert engines["ocr"]["loaded"] is True and engines["ocr"]["warm"] is False

    def test_get_status_reports_warmth_after_a_real_probe(self):
        service, mgr = self.service()
        assert all(e["warm"] is False for e in service.snapshot()["engines"])
        service.warm_engines()
        mgr.wait()
        engines = {e["name"]: e for e in service.snapshot()["engines"]}
        assert engines["detection"]["warm"] is True and engines["ocr"]["warm"] is True
        assert engines["detection"]["state"] == "ready"     # the field that used to lie


# ---------------------------------------------------------------------------
# Memory release
# ---------------------------------------------------------------------------

#: Runs in a fresh interpreter (see `TestMemoryRelease`): load, measure, unload, measure. The
#: file is deliberately standalone — it must not import anything that allocates a lot first,
#: or the measurement stops meaning "what one engine load costs".
_SLAB_SUBPROCESS_PROBE = '''
import gc, json, sys
from animetext_studio.core.memory import process_rss_mb, trim_heap
from animetext_studio.engines.base import BaseOcrEngine, EngineState


class Slab(BaseOcrEngine):
    name = "ocr"

    def __init__(self):
        super().__init__(backend="stub", device="cpu")
        self.slab = None

    def load(self, progress=None):
        with self._lock:
            slab = bytearray(96 * 1024 * 1024)
            slab[::4096] = b"\\x01" * len(slab[::4096])
            self.slab = slab
            self._state = EngineState.READY

    def ocr_crops(self, images, **kw):
        return ["" for _ in images]

    def _unload_locked(self):
        self.slab = None
        trim_heap()


engine = Slab()
baseline = process_rss_mb()
engine.load()
peak = process_rss_mb()
engine.unload()
gc.collect()
after = process_rss_mb()
print(json.dumps({"baseline": baseline, "peak": peak, "after": after,
                  "took": peak - baseline, "state": engine.status().state.value}))
'''

class SlabOcrEngine(BaseOcrEngine):
    """An engine that allocates a recognisable slab of RAM, so a release can be measured."""

    name = "ocr"

    def __init__(self, mb: int = 64) -> None:
        super().__init__(backend="stub", device="cpu")
        self.mb = mb
        self.slab: Optional[bytearray] = None

    def load(self, progress=None) -> None:
        with self._lock:
            slab = bytearray(self.mb * 1024 * 1024)
            slab[::4096] = b"\x01" * len(slab[::4096])       # touch it: untouched pages are free
            self.slab = slab
            self._state = EngineState.READY

    def ocr_crops(self, images, **kwargs):
        self._note_inference()
        return ["" for _ in images]

    def _unload_locked(self) -> None:
        self.slab = None
        from animetext_studio.core.memory import trim_heap

        trim_heap()          # the real engines do this (D62); the stub should behave the same


def _rss_or_skip() -> float:
    if not rss_available():
        pytest.skip("no RSS reading on this platform")
    gc.collect()
    return process_rss_mb()


class TestMemoryRelease:
    """What `unload` provably does, in the two measurements that can each be honest.

    One test used to claim "unload gives back at least half of what load took", and it was
    flaky: whether freed pages return to the OS depends on what the allocator had already done
    in this process (glibc raises its mmap threshold on the first big free, after which a slab
    of that size comes from the heap and stays there when freed). Run alone it passed; run after
    the rest of the suite it sometimes did not. That is a measurement artefact, not a regression
    we can pin, so the claim is split in two (P8.4):

    * **in this process** — the object goes, RSS never grows, and a *reload reuses* the space
      instead of stacking another slab on top. That is the property `unload_engines` is for,
      and it holds regardless of allocator history;
    * **in a clean process** — the strict "most of it comes back" number, measured where nothing
      else has touched the allocator, which is the only place that claim is well-defined.
    """

    def test_load_unload_cycles_do_not_leak(self):
        """In-process, the claim that survives the allocator: cycles do not stack.

        Whether a freed slab goes back to the OS here depends on what earlier tests allocated
        and freed in this same process (glibc raises its mmap threshold as it sees big frees, so
        the second 64 MB may come from the heap and stay resident). That is not ours to promise,
        and `test_in_a_clean_process_unload_returns_most_of_the_slab` makes it where the
        measurement is unambiguous instead.

        What *is* ours: dropping the references must not lose them anywhere else, so repeated
        load/unload cycles must not ratchet. Three cycles would show +128 MB if a reference
        escaped; the tolerance below is 32 MB.
        """
        _rss_or_skip()
        engine = SlabOcrEngine(mb=64)
        engine.load()
        peak = process_rss_mb()
        assert peak > 0
        engine.unload()
        gc.collect()
        assert engine.slab is None, "unload kept a reference to the slab"
        assert engine.status().state is EngineState.UNLOADED

        for _cycle in range(3):
            engine.load()
            assert engine.slab is not None
            engine.unload()
            gc.collect()
        final = process_rss_mb()
        assert final <= peak + 32, (
            f"three load/unload cycles left RSS at {final:.0f} MB against a first peak of "
            f"{peak:.0f} MB — something is holding the slab")

    def test_in_a_clean_process_unload_returns_most_of_the_slab(self, tmp_path):
        """The strict number, measured where the allocator has no history (see above)."""
        if not rss_available():
            pytest.skip("no RSS reading on this platform")
        script = tmp_path / "slab_probe.py"
        script.write_text(_SLAB_SUBPROCESS_PROBE, encoding="utf-8")
        env = dict(os.environ)
        root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        done = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=180, env=env)
        assert done.returncode == 0, done.stderr[-800:]
        data = json.loads(done.stdout.strip().splitlines()[-1])
        assert data["took"] > 40, f"the slab did not show up in RSS: {data}"
        assert data["after"] <= data["peak"] - 0.5 * data["took"], (
            f"unload returned {data['peak'] - data['after']:.0f} MB of the "
            f"{data['took']:.0f} MB the load took: {data}")
        assert data["state"] == "unloaded"

    def test_job_manager_unload_engines_clears_both_and_the_warm_flag(self):
        det, ocr = FakeDetectionEngine(), FakeOcrEngine()
        mgr = manager(det, ocr)
        mgr.start_warmup()
        mgr.wait()
        assert det.status().warm and ocr.status().warm
        mgr.unload_engines()
        assert det.status().state is EngineState.UNLOADED
        assert ocr.status().state is EngineState.UNLOADED
        assert det.status().inferences == 0 and ocr.status().inferences == 0


# ---------------------------------------------------------------------------
# Real weights: the only place ONNX Runtime's behaviour is asserted
# ---------------------------------------------------------------------------

def test_trim_heap_is_safe_wherever_it_runs():
    from animetext_studio.core.memory import trim_heap

    assert trim_heap() in (True, False)          # glibc: True; musl/Windows: False
    assert trim_heap() is trim_heap()            # idempotent, never raises


def test_live_sessions_counts_objects_not_our_own_bookkeeping():
    from animetext_studio.core.memory import live_sessions

    count = live_sessions()
    assert count is None or isinstance(count, int)   # None only when onnxruntime is absent
    assert count is None or count >= 0


@pytest.mark.models
def test_unload_releases_the_session_even_though_rss_has_a_floor():
    """What `unload_engines` actually buys, measured instead of assumed (decision D62).

    Two claims, both true on 2026-11-22 with onnxruntime 1.32, `yolo12m_animetext`, CPU
    provider: every `InferenceSession` is gone afterwards and a second load reuses that memory
    rather than stacking on top of it; RSS falls, but not to baseline, because ONNX Runtime
    keeps a few hundred MB in process-wide state that no Python API hands back (measured here:
    55 -> 999 MB peak -> ~808 MB after unload, ~755 MB after gc + `malloc_trim`).

    The old version of this test asserted "most of it comes back" and failed — against the
    runtime, not against us. What is ours to guarantee is that the *objects* go, the memory is
    *reusable*, and the `warm`/`loaded` flags stop lying. The floor belongs to the process, and
    the only real cure for it is exiting, which `unload_engines` does not claim to do.
    """
    from animetext_studio.core.memory import live_sessions
    from animetext_studio.engines.detection import DetectionEngine

    baseline = _rss_or_skip()
    engine = DetectionEngine(offline=True)
    engine.load("yolo12m_animetext")
    engine.detect(Image.open(SAMPLE), "yolo12m_animetext")     # warm: the session has run
    assert engine.status().warm is True
    peak = process_rss_mb()
    held = live_sessions()
    assert peak - baseline > 200, (
        f"loading a real ONNX session only moved RSS by {peak - baseline:.0f} MB; "
        "this test is measuring something other than the weights")
    assert held is not None and held >= 1, "the session object was never visible to gc"

    engine.unload()
    gc.collect()
    time.sleep(0.2)
    after = process_rss_mb()
    assert live_sessions() == 0, "unload left an InferenceSession alive: the weights are still held"
    assert engine.status().state is EngineState.UNLOADED
    assert engine.status().warm is False
    assert after < peak, f"unload released nothing at all ({peak:.0f} -> {after:.0f} MB)"
    assert peak - after > 50, f"only {peak - after:.0f} MB came back; the release has degraded"

    # The property that matters in a long-lived app: switching models reuses the memory. If
    # unload had only *hidden* the allocation, the second peak would stack.
    engine.load("yolo12s_animetext")
    engine.detect(Image.open(SAMPLE), "yolo12s_animetext")
    second = process_rss_mb()
    assert live_sessions() == 1
    assert second <= peak * 1.25, (
        f"reloading after unload cost {second:.0f} MB against a first peak of {peak:.0f} MB: "
        "the released memory was not reusable")
    engine.unload()


# ---------------------------------------------------------------------------
# GUI wiring: a warm-up must not look like a page
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6", reason="GUI wiring needs PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestGuiWiring:
    def test_the_bridge_routes_a_warmup_to_its_own_signal_not_finished(self, qapp):
        from animetext_studio.gui.workers import JobBridge

        mgr = manager(FakeDetectionEngine(), FakeOcrEngine())
        bridge = JobBridge(mgr)
        done, finished = [], []
        bridge.warmup_done.connect(lambda report: done.append(report))
        bridge.finished.connect(lambda result: finished.append(result))
        assert bridge.start_warmup() is True
        mgr.wait()
        for _ in range(50):
            qapp.processEvents()
            if done:
                break
            time.sleep(0.01)
        assert done and isinstance(done[0], WarmupReport)
        assert finished == [], "a warm-up re-emitted the previous page as a finished run"

    def test_start_preload_warms_the_window_and_says_so(self, qapp):
        from animetext_studio.gui.main_window import MainWindow

        mgr = manager(FakeDetectionEngine(), FakeOcrEngine())
        window = MainWindow(mgr, AppConfig.default(), mock=True)
        assert window.start_preload() is True
        assert window.start_preload() is False, "a second preload must not double-load"
        mgr.wait()
        for _ in range(50):
            qapp.processEvents()
            if "warm-up" in window.status_label.text():
                break
            time.sleep(0.01)
        assert "warm-up" in window.status_label.text()
        assert mgr.detection.status().warm is True
        window.close()


class TestDeviceSweep:
    """P9.4 — `warm_devices`: one cold load and one first inference per (engine, device).

    The question the sweep exists for is "is CUDA warm, and what did the first inference cost
    there?". What these tests hold is the part that is easy to get wrong: a row per device rather
    than one merged line, each device measured on a pair that was built for it and released
    afterwards, and the device the engine *got* printed next to the one that was *asked for*.
    """

    class Pair:
        """A pair that remembers being released — so "cold each time" is checkable."""

        def __init__(self, detection, ocr):
            self.detection, self.ocr = detection, ocr
            self.released = 0

        def unload_all(self) -> None:
            self.released += 1

    @classmethod
    def _factory(cls, built, *, detection_error=None, ocr_error=None):
        def build(device):
            pair = cls.Pair(StubDetection(load_error=detection_error),
                            StubOcr(ocr_error=ocr_error))
            built.append((device, pair))
            return pair
        return build

    def test_one_row_per_engine_per_device(self):
        built: list = []
        report = warm_devices(["cpu", "cuda"], AppConfig.default(),
                              factory=self._factory(built))
        assert [(o.engine, o.requested_device) for o in report.outcomes] == [
            ("detection", "cpu"), ("ocr", "cpu"), ("detection", "cuda"), ("ocr", "cuda")]
        assert report.devices == ("cpu", "cuda")
        assert [d for d, _pair in built] == ["cpu", "cuda"]

    def test_each_pair_is_released_so_the_next_row_starts_cold(self):
        built: list = []
        warm_devices(["cpu", "cuda"], AppConfig.default(), factory=self._factory(built))
        assert [pair.released for _d, pair in built] == [1, 1]

    def test_the_device_you_asked_for_survives_next_to_the_one_you_got(self):
        """The stub answers "cpu" whatever it is told — exactly the case the table must show."""
        built: list = []
        report = warm_devices(["cuda"], AppConfig.default(), factory=self._factory(built))
        row = report.outcomes[0]
        assert row.requested_device == "cuda" and row.device == "cpu"
        assert row.ok is True          # it warmed; it just did not warm where it was asked to

    def test_the_table_is_a_table(self):
        built: list = []
        report = warm_devices(["cpu", "cuda"], AppConfig.default(), factory=self._factory(built))
        text = report.table()
        lines = text.splitlines()
        assert lines[0].split() == ["requested", "engine", "device", "state",
                                    "load_ms", "probe_ms", "total_ms"]
        # header, rule, one row per (engine, device), blank line, summary line
        assert len(lines) == 2 + 4 + 2
        assert lines[6] == ""
        assert "4 row(s) over 2 device(s), none failed" in text
        assert "cuda" in text and "cpu" in text

    def test_a_failing_engine_is_a_row_not_a_crash(self):
        built: list = []
        report = warm_devices(["cpu"], AppConfig.default(),
                              factory=self._factory(built, ocr_error=StudioError(
                                  ErrorCode.ENGINE_LOAD_FAILED, "torch is not installed")))
        assert report.ok is False
        bad = report.failed[0]
        assert bad.engine == "ocr" and bad.requested_device == "cpu"
        text = report.table()
        assert "1 failed" in text and "ENGINE_LOAD_FAILED" in text
        with pytest.raises(StudioError) as ei:
            report.raise_for_failures()
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED

    def test_a_pair_that_cannot_be_built_is_still_a_row(self):
        def build(device):
            raise StudioError(ErrorCode.ENGINE_LOAD_FAILED, f"no provider for {device}")
        report = warm_devices(["tpu-ish"], AppConfig.default(), factory=build)
        assert report.ok is False
        assert report.outcomes[0].requested_device == "tpu-ish"
        assert report.outcomes[0].error["code"] == "ENGINE_LOAD_FAILED"

    def test_an_empty_device_list_is_refused(self):
        for bad in ([], [""], ["  "]):
            with pytest.raises(StudioError) as ei:
                warm_devices(bad, AppConfig.default())
            assert ei.value.code is ErrorCode.INVALID_ARGUMENT

    def test_duplicates_are_dropped_and_the_order_you_gave_is_kept(self):
        built: list = []
        report = warm_devices(["cuda", "cpu", "cuda"], AppConfig.default(),
                              factory=self._factory(built))
        assert report.devices == ("cuda", "cpu")
        assert [d for d, _p in built] == ["cuda", "cpu"]

    def test_a_cancelled_sweep_is_a_cancellation_not_a_short_table(self):
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(StudioError) as ei:
            warm_devices(["cpu", "cuda"], AppConfig.default(),
                         factory=self._factory([]), cancel_event=cancel)
        assert ei.value.code is ErrorCode.CANCELLED

    def test_the_default_sweep_is_the_two_devices_everyone_means(self):
        assert DEFAULT_SWEEP_DEVICES == ("cpu", "cuda")

    def test_total_ms_is_load_plus_probe_and_travels_in_the_dict(self):
        report = WarmupReport(outcomes=(WarmupOutcome(engine="detection", ok=True,
                                                      state="ready", device="cpu",
                                                      requested_device="cpu",
                                                      load_ms=400, probe_ms=90),))
        row = report.outcomes[0]
        assert row.total_ms == 490
        assert row.to_dict()["total_ms"] == 490
        assert row.to_dict()["requested_device"] == "cpu"
        assert report.summary().endswith("490 ms")
