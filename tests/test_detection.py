"""Part 2 tests — the real DetectionEngine, with the two lazy seams mocked.

Nothing here touches the network: `_import_yolo_model` and `_hf_download` are the
only places the wrapper talks to the outside world, and both are monkeypatched.
Real (network + gated weights) coverage lives in tests/test_models.py (`-m models`).
"""
from __future__ import annotations

import os
import sys
import threading
import types

import pytest
from PIL import Image

from animetext_studio.core.config import KNOWN_MODELS
from animetext_studio.core.types import (
    DETECTION_REPO, ErrorCode, StudioError, TEXT_LABEL,
)
from animetext_studio.engines import detection as det
from animetext_studio.engines.base import EngineState

W, H = 640, 960


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------

class GatedRepoError(Exception):
    """Mimics huggingface_hub.errors.GatedRepoError (matched by type name)."""


class EntryNotFoundError(Exception):
    pass


class LocalEntryNotFoundError(Exception):
    pass


class Response:
    def __init__(self, status_code):
        self.status_code = status_code


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.response = Response(status_code)


class FakeSession:
    """Stand-in for an onnxruntime InferenceSession (only get_providers is used)."""

    def __init__(self, providers):
        self._providers = list(providers)

    def get_providers(self):
        if self._providers == ["boom"]:
            raise RuntimeError("providers unavailable")
        return list(self._providers)


class OfflineSentinel:
    """Stands in for imgutils' private `_OFFLINE` SingletonMark (not a list)."""


class FakeYOLO:
    """Records every call; class attributes configure the behaviour."""

    names = ["yolo12n_animetext", "yolo12s_animetext", "yolo12m_animetext",
             "yolo12l_animetext", "yolo12x_animetext"]
    names_error: Exception | None = None
    predict_error: Exception | None = None
    results: list = []
    thresholds: dict = {}          # repo threshold.json values
    threshold_error: Exception | None = None
    session_providers: list = ["CPUExecutionProvider"]

    instances: list = []

    def __init__(self, repo_id, hf_token=None):
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.opened: list[str] = []
        self.calls: list[dict] = []
        self.cleared = 0
        FakeYOLO.instances.append(self)

    @property
    def model_names(self):
        if FakeYOLO.names_error is not None:
            raise FakeYOLO.names_error
        return FakeYOLO.names

    def _open_model(self, model_name):
        self.opened.append(model_name)
        return (FakeSession(FakeYOLO.session_providers), 640, [TEXT_LABEL], threading.Lock())

    def _get_default_thresholds(self, model_name):
        if FakeYOLO.threshold_error is not None:
            raise FakeYOLO.threshold_error
        return FakeYOLO.thresholds.get(model_name)

    def predict(self, image, model_name, conf_threshold=None, iou_threshold=0.7,
                allow_dynamic=False):
        self.calls.append({"image": image, "model_name": model_name,
                           "conf_threshold": conf_threshold, "iou_threshold": iou_threshold,
                           "allow_dynamic": allow_dynamic})
        if FakeYOLO.predict_error is not None:
            raise FakeYOLO.predict_error
        return list(FakeYOLO.results)

    def clear(self):
        self.cleared += 1


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Engine constructors touch ONNX_MODE / HF_HUB_CACHE; never leak into other tests."""
    for key in ("ONNX_MODE", "HF_HUB_CACHE", "HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    FakeYOLO.names = ["yolo12n_animetext", "yolo12s_animetext", "yolo12m_animetext",
                      "yolo12l_animetext", "yolo12x_animetext"]
    FakeYOLO.names_error = None
    FakeYOLO.predict_error = None
    FakeYOLO.results = []
    FakeYOLO.thresholds = {}
    FakeYOLO.threshold_error = None
    FakeYOLO.session_providers = ["CPUExecutionProvider"]
    FakeYOLO.instances = []
    yield


@pytest.fixture
def seams(monkeypatch):
    """Install the fakes; expose the download log and the last constructed YOLOModel."""
    downloads: list[dict] = []

    def fake_download(repo_id, filename, token=None, tqdm_class=None):
        if os.environ.get("DEMO_DOWNLOAD_FAILS") == "gated":
            raise GatedRepoError(f"Cannot access gated repo for {repo_id}/{filename}")
        downloads.append({"repo_id": repo_id, "filename": filename, "token": token,
                          "tqdm_class": tqdm_class})
        return f"/cache/{filename}"

    monkeypatch.setattr(det, "_import_yolo_model", lambda: FakeYOLO)
    monkeypatch.setattr(det, "_hf_download", fake_download)
    return types.SimpleNamespace(downloads=downloads, engine_yolo=lambda: FakeYOLO.instances[-1])


@pytest.fixture
def engine(seams):
    return det.DetectionEngine()


@pytest.fixture
def image():
    return Image.new("RGB", (W, H), "white")


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------

class TestConfigureHelpers:
    def test_provider_cpu_sets_onnx_mode(self, monkeypatch):
        monkeypatch.setenv("ONNX_MODE", "cuda")
        det.configure_onnx_provider("cpu")
        assert os.environ["ONNX_MODE"] == "cpu"

    def test_provider_auto_clears_onnx_mode(self, monkeypatch):
        monkeypatch.setenv("ONNX_MODE", "cpu")
        det.configure_onnx_provider("auto")
        assert "ONNX_MODE" not in os.environ
        det.configure_onnx_provider(None)
        assert "ONNX_MODE" not in os.environ

    def test_provider_unknown_rejected(self):
        with pytest.raises(StudioError) as ei:
            det.configure_onnx_provider("tpu")
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert "cpu" in ei.value.details["choices"]

    def test_cache_dir_sets_env_and_patches_imported_constant(self, tmp_path, monkeypatch):
        fake = types.ModuleType("huggingface_hub.constants")
        fake.HF_HUB_CACHE = "/old/cache"
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake)
        target = tmp_path / "hf cache"
        assert det.configure_hf_cache_dir(str(target)) == str(target)
        assert os.environ["HF_HUB_CACHE"] == str(target)
        assert fake.HF_HUB_CACHE == str(target)   # hub reads this constant at call time

    def test_cache_dir_expands_user_and_relative(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert det.configure_hf_cache_dir("relative-cache") == str(tmp_path / "relative-cache")
        assert det.configure_hf_cache_dir(None) is None   # no-op

    def test_engine_constructor_applies_both(self, seams, monkeypatch, tmp_path):
        fake = types.ModuleType("huggingface_hub.constants")
        fake.HF_HUB_CACHE = "/old"
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake)
        eng = det.DetectionEngine(provider="cpu", hf_cache_dir=str(tmp_path / "c"))
        assert os.environ["ONNX_MODE"] == "cpu"
        assert fake.HF_HUB_CACHE == str(tmp_path / "c")
        assert eng.status().device == "cpu"
        with pytest.raises(StudioError):
            det.DetectionEngine(provider="quantum")


class TestImportSeam:
    def test_missing_backend_maps_to_engine_load_failed(self, seams, monkeypatch):
        def boom():
            raise ImportError("No module named 'imgutils'")
        monkeypatch.setattr(det, "_import_yolo_model", boom)
        eng = det.DetectionEngine()
        with pytest.raises(StudioError) as ei:
            eng.load("yolo12n_animetext")
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED
        assert "pip install" in ei.value.details["hint"]

    def test_real_import_returns_class_or_structured_error(self):
        """With dghs-imgutils installed -> the class; without it -> ENGINE_LOAD_FAILED.

        Never a bare ImportError, so the GUI/MCP can show an actionable message.
        """
        try:
            assert det._import_yolo_model().__name__ == "YOLOModel"
        except StudioError as exc:
            assert exc.code is ErrorCode.ENGINE_LOAD_FAILED


# ---------------------------------------------------------------------------
# model listing
# ---------------------------------------------------------------------------

class TestListModels:
    def test_repo_list_with_presets(self, engine):
        models = engine.list_models()
        assert [m["name"] for m in models] == list(KNOWN_MODELS)
        assert models[0]["recommended_score_threshold"] == 0.251
        assert engine.status().detail == ""

    def test_unknown_repo_models_are_kept_and_sorted_last(self, seams, engine):
        FakeYOLO.names = ["my_custom_yolo", "yolo12m_animetext", "another_yolo"]
        names = [m["name"] for m in engine.list_models()]
        assert names == ["yolo12m_animetext", "another_yolo", "my_custom_yolo"]

    def test_auth_failure_falls_back_to_known_models(self, seams, engine):
        FakeYOLO.names_error = GatedRepoError("401 cannot access gated repo")
        models = engine.list_models()
        assert [m["name"] for m in models] == list(KNOWN_MODELS)
        assert "GatedRepoError" in engine.status().detail

    def test_offline_sentinel_falls_back(self, seams, engine):
        FakeYOLO.names = OfflineSentinel()   # not a list -> fallback
        assert [m["name"] for m in engine.list_models()] == list(KNOWN_MODELS)
        assert "fallback" in engine.status().detail

    def test_repo_list_is_cached(self, engine, seams, image):
        FakeYOLO.results = [((10, 10, 20, 20), TEXT_LABEL, 0.5)]
        for _ in range(3):
            engine.detect(image, "yolo12m_animetext")
        assert len(FakeYOLO.instances) == 1          # single YOLOModel instance
        assert len([d for d in seams.downloads if d["filename"].endswith("model.onnx")]) == 1

    def test_token_is_passed_to_the_underlying_class(self, seams, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        det.DetectionEngine()._ensure_yolo()
        assert seams.engine_yolo().hf_token is None   # env fallback happens inside imgutils
        det.DetectionEngine(hf_token="explicit")._ensure_yolo()
        assert seams.engine_yolo().hf_token == "explicit"


# ---------------------------------------------------------------------------
# load / download
# ---------------------------------------------------------------------------

class TestLoad:
    def test_downloads_weights_and_metadata_then_opens_session(self, engine, seams):
        events: list[tuple] = []
        engine.load("yolo12n_animetext", progress=lambda *a: events.append(a))
        files = [d["filename"] for d in seams.downloads]
        assert files == ["yolo12n_animetext/model.onnx",
                         "yolo12n_animetext/model_type.json",
                         "yolo12n_animetext/threshold.json"]
        assert all(d["repo_id"] == DETECTION_REPO for d in seams.downloads)
        assert seams.engine_yolo().opened == ["yolo12n_animetext"]
        status = engine.status()
        assert status.state is EngineState.READY and status.loaded_models == ("yolo12n_animetext",)
        # aux metadata downloads are silent; the ONNX download reports 0%, then the
        # session open (90%) and the completion (100%).
        assert [e[0] for e in events] == ["download", "load_session", "load_model"]
        assert events[-1][1] == 100.0

    def test_download_reports_progress_percent(self, seams, monkeypatch):
        from tqdm import tqdm

        def fake_download(repo_id, filename, token=None, tqdm_class=None):
            if tqdm_class is not None:
                bar = tqdm_class(total=100)
                bar.update(50)
                bar.update(50)
            return "/cache/x"

        monkeypatch.setattr(det, "_hf_download", fake_download)
        events: list[tuple] = []
        det.DetectionEngine().load("yolo12n_animetext", progress=lambda *a: events.append(a))
        percents = [e[1] for e in events if e[0] == "download"]
        assert 50.0 in percents and max(percents) < 100.0   # 100% comes from load_model

    def test_load_is_idempotent(self, engine, seams):
        engine.load("yolo12m_animetext")
        engine.load("yolo12m_animetext")
        assert len([d for d in seams.downloads if d["filename"].endswith("model.onnx")]) == 1

    def test_optional_metadata_failure_is_tolerated(self, seams, monkeypatch):
        def flaky(repo_id, filename, token=None, tqdm_class=None):
            if filename.endswith(("model_type.json", "threshold.json")):
                raise RuntimeError("metadata unavailable")
            return "/cache/x"
        monkeypatch.setattr(det, "_hf_download", flaky)
        engine = det.DetectionEngine()
        engine.load("yolo12s_animetext")
        assert engine.status().state is EngineState.READY

    @pytest.mark.parametrize("exc,expected", [
        (GatedRepoError("gated"), ErrorCode.HF_AUTH_REQUIRED),
        (HttpError(401), ErrorCode.HF_AUTH_REQUIRED),
        (HttpError(403), ErrorCode.HF_AUTH_REQUIRED),
        (EntryNotFoundError("no file"), ErrorCode.MODEL_NOT_FOUND),
        (HttpError(404), ErrorCode.MODEL_NOT_FOUND),
        (LocalEntryNotFoundError("no net"), ErrorCode.DOWNLOAD_FAILED),
        (ConnectionError("dns"), ErrorCode.DOWNLOAD_FAILED),
        (TimeoutError("slow"), ErrorCode.DOWNLOAD_FAILED),
        (ValueError("weird"), ErrorCode.ENGINE_LOAD_FAILED),
    ])
    def test_download_errors_are_classified(self, seams, monkeypatch, exc, expected):
        def raiser(repo_id, filename, token=None, tqdm_class=None):
            raise exc
        monkeypatch.setattr(det, "_hf_download", raiser)
        engine = det.DetectionEngine()
        with pytest.raises(StudioError) as ei:
            engine.load("yolo12n_animetext")
        assert ei.value.code is expected
        assert engine.status().state is EngineState.ERROR

    def test_auth_error_carries_actionable_steps(self, seams, monkeypatch):
        def raiser(repo_id, filename, token=None, tqdm_class=None):
            raise GatedRepoError("401 Cannot access gated repo")
        monkeypatch.setattr(det, "_hf_download", raiser)
        with pytest.raises(StudioError) as ei:
            det.DetectionEngine().load("yolo12n_animetext")
        steps = ei.value.details["steps"]
        assert any("accept the access terms" in s for s in steps)
        assert any("HF_TOKEN" in s for s in steps)
        assert det.GATED_REPO_URL in steps[0]

    def test_session_open_failure_is_classified(self, seams, monkeypatch):
        def bad_open(self, model_name):
            raise GatedRepoError("gated")
        monkeypatch.setattr(FakeYOLO, "_open_model", bad_open)
        with pytest.raises(StudioError) as ei:
            det.DetectionEngine().load("yolo12n_animetext")
        assert ei.value.code is ErrorCode.HF_AUTH_REQUIRED

    def test_token_is_forwarded_to_downloads(self, seams, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "tok-from-env")
        det.DetectionEngine().load("yolo12n_animetext")
        assert all(d["token"] == "tok-from-env" for d in seams.downloads)


# ---------------------------------------------------------------------------
# device reporting (GUI / MCP get_status must show what really runs)
# ---------------------------------------------------------------------------

class TestDeviceReporting:
    def test_device_is_the_configured_request_before_load(self, seams):
        assert det.DetectionEngine().status().device == "auto"
        assert det.DetectionEngine(provider="cpu").status().device == "cpu"

    def test_device_resolves_to_the_real_provider_after_load(self, seams):
        engine = det.DetectionEngine()                 # provider "auto"
        engine.load("yolo12n_animetext")
        assert engine.status().device == "cpu"         # session got CPUExecutionProvider

    def test_cuda_session_is_reported_as_cuda(self, seams):
        FakeYOLO.session_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        engine = det.DetectionEngine()
        engine.load("yolo12n_animetext")
        assert engine.status().device == "cuda"

    def test_unknown_provider_name_is_passed_through(self, seams):
        FakeYOLO.session_providers = ["SomethingCustomExecutionProvider"]
        engine = det.DetectionEngine()
        engine.load("yolo12n_animetext")
        assert engine.status().device == "SomethingCustomExecutionProvider"

    def test_provider_lookup_failures_never_break_loading(self, seams):
        FakeYOLO.session_providers = ["boom"]
        engine = det.DetectionEngine()
        engine.load("yolo12n_animetext")
        assert engine.status().device == "auto"        # unchanged, load still succeeded
        assert engine.status().state is EngineState.READY


# ---------------------------------------------------------------------------
# score-threshold resolution (recorded verbatim in the result JSON by P4)
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_repo_threshold_wins_when_backend_is_live(self, engine, seams):
        FakeYOLO.thresholds = {"yolo12m_animetext": 0.31}
        assert engine.recommended_threshold("yolo12m_animetext") == 0.299   # cold: preset
        engine._ensure_yolo()                                               # no network needed
        assert engine.recommended_threshold("yolo12m_animetext") == 0.31    # live: threshold.json
        assert engine.resolved_score_threshold("yolo12m_animetext", None) == 0.31

    def test_documented_preset_when_backend_is_cold(self, seams):
        assert det.DetectionEngine().recommended_threshold("yolo12m_animetext") == 0.299

    def test_explicit_value_always_wins(self, engine, seams):
        FakeYOLO.thresholds = {"yolo12m_animetext": 0.31}
        assert engine.resolved_score_threshold("yolo12m_animetext", 0.05) == 0.05

    def test_imgutils_fallback_for_models_without_threshold_file(self, engine, seams):
        engine._ensure_yolo()
        assert engine.recommended_threshold("some_custom_model") is None
        assert engine.resolved_score_threshold("some_custom_model", None) == 0.25

    def test_threshold_reader_failures_are_swallowed(self, engine, seams):
        engine._ensure_yolo()
        FakeYOLO.threshold_error = RuntimeError("metadata unavailable")
        assert engine.recommended_threshold("yolo12m_animetext") == 0.299  # preset fallback

    def test_detect_uses_repo_threshold_when_none_given(self, engine, seams, image):
        FakeYOLO.thresholds = {"yolo12m_animetext": 0.31}
        engine.detect(image, "yolo12m_animetext", score_threshold=None)
        assert seams.engine_yolo().calls[-1]["conf_threshold"] is None  # imgutils resolves it
        assert engine.resolved_score_threshold("yolo12m_animetext", None) == 0.31
        assert engine.resolved_score_threshold("yolo12m_animetext", 0.4) == 0.4


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

class TestDetect:
    def test_parameters_are_passed_through_verbatim(self, engine, seams, image):
        FakeYOLO.results = [((1, 2, 3, 4), TEXT_LABEL, 0.5)]
        engine.detect(image, "yolo12l_animetext", iou_threshold=0.42, score_threshold=0.11,
                      allow_dynamic=True)
        call = seams.engine_yolo().calls[-1]
        assert call["model_name"] == "yolo12l_animetext"
        assert call["iou_threshold"] == 0.42
        assert call["conf_threshold"] == 0.11
        assert call["allow_dynamic"] is True

    def test_score_none_delegates_to_repo_threshold(self, engine, seams, image):
        engine.detect(image, "yolo12m_animetext", score_threshold=None)
        assert seams.engine_yolo().calls[-1]["conf_threshold"] is None

    def test_results_are_clamped_filtered_and_sorted(self, engine, seams, image):
        FakeYOLO.results = [
            ((-40, 10, 60, 40), TEXT_LABEL, 0.42),      # clamped to x0=0
            ((10, 10, 9999, 50), TEXT_LABEL, 0.99),     # clamped to W
            ((5, 5, 5, 90), TEXT_LABEL, 0.9),           # zero width -> dropped
            ((30, 30, 40, 40), TEXT_LABEL, 0.7),
            ((50, 50, 60, 60), "other_label", 0.6),
        ]
        dets = engine.detect(image, "yolo12m_animetext")
        assert [d.score for d in dets] == [0.99, 0.7, 0.6, 0.42]
        assert dets[0].bbox.x1 == W and dets[-1].bbox.x0 == 0
        assert all(d.bbox.is_valid() for d in dets)
        assert {d.label for d in dets} == {TEXT_LABEL, "other_label"}

    def test_lazy_load_happens_inside_detect(self, engine, seams, image):
        assert engine.status().state is EngineState.UNLOADED
        engine.detect(image, "yolo12x_animetext")
        assert engine.status().loaded_models == ("yolo12x_animetext",)
        assert seams.engine_yolo().opened == ["yolo12x_animetext"]

    def test_progress_events(self, engine, seams, image):
        events: list[tuple] = []
        engine.detect(image, "yolo12m_animetext", progress=lambda *a: events.append(a))
        stages = [e[0] for e in events]
        assert stages[0] == "download" and stages[-1] == "detect"
        assert events[-1][1] == 100.0 and "text block" in events[-1][2]

    def test_invalid_arguments(self, engine, seams, image):
        with pytest.raises(StudioError) as ei:
            engine.detect("nope", "yolo12m_animetext")  # type: ignore[arg-type]
        assert ei.value.code is ErrorCode.INVALID_IMAGE
        with pytest.raises(StudioError) as ei:
            engine.detect(image, "yolo12m_animetext", iou_threshold=1.5)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        with pytest.raises(StudioError) as ei:
            engine.detect(image, "yolo12m_animetext", score_threshold=-0.1)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        with pytest.raises(StudioError) as ei:
            engine.detect(image, "not_in_repo")
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND
        assert "available" in ei.value.details

    def test_cancellation_before_inference(self, engine, seams, image):
        ev = threading.Event()
        ev.set()
        with pytest.raises(StudioError) as ei:
            engine.detect(image, "yolo12m_animetext", cancel_event=ev)
        assert ei.value.code is ErrorCode.CANCELLED
        assert all(inst.calls == [] for inst in FakeYOLO.instances)

    def test_predict_errors_are_classified(self, engine, seams, image):
        FakeYOLO.predict_error = RuntimeError("onnx session died")
        with pytest.raises(StudioError) as ei:
            engine.detect(image, "yolo12m_animetext")
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED
        assert "RuntimeError" in ei.value.details["reason"]

    def test_concurrent_detect_is_serialised(self, seams, monkeypatch, image):
        """The engine lock must keep two workers out of the shared ONNX session."""
        active = {"n": 0, "max": 0}
        guard = threading.Lock()

        class CountingYOLO(FakeYOLO):
            def predict(self, *args, **kwargs):
                with guard:
                    active["n"] += 1
                    active["max"] = max(active["max"], active["n"])
                try:
                    threading.Event().wait(0.02)   # hold the "session" for a moment
                    return [((1, 1, 20, 20), TEXT_LABEL, 0.5)]
                finally:
                    with guard:
                        active["n"] -= 1

        monkeypatch.setattr(det, "_import_yolo_model", lambda: CountingYOLO)
        eng = det.DetectionEngine()
        results: list = []
        errors: list = []

        def work():
            try:
                results.append(eng.detect(image, "yolo12m_animetext"))
            except Exception as exc:  # pragma: no cover - debugging aid
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == []
        assert len(results) == 4
        assert active["max"] == 1


class TestUnload:
    def test_unload_clears_sessions_and_forces_refetch(self, engine, seams, image):
        engine.detect(image, "yolo12m_animetext")
        yolo = seams.engine_yolo()
        engine.unload()
        assert yolo.cleared == 1
        assert engine.status().state is EngineState.UNLOADED
        assert engine.status().loaded_models == ()
        engine.detect(image, "yolo12m_animetext")
        assert len([d for d in seams.downloads if d["filename"].endswith("model.onnx")]) == 2
