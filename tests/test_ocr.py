"""Part 3 tests — the OCR engine contract (Hayai's real API shape, no torch needed).

Everything here runs offline and without PyTorch installed. The two lazy import seams in
`engines/ocr.py` (`_import_torch`, `_import_transformers`) are the only things a real
install touches, so stubbing them exercises load / chunking / cancellation / error mapping
exactly the way the pipeline (P4), the GUI (P5) and the MCP server (P7) will drive them.
Real weights are covered by `-m models` (the last test here + `tests/test_models.py`).

Contractual points:
* one string per crop, in input order, `""` for "nothing recognised";
* the **engine** owns the chunking — `batch_size` is never forwarded to `model.ocr`, and
  `escalate=False` always (single pass, no silent retries);
* progress per chunk, plus a cancellation point before *and* after every batch;
* every torch/transformers failure lands in the project's error taxonomy;
* weights resolve bundled-first (D26); offline never reaches for the network.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest
from PIL import Image

from animetext_studio.core.model_store import OCR_REQUIRED_FILES, ModelStore
from animetext_studio.core.types import BBox, OCR_REPO, ErrorCode, StudioError
from animetext_studio.engines import ocr as ocr_mod
from animetext_studio.engines.base import EngineState, FakeOcrEngine
from animetext_studio.engines.ocr import (
    DTYPE_CHOICES, OcrEngine, configure_hf_offline, configure_ocr_compile,
)


# ---------------------------------------------------------------------------
# Stubs: torch / transformers / the Hayai model itself
# ---------------------------------------------------------------------------

class _CudaStub:
    def __init__(self, count: int) -> None:
        self.count = count
        self.calls: list[str] = []

    def device_count(self) -> int:
        return self.count

    def is_available(self) -> bool:
        return self.count > 0

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")

    def ipc_collect(self) -> None:
        self.calls.append("ipc_collect")


class TorchStub:
    """Only the surface `OcrEngine` touches. dtype sentinels are compared by identity."""

    float32, float16, bfloat16 = "torch.float32", "torch.float16", "torch.bfloat16"

    def __init__(self, cuda_count: int = 0) -> None:
        self.cuda = _CudaStub(cuda_count)


class ModelStub:
    """Stands in for the Hayai model; records every `ocr()` call it receives."""

    def __init__(self, responder=None) -> None:
        self.calls: list[tuple[int, dict]] = []
        self.devices: list[str] = []
        self.evalled = False
        self.responder = responder

    def to(self, device: str) -> None:
        self.devices.append(device)

    def eval(self) -> None:
        self.evalled = True

    def ocr(self, images, processor, **kwargs):
        self.calls.append((len(images), kwargs))
        if self.responder is not None:
            return self.responder(images, kwargs)
        return [f"t{len(self.calls)}-{i}" for i in range(len(images))]


class TransformersStub:
    def __init__(self, model=None, processor=None, model_errors=(), load_delay: float = 0.0):
        self.model = model if model is not None else ModelStub()
        self.processor = processor if processor is not None else object()
        self.model_errors = list(model_errors)
        self.load_delay = load_delay
        self.model_sources: list[str] = []
        self.model_kwargs: list[dict] = []
        self.processor_sources: list[str] = []

    @property
    def load_calls(self) -> int:
        return len(self.model_sources)


def install(monkeypatch, stub: TransformersStub, torch: TorchStub | None = None) -> TransformersStub:
    """Patch the two lazy import seams. This is the only mocking the engine needs."""

    class AutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            stub.model_sources.append(str(source))
            stub.model_kwargs.append(kwargs)
            if stub.load_delay:
                time.sleep(stub.load_delay)
            if stub.model_errors:
                error = stub.model_errors.pop(0)
                if error is not None:
                    raise error
            return stub.model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            stub.processor_sources.append(str(source))
            return stub.processor

    monkeypatch.setattr(ocr_mod, "_import_transformers", lambda: (AutoModel, AutoProcessor))
    monkeypatch.setattr(ocr_mod, "_import_torch", lambda: torch or TorchStub(0))
    return stub


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def crop(seed: int = 0, size: tuple[int, int] = (48, 24)) -> Image.Image:
    """A deterministic non-blank crop; different seeds give visibly different pixels."""
    image = Image.new("RGB", size)
    pixels = image.load()
    for x in range(size[0]):
        for y in range(size[1]):
            pixels[x, y] = ((x * 7 + seed * 53) % 256, (y * 11 + seed) % 256, (seed * 31) % 256)
    return image


def blank(size: tuple[int, int] = (16, 16)) -> Image.Image:
    return Image.new("RGB", size, (0, 0, 0))


def recorder() -> tuple[list, callable]:
    seen: list[tuple[str, float, str]] = []
    return seen, (lambda stage, percent, message: seen.append((stage, percent, message)))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The engine writes HAYAI_DISABLE_COMPILE / HF_HUB_OFFLINE into the process env."""
    for var in ("HAYAI_DISABLE_COMPILE", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def bare(tmp_path) -> OcrEngine:
    """An engine whose models dir does not exist: nothing is bundled."""
    return OcrEngine(models_dir=str(tmp_path / "no-bundle"))


@pytest.fixture
def bundle(tmp_path):
    """A directory that satisfies `has_ocr_snapshot()` (the store only needs 1 byte/file)."""
    root = tmp_path / "bundle"
    snap = root / "hayai-ocr-v2.7-supernova"
    snap.mkdir(parents=True)
    for name in OCR_REQUIRED_FILES:
        (snap / name).write_text("stub", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Construction / pre-flight (no torch, no I/O)
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_construction_touches_nothing(self, bare):
        assert bare.status().state is EngineState.UNLOADED
        assert bare.status().backend == "torch"
        assert bare.weights_source is None

    @pytest.mark.parametrize("device", ["mps", "gpu", "CUDA0", "cuda-x"])
    def test_device_strings_are_validated(self, tmp_path, device):
        with pytest.raises(StudioError) as ei:
            OcrEngine(device=device, models_dir=str(tmp_path))
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert device in ei.value.message

    def test_dtype_strings_are_validated(self, tmp_path):
        with pytest.raises(StudioError) as ei:
            OcrEngine(dtype="fp8", models_dir=str(tmp_path))
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert set(DTYPE_CHOICES) <= set(ei.value.details["choices"])

    def test_plan_reports_the_missing_bundle_file_by_file(self, bare):
        plan = bare.plan()
        assert plan["bundled"] is False
        assert plan["snapshot_dir"] is None
        assert plan["missing_files"] == list(OCR_REQUIRED_FILES)
        assert plan["repo_id"] == OCR_REPO

    def test_plan_sees_the_bundle_and_still_does_no_io(self, bundle):
        engine = OcrEngine(models_dir=str(bundle))
        plan = engine.plan()
        assert plan["bundled"] is True
        assert plan["missing_files"] == []
        assert plan["snapshot_dir"].endswith("hayai-ocr-v2.7-supernova")
        assert engine.status().state is EngineState.UNLOADED

    def test_plan_states_the_defaults_the_card_documents(self, bare):
        plan = bare.plan()
        assert plan["dtype"] == "auto"
        assert plan["use_cuda_graph"] is True      # default *intent*; gated on CUDA at load
        assert plan["offline"] is False


# ---------------------------------------------------------------------------
# load(): resolution, dtype/device policy, import-time traps
# ---------------------------------------------------------------------------

class TestLoad:
    def test_bundled_snapshot_is_used_verbatim_and_offline_clean(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        snapshot = str(ModelStore(str(bundle)).ocr_dir())
        assert stub.processor_sources == [snapshot]
        assert stub.model_sources == [snapshot]
        assert engine.weights_source == snapshot
        assert "bundle" in engine.status().detail

    def test_trust_remote_code_is_passed_to_both_halves(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        OcrEngine(models_dir=str(bundle)).load()
        assert stub.model_kwargs[0].get("trust_remote_code") is True
        assert len(stub.processor_sources) == 1        # processor loaded once, not per dtype

    def test_model_is_moved_and_set_to_eval_mode(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle), device="cpu")
        engine.load()
        assert stub.model.devices == ["cpu"]
        assert stub.model.evalled is True
        assert engine.device == "cpu"

    def test_hub_repo_is_the_source_when_nothing_is_bundled(self, monkeypatch, bare):
        stub = install(monkeypatch, TransformersStub())
        bare.load()
        assert stub.model_sources == [OCR_REPO]
        assert "hub" in bare.status().detail

    def test_load_is_idempotent(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        engine.load()
        assert stub.load_calls == 1

    def test_load_reports_progress_and_ends_ready(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub())
        seen, cb = recorder()
        engine = OcrEngine(models_dir=str(bundle))
        engine.load(progress=cb)
        assert [stage for stage, _, _ in seen] and all(s == "load_ocr" for s, _, _ in seen)
        assert seen[-1][1] == 100.0
        assert engine.status().state is EngineState.READY
        assert engine.status().loaded_models == (OCR_REPO,)

    def test_compile_is_disabled_on_cpu_before_the_remote_import(self, monkeypatch, bundle):
        """HAYAI_DISABLE_COMPILE is read at *import* time, so it must be set beforehand."""
        calls: list[str] = []
        stub = install(monkeypatch, TransformersStub())
        original = ocr_mod._import_transformers

        def spy():
            calls.append(f"env={__import__('os').environ.get('HAYAI_DISABLE_COMPILE')}")
            return original()

        monkeypatch.setattr(ocr_mod, "_import_transformers", spy)
        OcrEngine(models_dir=str(bundle), device="cpu").load()
        assert calls == ["env=1"]

    def test_compile_is_left_alone_on_cuda(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(), torch=TorchStub(1))
        OcrEngine(models_dir=str(bundle), device="auto").load()
        import os
        assert "HAYAI_DISABLE_COMPILE" not in os.environ

    def test_an_explicit_compile_choice_is_respected(self, monkeypatch, bundle):
        monkeypatch.setenv("HAYAI_DISABLE_COMPILE", "0")
        assert configure_ocr_compile("cpu") is None
        import os
        assert os.environ["HAYAI_DISABLE_COMPILE"] == "0"

    def test_hf_offline_only_ever_disables_the_network(self, monkeypatch):
        configure_hf_offline(False)
        assert "HF_HUB_OFFLINE" not in __import__("os").environ
        configure_hf_offline(True)
        assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"

    def test_hf_offline_reaches_an_already_imported_hub_module(self, monkeypatch):
        hub = types.ModuleType("huggingface_hub.constants")
        hub.HF_HUB_OFFLINE = False
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hub)
        configure_hf_offline(True)
        assert hub.HF_HUB_OFFLINE is True

    def test_transformers_4x_fallback_when_dtype_is_renamed(self, monkeypatch, bundle):
        """`torch_dtype` -> `dtype` was renamed in transformers v5; both must work."""
        stub = install(monkeypatch, TransformersStub(
            model_errors=[TypeError("AutoModel.from_pretrained() got an unexpected keyword 'dtype'")]))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        assert stub.load_calls == 2
        assert "torch_dtype" in stub.model_kwargs[1]
        assert engine.status().state is EngineState.READY


class TestDeviceResolution:
    def test_auto_picks_cuda_when_visible(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(), torch=TorchStub(2))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        assert engine.device == "cuda:0"
        assert "cuda-graphs on" in engine.status().detail

    def test_auto_falls_back_to_cpu_and_says_so(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(), torch=TorchStub(0))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        assert engine.device == "cpu"
        assert "cuda-graphs off" in engine.status().detail

    def test_cpu_is_honoured_even_with_a_gpu_present(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(), torch=TorchStub(2))
        engine = OcrEngine(models_dir=str(bundle), device="cpu")
        engine.load()
        assert engine.device == "cpu"

    def test_a_missing_cuda_index_is_refused_not_downgraded(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(), torch=TorchStub(1))
        engine = OcrEngine(models_dir=str(bundle), device="cuda:3")
        with pytest.raises(StudioError) as ei:
            engine.load()
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED
        assert "3" in ei.value.message
        assert ei.value.details["cuda_visible_count"] == 1
        assert engine.status().state is EngineState.ERROR

    def test_broken_driver_reports_as_no_cuda(self, monkeypatch, bundle):
        class ExplodingTorch(TorchStub):
            pass

        torch = ExplodingTorch(0)
        torch.cuda.device_count = lambda: (_ for _ in ()).throw(RuntimeError("no NVML"))
        install(monkeypatch, TransformersStub(), torch=torch)
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        assert engine.device == "cpu"

    def test_cuda_graphs_can_be_forced_off_on_cuda(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub(), torch=TorchStub(1))
        engine = OcrEngine(models_dir=str(bundle), use_cuda_graph=False)
        engine.load()
        engine.ocr_crops([crop()])
        assert stub.model.calls[0][1]["use_cuda_graph"] is False
        assert "cuda-graphs off" in engine.status().detail


class TestLoadFailures:
    def test_offline_without_a_bundle_fails_before_touching_transformers(self, monkeypatch, bare):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bare.model_store.root), offline=True)
        with pytest.raises(StudioError) as ei:
            engine.load()
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND
        assert stub.load_calls == 0                      # no network attempt, at all
        assert any("fetch_models.py" in step for step in ei.value.details["steps"])
        assert engine.status().state is EngineState.ERROR

    def test_a_broken_backend_import_is_an_engine_load_failure(self, monkeypatch, bare):
        monkeypatch.setattr(ocr_mod, "_import_torch",
                            lambda: (_ for _ in ()).throw(ImportError("no torch")))
        with pytest.raises(StudioError) as ei:
            bare.load()
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED
        assert "[ocr]" in ei.value.details["hint"]

    def test_offline_mode_is_reported_as_a_download_problem(self, monkeypatch, tmp_path):
        class OfflineModeIsEnabled(Exception):
            pass
        OfflineModeIsEnabled.__name__ = "OfflineModeIsEnabled"
        stub = install(monkeypatch, TransformersStub(model_errors=[OfflineModeIsEnabled("offline")]))
        engine = OcrEngine(models_dir=str(tmp_path / "none"))
        with pytest.raises(StudioError) as ei:
            engine.load()
        assert ei.value.code is ErrorCode.DOWNLOAD_FAILED
        assert stub.load_calls == 1


# ---------------------------------------------------------------------------
# ocr_crops(): alignment, chunking, cancellation, arguments
# ---------------------------------------------------------------------------

class TestOcrCrops:
    def test_one_text_per_crop_in_order(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        crops = [crop(i) for i in range(5)]
        assert engine.ocr_crops(crops, batch_size=2) == ["t1-0", "t1-1", "t2-0", "t2-1", "t3-0"]

    def test_the_engine_owns_the_chunking(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        engine.ocr_crops([crop(i) for i in range(5)], batch_size=2)
        assert [n for n, _ in stub.model.calls] == [2, 2, 1]

    def test_batch_size_is_never_forwarded_and_escalate_is_always_off(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        engine.ocr_crops([crop()], batch_size=4, max_new_tokens=33, max_patches=256)
        kwargs = stub.model.calls[0][1]
        assert "batch_size" not in kwargs
        assert kwargs["escalate"] is False
        assert kwargs["max_new_tokens"] == 33
        assert kwargs["max_patches"] == 256

    def test_dtype_is_translated_for_the_model(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle), dtype="fp16")
        engine.load()
        engine.ocr_crops([crop()])
        assert stub.model.calls[0][1]["dtype"] == TorchStub.float16

    def test_auto_dtype_passes_none_and_lets_the_model_choose(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle), dtype="auto")
        engine.load()
        engine.ocr_crops([crop()])
        assert stub.model.calls[0][1]["dtype"] is None

    def test_a_lone_string_is_still_one_result(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub(model=ModelStub(responder=lambda im, kw: "solo")))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        assert engine.ocr_crops([crop()]) == ["solo"]

    def test_non_string_outputs_become_empty_not_crashes(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub(
            model=ModelStub(responder=lambda im, kw: [None, 12, "ok"])))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        assert engine.ocr_crops([crop(1), crop(2), crop(3)], batch_size=3) == ["", "", "ok"]

    def test_a_misaligned_response_is_fatal_not_silently_zipped(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub(
            model=ModelStub(responder=lambda im, kw: ["only one"])))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop(1), crop(2)], batch_size=2)
        assert ei.value.code is ErrorCode.OCR_FAILED
        assert "1 texts for 2 crops" in ei.value.message

    def test_progress_is_per_chunk_and_monotonic(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        seen, cb = recorder()
        engine.ocr_crops([crop(i) for i in range(5)], batch_size=2, progress=cb)
        percents = [p for stage, p, _ in seen if stage == "ocr"]
        assert percents == sorted(percents)
        assert percents[-1] == 100.0
        assert len(percents) == 3

    def test_cancellation_before_the_first_batch_costs_nothing(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        ev = threading.Event()
        ev.set()
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop()], cancel_event=ev)
        assert ei.value.code is ErrorCode.CANCELLED
        assert stub.model.calls == []

    def test_cancellation_between_chunks_stops_the_run(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        ev = threading.Event()

        def progress(stage, percent, message):
            if stage == "ocr":            # the first batch came back; stop right here
                ev.set()

        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop(i) for i in range(6)], batch_size=2, progress=progress,
                             cancel_event=ev)
        assert ei.value.code is ErrorCode.CANCELLED
        assert len(stub.model.calls) == 1               # second batch never started

    def test_empty_input_returns_empty_without_loading(self, monkeypatch, bare):
        stub = install(monkeypatch, TransformersStub())
        assert bare.ocr_crops([]) == []
        assert stub.load_calls == 0 and not bare.is_loaded()

    def test_loading_is_lazy_on_the_first_crop(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        assert engine.status().state is EngineState.UNLOADED
        engine.ocr_crops([crop()])
        assert stub.load_calls == 1 and engine.is_loaded()

    @pytest.mark.parametrize("bad", ["max_patches", "batch_size", "max_new_tokens"])
    def test_argument_validation(self, monkeypatch, bundle, bad):
        install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        kwargs = {"max_patches": "auto", "batch_size": 2, "max_new_tokens": 8}
        kwargs[bad] = {"max_patches": 64, "batch_size": 0, "max_new_tokens": 0}[bad]
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop()], **kwargs)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT

    def test_max_patches_accepts_auto_and_the_documented_budgets(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        for budget in ("auto", 128, 192, 256, 384, 512):
            engine.ocr_crops([crop()], max_patches=budget)
        assert [kw["max_patches"] for _, kw in stub.model.calls] ==             ["auto", 128, 192, 256, 384, 512]

    def test_a_non_pil_crop_is_named_by_index(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop(1), "not-an-image"])
        assert ei.value.code is ErrorCode.INVALID_IMAGE
        assert "1" in ei.value.message

    def test_normalization_is_opt_in(self, monkeypatch, bundle):
        zen = "\uff21\uff23\uff01 \uff34\uff45\uff53\uff54"     # ＡＣ！ Ｔｅｓｔ
        stub = install(monkeypatch, TransformersStub(
            model=ModelStub(responder=lambda im, kw: [zen] * len(im))))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        raw = engine.ocr_crops([crop(1), crop(2)], normalize_nfkc=False)
        tidy = engine.ocr_crops([crop(1), crop(2)], normalize_nfkc=True)
        assert raw[0] == zen
        assert tidy[0] == "AC! Test"
        assert len(tidy) == 2

    def test_line_breaks_survive_normalization(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(
            model=ModelStub(responder=lambda im, kw: ["\uff21\uff22\n\uff23\uff24"])))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        assert engine.ocr_crops([crop()], normalize_nfkc=True) == ["AB\nCD"]


class TestOcrFailureMapping:
    @pytest.mark.parametrize("error,code", [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), ErrorCode.OCR_FAILED),
        (ConnectionError("connection reset"), ErrorCode.DOWNLOAD_FAILED),
        (TimeoutError("read timed out"), ErrorCode.DOWNLOAD_FAILED),
        (ValueError("bad positional args"), ErrorCode.OCR_FAILED),
    ])
    def test_runtime_failures_map_onto_the_taxonomy(self, monkeypatch, bundle, error, code):
        install(monkeypatch, TransformersStub(
            model=ModelStub(responder=lambda im, kw: (_ for _ in ()).throw(error))))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop()])
        assert ei.value.code is code

    def test_oom_advice_names_the_two_knobs(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(
            model=ModelStub(responder=lambda im, kw: (_ for _ in ()).throw(
                RuntimeError("CUDA out of memory")))))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop()])
        hint = ei.value.details["hint"]
        assert "batch_size" in hint and "max_patches" in hint

    def test_the_failing_batch_is_identified(self, monkeypatch, bundle):
        seen = {"n": 0}

        def responder(images, kwargs):
            seen["n"] += 1
            if seen["n"] == 3:
                raise RuntimeError("out of memory")
            return ["ok"] * len(images)

        install(monkeypatch, TransformersStub(model=ModelStub(responder=responder)))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop(i) for i in range(4)], batch_size=1)
        assert "batch 2..3" in ei.value.message

    def test_a_studio_error_from_the_model_passes_through(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(model=ModelStub(
            responder=lambda im, kw: (_ for _ in ()).throw(
                StudioError(ErrorCode.INVALID_IMAGE, "processor said no")))))
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        with pytest.raises(StudioError) as ei:
            engine.ocr_crops([crop()])
        assert ei.value.code is ErrorCode.INVALID_IMAGE

    def test_missing_hub_files_point_at_the_fetch_script(self):
        class LocalEntryNotFoundError(Exception):
            pass
        LocalEntryNotFoundError.__name__ = "LocalEntryNotFoundError"
        err = ocr_mod._classify_ocr(LocalEntryNotFoundError("not there"), "load")
        assert err.code is ErrorCode.MODEL_NOT_FOUND
        assert "fetch_models.py" in err.details["hint"]

    def test_revision_and_repo_errors_are_model_not_found(self):
        for name in ("HFValidationError", "RepositoryNotFoundError", "RevisionNotFoundError"):
            exc = type(name, (Exception,), {})("nope")
            assert ocr_mod._classify_ocr(exc, "load").code is ErrorCode.MODEL_NOT_FOUND


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_clears_state_and_frees_the_cache(self, monkeypatch, bundle):
        torch = TorchStub(1)
        stub = install(monkeypatch, TransformersStub(), torch=torch)
        engine = OcrEngine(models_dir=str(bundle))
        engine.load()
        engine.unload()
        assert engine.status().state is EngineState.UNLOADED
        assert engine.status().loaded_models == ()
        assert engine.weights_source is None
        assert engine._model is None and engine._processor is None
        assert torch.cuda.calls == ["empty_cache", "ipc_collect"]

    def test_unload_on_cpu_does_not_try_the_gpu(self, monkeypatch, bundle):
        torch = TorchStub(0)
        install(monkeypatch, TransformersStub(), torch=torch)
        engine = OcrEngine(models_dir=str(bundle), device="cpu")
        engine.load()
        engine.unload()
        assert torch.cuda.calls == []

    def test_unloading_an_unloaded_engine_is_a_no_op(self, bare):
        bare.unload()
        assert bare.status().state is EngineState.UNLOADED

    def test_a_run_after_unload_reloads(self, monkeypatch, bundle):
        stub = install(monkeypatch, TransformersStub())
        engine = OcrEngine(models_dir=str(bundle))
        engine.ocr_crops([crop()])
        engine.unload()
        assert engine.ocr_crops([crop()])
        assert stub.load_calls == 2

    def test_concurrent_loads_load_once(self, monkeypatch, bundle):
        install(monkeypatch, TransformersStub(load_delay=0.05))
        engine = OcrEngine(models_dir=str(bundle))
        errors: list[BaseException] = []

        def go():
            try:
                engine.load()
            except BaseException as exc:              # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=go) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [] and engine.is_loaded()


# ---------------------------------------------------------------------------
# FakeOcrEngine — the engine P4/P5/P7 actually run against
# ---------------------------------------------------------------------------

class TestFakeOcrEngine:
    def test_same_pixels_same_text(self):
        fake = FakeOcrEngine()
        assert fake.text_for(crop(3)) == fake.text_for(crop(3))
        assert fake.text_for(crop(3)) in FakeOcrEngine.PHRASES

    def test_a_blank_crop_reads_as_nothing(self):
        assert FakeOcrEngine().text_for(blank()) == ""

    def test_alignment_and_blank_mixed(self):
        fake = FakeOcrEngine()
        out = fake.ocr_crops([crop(1), blank(), crop(2)])
        assert len(out) == 3 and out[1] == "" and out[0] and out[2]

    def test_it_loads_itself_on_demand(self):
        fake = FakeOcrEngine()
        assert fake.status().state is EngineState.UNLOADED
        fake.ocr_crops([crop()])
        assert fake.status().state is EngineState.READY
        assert fake.status().loaded_models and fake.status().backend == "fake"

    def test_calls_are_recorded_for_the_pipeline_test(self):
        fake = FakeOcrEngine()
        fake.ocr_crops([crop(i) for i in range(3)], max_patches=192, batch_size=2,
                       max_new_tokens=64)
        assert fake.calls == [{"count": 3, "max_patches": 192, "batch_size": 2,
                               "max_new_tokens": 64}]

    def test_chunking_and_progress(self):
        fake = FakeOcrEngine()
        seen, cb = recorder()
        fake.ocr_crops([crop(i) for i in range(5)], batch_size=2, progress=cb)
        percents = [p for stage, p, _ in seen if stage == "ocr"]
        assert len(percents) == 3 and percents[-1] == 100.0 and percents == sorted(percents)

    def test_cancellation(self):
        fake = FakeOcrEngine()
        ev = threading.Event()
        ev.set()
        with pytest.raises(StudioError) as ei:
            fake.ocr_crops([crop()], cancel_event=ev)
        assert ei.value.code is ErrorCode.CANCELLED

    @pytest.mark.parametrize("kwargs", [
        {"max_patches": 100}, {"batch_size": 0},
    ])
    def test_validation(self, kwargs):
        fake = FakeOcrEngine()
        with pytest.raises(StudioError) as ei:
            fake.ocr_crops([crop()], **kwargs)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT

    def test_a_non_pil_crop_is_rejected(self):
        fake = FakeOcrEngine()
        with pytest.raises(StudioError) as ei:
            fake.ocr_crops([object()])
        assert ei.value.code is ErrorCode.INVALID_IMAGE

    def test_normalization_is_opt_in_and_keeps_lines(self):
        fake = FakeOcrEngine()
        assert fake.ocr_crops([crop()], normalize_nfkc=True) == fake.ocr_crops([crop()])

    def test_unload_resets(self):
        fake = FakeOcrEngine()
        fake.load()
        fake.unload()
        assert not fake.is_loaded()


# ---------------------------------------------------------------------------
# Real weights (opt-in: `pytest -m models`) — offline, on the vendored bundle
# ---------------------------------------------------------------------------

@pytest.mark.models
def test_real_bundled_weights_recognise_a_sample_crop():
    from pathlib import Path

    store = ModelStore()
    if not store.has_ocr_snapshot():
        pytest.skip("OCR snapshot not vendored (python scripts/fetch_models.py --ocr)")
    page_path = Path("examples/sample_page.png")
    if not page_path.is_file():
        pytest.skip("examples/sample_page.png missing")

    from animetext_studio.core.imaging import crop as im_crop, load_image

    image, _info = load_image(page_path)
    piece = im_crop(image, BBox(530, 913, 699, 963))
    engine = OcrEngine(device="cpu", models_dir=str(store.root), offline=True)
    try:
        engine.load()
        assert engine.weights_source and "hayai" in engine.weights_source
        texts = engine.ocr_crops([piece], max_new_tokens=48, batch_size=1)
        assert len(texts) == 1 and texts[0].strip(), "the model read nothing from a known bubble"
    finally:
        engine.unload()
