"""OCR engine: Hayai OCR v2.7 Supernova (transformers, `trust_remote_code`).

API verified against the *vendored* snapshot of 2026-11-20 (`models/hayai-ocr-v2.7-supernova/`),
not the model card, because the card and the code had drifted:

    model.ocr(images, processor, max_new_tokens=128, max_patches="auto",
              use_cuda_graph=True, dtype=None, batch_size=None, escalate=False)
        -> str | list[str]          # list is aligned with the input, "" = nothing read

Things the code (not the card) told us, which shaped this engine:

* `batch_size` inside `model.ocr` only kicks in for `max_patches="auto"` **and** only
  when `len(images) > batch_size`; otherwise the whole list is preprocessed as one
  batch. It also offers no progress and no cancellation. So **we chunk ourselves**
  (passing `batch_size=None`) — memory stays bounded and the GUI/MCP get per-chunk
  progress and a cancellation point.
* CUDA graphs are gated on `device.type == "cuda"` inside `generate()`, so
  `use_cuda_graph=True` is harmless on CPU — we still pass False there to say what we mean.
* `HAYAI_DISABLE_COMPILE` is read at *module import* time by `_maybe_compile`, so it has
  to be set before the remote code is imported (same import-time trap as D13) — see
  :func:`configure_ocr_compile`.
* `generate()` already falls back (fast path -> fp32 -> reference) and re-raises OOM, so
  the engine must not swallow OOM into a retry: it maps it to OCR_FAILED with advice.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from typing import Any, Optional, Union

from PIL import Image

from ..core.model_store import ModelStore
from ..core.types import OCR_REPO, OCR_REVISION, ErrorCode, StudioError
from .base import BaseOcrEngine, EngineState, ProgressCallback, check_cancel, report

INSTALL_HINT = "pip install 'animetext-studio[ocr]'   (torch + transformers + accelerate)"

#: "auto" | "fp32" | "fp16" | "bf16"
DTYPE_CHOICES = ("auto", "fp32", "fp16", "bf16")

#: `cuda` or `cuda:<index>` — anything else must not reach `_resolve_device`, where a
#: sloppy suffix would silently be parsed as index 0 and reported back as `cuda:0`.
_CUDA_DEVICE_RE = re.compile(r"cuda(:\d+)?$")


# ---------------------------------------------------------------------------
# Lazy import seams (monkeypatched in unit tests; the app must import without torch)
# ---------------------------------------------------------------------------

def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise StudioError(
            ErrorCode.ENGINE_LOAD_FAILED,
            "OCR backend not installed (PyTorch).",
            {"hint": INSTALL_HINT, "reason": str(exc)},
        ) from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise StudioError(
            ErrorCode.ENGINE_LOAD_FAILED,
            "OCR backend not installed (transformers).",
            {"hint": INSTALL_HINT, "reason": str(exc)},
        ) from exc
    return AutoModel, AutoProcessor


def configure_hf_offline(offline: bool) -> None:
    """Never *enable* the network; only ever disable it (see detection.configure_offline)."""
    if not offline:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True


def configure_ocr_compile(device_type: str) -> Optional[str]:
    """Set `HAYAI_DISABLE_COMPILE=1` before the remote code is imported (CPU/other).

    Returns the value it set (None when it left things alone). An explicit value the
    operator set is always respected — this only ever *adds* the opt-out.
    """
    if device_type == "cuda":
        return None
    if os.environ.get("HAYAI_DISABLE_COMPILE") not in (None, ""):
        return None
    os.environ["HAYAI_DISABLE_COMPILE"] = "1"
    return "1"


def _classify_ocr(exc: Exception, context: str) -> StudioError:
    """Map torch/transformers failures onto the project's error taxonomy."""
    if isinstance(exc, StudioError):
        return exc
    name = type(exc).__name__
    text = str(exc)
    if isinstance(exc, ImportError):
        return StudioError(ErrorCode.ENGINE_LOAD_FAILED,
                           f"OCR backend is incomplete at {context}.",
                           {"hint": INSTALL_HINT, "reason": f"{name}: {text}"})
    if "out of memory" in text.lower() or name.endswith("OutOfMemoryError"):
        return StudioError(
            ErrorCode.OCR_FAILED,
            f"Out of memory while running {context}.",
            {"hint": "lower ocr.batch_size, or ocr.max_patches (256 -> 128), or use CPU",
             "reason": f"{name}: {text[:300]}"},
        )
    if name in ("HFValidationError", "RepositoryNotFoundError", "RevisionNotFoundError"):
        return StudioError(ErrorCode.MODEL_NOT_FOUND, f"OCR model not available: {context}.",
                           {"reason": f"{name}: {text[:300]}"})
    if name in ("LocalEntryNotFoundError", "EntryNotFoundError"):
        return StudioError(
            ErrorCode.MODEL_NOT_FOUND,
            f"OCR snapshot is neither on premises nor downloadable offline: {context}.",
            {"hint": "python scripts/fetch_models.py --ocr", "reason": f"{name}: {text[:300]}"},
        )
    if name in ("OfflineModeIsEnabled", "ConnectionError", "Timeout") or \
            isinstance(exc, (ConnectionError, TimeoutError)):
        return StudioError(ErrorCode.DOWNLOAD_FAILED,
                           f"Network needed for {context} but the run is offline.",
                           {"reason": f"{name}: {text[:300]}"})
    return StudioError(ErrorCode.OCR_FAILED, f"OCR failed at {context}.",
                       {"reason": f"{name}: {text[:300]}"})


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class OcrEngine(BaseOcrEngine):
    """Crop-level OCR with Hayai Supernova; weights come from the bundle when present.

    Construction is cheap and side-effect free: torch is imported, the snapshot resolved
    and the weights read only by :meth:`load` (PLAN: independent lazy engines).
    """

    def __init__(
        self,
        device: str = "auto",
        *,
        models_dir: Optional[str] = None,
        offline: bool = False,
        dtype: str = "auto",
        use_cuda_graph: Optional[bool] = None,
        repo_id: str = OCR_REPO,
        revision: str = OCR_REVISION,
    ) -> None:
        super().__init__(backend="torch", device=device)
        if device != "auto" and device != "cpu" and not _CUDA_DEVICE_RE.fullmatch(device):
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                f"device must be 'auto', 'cpu' or 'cuda[:N]', got {device!r}.",
                {"choices": ["auto", "cpu", "cuda", "cuda:0", "cuda:1", "..."]},
            )
        if dtype not in DTYPE_CHOICES:
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              f"dtype must be one of {DTYPE_CHOICES}, got {dtype!r}.",
                              {"choices": list(DTYPE_CHOICES)})
        self._requested_device = device
        self._store = ModelStore(models_dir, offline=offline)
        self._dtype_name = dtype
        self._use_cuda_graph = use_cuda_graph
        self._repo_id = repo_id
        self._revision = revision
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._source: Optional[str] = None      # snapshot dir or repo id actually used

    # -- introspection ------------------------------------------------------

    def _loaded_models(self) -> tuple[str, ...]:
        return (self._repo_id,) if self._state is EngineState.READY else ()

    @property
    def model_store(self) -> ModelStore:
        return self._store

    @property
    def weights_source(self) -> Optional[str]:
        """Where the loaded weights came from (bundle directory or Hub repo id)."""
        return self._source

    def plan(self) -> dict[str, Any]:
        """What `load()` would do — no torch, no downloads (GUI pre-flight / doctor)."""
        bundled = self._store.has_ocr_snapshot()
        return {
            "repo_id": self._repo_id,
            "revision": self._revision,
            "bundled": bundled,
            "snapshot_dir": str(self._store.ocr_dir()) if bundled else None,
            "missing_files": self._store.missing_ocr_files(),
            "offline": self._store.offline,
            "requested_device": self._requested_device,
            "dtype": self._dtype_name,
            "use_cuda_graph": (True if self._use_cuda_graph is None
                               else bool(self._use_cuda_graph)),
        }

    # -- loading ------------------------------------------------------------

    def _resolve_device(self, torch) -> str:
        """'auto' -> cuda:N when a usable CUDA device exists, else cpu. Never lies."""
        if self._requested_device == "cpu":
            return "cpu"
        count = 0
        try:
            count = int(torch.cuda.device_count())
        except Exception:                       # noqa: BLE001 - broken driver builds
            count = 0
        if self._requested_device.startswith("cuda"):
            index = self._requested_device.partition(":")[2]
            wanted = int(index) if index else 0
            if count <= wanted:
                raise StudioError(
                    ErrorCode.ENGINE_LOAD_FAILED,
                    f"CUDA device {wanted} is not available ({count} visible).",
                    {"hint": "use --device auto or --device cpu",
                     "cuda_visible_count": count,
                     "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
                )
            return f"cuda:{wanted}"
        return f"cuda:0" if count > 0 else "cpu"

    def _load_model(self, source: str, device: str, progress: Optional[ProgressCallback]):
        """Import + instantiate. Split out so tests can drive load() without torch."""
        AutoModel, AutoProcessor = _import_transformers()
        trust = {"trust_remote_code": True}
        report(progress, "load_ocr", 35, f"loading processor from {source}")
        processor = AutoProcessor.from_pretrained(source, **trust)

        report(progress, "load_ocr", 55, f"loading weights from {source}")
        model = None
        last_error: Optional[Exception] = None
        for kwargs in ({"dtype": "auto"}, {"torch_dtype": "auto"}, {}):
            # transformers renamed torch_dtype -> dtype in v5; try the modern name first
            # and fall back so a 4.x install still works.
            try:
                model = AutoModel.from_pretrained(source, **trust, **kwargs)
                break
            except TypeError as exc:            # unexpected keyword -> try the older name
                last_error = exc
        if model is None:
            raise _classify_ocr(last_error, f"from_pretrained({source})")

        report(progress, "load_ocr", 80, f"moving to {device}")
        model.to(device)
        model.eval()
        return model, processor

    def load(self, progress: Optional[ProgressCallback] = None) -> None:
        with self._lock:
            if self._state is EngineState.READY:
                report(progress, "load_ocr", 100, "OCR model already loaded")
                return
            self._state = EngineState.LOADING
            try:
                torch = _import_torch()
                self._torch = torch
                device = self._resolve_device(torch)
                self._device = device

                if self._store.has_ocr_snapshot():
                    source: str = str(self._store.require_ocr_snapshot())
                elif self._store.offline:
                    self._store.require_ocr_snapshot()     # raises MODEL_NOT_FOUND
                    raise AssertionError("unreachable")   # pragma: no cover
                else:
                    source = self._repo_id       # transformers resolves/downloads it

                configure_hf_offline(self._store.offline)
                configure_ocr_compile(device.split(":")[0])

                self._model, self._processor = self._load_model(source, device, progress)
                self._source = source

                graph = self._cuda_graph_wanted(device)
                self._detail = (f"{self._dtype_name}, cuda-graphs "
                                f"{'on' if graph else 'off'}, "
                                f"{'bundle' if source.startswith(str(self._store.root)) else 'hub'}")
                self._state = EngineState.READY
                report(progress, "load_ocr", 100, f"OCR ready on {device}")
            except Exception as exc:            # noqa: BLE001
                self._state = EngineState.ERROR
                raise _classify_ocr(exc, "load OCR model") from None

    def _cuda_graph_wanted(self, device: str) -> bool:
        if self._use_cuda_graph is None:
            return device.startswith("cuda")
        return bool(self._use_cuda_graph)

    def _torch_dtype(self):
        if self._dtype_name == "auto" or self._torch is None:
            return None
        return {"fp32": self._torch.float32,
                "fp16": self._torch.float16,
                "bf16": self._torch.bfloat16}[self._dtype_name]

    # -- inference ----------------------------------------------------------

    def ocr_crops(
        self,
        images: list[Image.Image],
        *,
        max_patches: Union[str, int] = "auto",
        batch_size: int = 16,
        max_new_tokens: int = 128,
        normalize_nfkc: Optional[bool] = None,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[str]:
        from ..core.imaging import normalize_text

        crops = list(images)
        for index, image in enumerate(crops):
            if not isinstance(image, Image.Image):
                raise StudioError(ErrorCode.INVALID_IMAGE,
                                  f"Crop {index} is not a PIL.Image.Image.")
        budget = self.check_max_patches(max_patches)
        if int(batch_size) < 1:
            raise StudioError(ErrorCode.INVALID_ARGUMENT, "batch_size must be >= 1.")
        if int(max_new_tokens) < 1:
            raise StudioError(ErrorCode.INVALID_ARGUMENT, "max_new_tokens must be >= 1.")
        if not crops:
            return []
        if self._state is not EngineState.READY:
            self.load(progress)

        model, processor = self._model, self._processor
        if model is None or processor is None:            # unloaded underneath us
            raise StudioError(ErrorCode.ENGINE_LOAD_FAILED, "OCR model is not loaded.")

        torch = self._torch
        dtype = self._torch_dtype()
        graph = self._cuda_graph_wanted(self._device)
        step = max(1, int(batch_size))
        total = len(crops)
        out: list[str] = []
        for start in range(0, total, step):
            check_cancel(cancel_event)
            chunk = crops[start:start + step]
            try:
                # batch_size stays None: we own the chunking so that progress and
                # cancellation exist (see module docstring). escalate=False = single pass.
                got = model.ocr(
                    chunk, processor,
                    max_patches=budget,
                    max_new_tokens=int(max_new_tokens),
                    use_cuda_graph=graph,
                    dtype=dtype,
                    escalate=False,
                )
            except Exception as exc:                      # noqa: BLE001
                raise _classify_ocr(exc, f"ocr(batch {start}..{start + len(chunk)})") from None
            if isinstance(got, str):
                got = [got]
            got = [text if isinstance(text, str) else "" for text in got]
            if len(got) != len(chunk):                    # would silently misalign blocks
                raise StudioError(
                    ErrorCode.OCR_FAILED,
                    f"OCR returned {len(got)} texts for {len(chunk)} crops.",
                    {"hint": "engine/processor version mismatch"},
                )
            out.extend(got)
            report(progress, "ocr", 100.0 * len(out) / total,
                   f"{len(out)}/{total} crops recognised")
            check_cancel(cancel_event)
        if normalize_nfkc:
            out = [normalize_text(text) or "" for text in out]
        self._note_inference()
        return out

    # -- teardown -----------------------------------------------------------

    def _unload_locked(self) -> None:
        model, processor, torch = self._model, self._processor, self._torch
        self._model = None
        self._processor = None
        self._source = None
        if model is None and processor is None:
            return
        try:
            del model
            del processor
        except Exception:                                 # noqa: BLE001
            pass
        try:
            import gc
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            from ..core.memory import trim_heap           # same reasoning as detection (D62)
            trim_heap()
        except Exception:                                 # noqa: BLE001
            pass   # freeing memory must never be an error
