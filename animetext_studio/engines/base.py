"""Engine scaffolding: status model, progress/cancel helpers, deterministic fakes.

The fake engines let the GUI (P5/P6), pipeline (P4) and MCP server (P7) be
developed and tested without model downloads, GPU or network (PLAN `--mock`).
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union

from PIL import Image

from ..core.config import KNOWN_MODELS, MODEL_PRESETS
from ..core.types import BBox, Detection, ErrorCode, StudioError, TEXT_LABEL

#: progress(stage, percent 0..100, message) — same shape the JobManager will use.
ProgressCallback = Callable[[str, float, str], None]


class EngineState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True)
class EngineStatus:
    name: str
    state: EngineState
    loaded_models: tuple[str, ...]
    backend: str
    device: str
    detail: str = ""
    #: True when the variant *list* came from the compiled presets + the bundle instead of the
    #: repo (offline, listing timed out, repo refused). The list is still usable — it is the claim
    #: "these are the variants that exist" that is not. It used to live only on the detection
    #: engine, so it reached the GUI's combo box and stopped there; on the status it reaches the
    #: badge, the MCP door and `--model-status` from one field (P9.2).
    model_list_is_fallback: bool = False
    #: Inferences completed since the engine was (un)loaded. Warmth is *evidence*, not intent:
    #: `--preload` is only verifiable if something records that a model actually ran (P8.3).
    inferences: int = 0

    @property
    def warm(self) -> bool:
        """Loaded **and** at least one inference has completed.

        The distinction is the whole point: a freshly loaded ONNX session on CUDA is not the
        same cost as one that has already built its context and picked its kernels. Reporting
        `ready` as if it were warm would let `--preload` claim something it had not done.
        """
        return EngineState(self.state) is EngineState.READY and self.inferences > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": EngineState(self.state).value,
            "loaded_models": list(self.loaded_models),
            "backend": self.backend,
            "device": self.device,
            "detail": self.detail,
            "model_list_is_fallback": self.model_list_is_fallback,
            "inferences": self.inferences,
            "warm": self.warm,
        }


def _short_model_name(status: EngineStatus) -> str:
    """`yolo12m_animetext` -> `yolo12m`; no model -> ``; several -> `3x`."""
    models = [str(m) for m in (status.loaded_models or ()) if m]
    if not models:
        return ""
    if len(models) == 1:
        return models[0].replace("_animetext", "")
    return f"{len(models)}x"


def engines_badge_text(detection: EngineStatus, ocr: EngineStatus) -> str:
    """One-line engine summary for the panel badge (P8).

    This exists because `unload_engines` used to change nothing on screen but a log row:
    the user could not tell a warm studio from an empty one (and therefore from a page
    that is about to take 20 seconds). Qt-free on purpose, so it is testable headless and
    shared by the panel and any future doctor view. Short by necessity — it sits next to
    the MOCK / OFFLINE / MCP badges in a narrow column.
    """
    parts: list[str] = []
    for label, status in (("det", detection), ("ocr", ocr)):
        state = EngineState(status.state)
        if state is EngineState.READY:
            parts.append(f"{label} {_short_model_name(status) or 'ready'}")
        elif state is EngineState.LOADING:
            parts.append(f"{label} loading")
        elif state is EngineState.ERROR:
            parts.append(f"{label} error")
        # UNLOADED adds nothing: "engines: unloaded" reads better than
        # "engines: det unloaded · ocr unloaded", which says the same thing.
    if not parts:
        return "engines: unloaded"
    return "engines: " + " · ".join(parts)


def models_badge_text(detection: EngineStatus) -> str:
    """The fifth badge: "models: presets", or "" when the list came from the repo (P9.2).

    The listing is bounded by 10 s and skipped offline (D27), so a studio that never reached the
    repo shows the compiled presets plus the bundle. That is the right behaviour, and it was
    recorded — but only in the combo box's disabled items, where nothing shouts. A badge that is
    usually empty is the cheapest way to make "we did not ask, or could not hear" visible without
    training the user to ignore it.

    Qt-free, like :func:`engines_badge_text`: the panel shows it, the tests assert on it, and a
    headless run can print the same verdict.
    """
    if not getattr(detection, "model_list_is_fallback", False):
        return ""
    return "models: presets"


def model_catalog(
    names: list[str], sources: Optional[dict[str, str]] = None
) -> list[dict[str, Any]]:
    """Stable model metadata records (used by GUI combo and MCP list_models).

    ``sources`` maps a model name to where its weights come from — "bundled"
    (on-premises, decision D26) or "hf" (Hub download/cache). Defaults to "hf", so a
    fake/offline catalog stays honest about not having local weights.
    """
    out = []
    for name in names:
        preset = MODEL_PRESETS.get(name, {})
        out.append({
            "name": name,
            "known": name in MODEL_PRESETS,
            "params_m": preset.get("params_m"),
            "flops_g": preset.get("flops_g"),
            "f1": preset.get("f1"),
            "recommended_score_threshold": preset.get("recommended_score_threshold"),
            "source": (sources or {}).get(name, "hf"),
        })
    return out


def report(progress: Optional[ProgressCallback], stage: str, percent: float, message: str) -> None:
    if progress is not None:
        progress(stage, max(0.0, min(100.0, float(percent))), message)


def check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy score-ordered NMS (same semantics as imgutils: keep if IoU <= thr)."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: -d.score):
        if all(_iou(det.bbox, k.bbox) <= iou_threshold for k in kept):
            kept.append(det)
    return kept


class BaseDetectionEngine:
    """Common state/locking; subclasses implement list_models/load/detect."""

    name = "detection"

    def __init__(self, backend: str, device: str) -> None:
        self._lock = threading.RLock()
        self._state = EngineState.UNLOADED
        self._loaded: set[str] = set()
        self._inferences = 0
        self._detail = ""
        self._backend = backend
        self._device = device

    def status(self) -> EngineStatus:
        with self._lock:
            return EngineStatus(
                name=self.name, state=self._state,
                loaded_models=tuple(sorted(self._loaded)),
                backend=self._backend, device=self._device, detail=self._detail,
                inferences=self._inferences,
            )

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()
            self._loaded.clear()
            self._inferences = 0      # unloading ends the warm streak (P8.3)
            self._state = EngineState.UNLOADED

    def _note_inference(self) -> None:
        """Count one completed inference. Called by every engine implementation on success."""
        with self._lock:
            self._inferences += 1

    def _unload_locked(self) -> None:  # hook
        pass

    #: imgutils' own fallback when a repo ships no threshold.json (YOLO official default)
    FALLBACK_SCORE_THRESHOLD = 0.25

    def list_models(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def recommended_threshold(self, model_name: str) -> Optional[float]:
        """F1-optimal score threshold published for this model (None if unknown).

        Drives the GUI's "recommended threshold" button and MCP `list_models`.
        """
        value = MODEL_PRESETS.get(model_name, {}).get("recommended_score_threshold")
        return float(value) if value is not None else None

    def resolved_score_threshold(self, model_name: str,
                                 requested: Optional[float]) -> float:
        """The number an inference run will actually use.

        Needed because the result JSON (§2.1) records a concrete score_threshold even
        when the caller asked for the repo-recommended one (`score=None`).
        """
        if requested is not None:
            return float(requested)
        recommended = self.recommended_threshold(model_name)
        return recommended if recommended is not None else self.FALLBACK_SCORE_THRESHOLD

    def load(self, model_name: str, progress: Optional[ProgressCallback] = None) -> None:
        raise NotImplementedError

    def detect(
        self,
        image: Image.Image,
        model_name: str,
        *,
        iou_threshold: float = 0.35,
        score_threshold: Optional[float] = 0.05,
        allow_dynamic: bool = False,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[Detection]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OCR side
# ---------------------------------------------------------------------------

class BaseOcrEngine:
    """Common state/locking for crop-level OCR; subclasses implement load/ocr_crops.

    The contract is deliberately crop-based (Hayai is a crop recognizer, not a
    page recognizer): the pipeline crops each detection with padding and hands the
    engine a list, so batching, progress and cancellation stay inside the engine.
    """

    name = "ocr"

    #: documented patch budgets of the Hayai processor ("auto" routes across these)
    #: Same list as `core.config.MAX_PATCHES_CHOICES` minus "auto" — the tiers the vendored
    #: processor implements, 512 included (P8.2b).
    MAX_PATCH_CHOICES: tuple[int, ...] = (128, 192, 256, 384, 512)

    def __init__(self, backend: str, device: str) -> None:
        self._lock = threading.RLock()
        self._state = EngineState.UNLOADED
        self._detail = ""
        self._backend = backend
        self._device = device
        self._inferences = 0

    def status(self) -> EngineStatus:
        with self._lock:
            return EngineStatus(
                name=self.name, state=self._state, loaded_models=self._loaded_models(),
                backend=self._backend, device=self._device, detail=self._detail,
                inferences=self._inferences,
            )

    def _loaded_models(self) -> tuple[str, ...]:
        return ()

    @property
    def device(self) -> str:
        return self._device

    def is_loaded(self) -> bool:
        with self._lock:
            return self._state is EngineState.READY

    def load(self, progress: Optional[ProgressCallback] = None) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()
            self._inferences = 0      # unloading ends the warm streak (P8.3)
            self._state = EngineState.UNLOADED

    def _note_inference(self) -> None:
        """Count one completed inference. Called by every engine implementation on success."""
        with self._lock:
            self._inferences += 1

    def _unload_locked(self) -> None:  # hook
        pass

    @staticmethod
    def check_max_patches(max_patches: Union[str, int]) -> Union[str, int]:
        """Validate the patch budget against the processor's documented values."""
        if max_patches == "auto":
            return "auto"
        try:
            value = int(max_patches)
        except (TypeError, ValueError):
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                f"max_patches must be 'auto' or one of {BaseOcrEngine.MAX_PATCH_CHOICES}, "
                f"got {max_patches!r}.",
                {"choices": ["auto", *BaseOcrEngine.MAX_PATCH_CHOICES]},
            ) from None
        if value not in BaseOcrEngine.MAX_PATCH_CHOICES:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                f"max_patches must be 'auto' or one of {BaseOcrEngine.MAX_PATCH_CHOICES}, "
                f"got {value}.",
                {"choices": ["auto", *BaseOcrEngine.MAX_PATCH_CHOICES]},
            )
        return value

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
        """One string per input crop ("" = nothing recognised). Same length as input."""
        raise NotImplementedError


class FakeOcrEngine(BaseOcrEngine):
    """Deterministic fake: text derived from the crop's own pixels (no model, no torch).

    Same crop bytes -> same text, always, which is what makes the P4 golden-result test
    and the P7 MCP harness reproducible. Empty/blank crops yield "", so the pipeline's
    "no text" branch is reachable in mock mode too.
    """

    #: multilingual filler, picked by crop hash — mirrors what Hayai can output (JA/ZH/KO/EN)
    PHRASES = (
        "この攻撃で決まりだ", "まだまだよ、私は", "絶対に負けない", "何!? その力は",
        "真相はいつも闇の中", "決戦の前夜", "風が止んだ", "その日、空は赤かった",
        "第4話", "風 stopped blowing", "决战的前夜", "결전의 전야",
    )

    def __init__(self, model_name: str = "hayai-ocr-v2.7-supernova (mock)") -> None:
        super().__init__(backend="fake", device="cpu")
        self._model_name = model_name
        self.calls: list[dict[str, Any]] = []      # what the pipeline asked for (test hook)

    def _loaded_models(self) -> tuple[str, ...]:
        return (self._model_name,) if self._state is EngineState.READY else ()

    def load(self, progress: Optional[ProgressCallback] = None) -> None:
        with self._lock:
            report(progress, "load_ocr", 0, "OCR model (mock)")
            self._state = EngineState.READY
            self._detail = "mock engine"
            report(progress, "load_ocr", 100, "OCR model ready (mock)")

    @staticmethod
    def text_for(image: Image.Image) -> str:
        from ..core.imaging import to_png_bytes

        digest = hashlib.sha256(to_png_bytes(image)).hexdigest()
        if not image.getbbox():          # fully blank crop -> nothing to read
            return ""
        return FakeOcrEngine.PHRASES[int(digest[:8], 16) % len(FakeOcrEngine.PHRASES)]

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
        self.check_max_patches(max_patches)
        if batch_size < 1:
            raise StudioError(ErrorCode.INVALID_ARGUMENT, "batch_size must be >= 1.")
        self.calls.append({"count": len(crops), "max_patches": max_patches,
                           "batch_size": int(batch_size),
                           "max_new_tokens": int(max_new_tokens)})
        if self._state is not EngineState.READY:
            self.load(progress)
        out: list[str] = []
        total = len(crops)
        for start in range(0, total, max(1, int(batch_size))):
            check_cancel(cancel_event)
            chunk = crops[start:start + int(batch_size)]
            out.extend(self.text_for(image) for image in chunk)
            report(progress, "ocr", 100.0 * len(out) / max(1, total),
                   f"{len(out)}/{total} crops (mock)")
        if normalize_nfkc:
            out = [normalize_text(text) or "" for text in out]
        self._note_inference()
        return out


class FakeDetectionEngine(BaseDetectionEngine):
    """Deterministic fake: fixed fractional layout, real threshold/NMS semantics.

    Same image size + same parameters -> identical boxes, always. One layout
    pair overlaps (IoU ~= 0.67) so the IOU threshold has a visible effect.
    """

    #: (x0f, y0f, x1f, y1f, score) as fractions of image size
    LAYOUT = (
        (0.62, 0.05, 0.78, 0.30, 0.93),
        (0.64, 0.07, 0.80, 0.32, 0.85),  # overlaps the first box
        (0.10, 0.12, 0.24, 0.42, 0.88),
        (0.40, 0.45, 0.56, 0.75, 0.81),
        (0.08, 0.60, 0.22, 0.92, 0.74),
        (0.70, 0.55, 0.90, 0.88, 0.66),
    )

    #: the fake's own "threshold.json": a plain constant, so None stays deterministic
    FAKE_DEFAULT_SCORE_THRESHOLD = 0.05

    def __init__(self) -> None:
        super().__init__(backend="fake", device="cpu")

    def list_models(self) -> list[dict[str, Any]]:
        return model_catalog(list(KNOWN_MODELS))

    def resolved_score_threshold(self, model_name: str,
                                 requested: Optional[float]) -> float:
        if requested is not None:
            return float(requested)
        return float(self.FAKE_DEFAULT_SCORE_THRESHOLD)

    def load(self, model_name: str, progress: Optional[ProgressCallback] = None) -> None:
        if model_name not in KNOWN_MODELS:
            raise StudioError(ErrorCode.MODEL_NOT_FOUND, f"Unknown model {model_name!r}.",
                              {"known_models": list(KNOWN_MODELS)})
        with self._lock:
            report(progress, "load_model", 0, f"{model_name} (mock)")
            self._loaded.add(model_name)
            self._state = EngineState.READY
            report(progress, "load_model", 100, f"{model_name} ready (mock)")

    def detect(
        self,
        image: Image.Image,
        model_name: str,
        *,
        iou_threshold: float = 0.35,
        score_threshold: Optional[float] = 0.05,
        allow_dynamic: bool = False,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[Detection]:
        check_cancel(cancel_event)
        if not isinstance(image, Image.Image):
            raise StudioError(ErrorCode.INVALID_IMAGE, "Expected a PIL.Image.Image.")
        if model_name not in KNOWN_MODELS:
            raise StudioError(ErrorCode.MODEL_NOT_FOUND, f"Unknown model {model_name!r}.",
                              {"known_models": list(KNOWN_MODELS)})
        if model_name not in self._loaded:
            self.load(model_name, progress=progress)
        thr = self.resolved_score_threshold(model_name, score_threshold)
        w, h = image.size
        report(progress, "detect", 0, f"{model_name} (mock)")
        dets = [
            Detection(BBox(x0 * w, y0 * h, x1 * w, y1 * h), TEXT_LABEL, score)
            for (x0, y0, x1, y1, score) in self.LAYOUT
            if score >= thr
        ]
        dets = _nms(dets, iou_threshold)
        check_cancel(cancel_event)
        self._note_inference()
        report(progress, "detect", 100, f"{len(dets)} text block(s) (mock)")
        return dets
