"""Part 4 — the pipeline: detect -> order + crop -> batched OCR -> the §2.1 document.

Three steps, exactly, GUI-independent (PLAN §3 Part 4):

1. **detect** — one `detect()` call for the whole page; junk boxes (area below
   `pipeline.min_box_area_px`) are dropped here, once, so they never reach the OCR stage
   or the document.
2. **order + crop** — boxes are put into reading order *before* cropping, because that
   order is what the result's `id` / `reading_order` are, and it lets the OCR batch follow
   the page instead of the detector's raw output. Each box is cropped padded + clamped
   (`imaging.crop_pair`), and a box that survives the area floor but is still unreadable
   (a 1x200 strip) is gated out of OCR and recorded as such instead of being fed to the
   model.
3. **assemble** — ids, palette colours, per-block `error`, timings, the image digest.

Deliberate choices, all mirrored in `tracking.md`:

* **One batched OCR call** (D33: the engine owns chunking, progress and cancellation). If
  that call fails in a way that *might* be one poisoned crop, the pipeline re-runs the same
  crops one at a time so a single bad crop cannot lose the run (D37). Load/auth/offline
  failures are not retried — they will fail identically per crop, and pretending otherwise
  just hides the real error behind N copies of it.
* **`bbox` in the document is the effective box** (padded + clamped), not the raw detector
  box: it is the truth about the pixels the text came from, and it is always inside the
  image (D38).
* **A block with `text: null` always says why** — either an `error` object, or the
  `SKIPPED_TOO_SMALL` code. `text: ""` means "OCR ran and read nothing", which is a
  different fact and stays distinguishable (D39).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence, Union

from PIL import Image

from .config import AppConfig
from .imaging import (
    MIN_CROP_SIDE, crop_pair, image_from_bytes, is_ocr_worthy, load_image,
)
from .types import (
    CONTAINMENT_RATIO, DETECTION_REPO, OCR_REPO, TEXT_LABEL, TRUNCATION_RATIO, BBox,
    Detection, DetectionInfo, ErrorCode, ImageInfo, OcrInfo, PipelineResult, StudioError,
    TextBlock, TimingMs, color_for_index,
)

__all__ = [
    "Pipeline", "PipelineEngines", "engines_from_config", "order_detections",
    "run_file", "SKIP_ERROR_CODE", "STAGE_SPANS",
    "annotate_blocks", "contained_in_map", "is_possibly_truncated",
]

#: Block-level error code for "OCR was not attempted on this box" (schema-legal: the
#: result schema types `error.code` as a string; the taxonomy in `types.ErrorCode` is for
#: *operations*, this is a per-block outcome).
SKIP_ERROR_CODE = "SKIPPED_TOO_SMALL"

#: Stage -> (start, end) of the overall 0..100 bar. Engine stages are forwarded, so the
#: pipeline translates their local percent into page-level progress. Names not listed
#: here are passed through with the span of the stage they arrived in.
STAGE_SPANS: dict[str, tuple[float, float]] = {
    "download": (0.0, 8.0),
    "load_model": (8.0, 20.0),
    "load_session": (8.0, 20.0),
    "load_ocr": (8.0, 20.0),
    "detect": (20.0, 45.0),
    "crop": (45.0, 50.0),
    "ocr": (50.0, 95.0),
    "assemble": (95.0, 100.0),
}

#: Failures that are never worth retrying per crop: they are about the engine, the model
#: or the operator's request, not about one image.
_FATAL_OCR_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.CANCELLED, ErrorCode.INVALID_ARGUMENT, ErrorCode.INVALID_IMAGE,
    ErrorCode.ENGINE_LOAD_FAILED, ErrorCode.MODEL_NOT_FOUND, ErrorCode.DOWNLOAD_FAILED,
    ErrorCode.HF_AUTH_REQUIRED, ErrorCode.INVALID_CONFIG,
})

ProgressCallback = Callable[[str, float, str], None]


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------

def _row_key(bbox: BBox) -> float:
    return bbox.y0 + bbox.height / 2.0


def _group_rows(detections: Sequence[Detection]) -> list[list[Detection]]:
    """Cluster boxes into visual rows by vertical overlap (row = shared reading line).

    Two boxes join a row when their vertical ranges overlap by more than half of the
    smaller height — the tategaki/yoko mix on a manga page makes pure y-sorting wrong and
    pure per-box sorting worse. Stable and deterministic: rows are ordered by the vertical
    centre of their topmost member.
    """
    rows: list[list[Detection]] = []
    for det in sorted(detections, key=lambda d: _row_key(d.bbox)):
        placed = False
        for row in rows:
            for member in row:
                shared = (min(det.bbox.y1, member.bbox.y1)
                          - max(det.bbox.y0, member.bbox.y0))
                smallest = min(det.bbox.height, member.bbox.height)
                if smallest > 0 and shared > smallest / 2.0:
                    row.append(det)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rows.append([det])
    return sorted(rows, key=lambda row: min(_row_key(d.bbox) for d in row))


def order_detections(detections: Sequence[Detection], mode: str = "manga_rtl",
                     ) -> list[Detection]:
    """Deterministic reading order. Unknown modes fall back to `manga_rtl` (the default).

    * `manga_rtl`      — rows top-to-bottom, boxes right-to-left inside a row (Japanese).
    * `top_to_bottom`  — rows top-to-bottom, boxes left-to-right (western / captions).
    * `score`          — most confident first, y/x as the tie-break.
    """
    items = list(detections)
    if mode == "score":
        return sorted(items, key=lambda d: (-d.score, d.bbox.y0, d.bbox.x0))
    reverse = mode != "top_to_bottom"
    ordered: list[Detection] = []
    for row in _group_rows(items):
        ordered.extend(sorted(row, key=lambda d: (d.bbox.x0, d.bbox.y0), reverse=reverse))
    return ordered


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineEngines:
    """The two engines as a pair, so one object is what the GUI/MCP share (D14)."""

    detection: Any
    ocr: Any

    @property
    def names(self) -> tuple[str, str]:
        return (self.detection.name, self.ocr.name)

    def unload_all(self) -> None:
        for engine in (self.ocr, self.detection):
            try:
                engine.unload()
            except Exception:                              # noqa: BLE001
                pass    # freeing memory must never be an error


def engines_from_config(config: AppConfig, *, mock: bool = False,
                        hf_token: Optional[str] = None,
                        provider: Optional[str] = None) -> PipelineEngines:
    """Build the engine pair from config. Cheap and side-effect free (lazy engines).

    `mock=True` returns the fakes: no torch, no ONNX, no downloads — the same code path the
    GUI and the MCP harness run against.
    """
    if mock:
        from ..engines.base import FakeDetectionEngine, FakeOcrEngine

        return PipelineEngines(FakeDetectionEngine(), FakeOcrEngine())

    from ..engines.detection import DetectionEngine
    from ..engines.ocr import OcrEngine

    token = hf_token
    if token is None:
        import os

        token = os.environ.get(config.hf.token_env) or None

    return PipelineEngines(
        DetectionEngine(hf_token=token, hf_cache_dir=config.paths.hf_cache_dir,
                        provider=provider, models_dir=config.paths.models_dir,
                        offline=config.paths.offline),
        OcrEngine(device=config.ocr.device, models_dir=config.paths.models_dir,
                  offline=config.paths.offline),
    )


# ---------------------------------------------------------------------------
# Progress translation
# ---------------------------------------------------------------------------

class _Bar:
    """Page-level progress: translate engine stages into one monotonic 0..100 bar."""

    def __init__(self, sink: Optional[ProgressCallback], cancel_event: Optional[threading.Event]):
        self._sink = sink
        self._high = -1.0
        self._last: Optional[tuple] = None
        self.cancel_event = cancel_event

    def emit(self, stage: str, percent: float, message: str = "") -> None:
        if self._sink is None:
            return
        span = STAGE_SPANS.get(stage)
        if span is None:
            value, message = float(percent), f"[{stage}] {message}" if message else stage
        else:
            start, end = span
            value = start + max(0.0, min(100.0, float(percent))) / 100.0 * (end - start)
        # Monotonic: a stage that starts late (the OCR engine loading its weights inside
        # the first batch) must never drag the bar backwards.
        value = max(self._high, min(100.0, value))
        self._high = value
        point = (stage, value, message)
        if point == self._last:
            return                      # the engine and the pipeline can report the same thing
        self._last = point
        self._sink(stage, round(value, 1), message)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """detect -> order + crop -> OCR -> `PipelineResult`, with one engine pair per instance.

    Not thread-safe by design: exactly one job runs at a time (`JobManager`, P4/P5/P7),
    which is also what keeps an ONNX session and a torch model on one worker thread (D14).
    """

    def __init__(self, detection: Any, ocr: Any, config: Optional[AppConfig] = None, *,
                 ocr_enabled: bool = True) -> None:
        self.detection = detection
        self.ocr = ocr
        self.config = config or AppConfig.default()
        self.ocr_enabled = ocr_enabled

    # -- entry points ------------------------------------------------------

    def run_file(self, path: Union[str, Any], *, progress: Optional[ProgressCallback] = None,
                 cancel_event: Optional[threading.Event] = None,
                 **kwargs: Any) -> PipelineResult:
        image, info = load_image(path)
        return self.run(image, info=info, progress=progress, cancel_event=cancel_event,
                        **kwargs)

    def run_bytes(self, data: bytes, *, file_name: Optional[str] = None,
                  source: str = "mcp-base64",
                  progress: Optional[ProgressCallback] = None,
                  cancel_event: Optional[threading.Event] = None,
                  **kwargs: Any) -> PipelineResult:
        image, info = image_from_bytes(data, file_name=file_name, source=source)
        return self.run(image, info=info, progress=progress, cancel_event=cancel_event,
                        **kwargs)

    def run(self, image: Image.Image, *, info: Optional[ImageInfo] = None,
            progress: Optional[ProgressCallback] = None,
            cancel_event: Optional[threading.Event] = None,
            ocr_enabled: Optional[bool] = None) -> PipelineResult:
        """`ocr_enabled` overrides `self.ocr_enabled` for *this run only* (P7 `detect`).

        Stage-1-only runs are a legitimate request ("show me the boxes, don't spend 3 s per
        crop reading them"). The override is a local, so two callers can never race on the
        instance flag - and `JobManager` runs one job at a time anyway (D14).
        """
        read_text = self.ocr_enabled if ocr_enabled is None else bool(ocr_enabled)
        # `info` is checked before anything expensive happens: a run that cannot record where
        # its pixels came from must not spend three seconds loading weights first.
        if not isinstance(image, Image.Image):
            raise StudioError(ErrorCode.INVALID_IMAGE,
                              "Pipeline.run needs a PIL.Image.Image.",
                              {"hint": "use run_file()/run_bytes(), or pass info= "
                                       "from load_image()/image_from_bytes()"})
        if info is None:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                "Pipeline.run needs an ImageInfo to record where the pixels came from.",
                {"hint": "use run_file()/run_bytes(), or pass info= from "
                         "load_image()/image_from_bytes()",
                 "sources": ["file", "clipboard", "mcp-base64", "mcp-path"]},
            )
        bar = _Bar(progress, cancel_event)
        started = time.monotonic()
        cfg = self.config

        # 1. detect -------------------------------------------------------
        floor = int(cfg.pipeline.min_box_area_px)
        detections = _as_detections(self.detection.detect(
            image, cfg.detection.model,
            iou_threshold=cfg.detection.iou_threshold,
            score_threshold=cfg.detection.score_threshold,
            allow_dynamic=cfg.detection.allow_dynamic,
            progress=bar_for(bar, "detect"), cancel_event=cancel_event,
        ), engine_name=str(getattr(self.detection, "name", "detection")))
        detect_ms = _ms(started)
        kept = [d for d in detections if d.bbox.area >= floor]
        dropped = len(detections) - len(kept)
        bar.emit("detect", 100, f"{len(kept)} text block(s)"
                                + (f", {dropped} below {floor} px dropped" if dropped else ""))

        # 2. order + crop -------------------------------------------------
        ordered = order_detections(kept, cfg.pipeline.reading_order)
        bar.emit("crop", 0, f"cropping {len(ordered)} box(es)")
        crops: list[tuple[int, Image.Image]] = []
        skipped: dict[int, dict[str, Any]] = {}
        for index, det in enumerate(ordered):
            if not read_text:
                skipped[index] = {"code": "SKIPPED", "message": "OCR disabled for this run"}
                continue
            piece, effective = crop_pair(image, det.bbox, padding=cfg.ocr.crop_padding_px)
            ordered[index] = _with_bbox(det, effective)
            if is_ocr_worthy(effective, min_side=MIN_CROP_SIDE, min_area=max(1, floor)):
                crops.append((index, piece))
            else:
                skipped[index] = {
                    "code": SKIP_ERROR_CODE,
                    "message": f"box too small to read "
                               f"({effective.width}x{effective.height} px, floor {floor} px)",
                }
        bar.emit("crop", 100, f"{len(crops)} crop(s) for OCR")

        # 3. OCR + assemble ----------------------------------------------
        texts: dict[int, str] = {}
        errors: dict[int, dict[str, Any]] = dict(skipped)
        ocr_started = time.monotonic()
        if crops:
            # Boundaries are emitted by the pipeline, not only by the engine: the bar must
            # be complete even if an engine reports nothing at all.
            bar.emit("ocr", 0, f"reading {len(crops)} crop(s)")
            texts, ocr_errors = self._recognise(crops, bar, cancel_event)
            errors.update(ocr_errors)
            bar.emit("ocr", 100, f"{len(texts)} crop(s) read")
        ocr_ms = _ms(ocr_started) if crops else 0

        blocks = annotate_blocks(
            [TextBlock(
                id=index, bbox=det.bbox, label=det.label, score=round(det.score, 4),
                text=texts.get(index), reading_order=index,
                color=color_for_index(index), error=errors.get(index),
            )
             for index, det in enumerate(ordered)],
            max_new_tokens=cfg.ocr.max_new_tokens,
        )
        flagged = sum(1 for b in blocks if b.flags or b.contained_in is not None)
        bar.emit("assemble", 100, f"{sum(1 for b in blocks if b.text)} block(s) with text"
                 + (f", {flagged} flagged" if flagged else ""))

        return PipelineResult(
            image=info,
            detection=self._detection_info(ordered),
            ocr=self._ocr_info(),
            timing_ms=TimingMs(detect=detect_ms, ocr=ocr_ms, total=_ms(started)),
            blocks=blocks,
            # The document has to say what it is: a stage-1 run and a full run whose OCR
            # read nothing are different facts, and `stage` is the only place that knows
            # (schema 1.1, P8).
            stage="full" if read_text else "detect",
        )

    # -- re-reading one block (P9.3) ---------------------------------------

    def reocr_blocks(self, image: Image.Image, result: PipelineResult,
                     block_ids: Sequence[int], *, info: Optional[ImageInfo] = None,
                     progress: Optional[ProgressCallback] = None,
                     cancel_event: Optional[threading.Event] = None) -> PipelineResult:
        """Re-read the named blocks of an existing document, without detecting again (P9.3).

        The request this exists for: block 7 came back `possible_truncation`, and the caller wants
        *that box* read again with a larger `max_new_tokens` — not a second opinion on the whole
        page, and not another 2 seconds of detection to get it. The machinery was already there
        (`OcrEngine.ocr_crops` takes any list of crops; `annotate_blocks` recomputes the flags
        from blocks alone), so this is an entry point, not a feature.

        What comes back is a **whole document** (D81), not a patch: same image, same detections,
        same ids, colours and reading order, new `created_at`/`timing_ms`, and new `text`/`error`
        on the selected blocks. `result://latest` and `save_result` promise one document describes
        one page, and a spliced half-document would break that for every reader. `stage` is
        `"reocr"` so the document says how it was made.

        The image is a required argument because the app keeps the *document*, not the pixels: the
        same file (or the same bytes) has to come back in. Boxes are re-cropped from the stored
        (already padding-adjusted) bbox, so the crop is the one the text came from.
        """
        ids = self._selected_ids(result, block_ids)
        if not isinstance(image, Image.Image):
            raise StudioError(ErrorCode.INVALID_IMAGE,
                              "Pipeline.reocr_blocks needs a PIL.Image.Image.",
                              {"hint": "use reocr_blocks_file()/reocr_blocks_bytes(), or pass "
                                       "info= from load_image()/image_from_bytes()"})
        if info is None:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                "Pipeline.reocr_blocks needs an ImageInfo to record where the pixels came from.",
                {"hint": "the image must be the page these blocks were read from"},
            )
        bar = _Bar(progress, cancel_event)
        started = time.monotonic()
        cfg = self.config
        floor = max(1, int(cfg.pipeline.min_box_area_px))
        by_id = {block.id: block for block in result.blocks}

        crops: list[tuple[int, Image.Image]] = []
        errors: dict[int, dict[str, Any]] = {}
        bar.emit("crop", 0, f"cropping {len(ids)} block(s)")
        for block_id in ids:
            piece, effective = crop_pair(image, by_id[block_id].bbox,
                                         padding=cfg.ocr.crop_padding_px)
            if is_ocr_worthy(effective, min_side=MIN_CROP_SIDE, min_area=floor):
                crops.append((block_id, piece))
            else:
                errors[block_id] = {
                    "code": SKIP_ERROR_CODE,
                    "message": f"box too small to read "
                               f"({effective.width}x{effective.height} px, floor {floor} px)",
                }
        bar.emit("crop", 100, f"{len(crops)} crop(s) for OCR")

        texts: dict[int, str] = {}
        ocr_started = time.monotonic()
        if crops:
            bar.emit("ocr", 0, f"re-reading {len(crops)} crop(s)")
            texts, ocr_errors = self._recognise(crops, bar, cancel_event)
            errors.update(ocr_errors)
            bar.emit("ocr", 100, f"{len(texts)} crop(s) read")
        ocr_ms = _ms(ocr_started) if crops else 0

        rebuilt = [replace(block, text=texts.get(block.id), error=errors.get(block.id))
                   if block.id in by_id and block.id in ids else block
                   for block in result.blocks]
        # Flags and containment are recomputed for the whole page, not patched per block: a
        # longer text can now contain, or be contained by, a neighbour (D32 owns that rule).
        blocks = annotate_blocks(rebuilt, max_new_tokens=cfg.ocr.max_new_tokens)
        return PipelineResult(
            image=info,
            detection=result.detection,          # nothing was re-detected; the boxes are the same
            ocr=self._ocr_info(),                # ... but the reading settings may not be
            timing_ms=TimingMs(detect=0, ocr=ocr_ms, total=_ms(started)),
            blocks=blocks,
            stage="reocr",
        )

    def reocr_blocks_file(self, path: Union[str, Any], result: PipelineResult,
                          block_ids: Sequence[int], **kwargs: Any) -> PipelineResult:
        image, info = load_image(path)
        return self.reocr_blocks(image, result, block_ids, info=info, **kwargs)

    def reocr_blocks_bytes(self, data: bytes, result: PipelineResult,
                           block_ids: Sequence[int], *, file_name: Optional[str] = None,
                           source: str = "mcp-base64", **kwargs: Any) -> PipelineResult:
        image, info = image_from_bytes(data, file_name=file_name, source=source)
        return self.reocr_blocks(image, result, block_ids, info=info, **kwargs)

    @staticmethod
    def _selected_ids(result: PipelineResult, block_ids: Sequence[int]) -> list[int]:
        """Validate the ids against the document — the same rule `get_result(block_ids=…)` uses."""
        return block_id_list(block_ids, available=[block.id for block in result.blocks])


    # -- OCR with per-block isolation -------------------------------------

    def _recognise(self, crops: list[tuple[int, Image.Image]], bar: _Bar,
                   cancel_event: Optional[threading.Event],
                   ) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
        cfg = self.config
        images = [piece for _, piece in crops]
        common = dict(max_patches=cfg.ocr.max_patches, batch_size=cfg.ocr.batch_size,
                      max_new_tokens=cfg.ocr.max_new_tokens,
                      normalize_nfkc=cfg.ocr.normalize_nfkc,
                      progress=bar_for(bar, "ocr"), cancel_event=cancel_event)
        try:
            got = _as_texts(self.ocr.ocr_crops(images, **common), len(images),
                            engine_name=str(getattr(self.ocr, "name", "ocr")))
            return _place(crops, got)
        except StudioError as exc:
            # `details.contract` is the engine breaking its own contract (wrong length, wrong
            # type). Retrying that crop by crop would run the same broken call N times and learn
            # nothing; a poisoned *image* is the only reason the per-crop path exists.
            if _is_contract_error(exc) or exc.code in _FATAL_OCR_CODES or len(crops) == 1:
                raise
            bar.emit("ocr", 50, f"batch failed ({exc.code.value}); retrying crop by crop")

        # One poisoned crop must not cost the whole page (D37): re-run the same crops
        # individually and record the failure where it belongs.
        texts: dict[int, str] = {}
        errors: dict[int, dict[str, Any]] = {}
        total = len(crops)
        for done, (index, piece) in enumerate(crops, start=1):
            try:
                got = _as_texts(self.ocr.ocr_crops([piece], **{**common, "batch_size": 1,
                                                              "progress": None}), 1,
                                engine_name=str(getattr(self.ocr, "name", "ocr")))
                placed, missing = _place([(index, piece)], got)
                texts.update(placed)
                errors.update(missing)
            except StudioError as exc:
                if exc.code is ErrorCode.CANCELLED:
                    raise
                errors[index] = exc.to_dict()
            bar.emit("ocr", 100.0 * done / total, f"{done}/{total} crops (isolated)")
        return texts, errors

    # -- document sections ------------------------------------------------

    def _detection_info(self, detections: Sequence[Detection]) -> DetectionInfo:
        engine = self.detection
        threshold = self.config.detection.score_threshold
        resolver = getattr(engine, "resolved_score_threshold", None)
        if resolver is not None:
            try:
                threshold = resolver(self.config.detection.model, threshold)
            except Exception:                               # noqa: BLE001
                pass                                        # keep the configured number
        return DetectionInfo(
            repo=DETECTION_REPO, model=self.config.detection.model,
            iou_threshold=float(self.config.detection.iou_threshold),
            score_threshold=float(threshold),
            allow_dynamic=bool(self.config.detection.allow_dynamic),
            backend=str(getattr(engine.status(), "backend", "onnxruntime")),
        )

    def _ocr_info(self) -> OcrInfo:
        status = self.ocr.status()
        device = getattr(status, "device", None) or self.config.ocr.device
        return OcrInfo(
            repo=OCR_REPO, device=str(device), max_patches=self.config.ocr.max_patches,
            batch_size=int(self.config.ocr.batch_size),
            max_new_tokens=int(self.config.ocr.max_new_tokens),
            crop_padding_px=int(self.config.ocr.crop_padding_px),
            normalize_nfkc=bool(self.config.ocr.normalize_nfkc),
        )


# ---------------------------------------------------------------------------
# Document annotations (result schema 1.2, P8.2b) — pure, deterministic, no I/O
# ---------------------------------------------------------------------------

def is_possibly_truncated(text: Optional[str], max_new_tokens: int) -> bool:
    """Did this text probably stop at the generation cap rather than at the end of the line?

    The threshold is the model author's own: `processing_hayai.check_escalation` treats
    `len(pred) >= 0.9 * max_new_tokens` as "this hit the cap". We flag it instead of acting
    on it — raising `max_new_tokens` automatically would change the document for a whole page
    because one block was long, and the caller may well prefer a truncated string it knows
    about over a slower run it did not ask for.
    """
    if not text or max_new_tokens < 1:
        return False
    return len(text) >= max_new_tokens * TRUNCATION_RATIO


def contained_in_map(blocks: Sequence[TextBlock],
                     ratio: float = CONTAINMENT_RATIO) -> dict[int, int]:
    """block id -> id of the tightest larger box covering >= `ratio` of it.

    The detector really does return nested boxes (a title band around its columns, a page-wide
    box around a smaller one) and NMS keeps them because their IoU is low. A reader who does
    not know that reads a nested crop as its own sentence — measured on 2026-11-22, a nested
    box inside a narration slab OCR'd as `った`. Geometry is the only thing needed to warn
    about that, so this is geometry only: no text comparison, no dropping, no merging.

    The *smallest* qualifying container wins (a box inside two boxes belongs to the inner one),
    ties broken by id so the map is deterministic across runs and platforms.
    """
    out: dict[int, int] = {}
    for block in blocks:
        if block.bbox.area <= 0:
            continue
        best: Optional[tuple[int, int]] = None      # (container area, container id)
        for other in blocks:
            if other.id == block.id or other.bbox.area <= block.bbox.area:
                continue                            # must be strictly larger
            if block.bbox.contained_ratio(other.bbox) < ratio:
                continue
            key = (other.bbox.area, other.id)
            if best is None or key < best:
                best = key
        if best is not None:
            out[block.id] = best[1]
    return out


def annotate_blocks(blocks: Sequence[TextBlock], *,
                    max_new_tokens: int,
                    containment_ratio: float = CONTAINMENT_RATIO) -> tuple[TextBlock, ...]:
    """Add `flags` + `contained_in` to freshly built blocks (schema 1.2).

    Applied to every run, including `stage: "detect"` documents: containment is known from the
    boxes alone, and a detect-only document that cannot warn about a nested fragment would be
    leaving the useful half of the feature on the floor.
    """
    containers = contained_in_map(blocks, containment_ratio)
    annotated: list[TextBlock] = []
    for block in blocks:
        flags: list[str] = []
        if is_possibly_truncated(block.text, max_new_tokens):
            flags.append("possible_truncation")
        annotated.append(replace(block, flags=tuple(flags),
                                 contained_in=containers.get(block.id)))
    return tuple(annotated)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bar_for(bar: _Bar, fallback_stage: str) -> ProgressCallback:
    """Adapt an engine's (stage, percent, message) report into the page bar.

    The engine's own stage name is kept when we know it (`download`, `load_model`,
    `load_session`, `load_ocr`, `detect`, `ocr`) so the GUI can say *what* is happening;
    anything unexpected is reported under the pipeline stage it happened in.
    """
    def report(stage: str, percent: float, message: str) -> None:
        bar.emit(stage if stage in STAGE_SPANS else fallback_stage, percent, message)

    return report


def _as_detections(value: Any, *, engine_name: str = "detection") -> list[Detection]:
    """Contract check on what a detection engine handed back (P8.4).

    The engine contract is `list[Detection]`. When it is not — a mock that returns `None`, a
    future engine that returns raw dicts — the failure would otherwise surface two lines later
    as `AttributeError: 'dict' object has no attribute 'bbox'`, reported as INTERNAL with no
    hint of which engine misbehaved. Naming the engine and the offending element is the whole
    difference between a five-minute fix and an afternoon. The check is one `isinstance` per
    box, which is free next to an inference.
    """
    if not isinstance(value, (list, tuple)):
        raise StudioError(
            ErrorCode.INTERNAL,
            f"The {engine_name} engine returned {type(value).__name__} instead of a list "
            f"of detections.",
            {"engine": engine_name, "got": type(value).__name__, "contract": True},
        )
    for index, item in enumerate(value):
        if isinstance(item, Detection):
            continue
        if isinstance(item, (tuple, list)) and len(item) == 4:
            # The (x0, y0, x1, y1) shape an engine *would* reasonably return; still our bug to
            # translate, but a cheap and lossless one.
            value[index] = Detection(BBox(*item), TEXT_LABEL, 1.0)   # type: ignore[index]
            continue
        raise StudioError(
            ErrorCode.INTERNAL,
            f"The {engine_name} engine returned {type(item).__name__} at index {index} "
            f"instead of a Detection.",
            {"engine": engine_name, "index": index, "got": type(item).__name__,
             "contract": True},
        )
    return list(value)


def _is_contract_error(exc: StudioError) -> bool:
    """True when the engine, not the image, is what failed (see `_as_detections`/`_as_texts`)."""
    return bool(isinstance(exc.details, dict) and exc.details.get("contract"))


def _as_texts(value: Any, expected: int, *, engine_name: str = "ocr") -> list[Optional[str]]:
    """One answer per crop, or say which ones are missing (P8.4).

    D39 is the invariant this protects: in the document, `text: null` always *says why* (an
    `error` object), and `text: ""` means "read, nothing there". An engine that returns a short
    list, or `None` for a crop, used to produce `null` with no reason attached — the one state
    the document is not allowed to be in. `None` is passed through for the pipeline to record as
    an explicit per-block `OCR_FAILED`; a wrong length is refused outright, because a short list
    silently shifts every later crop onto the wrong box.
    """
    if not isinstance(value, (list, tuple)):
        raise StudioError(
            ErrorCode.INTERNAL,
            f"The {engine_name} engine returned {type(value).__name__} instead of a list "
            f"of texts.",
            {"engine": engine_name, "got": type(value).__name__, "contract": True},
        )
    if len(value) != expected:
        raise StudioError(
            ErrorCode.INTERNAL,
            f"The {engine_name} engine returned {len(value)} text(s) for {expected} crop(s).",
            {"engine": engine_name, "expected": expected, "got": len(value),
             "contract": True,
             "hint": "the engine must answer once per crop, in order"},
        )
    out: list[Optional[str]] = []
    for index, item in enumerate(value):
        if item is None or isinstance(item, str):
            out.append(item)
            continue
        raise StudioError(
            ErrorCode.INTERNAL,
            f"The {engine_name} engine returned {type(item).__name__} at index {index} "
            f"instead of a string.",
            {"engine": engine_name, "index": index, "got": type(item).__name__,
             "contract": True},
        )
    return out


def _place(crops: Sequence[tuple[int, Image.Image]], answers: Sequence[Optional[str]],
           ) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    """Zip answers onto crop indices, turning a missing answer into a per-block error.

    Returns `(texts, errors)`. This is the single place that decides what "the engine said
    nothing about this crop" means in the document, so D39 (`null` always carries a reason,
    `""` means "read and empty") holds whichever engine produced the answer.
    """
    texts: dict[int, str] = {}
    errors: dict[int, dict[str, Any]] = {}
    for (index, _piece), answer in zip(crops, answers):
        if answer is None:
            errors[index] = _missing_text_error(index)
        else:
            texts[index] = answer
    return texts, errors


def _missing_text_error(index: int) -> dict[str, Any]:
    """Per-block error for "the engine gave no answer for this crop" (keeps D39 true)."""
    return {"code": ErrorCode.OCR_FAILED.value,
            "message": f"the OCR engine returned no text for crop {index}",
            "details": {"reason": "engine returned None for this crop"}}


def _with_bbox(det: Detection, bbox: BBox) -> Detection:
    return Detection(bbox=bbox, label=det.label, score=det.score)


def _ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


def run_file(path: Union[str, Any], config: Optional[AppConfig] = None, *,
             engines: Optional[PipelineEngines] = None, mock: bool = False,
             hf_token: Optional[str] = None, provider: Optional[str] = None,
             progress: Optional[ProgressCallback] = None,
             cancel_event: Optional[threading.Event] = None) -> PipelineResult:
    """One-shot convenience for scripts/tests: config -> engines -> pipeline -> result."""
    config = config or AppConfig.default()
    pair = engines or engines_from_config(config, mock=mock, hf_token=hf_token,
                                          provider=provider)
    return Pipeline(pair.detection, pair.ocr, config).run_file(
        path, progress=progress, cancel_event=cancel_event)


# ---------------------------------------------------------------------------
# The doors' shared `block_ids` check (P9.3)
# ---------------------------------------------------------------------------

def block_id_list(block_ids: Any, *, available: Optional[Sequence[int]] = None) -> list[int]:
    """Validate a `block_ids` argument; return the ids sorted and de-duplicated.

    Split out so a door with no document in hand yet (the MCP `reocr_blocks` precondition check)
    can still answer "is that even a list of ids?" before it reports what is missing. With
    ``available`` given, unknown ids are named alongside the ones that exist.
    """
    if isinstance(block_ids, (str, int)) or not isinstance(block_ids, Sequence):
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT, "block_ids must be a list of block ids.",
            {"got": type(block_ids).__name__,
             "hint": "e.g. [3] or [3, 7] — the ids from the result document"},
        )
    known = None if available is None else [block_id for block_id in available]
    wanted: list[int] = []
    for value in block_ids:
        if isinstance(value, bool) or not isinstance(value, int):
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT, f"Block id {value!r} is not an integer.",
                _ids_details(known),
            )
        wanted.append(value)
    if not wanted:
        raise StudioError(ErrorCode.INVALID_ARGUMENT,
                          "block_ids needs at least one block id.",
                          _ids_details(known))
    if known is not None:
        unknown = [i for i in wanted if i not in known]
        if unknown:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT, "Unknown block id(s) for this result.",
                {**_ids_details(known), "unknown": unknown,
                 "hint": "block ids come from the result document; they do not change between "
                         "a run and a re-read"},
            )
    # Reading order, de-duplicated: the same page read twice gives the same document.
    return sorted(set(wanted))


def _ids_details(known: Optional[Sequence[int]]) -> dict[str, Any]:
    """`{"available": […]}` when the document is known, `{}` when it is not (P9.3).

    A door that has no result yet says "that is not a list of ids"; one that has a document can
    also say which ids there are. Two different answers, and neither one invented.
    """
    return {} if known is None else {"available": list(known)}
