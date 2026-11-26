"""Domain types and error taxonomy shared by pipeline, GUI, CLI and MCP.

Pure stdlib. These dataclasses implement the data contracts in PLAN.md §2:
result JSON (schema 1.1), job progress, and the structured error model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

#: Result-document version we EMIT. Bumped to 1.1 in P8 for the additive ``stage`` key,
#: which is what makes a stage-1 (`detect`-only) document distinguishable from a full one
#: — before that, "OCR read nothing" and "OCR never ran" both looked like blocks with
#: empty text (P7 backlog). Documents without OCR are no longer a special case hiding in
#: the `text: null` column; they are a stage.
SCHEMA_VERSION = "1.2"

#: Versions we can still READ. A 1.0 document has no ``stage``; `from_dict` fills it in as
#: ``"full"``, which is the only thing a 1.0 document ever claimed to be.
SUPPORTED_RESULT_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "1.1", "1.2")

#: Closed vocabulary for `blocks[].flags` (result schema 1.2, P8.2b). Flags are *observations
#: about a block*, derived from data the document already carries — never a second opinion
#: about the text itself, and never a reason to drop or merge a block.
BLOCK_FLAGS: tuple[str, ...] = ("possible_truncation",)

#: A text this close to `ocr.max_new_tokens` probably ran into the generation cap. The ratio
#: is the model author's own (`processing_hayai.check_escalation` uses `>= 0.9 * max_new_tokens`),
#: which is why it is a flag and not a guess of ours.
TRUNCATION_RATIO = 0.9

#: Fraction of a block's area that another (larger) box must cover for us to call it the
#: container in `blocks[].contained_in`. 0.8 leaves room for the few-px slop the detector
#: produces without turning every overlapping pair into a parent/child pair.
CONTAINMENT_RATIO = 0.8

#: What a run actually did. `detect` = stage 1 only (boxes, no OCR attempted on any block);
#: `full` = detect + OCR (per-block failures still carry their own `error`);
#: `reocr` = P9.3 — an existing document's boxes, some of them read again. Still a *whole*
#: document (D81): every block is present, unselected ones carry their previous text.
RESULT_STAGES = ("detect", "full", "reocr")

#: Config documents have their own version line (decision D27). Bumped to 1.1 for the
#: additive on-premises keys ``paths.models_dir`` / ``paths.offline``; 1.0 documents stay
#: readable forever (additive change, defaults preserve the documented behaviour), which
#: is why the *result* document above stays at 1.0 — its shape did not change.
CONFIG_SCHEMA_VERSION = "1.2"
SUPPORTED_CONFIG_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "1.1", "1.2")

DETECTION_REPO = "deepghs/AnimeText_yolo"
OCR_REPO = "JustANormalTinkerer/hayai-ocr-v2.7-supernova"
# Pinned revision of the OCR repo (trust_remote_code reproducibility, decision D12).
# Bumped 2026-11-20 from 5289b2cfdfd3… to the repo HEAD that the on-premises bundle
# carries: the newer snapshot updates `processing_hayai.py` (the very API Part 3 calls)
# and adds `yolo26m-text.pt`. Keeping the older pin would make a bundled, fully offline
# install pull a *different* snapshot from the Hub (decision D26).
#: Bumped again 2026-11-22 to `b00317ed69fc…` (repo HEAD, = the re-vendored snapshot):
#: the weights were merged/fine-tuned for edge cases. The snapshot's Python and tokenizer
#: files are byte-identical to the previous revision, so the Part 3 API is unaffected.
OCR_REVISION = "b00317ed69fccee5f787187d913fcf8a96f37f65"
#: The detector's single class name. Verified against the repo's own
#: `<model>/labels.json` for all five variants: `["text_block"]` (an earlier draft of
#: PLAN.md had `text_bblock`, which was a typo — caught by tests/test_models.py).
TEXT_LABEL = "text_block"

IMAGE_SOURCES = ("file", "clipboard", "mcp-base64", "mcp-path")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorCode(str, Enum):
    NO_IMAGE = "NO_IMAGE"
    BUSY = "BUSY"
    CANCELLED = "CANCELLED"
    INVALID_IMAGE = "INVALID_IMAGE"
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    HF_AUTH_REQUIRED = "HF_AUTH_REQUIRED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    ENGINE_LOAD_FAILED = "ENGINE_LOAD_FAILED"
    OCR_FAILED = "OCR_FAILED"
    IO_ERROR = "IO_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    INTERNAL = "INTERNAL"


#: Process exit codes for headless/CLI use (0 = success).
EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.INTERNAL: 1,
    ErrorCode.INVALID_CONFIG: 2,
    ErrorCode.INVALID_ARGUMENT: 2,
    ErrorCode.NO_IMAGE: 3,
    ErrorCode.BUSY: 4,
    ErrorCode.CANCELLED: 5,
    ErrorCode.HF_AUTH_REQUIRED: 6,
    ErrorCode.DOWNLOAD_FAILED: 7,
    ErrorCode.ENGINE_LOAD_FAILED: 8,
    ErrorCode.OCR_FAILED: 9,
    ErrorCode.IO_ERROR: 10,
    ErrorCode.INVALID_IMAGE: 11,
    ErrorCode.MODEL_NOT_FOUND: 12,
    ErrorCode.NOT_IMPLEMENTED: 20,
}


class StudioError(Exception):
    """Structured error: {code, message, details?} everywhere (GUI/CLI/MCP)."""

    def __init__(
        self,
        code: Union[ErrorCode, str],
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = str(message)
        self.details = details
        super().__init__(f"[{self.code.value}] {self.message}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details is not None:
            out["details"] = self.details
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudioError":
        """Parse a {code, message, details} document — tolerant, because it is a *reader*.

        The constructor stays strict (a typo in a code we raise ourselves should be loud), but a
        document written by another — possibly newer — version of the app must not blow up on
        the way in: an unknown code becomes INTERNAL with the original string kept in
        `details.unknown_code`, and a missing message is said out loud instead of raising KeyError
        three frames away (P8.4).
        """
        if not isinstance(data, dict):
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              f"An error document must be an object, got {type(data).__name__}.",
                              {"got": type(data).__name__})
        raw_code = data.get("code")
        details = data.get("details")
        message = str(data.get("message") or "")
        try:
            code = ErrorCode(raw_code)
        except ValueError:
            code = ErrorCode.INTERNAL
            merged = {"unknown_code": raw_code}
            merged.update(details if isinstance(details, dict) else {"details": details})
            details = merged
            message = message or f"Unknown error code {raw_code!r}."
        return cls(code, message or "(the error document carried no message)", details)


def _build(cls: type, data: Any, factory: Callable[[Any], Any]):
    """Run a `from_dict` factory, translating malformed input into INVALID_ARGUMENT.

    Every public `from_dict` goes through here so that a payload coming from a file,
    the clipboard or an MCP client can only ever produce the structured error the
    GUI/CLI/MCP promise (PLAN 2.3) — never a bare KeyError/TypeError that would
    surface as INTERNAL with a cryptic message.
    """
    try:
        return factory(data)
    except StudioError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        keys = sorted(data) if isinstance(data, dict) else None
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Malformed {cls.__name__} payload ({type(exc).__name__}: {exc}).",
            {"available_keys": keys} if keys else None,
        ) from exc


# ---------------------------------------------------------------------------
# Geometry & detections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Integer pixel bounding box, xyxy, original-image space."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, int(round(getattr(self, name))))

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def clamped(self, image_w: int, image_h: int) -> "BBox":
        return BBox(
            max(0, min(self.x0, image_w)),
            max(0, min(self.y0, image_h)),
            max(0, min(self.x1, image_w)),
            max(0, min(self.y1, image_h)),
        )

    def contained_ratio(self, outer: "BBox") -> float:
        """Fraction of *this* box's area that `outer` covers (0.0 … 1.0).

        Asymmetric on purpose: `a.contained_ratio(b)` answers "is a inside b?", and
        `b.contained_ratio(a)` answers the same question about b. Used for the
        `contained_in` relation (P8.2b), where a nested fragment and a page-wide box are
        very different cases even when they overlap each other completely.
        """
        if self.area <= 0:
            return 0.0
        ix0, iy0 = max(self.x0, outer.x0), max(self.y0, outer.y0)
        ix1, iy1 = min(self.x1, outer.x1), min(self.y1, outer.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return ((ix1 - ix0) * (iy1 - iy0)) / self.area

    def expanded(self, padding: int, image_w: int, image_h: int) -> "BBox":
        """Grow by `padding` px on each side, clamped to the image."""
        return BBox(
            self.x0 - padding, self.y0 - padding,
            self.x1 + padding, self.y1 + padding,
        ).clamped(image_w, image_h)

    def to_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBox":
        return _build(cls, data,
                      lambda d: cls(d["x0"], d["y0"], d["x1"], d["y1"]))

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> "BBox":
        return cls(*t)


@dataclass(frozen=True)
class Detection:
    """One raw detector output (stage 1 of the pipeline)."""

    bbox: BBox
    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"bbox": self.bbox.to_dict(), "label": self.label, "score": self.score}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        return _build(cls, data,
                      lambda d: cls(BBox.from_dict(d["bbox"]), d["label"], float(d["score"])))


@dataclass(frozen=True)
class TextBlock:
    """One finalized block: bounding box associated with its OCR text."""

    id: int
    bbox: BBox
    label: str
    score: float
    text: Optional[str]
    reading_order: int
    color: str
    error: Optional[dict[str, Any]] = None
    #: Observations about this block (`BLOCK_FLAGS`), e.g. `possible_truncation`. Empty in
    #: the common case; 1.0/1.1 documents read as empty (schema 1.2, P8.2b).
    flags: tuple[str, ...] = ()
    #: id of the smallest strictly larger box that contains >= `CONTAINMENT_RATIO` of this
    #: one, or None. Geometry only — it says "a consumer should look at block N before
    #: reading this text on its own", nothing about which text to keep.
    contained_in: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "label": self.label,
            "score": self.score,
            "text": self.text,
            "reading_order": self.reading_order,
            "color": self.color,
            "error": self.error,
            "flags": list(self.flags),
            "contained_in": self.contained_in,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextBlock":
        def make(d: dict[str, Any]) -> "TextBlock":
            return cls(
                id=int(d["id"]),
                bbox=BBox.from_dict(d["bbox"]),
                label=d["label"],
                score=float(d["score"]),
                text=d["text"],
                reading_order=int(d["reading_order"]),
                color=d["color"],
                error=d.get("error"),
                # 1.0/1.1 documents have neither key; `null` for contained_in is also legal
                # input and means the same thing as absent (no container).
                flags=tuple(d.get("flags") or ()),
                contained_in=(int(d["contained_in"]) if d.get("contained_in") is not None
                              else None),
            )
        return _build(cls, data, make)


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    """Local-timezone ISO-8601 timestamp (seconds precision)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class ImageInfo:
    source: str  # one of IMAGE_SOURCES
    file_name: Optional[str]
    width: int
    height: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageInfo":
        return _build(cls, data, lambda d: cls(**{f.name: d[f.name] for f in fields(cls)}))


@dataclass(frozen=True)
class DetectionInfo:
    repo: str
    model: str
    iou_threshold: float
    score_threshold: float
    allow_dynamic: bool
    backend: str = "onnxruntime"

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionInfo":
        # `backend` is optional here (older result files); the rest are required by
        # the constructor and _build() turns a missing one into INVALID_ARGUMENT.
        return _build(cls, data,
                      lambda d: cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d}))


@dataclass(frozen=True)
class OcrInfo:
    repo: str
    device: str  # resolved device, e.g. "cuda:0" or "cpu"
    max_patches: Union[str, int]  # "auto" | 128 | 192 | 256 | 384 | 512
    batch_size: int
    max_new_tokens: int
    crop_padding_px: int
    normalize_nfkc: bool

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrInfo":
        return _build(cls, data, lambda d: cls(**{f.name: d[f.name] for f in fields(cls)}))


@dataclass(frozen=True)
class TimingMs:
    detect: int
    ocr: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return {"detect": self.detect, "ocr": self.ocr, "total": self.total}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimingMs":
        return _build(cls, data, lambda d: cls(**{f.name: int(d[f.name]) for f in fields(cls)}))


@dataclass(frozen=True)
class PipelineResult:
    """The primary output document (PLAN.md §2.1, result schema 1.1)."""

    image: ImageInfo
    detection: DetectionInfo
    ocr: OcrInfo
    timing_ms: TimingMs
    blocks: tuple[TextBlock, ...] = ()
    #: `RESULT_STAGES`: "full" (detect + OCR), "detect" (boxes only) or "reocr" (P9.3).
    stage: str = "full"
    created_at: str = field(default_factory=now_iso)
    schema_version: str = SCHEMA_VERSION

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def ocr_ran(self) -> bool:
        """Did this document have text read into it? (1.0 documents read as True.)

        `reocr` counts: it is a whole document whose blocks were read, only some of them again.
        Reading this as False would tell a caller there is nothing to show, in the one case where
        somebody had gone to extra trouble to get the text right.
        """
        return self.stage in ("full", "reocr")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "stage": self.stage,
            "image": self.image.to_dict(),
            "detection": self.detection.to_dict(),
            "ocr": self.ocr.to_dict(),
            "timing_ms": self.timing_ms.to_dict(),
            "blocks": [b.to_dict() for b in self.blocks],
            "block_count": self.block_count,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineResult":
        return _build(cls, data, _pipeline_result_from_dict)


def _pipeline_result_from_dict(data: dict[str, Any]) -> PipelineResult:
    version = data["schema_version"]
    if version not in SUPPORTED_RESULT_SCHEMA_VERSIONS:
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Unsupported result schema version {version!r}.",
            {"supported": list(SUPPORTED_RESULT_SCHEMA_VERSIONS)},
        )
    # 1.0 documents predate `stage`; they were always full runs (see above).
    stage = data.get("stage", "full")
    if stage not in RESULT_STAGES:
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Unknown result stage {stage!r}.",
            {"expected": list(RESULT_STAGES)},
        )
    return PipelineResult(
        image=ImageInfo.from_dict(data["image"]),
        detection=DetectionInfo.from_dict(data["detection"]),
        ocr=OcrInfo.from_dict(data["ocr"]),
        timing_ms=TimingMs.from_dict(data["timing_ms"]),
        blocks=tuple(TextBlock.from_dict(b) for b in data["blocks"]),
        stage=stage,
        created_at=data["created_at"],
        schema_version=version,
    )


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JobProgress:
    job_id: str
    state: JobState
    stage: str  # e.g. "load_model" | "detect" | "crop" | "ocr" | "assemble"
    percent: float  # 0..100
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": JobState(self.state).value,
            "stage": self.stage,
            "percent": max(0.0, min(100.0, float(self.percent))),
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Deterministic block palette (same index -> same color, always)
# ---------------------------------------------------------------------------

BLOCK_PALETTE: tuple[str, ...] = (
    "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231",
    "#911EB4", "#46F0F0", "#F032E6", "#BCF60C", "#FABEBE",
    "#008080", "#E6BEFF", "#9A6324", "#FFFAC8", "#800000",
    "#AAFFC3", "#808000", "#FFD8B1", "#000075", "#808080",
)


def color_for_index(index: int) -> str:
    return BLOCK_PALETTE[index % len(BLOCK_PALETTE)]
