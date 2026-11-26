"""P7 - pydantic output models for the MCP tools.

Two schemas, one truth:

* the **input** schema of a tool is derived by the SDK from the tool function signature
  (`mcp/tools.py`) - there is no second copy to drift;
* the **output** schema is these models. Each tool is annotated with one, so the SDK
  publishes `outputSchema` on the wire *and* validates the structured content it sends;
  `schemas/mcp-tools.json` dumps the same models, so a harness can validate a response
  offline without importing the app.

Deliberate `dict[str, Any]` holes: `config` and `result_document` are the §2.2 / §2.1
documents. They already have their own versioned JSON Schemas (`config.schema.json`,
`result.schema.json`) validated natively in `core/`; mirroring them again here would be a
second definition to keep in sync. Tests validate those two against their real schemas.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    """Common config: forbid unknown fields (a typo in a response is a bug we want loud)."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Small shared shapes
# ---------------------------------------------------------------------------

class BBoxModel(_Model):
    x0: int
    y0: int
    x1: int
    y1: int


class ImageInfoModel(_Model):
    """`image` section of the result document (PLAN 2.1)."""

    source: str
    file_name: Optional[str] = None
    width: int
    height: int
    sha256: str


class JobStateModel(_Model):
    job_id: str = ""
    #: "run" (produced a result document) or "warmup" (P8.3: engine warm-up, no result).
    #: Without it a caller cannot tell a finished warm-up from a finished page, because both
    #: end DONE and `has_result` still describes the *previous* page.
    kind: str = "run"
    state: str = "idle"
    stage: str = "idle"
    percent: float = 0.0
    message: str = ""
    busy: bool = False
    has_result: bool = False
    block_count: int = 0
    error: Optional[dict[str, Any]] = None


class EngineModel(_Model):
    name: str
    backend: str = ""
    device: str = ""
    #: "unloaded" | "loading" | "ready" | "error" — the engine's own state, so `loaded` is no
    #: longer the only clue (and no longer the only one that was ever wrong: see P8.3).
    state: str = "unloaded"
    loaded: bool = False
    models: list[str] = Field(default_factory=list)
    #: Inferences completed since load, and whether the engine is *warm* (loaded and at least
    #: one inference done). This is how `--preload` becomes observable (P8.3).
    inferences: int = 0
    warm: bool = False
    #: The engine's own note — why it is in the state it is in. Empty when all is ordinary.
    detail: str = ""
    #: True when the *list* of variants came from the compiled presets + the bundle rather than
    #: the repo (offline, timed out, or refused). The variants on it are still usable; the claim
    #: "these are the variants that exist" is not. Nothing else on this door says which of the
    #: two the caller is looking at, which is why the flag travels, not just the prose (P9.2).
    model_list_is_fallback: bool = False


class McpInfoModel(_Model):
    running: bool = False
    transport: str = "none"
    host: str = ""
    port: int = 0
    url: str = ""
    auth_required: bool = False
    endpoint: str = ""
    started_at: str = ""
    calls: int = 0
    errors: int = 0
    #: Cautions raised at start-up (P8): a non-loopback bind, with or without a token.
    warnings: list[str] = Field(default_factory=list)


class ErrorModel(_Model):
    """The §2.3 error taxonomy, as it appears in `isError` results."""

    code: str
    message: str
    details: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class GetStatusResult(_Model):
    app: str
    version: str
    result_schema_version: str
    config_schema_version: str
    busy: bool
    mock: bool
    offline: bool
    job: JobStateModel
    image: Optional[ImageInfoModel] = None
    engines: list[EngineModel] = Field(default_factory=list)
    models: dict[str, Any] = Field(default_factory=dict)
    mcp: McpInfoModel = Field(default_factory=McpInfoModel)
    gui: bool = False
    has_result: bool = False
    block_count: int = 0


class ModelEntry(_Model):
    name: str
    known: bool = True
    params_m: Optional[float] = None
    flops_g: Optional[float] = None
    f1: Optional[float] = None
    recommended_score_threshold: Optional[float] = None
    source: str = "hf"
    bundled: bool = False
    cached: bool = False
    loaded: bool = False
    available: bool = True
    size_mb: Optional[float] = None
    note: str = ""


class ListModelsResult(_Model):
    detection_repo: str
    ocr_repo: str
    ocr_revision: str
    default_model: str
    selected_model: str
    offline: bool
    models: list[ModelEntry]
    loaded_model: Optional[str] = None
    ocr: dict[str, Any] = Field(default_factory=dict)


class GetConfigResult(_Model):
    config: dict[str, Any]
    schema_version: str


class SetConfigResult(_Model):
    config: dict[str, Any]
    changed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: Changed keys the running process cannot re-apply to itself (P8.8): the MCP transport
    #: keys are read once, when the server is built. The new value *is* stored — an export is
    #: faithful and the next start uses it — but the socket you are talking to has not moved.
    restart_required: list[str] = Field(default_factory=list)


class ExportConfigResult(_Model):
    config: dict[str, Any]
    path: Optional[str] = None
    bytes: Optional[int] = None
    #: Secret keys replaced by the `<redacted>` placeholder when `redact_secrets` was asked
    #: for (P8). Empty unless something was actually hidden.
    redacted: list[str] = Field(default_factory=list)


class ImportConfigResult(_Model):
    config: dict[str, Any]
    changed: list[str] = Field(default_factory=list)
    source: str
    #: Secret keys that arrived as `<redacted>` and therefore kept their live value (P8).
    kept: list[str] = Field(default_factory=list)
    #: See `SetConfigResult.restart_required`.
    restart_required: list[str] = Field(default_factory=list)


class LoadImageResult(_Model):
    image: ImageInfoModel
    loaded: bool = True


class ImageInfoResult(_Model):
    image: ImageInfoModel


class DetectionItem(_Model):
    id: int
    bbox: BBoxModel
    label: str
    score: float


class DetectResult(_Model):
    detections: list[DetectionItem]
    count: int
    model: str
    iou_threshold: float
    score_threshold: float
    allow_dynamic: bool
    timing_ms: int
    job_id: str = ""
    dropped_small: int = 0


class RunPipelineResult(_Model):
    result: dict[str, Any] = Field(default_factory=dict,
                                   description="The full result document (result.schema.json).")
    job_id: str = ""
    block_count: int = 0
    timing_ms: dict[str, int] = Field(default_factory=dict)
    annotated_image: Optional[dict[str, Any]] = None
    progress_events: int = 0


class GetResultResult(_Model):
    result: dict[str, Any]
    block_count: int
    filtered: bool = False


class SaveResultResult(_Model):
    path: str
    bytes: int
    block_count: int


class AnnotatedImageMeta(_Model):
    """Metadata for the PNG that travels as MCP *image content*; the pixels are never part
    of the structured content (a full-resolution page would swamp a chat context)."""

    mime_type: str = "image/png"
    width: int
    height: int
    bytes: int
    blocks: int
    scale: float = 1.0


class OverlayResult(_Model):
    gui: dict[str, Any]
    overlay_visible: bool
    labels_visible: bool
    fill_opacity: float


class ZoomResult(_Model):
    zoom_mode: str
    zoom_percent: int
    scale: float


class HighlightResult(_Model):
    ok: bool
    block_id: int
    found: bool
    duration_ms: int
    text_preview: str = ""


class ResetResult(_Model):
    ok: bool = True
    cleared: list[str] = Field(default_factory=list)


class CancelResult(_Model):
    cancelled: bool
    job_id: str = ""
    state: str = ""


class UnloadResult(_Model):
    unloaded: list[str] = Field(default_factory=list)
    busy: bool = False


class WarmEnginesResult(_Model):
    """The warm-up is a job, so this answers with a job handle, not with a wait (P8.3)."""

    job_id: str = ""
    kind: str = "warmup"
    state: str = "running"
    busy: bool = True
    probe: bool = True
    note: str = "poll get_status (job.kind == 'warmup'); get_status.engines[].warm is the proof"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class McpLogResult(_Model):
    entries: list[dict[str, Any]] = Field(default_factory=list)
    calls: int = 0
    errors: int = 0
    capacity: int = 0


class ModelsResource(_Model):
    """`models://list` - the same payload as `list_models` (one shape, two doors)."""

    detection_repo: str
    ocr_repo: str
    models: list[ModelEntry]
    default_model: str
    selected_model: str
    offline: bool = False
