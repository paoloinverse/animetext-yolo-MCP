"""P7 - the MCP tools (PLAN §4) as thin wrappers over `StudioService`.

Every tool here does four things and nothing else:

1. validate/unpack its arguments (the SDK validates them against the derived input schema);
2. hand one `StudioService` method to the app-state thread (`service.aexecute`);
3. shape the answer into `CallToolResult` — structured JSON validated against the tool's
   output model, plus a short human-readable text block;
4. on failure, return `isError: true` with the §2.3 `{code, message, details}` object as
   structured content, so a harness can branch on `code` instead of parsing prose.

The activity entry for the call is written by `StudioService.execute` — one per call,
success or failure — which is why no tool mentions the log, the widgets or the status bar.
"""
from __future__ import annotations

import asyncio
import base64
import json as _json
import time
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field
from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations

try:                                              # the SDK's context type (v2 name; v1 fallback)
    from mcp.server.mcpserver.context import Context
except ImportError:                               # pragma: no cover - mcp<2
    from mcp.server.fastmcp import Context        # type: ignore[assignment]

from ..core.types import (
    CONFIG_SCHEMA_VERSION, ErrorCode, StudioError,
    SCHEMA_VERSION as RESULT_SCHEMA_VERSION,
)
from .schemas import (
    AnnotatedImageMeta, CancelResult, DetectResult, ExportConfigResult, GetConfigResult,
    GetResultResult, GetStatusResult, HighlightResult, ImageInfoResult, ImportConfigResult,
    ListModelsResult, LoadImageResult, OverlayResult, ResetResult, RunPipelineResult,
    SaveResultResult, SetConfigResult, UnloadResult, WarmEnginesResult, ZoomResult,
)
from .service import (           # MAX_BASE64_* are re-exported for callers that look here
    MAX_BASE64_BYTES, MAX_BASE64_CHARS, StudioService,
)

__all__ = ["TOOL_GUI_EFFECT", "register_tools", "tool_catalog"]

#: What each tool does to the visible window — documented once, published in the catalog,
#: and used by `gui/activity.py` to decide which widget to flash.
TOOL_GUI_EFFECT: dict[str, dict[str, Any]] = {
    "get_status":            {"flash": [], "effect": "activity log + status bar"},
    "list_models":           {"flash": ["model_combo"], "effect": "activity log; model list"},
    "get_config":            {"flash": [], "effect": "activity log"},
    "set_config":            {"flash": ["controls"], "effect": "affected widgets update + flash"},
    "export_config":         {"flash": ["export_button"], "effect": "Export button flash"},
    "import_config":         {"flash": ["controls"], "effect": "all controls update + flash"},
    "load_image":            {"flash": ["preview"], "effect": "input preview shows the image"},
    "get_image_info":        {"flash": [], "effect": "activity log"},
    "detect":                {"flash": ["canvas"], "effect": "boxes drawn on the output canvas"},
    "run_pipeline":          {"flash": ["canvas", "json"],
                              "effect": "progress bar, boxes appear, JSON pane fills"},
    "reocr_blocks":          {"flash": ["progress", "canvas", "json"],
                              "effect": "progress bar; the named blocks change text, the rest "
                                       "of the page is untouched"},
    "cancel_job":            {"flash": ["progress"], "effect": "progress bar aborts, log"},
    "get_result":            {"flash": ["json"], "effect": "JSON pane flash"},
    "save_result":           {"flash": ["save_button"], "effect": "Save button flash"},
    "get_annotated_image":   {"flash": ["canvas"], "effect": "canvas flash"},
    "set_overlay_visible":   {"flash": ["canvas"], "effect": "canvas toggles live"},
    "set_zoom":              {"flash": ["canvas"], "effect": "canvas zooms"},
    "highlight_block":       {"flash": ["canvas"], "effect": "box pulses + hover text panel"},
    "reset":                 {"flash": ["canvas", "preview", "json"],
                              "effect": "clears image, result and canvas"},
    "unload_engines":        {"flash": ["badges"], "effect": "activity log; engines freed"},
    "warm_engines":          {"flash": ["badges"],
                              "effect": "progress bar runs the `warmup` job; engines badge "
                                       "refreshes when it finishes"},
}

_READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)
_MUTATING = ToolAnnotations(open_world_hint=False)

#: The base64 size caps are in `service.py` (next to the code that enforces them) and
#: re-exported from there — see `MAX_BASE64_BYTES` below in the `.service` import.


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def ok(model: type[BaseModel], payload: Any, *, extra_content: Optional[list[Any]] = None
       ) -> CallToolResult:
    """Validate `payload` against the tool's output model and build the success result."""
    validated = model.model_validate(payload)
    data = validated.model_dump(mode="json")
    text = _summary_text(model, data)
    content: list[Any] = [TextContent(type="text", text=text)]
    content.extend(extra_content or [])
    return CallToolResult(content=content, structured_content=data)


def failure(exc: StudioError) -> CallToolResult:
    """`isError: true` carrying {code, message, details} as structured content (§2.3)."""
    payload = exc.to_dict()
    return CallToolResult(
        content=[TextContent(type="text", text=_json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
        is_error=True,
    )


# ---------------------------------------------------------------------------
# Errors the SDK produces before our handlers run
# ---------------------------------------------------------------------------

class ErrorShapeMiddleware:
    """Give every failed tool call the §2.3 shape — including the ones we never saw.

    The SDK validates arguments *before* calling a tool, so a mistyped, missing or unknown
    argument — and a call to a tool that does not exist — never reaches `call()`. It comes back
    as `isError: true` with a text message and **no** `structuredContent`. That is a reasonable
    default for a general-purpose SDK, and a bad one for this app: PLAN §2.3 promises a `code`
    on every tool error, argument mistakes are the single most common failure a harness will
    hit, and the GUI log did not show rejected calls at all — a client could hammer the server
    with garbage and the user would see nothing.

    So: after the fact, an `isError` result that carries no structured content is rewritten into
    `{code: "INVALID_ARGUMENT", message, details}` and recorded as one activity entry. Our own
    failures already carry structured content, so they pass through untouched and are logged
    exactly once (by `StudioService.execute`).

    The middleware list is the SDK's documented hook but marked provisional, so everything here
    is best-effort: `install_error_shaping()` reports whether it took, and a rewrite that goes
    wrong returns the original result. The server must never fail because its error cosmetics did.
    """

    def __init__(self, service: StudioService,
                 tool_names: Optional[Callable[[], list[str]]] = None) -> None:
        self._service = service
        self._tool_names = tool_names or (lambda: [])
        self.rewritten = 0

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        result = await call_next(ctx)
        try:
            return self._rewrite(ctx, result)
        except Exception:                                       # noqa: BLE001
            return result                       # never trade a readable error for no error

    # -- rewriting ----------------------------------------------------------

    def _known_tools(self) -> list[str]:
        """The server's tool names, resolved lazily (registration may still be in progress)."""
        try:
            return sorted(self._tool_names())
        except Exception:                                       # noqa: BLE001
            return []

    def _rewrite(self, ctx: Any, result: Any) -> Any:
        """Rewrite one unstructured `isError` tool result; anything else passes through.

        The chain hands back the *wire* shape (a plain dict, because the runner serialises the
        handler's `CallToolResult` before middleware returns), so both a dict and a
        `CallToolResult` are handled — whichever the SDK happens to be passing in this version.
        """
        if getattr(ctx, "method", "") != "tools/call":
            return result
        body = _result_as_dict(result)
        if body is None or not body.get("isError"):
            return result
        if body.get("structuredContent"):                        # ours already: leave it alone
            return result
        params = getattr(ctx, "params", None)
        params = params if isinstance(params, dict) else {}
        tool = str(params.get("name") or "")
        text = " ".join(str(block.get("text") or "")
                        for block in (body.get("content") or [])
                        if isinstance(block, dict) and block.get("type") == "text")[:600]
        exc = unshaped_tool_error(tool, text, known_tools=self._known_tools())
        payload = exc.to_dict()
        self.rewritten += 1
        self._service.activity.record(
            tool or "unknown_tool", ok=False, args=params.get("arguments") or {},
            code=exc.code.value, message=exc.message,
            transport=self._service.mcp_transport,
        )
        if isinstance(result, CallToolResult):        # keep the shape the caller handed us
            return CallToolResult(content=[TextContent(type="text",
                                                       text=_json.dumps(payload,
                                                                        ensure_ascii=False))],
                                  structured_content=payload, is_error=True)
        return {**body,
                "content": [{"type": "text",
                             "text": _json.dumps(payload, ensure_ascii=False)}],
                "structuredContent": payload,
                "isError": True}


def _result_as_dict(result: Any) -> Optional[dict[str, Any]]:
    """A tool result as a dict of its wire fields, or None if it is not a tool result."""
    if isinstance(result, dict):
        return result
    if isinstance(result, CallToolResult):
        return result.model_dump(by_alias=True, exclude_none=True)
    return None


def unshaped_tool_error(tool: str, text: str,
                        known_tools: Optional[list[str]] = None) -> StudioError:
    """The §2.3 form of an error the SDK produced for us (P8.4).

    Pydantic's text is multi-line and verbose; JSON-RPC asks for a concise message, so it is
    folded onto one line and capped. Nothing is invented here — the wording stays the SDK's, so
    a client reading it sees the same complaint the validator actually made.
    """
    folded = " ".join((text or "").split())[:400]
    if folded.lower().startswith("unknown tool") or not tool:
        return StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Unknown tool {tool!r}." if tool else "Unknown tool.",
            {"reason": "unknown_tool", "tool": tool,
             # Twenty names is a few hundred bytes, and this is the one error where the client
             # cannot guess its way out: it asked for something that is not there.
             "known_tools": list(known_tools or []),
             "hint": "call list_tools(); the catalog is also at schemas/mcp-tools.json"},
        )
    return StudioError(
        ErrorCode.INVALID_ARGUMENT,
        folded or f"{tool} rejected the arguments.",
        {"reason": "argument_validation", "tool": tool,
         "hint": "the tool's inputSchema is in list_tools() / schemas/mcp-tools.json"},
    )


def install_error_shaping(server: Any, service: StudioService) -> bool:
    """Attach `ErrorShapeMiddleware` to `server`; True when the SDK allowed it."""
    chain = getattr(server, "middleware", None)
    if chain is None or not hasattr(chain, "append"):
        return False
    try:
        chain.append(ErrorShapeMiddleware(service,
                                           tool_names=lambda: [tool.name
                                                               for tool in _registered_tools(server)]))
    except Exception:                                           # noqa: BLE001
        return False
    return True


def _summary_text(model: type[BaseModel], data: dict[str, Any]) -> str:
    """The text block: the JSON, trimmed where a base64 blob would swamp a chat window."""
    trimmed = dict(data)
    if model is AnnotatedImageMeta:
        trimmed.pop("base64_png", None)
    return _json.dumps(trimmed, ensure_ascii=False)


async def call(service: StudioService, name: str, fn: Any, args: dict[str, Any],
               model: type[BaseModel], *, timeout: Optional[float] = None) -> CallToolResult:
    """One dispatched service call, logged, shaped, and error-mapped."""
    try:
        payload = await service.aexecute(name, fn, args=args, timeout=timeout)
    except StudioError as exc:
        return failure(exc)
    return ok(model, payload)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_tools(server: Any, service: StudioService) -> None:
    """Register every §4 tool on `server` (an `MCPServer`), bound to `service`."""

    # -- status / models ------------------------------------------------------

    @server.tool(name="get_status", description=(
        "App state in one call: version, job state and progress, loaded engines and device, "
        "the current image, the on-premises model report and MCP counters."),
        annotations=_READ_ONLY)
    async def get_status() -> GetStatusResult:
        return await call(service, "get_status", service.snapshot, {}, GetStatusResult)

    @server.tool(name="list_models", description=(
        "The five AnimeText YOLO variants with params/F1/recommended thresholds, plus "
        "bundled / loaded / available flags from the on-premises model bundle."),
        annotations=_READ_ONLY)
    async def list_models() -> ListModelsResult:
        return await call(service, "list_models", service.models, {}, ListModelsResult)

    # -- config ---------------------------------------------------------------

    @server.tool(name="get_config", description=(
        # Interpolated from the types module rather than typed out: this string said "1.1" for a
        # document that had already moved to 1.2, and a catalog that misstates a version is worse
        # than one that says nothing about it (P8.2b). Same for `get_result` below.
        f"The live configuration document (config.schema.json, "
        f"schema_version {CONFIG_SCHEMA_VERSION})."),
        annotations=_READ_ONLY)
    async def get_config() -> GetConfigResult:
        return await call(service, "get_config", service.config_snapshot, {}, GetConfigResult)

    @server.tool(name="set_config", description=(
        "Apply a partial config: {\"detection\": {\"score_threshold\": 0.3}}. Sections and keys "
        "you leave out keep their current values; invalid values fail with INVALID_CONFIG and "
        "change nothing. Returns the full config, the list of changed key paths, and "
        "`restart_required`: changed keys the running process cannot re-apply (the MCP bind "
        "address, port, transport and bearer token are read once, at server start). The value "
        "is stored either way — the socket you are talking to has not moved."),
        annotations=_MUTATING)
    async def set_config(
        patch: Annotated[dict[str, Any],
                         Field(description="Partial config object (one or more sections).")],
    ) -> SetConfigResult:
        return await call(service, "set_config",
                          lambda: service.apply_config_patch(patch), {"patch": patch},
                          SetConfigResult)

    @server.tool(name="export_config", description=(
        "Return the live config; if `path` is given, also write it there (parents created). "
        "`redact_secrets` replaces secret values (mcp.auth_token) with the `<redacted>` "
        "placeholder in the file and in the reply, and lists what it hid."),
        annotations=_MUTATING)
    async def export_config(
        path: Annotated[Optional[str],
                        Field(description="Optional file to write the JSON to.")] = None,
        redact_secrets: Annotated[bool, Field(description=(
            "Replace secrets with the <redacted> placeholder. Importing such a file keeps the "
            "app's current token instead of setting the literal placeholder."))] = False,
    ) -> ExportConfigResult:
        return await call(service, "export_config",
                          lambda: service.export_config(path, redact_secrets=redact_secrets),
                          {"path": path, "redact_secrets": redact_secrets},
                          ExportConfigResult)

    @server.tool(name="import_config", description=(
        "Replace the whole configuration from `json` (a config document) or from `path` — "
        "exactly one of the two. Widgets, canvas and pipeline all follow it; "
        "`restart_required` lists any MCP transport key the file changed that only takes "
        "effect on the next start."),
        annotations=_MUTATING)
    async def import_config(
        json: Annotated[Optional[dict[str, Any]],
                        Field(description="Full or partial configuration document.")] = None,
        path: Annotated[Optional[str],
                        Field(description="Path to a configuration file.")] = None,
    ) -> ImportConfigResult:
        return await call(service, "import_config",
                          lambda: service.import_config(json, path),
                          {"json": json, "path": path}, ImportConfigResult)

    # -- image ----------------------------------------------------------------

    @server.tool(name="load_image", description=(
        "Load a page into the app from `path`, from `base64_png_jpg`, or from the desktop "
        "clipboard (`from_clipboard`: GUI only) — exactly one of the three."),
        annotations=_MUTATING)
    async def load_image(
        path: Annotated[Optional[str], Field(description="Image file on the app's machine.")] = None,
        base64_png_jpg: Annotated[Optional[str], Field(
            description="PNG/JPEG/WebP bytes, base64 (a data: URL is accepted).")] = None,
        from_clipboard: Annotated[bool, Field(
            description="Take the image from the desktop clipboard (needs the GUI).")] = False,
        file_name: Annotated[Optional[str],
                             Field(description="Name to record for a base64 payload.")] = None,
    ) -> LoadImageResult:
        # The size cap lives in the service, not here: a raise at this point happens *before*
        # the call is dispatched, so it would escape `call()`, produce no activity entry and
        # reach the client as an unshaped exception instead of the §2.3 error (P8.4).
        return await call(service, "load_image",
                          lambda: service.load_image(path, base64_png_jpg,
                                                      bool(from_clipboard), file_name),
                          {"path": path, "base64_png_jpg": base64_png_jpg,
                           "from_clipboard": from_clipboard, "file_name": file_name},
                          LoadImageResult)

    @server.tool(name="get_image_info", description=(
        "The current page: source, file name, size and sha256. NO_IMAGE when nothing is loaded."),
        annotations=_READ_ONLY)
    async def get_image_info() -> ImageInfoResult:
        return await call(service, "get_image_info", service.describe_image, {},
                          ImageInfoResult)

    # -- running --------------------------------------------------------------

    @server.tool(name="detect", description=(
        "Stage 1 only: detect text blocks and return the boxes, skipping OCR. Runs the same "
        "single-flight job path as run_pipeline (so it is BUSY while a job is running) and "
        "shows the boxes on the canvas. Overrides are applied to the live config."),
        annotations=_MUTATING)
    async def detect(
        ctx: Context,
        model: Annotated[Optional[str], Field(description="Detection model name.")] = None,
        iou_threshold: Annotated[Optional[float],
                                 Field(ge=0.0, le=1.0, description="NMS IoU threshold.")] = None,
        score_threshold: Annotated[Optional[float], Field(
            ge=0.0, le=1.0, description="Detector confidence threshold.")] = None,
        allow_dynamic: Annotated[Optional[bool], Field(
            description="Dynamic input size (tall pages: more boxes, slower).")] = None,
    ) -> DetectResult:
        patch = _detection_patch(model, iou_threshold, score_threshold, allow_dynamic)
        return await _detect(service, ctx, patch)

    @server.tool(name="run_pipeline", description=(
        "Full run on the loaded page: detect -> order -> crop -> OCR. Returns the result "
        "document (§2.1) and, with include_annotated_image, the overlay PNG as image content. "
        "Sends MCP progress notifications while the GUI progress bar moves. Any override is "
        "applied to the live config first, so the GUI shows what actually ran."),
        annotations=_MUTATING)
    async def run_pipeline(
        ctx: Context,
        model: Annotated[Optional[str], Field(description="Detection model name.")] = None,
        iou_threshold: Annotated[Optional[float],
                                 Field(ge=0.0, le=1.0, description="NMS IoU threshold.")] = None,
        score_threshold: Annotated[Optional[float], Field(
            ge=0.0, le=1.0, description="Detector confidence threshold.")] = None,
        allow_dynamic: Annotated[Optional[bool],
                                 Field(description="Dynamic input size.")] = None,
        ocr: Annotated[Optional[dict[str, Any]], Field(
            description="OCR section overrides: device, max_patches, batch_size, "
                        "max_new_tokens, crop_padding_px, normalize_nfkc.")] = None,
        reading_order: Annotated[Optional[str], Field(
            description="score | top_to_bottom | manga_rtl.")] = None,
        include_annotated_image: Annotated[bool, Field(
            description="Also return the annotated PNG as MCP image content.")] = False,
    ) -> RunPipelineResult:
        # The patch is built *inside* `_run_pipeline`: building it here would put a rejected
        # `ocr` override outside the try/except that shapes our error answer (P8.4).
        return await _run_pipeline(
            service, ctx,
            lambda: _run_patch(model, iou_threshold, score_threshold, allow_dynamic, ocr,
                               reading_order),
            bool(include_annotated_image))

    @server.tool(name="reocr_blocks", description=(
        "Re-read selected blocks of the current document without detecting again: crop the named "
        "boxes, OCR only those, and return a whole new result document (same boxes, ids, colours "
        "and reading order; new created_at/timing_ms; `stage: 'reocr'`). The way to act on "
        "`flags: [\"possible_truncation\"]` — usually with an `ocr` override raising "
        "`max_new_tokens`. Needs a loaded page and a result; BUSY while a job runs."),
        annotations=_MUTATING)
    async def reocr_blocks(
        ctx: Context,
        block_ids: Annotated[list[int], Field(
            description="Block ids to read again, e.g. [7] or [3, 7] — the ids from the result "
                        "document. Unknown ids are an INVALID_ARGUMENT listing what is available.")],
        ocr: Annotated[Optional[dict[str, Any]], Field(
            description="OCR section overrides for this read only: max_new_tokens, "
                        "crop_padding_px, batch_size, device, normalize_nfkc.")] = None,
        include_annotated_image: Annotated[bool, Field(
            description="Also return the annotated PNG as MCP image content.")] = False,
    ) -> RunPipelineResult:
        return await _run_pipeline(
            service, ctx,
            lambda: _ocr_patch(ocr),
            bool(include_annotated_image),
            tool_name="reocr_blocks",
            start_of=lambda: service.start_reocr(block_ids))

    @server.tool(name="cancel_job", description=(
        "Request cancellation of the running job. Returns the job id and the state; "
        "`cancelled: false` when there was nothing to cancel (that is not an error)."),
        annotations=_MUTATING)
    async def cancel_job() -> CancelResult:
        return await call(service, "cancel_job", service.cancel, {}, CancelResult)

    # -- results --------------------------------------------------------------

    @server.tool(name="get_result", description=(
        f"The latest result document (result.schema.json, schema_version "
        f"{RESULT_SCHEMA_VERSION}); `block_ids` returns only those blocks "
        "(INVALID_ARGUMENT when none of them exist). Each block carries `flags` "
        "(`possible_truncation`) and `contained_in` (id of the tightest larger box "
        "covering it) — observations, never edits."),
        annotations=_READ_ONLY)
    async def get_result(
        block_ids: Annotated[Optional[list[int]],
                             Field(description="Block ids to keep (default: all).")] = None,
    ) -> GetResultResult:
        return await call(service, "get_result", lambda: service.result_payload(block_ids),
                          {"block_ids": block_ids}, GetResultResult)

    @server.tool(name="save_result", description=(
        "Write the latest result document to `path` on the app's machine."),
        annotations=_MUTATING)
    async def save_result(
        path: Annotated[str, Field(description="Destination file (parents are created).")],
    ) -> SaveResultResult:
        return await call(service, "save_result", lambda: service.save_result(path),
                          {"path": path}, SaveResultResult)

    @server.tool(name="get_annotated_image", description=(
        "The current result drawn over the page, as a PNG image content block; metadata "
        "(size, block count, scale) comes back as structured content."),
        annotations=_READ_ONLY)
    async def get_annotated_image(
        scale: Annotated[Optional[float], Field(
            ge=0.05, le=8.0,
            description="Render scale (1.0 = full resolution of the source page).")] = None,
    ) -> AnnotatedImageMeta:
        return await _annotated_image(service, scale)

    # -- view -----------------------------------------------------------------

    @server.tool(name="set_overlay_visible", description=(
        "Show/hide the box overlay, the labels, or set the fill opacity. Applies to the "
        "canvas and to the gui.* config section at once (so it exports and imports)."),
        annotations=_MUTATING)
    async def set_overlay_visible(
        visible: Annotated[Optional[bool],
                           Field(description="Draw the boxes at all.")] = None,
        labels: Annotated[Optional[bool],
                          Field(description="Draw the id/text chips.")] = None,
        fill_opacity: Annotated[Optional[float],
                                Field(ge=0.0, le=1.0, description="Box fill opacity.")] = None,
    ) -> OverlayResult:
        return await call(service, "set_overlay_visible",
                          lambda: service.set_overlay(visible, labels, fill_opacity),
                          {"visible": visible, "labels": labels,
                           "fill_opacity": fill_opacity}, OverlayResult)

    @server.tool(name="set_zoom", description=(
        "Zoom the output canvas: mode \"fit\", or \"percent\" with percent 10..800."),
        annotations=_MUTATING)
    async def set_zoom(
        mode: Annotated[str, Field(description="\"fit\" or \"percent\".")],
        percent: Annotated[Optional[int],
                           Field(ge=10, le=800, description="Zoom percent for mode=percent.")] = None,
    ) -> ZoomResult:
        return await call(service, "set_zoom", lambda: service.set_zoom(mode, percent),
                          {"mode": mode, "percent": percent}, ZoomResult)

    @server.tool(name="highlight_block", description=(
        "Select a block: the canvas box is marked, its hover panel shows its text, the table "
        "row and the JSON pane follow. The panel clears after `duration_ms` (default 1500)."),
        annotations=_MUTATING)
    async def highlight_block(
        id: Annotated[int, Field(description="Block id from the result document.")],
        duration_ms: Annotated[Optional[int], Field(
            ge=0, le=60000, description="How long the hover panel stays up.")] = None,
    ) -> HighlightResult:
        return await call(service, "highlight_block",
                          lambda: service.highlight(id, duration_ms),
                          {"id": id, "duration_ms": duration_ms}, HighlightResult)

    @server.tool(name="reset", description=(
        "Throw the page away: clears the image, the result, the canvas, the table and the "
        "JSON pane. BUSY while a job is running (cancel_job first)."),
        annotations=_MUTATING)
    async def reset() -> ResetResult:
        return await call(service, "reset", service.reset, {}, ResetResult)

    @server.tool(name="unload_engines", description=(
        "Free the detector and OCR models (waits for the running job). The next run loads "
        "them again — useful when the GPU is shared. Note: the model objects are released and "
        "that memory becomes reusable, but ONNX Runtime keeps a few hundred MB of process-wide "
        "state that is only returned when the process exits."),
        annotations=_MUTATING)
    async def unload_engines() -> UnloadResult:
        return await call(service, "unload_engines", service.unload, {}, UnloadResult,
                          timeout=service._call_timeout * 4)

    @server.tool(name="warm_engines", description=(
        "Start the engine warm-up job — `--preload`'s remote twin, and the counterpart of "
        "`unload_engines`. Loads the weights AND runs one throwaway inference per engine, so the "
        "first real page is not slower than every page after it. Returns immediately with the job "
        "id (BUSY if a job is already running); poll `get_status`, where `job.kind == 'warmup'` "
        "and `engines[].warm` is the proof. `probe: false` loads without the inference: from a "
        "cold engine that leaves `loaded: true, warm: false`."),
        annotations=_MUTATING)
    async def warm_engines(probe: Annotated[
            Optional[bool], Field(
                description="Run the throwaway first inference (default true). "
                            "False = load the weights only.")] = None) -> WarmEnginesResult:
        return await call(service, "warm_engines",
                          lambda: service.warm_engines(probe=probe), {}, WarmEnginesResult)


# ---------------------------------------------------------------------------
# The two tools that need more than one service call
# ---------------------------------------------------------------------------

def _detection_patch(model: Optional[str], iou: Optional[float], score: Optional[float],
                     allow_dynamic: Optional[bool]) -> dict[str, Any]:
    detection: dict[str, Any] = {}
    if model is not None:
        detection["model"] = model
    if iou is not None:
        detection["iou_threshold"] = iou
    if score is not None:
        detection["score_threshold"] = score
    if allow_dynamic is not None:
        detection["allow_dynamic"] = allow_dynamic
    return {"detection": detection} if detection else {}


async def _run_job(service: StudioService, ctx: Any, name: str, patch: dict[str, Any],
                   *, ocr_enabled: bool = True, start_of: Any = None) -> tuple[str, Any, int]:
    """Apply overrides -> start the job -> stream progress notifications -> settle.

    Returns `(job_id, PipelineResult, progress_events)`; raises `StudioError` (BUSY, NO_IMAGE,
    CANCELLED, or whatever the job reported) so the caller can turn it into one `isError`
    result and one activity entry. The wait itself happens in the service (off the event
    loop, off the Qt thread); progress arrives through a `JobManager` listener that hops back
    onto this loop with `call_soon_threadsafe`.
    """
    if patch:
        await service.acall(lambda: service.apply_run_overrides(patch))
    # `start_of` is how a job that is not a page joins the same road: a re-read (P9.3) starts
    # through `service.start_reocr`, and gets the single-flight lock, the progress notifications
    # and the one activity entry for free.
    job_id = await service.acall(start_of or (lambda: service.start_job(ocr_enabled=ocr_enabled)))
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(state: Any) -> None:
        if state.job_id and state.job_id != job_id:
            return                      # a job someone else started is not ours to report
        loop.call_soon_threadsafe(queue.put_nowait,
                                  (float(state.percent), state.stage, state.message))

    service.subscribe(on_progress)
    events = 0
    try:
        while True:
            try:
                percent, stage, message = await asyncio.wait_for(queue.get(), 0.2)
            except asyncio.TimeoutError:
                status = service.manager.status()      # thread-safe read, no dispatch needed
                if status["job_id"] != job_id or not status["busy"]:
                    break
                continue
            events += 1
            if ctx is not None:
                await ctx.report_progress(round(percent, 1), 100.0, message or stage or None)
        # Deliberately *not* dispatched. The pump above only leaves its loop once the job has
        # settled, so this is a join on a thread that has already ended — and a dispatched wait
        # is the one thing that must never happen: on a window-attached service it would occupy
        # the app thread for the length of the job (frozen window, no progress, no cancel) and
        # then answer `INTERNAL: the application did not answer within N s` for a job that is
        # merely slow. `JobManager.wait` is thread-safe by design, so the worker thread may do
        # it directly (found by the P8.5 real-model probes, where jobs cost seconds).
        await asyncio.get_running_loop().run_in_executor(None, service.wait_for_job)
        result = await service.acall(lambda: service.job_result(job_id))
    finally:
        service.unsubscribe(on_progress)
    return job_id, result, events


async def _detect(service: StudioService, ctx: Any, patch: dict[str, Any]) -> CallToolResult:
    started = time.perf_counter()
    job_id = ""
    try:
        job_id, result, _events = await _run_job(service, ctx, "detect", patch,
                                                ocr_enabled=False)
    except StudioError as exc:
        _record(service, "detect", ok_=False, args=patch, started=started,
                code=exc.code.value, message=exc.message, job_id=job_id)
        return failure(exc)
    payload = {
        "detections": [{"id": block.id, "bbox": block.bbox.to_dict(),
                        "label": block.label, "score": block.score}
                       for block in result.blocks],
        "count": len(result.blocks),
        "model": result.detection.model,
        "iou_threshold": result.detection.iou_threshold,
        "score_threshold": result.detection.score_threshold,
        "allow_dynamic": result.detection.allow_dynamic,
        "timing_ms": result.timing_ms.detect,
        "job_id": job_id,
    }
    _record(service, "detect", ok_=True, args=patch, started=started,
            message=f"{len(result.blocks)} box(es) in {result.timing_ms.detect} ms",
            job_id=job_id)
    return ok(DetectResult, payload)


def _run_patch(model: Optional[str], iou: Optional[float], score: Optional[float],
               allow_dynamic: Optional[bool], ocr: Optional[dict[str, Any]],
               reading_order: Optional[str]) -> dict[str, Any]:
    patch = _detection_patch(model, iou, score, allow_dynamic)
    if ocr:
        if not isinstance(ocr, dict):
            raise StudioError(ErrorCode.INVALID_CONFIG, "`ocr` must be an object.",
                              {"got": type(ocr).__name__,
                               "hint": 'e.g. {"batch_size": 8, "max_new_tokens": 256}'})
        patch["ocr"] = dict(ocr)
    if reading_order is not None:
        patch["pipeline"] = {"reading_order": reading_order}
    return patch


def _ocr_patch(ocr: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The OCR section alone — what a re-read can be told differently (P9.3).

    Detection overrides are deliberately absent: a re-read works on the boxes that exist, so
    `iou_threshold` would be a knob with nothing to turn.
    """
    if ocr:
        if not isinstance(ocr, dict):
            raise StudioError(ErrorCode.INVALID_CONFIG, "`ocr` must be an object.",
                              {"got": type(ocr).__name__,
                               "hint": 'e.g. {"max_new_tokens": 256}'})
        return {"ocr": dict(ocr)}
    return {}


async def _run_pipeline(service: StudioService, ctx: Any, patch_of: Any,
                        include_annotated: bool, *, tool_name: str = "run_pipeline",
                        start_of: Any = None) -> CallToolResult:
    """start -> pump progress notifications -> settle -> shape. One activity entry, at the end.

    `patch_of` is a callable built here rather than a dict built by the caller, so that an
    override the app refuses (`ocr: "…"`, a bad `reading_order`) is reported through the same
    `failure()` path as everything else instead of escaping as an exception the SDK would
    flatten into an unstructured `isError`.
    """
    started = time.perf_counter()
    job_id = ""
    patch: dict[str, Any] = {}
    try:
        patch = patch_of()
        job_id, result, progress_events = await _run_job(service, ctx, tool_name, patch,
                                                         start_of=start_of)
    except StudioError as exc:
        _record(service, tool_name, ok_=False, args=patch, started=started,
                code=exc.code.value, message=exc.message, job_id=job_id)
        return failure(exc)
    except Exception as exc:                                        # noqa: BLE001
        # Anything else is an internal fault; report it the way the service would.
        wrapped = StudioError(ErrorCode.INTERNAL, f"Unexpected error: {exc}",
                              {"reason": f"{type(exc).__name__}: {exc}"})
        _record(service, tool_name, ok_=False, args=patch, started=started,
                code=wrapped.code.value, message=wrapped.message, job_id=job_id)
        return failure(wrapped)

    document = result.to_dict()
    payload = {
        "result": document,
        "job_id": job_id,
        "block_count": result.block_count,
        "timing_ms": {"detect": result.timing_ms.detect, "ocr": result.timing_ms.ocr,
                      "total": result.timing_ms.total},
        "progress_events": progress_events,
    }
    extra: list[Any] = []
    if include_annotated:
        png, meta = await service.acall(service.annotated)
        payload["annotated_image"] = meta
        extra.append(ImageContent(type="image", data=base64.b64encode(png).decode("ascii"),
                                  mimeType="image/png"))
    _record(service, tool_name, ok_=True, args=patch, started=started,
            message=f"{result.block_count} block(s) in {result.timing_ms.total} ms",
            job_id=job_id)
    return ok(RunPipelineResult, payload, extra_content=extra)


async def _annotated_image(service: StudioService,
                           scale: Optional[float]) -> CallToolResult:
    try:
        png, meta = await service.aexecute(
            "get_annotated_image", lambda: service.annotated(scale),
            args={"scale": scale},
            summarize=lambda pair: f"{pair[1]['width']}x{pair[1]['height']} PNG")
    except StudioError as exc:
        return failure(exc)
    return CallToolResult(
        content=[ImageContent(type="image", data=base64.b64encode(png).decode("ascii"),
                              mimeType="image/png")],
        structured_content=AnnotatedImageMeta.model_validate(meta).model_dump(mode="json"))


def _record(service: StudioService, name: str, *, ok_: bool, args: dict[str, Any],
            started: float, code: str = "", message: str = "", job_id: str = "") -> None:
    """The one activity entry a multi-step tool owes the log dock."""
    service.activity.record(name, ok=ok_, args=args, code=code, message=message,
                            job_id=job_id, transport=service.mcp_transport,
                            duration_ms=int(round((time.perf_counter() - started) * 1000)))


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

def _registered_tools(server: Any) -> list[Any]:
    """The SDK's own view of the registered tools (input schemas included)."""
    manager = getattr(server, "_tool_manager", None)
    if manager is not None:
        return list(manager.list_tools())
    return []


def tool_catalog(server: Any, *, transport: str = "streamable-http",
                 host: str = "127.0.0.1", port: int = 8765,
                 resources: Optional[list[dict[str, Any]]] = None,
                 version: str = "", name: str = "animetext-studio") -> dict[str, Any]:
    """The machine-readable catalog that ships as `schemas/mcp-tools.json`.

    Dumped from the live server object, so the published schemas are the ones on the wire -
    `tests/test_mcp_catalog.py` regenerates it and fails on any drift.
    """
    tools: list[dict[str, Any]] = []
    for tool in _registered_tools(server):
        gui = TOOL_GUI_EFFECT.get(tool.name, {})
        entry: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "annotations": (tool.annotations.model_dump(by_alias=True, exclude_none=True)
                            if tool.annotations is not None else None),
            "inputSchema": tool.parameters,
            "outputSchema": (tool.output_schema
                             if tool.output_schema is not None else _output_schema_of(tool.name)),
            "guiEffect": gui.get("effect", ""),
            "flash": gui.get("flash", []),
        }
        tools.append(entry)
    tools.sort(key=lambda item: item["name"])
    return {
        "catalogVersion": "1.0",
        "server": {"name": name, "version": version},
        "transport": {"default": transport, "available": ["streamable-http", "stdio", "none"],
                      "endpoint": "/mcp", "host": host, "port": port},
        "errorShape": {"code": "string (ErrorCode)", "message": "string",
                       "details": "object (optional)"},
        "errorCodes": [code.value for code in ErrorCode],
        "tools": tools,
        "resources": list(resources or []),
    }


#: Fallback output schemas, only used if the SDK ever fails to derive one.
_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_status": GetStatusResult, "list_models": ListModelsResult,
    "get_config": GetConfigResult, "set_config": SetConfigResult,
    "export_config": ExportConfigResult, "import_config": ImportConfigResult,
    "load_image": LoadImageResult, "get_image_info": ImageInfoResult,
    "detect": DetectResult, "run_pipeline": RunPipelineResult,
    "cancel_job": CancelResult, "get_result": GetResultResult,
    "save_result": SaveResultResult, "get_annotated_image": AnnotatedImageMeta,
    "set_overlay_visible": OverlayResult, "set_zoom": ZoomResult,
    "highlight_block": HighlightResult, "reset": ResetResult,
    "unload_engines": UnloadResult,
}


def _output_schema_of(tool_name: str) -> Optional[dict[str, Any]]:
    model = _OUTPUT_MODELS.get(tool_name)
    return model.model_json_schema() if model is not None else None
