"""P7 - the six read-only MCP resources of PLAN §4.

Resources are the *passive* half of the interface: a harness (or the Inspector) reads the
same state the window shows without asking the app to do anything. Each one is the same
service call the tool uses, so `result://latest` and `get_result` can never disagree.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from ..core.types import ErrorCode, StudioError
from .service import StudioService

__all__ = ["RESOURCE_CATALOG", "register_resources", "as_mcp_error", "report_resource_errors"]

#: JSON-RPC error codes (the spec fixes these two and reserves -32000..-32099 for us).
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

#: Conditions the *caller* can act on: send a better request, or run something first.
_CALLER_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.NO_IMAGE, ErrorCode.INVALID_ARGUMENT, ErrorCode.INVALID_CONFIG,
    ErrorCode.INVALID_IMAGE, ErrorCode.BUSY, ErrorCode.NOT_IMPLEMENTED,
})


def as_mcp_error(exc: StudioError) -> Any:
    """A `StudioError` as the `MCPError` the SDK passes through **unchanged** (P8.4).

    Before this, every resource failure reached the client as
    `{-32603, "Error reading resource result://latest"}`: our code, our message and our hint
    stopped at the server log, while `RESOURCE_CATALOG` promised "NO_IMAGE when there is none".
    The SDK documents that an `MCPError` raised by a resource function is not wrapped, so the
    taxonomy rides along in `data` as the same `{code, message, details}` object a failed tool
    call returns — one error shape for both halves of the interface.
    """
    from mcp.shared.exceptions import MCPError

    code = INVALID_PARAMS if exc.code in _CALLER_CODES else INTERNAL_ERROR
    return MCPError(code, exc.message, exc.to_dict())


def report_resource_errors(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Turn a resource's `StudioError` into an `MCPError` on the way out (see above)."""

    @functools.wraps(fn)
    async def run() -> Any:
        try:
            return await fn()
        except StudioError as exc:
            raise as_mcp_error(exc) from exc

    return run

#: Published verbatim in `schemas/mcp-tools.json` (order = the order §4 lists them).
RESOURCE_CATALOG: list[dict[str, Any]] = [
    {"uri": "config://current", "name": "current-config", "title": "Live configuration",
     "description": "The whole configuration document the app is running with "
                    "(config.schema.json).",
     "mimeType": "application/json"},
    {"uri": "result://latest", "name": "latest-result", "title": "Latest result document",
     "description": "The result JSON of the most recent successful run (§2.1). "
                    "NO_IMAGE when there is none.",
     "mimeType": "application/json"},
    {"uri": "image://current", "name": "current-image", "title": "Current page (PNG)",
     "description": "The loaded page as PNG, untouched by the overlay.",
     "mimeType": "image/png"},
    {"uri": "image://annotated", "name": "annotated-image", "title": "Annotated page (PNG)",
     "description": "The page with the current boxes drawn on it, at full resolution.",
     "mimeType": "image/png"},
    {"uri": "models://list", "name": "model-list", "title": "Model catalogue",
     "description": "The five YOLO variants with bundled/loaded/available flags.",
     "mimeType": "application/json"},
    {"uri": "log://mcp", "name": "mcp-log", "title": "MCP activity log",
     "description": "Recent tool calls: time, tool, arguments (redacted), outcome, duration.",
     "mimeType": "application/json"},
]


def register_resources(server: Any, service: StudioService) -> list[dict[str, Any]]:
    """Register every resource on `server`; returns the catalog entries."""

    @server.resource("config://current", name="current-config", title="Live configuration",
                     mime_type="application/json",
                     description="The whole configuration document the app is running with.")
    @report_resource_errors
    async def config_current() -> str:
        return await service.acall(service.config_document)

    @server.resource("result://latest", name="latest-result", title="Latest result document",
                     mime_type="application/json",
                     description="Result JSON of the most recent successful run.")
    @report_resource_errors
    async def result_latest() -> str:
        return await service.acall(service.result_document)

    @server.resource("image://current", name="current-image", title="Current page (PNG)",
                     mime_type="image/png", description="The loaded page, no overlay.")
    @report_resource_errors
    async def image_current() -> bytes:
        return await service.acall(service.image_png)

    @server.resource("image://annotated", name="annotated-image", title="Annotated page (PNG)",
                     mime_type="image/png",
                     description="The page with the current boxes drawn on it.")
    @report_resource_errors
    async def image_annotated() -> bytes:
        return await service.acall(service.annotated_png_bytes)

    @server.resource("models://list", name="model-list", title="Model catalogue",
                     mime_type="application/json",
                     description="YOLO variants with bundled/loaded/available flags.")
    @report_resource_errors
    async def models_list() -> str:
        return await service.acall(service.models_document)

    @server.resource("log://mcp", name="mcp-log", title="MCP activity log",
                     mime_type="application/json",
                     description="Recent tool calls, newest first (secrets redacted).")
    @report_resource_errors
    async def mcp_log() -> str:
        return await service.acall(service.log_document)

    return [dict(entry) for entry in RESOURCE_CATALOG]


def error_text(exc: StudioError) -> str:
    """Resources report failures as JSON-RPC errors; this is the text the SDK puts on them."""
    return exc.message
