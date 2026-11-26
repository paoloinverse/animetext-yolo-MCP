"""P7 - the MCP server package.

`bridge` and `service` are importable with nothing but Pillow and the stdlib; `tools`,
`resources` and `server` need the `mcp` SDK (pip install 'animetext-studio[mcp]') and are
imported lazily so that running the GUI with `mcp.enabled: false` - or just reading the
activity log - never depends on it.
"""
from __future__ import annotations

from typing import Any

from .bridge import (
    ActivityEvent, ActivityLog, Dispatcher, InlineDispatcher, QtDispatcher, redact_args,
)
from .service import StudioService

__all__ = [
    "ActivityEvent", "ActivityLog", "Dispatcher", "InlineDispatcher", "QtDispatcher",
    "StudioService", "redact_args",
    # lazy (need the `mcp` SDK):
    "StudioServer", "build_mcp_server", "write_catalog", "catalog_path", "HttpGateway",
    "register_tools", "register_resources", "tool_catalog", "RESOURCE_CATALOG",
]

_LAZY = {
    "StudioServer": ".server", "build_mcp_server": ".server", "write_catalog": ".server",
    "catalog_path": ".server", "HttpGateway": ".server", "SERVER_INSTRUCTIONS": ".server",
    "register_tools": ".tools", "tool_catalog": ".tools", "TOOL_GUI_EFFECT": ".tools",
    "register_resources": ".resources", "RESOURCE_CATALOG": ".resources",
}


def __getattr__(name: str) -> Any:                                   # PEP 562
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
