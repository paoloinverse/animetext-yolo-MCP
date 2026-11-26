"""Part 7 — the server: catalog, transports, lifecycle and bearer auth.

No client here except a raw `httpx` call to the gateway: this file is about what the
server *publishes* and how it starts and stops. The protocol round-trips live in
`test_mcp_http.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the MCP SDK is not installed (pip install 'animetext-studio[mcp]')")

from animetext_studio import __version__                        # noqa: E402
from animetext_studio.core.config import McpConfig              # noqa: E402
from animetext_studio.core.types import ErrorCode, StudioError  # noqa: E402
from animetext_studio.mcp import server as server_mod           # noqa: E402
from animetext_studio.mcp.server import (                       # noqa: E402
    HttpGateway, StudioServer, build_mcp_server, catalog_document, write_catalog,
)
from animetext_studio.mcp.tools import TOOL_GUI_EFFECT         # noqa: E402

CATALOG_PATH = (Path(__file__).resolve().parent.parent
                / "animetext_studio" / "schemas" / "mcp-tools.json")

#: The 18 tools of PLAN §4, plus the two that came after it: `unload_engines` (promised by
#: jobs.py, shipped in P7) and `warm_engines` (P8.3, its counterpart).
SPEC_TOOLS = {
    "get_status", "list_models", "get_config", "set_config", "export_config",
    "import_config", "load_image", "get_image_info", "detect", "run_pipeline",
    "cancel_job", "get_result", "save_result", "get_annotated_image",
    "set_overlay_visible", "set_zoom", "highlight_block", "reset",
}
#: Shipped after §4 was written, so they are not in `SPEC_TOOLS` but are expected all the
#: same: P8 added the two engine doors, P9.3 added the one that re-reads an existing answer.
POST_SPEC_TOOLS = {"unload_engines", "warm_engines", "reocr_blocks"}

SPEC_RESOURCES = {"config://current", "result://latest", "image://current",
                  "image://annotated", "models://list", "log://mcp"}


# ---------------------------------------------------------------------------
# The published catalog
# ---------------------------------------------------------------------------

def test_catalog_covers_the_spec_and_documents_every_tool():
    catalog = catalog_document()
    names = {t["name"] for t in catalog["tools"]}
    assert SPEC_TOOLS <= names
    assert names == SPEC_TOOLS | POST_SPEC_TOOLS
    assert {r["uri"] for r in catalog["resources"]} == SPEC_RESOURCES
    for tool in catalog["tools"]:
        assert tool["description"], f"{tool['name']} has no description"
        assert tool["inputSchema"]["type"] == "object"
        assert tool["outputSchema"] is not None, f"{tool['name']} has no output schema"
        assert tool["annotations"], f"{tool['name']} has no read-only/destructive hint"
        assert tool["guiEffect"], f"{tool['name']} documents no visual confirmation"
    assert "ctx" not in catalog["tools"][0]["inputSchema"]["properties"]


def test_catalog_is_the_file_that_ships_in_the_wheel():
    """The committed JSON must be exactly what the running server would publish."""
    on_disk = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert on_disk == catalog_document()


def test_catalog_records_the_error_taxonomy_and_versions():
    catalog = catalog_document()
    assert catalog["server"] == {"name": "animetext-studio", "version": __version__}
    assert catalog["transport"]["endpoint"] == "/mcp"
    assert catalog["errorShape"] == {"code": "string (ErrorCode)", "message": "string",
                                     "details": "object (optional)"}
    assert "BUSY" in catalog["errorCodes"] and "NO_IMAGE" in catalog["errorCodes"]


def test_write_catalog_writes_where_it_is_told_and_nowhere_else(tmp_path):
    """`path=` is honoured, and the shipped catalog is not a scratch file (D94).

    The first version of this test passed `tmp_path / "tools.json"` **positionally**, which
    bound to `service`: it regenerated `schemas/mcp-tools.json` twice and compared that file
    with itself. It passed, and it repaired the very artifact
    `test_catalog_is_the_file_that_ships_in_the_wheel` exists to catch — a check that fixes its
    own subject is not a check.
    """
    before = CATALOG_PATH.read_bytes()
    target = tmp_path / "nested" / "tools.json"
    first = write_catalog(path=target)
    again = write_catalog(path=target)
    assert Path(first) == Path(again) == target
    assert target.read_bytes() == before, "a copy must be the shipped catalog, byte for byte"
    assert CATALOG_PATH.read_bytes() == before, "generating a copy touched the shipped catalog"


# ---------------------------------------------------------------------------
# Building the MCP server object
# ---------------------------------------------------------------------------

def test_build_mcp_server_registers_tools_and_resources(mock_service):
    import anyio

    mcp_server = build_mcp_server(mock_service)
    assert mcp_server.name == "animetext-studio"
    assert mcp_server.version == __version__
    assert "single-flight" in (mcp_server.instructions or "").lower()

    async def collect():
        return await mcp_server.list_tools(), await mcp_server.list_resources()

    tools, resources = anyio.run(collect)
    assert {t.name for t in tools} == SPEC_TOOLS | POST_SPEC_TOOLS
    assert {str(r.uri) for r in resources} == SPEC_RESOURCES
    run = next(t for t in tools if t.name == "run_pipeline")
    assert "include_annotated_image" in run.input_schema["properties"]
    assert run.output_schema["properties"]["block_count"]["type"] == "integer"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_transport_none_never_binds_and_reports_itself(mock_service):
    server = StudioServer(mock_service, McpConfig(enabled=False, transport="none"))
    info = server.start()
    assert info["running"] is False and info["transport"] == "none"
    assert server.running is False
    server.stop()                                       # stopping a server that never ran


def test_http_server_starts_reports_and_stops(mcp_http):
    service, server, url = mcp_http["service"], mcp_http["server"], mcp_http["url"]
    assert url.startswith("http://127.0.0.1:") and url.endswith("/mcp")
    assert server.running is True
    assert service.snapshot()["mcp"] == {
        "running": True, "transport": "streamable-http", "host": "127.0.0.1",
        "port": server.port, "url": url, "auth_required": False, "endpoint": "/mcp",
        "started_at": service.snapshot()["mcp"]["started_at"], "calls": 0, "errors": 0,
        # P8: loopback bind => nothing to warn about. A non-loopback bind is covered in
        # tests/test_p8_hardening.py.
        "warnings": []}
    server.stop()
    server.stop()                                       # idempotent
    assert server.running is False
    assert service.snapshot()["mcp"]["running"] is False


def test_a_port_already_in_use_is_a_structured_error(mock_service, free_port):
    import socket

    port = free_port()
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        server = StudioServer(mock_service, McpConfig(port=port))
        with pytest.raises(StudioError) as excinfo:
            server.start()
        assert excinfo.value.code is ErrorCode.ENGINE_LOAD_FAILED
        assert excinfo.value.details["port"] == port
        assert "port" in excinfo.value.message.lower() or "bind" in excinfo.value.message.lower()
    finally:
        blocker.close()


# ---------------------------------------------------------------------------
# Bearer auth (gateway level, no MCP handshake needed)
# ---------------------------------------------------------------------------

def _get(url: str, headers: dict | None = None):
    import httpx

    with httpx.Client(timeout=5.0) as client:
        return client.get(url, headers=headers or {})


def _serve_gateway(gateway, port):
    """Run a gateway under uvicorn in a thread; returns (server, thread)."""
    import threading
    import time

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(gateway, host="127.0.0.1", port=port,
                                           log_level="critical", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(400):
        if server.started:
            return server, thread
        time.sleep(0.02)
    raise AssertionError("the gateway did not start")


def test_gateway_health_is_open_and_the_mcp_endpoint_is_not(free_port):
    import time

    async def inner(scope, receive, send):
        from starlette.responses import PlainTextResponse

        await PlainTextResponse("inner")(scope, receive, send)

    port = free_port()
    gateway = HttpGateway(inner, token="s3cret")
    server, thread = _serve_gateway(gateway, port)
    base = f"http://127.0.0.1:{port}"
    try:
        health = _get(f"{base}/health")
        assert health.status_code == 200 and health.json()["ok"] is True

        anonymous = _get(f"{base}/mcp")
        assert anonymous.status_code == 401
        assert anonymous.json()["error"] == "unauthorized"   # HTTP-level, not a tool error
        assert "Bearer" in anonymous.headers.get("www-authenticate", "")
        assert gateway.rejected == 1

        authorized = _get(f"{base}/mcp", {"Authorization": "Bearer s3cret"})
        assert authorized.status_code == 200 and authorized.text == "inner"
        assert gateway.rejected == 1

        wrong = _get(f"{base}/mcp", {"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert gateway.rejected == 2
    finally:
        server.should_exit = True
        thread.join(5)
        time.sleep(0.05)


def test_gateway_without_a_token_lets_everything_through(free_port):
    import time

    async def inner(scope, receive, send):
        from starlette.responses import PlainTextResponse

        await PlainTextResponse("open")(scope, receive, send)

    port = free_port()
    gateway = HttpGateway(inner, token=None)
    server, thread = _serve_gateway(gateway, port)
    try:
        response = _get(f"http://127.0.0.1:{port}/mcp")
        assert response.status_code == 200 and response.text == "open"
        assert gateway.rejected == 0
    finally:
        server.should_exit = True
        thread.join(5)
        time.sleep(0.05)


