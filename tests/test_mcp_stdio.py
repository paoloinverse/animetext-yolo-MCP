"""P7 - the stdio transport, handshake included, over a real subprocess.

Nothing here is in-process on purpose: the point of stdio is that a harness *spawns* the app
and talks to it down a pipe, so the test spawns `python -m animetext_studio --mock --serve
--mcp-transport stdio` exactly the way Claude Desktop or `mcp run` would. That proves three
things a same-process test cannot: the CLI reaches the server, stdout carries protocol bytes
only (the banner must go to stderr), and the child goes away when the pipe closes.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the MCP SDK is not installed (pip install 'animetext-studio[mcp]')")

from mcp import StdioServerParameters                                   # noqa: E402
from mcp.client.session import ClientSession                            # noqa: E402
from mcp.client.stdio import stdio_client                               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = str(ROOT / "examples" / "sample_page.png")
CATALOG = json.loads((ROOT / "animetext_studio" / "schemas" / "mcp-tools.json")
                     .read_text(encoding="utf-8"))
CATALOG_TOOLS = {tool["name"]: tool for tool in CATALOG["tools"]}
CATALOG_URIS = sorted(entry["uri"] for entry in CATALOG["resources"])


def server_params() -> StdioServerParameters:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH", "")]))
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "animetext_studio", "--mock", "--serve", "--mcp-transport", "stdio"],
        cwd=str(ROOT), env=env)


def run_stdio(probe):
    """Spawn the app, hand `probe` a live `ClientSession`, return whatever it returns."""

    async def go():
        async with contextlib.AsyncExitStack() as stack:
            log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
            stack.callback(log.close)
            read, write = await stack.enter_async_context(
                stdio_client(server_params(), errlog=log))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return await probe(session)

    return asyncio.run(asyncio.wait_for(go(), timeout=120))


def test_the_stdio_handshake_publishes_the_same_catalog_as_http() -> None:
    async def probe(session):
        listed = await session.list_tools()
        resources = await session.list_resources()
        return {t.name: t for t in listed.tools}, [str(r.uri) for r in resources.resources]

    by_name, uris = run_stdio(probe)
    assert sorted(by_name) == sorted(CATALOG_TOOLS), set(by_name) ^ set(CATALOG_TOOLS)
    assert sorted(uris) == CATALOG_URIS
    for name, entry in CATALOG_TOOLS.items():
        tool = by_name[name]
        assert tool.input_schema == entry["inputSchema"], name
        assert tool.output_schema == entry.get("outputSchema"), name


def test_a_tool_call_and_a_resource_read_work_over_stdio() -> None:
    async def probe(session):
        status = await session.call_tool("get_status", {})
        assert not status.is_error
        payload = status.structured_content
        assert payload["mock"] is True and payload["gui"] is False
        assert payload["mcp"]["transport"] == "stdio"

        # an error stays a structured error down the pipe - it does not break the session
        empty = await session.call_tool("get_result", {})
        assert empty.is_error and empty.structured_content["code"] == "NO_IMAGE"

        assert not (await session.call_tool("load_image", {"path": SAMPLE})).is_error
        detected = await session.call_tool("detect", {})
        assert not detected.is_error, detected.structured_content
        assert detected.structured_content["count"] > 0

        published = await session.call_tool("get_result", {})
        assert not published.is_error
        document = published.structured_content["result"]
        assert document["block_count"] == detected.structured_content["count"]
        assert all(not block.get("text") for block in document["blocks"]), \
            "stage 1 publishes blocks without text (no OCR ran)"

        models = await session.read_resource("models://list")
        assert json.loads(models.contents[0].text)["models"]
        return detected.structured_content["count"]

    assert run_stdio(probe) > 0        # the fake engine always finds something


def test_the_child_exits_when_the_pipe_closes() -> None:
    import subprocess
    import time

    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, "-m", "animetext_studio", "--mock", "--serve",
         "--mcp-transport", "stdio"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        cwd=str(ROOT), env=server_params().env)
    try:
        # A bare `initialize`, written by hand: if stdout carried anything but protocol
        # bytes, this line would not parse.
        hello = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": {"name": "stdio-test", "version": "0"}}}
        child.stdin.write((json.dumps(hello) + "\n").encode())
        child.stdin.flush()

        line, deadline = b"", time.time() + 90
        while time.time() < deadline:
            line = child.stdout.readline()
            if line.strip():
                break
            time.sleep(0.05)
        assert line, "the server answered nothing on stdout"
        answer = json.loads(line)
        assert answer["id"] == 1 and "serverInfo" in answer["result"], answer

        log.seek(0)
        assert "stdio serving" in log.read(), "the banner must reach stderr, not stdout"
    finally:
        child.stdin.close()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            raise
        log.close()
