"""P7 - the server over real HTTP, driven by the official SDK client.

Everything here is in-process (a `StudioServer` in a thread) and uses the **mock engines**,
so it is a default test: no model weights, no token, no display. What it proves is the wire
contract - that every tool's `structuredContent` validates against the schema the server
published for it, that errors are `isError` + `{code,message,details}`, that a second run
gets `BUSY` instead of a queue, and that the six resources answer.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the MCP SDK is not installed (pip install 'animetext-studio[mcp]')")
pytest.importorskip("jsonschema", reason="jsonschema is a dev extra")

from jsonschema import Draft202012Validator                     # noqa: E402
from mcp.client.session import ClientSession                    # noqa: E402
from mcp.client.streamable_http import streamable_http_client   # noqa: E402
from PIL import Image                                           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = str(ROOT / "examples" / "sample_page.png")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def catalog_of(tool_name: str) -> dict:
    document = json.loads((ROOT / "animetext_studio" / "schemas" / "mcp-tools.json")
                          .read_text(encoding="utf-8"))
    for tool in document["tools"]:
        if tool["name"] == tool_name:
            return tool
    raise AssertionError(f"{tool_name} is not in the published catalog")


def call(session, tool: str, arguments: dict | None = None):
    """`session.call_tool` + the contract check every tool must satisfy."""
    return asyncio.run(_call(session, tool, arguments or {}))


async def _call(session, tool: str, arguments: dict | None = None):
    result = await session.call_tool(tool, arguments or {})
    if not result.is_error:
        schema = catalog_of(tool).get("outputSchema")
        assert schema, f"{tool} has no published outputSchema"
        errors = list(Draft202012Validator(schema).iter_errors(result.structured_content))
        assert not errors, f"{tool} returned content its own schema rejects: {errors[0].message}"
    return result


def error_of(result) -> dict:
    assert result.is_error, "expected isError"
    payload = result.structured_content
    assert payload and set(payload) >= {"code", "message"}, payload
    assert result.content and result.content[0].text, "an error must also be readable as text"
    assert json.loads(result.content[0].text)["code"] == payload["code"]
    return payload


def read(session, uri: str) -> tuple[str | bytes, str]:
    result = asyncio.run(session.read_resource(uri))
    item = result.contents[0]
    if getattr(item, "blob", None):
        return base64.b64decode(item.blob), item.mime_type
    return item.text, item.mime_type


# ---------------------------------------------------------------------------
# The happy path, tool by tool
# ---------------------------------------------------------------------------

def test_initialize_announces_the_app(mcp_http):
    async def go():
        async with _session(mcp_http["url"]) as session:
            init = await session.initialize()
            assert init.server_info.name == "animetext-studio"
            assert init.capabilities.tools is not None
            assert init.capabilities.resources is not None
            assert "single-flight" in (init.instructions or "").lower()
            tools = await session.list_tools()
            # PLAN's 18 + unload_engines + warm_engines (P8.3) + reocr_blocks (P9.3)
            assert len(tools.tools) == 21
            resources = await session.list_resources()
            assert len(resources.resources) == 6
    asyncio.run(go())


def _session(url: str, token: str | None = None):
    """ClientSession over streamable-http, as an async context manager."""
    import contextlib

    @contextlib.asynccontextmanager
    async def open_():
        async with contextlib.AsyncExitStack() as stack:
            kwargs = {}
            if token:
                from mcp.client.streamable_http import create_mcp_http_client

                client = create_mcp_http_client(
                    headers={"Authorization": f"Bearer {token}"})
                stack.push_async_callback(client.aclose)
                kwargs["http_client"] = client
            streams = await stack.enter_async_context(streamable_http_client(url, **kwargs))
            session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
            await session.initialize()
            yield session

    return open_()


def test_a_full_session_round_trips_and_validates(mcp_http):
    url = mcp_http["url"]

    async def go():
        async with _session(url) as session:
            status = (await _call(session, "get_status")).structured_content
            assert status["app"] == "animetext-studio" and status["busy"] is False
            assert status["mcp"]["running"] is True and status["mcp"]["url"] == url

            models = (await _call(session, "list_models")).structured_content
            assert len(models["models"]) == 5

            loaded = (await _call(session, "load_image", {"path": SAMPLE})).structured_content
            assert loaded["image"]["source"] == "mcp-path"

            run = (await _call(session, "run_pipeline",
                               {"score_threshold": 0.1})).structured_content
            assert run["block_count"] >= 1
            assert run["result"]["schema_version"] == "1.2"
            assert run["result"]["stage"] == "full"

            filtered = (await _call(session, "get_result",
                                    {"block_ids": [0, 1]})).structured_content
            assert filtered["filtered"] is True and filtered["block_count"] == 2

            image = await session.call_tool("get_annotated_image", {"scale": 0.5})
            kinds = [block.type for block in image.content]
            assert kinds == ["image"], kinds
            decoded = Image.open(io.BytesIO(base64.b64decode(image.content[0].data)))
            assert decoded.size == (loaded["image"]["width"] // 2,
                                    loaded["image"]["height"] // 2)
            assert image.structured_content["blocks"] == run["block_count"]

            zoom = (await _call(session, "set_zoom",
                                {"mode": "percent", "percent": 150})).structured_content
            assert zoom["zoom_percent"] == 150
            overlay = (await _call(session, "set_overlay_visible",
                                   {"visible": False})).structured_content
            assert overlay["overlay_visible"] is False
            highlight = (await _call(session, "highlight_block",
                                     {"id": 0, "duration_ms": 20})).structured_content
            assert highlight["found"] is True

    asyncio.run(go())


def test_errors_are_iserror_with_the_shared_taxonomy(mcp_http):
    url = mcp_http["url"]

    async def go():
        async with _session(url) as session:
            assert error_of(await session.call_tool("get_image_info", {}))["code"] == "NO_IMAGE"
            assert error_of(await session.call_tool(
                "set_zoom", {"mode": "sideways"}))["code"] == "INVALID_ARGUMENT"
            assert error_of(await session.call_tool(
                "set_config", {"patch": {"detection": {"iou_threshold": 5}}}))["code"] \
                == "INVALID_CONFIG"
            assert error_of(await session.call_tool("import_config", {}))["code"] \
                == "INVALID_ARGUMENT"
            # Malformed arguments never reach a tool body: the SDK's own validation says so.
            bad = await session.call_tool("highlight_block", {"id": "not-an-int"})
            assert bad.is_error

    asyncio.run(go())


def test_progress_notifications_arrive_during_a_run(mcp_http):
    url = mcp_http["url"]
    seen: list[tuple[float, str | None]] = []

    async def on_progress(progress, total, message):
        seen.append((float(progress), message))

    async def go():
        async with _session(url) as session:
            await _call(session, "load_image", {"path": SAMPLE})
            result = await session.call_tool("run_pipeline", {},
                                             progress_callback=on_progress)
            assert not result.is_error

    asyncio.run(go())
    assert seen, "a run must report progress"
    assert seen[-1][0] >= 99.0, seen[-1]
    assert all(0.0 <= value <= 100.0 for value, _ in seen)
    assert [value for value, _ in seen] == sorted(value for value, _ in seen)


def test_a_second_run_is_busy_with_the_running_job_id(mcp_http, slow_gate):
    """Single-flight (D43): the second caller is told, not queued."""
    service = mcp_http["service"]
    url = mcp_http["url"]
    release, entered = slow_gate

    async def go():
        async with _session(url) as session:
            await _call(session, "load_image", {"path": SAMPLE})
            first = asyncio.create_task(session.call_tool("run_pipeline", {}))
            for _ in range(400):                       # wait for the job to be inside detect
                if service.manager.is_busy and entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert service.manager.is_busy, "the gate never held the job"
            second = await session.call_tool("detect", {})
            payload = error_of(second)
            assert payload["code"] == "BUSY"
            assert payload["details"]["job_id"].startswith("job-")
            release.set()
            done = await first
            assert not done.is_error

    asyncio.run(go())


def test_cancel_job_cancels_the_running_one(mcp_http, slow_gate):
    service = mcp_http["service"]
    url = mcp_http["url"]
    release, entered = slow_gate

    async def go():
        async with _session(url) as session:
            await _call(session, "load_image", {"path": SAMPLE})
            first = asyncio.create_task(session.call_tool("run_pipeline", {}))
            for _ in range(400):
                if service.manager.is_busy and entered.is_set():
                    break
                await asyncio.sleep(0.01)
            cancelled = (await _call(session, "cancel_job", {})).structured_content
            assert cancelled["cancelled"] is True
            release.set()
            done = await first
            payload = error_of(done)
            assert payload["code"] == "CANCELLED"

    asyncio.run(go())


def test_resources_answer(mcp_http):
    url = mcp_http["url"]

    async def read_uri(session, uri):
        item = (await session.read_resource(uri)).contents[0]
        if getattr(item, "blob", None):
            return base64.b64decode(item.blob), item.mime_type
        return item.text, item.mime_type

    async def go():
        async with _session(url) as session:
            text, mime = await read_uri(session, "config://current")
            assert mime == "application/json"
            assert json.loads(text)["detection"]["model"]

            # Nothing has run yet: the resource refuses instead of serving a stub.
            with pytest.raises(Exception):
                await session.read_resource("result://latest")

            await _call(session, "load_image", {"path": SAMPLE})
            await _call(session, "run_pipeline", {})

            document = json.loads((await read_uri(session, "result://latest"))[0])
            assert document["blocks"], "a finished run must be readable"

            for uri in ("image://current", "image://annotated"):
                png, mime = await read_uri(session, uri)
                assert mime == "image/png", uri
                assert png[:8] == PNG_MAGIC, uri

            models = json.loads((await read_uri(session, "models://list"))[0])
            assert len(models["models"]) == 5
            log = json.loads((await read_uri(session, "log://mcp"))[0])
            assert log["calls"] == 2, log["calls"]          # resources are not tool calls
            assert [e["tool"] for e in log["entries"][:2]] == ["run_pipeline", "load_image"]

    asyncio.run(go())
