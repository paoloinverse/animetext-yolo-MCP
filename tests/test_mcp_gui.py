"""P7 - the MCP bridge driving the *visible* window.

Part 7's acceptance criterion is not "the tools answer" (that is `test_mcp_http.py`, headless)
but "a remote call is visible": it must be marshalled onto the Qt thread, move the real
widgets, feed the same progress bar a local Generate does, and leave one row in the activity
dock. So these tests stand a real HTTP server in front of a real `MainWindow`, drive it with
the official SDK client **from a worker thread**, and assert on widget state.

The worker-thread dance is the whole point: the client cannot run on the Qt thread, because
the Qt thread is the thing being called into. `drive()` keeps the main thread pumping events
while the client thread waits, which is exactly what happens when a person uses the window
while Claude is working.

Marked `gui`: deselected by default, run with `pytest -m gui` (needs PySide6, no display).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed (pip install animetext-studio[gui])")
pytest.importorskip("mcp", reason="the MCP SDK is not installed (pip install 'animetext-studio[mcp]')")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace                                   # noqa: E402

from animetext_studio.core.config import AppConfig, McpConfig        # noqa: E402
from animetext_studio.core.jobs import JobManager                    # noqa: E402
from animetext_studio.core.pipeline import Pipeline                  # noqa: E402
from animetext_studio.engines.base import FakeDetectionEngine, FakeOcrEngine  # noqa: E402
from animetext_studio.gui.app import build_service, start_mcp        # noqa: E402
from animetext_studio.gui.main_window import MainWindow              # noqa: E402

from mcp.client.session import ClientSession                         # noqa: E402
from mcp.client.streamable_http import streamable_http_client        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = str(ROOT / "examples" / "sample_page.png")


# ---------------------------------------------------------------------------
# A window with a live server in front of it
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp_gui(qtbot, free_port):
    """{window, service, server, url} - the real window, the real server, offscreen."""
    from dataclasses import replace

    config = replace(AppConfig.default(),
                     mcp=McpConfig(enabled=True, transport="streamable-http",
                                   host="127.0.0.1", port=free_port()))
    engines = SimpleNamespace(detection=FakeDetectionEngine(), ocr=FakeOcrEngine())
    manager = JobManager(Pipeline(engines.detection, engines.ocr, config))
    win = MainWindow(manager, config, mock=True)
    qtbot.addWidget(win)
    win.show()

    service = build_service(manager, config, mock=True, window=win)
    server, info = start_mcp(service, config, window=win)
    assert info["running"], info
    try:
        yield {"window": win, "service": service, "server": server, "url": info["url"]}
    finally:
        server.stop()
        service.detach_window()


def drive(qapp, url: str, script):
    """Run `script(session)` on a worker thread while the Qt thread pumps events.

    Returns whatever the coroutine returns; its exception is re-raised here so the failure
    points at the test, not at a thread nobody is watching.
    """
    box: dict = {}

    async def run():
        async with contextlib.AsyncExitStack() as stack:
            read, write = await stack.enter_async_context(streamable_http_client(url))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return await script(session)

    def worker():
        try:
            box["value"] = asyncio.run(run())
        except BaseException as exc:                          # noqa: BLE001 - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=worker, name="mcp-client", daemon=True)
    thread.start()
    deadline = time.time() + 120
    while thread.is_alive() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    thread.join(30)
    assert not thread.is_alive(), "the MCP client never finished (deadlock between threads?)"
    if "error" in box:
        raise box["error"]
    return box["value"]


async def call(session, tool: str, arguments: dict | None = None, **kwargs):
    """`call(s, "set_zoom", mode="fit")` and `call(s, "load_image", {"path": p})` both work."""
    return await session.call_tool(tool, dict(arguments or {}, **kwargs))


async def payload(session, tool: str, arguments: dict | None = None, **kwargs) -> dict:
    result = await call(session, tool, arguments, **kwargs)
    assert not result.is_error, f"{tool}: {result.structured_content}"
    return result.structured_content


def last_activity(service) -> str:
    """What the activity log says about the most recent call (`""` when there is none)."""
    event = service.activity.last()
    return f"{event.tool} {event.code} {event.message}" if event else ""


# ---------------------------------------------------------------------------
# The remote call is visible
# ---------------------------------------------------------------------------

class TestBridge:
    def test_the_badge_announces_the_server(self, mcp_gui):
        assert mcp_gui["window"].controls.mcp_badge.text().startswith("MCP: http://127.0.0.1:")

    def test_get_status_knows_it_is_driving_a_window(self, qapp, mcp_gui):
        async def script(session):
            first = await payload(session, "get_status")
            second = await payload(session, "get_status")   # the first is counted by now
            return first, second

        first, second = drive(qapp, mcp_gui["url"], script)
        assert first["gui"] is True and first["mock"] is True
        assert first["mcp"]["running"] is True
        assert second["mcp"]["calls"] == first["mcp"]["calls"] + 1, "the counter counts calls"

    def test_load_image_and_run_pipeline_land_on_the_canvas(self, qapp, mcp_gui):
        win = mcp_gui["window"]
        assert win.image is None and not win.canvas.has_result

        async def script(session):
            await payload(session, "load_image", path=SAMPLE)
            run = await payload(session, "run_pipeline", {})
            return run

        run = drive(qapp, mcp_gui["url"], script)
        assert run["block_count"] > 0
        assert win.image is not None, "the remote load must set the window image"
        assert win.canvas.has_result and len(win.canvas.blocks) == run["block_count"]
        assert win.json_pane.text.strip(), "the JSON pane must show the result"
        assert win.progress.value() == 100

    def test_set_config_moves_the_real_widgets(self, qapp, mcp_gui):
        win = mcp_gui["window"]
        before = win.controls.score_spin.value()

        async def script(session):
            return await payload(session, "set_config",
                                 patch={"detection": {"score_threshold": 0.2}})

        after = drive(qapp, mcp_gui["url"], script)
        assert after["changed"][0].startswith("detection.score_threshold")
        assert win.controls.score_spin.value() == pytest.approx(0.2)
        assert win.config.detection.score_threshold == pytest.approx(0.2)
        assert win.controls.score_spin.value() != before

    def test_view_tools_touch_the_canvas(self, qapp, mcp_gui):
        win = mcp_gui["window"]

        async def script(session):
            await payload(session, "load_image", path=SAMPLE)
            await payload(session, "run_pipeline", {})
            await payload(session, "set_zoom", mode="percent", percent=150)
            await payload(session, "set_overlay_visible", visible=False)
            await payload(session, "highlight_block", id=1)
            return await call(session, "get_annotated_image")

        shot = drive(qapp, mcp_gui["url"], script)
        assert win.canvas.zoom_percent == 150
        assert win.canvas.style_spec.overlay is False
        assert win.canvas.selected_block_id == 1
        # pixels travel as image content, metadata as structured content (never both)
        png = base64.b64decode(shot.content[0].data)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert shot.structured_content["blocks"] == len(win.canvas.blocks)
        assert shot.structured_content["bytes"] == len(png)

    def test_reset_clears_the_window(self, qapp, mcp_gui):
        win = mcp_gui["window"]

        async def script(session):
            await payload(session, "load_image", path=SAMPLE)
            await payload(session, "run_pipeline", {})
            return await payload(session, "reset")

        drive(qapp, mcp_gui["url"], script)
        assert win.image is None and not win.canvas.has_result
        assert not win.json_pane.text.strip()

    def test_save_and_export_write_files_the_app_wrote(self, qapp, mcp_gui, tmp_path):
        saved, exported = tmp_path / "result.json", tmp_path / "config.json"

        async def script(session):
            await payload(session, "load_image", path=SAMPLE)
            await payload(session, "run_pipeline", {})
            wrote = await payload(session, "save_result", path=str(saved))
            conf = await payload(session, "export_config", path=str(exported))
            return wrote, conf

        wrote, conf = drive(qapp, mcp_gui["url"], script)
        assert saved.exists() and wrote["block_count"] > 0
        assert json.loads(saved.read_text(encoding="utf-8"))["blocks"]
        assert exported.exists() and conf["path"] == str(exported)

    def test_every_call_leaves_exactly_one_row_in_the_activity_dock(self, qapp, qtbot, mcp_gui):
        win, service = mcp_gui["window"], mcp_gui["service"]
        dock = win.mcp_dock

        async def script(session):
            await payload(session, "load_image", path=SAMPLE)
            await payload(session, "get_image_info")
            bad = await call(session, "get_result")
            assert bad.is_error and bad.structured_content["code"] == "NO_IMAGE"
            return service.activity.summary()

        summary = drive(qapp, mcp_gui["url"], script)
        tools = [event.tool for event in service.activity.events()]
        assert tools == ["get_result", "get_image_info", "load_image"], tools
        assert summary["calls"] == 3 and summary["errors"] == 1
        # the log is written synchronously, the dock follows on the UI thread
        qtbot.waitUntil(lambda: dock.table.rowCount() == 3, timeout=5000)
        assert dock.table.item(0, 1).text() == "get_result", "newest row on top"
        assert dock.table.item(0, 2).text() == "NO_IMAGE", "an error row carries its code"
        assert dock.table.item(0, 3).text().isdigit(), "a duration belongs on every row"
        assert dock.last_detail(), "the detail column must say something"
        assert win.status_label.text(), "every call must say something in the status bar"

    def test_pixels_never_appear_in_the_activity_log(self, qapp, mcp_gui):
        service = mcp_gui["service"]

        async def script(session):
            blob = base64.b64encode(Path(SAMPLE).read_bytes()).decode()
            await payload(session, "load_image", base64_png_jpg=blob, file_name="pasted.png")
            return blob, service.activity.events()[0].args

        blob, logged = drive(qapp, mcp_gui["url"], script)
        assert len(blob) > 10_000, "the sample must be big enough to prove the point"
        assert blob not in json.dumps(logged), "the pixels must not be logged"
        assert len(json.dumps(logged)) < 200, logged      # a shape/length note, not the payload

    def test_closing_the_window_detaches_the_service(self, qapp, mcp_gui):
        win, service = mcp_gui["window"], mcp_gui["service"]
        assert service.window is win
        win.close()
        assert service.window is None
        assert service.gui is False

    def test_closing_the_window_mid_flash_is_silent(self, qapp, mcp_gui):
        """Regression: the hover timer belongs to the canvas, not to the event loop.

        `highlight_block` used to schedule `QTimer.singleShot(..., lambda: canvas.set_hover(-1))`;
        quitting before it fired printed `RuntimeError: ... QLabel already deleted` from
        inside the Qt loop. pytest-qt turns exactly that into a teardown error, so this test
        has nothing to assert - surviving the pump *is* the result. (The tool still reports
        the duration it was asked for, GUI or no GUI.)
        """
        win = mcp_gui["window"]

        async def script(session):
            await payload(session, "load_image", path=SAMPLE)
            await payload(session, "run_pipeline", {})
            return await payload(session, "highlight_block", id=1, duration_ms=400)

        said = drive(qapp, mcp_gui["url"], script)
        assert said["duration_ms"] == 400 and win.canvas.selected_block_id == 1
        win.close()                                  # while the flash is still pending
        time.sleep(0.5)
        qapp.processEvents()                         # the timer would fire here if it leaked

    def test_a_remote_error_does_not_disturb_the_window(self, qapp, mcp_gui):
        win = mcp_gui["window"]

        async def script(session):
            bad = await call(session, "set_zoom", mode="diagonal")
            return bad.structured_content, last_activity(mcp_gui["service"])

        error, said = drive(qapp, mcp_gui["url"], script)
        assert error["code"] == "INVALID_ARGUMENT" and error["message"]
        assert "zoom" in json.dumps(error).lower()
        assert said.startswith("set_zoom INVALID_ARGUMENT"), said
        assert win.isVisible() and win.controls.generate_button.isEnabled()
