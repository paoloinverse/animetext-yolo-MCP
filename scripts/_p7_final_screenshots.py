
"""Offscreen GUI + live MCP session -> screenshots (the Part 7 acceptance, visual half)."""
import asyncio, contextlib, logging, os, threading, time
logging.disable(logging.CRITICAL)          # the SDK's INFO logs swallow our own prints
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path as P
from types import SimpleNamespace
from dataclasses import replace
from PySide6.QtWidgets import QApplication
from animetext_studio.core.config import AppConfig, McpConfig
from animetext_studio.core.jobs import JobManager
from animetext_studio.core.pipeline import Pipeline
from animetext_studio.engines.base import FakeDetectionEngine, FakeOcrEngine
from animetext_studio.gui.app import build_service, start_mcp
from animetext_studio.gui.main_window import MainWindow
import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()

app = QApplication.instance() or QApplication([])
cfg = replace(AppConfig.default(), mcp=McpConfig(enabled=True, transport="streamable-http",
                                                 host="127.0.0.1", port=port))
eng = SimpleNamespace(detection=FakeDetectionEngine(), ocr=FakeOcrEngine())
mgr = JobManager(Pipeline(eng.detection, eng.ocr, cfg))
win = MainWindow(mgr, cfg, mock=True); win.resize(1500, 950); win.show()
service = build_service(mgr, cfg, mock=True, window=win)
server, info = start_mcp(service, cfg, window=win)
out = P("out/p7_final"); out.mkdir(parents=True, exist_ok=True)
app.processEvents(); win.grab().save(str(out / "01_before.png"))

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
box = {}
async def run():
    async with contextlib.AsyncExitStack() as st:
        r, w = await st.enter_async_context(streamable_http_client(info["url"]))
        s2 = await st.enter_async_context(ClientSession(r, w))
        await s2.initialize()
        steps = [("load_image", {"path": "examples/sample_page.png"}),
                 ("run_pipeline", {}),
                 ("set_config", {"patch": {"detection": {"score_threshold": 0.2}}}),
                 ("set_zoom", {"mode": "percent", "percent": 110}),
                 ("highlight_block", {"id": 2, "duration_ms": 60000}),
                 ("get_annotated_image", {}),
                 ("set_overlay_visible", {"labels": False}),
                 ("get_result", {})]
        for name, args in steps:
            res = await s2.call_tool(name, args)
            if res.is_error:
                box["error"] = (name, res.structured_content)
                return
        box["ok"] = True
def worker():
    try:
        box["value"] = asyncio.run(run())
    except BaseException as exc:
        box["error"] = repr(exc)
t = threading.Thread(target=worker, daemon=True); t.start()
while t.is_alive():
    app.processEvents(); time.sleep(0.005)
t.join(20)
for _ in range(40):
    app.processEvents(); time.sleep(0.01)
print("session:", box)
win.grab().save(str(out / "02_after.png"))
win.canvas.grab().save(str(out / "03_canvas.png"))
win.mcp_dock.grab().save(str(out / "04_activity_dock.png"))
print("badge:", win.controls.mcp_badge.text())
print("status:", win.status_label.text())
print("dock rows:", win.mcp_dock.table.rowCount())
print("summary:", win.mcp_dock.summary.text())
print("rows:")
for row in range(win.mcp_dock.table.rowCount()):
    print("   ", " | ".join((win.mcp_dock.table.item(row, c).text()
                             if win.mcp_dock.table.item(row, c) else "") for c in range(5)))
print("zoom:", win.canvas.zoom_percent, "| overlay:", win.canvas.style_spec.overlay,
      "| selected:", win.canvas.selected_block_id, "| blocks:", len(win.canvas.blocks))
server.stop(); service.detach_window()
