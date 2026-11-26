"""Composition root for the GUI (Part 5).

Kept separate from `main_window.py` so the window can be constructed in tests with any
`JobManager` (mock engines, gated stubs) and nothing here is imported unless the user
actually asks for the GUI.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

from ..core.config import AppConfig
from ..core.jobs import JobManager
from ..core.pipeline import Pipeline, engines_from_config
from ..core.types import ErrorCode, StudioError

QT_HINT = "pip install 'animetext-studio[gui]'   (PySide6)"


def build_service(manager: JobManager, config: AppConfig, *, mock: bool = False,
                  window=None):
    """The MCP service for a window (P7): state + bridge + activity log, dock attached.

    Split out of `run_gui` so a test (or a harness embedding the window) can bring up the
    exact same MCP surface without starting an event loop.
    """
    from ..mcp.service import StudioService
    from .activity import attach_activity_dock

    service = StudioService(manager, config, mock=mock)
    if window is not None:
        service.attach_window(window)
        window.mcp_dock = attach_activity_dock(window, service)
    return service


def start_mcp(service, config: AppConfig, *, window=None) -> tuple[Any, dict]:
    """Start the MCP server for `service`; never fails the GUI if it cannot bind."""
    from ..mcp.server import StudioServer

    server = StudioServer(service, config.mcp)
    try:
        info = server.start()
    except StudioError as exc:
        if window is not None:
            window.controls.set_mcp_status("MCP: failed")
            window._say(f"{exc.code.value}: {exc.message}")
        return server, {"running": False, "transport": "none", "error": exc.to_dict()}
    if window is not None:
        if info.get("running"):
            where = info["url"] or f"{info['transport']}"
            window.controls.set_mcp_status(
                f"MCP: {where}" + ("  [auth]" if info.get("auth_required") else ""))
            window._say(f"MCP server listening on {where}")
            # P8: a bind outside loopback is the operator's call, but never a quiet one.
            for warning in info.get("warnings") or []:
                window.controls.set_mcp_status(f"MCP: {where}  [!]"
                                               + ("  [auth]" if info.get("auth_required")
                                                  else ""))
                window._say(f"MCP: {warning}")
        else:
            window.controls.set_mcp_status("MCP: off")
    return server, info


def build_manager(config: AppConfig, *, mock: bool = False,
                  hf_token: Optional[str] = None) -> JobManager:
    """Engines + pipeline + job manager, wired once and shared by the whole app.

    The manager owns the engines for the lifetime of the process: the GUI, the MCP server
    (P7) and `--run` all reuse one loaded model instead of each paying the load cost.
    """
    engines = engines_from_config(config, mock=mock, hf_token=hf_token)
    return JobManager(Pipeline(engines.detection, engines.ocr, config))


def run_gui(args, config: AppConfig, *, smoke: bool = False) -> int:
    """Build the window and run the Qt event loop. Returns the process exit code.

    `smoke=True` (`--gui-smoke`) builds and lays out the window, processes one round of
    events and exits 0: the packaging/CI question "does the GUI come up, with these
    engines and this config?" without needing a display or a human.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise StudioError(ErrorCode.ENGINE_LOAD_FAILED,
                          "The GUI needs PySide6, which is not installed.",
                          {"hint": QT_HINT, "reason": str(exc),
                           "steps": ["1. " + QT_HINT,
                                     "2. or run headless: --headless --run --image FILE"]}) from exc

    from .main_window import MainWindow

    manager = build_manager(config, mock=getattr(args, "mock", False),
                            hf_token=getattr(args, "hf_token", None))
    app = QApplication.instance() or QApplication(list(sys.argv[:1]))
    app.setApplicationName("AnimeText OCR Studio")
    window = MainWindow(manager, config, mock=getattr(args, "mock", False))
    service = build_service(manager, config, mock=getattr(args, "mock", False),
                            window=window)
    server, mcp_info = start_mcp(service, config, window=window)
    window.mcp_service, window.mcp_server = service, server
    window.show()
    if getattr(args, "image", None):
        window.load_path(args.image)
    # `--preload` (P8.3): the flag used to exist and do nothing. It now warms the engines on
    # the job worker *after* the window is up, so the first Generate does not pay for the CUDA
    # context and cuDNN kernel selection behind a progress bar that looks stuck. Skipped under
    # --mock (fake engines are already warm) and under --gui-smoke (the loop is about to end).
    if getattr(args, "preload", False) and not smoke:
        if getattr(args, "mock", False):
            window._say("preload: skipped (mock engines)")
        else:
            window.start_preload()
    elif getattr(args, "preload_off", False) and getattr(args, "verbose", False):
        # The flag's whole job is to name the baseline, so say which one this run is (P9.4).
        print("preload: off (--preload-off)", file=sys.stderr)
    if getattr(args, "verbose", False):
        plan = manager.ocr.plan()
        print(f"OCR plan: bundled={plan['bundled']} device={plan['requested_device']} "
              f"graphs={plan['use_cuda_graph']} offline={plan['offline']}", file=sys.stderr)
    if smoke:
        try:
            app.processEvents()
            size = window.size()
            where = mcp_info.get("url") or mcp_info.get("transport", "")
            mcp_part = f"on {where}" if mcp_info.get("running") else "off"
            print(f"GUI smoke OK: {window.objectName() or 'MainWindow'} "
                  f"{size.width()}x{size.height()}, "
                  f"models={window.controls.model_count}, "
                  f"image={'yes' if window.has_image else 'no'}, "
                  f"engines={'mock' if window.is_mock else 'real'}, "
                  f"canvas+json=ready, zoom={window.canvas.zoom_label.text()}, "
                  f"mcp={mcp_part}")
            window.close()
            manager.wait(2.0)
            return 0
        finally:
            server.stop()
            window.job_bridge.shutdown()
    try:
        return int(app.exec())
    finally:
        server.stop()
        window.job_bridge.shutdown()
        manager.wait(2.0)


def not_installed_error() -> StudioError:
    return StudioError(ErrorCode.ENGINE_LOAD_FAILED, "PySide6 is not installed.",
                       {"hint": QT_HINT,
                        "steps": ["1. " + QT_HINT,
                                  "2. or run headless: --headless --run --image FILE"]})
