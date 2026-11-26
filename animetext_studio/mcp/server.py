"""P7 - the MCP server: build it, serve it, stop it.

Layout of a running server:

```
uvicorn (thread, own asyncio loop)      or      stdio (thread, own asyncio loop)
   └── HttpGateway (bearer auth + /health)         └── MCPServer.run_stdio_async()
         └── MCPServer.streamable_http_app()  ← /mcp (JSON-RPC 2.0, SSE)
               └── tools.py / resources.py  →  StudioService  →  Qt main thread
```

The server never touches Qt and never touches an engine: every tool handler hops through the
service's dispatcher, which is what keeps the GUI responsive while a page is being read and
makes the same tools work with no window at all.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .. import APP_NAME, __version__
from ..core.config import McpConfig
from ..core.types import ErrorCode, StudioError
from .resources import RESOURCE_CATALOG, register_resources
from .service import StudioService
from .tools import install_error_shaping, register_tools, tool_catalog

__all__ = ["StudioServer", "HttpGateway", "build_mcp_server", "catalog_path",
           "write_catalog", "SERVER_INSTRUCTIONS", "MCP_PATH", "security_warnings",
           "LOOPBACK_HOSTS"]

MCP_PATH = "/mcp"
DEFAULT_CATALOG = "schemas/mcp-tools.json"

SERVER_INSTRUCTIONS = f"""{APP_NAME} {__version__}: anime/manga text-block detection \
(deepghs/AnimeText_yolo) and per-box OCR (hayai-ocr-v2.7-supernova), driven through one GUI \
app state.

Typical session: get_status -> load_image(path=... or base64_png_jpg=...) -> run_pipeline \
-> get_result / get_annotated_image. `detect` is stage 1 only (boxes, no OCR).

Rules worth knowing before you call anything:
* Single-flight: one job at a time. A second run_pipeline/detect while one is running returns isError BUSY \
with the running job id in details - it is never queued. cancel_job stops it.
* Overrides passed to run_pipeline/detect are applied to the live config, so the GUI shows \
exactly what ran; get_config always describes the current state.
* Errors come back as isError results whose structuredContent is {{"code", "message", \
"details"}} - branch on `code`, not on the text.
* Everything you do is visible: each call lands in the GUI activity log, and `log://mcp` \
reads it back.
"""


#: Addresses that mean "this machine only" (P8: the warning threshold for a live endpoint).
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def security_warnings(config: McpConfig) -> list[str]:
    """Cautions about this MCP configuration, in words (P8). Empty when there is nothing to say.

    The server binds what it is told — it runs on the operator's machine, and binding a
    LAN address is a legitimate thing to want. What it must *not* do is bind quietly: an
    unauthenticated endpoint outside loopback lets any host that can reach the port drive
    the app (load a page, write a file through `save_result`, change the config).
    """
    if not config.enabled or config.transport != "streamable-http":
        return []
    if config.host in LOOPBACK_HOSTS:
        return []
    where = f"{config.host}:{config.port}"
    if config.auth_token:
        return [f"MCP endpoint {where} is reachable from other hosts; bearer auth is on."]
    return [f"MCP endpoint {where} is reachable from other hosts with NO auth token: "
            f"any host that can reach the port can drive the app. "
            f"Set mcp.auth_token (or --mcp-token), or bind 127.0.0.1."]


# ---------------------------------------------------------------------------
# Building the SDK object
# ---------------------------------------------------------------------------

def build_mcp_server(service: StudioService, *, name: str = APP_NAME,
                     version: str = __version__) -> Any:
    """An `MCPServer` with all tools and resources bound to `service`."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:                                  # pragma: no cover - mcp<2
        raise StudioError(
            ErrorCode.ENGINE_LOAD_FAILED,
            "The MCP server needs the `mcp` package (2.x).",
            {"hint": "pip install 'animetext-studio[mcp]'  (or pip install 'mcp>=2.3')",
             "reason": str(exc)},
        ) from exc

    server = MCPServer(name=name, version=version, instructions=SERVER_INSTRUCTIONS)
    register_tools(server, service)
    register_resources(server, service)
    # P8.4: argument-validation and unknown-tool failures happen inside the SDK, before any
    # handler of ours runs. The middleware is what makes them answer with the same
    # {code,message,details} as everything else — and puts a row for them in the GUI log.
    install_error_shaping(server, service)
    return server


# ---------------------------------------------------------------------------
# HTTP gateway: bearer auth + a tiny health endpoint
# ---------------------------------------------------------------------------

class HttpGateway:
    """Plain ASGI wrapper: `GET /health` answers freely, everything else needs the token.

    Deliberately not a `BaseHTTPMiddleware`: the streamable-http transport holds SSE
    connections open for the life of a session, and Starlette's HTTP middleware wraps those
    in a task that has to be unwound carefully. A raw ASGI layer sees one `http` scope per
    request and never gets in the way.
    """

    def __init__(self, app: Any, *, token: Optional[str] = None,
                 health_payload: Optional[dict[str, Any]] = None) -> None:
        self._app = app
        self._token = token or None
        self._health = health_payload or {"ok": True}
        self.rejected = 0

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in ("/health", "/healthz"):
            await self._json(send, 200, self._health)
            return
        if self._token is not None and not self._authorised(scope):
            self.rejected += 1
            await self._json(send, 401, {
                "error": "unauthorized",
                "message": "Missing or wrong bearer token. Send "
                           "'Authorization: Bearer <token>'.",
            }, extra_headers=[(b"www-authenticate", b'Bearer realm="animetext-studio"')])
            return
        await self._app(scope, receive, send)

    def _authorised(self, scope: Any) -> bool:
        for name, value in scope.get("headers", []):
            if name.lower() != b"authorization":
                continue
            text = value.decode("latin-1").strip()
            scheme, _, offered = text.partition(" ")
            if scheme.lower() == "bearer" and offered.strip() == self._token:
                return True
        return False

    @staticmethod
    async def _json(send: Any, status: int, payload: dict[str, Any],
                    extra_headers: Optional[list[tuple[bytes, bytes]]] = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = [(b"content-type", b"application/json"),
                   (b"content-length", str(len(body)).encode("ascii"))]
        headers.extend(extra_headers or [])
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class StudioServer:
    """Start/stop the MCP server for one `StudioService` (HTTP in a thread, or stdio)."""

    def __init__(self, service: StudioService, config: Optional[McpConfig] = None, *,
                 path: str = MCP_PATH) -> None:
        self._service = service
        self._config = config or McpConfig()
        self._path = path
        self._mcp: Any = None
        self._uvicorn: Any = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Future] = None
        self._error: Optional[BaseException] = None
        self._ready = threading.Event()
        self._started_at: str = ""
        self._port = 0
        self._gateway: Optional[HttpGateway] = None
        #: Cautions from :func:`security_warnings`, filled in by :meth:`start`.
        self.warnings: list[str] = []

    # -- introspection --------------------------------------------------------

    @property
    def port(self) -> int:
        return int(self.info()["port"])

    @property
    def url(self) -> str:
        return str(self.info()["url"])

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    @property
    def transport(self) -> str:
        return str(self._config.transport)

    @property
    def mcp_server(self) -> Any:
        return self._mcp

    def url(self) -> str:
        if self._config.transport != "streamable-http":
            return ""
        host = self._config.host
        display = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        return f"http://{display}:{self._port}{self._path}"

    def info(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "transport": self.transport if self.running else "none",
            "host": self._config.host if self.running else "",
            "port": self._port if self.running else 0,
            "url": self.url() if self.running else "",
            "auth_required": bool(self._config.auth_token),
            "endpoint": self._path,
            "started_at": self._started_at,
            "warnings": list(self.warnings),
        }

    def catalog(self) -> dict[str, Any]:
        """The tool/resource catalog, built from the live server object."""
        server = self._mcp or build_mcp_server(self._service)
        return tool_catalog(server, transport=self.transport, host=self._config.host,
                            port=self._config.port or self._port, resources=RESOURCE_CATALOG,
                            version=__version__)

    # -- start / stop ---------------------------------------------------------

    def start(self, *, timeout: float = 15.0) -> dict[str, Any]:
        """Bring the server up. Returns `info()`; a failed start raises `StudioError`."""
        if self.running:
            return self.info()
        if not self._config.enabled or self._config.transport == "none":
            self._service.set_mcp_info(running=False, transport="none", url="", port=0)
            return self.info()
        if self._config.transport not in ("streamable-http", "stdio"):
            raise StudioError(ErrorCode.INVALID_CONFIG,
                              f"Unknown MCP transport {self._config.transport!r}.",
                              {"known": ["streamable-http", "stdio", "none"]})

        self._mcp = build_mcp_server(self._service)
        self._error = None
        self._ready.clear()
        if self._config.transport == "streamable-http":
            self._preflight_bind()
            self._start_http()
        else:
            self._start_stdio()

        if not self._ready.wait(timeout):
            self._abort(f"the {self._config.transport} server did not come up in {timeout} s")
        if self._error is not None:
            raise StudioError(
                ErrorCode.ENGINE_LOAD_FAILED,
                f"MCP server failed to start: {self._error}",
                {"transport": self._config.transport, "host": self._config.host,
                 "port": self._config.port,
                 "hint": "is the port in use? pass --mcp-port N, or set mcp.enabled=false"},
            )
        self._started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.warnings = security_warnings(self._config)
        info = self.info()
        self._service.set_mcp_info(**info)
        for warning in self.warnings:
            print(f"[mcp] {warning}", file=sys.stderr)       # every entry point sees it
        return info

    def _abort(self, why: str) -> None:
        self.stop()
        raise StudioError(ErrorCode.ENGINE_LOAD_FAILED, f"MCP server did not start: {why}",
                          {"transport": self._config.transport, "port": self._config.port})

    # -- streamable-http ------------------------------------------------------

    def _start_http(self) -> None:
        import uvicorn

        asgi = self._mcp.streamable_http_app(streamable_http_path=self._path,
                                             host=self._config.host)
        self._gateway = HttpGateway(
            asgi, token=self._config.auth_token,
            health_payload={"app": APP_NAME, "version": __version__,
                            "mcp": self._path, "transport": "streamable-http"})
        # port 0 -> the OS picks one (tests, and "just give me a server"); uvicorn honours it.
        self._uvicorn = uvicorn.Server(uvicorn.Config(
            self._gateway, host=self._config.host, port=int(self._config.port),
            log_level="warning", lifespan="on", access_log=False,
            timeout_graceful_shutdown=2))
        self._thread = threading.Thread(target=self._http_main, name="animetext-mcp-http",
                                        daemon=True)
        self._thread.start()
        if not self._wait_for_http(timeout=15.0):
            return
        self._port = self._bound_port() or int(self._config.port)

    def _preflight_bind(self) -> None:
        """Fail on the calling thread, not inside uvicorn's: a busy port is a clean error.

        Without this the user gets "MCP server failed to start: 3" (uvicorn exits the
        process-level way), which names neither the port nor the fix.
        """
        import socket

        try:
            with socket.socket() as probe:
                probe.bind((self._config.host, int(self._config.port)))
        except OSError as exc:
            raise StudioError(
                ErrorCode.ENGINE_LOAD_FAILED,
                f"Cannot bind the MCP server to {self._config.host}:{self._config.port}: "
                f"{exc.strerror or exc}",
                {"transport": self._config.transport, "host": self._config.host,
                 "port": int(self._config.port), "errno": exc.errno,
                 "hint": "another instance is probably running: pass --mcp-port N, "
                         "or set mcp.enabled=false"},
            ) from exc

    def _http_main(self) -> None:
        try:
            assert self._uvicorn is not None
            self._uvicorn.run()
            if not self._uvicorn.started:
                self._error = RuntimeError("uvicorn exited without binding "
                                           f"{self._config.host}:{self._config.port}")
        except BaseException as exc:            # noqa: BLE001 - SystemExit included
            self._error = (exc if isinstance(exc, Exception)
                           else RuntimeError(f"{type(exc).__name__}({exc}) while running uvicorn"))
        finally:
            self._ready.set()

    def _wait_for_http(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._error is not None:
                self._ready.set()
                return False
            assert self._uvicorn is not None and self._thread is not None
            if self._uvicorn.started:
                self._ready.set()
                return True
            if not self._thread.is_alive():
                self._error = self._error or RuntimeError("the uvicorn thread exited")
                self._ready.set()
                return False
            time.sleep(0.02)
        self._error = self._error or TimeoutError(f"no bind within {timeout} s")
        return False

    def _bound_port(self) -> int:
        """The port actually bound (matters when the config asked for port 0)."""
        try:
            for server in (self._uvicorn.servers or []):
                for sock in (getattr(server, "sockets", None) or []):
                    return int(sock.getsockname()[1])
        except Exception:                                         # noqa: BLE001
            return 0
        return 0

    # -- stdio ----------------------------------------------------------------

    def _start_stdio(self) -> None:
        self._port = 0
        self._thread = threading.Thread(target=self._stdio_main, name="animetext-mcp-stdio",
                                        daemon=True)
        self._thread.start()

    def _stdio_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        task = loop.create_task(self._mcp.run_stdio_async())
        self._task = task
        try:
            self._ready.set()
            loop.run_until_complete(task)
        except (asyncio.CancelledError, SystemExit):     # stop() pulled the plug: normal
            pass
        except BaseException as exc:                              # noqa: BLE001
            self._error = exc
        finally:
            self._ready.set()
            try:
                loop.close()
            except Exception:                                     # noqa: BLE001
                pass
            self._loop = None

    # -- stop -----------------------------------------------------------------

    def stop(self, *, timeout: float = 6.0) -> None:
        thread = self._thread
        if thread is None:
            return
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        elif self._loop is not None and self._task is not None:
            try:
                asyncio.run_coroutine_threadsafe(_cancel_task(self._task),
                                                 self._loop).result(2.0)
            except Exception:                                     # noqa: BLE001
                pass
        thread.join(timeout)
        if thread.is_alive():                                  # never block an app exit
            thread.daemon = True
        self._thread = None
        self._uvicorn = None
        self._task = None
        self._port = 0
        self._service.set_mcp_info(running=False, transport="none", url="", port=0,
                                   host="", started_at="")




async def _cancel_task(task: asyncio.Future) -> None:
    """Cancel the stdio serve task from the loop that owns it, and wait for it to unwind."""
    task.cancel()
    try:
        await task
    except BaseException:                                         # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# The published catalog
# ---------------------------------------------------------------------------

def catalog_path() -> Path:
    """Where the shipped catalog lives (inside the package, next to the other schemas)."""
    return Path(__file__).resolve().parent.parent / DEFAULT_CATALOG


def catalog_document(service: Optional[StudioService] = None) -> dict[str, Any]:
    """The machine-readable catalog as a dict, built from the live tool registration.

    Deterministic on purpose (no timestamps, sorted tools): a test compares it against the
    JSON that ships in the wheel, so a tool that changes shape without regenerating the
    file fails.
    """
    service = service or StudioService(_throwaway_manager())
    return tool_catalog(build_mcp_server(service), resources=RESOURCE_CATALOG,
                        version=__version__)


def write_catalog(service: Optional[StudioService] = None,
                  path: Optional[Any] = None) -> Path:
    """Regenerate `schemas/mcp-tools.json` (or `path`) from the live tool registration."""
    target = Path(str(path)) if path else catalog_path()
    catalog = catalog_document(service)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return target


def _throwaway_manager():
    """A mock-engine `JobManager`: the catalog needs an app, not a running one."""
    from ..core.config import AppConfig
    from ..core.jobs import JobManager
    from ..core.pipeline import Pipeline, engines_from_config

    config = AppConfig.default()
    engines = engines_from_config(config, mock=True)
    return JobManager(Pipeline(engines.detection, engines.ocr, config))
