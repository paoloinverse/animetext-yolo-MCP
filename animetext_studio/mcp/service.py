"""P7 - `StudioService`: everything the MCP tools do, in one GUI-agnostic object.

The rule this class exists to enforce: **one implementation of each behaviour**, whether a
human clicked a button or a harness called a tool. The service holds the app state (config,
page, latest result, `JobManager`) and performs each operation *on the app-state thread*
through its `Dispatcher` - the Qt main thread when a window is attached, the caller's thread
headless. `mcp/tools.py` is then a thin, boring layer: unpack arguments, call one method,
shape the result.

Threading contract
------------------
* Every `_x()` implementation runs inside `dispatcher.submit()`: with a window that is the Qt
  main thread, so it may touch widgets freely; headless it is the MCP worker thread.
* Nothing here imports PySide6 at module scope, and nothing touches a widget unless
  `self._window` is set - so `--headless` and the unit tests never need a display.
* Jobs are started and awaited through `JobManager` only (D14): the MCP layer never calls an
  engine or the pipeline directly, which is why an MCP job drives the GUI progress bar for
  free and why `BUSY` is one policy with one message.
"""
from __future__ import annotations

import asyncio
import functools
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from .. import APP_NAME, __version__
from ..core.annotation import OverlayStyle, annotated_png
from ..core.config import (
    KNOWN_MODELS, AppConfig, load_config_file, redacted_config_dict, restart_required,
    restore_redacted, save_config_file, strip_redacted, validate_config_dict, _flatten,
)
from ..core.imaging import MAX_IMAGE_PIXELS, image_from_base64, to_png_bytes
from ..core.jobs import JobManager
from ..core.model_store import ModelStore
from ..core.pipeline import Pipeline
from ..core.types import (
    CONFIG_SCHEMA_VERSION, DETECTION_REPO, IMAGE_SOURCES, OCR_REPO, OCR_REVISION,
    SCHEMA_VERSION, ErrorCode, ImageInfo, JobState, PipelineResult, StudioError,
)
from ..engines.base import model_catalog
from .bridge import ActivityLog, Dispatcher, InlineDispatcher

__all__ = ["StudioService"]

#: How long a single dispatched call may take before the client is told the app is stuck.
DEFAULT_CALL_TIMEOUT = 20.0
#: Largest image payload `load_image(base64_png_jpg=…)` accepts, measured on the **decoded**
#: bytes (64 MiB of pixels is already a very large page).
MAX_BASE64_BYTES = 64 * 1024 * 1024
#: The same cap expressed in base64 *characters*: every 3 bytes need 4 characters, and the old
#: `MAX_BASE64_BYTES * 2` check let ~96 MB through while its own message promised 64 (P8.4).
#: Whitespace is not counted — `imaging.image_from_base64` ignores it, so neither do we.
MAX_BASE64_CHARS = (MAX_BASE64_BYTES + 2) // 3 * 4
#: `highlight_block` default pulse.
DEFAULT_HIGHLIGHT_MS = 1500
NO_RESULT_HINT = "run run_pipeline (or press Generate) first"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))

#: Inclusive range `get_annotated_image(scale=…)` accepts — the same numbers the tool's input
#: schema publishes, so the service refuses exactly what the wire refuses (P8.4).
ANNOTATED_SCALE_MIN, ANNOTATED_SCALE_MAX = 0.05, 8.0


def _annotated_scale(scale: Any, *, width: int, height: int) -> float:
    """Validate a render scale instead of silently clamping it (P8.4).

    Two reasons the clamp had to go. First, consistency: `set_zoom(percent=5)` has always been
    an INVALID_ARGUMENT, and a caller that asks for `scale: 40` and quietly gets 8.0 has learned
    nothing. Second, and less theoretical: the old clamp let `scale: 100` through as `8.0`, and
    8x of a large page is a multi-gigapixel allocation — a single tool call could OOM the app.
    The area is therefore checked against `MAX_IMAGE_PIXELS` (the same ceiling `imaging` puts on
    *input*), so the worst render is bounded by a number the app already promises to survive.
    """
    if scale is None:
        return 1.0
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"scale must be a number, got {type(scale).__name__}.",
            {"got": str(scale)[:40],
             "range": [ANNOTATED_SCALE_MIN, ANNOTATED_SCALE_MAX]},
        ) from None
    if not ANNOTATED_SCALE_MIN <= factor <= ANNOTATED_SCALE_MAX:
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"scale must be within {ANNOTATED_SCALE_MIN}..{ANNOTATED_SCALE_MAX}, got {factor}.",
            {"got": factor, "range": [ANNOTATED_SCALE_MIN, ANNOTATED_SCALE_MAX]},
        )
    pixels = int(round(width * factor)) * int(round(height * factor))
    if pixels > MAX_IMAGE_PIXELS:
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"scale {factor} on a {width}x{height} page would render {pixels:,} px "
            f"(limit {MAX_IMAGE_PIXELS:,}).",
            {"scale": factor, "image": {"width": width, "height": height},
             "would_render_px": pixels, "max_px": MAX_IMAGE_PIXELS,
             "hint": f"use scale <= {max_scale_for(width, height):.2f} on this page"},
        )
    return factor


def max_scale_for(width: int, height: int) -> float:
    """Largest scale that keeps the render inside `MAX_IMAGE_PIXELS` (never above the cap)."""
    if width <= 0 or height <= 0:
        return ANNOTATED_SCALE_MIN
    return max(ANNOTATED_SCALE_MIN,
               min(ANNOTATED_SCALE_MAX,
                   (MAX_IMAGE_PIXELS / float(width * height)) ** 0.5))


def _as_block_ids(value: Any) -> set[int]:
    """Validate `block_ids=` (P8.4): a list of ids, or a clear INVALID_ARGUMENT.

    `{int(i) for i in value}` on a bare string iterates its characters, so `block_ids="10"` used
    to mean "blocks 1 and 0" and `block_ids="abc"` raised ValueError. Over the wire pydantic
    stops most of this, but `StudioService` is also the GUI/headless façade, and "the argument
    is not the shape I asked for" is always INVALID_ARGUMENT, never a crash.
    """
    if isinstance(value, (list, tuple, set, frozenset)):
        out: set[int] = set()
        for item in value:
            if isinstance(item, bool):
                raise StudioError(ErrorCode.INVALID_ARGUMENT,
                                  "block_ids must contain integers, not booleans.",
                                  {"got": repr(item)})
            if isinstance(item, int):
                out.add(item)
                continue
            if isinstance(item, float) and float(item).is_integer():
                out.add(int(item))
                continue
            if isinstance(item, str) and item.strip().lstrip("+-").isdigit():
                out.add(int(item.strip()))
                continue
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                f"block_ids must be a list of block ids, got {type(item).__name__} inside it.",
                {"got": repr(item)[:40], "hint": "e.g. [0, 2, 5]"},
            )
        return out
    raise StudioError(
        ErrorCode.INVALID_ARGUMENT,
        f"block_ids must be a list of block ids, got {type(value).__name__}.",
        {"got": repr(value)[:40], "hint": "e.g. [0, 2, 5], or omit it for every block"},
    )


class StudioService:
    """App state + tool behaviour, dispatchable to the Qt thread, testable without one."""

    def __init__(self, manager: JobManager, config: Optional[AppConfig] = None, *,
                 dispatcher: Optional[Dispatcher] = None, mock: bool = False,
                 activity: Optional[ActivityLog] = None,
                 window: Any = None,
                 call_timeout: float = DEFAULT_CALL_TIMEOUT) -> None:
        self._manager = manager
        self._config = config or AppConfig.default()
        self._mock = bool(mock)
        self._activity = activity or ActivityLog(capacity=self._config.mcp.log_capacity)
        self._dispatcher: Dispatcher = dispatcher or InlineDispatcher()
        self._window = None
        self._image = None
        self._info: Optional[ImageInfo] = None
        self._call_timeout = float(call_timeout)
        self._mcp: dict[str, Any] = {"running": False, "transport": "none", "host": "",
                                     "port": 0, "url": "", "auth_required": False,
                                     "warnings": []}
        self._job_lock = threading.Lock()
        self._last_scale = 1.0
        if window is not None:
            self.attach_window(window)

    # -- wiring --------------------------------------------------------------

    def attach_window(self, window: Any) -> None:
        """Adopt a `MainWindow`: build the Qt bridge and make it the source of truth.

        Called from the main thread right after the window is built. From now on the page,
        the result and the config live in the window (it is what the user sees); the service
        reads them there, so a Browse click and a `load_image` call cannot disagree.
        """
        from .bridge import QtDispatcher

        self._window = window
        self._dispatcher = QtDispatcher(self._app_of(window))
        window.set_closing_hook(self._on_window_closing)
        self._push_config(self._config)

    @staticmethod
    def _app_of(window: Any) -> Any:
        from PySide6.QtCore import QCoreApplication

        # `QWidget.application()` does not exist in PySide6; the instance lookup is the
        # documented way (and returns the QApplication we are running in).
        return QCoreApplication.instance()

    def _on_window_closing(self) -> None:
        dispatcher = self._dispatcher
        if isinstance(dispatcher, InlineDispatcher) is False and hasattr(dispatcher, "close"):
            dispatcher.close()
        self._window = None

    def detach_window(self) -> None:
        """Go back to headless state (tests, and `run_gui` teardown)."""
        if isinstance(self._dispatcher, InlineDispatcher) is False:
            closer = getattr(self._dispatcher, "close", None)
            if callable(closer):
                closer()
        self._window = None
        self._dispatcher = InlineDispatcher()

    @property
    def window(self) -> Any:
        return self._window

    @property
    def gui(self) -> bool:
        return self._window is not None

    @property
    def manager(self) -> JobManager:
        return self._manager

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    @property
    def activity(self) -> ActivityLog:
        return self._activity

    @property
    def is_mock(self) -> bool:
        return self._mock

    # -- state (read on the app thread only) ---------------------------------

    @property
    def config(self) -> AppConfig:
        if self._window is not None:
            return self._window.config
        return self._config

    @property
    def image(self) -> Any:
        return self._window.image if self._window is not None else self._image

    @property
    def image_info(self) -> Optional[ImageInfo]:
        return self._window.image_info if self._window is not None else self._info

    @property
    def result(self) -> Optional[PipelineResult]:
        """The latest successful run: the window's shown result, else the manager's cache."""
        if self._window is not None and self._window.current_result is not None:
            return self._window.current_result
        return self._manager.result

    def set_mcp_info(self, **info: Any) -> None:
        """Called by `StudioServer` when it starts/stops (shows up in `get_status`)."""
        self._mcp.update(info)

    # -- dispatch + activity -------------------------------------------------

    @property
    def call_timeout(self) -> float:
        return self._call_timeout

    @property
    def mcp_transport(self) -> str:
        """Which transport the running server uses ("" before one has started)."""
        return str(self._mcp.get("transport") or "")

    def submit(self, fn: Callable[[], Any], *, timeout: Optional[float] = None):
        return self._dispatcher.submit(fn, timeout=timeout)

    def call(self, fn: Callable[[], Any], *, timeout: Optional[float] = None) -> Any:
        """Dispatch without an activity entry (multi-step tools log once, at the end)."""
        return self._dispatcher.call(fn, timeout=timeout or self._call_timeout)

    async def acall(self, fn: Callable[[], Any], *, timeout: Optional[float] = None) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.call, fn, timeout=timeout))

    def execute(self, name: str, fn: Callable[[], Any], *,
                args: Optional[dict[str, Any]] = None,
                summarize: Optional[Callable[[Any], str]] = None,
                timeout: Optional[float] = None) -> Any:
        """Run `fn` on the app thread, time it, and record one activity entry either way.

        This is the single place an MCP call becomes visible to the user (the log dock and
        the status bar listen to `ActivityLog`), and the single place a `StudioError` keeps
        its code on the way out to the wire.
        """
        started = time.perf_counter()
        try:
            value = self._dispatcher.call(fn, timeout=timeout or self._call_timeout)
        except StudioError as exc:
            self._activity.record(name, ok=False, args=args, code=exc.code.value,
                                  message=exc.message, transport=self._mcp.get("transport", ""),
                                  duration_ms=_ms(started))
            raise
        except Exception as exc:                                    # noqa: BLE001
            wrapped = StudioError(ErrorCode.INTERNAL, f"Unexpected error: {exc}",
                                  {"reason": f"{type(exc).__name__}: {exc}"})
            self._activity.record(name, ok=False, args=args, code=wrapped.code.value,
                                  message=wrapped.message,
                                  transport=self._mcp.get("transport", ""),
                                  duration_ms=_ms(started))
            raise wrapped from exc
        message = "" if summarize is not None else default_summary(value)
        if summarize is not None:
            try:
                message = str(summarize(value) or "")
            except Exception:                                       # noqa: BLE001
                message = ""
        self._activity.record(name, ok=True, args=args, message=message,
                              transport=self._mcp.get("transport", ""),
                              duration_ms=_ms(started))
        return value

    async def aexecute(self, name: str, fn: Callable[[], Any], *,
                       args: Optional[dict[str, Any]] = None,
                       summarize: Optional[Callable[[Any], str]] = None,
                       timeout: Optional[float] = None) -> Any:
        """`execute()` off the event loop (the tool handlers are async)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self.execute, name, fn, args=args,
                              summarize=summarize, timeout=timeout))

    # -- job plumbing --------------------------------------------------------

    def subscribe(self, listener: Callable[[Any], None]) -> None:
        self._manager.add_listener(listener)

    def unsubscribe(self, listener: Callable[[Any], None]) -> None:
        self._manager.remove_listener(listener)

    # =======================================================================
    # Tool implementations (run on the app thread)
    # =======================================================================

    # -- status / models ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        manager = self._manager
        job = manager.status()
        result = self.result
        store = ModelStore(self.config.paths.models_dir, offline=self.config.paths.offline)
        engines = [self._engine_info(manager.detection, "detection"),
                   self._engine_info(manager.ocr, "ocr")]
        return {
            "app": APP_NAME,
            "version": __version__,
            "result_schema_version": SCHEMA_VERSION,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "busy": bool(job["busy"]),
            "mock": self._mock,
            "offline": bool(self.config.paths.offline),
            "job": job,
            "image": self._image_dict(self.image_info),
            "engines": engines,
            "models": store.status(),
            "mcp": self._mcp_status(),
            "gui": self.gui,
            "has_result": result is not None,
            "block_count": result.block_count if result is not None else 0,
        }

    def _mcp_status(self) -> dict[str, Any]:
        summary = self._activity.summary()
        return {**self._mcp, "calls": summary["calls"], "errors": summary["errors"]}

    @staticmethod
    def _engine_info(engine: Any, kind: str) -> dict[str, Any]:
        """`EngineStatus` as JSON (never raises: a status call answers, or says nothing)."""
        info: dict[str, Any] = {"name": kind, "backend": "", "device": "", "state": "unloaded",
                                "loaded": False, "models": [],
                                "inferences": 0, "warm": False,
                                "detail": "", "model_list_is_fallback": False}
        try:
            status = engine.status()
        except Exception:                                           # noqa: BLE001
            return info
        info["backend"] = getattr(status, "backend", "") or ""
        info["device"] = getattr(status, "device", "") or ""
        models = [str(m) for m in (getattr(status, "loaded_models", ()) or ())]
        info["models"] = models
        # Enum-safe on purpose. `EngineState` is a `(str, Enum)` mixin and `str(member)` is
        # "EngineState.READY", not "ready" — the comparison this replaces had therefore been
        # answering `loaded: false` for every engine, warm or not, since P7 (found by P8.3, when
        # warm-up made the field something a caller would actually check). `.value` is the wire
        # string; `models` is no longer part of the test, because readiness is the state's job,
        # not the presence of a name.
        state = getattr(getattr(status, "state", None), "value", getattr(status, "state", ""))
        info["state"] = str(state or "")
        info["loaded"] = str(state or "").lower() == "ready"
        # `warm` is the engine's own judgement (loaded + an inference has run), not ours: the
        # whole point of P8.3 is that `--preload` can be verified instead of assumed.
        info["inferences"] = int(getattr(status, "inferences", 0) or 0)
        info["warm"] = bool(getattr(status, "warm", False))
        # P9.2: this note used to stop at the GUI's combo box. A remote caller could see five
        # variants and had no way to know the repo had not answered — the asymmetry D78 exists to
        # prevent. `detail` is the reason, the boolean is the machine-readable form of it.
        info["detail"] = str(getattr(status, "detail", "") or "")
        # From the status, never from the engine property: that one refreshes the listing, and a
        # status call must not be able to spend ten seconds on a socket.
        info["model_list_is_fallback"] = bool(getattr(status, "model_list_is_fallback", False))
        return info

    def models(self) -> dict[str, Any]:
        """Catalog + on-premises availability + what is loaded right now."""
        store = ModelStore(self.config.paths.models_dir, offline=self.config.paths.offline)
        report = store.status()
        variants = report["detection_variants"]
        names = list(KNOWN_MODELS)
        for extra in report["detection_bundled"]:
            if extra not in names:
                names.append(extra)
        sources = {n: ("bundled" if variants.get(n, {}).get("bundled") else "hf")
                   for n in names}
        loaded = ""
        try:
            in_memory = [str(m) for m in
                         (self._manager.detection.status().loaded_models or ())]
            selected = self.config.detection.model
            loaded = selected if selected in in_memory else (in_memory[-1] if in_memory else "")
        except Exception:                                           # noqa: BLE001
            loaded = ""
        entries: list[dict[str, Any]] = []
        for record in model_catalog(names, sources):
            name = record["name"]
            variant = variants.get(name, {})
            size = variant.get("bytes") or 0
            available = bool(variant.get("bundled")) or not bool(self.config.paths.offline)
            note = ""
            if not available:
                note = (f"not bundled and offline: python scripts/fetch_models.py "
                        f"--variants {name}")
            entries.append({
                **record,
                "bundled": bool(variant.get("bundled")),
                "cached": bool(variant.get("bundled")),
                "loaded": name == loaded,
                "available": available,
                "size_mb": round(size / 1_048_576, 1) if size else None,
                "note": note,
            })
        return {
            "detection_repo": DETECTION_REPO,
            "ocr_repo": OCR_REPO,
            "ocr_revision": OCR_REVISION,
            "default_model": AppConfig.default().detection.model,
            "selected_model": self.config.detection.model,
            "offline": bool(self.config.paths.offline),
            "models": entries,
            "loaded_model": loaded or None,
            "ocr": {"repo": OCR_REPO, "bundled": bool(report["ocr"].get("bundled")),
                    "revision": report["ocr"].get("revision")},
        }

    # -- config ---------------------------------------------------------------

    def config_snapshot(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(),
                "schema_version": self.config.schema_version}

    def apply_config_patch(self, patch: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(patch, dict) or not patch:
            raise StudioError(ErrorCode.INVALID_CONFIG,
                              "set_config needs a non-empty object of sections.",
                              {"hint": "e.g. {\"detection\": {\"score_threshold\": 0.3}}"})
        # A `<redacted>` placeholder in a patch means "keep what is set" (P8): a config file
        # exported for sharing must not blank out a working token when it comes back in.
        patch, kept = strip_redacted(patch)
        warnings = _patch_warnings(patch)
        if not patch:
            return {"config": self.config.to_dict(), "changed": [],
                    "warnings": warnings + [f"kept current value for {k}" for k in kept],
                    "restart_required": []}
        new_config, changed = self.config.merged(patch)
        self._push_config(new_config)
        return {"config": new_config.to_dict(), "changed": changed,
                "warnings": warnings + [f"kept current value for {k}" for k in kept],
                "restart_required": restart_required(changed)}

    def _push_config(self, config: AppConfig) -> None:
        """One config object, three consumers: the service, the window, the pipeline."""
        self._config = config
        self._manager.pipeline.config = config
        self._activity.set_capacity(config.mcp.log_capacity)
        if self._window is not None:
            self._window.set_config(config)

    def export_config(self, path: Optional[str] = None, *,
                      redact_secrets: bool = False) -> dict[str, Any]:
        """The live config, optionally written to `path`.

        `redact_secrets` (P8) swaps every secret for the `<redacted>` placeholder in the file
        *and* in the returned copy, and reports which keys it touched — a config file is the
        thing people attach to bug reports and commit by accident, and `mcp.auth_token` has no
        business being in either. Off by default: an export has to round-trip back into the
        app unchanged unless the caller asked for the shareable version.
        """
        config = self.config
        data, touched = (redacted_config_dict(config) if redact_secrets
                         else (config.to_dict(), []))
        if not path:
            return {"config": data, "path": None, "bytes": None, "redacted": touched}
        written = save_config_file(config, path, redact=redact_secrets)
        return {"config": data, "path": str(written),
                "bytes": Path(str(written)).stat().st_size, "redacted": touched}

    def import_config(self, data: Optional[dict[str, Any]] = None,
                       path: Optional[str] = None) -> dict[str, Any]:
        if (data is None) == (path is None):
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              "import_config takes exactly one of `json` or `path`.",
                              {"got_json": data is not None, "got_path": path is not None})
        before = self.config.to_dict()
        if path is not None:
            config = load_config_file(path)
            source = str(path)
        else:
            if not isinstance(data, dict):
                raise StudioError(ErrorCode.INVALID_CONFIG,
                                  "`json` must be a configuration object.",
                                  {"got": type(data).__name__})
            config = AppConfig.from_dict(data)
            source = "mcp-json"
        kept = restore_redacted(config, AppConfig.from_dict(before))
        after = config.to_dict()
        flat_before, flat_after = _flatten(before), _flatten(after)
        changed = sorted(k for k in set(flat_before) | set(flat_after)
                         if flat_before.get(k, _MISSING) != flat_after.get(k, _MISSING))
        self._push_config(config)
        return {"config": config.to_dict(), "changed": changed, "source": source,
                "kept": kept, "restart_required": restart_required(changed)}

    # -- image ----------------------------------------------------------------

    def load_image(self, path: Optional[str] = None, b64: Optional[str] = None,
                    from_clipboard: bool = False,
                   file_name: Optional[str] = None) -> dict[str, Any]:
        # "you sent two sources", "you sent none" and "you sent one that was empty" are three
        # different mistakes, and the third one used to be reported as the second (P8.4): a
        # client whose base64 encoder produced an empty string read the reply as "I did not
        # know what to do" rather than "what you handed me has nothing in it".
        given = [name for name, value in (("path", path), ("base64_png_jpg", b64),
                                         ("from_clipboard", from_clipboard))
                 if value is not None and value is not False]
        usable = [name for name, value in (("path", path), ("base64_png_jpg", b64),
                                          ("from_clipboard", from_clipboard)) if value]
        if len(usable) > 1:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                "load_image takes exactly one of `path`, `base64_png_jpg`, `from_clipboard`.",
                {"given": usable,
                 "hint": "send one source per call"},
            )
        if not usable:
            if given:
                raise StudioError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"`{given[0]}` was given but is empty.",
                    {"given": given,
                     "hint": "an empty string is not an image; send the payload or omit the key"},
                )
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                "load_image takes exactly one of `path`, `base64_png_jpg`, `from_clipboard`.",
                {"path": bool(path), "base64_png_jpg": bool(b64),
                 "from_clipboard": bool(from_clipboard)},
            )
        if from_clipboard:
            return self._load_clipboard()
        if path:
            return self._load_path(path)
        return self._load_base64(b64 or "", file_name)

    def _load_path(self, path: str) -> dict[str, Any]:
        if self._window is not None:
            if not self._window.load_path(path, source="mcp-path"):
                raise _window_failure(self._window)
            return {"image": self._image_dict(self._window.image_info), "loaded": True}
        from ..core.imaging import load_image

        image, info = load_image(path, source="mcp-path")
        self._image, self._info = image, info
        return {"image": self._image_dict(info), "loaded": True}

    def _load_base64(self, b64: str, file_name: Optional[str]) -> dict[str, Any]:
        # The cap is checked here, on the dispatched call, so an oversized payload is a normal
        # structured failure: one activity entry, one {code,message,details} answer (P8.4).
        # Counting characters rather than bytes is deliberate — the point is to refuse the
        # allocation, and the string is what has to be decoded before its size is known.
        if len(b64) > MAX_BASE64_CHARS:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                f"base64 payload is too large: {len(b64):,} characters "
                f"(limit {MAX_BASE64_CHARS:,} characters / {MAX_BASE64_BYTES // (1024 * 1024)} MB "
                f"decoded).",
                {"chars": len(b64), "max_chars": MAX_BASE64_CHARS,
                 "max_decoded_bytes": MAX_BASE64_BYTES,
                 "hint": "downscale the page, or load it from `path` on this machine"},
            )
        image, info = image_from_base64(b64, file_name=file_name, source="mcp-base64")
        if self._window is not None:
            if not self._window.load_bytes(to_png_bytes(image), file_name=file_name,
                                           source="mcp-base64"):
                raise _window_failure(self._window)
            return {"image": self._image_dict(self._window.image_info), "loaded": True}
        self._image, self._info = image, info
        return {"image": self._image_dict(info), "loaded": True}

    def _load_clipboard(self) -> dict[str, Any]:
        if self._window is None:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                "`from_clipboard` needs the GUI (the clipboard belongs to the desktop session).",
                {"hint": "send `base64_png_jpg`, or run the GUI"},
            )
        if not self._window.paste_from_clipboard():
            raise _window_failure(self._window)
        return {"image": self._image_dict(self._window.image_info), "loaded": True}

    def describe_image(self) -> dict[str, Any]:
        return {"image": self.require_image_dict()}

    def require_image_dict(self) -> dict[str, Any]:
        info = self.image_info
        if info is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No image is loaded.",
                              {"hint": "load_image(path=... | base64_png_jpg=...)"})
        return self._image_dict(info)

    @staticmethod
    def _image_dict(info: Optional[ImageInfo]) -> Optional[dict[str, Any]]:
        if info is None:
            return None
        return {"source": info.source, "file_name": info.file_name, "width": info.width,
                "height": info.height, "sha256": info.sha256}

    # -- running --------------------------------------------------------------

    def apply_run_overrides(self, overrides: Optional[dict[str, Any]]) -> list[str]:
        """Apply a `run_pipeline`/`detect` override patch to the live config.

        Deliberately *not* run-local: the GUI must show the thresholds the page was actually
        processed with, and the next run must be reproducible from `get_config` alone. A
        client that wants the old values back can `set_config` them (or `import_config`).
        """
        if not overrides:
            return []
        if not isinstance(overrides, dict):
            raise StudioError(ErrorCode.INVALID_CONFIG,
                              "Overrides must be an object of config sections.",
                              {"got": type(overrides).__name__})
        new_config, changed = self.config.merged(overrides)
        self._push_config(new_config)
        return changed

    def start_job(self, *, ocr_enabled: bool = True) -> str:
        """Start one pipeline job on the current page. Returns the job id.

        Raises `BUSY` straight through: that is the single-flight policy (D43), and the
        `details.job_id` it carries is how a second client learns what is holding the model.
        """
        image, info = self.image, self.image_info
        if image is None or info is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No image is loaded.",
                              {"hint": "load_image(path=... | base64_png_jpg=...)"})
        with self._job_lock:
            started = self._manager.start(image, info=info, ocr_enabled=ocr_enabled)
        return started.job_id

    def start_reocr(self, block_ids: Any) -> str:
        """Start a re-read of selected blocks on the loaded page (P9.3). Returns the job id.

        Two preconditions, both stated as errors rather than behaved around: a page must be loaded
        (the app keeps the document, not the pixels) and a document must exist ("which blocks?"
        has no answer before the first read). The new document replaces the old one as `latest`,
        because a re-read is a newer answer about the same page, not an edit to the old one (D81).

        BUSY passes straight through, as in `start_job`: same single-flight lock, same policy (D43).
        """
        from ..core.pipeline import block_id_list
        # Shape first: "that is not a list of ids" is true whether or not a page is loaded, and
        # answering it does not depend on state the caller may not have set up yet.
        block_id_list(block_ids)
        image, info = self.image, self.image_info
        if image is None or info is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No image is loaded.",
                              {"hint": "load_image(path=... | base64_png_jpg=...), then "
                                       "run_pipeline; reocr_blocks reads blocks of that result"})
        result = self.result
        if result is None:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT, "There is no result to re-read yet.",
                {"hint": "run_pipeline first — reocr_blocks re-reads boxes that already exist",
                 "asked_for": block_ids},
            )
        # The document is in hand, so an id that does not exist is answered here rather than
        # three seconds later as a failed job: the door can see the list, the caller cannot.
        block_id_list(block_ids, available=[block.id for block in result.blocks])
        with self._job_lock:
            started = self._manager.start_reocr(image, result, block_ids, info=info)
        return started.job_id

    def wait_for_job(self) -> dict[str, Any]:
        """Blocking wait for the running job — **callers must not dispatch this**.

        It only touches `JobManager`, which is thread-safe to interrogate from anywhere, so
        there is nothing here the app thread is needed for. Routing it through `call()`/`acall()`
        would put a *wait* on the app thread: with a window attached that freezes the GUI for the
        whole job and then reports `INTERNAL` at the dispatch timeout while the job carries on
        running (the P8.5 probes are what made this worth writing down).
        """
        self._manager.wait(self._call_timeout * 60)
        return self._manager.status()

    def job_result(self, job_id: str = "") -> PipelineResult:
        status = self._manager.status()
        if status["state"] == JobState.RUNNING.value or status["state"] == JobState.CANCELLING.value:
            raise StudioError(ErrorCode.BUSY, "The job is still running.",
                              {"job_id": job_id, "state": status["state"]})
        if status["state"] == JobState.CANCELLED.value:
            raise StudioError(ErrorCode.CANCELLED, "Job cancelled.",
                              {"job_id": job_id or status["job_id"]})
        if status["error"] is not None:
            raise StudioError.from_dict(status["error"])
        result = self.result
        if result is None:
            raise StudioError(ErrorCode.INTERNAL, "Job finished without a result.",
                              {"job_id": job_id})
        return result

    def result_payload(self, block_ids: Optional[Sequence[int]] = None) -> dict[str, Any]:
        result = self.result
        if result is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No result to return yet.",
                              {"hint": NO_RESULT_HINT})
        document = result.to_dict()
        if not block_ids:
            return {"result": document, "block_count": result.block_count,
                    "filtered": False}
        wanted = _as_block_ids(block_ids)
        blocks = [b for b in document["blocks"] if b["id"] in wanted]
        missing = sorted(wanted - {b["id"] for b in blocks})
        if not blocks:
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              "None of the requested block ids are in the result.",
                              {"requested": sorted(wanted), "missing": missing,
                               "available": [b["id"] for b in document["blocks"]]})
        document["blocks"] = blocks
        return {"result": document, "block_count": len(blocks), "filtered": True}

    def save_result(self, path: str) -> dict[str, Any]:
        result = self.result
        if result is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No result to save.",
                              {"hint": NO_RESULT_HINT})
        if not str(path).strip():
            raise StudioError(ErrorCode.INVALID_ARGUMENT, "`path` must not be empty.",
                              {"hint": "save_result(path='/abs/path/result.json')"})
        target = Path(str(path)).expanduser()
        try:
            if target.parent and not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            text = result.to_json()
            target.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            raise StudioError(ErrorCode.IO_ERROR, f"Cannot write {path}",
                              {"reason": str(exc)}) from exc
        return {"path": str(target), "bytes": target.stat().st_size,
                "block_count": result.block_count}

    # -- annotated image ------------------------------------------------------

    def annotated(self, scale: Optional[float] = None) -> tuple[bytes, dict[str, Any]]:
        """The overlay PNG, rendered in `core/` so it works with no window at all."""
        image = self.image
        result = self.result
        if image is None or result is None:
            raise StudioError(ErrorCode.NO_IMAGE,
                              "Nothing to annotate: load an image and run the pipeline.",
                              {"hint": NO_RESULT_HINT})
        style = (self._window.canvas.style_spec if self._window is not None
                 else OverlayStyle.from_config(self.config.gui))
        png = annotated_png(image, result.blocks, style)
        width, height = image.width, image.height
        applied = 1.0
        factor = _annotated_scale(scale, width=width, height=height)
        if abs(factor - 1.0) > 1e-3:
            from PIL import Image

            resized = image.resize((max(1, int(round(width * factor))),
                                    max(1, int(round(height * factor)))),
                                   Image.LANCZOS)
            png = annotated_png(resized, result.blocks, style)
            width, height, applied = resized.width, resized.height, factor
        self._last_scale = applied
        # The pixels go out as an MCP image block; the structured content is metadata only.
        meta = {"mime_type": "image/png", "width": width, "height": height,
                "bytes": len(png), "blocks": len(result.blocks), "scale": applied}
        return png, meta

    # -- view -----------------------------------------------------------------

    def set_overlay(self, visible: Optional[bool] = None, labels: Optional[bool] = None,
                     fill_opacity: Optional[float] = None) -> dict[str, Any]:
        gui: dict[str, Any] = {}
        if visible is not None:
            gui["overlay_visible"] = bool(visible)
        if labels is not None:
            gui["labels_visible"] = bool(labels)
        if fill_opacity is not None:
            gui["fill_opacity"] = _clamp(fill_opacity, 0.0, 1.0)
        if not gui:
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              "set_overlay_visible needs at least one of `visible`, "
                              "`labels`, `fill_opacity`.")
        applied = self.apply_config_patch({"gui": gui})
        current = applied["config"]["gui"]
        return {"gui": current, "overlay_visible": current["overlay_visible"],
                "labels_visible": current["labels_visible"],
                "fill_opacity": current["fill_opacity"]}

    def set_zoom(self, mode: str, percent: Optional[int] = None) -> dict[str, Any]:
        if mode not in ("fit", "percent"):
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              f"zoom mode must be 'fit' or 'percent', got {mode!r}.")
        gui: dict[str, Any] = {"zoom_mode": mode}
        if mode == "percent":
            value = int(percent) if percent is not None else self.config.gui.zoom_percent
            if not 10 <= value <= 800:
                raise StudioError(ErrorCode.INVALID_ARGUMENT,
                                  "zoom percent must be within 10..800.", {"got": value})
            gui["zoom_percent"] = value
        self.apply_config_patch({"gui": gui})
        canvas = self._window.canvas if self._window is not None else None
        return {"zoom_mode": gui["zoom_mode"],
                "zoom_percent": int(canvas.zoom_percent if canvas is not None
                                    else self.config.gui.zoom_percent),
                "scale": round(float(canvas.current_scale() if canvas is not None else 1.0), 4)}

    def highlight(self, block_id: int, duration_ms: Optional[int] = None) -> dict[str, Any]:
        result = self.result
        if result is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No result: nothing to highlight.",
                              {"hint": NO_RESULT_HINT})
        block = next((b for b in result.blocks if b.id == int(block_id)), None)
        if block is None:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT, f"Block {block_id} is not in the current result.",
                {"available": [b.id for b in result.blocks]},
            )
        duration = int(duration_ms) if duration_ms is not None else DEFAULT_HIGHLIGHT_MS
        if self._window is not None:
            canvas = self._window.canvas
            canvas.select_block(int(block_id), emit=True)   # syncs table + JSON pane too
            canvas.flash_hover(int(block_id), max(50, duration))
        return {"ok": True, "block_id": int(block_id), "found": True,
                "duration_ms": duration,
                "text_preview": (block.text or "")[:80]}

    def reset(self) -> dict[str, Any]:
        cleared: list[str] = []
        if self._manager.is_busy:
            raise StudioError(ErrorCode.BUSY, "Cannot reset while a job is running.",
                              {"job_id": self._manager.job_id,
                               "hint": "cancel_job first"})
        self._manager.clear_result()
        cleared.append("result")
        if self._window is not None:
            self._window.clear_image()
            cleared += ["image", "canvas", "json", "table"]
        else:
            self._image, self._info = None, None
            cleared += ["image"]
        return {"ok": True, "cleared": cleared}

    def cancel(self) -> dict[str, Any]:
        manager = self._manager
        job_id = manager.job_id
        cancelled = manager.cancel()
        return {"cancelled": bool(cancelled), "job_id": job_id,
                "state": manager.state.value}

    def unload(self) -> dict[str, Any]:
        names = [str(getattr(e, "name", "")) for e in
                 (self._manager.detection, self._manager.ocr)]
        self._manager.unload_engines()
        # P8: unloading has to be *visible*, not just a row in the activity log — the
        # window repaints its engines badge from the engines, so ask it after the fact.
        refresh = getattr(self._window, "refresh_engine_badge", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:                                          # noqa: BLE001
                pass                    # a badge must never fail a tool call
        return {"unloaded": [n for n in names if n], "busy": self._manager.is_busy}

    def warm_engines(self, *, probe: Optional[bool] = None) -> dict[str, Any]:
        """Start the engine warm-up (P8.3) and answer with the job handle, not with a wait.

        `unload_engines` has had no counterpart since P7: an agent that freed the models to lend
        the GPU to something else could only bring them back by running a page it did not want.
        This is `--preload` over the wire. It is a job — single-flight, progress-reporting,
        cancellable — so `BUSY` here means exactly what it means for `run_pipeline`.

        `probe: false` loads the weights without the throwaway inference. From a cold engine that
        leaves `loaded: true, warm: false` — the honest difference, and the reason the two flags
        are separate. (An engine that was already warm stays warm: `warm` describes the current
        load, and an inference has already run against it.)
        """
        do_probe = True if probe is None else bool(probe)
        started = self._manager.start_warmup(probe=do_probe)
        return {
            "job_id": started.job_id,
            "kind": "warmup",
            "state": str(getattr(started.state, "value", started.state)),
            "busy": True,
            "probe": do_probe,
        }

    # -- resources ------------------------------------------------------------

    def config_document(self) -> str:
        return json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False)

    def result_document(self) -> str:
        result = self.result
        if result is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No result published yet.",
                              {"hint": NO_RESULT_HINT})
        return result.to_json()

    def image_png(self) -> bytes:
        image = self.image
        if image is None:
            raise StudioError(ErrorCode.NO_IMAGE, "No image loaded.",
                              {"hint": "load_image(path=... | base64_png_jpg=...)"})
        return to_png_bytes(image)

    def annotated_png_bytes(self) -> bytes:
        return self.annotated(None)[0]

    def models_document(self) -> str:
        payload = self.models()
        return json.dumps({"detection_repo": payload["detection_repo"],
                           "ocr_repo": payload["ocr_repo"],
                           "models": payload["models"],
                           "default_model": payload["default_model"],
                           "selected_model": payload["selected_model"],
                           "offline": payload["offline"]},
                          indent=2, ensure_ascii=False)

    def log_document(self, limit: Optional[int] = None) -> str:
        summary = self._activity.summary()
        return json.dumps({"entries": self._activity.entries(limit or 50),
                           "calls": summary["calls"], "errors": summary["errors"],
                           "capacity": summary["capacity"]},
                          indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MISSING = object()


def default_summary(value: Any) -> str:
    """One line describing a successful result, for the dock and the status bar.

    The human watching the window should not have to open `log://mcp` to learn what a call
    did. Every tool gets this for free; a tool with something better to say passes
    `summarize=`. Anything unrecognised stays empty (the row then reads `ok`).
    """
    if not isinstance(value, dict):
        return ""
    document = value.get("result") if isinstance(value.get("result"), dict) else None
    if document is not None and "blocks" in document:
        total = (document.get("timing_ms") or {}).get("total")
        return (f"{len(document['blocks'])} blocks in {total} ms"
                if total is not None else f"{len(document['blocks'])} blocks")
    if isinstance(value.get("path"), str) and "bytes" in value:      # save_result
        return f"{value['path']} ({value['bytes']} bytes)"
    if "app" in value and "job" in value:                             # get_status
        job = value.get("job") or {}
        state = str(job.get("state") or "idle")
        return f"{state} {job.get('percent', 0):.0f} %" if value.get("busy") else state
    if "zoom_mode" in value:                                          # set_zoom
        return (f"zoom {value['zoom_mode']}"
                + (f" {value['zoom_percent']} %" if value.get("zoom_mode") == "percent" else ""))
    if "block_id" in value:                                           # highlight_block
        preview = str(value.get("text_preview") or "").strip()
        return f"block {value['block_id']}" + (f": {preview[:24]}" if preview else "")
    if "overlay_visible" in value:                                    # set_overlay_visible
        return (f"overlay {'on' if value['overlay_visible'] else 'off'}, "
                f"labels {'on' if value.get('labels_visible') else 'off'}")
    for key, word in (("block_count", "blocks"), ("count", "detections")):
        if key in value:
            return f"{value[key]} {word}"
    for key in ("changed", "cleared", "unloaded"):
        if key in value and isinstance(value[key], list):
            return f"{key}: {', '.join(map(str, value[key])) or 'nothing'}"
    if "cancelled" in value:
        return f"cancelled {value.get('job_id') or 'job'}" if value["cancelled"] else "nothing running"
    image = value.get("image")
    if isinstance(image, dict) and "width" in image:
        name = image.get("file_name") or image.get("source") or "image"
        return f"{image['width']}x{image['height']} {name}"
    return ""


def _ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))


def _window_failure(window: Any) -> StudioError:
    """A `MainWindow` loader returned False: it already told the user, we tell the client."""
    text = ""
    try:
        text = window.status_label.text()
    except Exception:                                               # noqa: BLE001
        pass
    for code in ErrorCode:
        if text.startswith(f"{code.value}:"):
            return StudioError(code, text.split(": ", 1)[-1] or text)
    return StudioError(ErrorCode.INVALID_IMAGE, text or "The image could not be loaded.")


def _patch_warnings(patch: dict[str, Any]) -> list[str]:
    """Unknown-key warnings for a partial config, without failing the call.

    `AppConfig.merged()` validates the *merged* document, so an unknown key inside a patch is
    silently absorbed by the defaults it lands on. Validating the patch on its own turns that
    into the warning the §2.2 semantics promise (unknown key = warning, invalid value =
    error, and the error still comes from `merged()`).
    """
    try:
        _errors, warnings = validate_config_dict(patch, strict=False)
    except StudioError:
        return []
    return [f"{w.get('path')}: {w.get('message')}" for w in warnings]
