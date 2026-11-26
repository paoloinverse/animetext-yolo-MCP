"""Engine warm-up (P8.3): load the weights, then pay for one real inference.

Why: the first inference after a load is never representative. The ONNX Runtime session builds
its CUDA context, torch picks cuDNN kernels, the allocator fills its arenas, the processor and
tokenizer objects get built on first use. A user who hits Generate on a cold app pays all of
that inside the first job, behind a progress bar that looks stuck. `--preload` moves the cost to
startup, where it is expected.

What "warm" means here is physical, not aspirational: **loaded and at least one inference has
completed**. That is what `EngineStatus.warm` reports, so `--preload` is verifiable from
`get_status` instead of being something the launcher hopes it did.

Runs through `JobManager` (see `core/jobs.py:start_warmup`), so it holds the single-flight lock
like any other job, reports on the normal progress channel, and is cancellable. It produces **no
result document**: a warm-up is not a page, and the previous result must survive it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence

from PIL import Image

from .config import AppConfig
from .types import ErrorCode, StudioError

__all__ = ["WARM_STAGE", "PROBE_SIZE", "WarmupOutcome", "WarmupReport",
           "warm_engines", "warm_devices", "DEFAULT_SWEEP_DEVICES"]

#: What `--warm-devices` sweeps when asked for "the usual ones".
DEFAULT_SWEEP_DEVICES = ("cpu", "cuda")

#: Progress stage while warming. Deliberately not one of the run stages (`detect`/`crop`/`ocr`/
#: `assemble`), so a GUI or an agent can tell "getting ready" from "working on your page".
WARM_STAGE = "warmup"

#: The probe image is tiny on purpose. Its only job is to make the backend execute; nothing
#: about its content matters and its output is thrown away. (A blank crop can make an OCR model
#: invent characters — irrelevant here, and a reason not to show probe output anywhere.)
PROBE_SIZE = 64

ProgressCallback = Callable[[str, float, str], None]


def _emit(progress: Optional[ProgressCallback], percent: float, message: str) -> None:
    """Report on the warm-up stage. A GUI slot must never be able to break a warm-up."""
    if progress is None:
        return
    try:
        progress(WARM_STAGE, float(percent), message)
    except Exception:                                         # noqa: BLE001
        pass


def _ms(then: float) -> int:
    return int(round((time.monotonic() - then) * 1000))


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")


def _is_cancelled(exc: BaseException) -> bool:
    return isinstance(exc, StudioError) and exc.code is ErrorCode.CANCELLED


def _probe_image() -> Image.Image:
    return Image.new("RGB", (PROBE_SIZE, PROBE_SIZE), "white")


def _error_dict(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, StudioError):
        return exc.to_dict()
    return StudioError(ErrorCode.INTERNAL,
                       f"Unexpected error: {exc}",
                       {"reason": f"{type(exc).__name__}: {exc}"}).to_dict()


@dataclass(frozen=True)
class WarmupOutcome:
    """One engine's warm-up: what state it reached, how long each half took, or why not."""

    engine: str
    ok: bool
    state: str = "unloaded"
    device: str = ""
    #: The device this row was *asked* for. Empty for an ordinary warm-up, which warms whatever
    #: the config resolved to and has no opinion about it. In a device sweep (P9.4) the pair
    #: `requested_device` → `device` is the whole point: "asked cuda, got cpu" is a fact about
    #: the machine that the summary line alone would hide.
    requested_device: str = ""
    loaded_models: tuple[str, ...] = ()
    load_ms: int = 0
    probe_ms: int = 0
    error: Optional[dict[str, Any]] = None

    @property
    def total_ms(self) -> int:
        """Load + first inference — the number a cold start actually costs."""
        return int(self.load_ms) + int(self.probe_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine, "ok": self.ok, "state": self.state,
            "device": self.device, "requested_device": self.requested_device,
            "loaded_models": list(self.loaded_models),
            "load_ms": self.load_ms, "probe_ms": self.probe_ms,
            "total_ms": self.total_ms, "error": self.error,
        }


@dataclass(frozen=True)
class WarmupReport:
    """What a warm-up attempt did. Emitted as data, so `--verbose` and the log can agree."""

    outcomes: tuple[WarmupOutcome, ...] = ()
    total_ms: int = 0
    probe: bool = True
    #: The devices swept, in order — non-empty only for `warm_devices()` (P9.4).
    devices: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(o.ok for o in self.outcomes)

    @property
    def failed(self) -> tuple[WarmupOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.ok)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, "probe": self.probe, "total_ms": self.total_ms,
                               "outcomes": [o.to_dict() for o in self.outcomes]}
        if self.devices:
            out["devices"] = list(self.devices)
        return out

    def table(self) -> str:
        """The per-device report as a fixed-width table (P9.4).

        Printed, not shipped: a per-device latency table is machine-specific, and nothing on the
        other side of the MCP wire can act on it (D83). What belongs on the wire is `warm` —
        which it already has.

        `requested` and `device` are two columns on purpose. When they differ, the row is telling
        you the machine did not have what you asked for; when they match, the two timings are the
        cost of a cold start on that device.
        """
        if not self.outcomes:
            return "warm-up: nothing to do"
        headers = ("requested", "engine", "device", "state", "load_ms", "probe_ms", "total_ms")
        rows = [(o.requested_device or "-", o.engine, o.device or "-", o.state,
                 str(o.load_ms), str(o.probe_ms), str(o.total_ms)) for o in self.outcomes]
        widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
        lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip(),
                 "  ".join("-" * w for w in widths)]
        for row in rows:
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        failures = self.failed
        lines.append("")
        lines.append(f"{len(self.outcomes)} row(s) over {len(self.devices) or 1} device(s)"
                     + (f", {len(failures)} failed" if failures else ", none failed")
                     + f" · {self.total_ms} ms wall")
        for bad in failures:
            code = (bad.error or {}).get("code", "?")
            lines.append(f"  {bad.engine}@{bad.requested_device or bad.device}: {code}")
        return "\n".join(lines)

    def summary(self) -> str:
        if not self.outcomes:
            return "warm-up: nothing to do"
        parts = [f"{o.engine} {'warm' if o.ok else 'FAILED'}"
                 + (f"[{o.device}]" if o.device else "")
                 + f" {o.total_ms} ms" for o in self.outcomes]
        return "warm-up: " + " · ".join(parts)

    def raise_for_failures(self) -> "WarmupReport":
        """Turn a partial warm-up into the error the job should end with (loud, not quiet).

        `--preload` failing is a real fact about the machine — usually a missing token or a
        cold cache with `--offline`. Swallowing it would leave the user staring at an app that
        looks warm and is not, so the job ends ERROR with the engine's own code, and the full
        report travels in `details` (including the engine that *did* come up).
        """
        bad = self.failed
        if not bad:
            return self
        first = bad[0]
        err = first.error or {}
        try:
            code = ErrorCode(str(err.get("code", ErrorCode.INTERNAL.value)))
        except ValueError:
            code = ErrorCode.INTERNAL
        raise StudioError(
            code,
            str(err.get("message") or f"{first.engine} warm-up failed"),
            {"engine": first.engine, "warmup": self.to_dict()},
        )


def _outcome(name: str, engine: Any, *, load_ms: int = 0, probe_ms: int = 0,
             error: Optional[dict[str, Any]] = None) -> WarmupOutcome:
    state, device, models = "unloaded", "", ()
    try:
        status = engine.status()
        raw_state = getattr(status, "state", "")
        # `.value`, not `str()`: `EngineState` is a `(str, Enum)` mixin, and `str()` of a
        # member is "EngineState.READY" on this Python — comparing that to "ready" silently
        # made every successful warm-up look like a failure.
        state = str(getattr(raw_state, "value", raw_state)).lower()
        device = str(getattr(status, "device", "") or "")
        models = tuple(str(m) for m in (getattr(status, "loaded_models", ()) or ()))
    except Exception:                                         # noqa: BLE001
        pass
    return WarmupOutcome(engine=name, ok=error is None and state == "ready", state=state,
                         device=device, loaded_models=models, load_ms=load_ms,
                         probe_ms=probe_ms, error=error)


def _attempt(name: str, engine: Any, load_ms: int, probe_ms: int,
             exc: BaseException) -> WarmupOutcome:
    """Record a failed half, but never let one engine stop the other from trying."""
    return _outcome(name, engine, load_ms=int(load_ms), probe_ms=probe_ms,
                    error=_error_dict(exc))


def warm_engines(detection: Any = None, ocr: Any = None,
                 config: Optional[AppConfig] = None, *,
                 progress: Optional[ProgressCallback] = None,
                 cancel_event: Optional[threading.Event] = None,
                 probe: bool = True) -> WarmupReport:
    """Load both engines and run one throwaway inference through each.

    Never raises for an engine that simply failed (see `WarmupReport.raise_for_failures`), but
    cancellation always propagates: a cancelled warm-up is a cancelled job, not a report.
    """
    cfg = config or AppConfig.default()
    started = time.monotonic()
    outcomes: list[WarmupOutcome] = []
    _emit(progress, 0, "warm-up: starting")

    if detection is not None:
        _check_cancel(cancel_event)
        name, t0 = "detection", time.monotonic()
        _emit(progress, 5, f"detection: loading {cfg.detection.model}")
        try:
            detection.load(cfg.detection.model, progress=progress)
        except BaseException as exc:                          # noqa: BLE001
            if _is_cancelled(exc):
                raise
            outcomes.append(_attempt(name, detection, _ms(t0), 0, exc))
        else:
            load_ms, probe_ms = _ms(t0), 0
            if probe:
                _check_cancel(cancel_event)
                _emit(progress, 45, "detection: first inference")
                t1 = time.monotonic()
                try:
                    detection.detect(_probe_image(), cfg.detection.model,
                                     iou_threshold=cfg.detection.iou_threshold,
                                     score_threshold=0.95, progress=None,
                                     cancel_event=cancel_event)
                except BaseException as exc:                  # noqa: BLE001
                    if _is_cancelled(exc):
                        raise
                    outcomes.append(_attempt(name, detection, load_ms, _ms(t1), exc))
                else:
                    probe_ms = _ms(t1)
                    outcomes.append(_outcome(name, detection, load_ms=load_ms,
                                             probe_ms=probe_ms))
            else:
                outcomes.append(_outcome(name, detection, load_ms=load_ms, probe_ms=0))

    if ocr is not None:
        _check_cancel(cancel_event)
        name, t0 = "ocr", time.monotonic()
        _emit(progress, 55, "ocr: loading")
        try:
            ocr.load(progress=progress)
        except BaseException as exc:                          # noqa: BLE001
            if _is_cancelled(exc):
                raise
            outcomes.append(_attempt(name, ocr, _ms(t0), 0, exc))
        else:
            load_ms, probe_ms = _ms(t0), 0
            if probe:
                _check_cancel(cancel_event)
                _emit(progress, 80, "ocr: first inference")
                t1 = time.monotonic()
                try:
                    ocr.ocr_crops([_probe_image()], max_patches=128, batch_size=1,
                                  max_new_tokens=8, progress=None, cancel_event=cancel_event)
                except BaseException as exc:                  # noqa: BLE001
                    if _is_cancelled(exc):
                        raise
                    outcomes.append(_attempt(name, ocr, load_ms, _ms(t1), exc))
                else:
                    probe_ms = _ms(t1)
                    outcomes.append(_outcome(name, ocr, load_ms=load_ms, probe_ms=probe_ms))
            else:
                outcomes.append(_outcome(name, ocr, load_ms=load_ms, probe_ms=0))

    report = WarmupReport(tuple(outcomes), _ms(started), probe)
    _emit(progress, 100, report.summary())
    return report


# ---------------------------------------------------------------------------
# The device sweep (P9.4)
# ---------------------------------------------------------------------------

def _release(pair: Any) -> None:
    """Free the pair one sweep row used, so the next row is measured cold. Best effort.

    "Cold" is the point of a sweep: warm the same process twice on two devices and the second
    number is measuring the first one's allocator, cuDNN handles and page cache. What this cannot
    take back it says so about — ONNX Runtime keeps process-wide state until exit, which is why
    the table is a *comparison*, not an absolute claim per device.
    """
    if pair is None:
        return
    unload_all = getattr(pair, "unload_all", None)
    if callable(unload_all):
        try:
            unload_all()
            return
        except Exception:                                       # noqa: BLE001
            pass
    for name in ("detection", "ocr"):
        engine = getattr(pair, name, None)
        if engine is None:
            continue
        try:
            engine.unload()
        except Exception:                                       # noqa: BLE001
            pass


def engine_factory(config: AppConfig, *, mock: bool = False,
                   hf_token: Optional[str] = None) -> Callable[[str], Any]:
    """Build a fresh engine pair *for one device* (P9.4).

    The device goes to both halves the way each of them takes it: the OCR engine has a `device`
    in the config, the detector takes an ONNX Runtime `provider`. One place spells that out, so a
    sweep row and a normal run cannot quietly disagree about what "on cuda" meant.
    """

    def build(device: str) -> Any:
        from .pipeline import engines_from_config

        cfg = replace(config, ocr=replace(config.ocr, device=device))
        return engines_from_config(cfg, mock=mock, hf_token=hf_token, provider=device)

    return build


def warm_devices(devices: Optional[Sequence[str]] = None, config: Optional[AppConfig] = None, *,
                 mock: bool = False, hf_token: Optional[str] = None,
                 progress: Optional[ProgressCallback] = None,
                 cancel_event: Optional[threading.Event] = None,
                 probe: bool = True, factory: Optional[Callable[[str], Any]] = None
                 ) -> WarmupReport:
    """Load and first-infer once per (engine, device), and report the table (P9.4).

    The question this exists for is the one a GPU user actually asks: *is CUDA warm, and what did
    the first inference cost on it?* An ordinary warm-up warms whatever `ocr.device` resolved to,
    once, and cannot answer it. So each device gets its own pair, built cold, warmed, measured and
    released — and the row records both the device asked for and the one the engine says it got,
    because "asked cuda, got cpu" is the answer that matters most and the easiest one to hide.

    It is a benchmark, not a startup path: it loads the weights once per device, so it costs
    memory and seconds by design. That is also why the table stays on the terminal (D83) — the
    MCP door already reports `warm`, which is the only part a remote caller can act on.
    """
    # `None` means "the usual two"; an empty list means "you forgot to say" and is refused. A
    # caller that passes no devices must never get a surprise sweep of the machine.
    requested = list(DEFAULT_SWEEP_DEVICES) if devices is None else list(devices)
    wanted = [str(d).strip() for d in requested if str(d).strip()]
    wanted = list(dict.fromkeys(wanted))               # order kept, duplicates dropped
    if not wanted:
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT, "warm_devices needs at least one device.",
            {"hint": "e.g. --warm-devices cpu,cuda", "usual": list(DEFAULT_SWEEP_DEVICES)},
        )
    cfg = config or AppConfig.default()
    build = factory or engine_factory(cfg, mock=mock, hf_token=hf_token)
    started = time.monotonic()
    outcomes: list[WarmupOutcome] = []

    for device in wanted:
        _check_cancel(cancel_event)
        _emit(progress, 0, f"{device}: building engines")
        try:
            pair = build(device)
        except BaseException as exc:                            # noqa: BLE001
            if _is_cancelled(exc):
                raise
            outcomes.append(WarmupOutcome(engine="pair", ok=False, state="error",
                                          device="", requested_device=device,
                                          error=_error_dict(exc)))
            continue

        def device_progress(_stage: str, _percent: float, _message: str = "",
                            *, _device: str = device) -> None:
            """Same channel, one prefix: a sweep's progress bar says which device it is on."""
            _emit(progress, _percent, f"{_device}: {_message}")

        try:
            report = warm_engines(getattr(pair, "detection", None), getattr(pair, "ocr", None),
                                  cfg, progress=device_progress, cancel_event=cancel_event,
                                  probe=probe)
        finally:
            _release(pair)
        outcomes.extend(replace(outcome, requested_device=device) for outcome in report.outcomes)

    return WarmupReport(outcomes=tuple(outcomes), total_ms=_ms(started), probe=probe,
                        devices=tuple(wanted))
