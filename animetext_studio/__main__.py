"""Entry point. Part 1 provides the foundation actions; later parts plug in:

- Part 4: --headless --run  (pipeline)
- Part 5: GUI (default action)
- Part 7: MCP server startup

Unimplemented paths return NOT_IMPLEMENTED (exit 20) — never silent success
(decision D8).
"""
from __future__ import annotations

import json
import sys
from typing import Optional, Sequence

from . import APP_NAME, __version__
from .core.cli import parse_args
from .core.config import AppConfig, resolve_config, save_config_file, validate_config_file
from .core.model_store import ModelStore
from .core.types import ErrorCode, StudioError


def _print_error(err: StudioError) -> None:
    print(json.dumps({"error": err.to_dict()}, ensure_ascii=False), file=sys.stderr)


def _doctor_action(args, config) -> int:
    """`--doctor` (P9.5): the read-only environment report, prose first, JSON on --verbose.

    The token is passed through *unresolved* — the report's whole point is to say where it
    came from (``--hf-token``, ``$HF_TOKEN``, a local ``HF_token.txt``), and resolving it
    here would flatten three honest answers into one. Only the source and the length of a
    token ever reach the report; the value never does.
    """
    from .core.doctor import run_doctor

    report = run_doctor(config, mock=args.mock, hf_token=args.hf_token)
    print(report.text())
    if args.verbose:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return report.exit_code


def _warm_devices_action(args, config) -> int:
    """`--warm-devices`: the per-device cold-start table (P9.4). It prints, then exits.

    Loads and releases engines; writes nothing. The table goes to stdout, progress to stderr, so
    `--warm-devices cpu,cuda > warm.txt` captures the numbers. A device that failed to warm is
    reported in the table *and* decides the exit code — a benchmark whose GPU row is a failure is
    not a benchmark that succeeded.
    """
    from .core.config import DEVICE_RE
    from .core.warmup import DEFAULT_SWEEP_DEVICES, warm_devices

    raw = str(args.warm_devices or "").strip()
    devices = ([str(d) for d in DEFAULT_SWEEP_DEVICES]
               if raw.lower() in ("all", "*")
               else [part.strip() for part in raw.split(",") if part.strip()])
    bad = [d for d in devices if not DEVICE_RE.fullmatch(d)]
    if not devices or bad:
        err = StudioError(
            ErrorCode.INVALID_ARGUMENT, "--warm-devices needs device names (e.g. cpu,cuda).",
            {"asked": raw, "unknown": bad,
             "expected": "auto | cpu | cuda | cuda:N, comma-separated, or 'all'"},
        )
        _print_error(err)
        return err.exit_code

    def say(_stage: str, _percent: float, message: str = "") -> None:
        print(f"warm-devices: {message}", file=sys.stderr)

    try:
        report = warm_devices(devices, config, mock=args.mock, hf_token=args.hf_token,
                              progress=say)
    except StudioError as exc:
        _print_error(exc)
        return exc.exit_code

    print(report.table())
    if args.verbose:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    try:
        report.raise_for_failures()
    except StudioError as exc:
        _print_error(exc)
        return exc.exit_code
    return 0


def _run_headless(args, config) -> int:
    """`--run`: the same `JobManager` path the GUI and the MCP server use (Part 4).

    stdout is the result document and nothing else, so `--run > result.json` works;
    progress (--verbose) and the "written to" note go to stderr.
    """
    from .core.jobs import JobManager
    from .core.pipeline import Pipeline, engines_from_config

    if not args.image:
        err = StudioError(ErrorCode.NO_IMAGE, "--run needs an image to process.",
                          {"hint": "add --image PATH (or --mock with any image)"})
        _print_error(err)
        return err.exit_code

    try:
        engines = engines_from_config(config, mock=args.mock, hf_token=args.hf_token)
        manager = JobManager(Pipeline(engines.detection, engines.ocr, config))
        if args.verbose:
            manager.add_listener(lambda p: print(f"[{p.stage}] {p.percent:5.1f}%  {p.message}",
                                                 file=sys.stderr))
        result = manager.run_file(args.image)
    except StudioError as exc:
        _print_error(exc)
        return exc.exit_code

    document = result.to_json()
    print(document)

    if args.out:
        from pathlib import Path

        try:
            path = Path(args.out).expanduser()
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(document + "\n", encoding="utf-8")
        except OSError as exc:
            err = StudioError(ErrorCode.IO_ERROR, f"Cannot write {args.out}",
                              {"reason": str(exc)})
            _print_error(err)
            return err.exit_code
        print(f"Result written to {path}", file=sys.stderr)
    return 0


def _serve_headless(args, config) -> int:
    """`--serve` / `--mcp-transport stdio`: the MCP server, no window (PLAN §3 Part 7).

    Same `JobManager`, same `StudioService` as the GUI - only the dispatcher is different
    (inline instead of the Qt thread), which is exactly why the tools behave identically.
    stdout is the protocol when the transport is stdio, so every word this command says goes
    to stderr.
    """
    import signal
    import threading

    from .core.jobs import JobManager
    from .core.pipeline import Pipeline, engines_from_config
    from .mcp.server import StudioServer
    from .mcp.service import StudioService

    stop = threading.Event()

    def _stop(signum, _frame):                       # noqa: ANN001 - signal handler
        stop.set()

    try:
        engines = engines_from_config(config, mock=args.mock, hf_token=args.hf_token)
        manager = JobManager(Pipeline(engines.detection, engines.ocr, config))
        service = StudioService(manager, config, mock=bool(args.mock))
        server = StudioServer(service, config.mcp)
        info = server.start()
    except StudioError as exc:
        _print_error(exc)
        return exc.exit_code

    print(f"MCP {info['transport']} serving" + (f" at {info['url']}" if info.get("url")
                                                else " on stdio")
          + (" (auth)" if info.get("auth_required") else "")
          + (" [mock engines]" if args.mock else ""), file=sys.stderr)
    if getattr(args, "preload", False) and not args.mock:
        # Same job the GUI runs, so an MCP client sees the truth in `get_status` (kind=warmup)
        # and a cold first request does not silently pay for the load. On a thread, because
        # "warm the engines" must never delay the first `initialize`.
        def _warm() -> None:
            try:
                manager.start_warmup()
                done = manager.wait()
                report = manager.warmup_report
                if report is not None and done.state.value == "done":
                    print(report.summary(), file=sys.stderr)
                elif done.error is not None:
                    print(f"preload: {done.error.code.value}: {done.error.message}",
                          file=sys.stderr)
            except Exception as exc:                              # noqa: BLE001
                print(f"preload: {exc}", file=sys.stderr)         # never kills the server

        threading.Thread(target=_warm, name="animetext-preload", daemon=True).start()

    for number in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(number, _stop)
        except (ValueError, OSError):    # not the main thread, or the platform says no
            pass
    try:
        while not stop.is_set() and server.running:
            stop.wait(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        manager.wait(2.0)
    return 0


def _dump_mcp_tools(target: str) -> int:
    """`--dump-mcp-tools [PATH|-]`: regenerate / print the machine-readable tool catalog.

    PATH is where the copy goes (D94). It used to be dropped on the floor — the value was
    passed positionally into `write_catalog(service, path)`, so every call rewrote the catalog
    that *ships in the wheel* instead, and a generator that repairs the artifact it is meant to
    be checked against is a generator with no check at all.
    """
    from pathlib import Path

    from .mcp.server import write_catalog

    try:
        path = write_catalog(path=None if target == "-" else target)
        text = Path(str(path)).read_text(encoding="utf-8")
    except StudioError as exc:
        _print_error(exc)
        return exc.exit_code
    if target == "-":
        print(text)
        return 0
    print(f"MCP tool catalog written to {path}")
    return 0


#: Options whose value decides *what this command does*. Typing `--warm-devices ""` gave us
#: something: an empty value. Treating that as "the flag was not given" made the default action
#: run behind the user's back — a window opened instead of an answer, and `--out ""` quietly
#: skipped the write it was asked to make (found by the P9.6 sign-off run, D93).
#:
#: Deliberately not here: `--config`/`--models-dir`/`--hf-token`, where an empty value already
#: means "unset, use the next source", and `--mcp-token ""`, which legitimately clears a token.
#: An option that *chooses an action* has no such reading: doing nothing is not what it asked for.
EMPTY_VALUE_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("--warm-devices", "warm_devices", "name at least one device, e.g. `cpu,cuda` or `all`"),
    ("--export-config", "export_config", "name the file to write the config to"),
    ("--validate-config", "validate_config", "name the config file to check"),
    ("--dump-mcp-tools", "dump_mcp_tools", "a path to write to, or `-` for stdout"),
    ("--out", "out", "name the file to write the result document to"),
)


def _empty_value_error(args) -> Optional[StudioError]:
    """The one refusal that has to happen before any action is chosen (D93)."""
    for option, dest, hint in EMPTY_VALUE_OPTIONS:
        value = getattr(args, dest, None)
        if value is not None and not str(value).strip():
            return StudioError(
                ErrorCode.INVALID_ARGUMENT,
                f"{option} was given an empty value.",
                {"option": option, "hint": f"{option} … — {hint}"},
            )
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # First, and before anything is dispatched: an action option with an empty value is a
    # mistake about this command, not a request to run the default one.
    empty = _empty_value_error(args)
    if empty is not None:
        _print_error(empty)
        return empty.exit_code

    if args.dump_default_config:
        print(AppConfig.default().to_json())
        return 0

    if getattr(args, "dump_mcp_tools", None):
        return _dump_mcp_tools(args.dump_mcp_tools)

    if args.validate_config:
        try:
            report = validate_config_file(args.validate_config)
        except StudioError as exc:
            _print_error(exc)
            return exc.exit_code
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["valid"] else 2

    try:
        config = resolve_config(args)
    except StudioError as exc:
        _print_error(exc)
        return exc.exit_code

    # On-premises model report (decision D26) — pure stdlib, works without the
    # [detect]/[ocr] extras and without a token.
    if args.model_status:
        print(json.dumps(ModelStore(config.paths.models_dir,
                                    offline=config.paths.offline).status(),
                         indent=2, ensure_ascii=False))
        return 0

    # The doctor (P9.5) runs before anything that could load a model: it is read-only,
    # offline-safe and never imports a backend. Its exit code is the one the next real
    # command would give, so a CI job can gate on it without parsing prose.
    if getattr(args, "doctor", False):
        return _doctor_action(args, config)

    # The device sweep (P9.4) is an action, not a startup option: it owns its engine pairs, and
    # it exits rather than becoming a window that has already spent a minute loading weights.
    if getattr(args, "warm_devices", None):
        return _warm_devices_action(args, config)

    # A config file with unknown/missing keys still loads (decision D10), but silently
    # is not the same as invisibly: --verbose surfaces what was defaulted or ignored.
    if args.verbose and args.config:
        try:
            for warning in validate_config_file(args.config).get("warnings", []):
                print(f"config warning: {warning['path']}: {warning['message']}",
                      file=sys.stderr)
        except StudioError:
            pass   # already loaded successfully above; nothing more to say

    if args.verbose:
        # stderr, always: stdout carries the result document for `--run` and the JSON of the
        # other actions, and must stay machine-parseable.
        print("Resolved configuration:", file=sys.stderr)
        print(config.to_json(), file=sys.stderr)

    if args.export_config:
        redact = bool(getattr(args, "redact_secrets", False))
        try:
            path = save_config_file(config, args.export_config, redact=redact)
        except StudioError as exc:
            _print_error(exc)
            return exc.exit_code
        # Say it out loud: a file that looks like a config but carries no token is a
        # support artifact, not a backup, and the user should know which one they made.
        print(f"Config written to {path}" + (" (secrets redacted)" if redact else ""))
        return 0

    # --run is the headless pipeline (Part 4). --headless is accepted with it and implied
    # by it: nobody types --run meaning anything else.
    if args.run:
        return _run_headless(args, config)

    # --serve (or plain stdio: there is nothing else to do with a protocol on stdout) is the
    # headless MCP server (Part 7). --gui-smoke still means "show me the window line".
    if (args.serve or args.mcp_transport == "stdio") and not args.gui_smoke:
        return _serve_headless(args, config)

    if args.headless:
        err = StudioError(
            ErrorCode.INVALID_ARGUMENT,
            "--headless needs an action: --run (pipeline -> JSON) or --serve "
            "(MCP server, no window).",
            {"hint": "python -m animetext_studio --headless --mock --image "
                     "examples/sample_page.png --run"},
        )
        _print_error(err)
        return err.exit_code

    # Default action: the GUI (Part 5). Without PySide6 this is a structured error naming
    # the install command and the headless alternative - never a traceback.
    from .gui.app import run_gui

    try:
        return run_gui(args, config, smoke=bool(getattr(args, "gui_smoke", False)))
    except KeyboardInterrupt:
        return 130
    except StudioError as exc:
        _print_error(exc)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
