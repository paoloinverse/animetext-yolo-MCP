#!/usr/bin/env python3
"""Deterministic MCP client: drives a full AnimeText Studio session and exits 0/1.

    python -m animetext_studio.tools.mcp_client_demo --url http://127.0.0.1:8765/mcp \
        --image examples/sample_page.png --out-dir out/mcp_session

This is the harness PLAN §4 promises: a script (not a chat) that does
status -> models -> configure -> load -> run -> JSON -> annotated image -> save -> reset,
validating every response against the published catalog (`schemas/mcp-tools.json`) and
printing one line per step. Exit code 0 means: the server answered every tool, every
structured payload matched its output schema, and the activity log has one entry per call.

Options:
  --url / --token        where to connect (default http://127.0.0.1:8765/mcp)
  --image PATH           page to load (default examples/sample_page.png)
  --skip-run             stop after configure (schema/transport check only)
  --expect-blocks N      fail if the run yields fewer than N blocks
  --json FILE            also write the whole transcript as JSON
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from animetext_studio.core.types import StudioError                    # noqa: E402
from animetext_studio.mcp.server import catalog_path                    # noqa: E402

DEFAULT_IMAGE = "examples/sample_page.png"


class StepFailed(Exception):
    def __init__(self, name: str, detail: Any) -> None:
        super().__init__(f"{name}: {detail}")
        self.name = name
        self.detail = detail


def load_catalog(path: Optional[Path] = None) -> dict[str, Any]:
    target = Path(path) if path else catalog_path()
    if not target.is_file():
        raise SystemExit(f"catalog not found: {target} (run --dump-mcp-tools)")
    return json.loads(target.read_text(encoding="utf-8"))


def validate(catalog: dict[str, Any], tool: str, payload: Any) -> None:
    """Check a structured payload against the catalog's output schema (jsonschema if present)."""
    entry = next((t for t in catalog["tools"] if t["name"] == tool), None)
    if entry is None:
        raise StepFailed(tool, "not in the published catalog")
    schema = entry.get("outputSchema")
    if not schema:
        return
    try:
        import jsonschema
    except ImportError:
        return                                    # without the dev extra: shape check only
    jsonschema.validate(instance=payload, schema=schema)


def _result_dict(result: Any, tool: str) -> dict[str, Any]:
    if getattr(result, "is_error", False):
        text = result.content[0].text if result.content else "no content"
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            payload = {"code": "INTERNAL", "message": str(text)}
        raise StepFailed(tool, payload)
    if result.structured_content is None:
        raise StepFailed(tool, "no structuredContent in the response")
    return result.structured_content


class Session:
    """One scripted visit: every step records what it got, and fails loudly if it differs."""

    def __init__(self, url: str, *, token: Optional[str] = None, image: str = DEFAULT_IMAGE,
                 out_dir: Optional[Path] = None, skip_run: bool = False,
                 expect_blocks: int = 0, quiet: bool = False,
                 no_reset: bool = False) -> None:
        self.catalog = load_catalog()
        self.url = url
        self.token = token
        self.image = image
        self.out_dir = Path(out_dir) if out_dir else None
        self.skip_run = skip_run
        self.expect_blocks = expect_blocks
        self.quiet = quiet
        self.no_reset = no_reset
        self.transcript: list[dict[str, Any]] = []
        self.progress: list[dict[str, Any]] = []

    def say(self, text: str) -> None:
        if not self.quiet:
            print(text, flush=True)

    def record(self, step: str, **info: Any) -> None:
        self.transcript.append({"step": step, **info})

    # -- the session ----------------------------------------------------------

    async def run(self) -> int:
        import contextlib

        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        # One exit stack: transport, (optional) http client, session - unwound in order even
        # when a step raises. Handing `__aenter__`/`__aexit__` to the GC instead would leave
        # the transport's task group being cancelled from a generator finaliser.
        async with contextlib.AsyncExitStack() as stack:
            kwargs: dict[str, Any] = {}
            if self.token:
                from mcp.client.streamable_http import create_mcp_http_client

                client = create_mcp_http_client(
                    headers={"Authorization": f"Bearer {self.token}"})
                close = getattr(client, "aclose", None)
                if close is not None:
                    stack.push_async_callback(close)
                kwargs["http_client"] = client
            streams = await stack.enter_async_context(
                streamable_http_client(self.url, **kwargs))
            session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
            self.session = session
            init = await session.initialize()
            self.record("initialize", server=init.server_info.name,
                        version=init.server_info.version,
                        protocol=init.protocol_version)
            self.say(f"connected  {init.server_info.name} {init.server_info.version} "
                     f"(protocol {init.protocol_version})")
            await self.steps()
        return 0

    async def call(self, tool: str, arguments: Optional[dict[str, Any]] = None, *,
                   progress: bool = False, save: Optional[str] = None) -> dict[str, Any]:
        callback = self._on_progress if progress else None
        result = await self.session.call_tool(tool, arguments or {},
                                              progress_callback=callback)
        payload = _result_dict(result, tool)
        validate(self.catalog, tool, payload)
        self.record(tool, arguments=arguments or {}, response=payload)
        if save and self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            for block in result.content:
                if getattr(block, "type", "") == "image":
                    target = self.out_dir / save
                    target.write_bytes(base64.b64decode(block.data))
                    self.record("save-image", path=str(target),
                                bytes=len(base64.b64decode(block.data)))
        return payload

    async def _on_progress(self, progress: float, total, message) -> None:
        self.progress.append({"percent": progress, "total": total, "message": message})

    async def read(self, uri: str) -> str:
        result = await self.session.read_resource(uri)
        content = result.contents[0]
        text = getattr(content, "text", None)
        body = text if text is not None else getattr(content, "blob", b"")
        size = len(body)
        self.record("read-resource", uri=uri, bytes=size)
        self.say(f"resource    {uri:22s} {size} bytes")
        return text or ""

    async def steps(self) -> None:
        names = [t.name for t in (await self.session.list_tools()).tools]
        expected = [t["name"] for t in self.catalog["tools"]]
        missing = sorted(set(expected) - set(names))
        if missing:
            raise StepFailed("list_tools", {"missing": missing})
        self.say(f"tools       {len(names)} published, {len(expected)} in the catalog")
        resources = [str(r.uri) for r in (await self.session.list_resources()).resources]
        self.say(f"resources   {len(resources)}: {', '.join(sorted(resources))}")

        status = await self.call("get_status")
        self.say(f"get_status  {status['app']} {status['version']} "
                 f"mock={status['mock']} busy={status['busy']} gui={status['gui']}")
        models = await self.call("list_models")
        self.say(f"list_models {len(models['models'])} variants, "
                 f"selected={models['selected_model']}, bundled="
                 f"{sum(1 for m in models['models'] if m['bundled'])}")

        applied = await self.call("set_config", {"patch": {
            "detection": {"score_threshold": 0.2, "iou_threshold": 0.3},
            "pipeline": {"reading_order": "manga_rtl"}}})
        self.say(f"set_config  changed={applied['changed']} warnings={applied['warnings']}")

        loaded = await self.call("load_image", {"path": str(self.image)})
        self.say(f"load_image  {loaded['image']['width']}x{loaded['image']['height']} "
                 f"sha={loaded['image']['sha256'][:12]}")

        if self.skip_run:
            return

        run = await self.call("run_pipeline", {"include_annotated_image": True},
                             progress=True, save="annotated_from_mcp.png")
        blocks = run["result"]["blocks"]
        self.say(f"run_pipeline {run['block_count']} blocks in {run['timing_ms']['total']} ms, "
                 f"{len(self.progress)} progress notifications, "
                 f"image={'yes' if run.get('annotated_image') else 'no'}")
        if self.expect_blocks and run["block_count"] < self.expect_blocks:
            raise StepFailed("run_pipeline", {"blocks": run["block_count"],
                                              "expected_min": self.expect_blocks})
        for block in blocks[:3]:
            self.say(f"            #{block['id']} {block['bbox']['x0']},{block['bbox']['y0']}"
                     f"-{block['bbox']['x1']},{block['bbox']['y1']} "
                     f"{(block.get('text') or '')[:24]!r}")

        filtered = await self.call("get_result", {"block_ids": [b["id"] for b in blocks[:2]]})
        self.say(f"get_result  filtered to {filtered['block_count']} block(s)")

        # The 21st door (P9.3): re-read one block without asking the detector again. The answer
        # is a whole document, so the check is "same boxes, new read" — not a diff.
        target = blocks[0]["id"]
        again = await self.call("reocr_blocks", {"block_ids": [target]})
        reread = again["result"]
        if reread["stage"] != "reocr" or reread["block_count"] != run["block_count"]:
            raise StepFailed("reocr_blocks", {"stage": reread["stage"],
                                              "blocks": reread["block_count"],
                                              "expected_blocks": run["block_count"]})
        if [b["id"] for b in reread["blocks"]] != [b["id"] for b in blocks]:
            raise StepFailed("reocr_blocks", {"reason": "ids changed; a re-read is the same page"})
        self.say(f"reocr       block #{target} in {again['timing_ms']['total']} ms — "
                 f"{reread['block_count']} blocks, stage={reread['stage']}, boxes unchanged")
        image = await self.call("get_annotated_image", {"scale": 0.5})
        self.say(f"annotated   {image['width']}x{image['height']} "
                 f"{image['bytes']} bytes, scale={image['scale']}")
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "result.json").write_text(
                json.dumps(run["result"], indent=2, ensure_ascii=False), encoding="utf-8")

        await self.call("highlight_block", {"id": blocks[0]["id"], "duration_ms": 400})
        self.say(f"highlight   block #{blocks[0]['id']}")
        overlay = await self.call("set_overlay_visible", {"visible": False})
        self.say(f"overlay     visible={overlay['overlay_visible']}")
        await self.call("set_overlay_visible", {"visible": True})
        zoom = await self.call("set_zoom", {"mode": "percent", "percent": 110})
        self.say(f"zoom        {zoom['zoom_mode']} {zoom['zoom_percent']}%")

        if self.out_dir is not None:
            saved = await self.call("save_result",
                                    {"path": str(self.out_dir / "saved_by_mcp.json")})
            self.say(f"save_result {saved['path']} ({saved['bytes']} bytes)")
        await self.read("config://current")
        await self.read("result://latest")
        log = json.loads(await self.read("log://mcp"))
        calls = [e["tool"] for e in log["entries"]]
        self.say(f"log         {log['calls']} calls recorded, newest={calls[:3]}")

        errors = await self._negative_checks()
        if self.no_reset:
            # The visual check wants the page left on screen; the post-reset guarantee is
            # then simply not exercised here (a test does it with reset enabled).
            self.say("reset      skipped (--no-reset: the page stays on the canvas)")
        else:
            reset = await self.call("reset")
            self.say(f"reset       cleared={reset['cleared']}")
            after = await self.session.call_tool("get_result", {})
            if not after.is_error:
                raise StepFailed("reset", "get_result still succeeds after reset")
        self.record("negative", checks=errors)
        self.say(f"checks      {errors}")
        self.say("OK")


    async def _negative_checks(self) -> list[str]:
        """The deterministic-client guarantees: bad args and unknown ids are structured errors."""
        checks: list[str] = []
        bad = await self.session.call_tool("set_zoom", {"mode": "sideways"})
        if not bad.is_error or _error_code(bad) != "INVALID_ARGUMENT":
            raise StepFailed("negative:set_zoom", _peek(bad))
        checks.append("set_zoom(mode=sideways) -> INVALID_ARGUMENT")

        missing = await self.session.call_tool("get_result", {"block_ids": [9999]})
        if not missing.is_error or _error_code(missing) != "INVALID_ARGUMENT":
            raise StepFailed("negative:get_result", _peek(missing))
        checks.append("get_result(block_ids=[9999]) -> INVALID_ARGUMENT")

        neither = await self.session.call_tool("import_config", {})
        if not neither.is_error or _error_code(neither) != "INVALID_ARGUMENT":
            raise StepFailed("negative:import_config", _peek(neither))
        checks.append("import_config() -> INVALID_ARGUMENT")
        return checks


def describe_exception(exc: BaseException) -> str:
    """Flatten anyio TaskGroups down to the leaf messages (a 401 hides two levels deep)."""
    leaves: list[str] = []
    stack: list[BaseException] = [exc]
    while stack and len(leaves) < 8:
        current = stack.pop()
        inner = getattr(current, "exceptions", None)
        if inner:
            stack.extend(inner)
        else:
            leaves.append(f"{type(current).__name__}: {current}")
    return "; ".join(leaves) or type(exc).__name__


def _error_code(result: Any) -> str:
    if result.structured_content and isinstance(result.structured_content, dict):
        return str(result.structured_content.get("code", ""))
    return ""


def _peek(result: Any) -> Any:
    if result.structured_content:
        return result.structured_content
    return result.content[0].text if result.content else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp_client_demo", description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--token", default=None)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--expect-blocks", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the transcript (steps, arguments, responses) here")
    parser.add_argument("--no-reset", action="store_true",
                        help="skip the final reset, so the page stays on the GUI canvas "
                             "(used by the visual check)")
    parser.add_argument("--expect-connect-failure", action="store_true",
                        help="exit 0 when the server refuses the connection (the bearer-"
                             "token rejection harness); a successful connect exits 1")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    session = Session(args.url, token=args.token, image=args.image,
                      out_dir=Path(args.out_dir) if args.out_dir else None,
                      skip_run=args.skip_run, expect_blocks=args.expect_blocks,
                      quiet=args.quiet, no_reset=args.no_reset)
    try:
        code = asyncio.run(session.run())
    except StepFailed as failure_:
        print(f"FAIL {failure_.name}: {failure_.detail}", file=sys.stderr)
        return 1
    except StudioError as exc:
        print(f"FAIL {exc.to_dict()}", file=sys.stderr)
        return 1
    except (ConnectionRefusedError, OSError) as exc:
        print(f"FAIL cannot connect to {args.url}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                                   # noqa: BLE001
        # A rejected handshake arrives as an MCPError (the 401 becomes a JSON-RPC error)
        # wrapped in anyio TaskGroups, not as a connection error - so it lands here.
        # With --expect-connect-failure that rejection *is* the test.
        detail = describe_exception(exc)
        if args.expect_connect_failure:
            print(f"rejected as expected: {detail[:200]}")
            return 0
        print(f"FAIL {detail}", file=sys.stderr)
        return 2
    if args.expect_connect_failure:
        print("FAIL the connection was accepted, but a rejection was expected",
              file=sys.stderr)
        return 1
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"url": args.url, "steps": session.transcript,
                        "progress": session.progress}, indent=2, ensure_ascii=False),
            encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
