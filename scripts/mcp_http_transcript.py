#!/usr/bin/env python3
"""Print a raw JSON-RPC transcript against a running AnimeText Studio MCP server.

Same requests `scripts/mcp_curl_demo.sh` makes, in Python (httpx) so it also runs where the
curl binary is restricted - and so the transcript in docs/MCP.md is captured, not hand-typed.
Every line is the real wire exchange: the POST body we sent, the response status, the session
header, and the JSON body (long values elided, marked `...`).

    python -m animetext_studio --mock --headless --serve --mcp-port 8791
    python scripts/mcp_http_transcript.py --port 8791
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

ACCEPT = "application/json, text/event-stream"
ELIDE_AFTER = 240


def shrink_images(payload):
    """A PNG in the transcript would be 90 000 characters of noise; keep the shape only."""
    if isinstance(payload, dict):
        result = dict(payload)
        for key in ("result", "contents"):
            if key in result:
                inner = result[key]
                if isinstance(inner, dict) and isinstance(inner.get("content"), list):
                    inner = {**inner, "content": [shrink_images(c) for c in inner["content"]]}
                elif isinstance(inner, list):
                    inner = [shrink_images(c) for c in inner]
                result[key] = inner
        if payload.get("type") == "image" and isinstance(payload.get("data"), str):
            result["data"] = f"<base64 PNG, {len(payload['data'])} chars>"
        return result
    if isinstance(payload, list):
        return [shrink_images(item) for item in payload]
    return payload


def elide(value):
    """Shorten long strings/lists so the transcript stays readable (and no pixels leak in)."""
    if isinstance(value, str):
        return value if len(value) <= ELIDE_AFTER else value[:ELIDE_AFTER] + f"... ({len(value)} chars)"
    if isinstance(value, dict):
        return {k: elide(v) for k, v in value.items()}
    if isinstance(value, list):
        keep = [elide(v) for v in value[:6]]
        if len(value) > 6:
            keep.append(f"... {len(value) - 6} more")
        return keep
    return value


class Transcript:
    def __init__(self, url: str) -> None:
        self.url = url
        self.client = httpx.Client(timeout=120)
        self.session_id = ""
        self.step = 0

    def rpc(self, method: str, params: dict | None = None, *, notify: bool = False,
            show: bool = True) -> dict:
        self.step += 1
        body: dict = {"jsonrpc": "2.0", "method": method}
        if not notify:
            body["id"] = self.step
        if params is not None:
            body["params"] = params
        headers = {"Content-Type": "application/json", "Accept": ACCEPT}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        print(f"\n=== step {self.step}: POST {self.url}")
        print("--> " + json.dumps(body, ensure_ascii=False))
        for name in ("Mcp-Session-Id",):
            if name in headers:
                print(f"--> header {name}: {headers[name]}")
        response = self.client.post(self.url, headers=headers, content=json.dumps(body).encode())
        if not self.session_id and "mcp-session-id" in response.headers:
            self.session_id = response.headers["mcp-session-id"]
            print(f"<-- header mcp-session-id: {self.session_id}")
        print(f"<-- {response.status_code} {response.headers.get('content-type', '')}")
        if notify:
            return {}
        result: dict = {}
        for message in self.events(response):        # JSON, or SSE when the call streams
            if "method" in message:
                print("<-- " + json.dumps(elide(message), ensure_ascii=False))
            else:
                result = message
                if show:
                    print("<-- " + json.dumps(elide(shrink_images(result)), ensure_ascii=False))
        return result

    def show(self, line: str) -> None:
        print("# " + line)

    def events(self, response: httpx.Response) -> list[dict]:
        """Parse either a JSON body or an SSE body (`text/event-stream`) into messages."""
        if "text/event-stream" in response.headers.get("content-type", ""):
            out = []
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk:
                        out.append(json.loads(chunk))
            return out
        return [json.loads(response.text)]

    def call(self, tool: str, arguments: dict | None = None) -> dict:
        """tools/call, printing any progress notifications the server pushes first."""
        self.step += 1
        body = {"jsonrpc": "2.0", "id": self.step, "method": "tools/call",
                "params": {"name": tool, "arguments": arguments or {}}}
        headers = {"Content-Type": "application/json", "Accept": ACCEPT,
                   "Mcp-Session-Id": self.session_id}
        print(f"\n=== step {self.step}: POST {self.url}  (tools/call {tool})")
        print("--> " + json.dumps(body, ensure_ascii=False))
        response = self.client.post(self.url, headers=headers, content=json.dumps(body).encode())
        print(f"<-- {response.status_code} {response.headers.get('content-type', '')}")
        result: dict = {}
        self.progress_seen = 0
        for message in self.events(response):
            if "method" in message:                      # notifications/progress etc.
                self.progress_seen += 1
                print("<-- " + json.dumps(elide(message), ensure_ascii=False))
            else:
                result = message
                print("<-- " + json.dumps(elide(result), ensure_ascii=False))
        return result.get("result", {})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--image", default="examples/sample_page.png")
    args = parser.parse_args(argv)

    tx = Transcript(f"http://{args.host}:{args.port}/mcp")
    print("# AnimeText Studio MCP - raw JSON-RPC transcript")
    print(f"# server: {tx.url}   python: {sys.version.split()[0]}")

    init = tx.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "transcript", "version": "1"}})
    print("# the session id came back as a header; every later request must echo it")
    tx.rpc("notifications/initialized", {}, notify=True)

    listing = tx.rpc("tools/list", show=False)      # the full schemas are 60 KB - names only
    tools = listing["result"]["tools"]
    tx.show(f"tools/list -> {len(tools)} tools: " + " ".join(t["name"] for t in tools))
    resources = tx.rpc("resources/list", show=False)["result"]["resources"]
    tx.show(f"resources/list -> {len(resources)} resources: "
            + " ".join(r["uri"] for r in resources))

    tx.rpc("tools/call", {"name": "get_status", "arguments": {}})
    tx.rpc("tools/call", {"name": "load_image", "arguments": {"path": args.image}})
    run = tx.call("run_pipeline", {})
    blocks = (run.get("structuredContent") or {}).get("result", {}).get("blocks", [])
    streamed = getattr(tx, "progress_seen", 0)
    how = (f"{streamed} progress notification(s) streamed first" if streamed
           else "no progress notification - the mock engines finish inside one tick")
    tx.show(f"run_pipeline published {len(blocks)} block(s); {how}")
    tx.rpc("tools/call", {"name": "get_result", "arguments": {"block_ids": [1, 2]}})
    shot = tx.rpc("tools/call", {"name": "get_annotated_image", "arguments": {}})["result"]
    tx.show(f"annotated PNG: {shot['structuredContent']['bytes']} bytes travel as image "
            f"content, never as structuredContent")
    bad = tx.rpc("tools/call", {"name": "set_config",
                                "arguments": {"patch": {"detection": {"iou_threshold": 42}}}})
    code = (bad.get("result", {}).get("structuredContent") or {}).get("code", "?")
    tx.show(f"a rejected call is still HTTP 200 + isError + a machine-readable code: {code}")
    tx.rpc("resources/read", {"uri": "log://mcp"})
    print("\n# end of transcript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
