# MCP — remote control with visual confirmation

AnimeText Studio is an MCP **server**. An agent (Claude Desktop, Claude Code, `mcp run`, or
90 lines of Python) can load a page, run detection and OCR, read the JSON back, and move the
viewer — while the person sitting at the machine watches the same widgets move that their own
mouse would move. There is no second, headless copy of the state: every tool call is
marshalled onto the Qt thread and applied through the same doors the GUI uses
(`MainWindow.set_result()`, `ControlPanel.apply_config()`, `JobManager`).

Three properties worth knowing up front:

* **One state.** `set_config` from a client moves the score slider; dragging the slider
  changes what the next `run_pipeline` from the same client uses. Nothing forks.
* **Nothing happens silently.** Each call adds a row to the MCP activity dock, a line to the
  status bar, and a brief flash on the widget it touched.
* **A failure is a result.** Errors come back as `isError: true` plus
  `{code, message, details}` — the same taxonomy the CLI exits with — not as a broken socket.
  That includes the failures the MCP SDK produces on our behalf (an argument that fails
  validation, a tool that does not exist) and resource reads, which carry the same object in
  `error.data`. See [the error matrix](ERROR-MATRIX.md).

---

## 1. Starting the server

| How | Command | Transport |
|---|---|---|
| GUI (default) | `animetext-studio` | `streamable-http` on `http://127.0.0.1:8765/mcp` |
| GUI, explicit | `animetext-studio --mcp-port 9000 --mcp-token "$(openssl rand -hex 16)"` | as above, with auth |
| Headless | `animetext-studio --headless --serve` | `streamable-http`, no window |
| Harness-spawned | `animetext-studio --mcp-transport stdio` | stdio (stdout is the protocol; logs go to stderr) |
| Off | `animetext-studio --mcp-transport none` | — |

The GUI shows what it did about itself, in the control panel:

```
MCP: http://127.0.0.1:8765/mcp        M
```

and `MCP: failed` with the reason in the status bar if the port was taken — a server that
cannot bind never takes the window down with it.

The `mcp` section of the configuration file (see `examples/config.default.json`) is the same
state the tools see:

```json
{ "mcp": { "enabled": true, "transport": "streamable-http",
           "host": "127.0.0.1", "port": 8765, "auth_token": null } }
```

`--mcp-host/--mcp-port/--mcp-token` override it. `get_status` reports `auth_required`; the
full config (`get_config`, `config://current`) contains the literal `mcp.auth_token`, because
that document is the app's state and round-trips through `import_config` unchanged. The
activity log and `log://mcp` redact it — but an `export_config(path=…)` file is a file with a
secret in it, so treat exported configs like one (and keep the server on loopback).

### Auth

With a token set, every request must carry `Authorization: Bearer <token>`; anything else gets
`401` before the MCP handshake and is counted in the gateway (`GET /health` is exempt, so a
load balancer can probe it). Without a token the server answers anyone on the bind address —
which is why the default bind is loopback. **If you set `--mcp-host 0.0.0.0`, set a token.**

### Connecting a client

Claude Desktop / Claude Code (`claude mcp add` writes the same JSON):

```json
{
  "mcpServers": {
    "animetext-studio": { "type": "http",
                          "url": "http://127.0.0.1:8765/mcp",
                          "headers": { "Authorization": "Bearer <token>" } }
  }
}
```

```json
{
  "mcpServers": {
    "animetext-studio-stdio": { "command": "animetext-studio",
                                "args": ["--mcp-transport", "stdio", "--mock"] }
  }
}
```

The MCP Inspector works too — GUI (`npx @modelcontextprotocol/inspector`, then connect to
`http://127.0.0.1:8765/mcp`) and headless:

```bash
npx -y @modelcontextprotocol/inspector --cli --transport http \
     --server-url http://127.0.0.1:8765/mcp --method tools/list
npx -y @modelcontextprotocol/inspector --cli --transport http \
     --server-url http://127.0.0.1:8765/mcp \
     --method tools/call --tool-name load_image --tool-arg path=page.png
```

---

## 2. The tools

21 tools, each with a JSON input schema **and** an output schema (both are in
`schemas/mcp-tools.json`, see §5). "What the user sees" is the point of Part 7 — every one of
them is visible, even the read-only ones.

### State

| Tool | What it does | What the user sees |
|---|---|---|
| `get_status` | Version, job state and progress, engines and device, current image, model bundle report, MCP counters | activity row + status bar |
| `list_models` | The five AnimeText YOLO variants (params, F1, recommended threshold) with `bundled` / `cached` / `loaded` / `available` flags, plus OCR state | activity row |
| `unload_engines` | Free the detector and OCR models (they reload on the next call). The model objects go and that memory is reusable; ONNX Runtime keeps a few hundred MB of process-wide state until the process exits | activity row; engines badge goes cold; next run is slower |
| `warm_engines` | Start the warm-up job (`--preload` remotely): load the weights **and** run one throwaway inference per engine. Answers with a job id; `probe: false` loads only. `BUSY` if a job is running | progress bar runs a `warmup` job; engines badge refreshes when it lands |

### Engine warmth (`--preload`, `warm_engines`)

An engine reports three separate things:

| Field | True when |
|---|---|
| `engines[].state` | `unloaded` / `loading` / `ready` / `error` — the engine's own state |
| `engines[].loaded` | that state is `ready`: the weights are in memory |
| `engines[].warm` | at least one inference has run since that load (`engines[].inferences` counts them) |
| `engines[].detail` | a short second line for whoever displays the engine — empty on an ordinary one |
| `engines[].model_list_is_fallback` | the model list came from the built-in presets rather than the repo: the hub was unreachable, gated or too slow to answer, so the five names are what this app can *offer*, not a list of what the repo has. `false` means a real answer arrived. |

That last field is one boolean on the engine, read by exactly the doors that show a model list: the
GUI's fifth badge (`models: presets — <why>`, next to `engines: …`), the `list_models` /
`get_status` payloads, and `--model-status`. It is not a state to manage — an engine that cannot
reach the repo still has models, and saying which list you are looking at is cheaper than
inventing a sixth engine state nobody can act on (D90).

The gap between the last two is the point. A loaded-but-cold session is slower on its first call
than on every call after it (execution context, kernel selection, allocator arenas), so `--preload`
— or `warm_engines` over MCP — pays that cost at startup instead of inside the user's first page.
Warm-up is a **job**: single-flight (`BUSY`, like `run_pipeline`), progress-reporting
(`job.kind == "warmup"`, stage `warmup`), cancellable — and it produces **no result document**,
so `has_result` and `result://latest` keep pointing at the last real page. `probe: false` loads
without the inference, which from a cold engine leaves `loaded: true, warm: false`.

`unload_engines` resets the flags (`loaded: false`, `warm: false`, `inferences: 0`) and really does
release the models — but it does not return every byte. Measured 2026-11-22, onnxruntime 1.32,
`yolo12m_animetext` on CPU: a load plus one inference cost 975 MB of RSS, unload gave back 191 MB
(244 MB with an explicit heap trim), and **zero** `InferenceSession` objects remained. The rest is
ONNX Runtime's process-wide state: reusable by the next load, returned to the OS only at exit.
`test_p8_lifecycle.py::test_unload_releases_the_session_even_though_rss_has_a_floor` pins exactly
that, so nobody has to rediscover it.

### Image in

| Tool | What it does | What the user sees |
|---|---|---|
| `load_image` | Load from `path`, from `base64_png_jpg` (≤ 64 MB), or from the clipboard (`from_clipboard`, GUI only) — exactly one of the three | the preview shows the page |
| `get_image_info` | Source, file name, width/height, sha256 of what is loaded | activity row |

### Running

| Tool | What it does | What the user sees |
|---|---|---|
| `detect` | Stage 1 only: boxes, no OCR. Publishes the result too (with `stage: "detect"`), so `get_result` works afterwards | boxes on the canvas |
| `run_pipeline` | Detection + OCR. `score_threshold`, `iou_threshold`, `model`, `device` and `reading_order` may be overridden **for this run** without touching the app config | progress bar, then boxes + JSON |
| `reocr_blocks` | Re-read the named blocks of the current document **without detecting again**: `block_ids: [7]`, an optional `ocr` override (`max_new_tokens` above all), and a whole new document comes back — same boxes, ids, colours and reading order, new `text` on the blocks you asked for, `stage: "reocr"`. The way to act on `flags: ["possible_truncation"]`. Needs a loaded page and a result; `INVALID_ARGUMENT` names the ids that exist when you ask for one that does not | progress bar; those rows change text, the rest of the page stands |
| `cancel_job` | Ask the running job to stop; returns its id and state | progress bar aborts |

One job at a time (single-flight). A second `run_pipeline`/`detect` while one is running gets
`BUSY` with the running `job_id` in `details` — nothing is queued, because a queue would mean
the human's Generate button silently loses.

**`stage` tells the two kinds of document apart.** Every result document carries
`"stage": "full" | "detect"` (result schema **1.1**). A `detect` document has
`blocks[].text == null` with `error.code == "SKIPPED"` on every block *by design* — OCR was
never asked. A `full` document with empty texts means OCR ran and read nothing (or failed per
block, with the engine's code in `error`). 1.0 documents have no `stage` and are read as
`full`; the frozen 1.0 schema ships as `schemas/result-1.0.schema.json`.

**`flags` and `contained_in` mark the blocks worth a second look** (result schema **1.2**).
Both are observations, never edits: nothing is dropped, merged or re-read, and geometry and
text are untouched.

| Field | Meaning | Why it can be trusted |
|---|---|---|
| `blocks[].flags` | list, currently `"possible_truncation"` or empty | the text length reached 90 % of `ocr.max_new_tokens` — the model author's own "this hit the generation cap" ratio (`processing_hayai.check_escalation`). It says *possibly cut off*, not *is cut off* |
| `blocks[].contained_in` | id of the tightest **strictly larger** box covering >= 80 % of this one, else `null` | pure geometry. The detector does return nested boxes and NMS keeps them (low IoU), and a nested fragment read as its own sentence is a real failure mode — measured on the sample page, a box inside a narration slab OCR'd as `った` |

Both are present in `detect` documents too (containment needs no OCR), and both are optional on
input: 1.0/1.1 documents read with `flags: []` and `contained_in: null`. Frozen copies ship as
`schemas/result-1.0.schema.json` and `schemas/result-1.1.schema.json`.

A consumer that wants to *act* on these — re-run a flagged block with a higher `max_new_tokens`,
or prefer a container's text over its fragment — does so outside this app; the app deliberately
reports rather than decides (see [`ADDENDUM.md`](../ADDENDUM.md)).

They are reported **in the window as well as in the document**: the block table grows a `flags`
column (`⚠` for `possible_truncation`, `⊂n` for `contained_in n`, with a sentence in the tooltip),
the summary line ends in `· 2 flagged` when there is anything to say, and the hover panel spells
both out under the full text. Painting them changes no bytes of the document — `render_overlay`
and the canvas only ever read the blocks (P8.6).

### Result out

| Tool | What it does | What the user sees |
|---|---|---|
| `get_result` | The full result document (`result.schema.json`), optionally filtered by `block_ids` | JSON pane flash |
| `save_result` | Write that document to `path` on the app's machine | Save button flash |
| `get_annotated_image` | The page with boxes drawn, `scale` 0.05–8.0 (outside that range is refused, and the rendered area is capped). **Pixels travel as image content**; `structuredContent` is metadata only (size, blocks, scale) | canvas flash |

### View

| Tool | What it does | What the user sees |
|---|---|---|
| `set_overlay_visible` | `visible`, `labels`, `fill_opacity` (any subset) | overlay toggles live |
| `set_zoom` | `mode: "fit"` or `mode: "percent"` with `percent` 10–800 | canvas zooms |
| `highlight_block` | Select a block and pulse it for `duration_ms` (default 1200) | box pulses + hover panel |
| `reset` | Clear image, result, canvas and JSON pane (refused while a job runs) | an empty window |

**Labels never sit on each other** (P8.6). Chips are claimed from the smallest box upwards — the
same rule that decides which box a click lands on — and one that does not fit tries the other
three edges of its box, then drops its words and keeps its number (`#7`), and only then steps
out entirely; the block it drops is reported by `ResultCanvas.hidden_labels` and keeps its row in
the table and its full text in the hover panel. Zooming in brings labels back, because the room a
chip needs is measured in screen pixels. The saved PNG and the canvas run the same rule, so an
export shows the page the user was looking at; the *document* is unaffected either way.

### Configuration

| Tool | What it does | What the user sees |
|---|---|---|
| `get_config` | The live config document (`config.schema.json`, v1.2) | activity row |
| `set_config` | Apply a **partial** patch (`{"detection": {"score_threshold": 0.3}}`); returns the full config, the changed key paths and `restart_required` | affected widgets update + flash |
| `export_config` | The config, and optionally written to `path` | Export button flash |
| `import_config` | Replace the config from `json` or `path` (exactly one); also reports `restart_required` | every control updates + flash |

**Everything in the config is live except five keys.** Thresholds, device, patch budget, reading
order, overlay, even `mcp.log_capacity` take effect on the call that changes them. `mcp.enabled`,
`mcp.transport`, `mcp.host`, `mcp.port` and `mcp.auth_token` are read once, when the server is
built, so they come back in `restart_required`: the value is stored (an export is faithful and
the next start uses it), but the socket you are talking to has not moved. There is no endpoint
change over MCP by design — a client that could move the port it is connected to could orphan
itself mid-session.

## 3. The resources

Six read-only resources, for clients that prefer URIs to calls:

| URI | MIME | Contents |
|---|---|---|
| `config://current` | `application/json` | the whole live configuration document |
| `result://latest` | `application/json` | the last published result (refuses before the first run) |
| `image://current` | `image/png` | the loaded page, no overlay |
| `image://annotated` | `image/png` | the page with the current boxes |
| `models://list` | `application/json` | the model catalogue with bundle state |
| `log://mcp` | `application/json` | recent tool calls, newest first, secrets redacted |

`result://latest` deliberately **refuses** rather than serving an empty document: an agent
reading it should learn that nothing has run, not that the page had no text.

## 4. Errors, progress, and what the human sees

### Errors

A failed call is a normal response: HTTP 200, `result.isError = true`, the same JSON in
`content[0].text` (for humans) and `structuredContent` (for machines):

```json
{ "code": "INVALID_CONFIG",
  "message": "Invalid configuration (1 issue(s)).",
  "details": { "issues": [ { "path": "detection.iou_threshold",
                             "message": "value 42 out of range [0.0, 1.0]" } ] } }
```

| Code | Means | CLI exit |
|---|---|---|
| `INVALID_ARGUMENT` | Wrong arguments for this tool | 2 |
| `INVALID_CONFIG` | A config document or patch is invalid (`details.issues[]`) | 2 |
| `NO_IMAGE` | Nothing loaded, or nothing run yet | 3 |
| `BUSY` | A job is already running (`details.job_id`) | 4 |
| `CANCELLED` | The job you awaited was cancelled | 5 |
| `HF_AUTH_REQUIRED` | Gated detection repo needs a token | 6 |
| `DOWNLOAD_FAILED` | Network/Hub failure | 7 |
| `ENGINE_LOAD_FAILED` | Engine could not load — **also a port that will not bind** | 8 |
| `OCR_FAILED` | OCR raised (whole run, or one crop in `blocks[].error`) | 9 |
| `IO_ERROR` | File could not be read or written | 10 |
| `INVALID_IMAGE` | Not a decodable image, or too large to decode | 11 |
| `MODEL_NOT_FOUND` | Not one of the known variants and not on disk | 12 |
| `INTERNAL` | Bug, including "the GUI thread did not answer in 20 s" | 1 |
| `NOT_IMPLEMENTED` | Declared for forward compatibility; nothing raises it today | 20 |

The exit column is `EXIT_CODES` in `core/types.py`, and
`test_the_mcp_doc_agrees_with_the_error_taxonomy` fails if this table and that mapping ever
disagree. [`docs/ERROR-MATRIX.md`](ERROR-MATRIX.md) is the full picture: every fault, every door,
the code each one produces, and what the caller should do.

`MODEL_NOT_FOUND` is deliberately *not* a "custom model" error: an unknown `detection.model`
is accepted with a warning, and only fails at load time, when the ONNX really is missing.

**Two things the SDK decides before our code runs** (P8.4), and both are handled so that the
promise above is actually true:

* a call whose arguments fail validation, or that names a tool that does not exist, also comes
  back as `{code: "INVALID_ARGUMENT", message, details}` (`details.reason` is
  `"argument_validation"` or `"unknown_tool"`, and an unknown tool gets the list of real ones).
  Without this the most common client mistake — a typo — was the one failure with no code, and it
  left no trace in the window. Every rejected call is now in `log://mcp`;
* a **resource** that cannot be read raises so that the JSON-RPC error carries the same document
  in `error.data` (`-32602` when the caller can fix it, `-32603` otherwise). Reading
  `result://latest` before anything has run used to arrive as a contentless
  `"Error reading resource result://latest"`.

### Progress

`run_pipeline` and `detect` pass a progress token through the SDK, so a client that registers
a callback receives `notifications/progress` (0–100, with the stage message) on the same
stream — while the GUI's own progress bar moves in lockstep. With the mock engines the run
finishes inside one tick and no notification is emitted; with real engines you get one per
stage and per OCR batch.

### What the human sees per call

* one row in the **MCP activity** dock: time, tool, status, duration, one-line detail
  (errors in red, carrying their code);
* a line in the **status bar**: `MCP run_pipeline: 5 block(s) in 842 ms`;
* a **flash** on the widget the call touched (canvas, score slider, Export button…);
* counters in `get_status().mcp` (`calls`, `errors`) and in `log://mcp`.

Arguments are summarised, never dumped. A long string is cut to 120 characters plus
`... (+1638280 chars)` — loading a page logs `base64_png_jpg: iVBORw0KGgoAAAANSUhEUgAA... (+1638280 chars)`,
not the page — nested structures are cut at two levels (`<8 keys>`), lists at 12 items, and
any key named `auth_token`, `token`, `hf_token`, `password`, `secret` or `api_key` becomes
`<N chars>` before it can reach the log, the dock or `log://mcp`.

---

## 5. Talking to it without an SDK

The streamable-http transport is JSON-RPC 2.0 over POST, with three rules:

1. `Accept: application/json, text/event-stream` — the server answers a call with **SSE**
   (`text/event-stream`), because a call may stream progress notifications before its result;
2. the `initialize` response carries **`Mcp-Session-Id`** in a *header*, and every later
   request must echo it (`404` otherwise);
3. the handshake is not complete until the client POSTs `notifications/initialized`.

A real session, captured by `scripts/mcp_http_transcript.py` against a live server
(`--mock`, so the numbers are reproducible; the full 11-step capture is in
`examples/mcp_raw_transcript.txt`):

```text
--> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18",
     "capabilities":{},"clientInfo":{"name":"transcript","version":"1"}}}
<-- header mcp-session-id: 2ae856a7b5c74d7caa0f77ed185e9606
<-- 200 text/event-stream
<-- {"jsonrpc":"2.0","id":1,"result":{"capabilities":{...},"protocolVersion":"2025-06-18",
     "serverInfo":{"name":"animetext-studio","version":"0.1.0"},
     "instructions":"animetext-studio 0.1.0: anime/manga text-block detection ... "}}

--> {"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
--> header Mcp-Session-Id: 2ae856a7b5c74d7caa0f77ed185e9606
<-- 202 application/json

--> {"jsonrpc":"2.0","method":"tools/call","id":6,
     "params":{"name":"load_image","arguments":{"path":"examples/sample_page.png"}}}
<-- 200 text/event-stream
<-- {"jsonrpc":"2.0","id":6,"result":{"content":[{"type":"text","text":"{...}"}],"isError":false,
     "structuredContent":{"image":{"source":"mcp-path","file_name":"sample_page.png",
     "width":900,"height":1300,"sha256":"b35b8283c659..."},"loaded":true}}}

--> {"jsonrpc":"2.0","method":"tools/call","id":10,
     "params":{"name":"set_config","arguments":{"patch":{"detection":{"iou_threshold":42}}}}}
<-- 200 text/event-stream
<-- {"jsonrpc":"2.0","id":10,"result":{"content":[{"type":"text","text":"{\"code\":
     \"INVALID_CONFIG\", ...}"}],"isError":true,
     "structuredContent":{"code":"INVALID_CONFIG",
       "message":"Invalid configuration (1 issue(s)).",
       "details":{"issues":[{"path":"detection.iou_threshold",
                             "message":"value 42 out of range [0.0, 1.0]"}]}}}}
```

The same session in shell, end to end (`./scripts/mcp_curl_demo.sh 8765`) — it prints each
request before making it, so it doubles as documentation:

```bash
URL=http://127.0.0.1:8765/mcp
HDR=(-H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream')

curl -si "${HDR[@]}" -X POST "$URL" -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
  "params":{"protocolVersion":"2025-06-18","capabilities":{},
            "clientInfo":{"name":"curl","version":"1"}}}' | grep -i mcp-session-id
SESSION=…                                     # from the header above
SID=(-H "Mcp-Session-Id: $SESSION")

curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | jq -r '.result.tools[].name'

curl -s "${HDR[@]}" "${SID[@]}" -X POST "$URL" \
  -d '{"jsonrpc":"2.0","id":6,"method":"tools/call",
       "params":{"name":"run_pipeline","arguments":{}}}' \
  | jq -c '.result.structuredContent | {block_count, timing_ms: .result.timing_ms}'
```

Three programs ship with the app, at three levels of "I am not using the SDK":

| Program | What it is |
|---|---|
| `python -m animetext_studio.tools.mcp_client_demo --url …` | the deterministic SDK session (19 steps, 3 of them negative checks); validates every payload against the published schemas and **exits 0** — the Part 7 acceptance harness |
| `scripts/mcp_curl_demo.sh [port]` | the same session in `curl`, printed as it runs |
| `scripts/mcp_http_transcript.py --port …` | the raw capture above (what the docs embed) |

## 6. The machine-readable catalog

`schemas/mcp-tools.json` is generated from the server itself, never written by hand:

```bash
animetext-studio --dump-mcp-tools            # regenerate schemas/mcp-tools.json
animetext-studio --dump-mcp-tools -          # print it instead
```

It contains `catalogVersion`, the server info, the transport, the error shape and code list,
the six resources, and for each of the 21 tools: `description`, `inputSchema`,
`outputSchema`, the tool annotations (`readOnlyHint`, `destructiveHint`, …) and
`guiEffect` + `flash` — what the human will see when that tool runs. A test
(`test_mcp_server.py`) fails if the shipped file and a live server ever disagree, so the file
is safe to generate client bindings from.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ENGINE_LOAD_FAILED` at startup with `details.port` set | the port is taken; `--mcp-port 0` asks for a free one (the GUI badge shows what it got) |
| Every request refused before the handshake | a bearer secret is configured and the client is not sending it; `GET /health` still answers, so a probe can tell "down" from "locked" |
| `404` on a request that carries a session id | the id came from an earlier session; re-`initialize` |
| Client hangs, then `INTERNAL` | the GUI thread did not answer within 20 s — the app is frozen or closing |
| `get_status` fine, `run_pipeline` says `HF_AUTH_REQUIRED` | the detection repo is gated; pass `--hf-token` once (it is never written into a config file) |
| stdio: "server exited immediately" | something wrote to stdout. Only the protocol may go there; the app's own banner goes to stderr, and a test enforces that |
| An agent claims the result is stale | it is not: a result is published only when a run finishes, and `result://latest` refuses before that |

### Limits worth knowing

* **Single-flight**, deliberately: no queue, so the human's own run is never deferred silently.
* `load_image` accepts at most 64 MB of base64; `get_annotated_image` renders the current
  overlay and sends pixels as image content, never inside `structuredContent`.
* The server binds loopback and does not encrypt. Anything beyond loopback needs a bearer
  secret and a reverse proxy — on the P8 list, together with a startup warning when the bind
  address is not loopback.

### Tests

```bash
python -m pytest -q                              # the default suite: no weights, no display
python -m pytest tests/test_mcp_http.py -q       # every tool over real HTTP, payloads schema-checked
python -m pytest tests/test_mcp_stdio.py -q      # spawn the app, handshake down a pipe
python -m pytest tests/test_mcp_gui.py -q        # the bridge driving the visible window
```
