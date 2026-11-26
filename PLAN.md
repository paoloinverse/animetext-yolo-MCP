# AnimeText OCR Studio — Implementation Plan

**Goal:** A Python desktop app (PySide6) that detects anime/manga text blocks with the
`deepghs/AnimeText_yolo` YOLO12 models, OCRs each detected crop with
`JustANormalTinkerer/hayai-ocr-v2.7-supernova`, and produces a copy-able JSON mapping
bounding boxes → text, with an interactive annotated-image view. The app is fully
remote-controllable through an MCP server (configurable bind address/port, CLI flags),
and **every MCP interaction is visually reflected in the GUI**.

This plan is divided into **8 independent parts**, each with its own scope, files,
interfaces, acceptance criteria, and demo/test command, so we can implement and verify
one part per turn.

---

## 0. Verified technical facts (researched, not assumed)

### 0.1 Detection — `deepghs/AnimeText_yolo`

- 5 variants, each shipping `model.onnx` + `model.pt` + `labels.json` + `threshold.json`:

  | Model | Params | F1 | Recommended score threshold (F1-optimal) |
  |---|---|---|---|
  | `yolo12n_animetext` | 2.57M | 0.86 | 0.251 |
  | `yolo12s_animetext` | 9.25M | 0.88 | 0.272 |
  | `yolo12m_animetext` | 20.1M | 0.89 | 0.299 |
  | `yolo12l_animetext` | 26.4M | 0.89 | 0.426 |
  | `yolo12x_animetext` | 59.1M | 0.90 | 0.425 |

  Single class label: **`text_block`** (verified from each variant's `labels.json` with a
  token; an earlier draft of this plan said `text_bblock` — typo, fixed in code). F1 and
  the F1-optimal thresholds were also re-verified against the shipped `threshold.json`
  (n 0.251 / s 0.272 / m 0.299 / l 0.426 / x 0.425 — all match the table above).
  ONNX sizes: n 10.5 MB, s 37.2 MB, m 80.7 MB, l 105.8 MB, x 236.7 MB.
  License GPL-3.0. **Repo is gated (`gated: auto`)**
  → downloading weights requires an HF token whose account accepted the repo terms
  (`HF_TOKEN` env var or `--hf-token`). Must be handled with a clear error + GUI hint.

- The official Space (`app.py`) is exactly:

  ```python
  from imgutils.generic import YOLOModel
  model = YOLOModel('deepghs/AnimeText_yolo')
  model.launch_demo(default_iou_threshold=0.35, default_conf_threshold=0.05,
                    default_model_name='yolo12m_animetext')
  ```

  → We use the same library (`dghs-imgutils`), same defaults (**iou 0.35, score 0.05,
  model m**), and additionally offer per-model "recommended F1 threshold" presets.

- Verified `imgutils` API (source inspected, saved at `research/imgutils_yolo.py`):

  ```python
  YOLOModel(repo_id: str, hf_token: str | None = None)   # honors HF_TOKEN env
  .model_names -> list[str]                              # globs */model.onnx in repo
  .predict(image, model_name: str,
           conf_threshold: float | None = None,          # None → threshold.json, else 0.25
           iou_threshold: float = 0.7,
           allow_dynamic: bool = False)                  # True keeps aspect ratio
      -> list[((x0, y0, x1, y1), label, score)]            # int pixel coords, original image space
  .clear()                                               # free cached sessions
  ```

  - Inference is **ONNX Runtime** (CPU by default; `onnxruntime-gpu` optional) — no torch
    needed for detection. Input size comes from ONNX metadata (`imgsz`); default
    preprocessing resizes to a fixed square, `allow_dynamic=True` preserves aspect ratio
    (better for tall manga pages → expose as an advanced option, default matches the demo).
  - Downloads/caches via `huggingface_hub` (standard HF cache, `HF_HOME` respected).

### 0.2 OCR — `JustANormalTinkerer/hayai-ocr-v2.7-supernova`

- ~157M-param SigLIP2-NaFlex encoder + 12-layer GQA decoder; crop-level transcription
  for JA/ZH/KO/EN, handles vertical manga text natively. **Not gated.** Apache-2.0.
  Weights ≈ 634 MB (F32 safetensors). Custom code in-repo → `trust_remote_code=True`.
- Verified usage from the model card:

  ```python
  from transformers import AutoModel, AutoProcessor
  model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).cuda().eval()
  processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

  text  = model.ocr(image, processor, max_patches="auto",
                    max_new_tokens=128, use_cuda_graph=True)          # single crop
  texts = model.ocr([crop1, crop2, ...], processor, max_patches="auto",
                    batch_size=64, escalate=False, max_new_tokens=128) # batched
  ```

  - `max_patches`: `"auto"` (recommended) or fixed `128 | 192 | 256 | 384`
    (128 = lowest VRAM/fastest, 384 = dense multi-line blocks).
  - CUDA: fp16 + `use_cuda_graph=True` (~2.3 GB peak VRAM, >100 crops/s on a T4).
    CPU fallback: fp32, `use_cuda_graph=False` (works, slower — fine for single pages).
  - `HAYAI_DISABLE_COMPILE=1` env flag disables `torch.compile` on custom kernels.
  - Optional NFKC normalization snippet provided in the card → config flag.
  - Pin the repo revision at implementation time for reproducibility
    (pinned sha `f5f0f617f5148f760074f743e25576f381992c34`, verified 2026-11-20; the
    earlier draft pinned `5289b2cfdfd3…`, an older snapshot that lacks
    `yolo26m-text.pt` and carries an older `processing_hayai.py`).
  - **On-premises weights (D26):** both repos are vendored into `animetext_studio/models/`
    by `scripts/fetch_models.py`, with `MANIFEST.json` recording repo, revision, size and
    sha256. Resolution is bundled-first, Hub second; `--offline` never opens a socket.

### 0.3 Consequences for the design

- Detection (onnxruntime) and OCR (torch/transformers) are **independent engines** with
  independent lazy loading — parts 2 and 3 can be built/tested in isolation.
- First run needs network + HF token (detection only). Everything must degrade to clear,
  actionable errors, and the GUI must remain usable (model list falls back to the 5 known
  names when offline).
- Both engines are CPU-capable → the whole app runs without a GPU.

---

## 1. Architecture overview

```
┌────────────────────────────── one process ──────────────────────────────┐
│                                                                         │
│  Qt main thread (GUI)                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ MainWindow: controls │ input view │ annotated output │ JSON pane  │  │
│  │ MCP activity log dock │ status bar │ progress                     │  │
│  └──────────────▲──────────────────────────────▲─────────────────────┘  │
│                 │ Qt signals (queued)          │ GuiBridge (QObject)    │
│  ┌──────────────┴──────────────┐   ┌───────────┴─────────────────────┐  │
│  │ JobManager + worker thread  │   │ MCP server thread               │  │
│  │ Pipeline: detect→crop→OCR   │◄──┤ asyncio loop + uvicorn          │  │
│  │ (single-flight, cancelable) │   │ FastMCP: streamable-http / stdio│  │
│  └──────┬───────────┬──────────┘   └───────────▲─────────────────────┘  │
│         │           │                          │ JSON-RPC (MCP)         │
│  DetectionEngine  OcrEngine            agents / harnesses / scripts     │
│  (imgutils/ONNX)  (hayai/torch)                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Threading rules (critical):**
1. All Qt widget access happens on the **main thread only**.
2. Inference runs on a **worker thread** owned by `JobManager` (single-flight: one job
   at a time; other requests get a structured `BUSY` error; jobs are cancelable).
3. The MCP server runs in a **dedicated thread with its own asyncio loop**. Tool
   handlers never touch Qt directly — they call `GuiBridge`, which marshals operations
   to the main thread via queued invocations and returns `concurrent.futures.Future`s
   (with timeouts). Long jobs report MCP progress notifications while the same progress
   drives the GUI progress bar.
4. Every MCP tool call emits an `mcp_activity` signal → activity log entry + transient
   highlight ("flash") of the affected widget + status-bar message. This is the
   "visual confirmation" requirement, implemented once in the bridge, not per-tool.

**Headless parity:** `--headless --run` executes the exact same `Pipeline` without Qt,
printing the result JSON to stdout — the deterministic fallback path.

### Repository layout

```
animetext-studio/
├── pyproject.toml              # deps + extras: [gui] [ocr] [mcp] [dev]
├── README.md                   # part 8
├── PLAN.md                     # this document
├── schemas/                    # (moved into package in P1, see D6 in tracking.md)
├── animetext_studio/
│   ├── __init__.py  __main__.py
│   ├── schemas/
│   │   ├── config.schema.json  # part 1 (shipped inside the wheel)
│   │   ├── result.schema.json  # part 1
│   │   └── mcp-tools.json      # part 7 (machine-readable tool catalog)
│   ├── core/
│   │   ├── types.py            # dataclasses + error taxonomy      (P1)
│   │   ├── config.py           # load/save/merge/validate, presets (P1)
│   │   ├── cli.py              # argparse → bootstrap              (P1)
│   │   ├── imaging.py          # image IO, crops, palette          (P3)
│   │   ├── pipeline.py         # detect→crop→OCR→assemble          (P4)
│   │   └── jobs.py             # JobManager, progress, cancel      (P4)
│   ├── engines/
│   │   ├── base.py             # Engine protocol + Fake engines    (P2)
│   │   ├── detection.py        # imgutils YOLOModel wrapper        (P2)
│   │   └── ocr.py              # hayai wrapper                     (P3)
│   ├── gui/
│   │   ├── main_window.py  controls.py  workers.py                 (P5)
│   │   ├── image_input.py      # drop/paste/browse                 (P5)
│   │   ├── canvas.py           # annotated view + hover overlay    (P6)
│   │   ├── json_pane.py        # copyable JSON + save              (P6)
│   │   └── activity.py         # MCP activity log + flashes        (P7)
│   ├── mcp/
│   │   ├── server.py           # FastMCP app, transports           (P7)
│   │   ├── bridge.py           # thread-safe GUI marshaling        (P7)
│   │   └── tools.py            # tool implementations              (P7)
│   └── tools/                  # dev/demo entry points
│       ├── detect_demo.py  ocr_demo.py  mcp_client_demo.py
├── tests/                      # per-part unit tests + opt-in model tests
├── examples/                   # sample config, sample image, session transcript
└── research/imgutils_yolo.py   # vendored reference (already saved)
```

---

## 2. Data contracts (defined in Part 1, used everywhere)

### 2.1 Result JSON (`schema_version: "1.0"`) — the primary output

```json
{
  "schema_version": "1.0",
  "created_at": "2026-11-19T12:00:00+08:00",
  "image": { "source": "file|clipboard|mcp-base64|mcp-path",
             "file_name": "page_01.png", "width": 1200, "height": 1700,
             "sha256": "…" },
  "detection": { "repo": "deepghs/AnimeText_yolo", "model": "yolo12m_animetext",
                 "iou_threshold": 0.35, "score_threshold": 0.05,
                 "allow_dynamic": false, "backend": "onnxruntime" },
  "ocr": { "repo": "JustANormalTinkerer/hayai-ocr-v2.7-supernova",
           "device": "cuda:0|cpu", "max_patches": "auto", "batch_size": 16,
           "max_new_tokens": 128, "crop_padding_px": 2, "normalize_nfkc": false },
  "timing_ms": { "detect": 412, "ocr": 1830, "total": 2260 },
  "blocks": [
    { "id": 0, "bbox": { "x0": 812, "y0": 96, "x1": 918, "y1": 402 },
      "label": "text_block", "score": 0.91,
      "text": "……なに!?", "reading_order": 0,
      "color": "#E6194B", "error": null }
  ],
  "block_count": 1
}
```

- `bbox` = integer pixels, xyxy, original image space. `id` = stable index after
  ordering. `color` = deterministic palette assignment (same image+params → same colors).
- Per-block OCR failures don't abort the job: `text: null`, `error: {code, message}`.
- `created_at`/`timing_ms` are the only non-deterministic fields (documented for harnesses).

### 2.2 Config JSON (fully exportable/importable)

```json
{
  "schema_version": "1.0",
  "detection": { "model": "yolo12m_animetext", "iou_threshold": 0.35,
                 "score_threshold": 0.05, "allow_dynamic": false },
  "ocr": { "device": "auto", "max_patches": "auto", "batch_size": 16,
           "max_new_tokens": 128, "crop_padding_px": 2, "normalize_nfkc": false },
  "pipeline": { "reading_order": "manga_rtl", "min_box_area_px": 16 },
  "gui": { "overlay_visible": true, "fill_opacity": 0.18, "labels_visible": true,
           "hover_opacity": 0.5, "zoom_mode": "fit" },
  "mcp": { "enabled": true, "transport": "streamable-http",
           "host": "127.0.0.1", "port": 8765, "auth_token": null },
  "paths": { "hf_cache_dir": null, "last_image_dir": null, "export_dir": null },
  "hf": { "token_env": "HF_TOKEN" }
}
```

- Precedence: **CLI flags > imported config file > built-in defaults**.
- `reading_order`: `score` (detector order) | `top_to_bottom` | `manga_rtl`
  (column-aware right-to-left, top-to-bottom within columns — default for manga).
- Import validates against `schemas/config.schema.json`; unknown keys → warning;
  invalid values → structured error, config unchanged. Export writes the *live* state.
- Applying MCP host/port changes at runtime restarts the server thread (documented).

### 2.3 Error taxonomy (shared by GUI, CLI, MCP)

`NO_IMAGE, BUSY, CANCELLED, INVALID_IMAGE, INVALID_CONFIG, INVALID_ARGUMENT,
MODEL_NOT_FOUND, HF_AUTH_REQUIRED, DOWNLOAD_FAILED, ENGINE_LOAD_FAILED, OCR_FAILED,
IO_ERROR, NOT_IMPLEMENTED, INTERNAL`

*(14 codes as implemented: `INVALID_ARGUMENT` — bad tool/engine arguments — and
`NOT_IMPLEMENTED` — part stubs, decision D8 — were added during implementation.
Every code maps to a fixed process exit code in `core/types.py:EXIT_CODES`.)*

Every error = `{code, message, details?}`. MCP returns these as structured tool errors
(`isError: true` + JSON payload) so deterministic clients can branch on `code`.

---

## 3. The 8 parts

> Dependency graph: **P1 → (P2 ∥ P3) → P4 → P5 → P6 → P7 → P8.**
> P5/P6 can be developed against fake engines (`--mock`) before real models are wired,
> so GUI work is never blocked by downloads/GPU. Each part ends with tests + a demo.

### Part 1 — Foundation: domain types, config system, CLI skeleton
**Scope:** pure Python, zero GUI/model deps. Everything later builds on these contracts.
- `pyproject.toml` with extras; package skeleton; `python -m animetext_studio --help`.
- `core/types.py`: `BBox`, `Detection`, `TextBlock`, `ImageInfo`, `PipelineResult`,
  `JobProgress`, `StudioError` + error codes; (de)serialization to the §2 schemas.
- `core/config.py`: `AppConfig` dataclass tree, defaults (demo defaults + per-model F1
  presets from §0.1), load/save/export/import, partial merge (`set_config` semantics),
  validation with precise messages, CLI-override layering.
- `core/cli.py`: full argument surface (§5) parsed into a bootstrap object.
- `schemas/config.schema.json`, `schemas/result.schema.json` + validation.
  *(Implemented as: native Python validation for precise path-based errors; JSON
  Schemas shipped in-package as the published contract, cross-checked in tests
  via `jsonschema` — decision D5 in tracking.md.)*
- **Tests:** round-trip, merge, precedence, validation failures, schema conformance.
- **Accept:** `pytest tests/test_config.py` green; `--help` shows the full CLI;
  export→import of a default config is byte-stable.
- **Demo:** `python -m animetext_studio --dump-default-config`.

### Part 2 — Detection engine (AnimeText YOLO)
**Scope:** `engines/base.py` + `engines/detection.py`, no GUI.
- `Engine` protocol (load/status/progress/clear) + `FakeDetectionEngine` (deterministic
  boxes for a given image size — used by all later GUI/MCP tests).
- `DetectionEngine`: wraps `YOLOModel('deepghs/AnimeText_yolo', hf_token=…)`;
  `list_models()` (network, with offline fallback to the 5 known names + metadata);
  lazy per-model load with download progress callback; `detect(image, model, iou,
  score, allow_dynamic) -> list[Detection]` (score=None → repo-recommended threshold);
  thread-safe; `unload_all()`; HF gating errors → `HF_AUTH_REQUIRED` with instructions.
- Optional `onnxruntime-gpu` provider selection (config/CLI), CPU default.
- **Tests:** mocked `YOLOModel` (order/format/threshold pass-through, error mapping);
  opt-in `@pytest.mark.models` real-download test (needs HF token).
- **Accept:** unit tests green; with a token, real image → plausible `text_block` boxes.
- **Demo:** `python -m animetext_studio.tools.detect_demo img.png --model yolo12n_animetext --iou 0.35 --score 0.05` → prints detections JSON.

### Part 3 — OCR engine (Hayai Supernova) + imaging utilities
**Scope:** `engines/ocr.py` + `core/imaging.py`, no GUI.
- `core/imaging.py`: load image from path/bytes (EXIF-aware, RGB), PIL↔QImage helpers,
  crop extraction with clamped padding + min-size guard, deterministic 20-color palette.
- `OcrEngine`: lazy `AutoModel/AutoProcessor` (`trust_remote_code=True`, pinned revision);
  device auto: CUDA→fp16+cuda-graphs, CPU→fp32 eager (`HAYAI_DISABLE_COMPILE=1` on CPU);
  `ocr_crops(crops, max_patches, batch_size, max_new_tokens) -> list[str]` using the
  batched `model.ocr([...], escalate=False)` API; internal chunking with progress
  callback + cancellation between chunks; optional NFKC normalization; `unload()`
  frees VRAM/RAM; `FakeOcrEngine` returns deterministic pseudo-text per crop.
- **Tests:** fake-model batching/chunking/cancel/normalize; opt-in real-model smoke
  (repo is public; CPU feasible for a few crops).
- **Accept:** unit tests green; real smoke prints recognized text for a sample crop.
- **Demo:** `python -m animetext_studio.tools.ocr_demo crop.png --max-patches auto`.

### Part 4 — Pipeline orchestrator + JobManager + headless run
**Scope:** the exact 3-step flow the user specified, GUI-independent.
- `core/pipeline.py`: `run(image, config, progress_cb, cancel_event) -> PipelineResult`:
  1. detect → detections; 2. order blocks (`reading_order`) → crops → batched OCR;
  3. assemble §2.1 JSON (ids, colors, per-block error isolation, timings, sha256).
- `core/jobs.py`: single-flight `JobManager` (states idle/running/cancelling, stage +
  percent progress, cancel, result cache = "latest result"), thread-safe, used
  identically by the GUI worker and MCP tools.
- Headless entry: `--headless --image p.png --run [--out result.json]` → stdout JSON,
  exit codes mapped to error taxonomy. `--mock` works here too (no downloads).
- **Tests:** end-to-end with fake engines → golden result JSON (schema-validated);
  cancellation mid-OCR; per-block OCR failure isolation; busy rejection.
- **Accept:** `python -m animetext_studio --headless --mock --image examples/page.png --run`
  emits a valid result JSON; golden tests green.

### Part 5 — GUI shell: controls, image input, worker wiring
**Scope:** a fully interactive window running the pipeline (real or `--mock` engines).
- `MainWindow`: left control panel; right side input preview (top) + output area (bottom)
  in splitters; **whole central widget inside a QScrollArea with H+V scrollbars as
  needed** (large images scroll both ways at 100% zoom); status bar; progress bar + stage.
- Controls: model combo (5 models, tooltips with params/F1/recommended threshold),
  IOU + Score threshold slider⊕spinbox pairs (0–1, 0.01 steps), "recommended threshold"
  button, OCR group (device, max_patches, batch size, max_new_tokens, padding, NFKC),
  reading-order combo, **Generate** + **Cancel**, config **Export/Import** buttons,
  MCP status indicator (placeholder until P7).
- Image input: toolbar Browse; **drag-and-drop** files anywhere on the window;
  **Ctrl+V paste** (clipboard image or copied file); input preview with filename + size.
- `gui/workers.py`: QThread adapter over JobManager; queued signals
  progress/finished/error; buttons disable appropriately while busy.
- **Tests:** pytest-qt — drop/paste/browse set the image; generate with `--mock`
  produces a result; cancel works; config export/import updates widgets.
- **Accept:** offscreen-rendered screenshots confirm layout; full mock flow works
  end-to-end via GUI actions.

### Part 6 — Output visualization + JSON pane
**Scope:** the annotated interactive view and copy-able JSON.
- `gui/canvas.py` (`ResultCanvas`, QGraphicsView-based):
  - draws a **copy** of the input image; per-block **colored translucent fill**
    (default 18% alpha, configurable) + solid border + label chip (`#id · score · text…`);
  - **hover** → box highlights and a **floating 50%-transparent text panel** appears near
    the cursor with the full OCR text (word-wrapped, CJK-capable font); leaves on exit;
  - click → selects block, syncs JSON pane scroll + block list;
  - overlay/labels toggles, fill-opacity slider, zoom (fit / 100% / Ctrl+wheel),
    **Save annotated PNG** (bakes boxes+labels at full resolution).
- `gui/json_pane.py`: pretty JSON (`ensure_ascii=False`, CJK readable), read-only,
  **Copy All** button, **Save As…**, right-click a block → copy its text/JSON.
- **Tests:** hover overlay geometry/opacity (offscreen grab), copy puts exact JSON on
  clipboard, annotated export matches block count.
- **Accept:** visual screenshot review of boxes/labels/hover panel; copy/save verified.

### Part 7 — MCP server + GUI bridge (remote control with visual confirmation)
**Scope:** full §4 spec implemented; the largest single part (may split into 7a server +
7b tools if needed).
- Official **`mcp` Python SDK (FastMCP)**; transports: `streamable-http` (default,
  endpoint `http://host:port/mcp`), `stdio` (harness-spawned), `none`. Server thread +
  own asyncio loop; optional bearer token; default bind `127.0.0.1:8765`.
- `mcp/bridge.py`: queued-invocation marshaling to the Qt main thread, futures with
  timeouts, single-flight busy policy, and the universal **activity hook**:
  every call → log dock entry (time, tool, args, result/error, duration) + widget flash
  + status-bar message; MCP-driven jobs drive the same GUI progress bar and send MCP
  progress notifications.
- `mcp/tools.py`: the 18 tools + 6 resources of §4, each with JSON input/output schemas;
  machine-readable catalog dumped to `schemas/mcp-tools.json`.
- `tools/mcp_client_demo.py`: deterministic SDK client running a full session
  (status → configure → load image → run → fetch JSON → fetch annotated image);
  raw `curl` JSON-RPC transcript in `docs/MCP.md` for non-SDK programs.
- **Tests:** in-process server+client over HTTP with fake engines: every tool
  round-trips, schemas validate, BUSY/CANCELLED paths, activity signal emitted per call;
  stdio handshake test; auth rejection test.
- **Accept:** demo client drives the visible GUI (verified via screenshots); harness
  script exits 0; `mcp dev`/Inspector compatibility.

### Part 8 — Integration, hardening, docs, packaging
- Opt-in real-model end-to-end test (gated: needs HF token; CPU-tolerant settings).
- Error-path suite: no token, offline, corrupt image, huge image, cancel storm,
  config-import garbage, MCP malformed args.
- Performance/UX pass: optional `--preload` (warm engines at startup), model load
  caching, OCR chunk size tuning, memory release on unload.
- `README.md`: install (CPU & CUDA extras), **HF gated-repo token setup**, GUI guide,
  CLI reference, MCP reference (link `docs/MCP.md`), config reference, troubleshooting.
- `examples/`: sample config, sample page, recorded MCP session transcript.
- Final QA checklist against every original requirement (§7).

---

## 4. MCP interface specification (draft v1 — finalized in Part 7)

**Server identity:** name `animetext-studio`, version = app version.
**Transports:** Streamable HTTP (POST/GET/DELETE `/mcp`, MCP session header, JSON-RPC 2.0;
SSE stream for server→client notifications) and stdio. Bind host/port/token via config
file **and** CLI (`--mcp-host`, `--mcp-port`, `--mcp-transport`, `--mcp-token`).
**State:** app-level state is global (one GUI, one latest result) — shared across MCP
sessions; single-flight jobs make concurrent callers deterministic (`BUSY` + job id).
**Conventions:** snake_case tools; every tool returns structured JSON matching its
output schema; errors = `isError:true` with `{code,message,details}` (§2.3); every call
is logged + flashed in the GUI; long tools accept MCP progress tokens.

### Tools

| Tool | Input (JSON schema fields) | Returns | GUI effect (visual confirmation) |
|---|---|---|---|
| `get_status` | – | app version, busy/job state, loaded engines, device, image info, mcp info | activity log + status bar |
| `list_models` | – | 5 models + params/F1/recommended thresholds, cached/loaded flags | log; model combo refreshed |
| `get_config` | – | full live config (§2.2) | log |
| `set_config` | partial config object | applied config + list of changed keys | affected widgets flash & update |
| `export_config` | `path?` | config JSON (+ written path if given) | Export button flash |
| `import_config` | `json?` \| `path?` (exactly one) | applied config | all controls update + flash |
| `load_image` | `path?` \| `base64_png_jpg?` \| `from_clipboard?:bool` | image info (w,h,sha256,source) | input preview shows image |
| `get_image_info` | – | image info or `NO_IMAGE` | log |
| `detect` | `iou_threshold?, score_threshold?, model?, allow_dynamic?` | detections array (boxes only, stage 1) | boxes drawn on output canvas |
| `run_pipeline` | overrides: `model?, iou_threshold?, score_threshold?, ocr:{…}?, reading_order?, include_annotated_image?:bool` | full result JSON (§2.1) (+ annotated PNG as MCP image content if requested); progress notifications | progress bar, boxes appear, JSON pane fills, canvas updates |
| `cancel_job` | – | cancelled job id / state | progress bar aborts, log |
| `get_result` | `block_ids?:int[]` | latest result JSON (optionally filtered) | JSON pane flash |
| `save_result` | `path` | written path + byte count | Save button flash |
| `get_annotated_image` | `scale?:float` | MCP ImageContent (base64 PNG) | canvas flash |
| `set_overlay_visible` | `visible:bool`, `labels?:bool`, `fill_opacity?:float` | applied gui config | canvas toggles live |
| `set_zoom` | `mode:"fit"\|"percent"`, `percent?:int` | applied zoom | canvas zooms |
| `highlight_block` | `id:int`, `duration_ms?:int` | ok | box pulses + hover text panel shown |
| `reset` | – | ok | clears image, result, canvas |

### Resources (read-only subscriptions)

`config://current` · `result://latest` · `image://current` (PNG) ·
`image://annotated` (PNG) · `models://list` · `log://mcp` (recent activity)

### Deterministic-client guarantees

- Stable block ids/colors for identical image+params; only `created_at`/`timing_ms` vary.
- All schemas published in `schemas/mcp-tools.json`; `docs/MCP.md` contains a full
  curl-level session transcript (initialize → notifications/initialized → tools/call …).
- `BUSY` errors include the running job id; clients may poll `get_status` or retry.
- Optional bearer auth: `Authorization: Bearer <token>`; default loopback-only bind.

---

## 5. CLI specification (Part 1 defines, later parts consume)

```
python -m animetext_studio [options]

GUI/app:      --config PATH  --image PATH  --model NAME  --iou F  --score F
              --device {auto,cpu,cuda,cuda:N}  --mock  --preload  --headless
Headless:     --run  --out PATH            (with --headless: pipeline → JSON, exit)
MCP:          --mcp-transport {streamable-http,stdio,none}
              --mcp-host ADDR (default 127.0.0.1)  --mcp-port N (default 8765)
              --mcp-token TOKEN
HF/models:    --hf-token TOKEN  --hf-cache-dir PATH
Misc:         --dump-default-config  --validate-config PATH  --export-config PATH
              --version  --verbose
```

Precedence: CLI > config file > defaults. `--mcp-*` flags also update the live config
shown/exported by the GUI.

---

## 6. Dependencies & environment strategy

| Package | Purpose | Notes |
|---|---|---|
| `PySide6 >= 6.6` | GUI | LGPL; wheels for win/mac/linux |
| `dghs-imgutils >= 0.9` + `onnxruntime` | detection | same lib as official Space; `[gpu]` → onnxruntime-gpu optional |
| `huggingface_hub >= 0.26` | downloads/auth | `HF_TOKEN`, `HF_HOME` |
| `transformers >= 4.57` (verify at P3), `torch >= 2.4` | OCR | CUDA wheels optional; CPU works. **User hardware = 3× RTX 5060 Ti (Blackwell sm_120) → torch ≥ 2.7 + cu128 wheels required on the target machine** |
| `mcp >= 1.9` | MCP server+client | FastMCP, streamable-http + stdio |
| `pillow`, `numpy`, `jsonschema` | imaging/contracts | |
| dev: `pytest`, `pytest-qt`, `pytest-timeout` | tests | |

- Python ≥ 3.10 (3.11 recommended). Extras: `pip install .[gui,ocr,mcp]` vs minimal
  headless-detection installs.
- **Sandbox/dev strategy:** no GPU/display here → GUI developed with `--mock` engines
  and `QT_QPA_PLATFORM=offscreen`, verified by grabbing widget screenshots into the
  workspace for visual review. Real-model integration tests are opt-in
  (`pytest -m models`); detection needs the user's HF token (gated repo), OCR is public.
- Fonts: bundle/fallback to a CJK-capable font for labels & hover panels (Noto Sans CJK).

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Gated detection repo → 401/403 | `HF_AUTH_REQUIRED` error with exact steps (accept terms, set token); GUI banner; offline model-list fallback |
| `trust_remote_code` custom OCR code breaks with new transformers | pin repo revision + verified transformers range at P3; smoke test in CI-like script |
| Qt/asyncio cross-thread deadlocks | strict GuiBridge pattern (queued invoke + futures + timeouts); no Qt imports in mcp/ except bridge; tests for concurrent calls |
| Tall manga pages squashed by fixed-size YOLO preprocessing | expose `allow_dynamic` (aspect-preserving) option; document trade-off vs demo defaults |
| CPU OCR slow on many blocks | batched `model.ocr`, chunked progress + cancel; `max_patches=128` fast preset |
| VRAM/RAM pressure (x-model + OCR) | lazy per-model loading, `unload`, engine status in `get_status` |
| Non-determinism for harnesses | stable ordering/colors/ids; documented volatile fields; golden tests with fake engines |
| GPL-3.0 detection weights | note licensing in README (weights GPL; app code license chosen by user) |

## 8. Execution order (one part per turn)

| Turn | Part | Exit demo |
|---|---|---|
| 1 | P1 Foundation | `--dump-default-config`, config tests green |
| 2 | P2 Detection | `detect_demo` JSON (mock always; real with token) |
| 3 | P3 OCR + imaging | `ocr_demo` text (mock always; real smoke on CPU) |
| 4 | P4 Pipeline/headless | headless `--run` golden JSON |
| 5 | P5 GUI shell | offscreen screenshots + mock GUI flow |
| 6 | P6 Visualization | annotated canvas + hover + copy/save verified |
| 7 | P7 MCP | demo client drives GUI; harness script green |
| 8 | P8 Hardening/docs | full QA checklist + README |

**Open decision points (defaults chosen, flag if you disagree):**
1. MCP via **official SDK / Streamable HTTP + stdio** (not a bespoke raw-TCP protocol) —
   maximizes compatibility with existing agents/harnesses (Inspector, Claude, etc.).
2. Detection via **dghs-imgutils/ONNX** exactly like the Space (ultralytics `.pt` path
   possible later as an alternative engine, not in scope now).
3. Default thresholds = Space demo (0.35 / 0.05) with one-click per-model F1 presets.
4. Default reading order = `manga_rtl`; `top_to_bottom` and `score` available.
5. App name `animetext-studio` (rename freely).
