# AnimeText OCR Studio

Detect the text blocks in an anime/manga page and read them — **in one app, with three doors**:
a PySide6 window a human can watch, a headless CLI that prints one JSON document, and an MCP
server an agent can drive. Every door moves the *same* state; nothing forks.

**Repository:** <https://github.com/paoloinverse/animetext-yolo-MCP> — code, tests, docs and
the build trail. A clone carries **no model weights and no scans** (decision D95): run
`python scripts/fetch_models.py` to build the bundle, then `animetext-studio --doctor` to ask
whether the machine can run it.

```text
page.png ──► AnimeText YOLO (deepghs/AnimeText_yolo)  ──► boxes ──► Hayai Supernova crop OCR
             5 variants, n · s · m · l · x                          JustANormalTinkerer/hayai-ocr-v2.7-supernova
                                                                        └─► box → text JSON (schema 1.2)
```

| Door | What you get |
|---|---|
| **GUI** | Boxes on the page, a block table, the JSON, live controls — and a dock showing every remote call an agent makes |
| **CLI** | `--headless --run` → one result document on stdout, real exit codes, no display needed |
| **MCP** | 21 tools + 6 resources over streamable-HTTP or stdio; 14 error codes, never a broken socket |

Two properties hold everywhere:

* **It runs offline.** Both models are vendored under `animetext_studio/models/` (≈1.3 GB) with
  a sha256 manifest and pinned revisions. `--offline` is a promise the app keeps, not a wish.
* **It is deterministic.** Same page + same config → same boxes, same order, same text. Only
  `created_at` and `timing_ms` change between runs.

---

## Contents

[Install](#install) · [Models & offline bundle](#models--offline-bundle) ·
[Hugging Face token setup](#hugging-face-token-setup) · [Quick start](#quick-start) ·
[GUI guide](#gui-guide) · [CLI reference](#cli-reference) · [MCP reference](#mcp-reference) ·
[Configuration reference](#configuration-reference) · [Result document](#result-document) ·
[Performance notes](#performance-notes) ·
[Troubleshooting](#troubleshooting) · [Development](#development) · [Licensing](#licensing)

---

## Install

**Python 3.10 – 3.13.** (3.14 works for the app but not for `dghs-imgutils`, which still uses
the `ast.Str`/`ast.Num` aliases removed in 3.14 — see [Troubleshooting](#troubleshooting).)

```bash
git clone <this repo> && cd animetext-studio
pip install -e ".[gui,detect,ocr,mcp]"        # or ".[all]"
```

| Extra | Pulls in | Needed for |
|---|---|---|
| *(core)* | `pillow`, `numpy` | headless result documents, config tools, `--model-status` — no engine |
| `gui` | `PySide6>=6.6` | the window |
| `detect` | `dghs-imgutils`, `onnxruntime`, `huggingface_hub` | stage 1 (YOLO) |
| `ocr` | `torch`, `transformers>=4.57`, `accelerate` | stage 2 (Hayai) |
| `mcp` | `mcp>=2.3`, `uvicorn` | the MCP server |
| `dev` | `pytest`, `pytest-qt`, `pytest-timeout`, `jsonschema` | the test suite |

Everything is optional by design: `pip install -e .` on a machine with neither PySide6 nor
torch still gives you a working CLI that reports the bundle and explains, with a structured
error and the right exit code, what it cannot do yet.

### CUDA (OCR on the GPU)

`pip install .[ocr]` gives you CPU torch. For an NVIDIA card install a CUDA build **first**,
from the index that matches the card — the app never asks for a specific one, it just uses
whatever torch reports:

```bash
# NVIDIA Blackwell (RTX 50xx, sm_120) needs torch >= 2.7 with CUDA >= 12.8 wheels:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[gui,detect,ocr,mcp]"

animetext-studio --device cuda:1 --model-status   # the report says what is actually there
```

`ocr.device: "auto"` (the default) picks `cuda:0` when torch sees a usable device and `cpu`
when it does not — and says so in `get_status` / the document, so a device that lies is
impossible. Asking for `cuda:7` when there are two cards is an `ENGINE_LOAD_FAILED` with the
visible-device count, not a silent fall-back.

### The `dghs-imgutils` wrinkle

`dghs-imgutils` pins `numpy<2`, which has no wheels for Python 3.13 and would downgrade numpy
underneath torch. On a numpy-2 install, put it in without its dependency set:

```bash
pip install --no-deps dghs-imgutils
pip install hfutils hbutils onnxruntime opencv-contrib-python pyrfc6266 piexif \
            pyclipper shapely emoji pilmoji
```

That is the verified working set (the list is what `imgutils.generic.yolo` actually imports).
`scripts/setup_dev.sh --detect --gui` does it for you and handles both a rich interpreter and
a bare `--venv`.

---

## Models & offline bundle

The weights are **part of the deliverable** (decision D26): a copy of this folder — or a built
wheel — is a complete offline installation. No download at first run, no cache warm-up, no
token for the OCR model.

**A clone is not a copy** (decision D95, amending D26). The AnimeText weights are their
author's to distribute, the Hayai snapshot is 604 MB on its own, and four files under
`models/` pass GitHub's 100 MB hard limit — so the *repository* publishes no `models/` at
all, and no `sample_scans/` either. One command builds the bundle described below, and it is
the same command that repairs a damaged one:

```bash
python scripts/fetch_models.py --all --ocr    # a HF token is needed for the gated detection repo
```

Everything on this page describes a folder that has been built — by copy or by fetch — and the
app does not care which: a folder with no bundle still starts, still answers `--model-status`,
fails with `MODEL_NOT_FOUND` naming that command, and runs `--mock` regardless.

```text
animetext_studio/models/
├── MANIFEST.json                     provenance: repo, revision, size, sha256 (33 files)
├── animetext-yolo/<variant>/         model.onnx + threshold.json + labels.json + model_type.json
│                                     (l and x also carry the upstream model.pt)
└── hayai-ocr-v2.7-supernova/         model.safetensors + processor/tokenizer files + remote code
                                      .origin.json records the pinned revision
```

| Variant | Params | F1 | Recommended score threshold | `model.onnx` |
|---|---|---|---|---|
| `yolo12n_animetext` | 2.57 M | 0.86 | 0.251 | 10 MB |
| `yolo12s_animetext` | 9.25 M | 0.88 | 0.272 | 36 MB |
| `yolo12m_animetext` *(default)* | 20.1 M | 0.89 | 0.299 | 77 MB |
| `yolo12l_animetext` | 26.4 M | 0.89 | 0.426 | 101 MB |
| `yolo12x_animetext` | 59.1 M | 0.90 | 0.425 | 226 MB |

Hayai Supernova is a ~157 M-parameter crop OCR (SigLIP2 NaFlex encoder + 12-layer GQA decoder)
for Japanese, Chinese, Korean and English. It reads **crops**, not pages — which is exactly
what stage 1 produces.

**Check the bundle** (pure stdlib, no extras, no network, no token):

```bash
animetext-studio --model-status                 # JSON report: variants, sizes, revision, sha256
python scripts/fetch_models.py --verify         # human-readable, exits 1 if incomplete
animetext-studio --doctor                       # the whole machine: deps, pins, bundle, token, font
```

`--model-status` answers *what is on disk*; `--doctor` answers *whether this machine can run it*,
and it is the one command to run before filing a bug — it names the two `dghs-imgutils` behaviours
that make a healthy-looking install fail ([`docs/UPSTREAM.md`](docs/UPSTREAM.md)). It is read-only
and offline by construction: it never installs, never downloads, never writes, and it never
imports `imgutils` — that import is the thing it is checking.

**Re-vendor or extend it** — never re-downloads what is already present and hash-matching:

```bash
python scripts/fetch_models.py --all --ocr      # all five variants + Hayai
python scripts/fetch_models.py --variants yolo12n_animetext --with-sha256
python scripts/fetch_models.py --dry-run --all  # the plan, no bytes moved
```

**Move it.** `--models-dir PATH`, `paths.models_dir` in the config, or `$ANIMETEXT_MODELS_DIR`
(precedence: explicit → env → `<package>/models`). Useful when 1.3 GB should not live inside
site-packages: keep the weights on a shared volume and point every install at it.

---

## Hugging Face token setup

**You do not need a token to run this project as shipped.** The detection weights are already
on disk. You need one only to (re)download from `deepghs/AnimeText_yolo`, which is a **gated**
repository.

1. Open <https://huggingface.co/deepghs/AnimeText_yolo> while logged in and accept the terms
   (the OCR snapshot, `JustANormalTinkerer/hayai-ocr-v2.7-supernova`, is public — no step).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Give it to the app, in this discovery order (first hit wins):

```bash
animetext-studio --hf-token hf_…            # 1. explicit flag (never written anywhere)
export HF_TOKEN=hf_…                        # 2. environment ($HF_TOKEN, or hf.token_env)
printf 'hf_…' > HF_token.txt                # 3. a file: cwd, project root, or models root
export ANIMETEXT_TOKEN_FILE=/secrets/hf     # 4. an explicit file elsewhere
```

What the app does with it: sends it to `huggingface_hub` for the gated repo. What it does
**not** do: write it into a config file, print it, log it, or put it in `log://mcp`. Deleting
`HF_token.txt` removes the capability completely — there is no second place it is stored.

Two things to know:

* `export_config` writes the **live** config, which includes `mcp.auth_token` (the *MCP* token,
  not the HF one). Add `--redact-secrets` / `redact_secrets: true` when the file is going to a
  bug report.
* `python scripts/check_hf_access.py` answers "will this token work?" in one second, with
  actionable messages and the app's own exit codes, before you wait on a 100 MB download.

---

## Quick start

```bash
# 1. The window (MCP server is on by default, loopback, no token)
animetext-studio --image examples/sample_page.png --preload

# 2. One page, one document, no display
animetext-studio --headless --image examples/sample_page.png --run > result.json

# 3. Everything offline, nothing downloaded, no network at all
animetext-studio --offline --headless --image page.png --run --out result.json

# 4. No weights, no torch: the deterministic fake engines (development, demos, CI)
animetext-studio --mock --image examples/sample_page.png

# 5. A headless MCP server an agent can drive
animetext-studio --headless --serve --preload --mcp-port 8765
```

`--mock` is not a toy: it runs the whole product — the same pipeline, the same document, the
same 21 tools — with engines that return fixed boxes and text. It is how the suite stays
offline and how you can wire an agent before the model finishes loading.

---

## GUI guide

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ [MOCK/REAL ENGINES] [OFFLINE] [MCP: 127.0.0.1:8765/mcp] [engines: …] [models: …]│
│ Browse…  Generate  Cancel │ Export config…  Import config…                     │
├──────────────┬─────────────────────────────────────────────────────────────────┤
│ Detection    │  Input                    ┌─────────────────────────────────┐   │
│  model ▾     │   preview · Browse…       │  canvas   [Fit][100%][Save PNG…] │   │
│  IoU ──●──   ├───────────────────────────│  (boxes, labels, hover panel)      │   │
│  Score ──●── │  Output                   ├─────────────────────────────────┤   │
│  [Recommended]│  12 block(s) · yolo12m…  │  # │ score │ text / error │ flags │   │
│ OCR          │                           ├─────────────────────────────────┤   │
│  device ▾    │                           │  JSON  [Copy all][Copy block]…   │   │
│  max patches ▾│                          └─────────────────────────────────┘   │
│  batch/tokens│  ─────────────────────────────────────────────────────────────── │
│ Layout/View  │  MCP activity            time │ tool │ outcome │ ms │ detail      │
│ Run/Config   │  ─────────────────────────────────────────────────────────────── │
├──────────────┴─────────────────────────────────────────────────────────────────┤
│ status text                                        stage ▸ 62%  ▓▓▓▓▓░░░░░     │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Getting a page in:** Browse…, drag and drop onto the window, or **Ctrl+V** a screenshot.
**Getting it read:** Generate (it is the default button, so Enter works).
**Getting one box read again:** right-click its row → **Re-read this block**. It crops the box you
already have and runs OCR on it — the detector is not asked twice — and **Refresh models** asks
the repo for its variant list again without waiting for a restart.

**Controls.** *Detection*: variant, IoU, and score threshold — **Recommended** applies that
variant's F1-optimal score from the repo's `threshold.json`. *OCR*: device, patch budget,
batch size, `max_new_tokens`, crop padding, NFKC normalisation. *Layout*: reading order
(`manga_rtl` rows top→bottom, boxes right→left · `top_to_bottom` · `score`). *View*: overlay,
labels, fill opacity, fit/percent zoom. Everything you touch is the live config an MCP client
sees, and vice versa.

**On the canvas.** Click a box to select it; hover for a panel with the full text; Ctrl+wheel
to zoom; **Save annotated PNG…** bakes the boxes at full resolution. Label chips are
automatically de-collided — smallest box first, and if there is truly no room the chip keeps
`#7` and drops the words, or is dropped and reported, never silently overlapping.

**Flags.** The table has a `flags` column and the summary line says `· 2 flagged` when it
matters:

| Mark | Meaning | What to do |
|---|---|---|
| `⚠` | `possible_truncation` — the text ran to `max_new_tokens`, so it may stop mid-sentence | raise `ocr.max_new_tokens`, then right-click the row → **Re-read this block** (`reocr_blocks` over MCP): one crop, one read, no re-detection |
| `⊂n` | `contained_in` block *n* — a bigger box holds this one | read the parent first; nested boxes are reported, never merged or dropped |

Both are *observations*: they change nothing about the run, and both are in the JSON
(`blocks[].flags`, `blocks[].contained_in`).

**Badges.** `MOCK ENGINES` / `REAL ENGINES`, `OFFLINE`, `MCP: <url>` (or `MCP: failed` — a
server that cannot bind never takes the window down), `engines: det yolo12m · ocr hayai`
so a warmed studio is distinguishable from an empty one at a glance, and `models: presets — <why>`
while the variant list is the one this app ships rather than the one the repo served. That last
one appears the moment the repo stops answering and disappears when it does; it is the same
boolean an MCP client reads as `engines[].model_list_is_fallback`, so the window and the wire
cannot disagree about where the five model names came from.

**The MCP activity dock** is the point of the whole design: every remote call is a row — time,
tool, redacted arguments, outcome, duration — and the widget it touched flashes. When an agent
is driving your screen, you can read over its shoulder.

---

## CLI reference

```
animetext-studio [flags]            # default action: the GUI
```

| Flag | Default | Meaning |
|---|---|---|
| `--help` | — | this table, in the terminal, exit 0 |
| `--version` | — | `AnimeText OCR Studio 0.1.0` — the app version, which is also the MCP `serverInfo.version` |
| `--config PATH` | — | load a config document (CLI flags still win) |
| `--image PATH` | — | page to load at startup |
| `--model NAME` | `yolo12m_animetext` | detection variant |
| `--iou F` | `0.35` | NMS IoU threshold |
| `--score F` | `0.05` | confidence threshold |
| `--device DEV` | `auto` | `auto` · `cpu` · `cuda` · `cuda:N` |
| `--mock` | off | deterministic fake engines |
| `--preload` | off | load the weights **and** run one throwaway inference, so the first real page is not 2–3× slower |
| `--preload-off` | off | no warm-up anywhere in this run — the cold baseline, named (mutually exclusive with `--preload`) |
| `--warm-devices DEV,DEV` | — | benchmark: cold load + one first inference **per device** (`cpu,cuda`, or `all`), printed as a table, then exit |
| `--headless` | off | no GUI |
| `--run` | — | with `--headless`: run the pipeline, print the document |
| `--out PATH` | — | with `--run`: also write the document to a file |
| `--serve` | off | headless MCP server and stay up (implied by `--mcp-transport stdio`) |
| `--mcp-transport {streamable-http,stdio,none}` | `streamable-http` | transport |
| `--mcp-host ADDR` · `--mcp-port N` | `127.0.0.1` · `8765` | bind (`--mcp-port 0` lets the OS pick) |
| `--mcp-token TOKEN` | none | require `Authorization: Bearer …` |
| `--dump-mcp-tools [PATH\|-]` | — | write/print the machine-readable tool catalog and exit |
| `--hf-token TOKEN` | — | HF token for the gated repo (never stored) |
| `--hf-cache-dir PATH` | HF default | hub cache location |
| `--models-dir PATH` | `<package>/models` | on-premises bundle |
| `--offline` | off | never use the network; bundled/cached weights only |
| `--model-status` | — | print the bundle report as JSON and exit |
| `--doctor` | — | read-only environment report — the Python floor, the imgutils import-time install, the numpy pin, torch/CUDA, the bundle, the token, a CJK font — and exit; never installs, never downloads, never writes (`--verbose` adds the JSON) |
| `--dump-default-config` | — | print the built-in defaults and exit |
| `--validate-config PATH` | — | validate a document, print `{valid, errors, warnings}`, exit 2 if invalid |
| `--export-config PATH` | — | write the resolved config (defaults + file + CLI) |
| `--redact-secrets` | off | with `--export-config`: replace `mcp.auth_token` with `<redacted>` |
| `--gui-smoke` | — | build the window, print one line, exit 0 (headless CI) |
| `--verbose` | off | config warnings + the resolved config, always on **stderr** |

**stdout is a document, stderr is prose.** Anything you plan to pipe is on stdout and parses
as JSON; diagnostics never pollute it.

**Exit codes** — the same taxonomy the MCP returns as `{code, message, details}`, so a shell
script and an agent are reading one language (pinned against `EXIT_CODES` by
`tests/test_p8_packaging.py`):

| Code | Exit | Meaning |
|---|---|---|
| — | `0` | success |
| `INTERNAL` | `1` | the app broke its own contract; please report it |
| `INVALID_ARGUMENT` | `2` | the request itself: bad type, two sources where one is expected, out-of-range number |
| `INVALID_CONFIG` | `2` | a config document or patch; `details.issues` lists `{path, message}` and nothing changed |
| `NO_IMAGE` | `3` | no page loaded (or no result published yet); `details.hint` says what to call first |
| `BUSY` | `4` | single-flight: another job owns the machine; `details.job_id` names it |
| `CANCELLED` | `5` | you cancelled it; the previous page and result are untouched |
| `HF_AUTH_REQUIRED` | `6` | Hugging Face refused the gated repo; `details.steps` |
| `DOWNLOAD_FAILED` | `7` | the network answered badly, or `--offline` with an empty cache |
| `ENGINE_LOAD_FAILED` | `8` | missing extra, broken weights, no provider; `details.hint` names the pip extra or the file |
| `OCR_FAILED` | `9` | OCR failed on the page, or on one crop (`blocks[].error`) |
| `IO_ERROR` | `10` | a file could not be read or written |
| `INVALID_IMAGE` | `11` | not a readable image, or too large to decode (a truncated download looks like this) |
| `MODEL_NOT_FOUND` | `12` | not on premises and cannot be fetched; `scripts/fetch_models.py`, or clear `--offline` |
| `NOT_IMPLEMENTED` | `20` | reserved for forward compatibility; **no code path produces it** |
| — | `99` | an unexpected `ValueError` escaped a command path |

The demo modules are the same engines without the app, useful for bisecting a problem:
`python -m animetext_studio.tools.detect_demo page.png`,
`python -m animetext_studio.tools.ocr_demo page.png --bbox 812,96,918,402`,
`python -m animetext_studio.tools.mcp_client_demo --url http://127.0.0.1:8765/mcp`.

---

## MCP reference

Full specification — every tool's schema, per-call GUI effect, progress, the raw-HTTP
walkthrough: **[docs/MCP.md](docs/MCP.md)**. The short version:

```bash
animetext-studio --headless --serve --preload            # http://127.0.0.1:8765/mcp
animetext-studio --mcp-transport stdio                   # harness-spawned; logs to stderr
```

| | |
|---|---|
| **Transports** | `streamable-http` (default) and `stdio`; `none` disables |
| **Auth** | optional bearer token; `401` before the handshake; `GET /health` is exempt |
| **State** | one app state, shared by every session; **single-flight** — a second run gets `BUSY` + the incumbent's `job_id`, never a silent queue |
| **Errors** | `isError: true` with `{code, message, details}` in both `content` and `structuredContent`; resources carry the same object in `error.data` |
| **Progress** | MCP progress tokens on `run_pipeline` / `detect` / `warm_engines` |
| **Catalog** | `animetext-studio --dump-mcp-tools -`, or `animetext_studio/schemas/mcp-tools.json` |

**21 tools:** `get_status` · `list_models` · `load_image` · `get_image_info` · `reset` ·
`run_pipeline` · `reocr_blocks` · `detect` · `cancel_job` · `get_result` · `save_result` ·
`get_annotated_image` · `highlight_block` · `set_zoom` · `set_overlay_visible` ·
`get_config` · `set_config` · `export_config` · `import_config` · `warm_engines` ·
`unload_engines`.

**6 resources:** `result://latest` · `image://current` · `image://annotated` ·
`config://current` · `models://list` · `log://mcp` (the activity ring, token-redacted).

```jsonc
// Claude Desktop / Claude Code
{"mcpServers": {"animetext-studio": {
   "type": "http", "url": "http://127.0.0.1:8765/mcp",
   "headers": {"Authorization": "Bearer <token>"}}}}
```

```bash
npx -y @modelcontextprotocol/inspector --cli --transport http \
     --server-url http://127.0.0.1:8765/mcp --method tools/list
./scripts/mcp_curl_demo.sh          # the same protocol with curl, no SDK
```

A refusal is a result. Every way the app can say *no* — 91 named faults across eight entry
points, each one a test — is in **[docs/ERROR-MATRIX.md](docs/ERROR-MATRIX.md)**.

And every promise the plan made is accounted for: **132 requirements, each with the test or
command that proves it**, in **[docs/QA.md](docs/QA.md)** (generated from
`tests/test_p8_qa.py`, so a row whose evidence disappeared fails the suite).

---

## Configuration reference

One JSON document, `schema_version` **1.2** (documents written against 1.0 and 1.1 still load
and are migrated; the app writes 1.2). The machine-readable version is
`animetext_studio/schemas/config.schema.json`; the built-in defaults are one command away:

```bash
animetext-studio --dump-default-config
animetext-studio --validate-config my.json     # {valid, errors[], warnings[]}, exit 2 if invalid
```

`examples/config.default.json` is that output, byte for byte. `examples/config.custom.json` is
deliberately still a **1.0** document: it loads, the keys it does not mention take their
defaults, and `--verbose` lists what was defaulted — an old file is never a reason to stop.

| Key | Default | Range / values | Notes |
|---|---|---|---|
| `detection.model` | `yolo12m_animetext` | the five variants | |
| `detection.iou_threshold` | `0.35` | 0–1 | NMS |
| `detection.score_threshold` | `0.05` | 0–1 | deliberately permissive; the GUI's **Recommended** sets the repo's F1-optimal value |
| `detection.allow_dynamic` | `false` | | keeps aspect ratio, **and** loads the variant's `.pt` instead of `.onnx` (~2× slower, a few boxes' difference) |
| `ocr.device` | `auto` | `auto` · `cpu` · `cuda` · `cuda:N` | never reports a device it does not have |
| `ocr.max_patches` | `auto` | `auto` · 128 · 192 · 256 · 384 · **512** | Hayai's patch budget; `auto` routes per crop. 512 is the long-text tier |
| `ocr.batch_size` | `16` | 1–512 | crops per OCR batch (the GUI exposes 1–64) |
| `ocr.max_new_tokens` | `128` | 1–1024 | raise it when blocks carry `⚠` |
| `ocr.crop_padding_px` | `2` | 0–64 | pixels added around each box before cropping |
| `ocr.normalize_nfkc` | `false` | | NFKC-normalise the output |
| `pipeline.reading_order` | `manga_rtl` | `manga_rtl` · `top_to_bottom` · `score` | deterministic |
| `pipeline.min_box_area_px` | `16` | 0–100000 | drop boxes below this area |
| `gui.overlay_visible` | `true` | | |
| `gui.fill_opacity` | `0.18` | 0–1 | box fill |
| `gui.labels_visible` | `true` | | |
| `gui.hover_opacity` | `0.5` | 0–1 | dimming of non-hovered boxes |
| `gui.zoom_mode` / `gui.zoom_percent` | `fit` / `100` | `fit` · `percent`; 10–800 | |
| `mcp.enabled` | `true` | | the GUI serves MCP by default · **restart** |
| `mcp.transport` | `streamable-http` | `streamable-http` · `stdio` · `none` | **restart** |
| `mcp.host` / `mcp.port` | `127.0.0.1` / `8765` | | non-loopback warns at startup, louder without a token · **restart** |
| `mcp.auth_token` | `null` | | the **only** secret in the document; `--redact-secrets` on export · **restart** |
| `mcp.log_capacity` | `200` | 10–10000 | the activity ring / `log://mcp`; live, no restart |
| `paths.hf_cache_dir` | `null` | | `null` = HF default (`HF_HOME`) |
| `paths.last_image_dir` / `paths.export_dir` | `null` | | remembered by the file dialogs |
| `paths.models_dir` | `null` | | `null` = `$ANIMETEXT_MODELS_DIR` or `<package>/models` |
| `paths.offline` | `false` | | sets `HF_HUB_OFFLINE` |
| `hf.token_env` | `HF_TOKEN` | | which environment variable to read |

**Live versus restart.** Every key except the five marked **restart** is applied to the running
app the moment it changes — thresholds, device, patch budget, overlay, reading order, even
`mcp.log_capacity`. The five MCP transport keys are read once, when the server is built, so a
running process stores them (an export is faithful, and the next start honours them) but cannot
move the socket you are talking to. Nothing pretends otherwise: `set_config` and `import_config`
answer with `restart_required: ["mcp.port", …]`, and the GUI says the same in its status bar when
you import a file that changes them. Restart the app to move the endpoint.

**Precedence:** built-in defaults ← `--config FILE` ← CLI flags. Unknown keys are *not* fatal
— the app applies what it knows and reports the rest (`--verbose` prints the list; MCP
`set_config` refuses an unknown key outright, because a remote caller means to change
something).

**Round trip:** `export_config` → `import_config` is loss-free, including the token. A literal
`"<redacted>"` arriving in a patch means *keep the live value*, never *set the value to
`<redacted>`*.

---

## Result document

`result.schema.json` (version **1.2**), produced by `--run`, `get_result` and `result://latest`:

```jsonc
{
  "schema_version": "1.2",
  "created_at": "2026-11-24T15:33:35+01:00",
  "stage": "full",                       // "detect" | "full" | "reocr" (see below)
  "image":   {"source": "file", "file_name": "page.png", "width": 900, "height": 1300, "sha256": "…"},
  "detection": {"repo": "deepghs/AnimeText_yolo", "model": "yolo12m_animetext",
                "iou_threshold": 0.35, "score_threshold": 0.05, "allow_dynamic": false, "backend": "onnx"},
  "ocr":     {"repo": "JustANormalTinkerer/hayai-ocr-v2.7-supernova", "device": "cpu",
              "max_patches": "auto", "batch_size": 16, "max_new_tokens": 128,
              "crop_padding_px": 2, "normalize_nfkc": false},
  "timing_ms": {"detect": 2008, "ocr": 2031, "total": 4040},
  "block_count": 12,
  "blocks": [{"id": 0, "bbox": {"x0": 556, "y0": 63, "x1": 704, "y1": 392},
              "label": "text_block", "score": 0.93, "text": "何!? その力は",
              "reading_order": 0, "color": "#E6194B",
              "error": null, "flags": [], "contained_in": null}]
}
```

* `blocks[].error` is per-crop: one failed crop (`OCR_FAILED` on a 4-pixel box) never loses
  you the other eleven.
* `blocks[].flags` carries `possible_truncation`; `blocks[].contained_in` carries the id of
  the smallest box that fully holds this one — nesting is **reported, never resolved**.
  Merging, dropping or splitting is the caller's policy, not the pipeline's.
* `blocks[].color` is the stable id→colour used by the canvas and the export.
* `stage: "detect"` documents keep the **raw** detection boxes; `full` documents carry the
  padded **effective** boxes actually fed to the OCR — the delta is `ocr.crop_padding_px`.
* `stage: "reocr"` is a whole document re-read a block at a time: same boxes, same ids, same
  colours, same reading order, new `text` on the blocks you asked for (`reocr_blocks`). It is a
  document, not a diff — a client that could read the old one can read this one.
* Determinism: everything except `created_at` and `timing_ms` is reproducible byte for byte.

---

## Performance notes

Measured on `examples/sample_page.png` (900×1300, 12 blocks), **CPU only**, no warm-up:

| Model | Blocks | detect | ocr | total |
|---|---|---|---|---|
| `yolo12m_animetext` | 12 | 2.0 s | 2.0 s | 4.0 s |
| `yolo12x_animetext` | 13 | 2.6 s | 2.4 s | 4.9 s |

* **Warm it up.** `--preload` (GUI and headless) and `warm_engines` (MCP) load the weights
  *and* run one throwaway inference, so the first real page does not pay for both. Warm means
  warm: `get_status` reports `engines[].warm` and `inferences`, not a guess.
* **Which device, really?** `--warm-devices cpu,cuda` loads a cold engine pair per device, runs one
  inference through each, releases it, and prints the table:

  ```
  requested  engine     device  state  load_ms  probe_ms  total_ms
  ---------  ---------  ------  -----  -------  --------  --------
  cpu        detection  cpu     ready    1404         96      1500
  cpu        ocr        cpu     ready     812        742      1554
  cuda       detection  cuda    ready    2210       4180      6390
  cuda       ocr        cpu     FAILED      —           —         —   OCR_FAILED

  4 row(s) over 2 device(s), 1 failed · 8213 ms wall
  ```

  Two device columns on purpose: **requested** is what you asked for, **device** is what the
  engine answered with. "asked cuda, got cpu" is the sentence this table exists to make
  unnecessary — and it is the one the old one-line summary could not say. `--verbose` adds the
  JSON. The table is a local measurement and stays local: the MCP door still reports `warm`,
  which is the only part a remote caller can act on.
* **Patch budgets.** Hayai routes each crop to a tier in 128/192/256/384; `512` is the fixed
  long-text tier and is worth setting when a page has full-width narration slabs. Lower tiers
  are for throughput, not for saving your eyes: on the model card's own benchmark `p128` runs
  148.8 images/s against `auto`'s 92.0 and gives up ~0.2 composite points (98.61 vs 98.84)
  and roughly 1.8 CER points.
* **Batch size** trades memory for wall time on the OCR stage; 16 is a sane CPU default.
* **`unload_engines` frees the objects, not the RSS.** A warm `yolo12m` on CPU costs about
  975 MB; unloading returns 191–244 MB and the rest stays with the process until it exits.
  That is the allocator and torch, not a leak — the test suite pins the measured floor
  deliberately instead of asserting a nicer fiction. If you need the memory back, restart the
  process; if you need latency back, don't unload.

---

## Troubleshooting

Start with `animetext-studio --doctor`: it reads the machine the way this table reads the
symptoms, and it exits with the code a real run would have hit. The two `dghs-imgutils` rows are
upstream behaviour, drafted as issues in [`docs/UPSTREAM.md`](docs/UPSTREAM.md).

| Symptom | Cause | Fix |
|---|---|---|
| `HF_AUTH_REQUIRED` (exit 6) | the detection repo is gated and the token cannot see it | accept the terms, then `python scripts/check_hf_access.py`; with the bundle present, `--offline` sidesteps it entirely |
| `MODEL_NOT_FOUND` (exit 12) | that variant is not in the bundle | `python scripts/fetch_models.py --variants NAME`, or pick a bundled one from `--model-status` |
| `ENGINE_LOAD_FAILED` (exit 8) naming a bundled file, with `expected` and `got` in its details | the file **is** there and is not the bytes `MANIFEST.json` recorded — a half-copy, a hand-placed LFS pointer, a bad merge | `python scripts/fetch_models.py --verify`, then `--variants NAME` to re-vendor that one file. Absent and unusable are different faults and are answered differently: the hash is only checked once a load has already failed, and offline the message says it did not retry (online it retried once, and says that too) |
| `ENGINE_LOAD_FAILED: OCR backend is incomplete` | the `[ocr]` extra is missing | `pip install -e ".[ocr]"` (or run `--mock` to work without torch) |
| detection dies on Python 3.14 with `AttributeError: module 'ast' has no attribute 'Str'` | `dghs-imgutils` 0.19 evaluates label metadata with the `ast.Str`/`ast.Num` aliases removed in 3.14 | run the detection stage on Python ≤ 3.13 (`--doctor` says so before you load a page); issue drafted in [`docs/UPSTREAM.md`](docs/UPSTREAM.md#2) |
| numpy got downgraded to 1.x and torch broke | `dghs-imgutils` pins `numpy<2` | reinstall numpy, then `pip install --no-deps dghs-imgutils` (see [Install](#install)) |
| A wall of `Collecting onnxruntime` on first import | imgutils installs a missing ONNX provider *at import time* via pip | install `onnxruntime` (or `onnxruntime-gpu`) yourself, up front; issue drafted in [`docs/UPSTREAM.md`](docs/UPSTREAM.md#1) |
| `torch not built for this GPU` / `no kernel image available` | the wheel predates the card | Blackwell (RTX 50xx, sm_120): `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `MCP: failed` in the panel, window fine | the port is taken | `--mcp-port 0` for any free port, or `--mcp-transport none` |
| `QPA platform: cannot connect to X` | no display | `QT_QPA_PLATFORM=offscreen` (that is how the suite runs); `--gui-smoke` proves the window builds |
| `--headless needs an action` | `--headless` alone | add `--run` or `--serve` |
| `⚠` on blocks | text hit `max_new_tokens` | raise `ocr.max_new_tokens` (128 → 256+) and re-run |
| `⊂n` everywhere, text repeated | the detector found nested boxes | expected and reported; filter on `contained_in` if you want only the outer boxes |
| Narration slab reads as one run-on line | Hayai is a *crop* reader and got a whole-paragraph box | splitting is the caller's job: take the big box, split it yourself, OCR the parts |
| Non-Japanese text looks worse than Japanese | Japanese was trained on real pages; zh/ko/en on synthetic data | expect it; verify on your own pages before trusting a threshold |
| Labels are boxes-of-doom (□□□) | no CJK font on the system | install one (`fonts-noto-cjk`). The canvas uses the Qt UI font; the **exported PNG** looks for `NotoSansCJK` at its known paths and falls back to DejaVu, which has no Japanese glyphs |
| First page is much slower than the rest | cold engines | `--preload` / `warm_engines`; `--warm-devices cpu,cuda` says what cold costs on this machine |
| Memory does not come back after `unload_engines` | allocator floor, not a leak | see [Performance notes](#performance-notes) |
| A config key is ignored | typo or a key from a newer version | `--validate-config FILE` lists it; `--verbose` prints every warning |

---

## Development

```bash
QT_QPA_PLATFORM=offscreen python -m pytest          # 1259 passed, 41 deselected, ~39 s
QT_QPA_PLATFORM=offscreen python -m pytest -m models  # real weights, offline, ~40 s
python scripts/gen_error_matrix.py --check          # docs/ERROR-MATRIX.md matches the tests
python scripts/gen_qa_report.py --check             # docs/QA.md matches the requirement table
animetext-studio --gui-smoke --mock                 # the window builds headless
```

The default run touches **no network, no weights, no display**; `-m models` opts into the real
ones (still offline, still no token — they are on premises). Docs are pinned: the error matrix
and the QA checklist are generated from the tables in `tests/`, the MCP catalog from the live
server, and `tests/test_p8_packaging.py` fails if the README stops matching the parser, the
config schema or the shipped file list. `tests/test_p8_qa.py` is the requirement traceability
matrix itself — [docs/QA.md](docs/QA.md) is its printout, and it fails if a cited test or
command stops existing.

| Path | What lives there |
|---|---|
| `animetext_studio/core/` | Qt-free: config, types, imaging, pipeline, jobs, annotation, model store, CLI |
| `animetext_studio/engines/` | YOLO detector, Hayai OCR, fakes, engine lifecycle |
| `animetext_studio/gui/` | window, controls, canvas, JSON pane, MCP activity dock |
| `animetext_studio/mcp/` | service (the door), tools, resources, bridge, server/gateway |
| `animetext_studio/schemas/` | `result.schema.json`, `config.schema.json`, `mcp-tools.json` |
| `animetext_studio/models/` | the offline bundle + `MANIFEST.json` — **built by `scripts/fetch_models.py`, never published** (D95) |
| `sample_scans/` | the user's own pages, if they choose to keep them here; never published (D95) |
| `docs/` | [MCP.md](docs/MCP.md) · [ERROR-MATRIX.md](docs/ERROR-MATRIX.md) · [QA.md](docs/QA.md) · [UPSTREAM.md](docs/UPSTREAM.md) |
| `scripts/` | `fetch_models.py` · `check_hf_access.py` · `setup_dev.sh` · `gen_error_matrix.py` · `gen_qa_report.py` · `make_sample_image.py` · `mcp_curl_demo.sh` |
| `examples/` | `sample_page.png`, config examples, a recorded MCP session |
| `PLAN.md`, `tracking.md`, `P9.md` | the build log: the plan, then decisions D1–D95 and one entry per turn |

**Deliberately not done** (each one is a decision, not an oversight): no de-duplication of
nested boxes (flag, don't drop — D58); no language policy or "expected language" filtering
(the caller knows what a manga should contain, this app does not — D60); no sub-crop splitting
of dense slabs (same reason); no concurrent jobs (single-flight makes every caller
deterministic — D32).

---

## Licensing

**The application code is Apache-2.0.** The canonical text is [`LICENSE`](LICENSE) (the
unmodified Apache-2.0 body, with only the appendix's copyright line filled in), and
[`NOTICE`](NOTICE) is the inventory of everything in here that is not ours. Under it you may
use, modify and redistribute the code; contributions arrive under the same terms automatically
(§5); there is a patent grant (§3); the project's names and marks are not part of it (§6).

The **model weights are a separate matter** — they arrive in the tree when the bundle is built,
and nobody here can relicense them. The repository itself publishes none of them (D95):

| Component | What it is | Licence | Its text ships at |
|---|---|---|---|
| app code — `animetext_studio/` (minus `models/`), `scripts/`, `tests/`, `examples/` | this application | **Apache-2.0** | [`LICENSE`](LICENSE) |
| `deepghs/AnimeText_yolo` | detection weights, 5 ONNX variants | **GPL-3.0** | `animetext_studio/models/animetext-yolo/LICENSE.GPL-3.0.txt` |
| `JustANormalTinkerer/hayai-ocr-v2.7-supernova` | OCR weights + the `trust_remote_code` sources | **Apache-2.0** | same licence as the code; provenance in the snapshot's `.origin.json` |
| PySide6 / shiboken6 | Qt bindings | LGPL-3.0 (dynamic linking) | pip metadata |
| `dghs-imgutils`, `onnxruntime`, `torch`, `transformers`, `huggingface_hub`, Pillow, NumPy, `mcp`, `uvicorn` | runtime deps | MIT / BSD / Apache, per package — see [`NOTICE`](NOTICE) | pip metadata |

In practice:

* **Running it** — GUI, CLI or MCP, on your machine or on your own network — asks nothing of you.
* **Copying, modifying or shipping the code** keeps one obligation: ship `LICENSE` and `NOTICE`
  with it and note what you changed. `pyproject.toml` declares `license = "Apache-2.0"` and
  `license-files = ["LICENSE", "NOTICE"]`, so a built wheel carries both by itself.
* **Shipping the bundled weights** carries the GPL-3.0 obligations for *those files* in addition:
  pass the text along, offer the same terms, and make the corresponding source of the weights
  available. Apache-2.0 is GPL-3.0-compatible in that direction, so the two licences do not
  conflict — but the thing to reason about is the bundled artifact you distribute, not the code
  in isolation.
* **If you cannot meet those**, ship the slim wheel — which is now also what a clone is: a
  checkout with no `animetext_studio/models/`, users running `python scripts/fetch_models.py`
  themselves. Nothing in the app assumes the weights are present — `--offline`,
  `--model-status` and the fallback model list are built for exactly that, so it is a
  packaging decision and not a fork. And whatever you do, **do not re-publish the AnimeText
  weights**: they are GPL-3.0, and their author's choice about distributing them is not yours
  to make on their behalf (D95).
* **`sample_scans/`** are the user's own pages. Nothing in this repository licenses them to
  anyone, the app never renders them into documentation or test output, and the repository
  does not carry them (D95) — the tests that want a page use `examples/sample_page.png`,
  which `scripts/make_sample_image.py` draws.

This section is a summary, not legal advice: read [`LICENSE`](LICENSE) and the two upstream
model cards before you redistribute a bundle.
