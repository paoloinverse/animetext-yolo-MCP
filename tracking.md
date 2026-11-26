# AnimeText OCR Studio — Progress Tracker

Plan: [`PLAN.md`](PLAN.md) · Parts are implemented one per turn, in dependency order.

> **All eight parts are complete (2026-11-24).** This file is the history: status board below,
> decisions log (D1–D79), one entry per turn, then the standing rules and the open follow-ups.
> The requirement-by-requirement proof is [`docs/QA.md`](docs/QA.md) (generated from
> `tests/test_p8_qa.py`); the error surface is [`docs/ERROR-MATRIX.md`](docs/ERROR-MATRIX.md);
> how to use the product is [`README.md`](README.md). The per-part scratch files
> (`P8.md`, `STATE.md`) were folded in here and deleted.

**Legend:** ✅ done · 🔄 in progress · ⬜ not started · ⛔ blocked

## Status board

| Part | Scope | Status | Exit demo / verification |
|---|---|---|---|
| P1 | Foundation: domain types, config system, schemas, CLI | ✅ **re-inspected (T4)** | 110 tests at authoring time; **Turn 4 audit found + fixed 5 defects** (wrong `TEXT_LABEL`, 3 native↔schema contract divergences, `KeyError` on malformed payloads). Suite now **232 offline + 7 real-model** |
| P2 | Detection engine (AnimeText YOLO via imgutils/ONNX) | ✅ **verified on real models + runs fully offline (T5)** | **13/13 real-model tests green** (7 Hub + 6 bundled); yolo12n/s/m vendored on premises and **infer with `--offline`, no token, no socket** (yolo12m: 12 boxes on the sample page in 2.1 s CPU, threshold 0.299 read from the bundled `threshold.json`); repo `labels.json`/`threshold.json` match our presets for all 5 variants; gated / threshold / provider paths exercised live; suite now **278 offline + 13 real-model** |
| P3 | OCR engine (Hayai Supernova) + imaging utils | ✅ **verified on the real vendored weights, fully offline (T6)** | **76 new unit tests** (`tests/test_ocr.py`) driving the engine through two lazy import seams — no torch needed; opt-in `-m models` runs **real Hayai inference on the 666 MB vendored snapshot, `offline=True`, CPU, 7.8 s**; `ocr_demo --offline` on the sample page → 8 crops → 8 recognitions (6.7 s, `detail: "auto, cuda-graphs off, bundle"`); `FakeOcrEngine` deterministic + blank-crop→`""`; suite now **386 offline + 14 real-model** |
| P4 | Pipeline orchestrator + JobManager + headless run | ✅ **verified end-to-end on the real vendored weights, offline (T7)** | **71 new tests** (`test_pipeline.py` 42 + 1 opt-in real-weights, `test_jobs.py` 28); `--headless --mock --image examples/sample_page.png --run` → schema-valid document, byte-stable across runs; **real offline run** (`out/pipeline_real.json`): 12 blocks, 12 real Japanese strings, `detect 2289 ms / ocr 4580 ms / total 6869 ms` on CPU; one poisoned crop isolates instead of losing the page; busy/cancel/listener semantics tested against real threading; suite now **457 offline + 15 real-model** |
| P5 | GUI shell: controls, image input (drop/paste/browse), workers | ✅ **built and driven offscreen, screenshots reviewed (T8)** | **23 `tests/test_gui.py` tests** (PySide6 6.12, offscreen): browse/drop/paste set the image, Generate runs the *same* `JobManager` path as `--run`, buttons follow the job, cancel works, config export/import round-trips through the widgets, unbundled variants are disabled offline; **`--gui-smoke`** builds the window and exits 0 for CI; `out/gui_mock.png` reviewed (CJK renders). Suite now **481 offline + 15 real-model** |
| P6 | Output visualization: boxes, hover overlay, JSON pane | ✅ **built; reviewed on mock *and* real offline runs (T9)** | `core/annotation.py` (Qt-free recipe) + `gui/canvas.py` (`QGraphicsView`, hover panel, zoom, full-res export) + `gui/json_pane.py` (byte-exact JSON, copy/save, per-block copy), View group in the panel wired through the same `apply_config`/`collect_into` pair, three views synced on the block id; **80 new tests** (41 pure + 39 offscreen GUI) incl. pixel probes; **7 defects found and fixed** (two `QRect`/`QRectF` xyxy traps, `clicked`→`path=False` on four buttons, chips off-page, a raising `QSyntaxHighlighter`, buried chips, an unconnected `view_changed`); suite now **561 offline + 15 real-model** |
| P7 | MCP server + GUI bridge (18 tools, 6 resources) | ✅ **complete (T11)** | `StudioService` façade + `StudioServer` (stdio / streamable-HTTP, optional bearer token), 18 tools + 6 resources, activity log bridged into the GUI dock; demo client drives the GUI and the harness script is green. Suite at close: **639 offline + 15 real-model** |
| P8 | Hardening, real-model tests, docs, packaging | ✅ **complete (T19)** | 8.1 result document **1.1** (`stage`) · 8.2 security/config hardening · 8.2b result **1.2** flags · 8.3 warm-up as a job + measured release · 8.4 error matrix (84 faults × 8 doors, D64–D71) · 8.5 real-model end-to-end (25 tests, D72) · 8.6 annotation polish (D73) · 8.7 README + packaging (D74–D76) · **8.8 final QA: 115 requirements traced in `docs/QA.md`, licence applied (D77), restart-required reporting (D78), the traceability table itself (D79)**. Suite at close: **1092 offline + 40 `-m models`** |
| P9 | The five approved enhancements, and the trail | ✅ **complete (T20)** | 9.1 a corrupt bundled file is diagnosed, not mistaken for absent — retried once online, recorded in `status().detail` (D84, D85) · 9.2 the model list's provenance on every door + the fifth badge (D90) · 9.3 `reocr_blocks`, the 21st tool: one crop, no re-detection, a whole document back, right-click in the table (D81, D82) · 9.4 per-device cold latency, `--warm-devices`, `--preload-off` (D83, D86) · 9.5 `--doctor` + `docs/UPSTREAM.md` (D87–D92) · **9.6 the trail: matrix 84→91, checklist 115→131, floors, README / MCP / UPSTREAM, the sign-off** (D93 an option given `""` is refused, not ignored; D94 `--dump-mcp-tools PATH` writes to PATH). Suite at close: **1253 offline + 40 `-m models`** |
| REPO | Putting the project in a repository, and deciding what a repository may hold (D95) | ✅ **pushed: `github.com/paoloinverse/animetext-yolo-MCP`, `main` = `e95e8be` (T21)** | `.gitignore` rewritten: **no `animetext_studio/models/`** (the AnimeText weights are their author's to distribute; the Hayai snapshot is 604 MB alone and four files pass GitHub's 100 MB hard limit), **no `sample_scans/`**, no `.npm/`/`uploads/`/`research/`/`tmp*`, both token files named rather than globed. `examples/` carried this machine's absolute paths in 45 places, local account name included — now `/home/you/animetext-studio`. New `TestWhatTheRepositoryPublishes` checks `git ls-files` (the published set, not the folder); the secret scan now knows `ghp_…` and `github_pat_…` as well as `hf_…`. README/NOTICE tell a clone to run `scripts/fetch_models.py`. Checklist **131 → 132**, suite **1253 → 1259**; `[project.urls]` and the README now name the repository. Pushing needed `contents:write` — a pull-request token authenticates as the right account and is still refused (403) — and no fine-grained token of any shape can file into someone else's repository |

## Environment notes

> Re-probed on Turn 3 — the sandbox changed since Turn 1 (it was Python 3.11.2 + no pip).

- **Sandbox (dev/CI here):** Python **3.13.15**, no GPU visible to torch
  (`torch.cuda.is_available() == False`, although `nvidia-smi` exists), no display. GUI work
  will use `QT_QPA_PLATFORM=offscreen` + widget screenshots.
- The default interpreter is the Unsloth Studio env and already ships pytest 9.1.1,
  jsonschema, pillow 12.3, numpy 2.5.5, `torch 2.11.0+cu130`, transformers 5.5.4,
  `mcp 2.3.0`, uvicorn, fastapi, tqdm → **no `.venv` needed here**. pip installs land in
  `./.unsloth-packages` (`PIP_PREFIX`, already on `PYTHONPATH`, `PIP_IGNORE_INSTALLED=1`);
  `pip --target` fails while `PIP_PREFIX` is set (`env -u PIP_PREFIX …` if ever needed).
  `scripts/setup_dev.sh` now handles both worlds (rich interpreter vs `--venv`).
- **Detection deps installed and verified importable** (`imgutils.generic.yolo`):
  dghs-imgutils **0.19.0** via `--no-deps`, plus hfutils, hbutils, opencv-contrib-python,
  pyrfc6266, piexif, pyclipper, shapely, emoji, pilmoji. Two gotchas worth remembering:
  - dghs-imgutils pins `numpy<2` (no py3.13 wheels) → install it `--no-deps`; the YOLO path
    runs fine on numpy 2.5.5 (verified).
  - `imgutils.utils.onnxruntime` **silently pip-installs onnxruntime at import time** when
    it is missing (network during import!). Here it pulled onnxruntime 1.32.0 (CPU only:
    `Azure`/`CPU` providers exposed). The provider comes from the `ONNX_MODE` env var.
- **HF cache:** the sandbox pre-sets `HF_HOME=./.cache/huggingface` — a shared cache that
  also holds unrelated GGUF models. Use `--hf-cache-dir` to isolate project weights, and
  see D13 for why the env var alone is not enough.
- **Gating verified for real:** anonymous *listing* of `deepghs/AnimeText_yolo` succeeds
  (so `list_models()` returns the 5 real variants without a token), but every file
  download returns **401 GatedRepoError** → our `HF_AUTH_REQUIRED` path was exercised
  against the live repo (exit 6 with the setup steps).
- **OCR repo verified public** (`gated=False`) and **already vendored** (see D26):
  `animetext_studio/models/hayai-ocr-v2.7-supernova/` at revision `f5f0f617f514…` —
  `model.safetensors` (604 MB, 395 tensors F32), `modeling_hayai.py`,
  `processing_hayai.py`, `configuration_hayai.py`, preprocessor/processor configs,
  tokenizer, plus the repo's own `yolo26m-text.pt`. **Part 3 needs no download at all**
  and can be smoke-tested here on CPU with no token.
- **On-premises bundle:** `animetext_studio/models/` = 773.3 MB, `MANIFEST.json` with
  sha256 for every file. Check it with `python scripts/fetch_models.py --verify`
  (→ *RESULT: usable offline*) or `animetext-studio --model-status`. Detection weights
  n/s/m only (l/x exceed this sandbox's ~100 MB per-file limit); `paths.models_dir` /
  `--models-dir` / `$ANIMETEXT_MODELS_DIR` relocate the whole bundle.
- CJK fonts are available (`/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc`), so the
  sample page, label chips and the Part 6 hover panel can be reviewed visually here.
- Installed for tests: `pytest-timeout` (used by the `models` marker).
- **Installed for Part 5: PySide6-Essentials 6.12 + shiboken6** (`pip install --no-deps
  PySide6-Essentials shiboken6`; the `PySide6` *meta*-package contains no code, and
  `--no-deps` avoids pulling Addons — the Essentials wheel is 84 MB, inside the download
  limit) **and pytest-qt 4.5** (it upgraded pytest 9.1.1 → 9.2.0 inside `./.unsloth-packages`;
  the suite is green afterwards). Three PySide6 6.12 gotchas, each hit for real:
  - **`pyqtSignal` / `pyqtSlot` no longer exist** → use `Signal` / `Slot`.
  - `QComboBox` has no `setItemEnabled` → `combo.model().item(i).setEnabled(False)`.
  - `QDragEnterEvent(pos, actions, data, buttons, modifiers)` — actions *before* data, and
    `QEnterEvent` has no `mimeData()` (use `QDragEnterEvent`).
  GUI tests and screenshots need `QT_QPA_PLATFORM=offscreen` (`tests/test_gui.py` sets it at
  import). The `Fontconfig error: Cannot load default config file` warning is harmless: Qt
  still finds **Noto Sans CJK JP**, so Japanese renders correctly in offscreen grabs.
- **Verification corpus:** `sample_scans/` (7 JPEG pages, ~1000×1500, added by the user on
  Turn 3). Used as **opaque binary inputs only** — I/O, sha256 fingerprinting and box
  geometry. Their content is not inspected, and annotated renders of them are not written
  into the workspace; run `--annotate` locally if you want previews of your own scans.
- Workspace noise from environment probing (not part of the project, safe to delete):
  `.venv/` (empty nested-venv experiment) and `.probe/` (a few downloaded wheels).
- **GPU note for detection:** ONNX Runtime's CUDA EP support for Blackwell (sm_120) depends
  on the `onnxruntime-gpu` build, and imgutils always appends `CPUExecutionProvider`, so a
  missing CUDA EP degrades to CPU instead of failing. Detection is cheap (the official Space
  runs on cpu-basic); the GPU matters for the Part 3 OCR engine (torch cu128).
- **User target machine (from Space page metadata):** 3× NVIDIA **RTX 5060 Ti 16 GB**
  (Blackwell, sm_120) → Part 3 must use **torch ≥ 2.7 with CUDA 12.8 wheels**
  (`--index-url https://download.pytorch.org/whl/cu128`); older cu121 wheels will NOT
  run on sm_120. Detection (ONNX Runtime) is CPU-friendly; the official Space itself
  runs on cpu-basic.
- **HF access:** `deepghs/AnimeText_yolo` is gated (`gated: auto`) → user must accept
  terms on the model page and provide `HF_TOKEN` (env or `--hf-token`). The Space
  confirms this pattern (it uses an HF_TOKEN secret). OCR repo is public.
- Security: never persist HF tokens in exported configs; `mcp.auth_token` is included
  only because the user explicitly wants fully exportable MCP settings.

## Decisions log

| # | Decision | Rationale |
|---|---|---|
| D1 | MCP via official `mcp` SDK: Streamable HTTP (default 127.0.0.1:8765) + stdio | agent/harness compatibility (PLAN §4) |
| D2 | Detection via `dghs-imgutils` `YOLOModel` (ONNX), exactly like the Space | verified API; CPU-capable; no torch for detection |
| D3 | Defaults = Space demo (m / iou 0.35 / score 0.05) + per-model F1 presets | matches official demo behavior |
| D4 | Default reading order `manga_rtl` | manga use-case |
| D5 | Config validation is **native Python** (precise path-based errors); JSON Schemas shipped as the published contract and cross-checked in tests via `jsonschema` (dev dep) | zero runtime deps for P1; better error messages; schemas still authoritative for MCP clients |
| D6 | Schemas live **inside the package** (`animetext_studio/schemas/`) so wheels carry them; PLAN layout updated | packaging correctness |
| D7 | Extras split: `[gui] [detect] [ocr] [mcp] [dev] [all]` (PLAN had no `[detect]`) | headless-detection installs without torch/Qt |
| D8 | Stub entry points return `NOT_IMPLEMENTED` (exit 20) until their part lands | deterministic harnesses never get silent success |
| D9 | CLI adds `--validate-config PATH` and `--export-config PATH` | makes P1 independently useful/testable |
| D10 | Lenient import: missing config keys → defaults (+warning); unknown keys → warning (error in strict mode); invalid values → always error | friendly import, strict correctness |
| D11 | HF token never written to config files; passed via env/CLI only | secret hygiene |
| D12 | OCR repo revision pinned (`5289b2c…`) in `core/types.py` | reproducible `trust_remote_code` |
| D13 | `configure_hf_cache_dir()` sets `HF_HUB_CACHE` **and** patches `huggingface_hub.constants.HF_HUB_CACHE` | hub resolves that constant at import time, so the env var alone is silently ignored in a long-running GUI/MCP process (verified experimentally) |
| D14 | `load()` warming is *logical*, not a session handle — imgutils caches sessions per `(pid, thread_ident, model)` | warming on thread A does not reuse the session on thread B → **P4's JobManager must keep inference on one dedicated worker thread** to get session reuse (re-opening from the local cache is cheap but not free) |
| D15 | `load()` pre-fetches `model.onnx` + `model_type.json` + `threshold.json` (auxiliary failures tolerated) | auth errors surface at load time with the right code instead of mid-`predict`; later runs work offline |
| D16 | `_import_yolo_model()` falls back from `imgutils.generic` to `imgutils.generic.yolo` | `imgutils/generic/__init__.py` eagerly imports every sibling (opencv/timm-backed); a minimal `[detect]` install must still work |
| D17 | ONNX provider selection = imgutils' own `ONNX_MODE` env var, exposed as an engine/`detect_demo` flag `--provider {auto,cpu,cuda,gpu,tensorrt,trt,coreml,openvino,directml}`; **not** added to config schema 1.0 | keeps schema 1.0 frozen; `auto` = CUDA when the installed onnxruntime exposes it, else CPU. Main CLI flag + GUI control arrive with the P4/P5 wiring |
| D18 | Engine contract gained `recommended_threshold(model)` and `resolved_score_threshold(model, requested)` (base + real + fake) | the result JSON must record a *concrete* `detection.score_threshold` even when the caller passed `score=None`; the fake's `None → 0.05` constant keeps mock runs deterministic. Also drives the GUI "recommended threshold" button (P5) and MCP `list_models` (P7) |
| D19 | Default test run excludes `models`-marked tests (`addopts = -m 'not models'`) | `pytest` stays offline and fast; `pytest -m models` opts into downloads + token (an explicit `-m` on the CLI overrides the default) |
| D20 | `examples/sample_page.png` is **generated** by `scripts/make_sample_image.py` | synthetic manga-like page (speech bubbles, tategaki columns, narration band) gives the demos/tests a realistic CJK page without shipping copyrighted artwork |
| D21 | Two config shapes: `config.schema.json` = **canonical/emitted** (every key, unknowns forbidden); `config.partial_config_schema()` = **accepted** shape (`config-partial-1.0`: sections optional, unknown keys allowed), *derived* from the canonical one in code | the native validator is lenient by design (D10) while the published schema was strict, so MCP clients would be handed a contract the server does not honour; deriving it makes drift impossible (test-enforced) |
| D22 | Integer config fields accept **integral floats** (`16.0`) and are coerced to `int` on load (`_INT_FIELDS` / `_normalise`) | JSON Schema counts `16.0` as an integer, so native and schema disagreed; coercion also keeps `merged()` honest for MCP `set_config` (no phantom changed keys) |
| D23 | Every `from_dict` maps malformed payloads to `StudioError(INVALID_ARGUMENT)` through `types._build()` | a bare `KeyError` from a file/clipboard/MCP payload would surface as `INTERNAL`, breaking the {code,message} contract (PLAN 2.3) |
| D24 | `DetectionEngine.status().device` is refreshed from the session's real ORT providers after `load()` | "auto" hides the fact that a machine without a usable CUDA EP is really on CPU; GUI + MCP `get_status` must tell the truth |
| D25 | `--verbose` prints config-file warnings (unknown / defaulted keys) | D10's lenient import must not be invisible |
| D26 | **On-premises model bundle**: weights are vendored into `<package>/models/` (`scripts/fetch_models.py`) and resolved **bundled-first, Hub second**; `paths.models_dir` / `--models-dir` / `$ANIMETEXT_MODELS_DIR` override the root; `paths.offline` / `--offline` forbids the network entirely | the user must be able to back the app up as one folder and run it with no HF account, no token and no network ("safer in case HF ever goes under"). The Hub becomes a *fetch* fallback, not a runtime dependency. Implemented as `core/model_store.py` (pure stdlib) + a `BundledYOLOModel` subclass of imgutils' `YOLOModel` |
| D27 | The HF model listing is **bounded** (`LISTING_TIMEOUT_SECONDS = 10`, engine arg `listing_timeout`) and skipped altogether when offline; past the bound we answer from bundle + presets and record it in `status().detail` / `model_list_is_fallback` | found live: imgutils resolves `model_names` with an fsspec glob that ignores `HF_HUB_OFFLINE` and **blocks forever** on a stalled/401-looping connection — the listing sits on the first `detect()`, so a GUI/MCP process would wedge. Reproduced (90 s hang, faulthandler trace) and fixed |
| D28 | One definition of "bundled": the backend asks the `ModelStore` (not its own directory scan) which variants are usable (≥ `_MIN_WEIGHTS_BYTES`), and `write_manifest()` **merges** with the previous manifest while refusing to record a file that is not on disk | engine listing, `source` labels and `--model-status` can no longer disagree, and a partial run (`--ocr` alone) or a failed download cannot leave a phantom entry claiming a model we do not have (both reproduced during vendoring: 6 phantom 0-byte entries; `--verify` now reports `RECORDED BUT ABSENT`) |
| D29 | Unit tests are **isolated from the real bundle** (autouse `conftest` fixture points `$ANIMETEXT_MODELS_DIR` at an empty temp dir; `-m models` opts back in) | vendoring weights into the package must not change what an unrelated test observes — otherwise the suite would pass here and fail on a clean checkout (or vice versa) |
| D30 | `OCR_REVISION` bumped `5289b2cfdfd3…` → **`f5f0f617f514…`** (repo HEAD, = the vendored snapshot) | the Part 1 pin had gone stale: the newer snapshot updates `processing_hayai.py` (the API Part 3 calls) and adds `yolo26m-text.pt`. Keeping the old pin would make a bundled offline install pull a *different* snapshot from the Hub. Guarded by a real-model test comparing `.origin.json` with `OCR_REVISION` |
| D31 | **Unsupported `QImage` formats are reported, never converted.** `qimage_to_pil` accepts only `Format_RGB888`; the duck-type interface is probed in two passes (`isNull`/`format`, then `width`/`height`/`bytesPerLine`/`constBits`) so a wrong object's error names *every* missing method, and a wrong format's error names the format plus the conversion the caller must do | a blind `convertToFormat()` turns "wired the format wrong" into garbled OCR input, and the lossy/intent-dependent choice (greyscale? premultiplied ARGB?) belongs to the GUI layer that knows what was pasted. Check order is contractual: head → null → **format** → buffer |
| D32 | The engine sets `HAYAI_DISABLE_COMPILE=1` **before** importing the remote code (CPU/other only) and never overrides a value the operator set | `_maybe_compile` reads it at *module import* time (same trap as D13); setting it after `from_pretrained` is a no-op that silently leaves `torch.compile` on for CPU. The test asserts the env value **at import time**, not afterwards |
| D33 | The OCR engine owns the chunking: `batch_size` is **never** forwarded to `model.ocr`, and `escalate=False` always | in the vendored `processing_hayai.py`, `batch_size` only applies when `max_patches="auto"` **and** `len(images) > batch_size`, and it offers no progress and no cancellation point. Chunking in the engine keeps memory bounded *and* yields per-chunk progress with a cancel check before **and** after every batch |
| D34 | A misaligned `model.ocr` reply (`len(texts) != len(crops)`) is fatal — `OCR_FAILED`, "N texts for M crops" | zipping anyway would silently attach the wrong string to the wrong bubble: the worst failure mode in this app, because it looks like success |
| D35 | OCR device strings are validated at construction by `_CUDA_DEVICE_RE` (`cuda` \| `cuda:<n>`) | `startswith("cuda")` accepted `cuda-x`, `_resolve_device` parsed it as index 0 and then reported `cuda:0` — a lie in `status()`. A missing `cuda:N` stays a refusal (`ENGINE_LOAD_FAILED`), never a silent downgrade. The rule is deliberately the same one `config.py:DEVICE_RE` and `config.schema.json` already enforce: the engine had been *laxer* than the config layer, so a hand-built engine (the GUI/MCP path, which may construct it directly) accepted garbage the schema would have rejected |
| D36 | OCR weights resolve **bundled-first** like detection (extends D26) and `OcrEngine.plan()` is a **no-torch, no-I/O** pre-flight of exactly what `load()` would do (repo, revision, bundled, missing files, device, dtype, graphs); `weights_source` records which source won | P5 needs a doctor panel that answers "can this run offline?" without importing torch or touching the network, and P7 must be able to explain a `MODEL_NOT_FOUND` before starting a job |
| D37 | **Per-block OCR isolation with one bounded retry pass.** The pipeline makes a *single* batched `ocr_crops` call (D33). If it fails with a non-fatal code, the same crops are re-run one at a time and the failure is recorded on the block(s) that actually failed. Fatal codes (`ENGINE_LOAD_FAILED`, `MODEL_NOT_FOUND`, `DOWNLOAD_FAILED`, `HF_AUTH_REQUIRED`, `INVALID_ARGUMENT`, `INVALID_IMAGE`, `CANCELLED`) propagate immediately, and a single-crop page is never retried | one unreadable crop must not cost the user the other 11 blocks; but retrying a missing model or a bad argument N times just hides the real error behind N copies of it and multiplies the wait |
| D38 | The document's `block.bbox` is the **effective** box (padded by `ocr.crop_padding_px`, clamped to the image), not the raw detector box | it is the truth about the pixels the text came from, it is always inside the image, and P6's hover overlay then matches what OCR actually read. The raw box is recoverable by shrinking the padding; the reverse is not |
| D39 | `text: null` **always** carries an `error` object. Boxes skipped for size get the schema-legal code `SKIPPED_TOO_SMALL` (`pipeline.SKIP_ERROR_CODE`); an OCR-disabled run uses `SKIPPED`. `text: ""` means "OCR ran and read nothing" | three different facts (unreadable / not attempted / nothing there) must stay distinguishable in the JSON pane and in P7's answers, and "null with no reason" is the kind of ambiguity that turns into a bug report |
| D40 | Two effects, one configured floor: `pipeline.min_box_area_px` **drops** detections (they never enter the document), `imaging.is_ocr_worthy` **skips OCR** but keeps the block | a 5×3 junk box is not a text block at all, while a 1×200 strip *is* a box the detector found and the user should see — it just cannot be read. Both read the same number, so one config value still moves both |
| D41 | Reading order is computed **before** cropping/OCR, so `id` == `reading_order` == document order. Rows are clustered when boxes overlap vertically by more than half of the *smaller* box; `manga_rtl` = rows top→bottom, right→left inside a row; `top_to_bottom` = left→right; `score` = confidence with y/x tie-break; an unknown mode falls back to `manga_rtl` | ordering first means the OCR batch follows the page (useful for the user and for chunk progress) and the palette is stable per block. Falling back instead of raising keeps a hand-edited config from failing a run |
| D42 | Page progress is translated by the pipeline (`STAGE_SPANS`) with a **high-water mark**, and the pipeline emits the `detect`/`crop`/`ocr`/`assemble` boundaries itself | the OCR engine loads lazily *inside* the first batch and reports `load_ocr` 0→100 then; a naive bar would jump backwards. And a bar that depends on engine cooperation is a bar that stalls when an engine reports nothing |
| D43 | `JobManager` is **single-flight**: a second `start()` raises `StudioError(BUSY)` (exit 4) instead of queueing; `cancel()` is a *request* (state → `cancelling`, the worker settles it); a job that finishes after a cancel request is `done`, not `cancelled` | queuing silently reorders the user's intent and hides that the GPU is busy; and a finished page is a finished page — reporting it as cancelled would throw away real work in the GUI and in P7's cache |
| D44 | Listeners are notified with the state **captured under the lock**, **outside** the lock, and **before** the worker thread starts | re-entrancy (a GUI slot calling `status()`/`cancel()` from the callback) must not deadlock; events must not be reordered or skipped because the worker raced ahead; and a cancel issued from inside a callback must land before inference starts |
| D45 | The GUI owns **no second thread**: `gui/workers.JobBridge` is a `QObject` that subscribes to `JobManager` and re-emits its callbacks as Qt signals | PLAN said "QThread adapter over JobManager", but the manager already runs the one worker D14 allows; a QThread wrapper would either duplicate it or sit idle. Signals crossing threads are queued by Qt, so the GUI thread stays responsive and the MCP server keeps the same single state machine |
| D46 | Button/progress state is derived **only** from job events (`_set_busy` from progress/finished/failed/cancelled), never from statements around a `start()` call | `start()` announces "queue" before the worker runs (D44), so any `self._busy = True` placed after `start()` would race with the first event; deriving from events makes the order irrelevant and the widgets provably consistent |
| D47 | New CLI action **`--gui-smoke`**: build the window, lay it out, print one summary line, exit 0 | the packaging/CI question "does the GUI come up with these engines and this config?" has to be answerable without a display or a human; also the only sane way to test the composition root |
| D48 | `--verbose` writes to **stderr only** | stdout is the result document for `--run` and the JSON of the other actions; a human-readable dump there would break `--run > result.json` |
| D49 | The model combo is filled from the **on-disk bundle** at startup (no network, D27); offline, variants that are not bundled are *disabled with the fix in their tooltip* rather than hidden | a hidden variant makes the combo change shape between machines and confuses comparison; a disabled one says exactly what to run (`scripts/fetch_models.py --variants …`). A live Hub listing can still be pushed in later via `set_models()` |
| D50 | **The drawing recipe lives in `core/annotation.py` and imports no Qt.** Both painters call it: `render_overlay()` bakes full-resolution PNGs (Save annotated PNG, `detect_demo`, and P7's image tool headless), `gui/canvas.py` paints the same recipe interactively. Colour, chip text, chip placement, draw order and hover picking are decided *there*, once | two renderers that each decide "where does the label go" drift, and the drift shows up as "the file I saved is not what I saw". Keeping the module Qt-free is also what lets the MCP server render an annotated page without a window or a display |
| D51 | The canvas draws **geometry in scene coordinates, sizes in device pixels**: pen 1…4 px, chip font 8…20 px, both scaled by the current zoom and clamped | at fit zoom of a tall scan an image-pixel pen disappears and an image-pixel chip is wider than its box; at 400 % it would be a slab. The export stays at the natural (scale 1.0) sizes, so a screenshot and a saved PNG still *mean* the same thing |
| D52 | `gui.*` view settings are owned by the **control panel** (the same `apply_config`/`collect_into` pair as everything else); the canvas exposes setters and reports zoom back through `zoom_changed` → `set_view_zoom()` | one owner per value, so Export/Import and (later) MCP `get/set_config` cover the view state for free; and a canvas-only toggle would have made "what did the user actually see?" unanswerable from the config alone |
| D53 | A dialog-opening slot is **never connected to `clicked` directly**: it goes through a no-arg `_*_pressed` method | `clicked` carries a `checked` bool and PySide6 passes it to an optional first parameter, so `connect(self.export_config)` handed `False` to `path` and silently wrote a file named `False`. Only a real click shows this — every test passed a path, so the bug survived a whole part; the regression tests click |
| D54 | **An optional local HF token file is a first-class token source**: `--hf-token` > `$HF_TOKEN` > `HF_token.txt` / `.hf_token` at the project root (or `$ANIMETEXT_TOKEN_FILE` pointing anywhere), via `model_store.discover_hf_token()` | the gated repo needs a token for Hub fallback, live listing and `fetch_models.py`, and the app is launched from shells, the GUI, MCP and subprocesses — requiring an `export` in each of those is exactly how tokens end up baked into configs, shortcuts and shell history. Discovery is read-only, never logged, never exported (`redact_args` scrubs `hf_token`/`token`; `export_config` only ever writes `mcp.auth_token`), and every accepted file name is in `.gitignore` |
| D55 | **The bundle now carries all five detection variants (1.26 GB) and also records the PyTorch originals it did not fetch**; `OCR_REVISION` re-pinned `f5f0f617…` → **`b00317ed…`** (the merged, edge-case fine-tuned weights) | the user's rule is that *a folder copy is the backup*, so partial variant coverage is a defect: a machine that ran fine yesterday could not load `l`/`x` today. `model.pt` is never read at runtime, but recording it makes `--verify` notice a truncated copy and makes the manifest describe the folder as it actually is. The OCR snapshot's Python/tokenizer files are **byte-identical** across the two revisions (checked by hash, not by reading the card), so the Part 3 API is untouched |
| D56 | **The engines badge is derived, never remembered**: `MainWindow.refresh_engine_badge()` asks both engines for `status()` and renders it through `engines_badge_text()`, which lives in `engines/base.py` and imports no Qt | `unload_engines` used to leave no visible trace at all, and a cached string would be wrong the moment an MCP call touched the engines; asking is cheap (a lock plus a dataclass) and cannot drift. The window also has to survive an engine whose `status()` raises — a badge must never be able to break a job |
| D57 | **`--redact-secrets` is an explicit CLI flag, not the default**, mirroring the MCP `export_config(redact_secrets=)`, and the CLI prints "(secrets redacted)" when it writes | a config export is the app's own backup and must round-trip a working token by default; the shareable form exists for bug reports. The two files look identical, so the command output has to say which one was just made |
| D58 | **Result schema 1.2: `blocks[].flags` + `blocks[].contained_in`, emitted always, acted on never** | both are derivable from data the pipeline already holds (text length vs `ocr.max_new_tokens`; box geometry), so they cost nothing and cannot disagree with the rest of the document. The alternative — letting the app *fix* what they detect (auto-raise the cap, drop nested boxes) — would change a whole page's cost or content on the caller's behalf, which is the caller's decision (D60). `flags` is a list rather than a boolean so a second observation can be added without another schema bump; `contained_in` names the *tightest strictly larger* container so a box inside two boxes gets the inner one, and ties break by id, so the map is reproducible byte for byte. Both keys are required in 1.2 and optional on input: 1.0/1.1 documents read as unflagged, and the frozen `result-1.1.schema.json` keeps validating what 1.1 wrote |
| D59 | **`ocr.max_patches` accepts 512** (config, JSON schemas, GUI combo, engine `MAX_PATCH_CHOICES`) | the vendored processor implements tiers {128, 192, 256, 384, **512**} and its own escalation targets 512 (`processing_hayai._select_best_max_patches` / `check_escalation`), while our validation stopped at 384 — the top tier was unreachable through the app even though the model supported it. `max_scale=4.0` still decides per crop whether a larger budget is usable, so asking for 512 on a small crop stays a legal no-op |
| D60 | **Language/script policy and sub-crop splitting are out of scope for this app** (user decision, 2026-11-22) | deciding which languages *should* appear on a page, and what a "block" means (page? slab? line?), are the caller's questions; putting either in the app would bake in an opinion the caller then has to override with a config flag. Hayai also has **no language argument** anywhere in `ocr()`/`generate()`/the processor, so "retry with better language parameters" cannot be implemented honestly — only geometry and budget are levers here. What the app owes the caller instead is enough truth to decide: per-block provenance, geometry, and now the 1.2 flags. Evidence and the rejected-by-design analysis live in `ADDENDUM.md` §1, which is now written as *consumer* documentation |
| D61 | **Warm means "loaded *plus* one inference", and warm-up is a job** (`JobManager.start_warmup`, `kind: "warmup"`, stage `warmup`); `EngineStatus.warm` / `.inferences` report it, `--preload` and the new `warm_engines` tool both go through it | Loading is only half the cost: the first call on a cold session also pays the execution context, kernel selection and allocator arenas. Reporting `loaded` alone would let `--preload` claim success while the user's first page stayed slow — `warm` is the field that can be checked. As a job it inherits single-flight (`BUSY`, not a race), progress, cancellation and the GUI badge refresh for free, and because it makes no page it produces **no result document**: `has_result` and `result://latest` keep pointing at the last real run |
| D62 | **`unload_engines` guarantees released objects and reusable memory, not memory returned to the OS** — and the test says so | The first version of the real-model test asserted "most of it comes back" and failed. Measured (onnxruntime 1.32, `yolo12m_animetext`, CPU): load + one inference = +975 MB RSS; unload + gc + `malloc_trim(0)` gave back 244 MB, while `live_sessions()` went to **zero** and a second load reused the space instead of stacking. The remainder is ONNX Runtime's process-wide state with no Python API to free it. So the honest contract is: sessions gone, `warm`/`loaded` cleared, memory reusable, process exit the only full cure — pinned by `test_unload_releases_the_session_even_though_rss_has_a_floor` rather than by prose |
| D63 | **Compare `Enum` members by `.value`, never by `str(member)`** — and status fields nobody queries are fields nobody tests | `EngineState` is a `(str, Enum)` mixin, so `str(EngineState.READY)` is `"EngineState.READY"`. `StudioService._engine_info` compared that to `"ready"`, which means `get_status` had answered `engines[].loaded: false` for every engine since P7 — warm, cold, or mid-run — and no test caught it because nothing asserted on the field. Same bug class bit the new warm-up code on the way in. `state` is now published alongside `loaded` so a caller can see what the engine actually said |
| D64 | **Every door validates its own inputs; `pathlib` and `int()` never get to answer a caller** | `imaging._as_path` (a `None`/`bytes`/empty path is `INVALID_ARGUMENT`, not a `TypeError` three frames away), `service._as_block_ids` (a list of ints, no silent `int("n")` or truncating `1.7`), `config._read_text_file` (a UTF-16 file is `INVALID_CONFIG` naming the byte offset, not a bare `UnicodeDecodeError`; a UTF-8 BOM is forgiven). Found by throwing the matrix at the app, not by reading the code |
| D65 | **The pipeline enforces the engine contract at the boundary** | `_as_detections` / `_as_texts` / `_place` in `core/pipeline.py`: an engine returning `None`, dicts, a wrong-length list or a non-string answers `INTERNAL` naming the engine and what it got, marked `details.contract` so the per-crop retry path skips it (retrying a broken engine N times learns nothing). A *missing* or `None` per-crop answer becomes a per-block `error` instead of a silent `text: null` — the invariant D39 promised and the old `zip()` quietly broke |
| D66 | **base64 is whitespace-tolerant, and the size cap is a service rule measured in decoded bytes** | clients that wrap at 76 columns are normal, so `image_from_base64` strips ASCII whitespace before `validate=True` decoding; `MAX_BASE64_BYTES` (64 MiB decoded) moved to `mcp/service.py` with `MAX_BASE64_CHARS` derived from it — the old `* 2` check admitted ~96 MB while promising 64, and raised outside the dispatch path so it returned no §2.3 shape and no log row |
| D67 | **Out-of-range numbers are refused, not clamped** | `annotated(scale=…)` used to clamp 100 → 8.0 and happily allocate 8× a page; it now rejects outside 0.05–8.0 (the range the tool schema publishes) and bounds the *rendered area* by `MAX_IMAGE_PIXELS`. One rule for the service and the wire, matching `set_zoom`, which has always refused |
| D68 | **Every failed tool call carries `{code,message,details}` — including the ones the SDK produces for us** | `mcp/tools.ErrorShapeMiddleware`: an `isError` result with no `structuredContent` (failed argument validation, unknown tool) is rewritten into the §2.3 shape and recorded as one activity entry. The SDK's hook is marked provisional, so installation and rewriting are best-effort: the server must never fail because its error cosmetics did |
| D69 | **A failed resource read carries the same document in JSON-RPC `error.data`** | `mcp/resources.report_resource_errors` raises `MCPError(-32602 caller-fixable / -32603 otherwise, our message, exc.to_dict())`; the SDK passes an `MCPError` through unwrapped, so `NO_IMAGE` finally reaches the client instead of dying in the server log as `"Error reading resource result://latest"` |
| D70 | **The error matrix is data, and the documentation is checked against it** | `MATRIX` (84 rows × entry point/fault/code) in `tests/test_p8_errors.py` is the source of truth; `scripts/gen_error_matrix.py` renders `docs/ERROR-MATRIX.md` from it (`--check` for CI), and three tests keep the prose honest: every row documented, every code and every tool named, and both documents' exit columns compared against `EXIT_CODES` (the first run of that check caught six wrong numbers in my own draft) |
| D71 | **A memory claim belongs where the measurement is unambiguous** | the P8.3 unload test failed intermittently in a full run (glibc raises its mmap threshold as it sees big frees, so whether a freed slab leaves RSS depends on what ran before it). In-process it now claims only that load/unload cycles do not ratchet (+32 MB tolerance where a leak would show +128 MB); the strict "unload returns most of the slab" claim runs in a subprocess, where the slab is the dominant allocation (measured: took 96 MB, 23 MB resident afterwards) |
| D72 | **A wait is never dispatched to the app thread** | `mcp/tools.py` routed `wait_for_job` through `acall()`. Harmless only because the progress pump leaves its loop after the job settles, so the join returns at once — but it contradicted the method's own docstring, and a dispatched *wait* is the one call that must never happen: with a window attached it occupies the GUI thread for the whole job (frozen window, dead progress bar, unresponsive cancel) and then answers `INTERNAL: the application did not answer within 20 s` for a job that is merely slow and goes on to finish. Mock jobs take milliseconds, which is why the MCP suite never saw it since P7. The wait now runs on the caller's thread (`JobManager.wait` is thread-safe by design); pinned by a test that fails when the dispatch is restored (P8.5). |
| D73 | **The font that measures a label is the font that draws it** | `canvas.paint_overlay` built a `QFont` + `QFontMetrics` to size every chip and never called `painter.setFont(font)`, so `drawText` used the widget default: every label was wider than the chip it had been measured for and Qt clipped both ends — `#0·0.80·この攻撃で決まりだ！` rendered as `.80 この攻撃で決ま`, in the window and in every saved PNG, since P6 and visible in the Turn 16 screenshot. No test caught it because none compared drawn text with measured text. Found by looking at the 8.6 before/after pictures. Pinned by a pixel probe that walks up a chip's padding and refuses any glyph outside it — verified to fail when the line is removed. The general shape: **when one component measures and another paints, the measurement is a promise, and a promise nothing checks is a bug waiting for a font change** |
| D77 | **The application code is Apache-2.0; the weights keep their own licence** | the one open question 8.7 left for the distributor was answered on 2026-11-24: **Apache-2.0** for the code. `LICENSE` is the canonical text (pinned by the sha256 of everything before the appendix, so the copyright line can be personalised without the *terms* drifting), `NOTICE` inventories what is not ours, and `pyproject` declares it the PEP 639 way (`license = "Apache-2.0"` + `license-files`, no `License ::` classifier — setuptools 77+ deprecated those, and two spellings of one fact is two chances to disagree). Bundling changes nothing about the weights: the AnimeText ONNX variants stay **GPL-3.0** (their licence text now ships beside them at `models/animetext-yolo/LICENSE.GPL-3.0.txt` — redistributing GPL material without its text is the one thing nobody should have to chase), Hayai stays Apache-2.0, and README §Licensing states the boundary plus the slim-wheel escape for a distributor who cannot carry the GPL files. Two findings from writing it: **`dghs-imgutils` is MIT, not Apache-2.0** (my memory, caught by `importlib.metadata`, so the test now checks every NOTICE claim against what the installed distribution itself reports), and `license-files` is not optional — without it a wheel carries no licence text at all. |
| D78 | **Report the restart; do not fake one** (found by the 8.8 QA pass) | PLAN §2.2 promised that changing `mcp.host`/`port` at runtime "restarts the server thread". Nothing ever did — `StudioServer` is built once from `config.mcp` and the gateway captures the bind address and bearer token at that moment — and neither the docs nor the tool output said so, so a client that patched `mcp.port` had a stored value, an unchanged socket and no signal. Restarting a listening socket from inside a config patch is the part worth *not* building (a client that can move the port it is connected to can orphan itself mid-session, and it is the same "report, don't act" line as D58/D60). What shipped instead: `core/config.py:RESTART_REQUIRED_KEYS` (`mcp.enabled`, `transport`, `host`, `port`, `auth_token`), `restart_required()` in the `set_config`/`import_config` replies (schemas + regenerated catalog), the window says it out loud when an imported file moves one, and README/docs/MCP.md name the five keys. The value is still stored — an export is faithful and the next start uses it — but nobody is left believing the socket moved. |
| D79 | **A QA checklist is a table with rules, not prose** | "Final QA checklist against every original requirement" (PLAN §3 Part 8) becomes `REQUIREMENTS` in `tests/test_p8_qa.py`: 115 rows, each with a source (`PLAN §…`), a status and **evidence that must exist** — a test node id, a command whose flags are validated against the real parser, or a check in that same file. Four rules keep it honest: no row without evidence; no evidence that does not resolve (a cited test is looked up in the suite's ASTs, so a renamed test fails the suite instead of quietly turning the checklist into fiction); nothing may be marked unmet (what is not done lives in `FOLLOW_UPS`, labelled as *not a requirement*, with the reason); and `docs/QA.md` is generated from the table with `--check` like `docs/ERROR-MATRIX.md`. Also floors, not targets: the sign-off counts fail if the suite shrinks. Two rot cases this immediately caught in our own docs: `docs/MCP.md` still advertised the config document as **v1.1** (it is 1.2), and the README's development block quoted a stale test count. |

## Turn log

### Turn 2 — 2026-11-19 — Part 1 (Foundation) — ✅ COMPLETE
- Read new attachment (Space page HTML): confirmed cpu-basic hosting, HF_TOKEN secret,
  user GPU hardware (see Environment notes). Session tokens in the HTML were ignored.
- Created:
  - `pyproject.toml` (extras gui/detect/ocr/mcp/dev/all; console script; pytest markers)
  - `animetext_studio/core/types.py` — BBox/Detection/TextBlock/ImageInfo/DetectionInfo/
    OcrInfo/TimingMs/PipelineResult/JobProgress/JobState, StudioError + 14 error codes,
    exit-code map, deterministic 20-color palette, sha256 helper, pinned OCR revision
  - `animetext_studio/core/config.py` — 7-section AppConfig, model presets (F1 thresholds),
    native validation (errors/warnings with dotted paths), deep-merge `merged()` returning
    changed keys, file import/export, `validate_config_file` report, CLI layering
  - `animetext_studio/core/cli.py` + `__main__.py` — full CLI surface; foundation actions
    implemented; GUI/headless paths stubbed with NOT_IMPLEMENTED (exit 20)
  - `animetext_studio/schemas/{config,result}.schema.json` (draft 2020-12, in-package)
  - `tests/` — 110 tests (types/config/CLI/schema cross-checks), `scripts/setup_dev.sh`
  - `examples/config.default.json`, `examples/config.custom.json` (x-model + cuda:0 +
    MCP 0.0.0.0:9000 + token placeholder — tailored to the user's 3-GPU machine)
- Verification:
  - `.venv/bin/python -m pytest -q` → **110 passed** (0.22 s)
  - `--dump-default-config` → valid schema-1.0 JSON, demo defaults (m / 0.35 / 0.05)
  - `--export-config` with CLI overrides → resolved file; `--validate-config` → valid
  - `--headless --run` → exit 20 + structured NOT_IMPLEMENTED error (D8)
  - `pip install -e .` → OK; `animetext-studio --version` → 0.1.0
  - Native validator ↔ JSON Schema agreement tested on 17 invalid + 8 valid mutations
- PLAN.md updated: schemas in-package (D6), CLI additions (D9), Blackwell torch note.

### Turn 3 — 2026-11-19 — Part 2 (Detection engine) — ✅ COMPLETE (real inference awaiting the token)
- **P1 re-verified** on the new interpreter: `pytest` → 110 passed; `--dump-default-config`,
  `--validate-config`, `--export-config` and the `--headless --run` stub (exit 20) all match
  the Turn 2 log.
- P2's two engine modules were already on disk but had **no tests and no demo**; finished
  them and fixed three real defects found by re-checking `research/imgutils_yolo.py`
  *against the installed imgutils 0.19.0*:
  1. **`--hf-cache-dir` was silently ignored** — the env var was set after
     `huggingface_hub` had been imported → `configure_hf_cache_dir()` (D13).
  2. **`detect()` re-ran the repo model-list glob on every call** → `refresh_models()`
     caches the list; `unload()` invalidates it.
  3. Backend `ImportError`s were reported without a fix hint → `_classify()` now attaches
     the install hint (imgutils lazily imports onnxruntime/gradio itself).
- Additions to the engine: `--provider`/`ONNX_MODE` support (D17), `recommended_threshold()`
  + `resolved_score_threshold()` (D18), import fallback (D16), and module docs recording the
  verified upstream behaviours (`_OFFLINE` sentinel, per-thread session cache, lazy metadata).
- Created:
  - `animetext_studio/tools/__init__.py` + `tools/detect_demo.py` — stage-1 demo:
    `--mock`, `--score recommended`, `--allow-dynamic`, `--provider`, `--list-models`,
    `--annotate out.png` (colour-coded boxes + `#id score` chips), `--out`, `--limit`;
    JSON on stdout, structured error + StudioError exit code on stderr.
  - `tests/test_engines_base.py` (23 tests) — status/progress/cancel/NMS helpers and the
    fake engine's contract: determinism, size scaling, score & IOU thresholds (the fake's
    overlapping pair has IoU **0.6736** → 5 boxes at 0.35, 6 at 0.9), catalog/palette, unload.
  - `tests/test_detection.py` (43 tests) — the real engine behind mocked seams: parameter
    pass-through (`conf`/`iou`/`allow_dynamic`; `score=None` stays `None`), box clamping /
    invalid-box dropping / score sorting, model-list ordering + both fallback paths (auth
    error, non-list `_OFFLINE` sentinel), caching, download sequence, tqdm progress,
    idempotent load, 9-way error classification, token forwarding, cancellation,
    concurrent-detect serialisation, unload→refetch, HF-cache/provider helpers.
  - `tests/test_models.py` (6 tests, `@pytest.mark.models`) — repo list vs presets,
    download+load+detect, ≥ 3 blocks on the sample page, threshold monotonicity,
    `allow_dynamic`, unknown-model rejection; auto-skips without `HF_TOKEN`.
  - `scripts/check_hf_access.py` — token/access preflight: deps → `whoami` → repo visibility
    → **tiny `threshold.json` probe** (proves download rights, not just listing) → optional
    `--full` weights and `--detect IMAGE` end-to-end → public OCR repo check. Masked token
    fingerprint only, `--json` mode, exit codes from the shared taxonomy.
  - `scripts/make_sample_image.py` + `examples/sample_page.png` (D20).
  - `scripts/setup_dev.sh` rewritten (rich interpreter vs `--venv`; `--detect/--gui/--ocr`).
  - `pyproject.toml`: `addopts = -m 'not models'` (D19).
- Verification:
  - `python3 -m pytest -q` → **171 passed, 6 deselected** (Part 1 baseline was 110).
  - `python3 -m pytest -m models -rs` → 6 skipped, each with the exact token instructions.
  - Mock demo: 5 boxes at IOU 0.35 / 6 at IOU 0.9; JSON + annotated PNG reviewed visually
    (`out/mock_boxes.png`).
  - **Live gated repo without a token:** `detect_demo … --model yolo12n_animetext` →
    `HF_AUTH_REQUIRED`, exit 6, 3 actionable steps; `--list-models` → the 5 real variants
    (listing works anonymously); `check_hf_access.py` → “not ready”, exit 6.
  - Error paths: missing file → `IO_ERROR`/10, unknown model → `MODEL_NOT_FOUND`/12.
  - All 7 pages in `sample_scans/` load through the stage-1 path (EXIF-aware RGB, sha256,
    deterministic fake detection), exit 0 — inputs treated as opaque, content not inspected.
  - `python -m compileall` clean over package, tests and scripts.

### Turn 4 - 2026-11-19 - Real-model verification (token) + Part 1 inspection - COMPLETE

**A. Part 2 closed against the real gated repo** (token from the user; env-only, never
written to any file - verified nothing in the workspace contains it):
- `scripts/check_hf_access.py` -> all PASS, exit 0 (account resolved via whoami).
- Real ONNX sizes: n 10.5 / s 37.2 / m 80.7 / l 105.8 / x 236.7 MB (only n/s/m fit the
  ~100 MB sandbox limit).
- **Defect found and fixed: `TEXT_LABEL` was `text_bblock`; every model emits
  `text_block`** (verified in all five `labels.json`). PLAN.md corrected (3 places).
- All five `threshold.json` + `f1_score` values match `MODEL_PRESETS` exactly - now
  locked by a real-model test, so the P1 preset table can no longer rot silently.
- yolo12n and yolo12m infer correctly on CPU: 40-300 ms/page; annotated review of the
  synthetic page shows the 4 bubbles, 3 tategaki columns and the narration band all
  found. Quantitative pass over 3 of the user's scans (content not inspected, no
  renders): e.g. yolo12m on 003.jpg -> 46 boxes at score 0.05 vs 25 at 0.299.
- Measured D14 for real: first inference on a brand-new thread costs +70-100 ms to
  re-create the ORT session -> P4 must reuse one worker thread.
- Provider path verified on a CPU-only onnxruntime: `--provider cuda` neither crashes
  nor lies (ORT warns, falls back, `status().device` says `cpu` - D24).
- `pytest -m models` -> **7/7 green** (added the metadata-vs-presets test).

**B. Part 1 inspection** (code authored elsewhere):
- Differential audit of the native validator vs the published JSON Schema found **3
  real contract divergences**: integral floats for integer fields (native rejected,
  schema accepted), unknown keys and missing sections (native accepted per D10, schema
  rejected). Fixed by D21 + D22, then locked with a parametrised agreement table plus a
  seeded randomised walk over all 37 config fields.
- **Fixed:** every `from_dict` raised bare `KeyError` on malformed payloads (violates
  the {code,message} contract) -> now `INVALID_ARGUMENT` with the available keys (D23).
- **Fixed:** `status().device` reported the requested provider, not the real one (D24).
- **Improved:** `--verbose` surfaces config-file warnings instead of silently defaulting
  (D25). PLAN.md updated: label + ONNX sizes, 14-code error taxonomy.
- Claims that held up under checking: D11 (`--hf-token` never appears in an exported
  config), demo defaults, byte-stable round-trip, CLI>file>defaults precedence, exit
  codes, both example configs validate clean.

**C. Totals:** `pytest` -> **232 passed, 7 deselected**; `pytest -m models` -> **7
passed** with a token; `compileall` clean.

### Turn 5 - 2026-11-20 - On-premises model bundle (D26) + Part 2 hardening - COMPLETE

*Resumed after an external interruption mid-way through the model-store work (the
network dropped, then an accidental keystroke stopped the run). Everything below was
re-derived from the workspace, verified, finished and tested.*

**A. Weights are now part of the app (D26)** — the user's requirement: a copy of the
folder is a complete, offline, backup-able installation.
- `core/model_store.py` (new, pure stdlib): root resolution (explicit > env > package),
  variant completeness, OCR snapshot completeness, `require_*` errors that name
  `scripts/fetch_models.py`, structural `verify_safetensors()` (header + tensor ranges —
  catches truncated copies and unresolved LFS pointers without loading the file),
  `status()` report, `MANIFEST.json` read/write.
- `scripts/fetch_models.py` (new): vendors either repo, idempotent (size/sha aware),
  `--verify`, `--dry-run`, `--force`, `--with-sha256`, `--ocr`, `--all`, `--variants`.
- `engines/detection.py`: `BundledYOLOModel` (built at call time over imgutils'
  `YOLOModel`) resolves `model.onnx` / `threshold.json` / `model_type.json` locally first
  and delegates to the Hub otherwise; `load()` skips the download entirely for bundled
  weights; `--offline` never even constructs the backend.
- CLI: `--models-dir`, `--offline`, `--model-status` (+ `paths.models_dir`,
  `paths.offline` in config and schema); `detect_demo` gained the same flags.
- **Bundle on premises** (`animetext_studio/models/`, 773.3 MB, `MANIFEST.json` with
  sha256 for all 23 files, revisions `a180c191bfdb` / `f5f0f617f514`):
  yolo12n 10.0 MB + yolo12s 35.5 MB + yolo12m 77.0 MB (each with `threshold.json`,
  `labels.json`, `model_type.json`) and the complete Hayai OCR snapshot (650.9 MB,
  395 tensors, F32, structurally verified). `l` (105.8 MB) and `x` (236.7 MB) are
  deliberately **not** vendored: over this sandbox's ~100 MB per-file limit — fetch them
  with `python scripts/fetch_models.py --variants yolo12l_animetext` on a normal machine.
- `.gitignore` added; the bundle is intentionally **not** ignored (with git-lfs notes).

**B. Bugs found while doing it** (all reproduced first, then fixed and covered by tests):
1. **Offline detection hung forever.** `model_names` -> imgutils -> fsspec glob ignores
   `HF_HUB_OFFLINE` and blocked on a live HTTPS call (90 s faulthandler trace on the
   first `detect()`). Fixed by D27: offline never asks; online is bounded to 10 s and
   degrades to bundle + presets, reported in `status().detail` and the new
   `model_list_is_fallback`.
2. `BundledYOLOModel` assumed imgutils' private fields (`_global_lock`, `_models`, …)
   existed -> `AttributeError` swallowed into a silent "model list fallback". Now read
   defensively, and the fake base in the tests deliberately lacks them.
3. `write_manifest()` overwrote provenance on a partial run and recorded **phantom
   0-byte entries** for failed downloads (6 of them after a token-less `--all-both`).
   Fixed by D28; `--verify` now fails on `RECORDED BUT ABSENT`.
4. `OCR_REVISION` was stale (D30) — would have pulled a second, older OCR snapshot on an
   otherwise-offline machine.
5. `detect_demo --list-models` read `engine.status()` *before* `list_models()`, so the
   JSON always showed an empty `detail` (dict literals evaluate in order).
6. `hf_hub_download(local_dir=…)` left a `.cache/` tree inside the vendored snapshot —
   now cleaned by the fetch script.

**C. Tests:** `tests/test_model_store.py` (new, 46 tests: precedence, completeness,
truncated/phantom files, safetensors integrity incl. LFS pointers, manifest merge,
engine integration incl. a *wedged-listing* test proving the 10 s bound, and an
offline-must-not-construct-the-backend test); `tests/test_models.py` gained
`TestBundledDetection` — **real inference with no token and no network**; conftest
isolates the real bundle from unit tests (D29).

**D. Verification (all re-run at the end of the turn):**
- `pytest` -> **278 passed, 13 deselected** · `pytest -m models` -> **13 passed** with a
  token, **6 passed / 7 skipped** without one · `compileall` clean.
- `python scripts/fetch_models.py --verify` -> *RESULT: usable offline* (23 files, hashes
  verified, safetensors sound).
- `env -u HF_TOKEN … detect_demo examples/sample_page.png --offline --model yolo12m_animetext`
  -> exit 0, 12 boxes, `detect: 2133 ms`, `recommended_score_threshold: 0.299` (from the
  bundled file), annotated PNG visually confirmed: all 4 bubbles, 3 tategaki columns,
  title and narration band boxed.
- `--list-models --offline` -> 0.05 s, `source` = bundled for n/s/m, `hf` for l/x,
  `detail: "model list: offline (bundled + presets)"`.
- `--offline --model yolo12l_animetext` -> exit **12** `MODEL_NOT_FOUND` with
  `1. python scripts/fetch_models.py --variants yolo12l_animetext` as step 1.

**E. Note on token handling:** the user's HF token was used only in process environments
(`HF_TOKEN=… python3 …`) and is not written into any file; `grep` over the workspace
finds it nowhere except this session's commands. The bundle means it is no longer needed
for normal operation.

### Turn 6 - 2026-11-20 - Part 3 (OCR engine + imaging) - COMPLETE

**A. New code**
- `core/imaging.py` — EXIF-aware load from path/bytes/base64 with structured errors
  (`IO_ERROR` / `INVALID_IMAGE` / a pixel-count guard instead of Pillow's bare raise),
  `ImageInfo` (sha256 of the **original** bytes, so a re-encoded thumbnail can't fake an
  identity), `crop` / `crop_pair` / `crop_of` (padded, clamped, min-side upscale; an
  *empty* box is an argument error — a 0-area crop OCR'd into a hallucinated string is the
  failure mode to forbid), `is_ocr_worthy`, `normalize_text` (NFKC + per-line tidy that
  **preserves line breaks**; `collapse_lines` is opt-in), `pil_to_qimage` / `qimage_to_pil`
  (duck-typed, so Qt is not needed to test it), PNG + thumbnail helpers, `color_for_index`.
- `engines/base.py` — `BaseOcrEngine` (a crop-based contract on purpose: Hayai reads crops,
  not pages, so batching / progress / cancellation belong to the engine) + `FakeOcrEngine`
  (pixel-hash-deterministic text from a JA/ZH/KO/EN bank; **blank crop → `""`**, so P4's
  "no text" branch is reachable in mock mode; records `calls` so pipeline tests can assert
  what was asked for).
- `engines/ocr.py` — `OcrEngine`: bundled-first snapshot resolution (D26/D36), device +
  dtype validation (D35), `HAYAI_DISABLE_COMPILE` before the remote import (D32), own
  chunking loop (D33), misalignment guard (D34), error mapping (OOM → `OCR_FAILED` naming
  the two knobs; `LocalEntryNotFoundError` → `MODEL_NOT_FOUND` naming `fetch_models.py`;
  `OfflineModeIsEnabled`/`ConnectionError`/`Timeout` → `DOWNLOAD_FAILED`),
  `torch_dtype`→`dtype` fallback so a transformers 4.x install still loads, `unload()` with
  `gc` + `empty_cache`/`ipc_collect` (freeing memory must never raise).
- `tools/ocr_demo.py` — crops from `--bbox` args (or whole images), `--offline`, `--mock`,
  `--device/--dtype/--cuda-graphs/--max-patches/--batch-size/--max-new-tokens/
  --normalize-nfkc`, result document to stdout or `--out`.

**B. Facts read out of the vendored code, not the model card** (the card had drifted, so
the engine was written against `processing_hayai.py` at `f5f0f617f514…`):
`model.ocr(images, processor, max_patches, max_new_tokens, use_cuda_graph, dtype,
batch_size, escalate) -> str | list[str]`; `batch_size` only bites for `max_patches="auto"`
**and** only when `len(images) > batch_size`; CUDA graphs are gated on
`device.type == "cuda"` inside `generate()`; `HAYAI_DISABLE_COMPILE` is read at import
time; `generate()` already falls back fast → fp32 → reference and **re-raises** OOM, so the
engine maps it and never retries it.

**C. Defects the new tests caught (fixed in the same turn)**
1. `qimage_to_pil` called `convertToFormat()` blindly, and its "not a QImage" message was
   hardcoded so it never named the actually-missing methods → **D31**.
2. `OcrEngine(device="cuda-x")` was accepted and silently became `cuda:0` in `status()`
   → **D35** (`_CUDA_DEVICE_RE`).

**D. Verification (all re-run at the end of the turn)**
- `pytest -q` → **386 passed, 14 deselected** (`tests/test_ocr.py` = 76 of them; the suite
  was 310 when this turn started) · `compileall` clean.
- `pytest -m models` → **7 passed, 7 skipped** with no token in the environment (the skips
  are the token-gated Hub listing tests). New among the passes:
  `test_real_bundled_weights_recognise_a_sample_crop` — **real Hayai inference on the
  vendored 666 MB snapshot, `offline=True`, CPU, 7.8 s**.
- `python -m animetext_studio.tools.ocr_demo examples/sample_page.png --bbox … --offline
  --batch-size 8 --out out/ocr_real.json` → 8 crops → 8 recognitions,
  `timing_ms.ocr = 6726`, `engine.detail = "auto, cuda-graphs off, bundle"`,
  `weights_source` = the bundle directory. Read: 何!?その力は ·
  決戦の前夜風が止んだ第4話その日、空は赤かった · 絶対に負けない · まだだよ、私は ·
  この攻撃で決まりだ.
- `--mock` is deterministic (the same crop yields 「風が止んだ」 every run).
- Nothing here needed the HF token or a socket: the OCR repo is public **and** vendored,
  and offline is asserted *behaviourally* — the transformers seam is never called when the
  bundle is missing and offline is on.

**E. The 667 MB question, closed.** Turns 4–5 assumed the OCR weights could not be fetched
here, so P3 had been scoped down to "mock + a smoke test on the user's machine". The user
copied the snapshot into `animetext_studio/models/hayai-ocr-v2.7-supernova/` instead, so
real OCR is now part of this sandbox's normal loop (opt-in `-m models`) and the mock-only
fallback is unnecessary. **Do not re-download it.**

### Turn 7 - 2026-11-20 - Part 4 (pipeline + JobManager + headless run) - COMPLETE

**A. New code**
- `core/pipeline.py` — `Pipeline.run/run_file/run_bytes` (the exact 3-step flow),
  `order_detections` (D41), `engines_from_config` (mock or real, one place for both),
  `_Bar` progress translation (D42), `SKIP_ERROR_CODE` (D39). `run()` refuses to invent
  provenance: an in-memory image must come with the `ImageInfo` from `load_image()` /
  `image_from_bytes()`, because the document's `source` and `sha256` are its identity.
- `core/jobs.py` — `JobManager`: single-flight worker (D43/D44), `start*/cancel/wait/
  run_file/run_bytes`, listener add/remove, `status()` for the MCP job resource, the
  latest-result cache that *is* the app's current result, `unload_engines()`.
- `__main__.py` — `--run` wired to the same `JobManager` path the GUI and MCP will use:
  stdout is the document and nothing else (progress with `--verbose`, and the "written to"
  note, go to stderr), `--out` creates parent dirs, every failure maps to its exit code.

**B. Defects found while building it (fixed in the same turn)**
1. Row clustering initially compared box *centres* instead of their overlap, which merged
   a tall tategaki column with everything beside it → real vertical-range overlap against
   the smaller box (D41).
2. `cancel()` on an already-`cancelling` job would have raised `UnboundLocalError` in the
   notify path, and notification originally ran with the lock held → D44's snapshot rule.
3. Progress was only as complete as the engines' own reporting; the pipeline now emits its
   own stage boundaries (D42).

**C. Verification (all re-run at the end of the turn)**
- `pytest -q` → **457 passed, 15 deselected** (+71 this turn: `test_pipeline.py` 42,
  `test_jobs.py` 28, and the Part-1 `--run` stub test replaced by two that match the new
  dispatch) ·
  `pytest -m models` → **8 passed, 7 skipped** (no token here) · `compileall` clean.
- Acceptance command: `python -m animetext_studio --headless --mock --image
  examples/sample_page.png --run [--out …]` → exit 0, schema-validated document, identical
  across runs apart from `created_at`/`timing_ms`.
- **Real weights, offline, one call** (`out/pipeline_real.json`): yolo12m → 12 boxes →
  12 crops → 12 real strings (この攻撃で決まりだ / まだだよ、私は / その日、空は赤かった /
  絶対に負けない / 何!?その力は / 真相はいつも闇の中 …),
  `timing_ms = {detect: 2289, ocr: 4580, total: 6869}` on CPU, `ocr.device = cpu`,
  reading order correct for the tategaki columns (right→left inside each row).
- Concurrency tests (busy refusal, cancel mid-inference, cancel between OCR chunks,
  re-entrant listener, throwing listener) run against real threads and passed 5× in a row.

**D. Observation for P6/P8 (not a defect):** the detector legitimately emits *nested*
boxes — a title band that contains the individual column boxes, and a whole-page box that
contains a smaller one. NMS does not merge them (containment is low IoU), so the document
has overlapping blocks by design. P6 must render that sanely (draw order, hover picking);
P8 may add an opt-in containment dedup.

### Turn 8 - 2026-11-21 - Part 5 (GUI shell: controls, image input, worker wiring) - COMPLETE

**A. New code** (`animetext_studio/gui/`, 891 lines) — Qt lives only here; `core/` and
`engines/` still import without PySide6 installed:
- `gui/workers.py` — `JobBridge(QObject)`: subscribes to `JobManager` once and re-emits
  `progress` / `finished` / `failed` / `cancelled` (D45). The settled-state event carries the
  result or the `StudioError`, so consumers never poll the manager.
- `gui/controls.py` — `ControlPanel`: model combo (tooltips with params / GFLOPS / F1 /
  recommended threshold / where the weights are), IOU and Score slider+spin pairs, a
  **Recommended** button, the OCR group (device, max_patches, batch, max_new_tokens,
  padding, NFKC), reading order, Generate/Cancel, Export/Import, and the MOCK / OFFLINE /
  **MCP: off** badges. `apply_config()` / `collect_into(config)` are the only two ways
  between widgets and configuration.
- `gui/main_window.py` — `MainWindow`: toolbar (Browse, Generate, Cancel, Export, Import),
  control panel at the left, input preview above the output in a splitter, the whole body
  inside a `QScrollArea` (H+V as needed), status bar with message + stage + progress bar.
  Image input three ways: `load_path`, `load_bytes`, and `paste_from_clipboard` (a copied
  file URL first, then a `QImage` — converted to `Format_RGB888` **here**, per D31);
  `dragEnterEvent`/`dropEvent` accept image files. Public API for tests and P7:
  `controls`, `job_bridge`, `job_manager`, `has_image`, `current_result`, `result_ready`.
- `gui/app.py` — `build_manager(config, mock=…)` (engines → pipeline → manager, built once
  and shared) and `run_gui(args, config, smoke=…)`. `__main__.py` now dispatches `--run` to
  the pipeline and everything else to the GUI, with a structured
  `ENGINE_LOAD_FAILED` + install hint + headless alternative when PySide6 is absent.

**B. Defects the GUI tests caught (fixed in the same turn)**
1. `paste_from_clipboard()` set `_info` and repainted the preview but **not `_image`**: the
   window would have shown one picture and run another. Now routed through `_accept_image`.
2. `_set_busy` touched a non-existent `panel.browse_button` → `AttributeError` on the first
   progress event of every run.
3. `_result` was never initialised and the public properties the composition root needed
   (`controls`, `job_bridge`, `has_image`, `is_mock`, `current_result`) did not exist.
4. Test-harness trap worth remembering: blocking only `sys.modules["PySide6"]` does **not**
   make `from PySide6.QtWidgets import QApplication` fail when the submodule is already
   cached — the CLI test then started the real event loop and hung the suite. Block both
   entries (the test does, with a comment).

**C. Verification (all re-run at the end of the turn)**
- `pytest -q` → **481 passed, 15 deselected** (+24: `tests/test_gui.py` 23, and the Part-1
  GUI stub test replaced by `--gui-smoke` + a no-PySide6 structured-error test) ·
  `pytest -m models` → **8 passed, 7 skipped** · `compileall` clean.
- GUI tests run offscreen and include: window builds and lists the 5 variants; controls come
  up with the config; **generate with the mock engines produces a result** and fills the
  block table; buttons disable while a gated engine is busy; **cancel from the panel lands as
  a `cancelled` signal**; a failing engine shows `MODEL_NOT_FOUND` in the status bar instead
  of crashing; browse/drop/paste (including a **Grayscale8** `QImage`, which needs the D31
  conversion) all set the image; export→edit→import restores the widgets; garbage config is
  reported and the widgets are left alone; offline unbundled variants are unselectable.
- `QT_QPA_PLATFORM=offscreen python -m animetext_studio --mock --gui-smoke --image
  examples/sample_page.png` → `GUI smoke OK: AnimeTextStudioMainWindow 1280x860, models=5,
  image=yes, engines=mock`, exit 0.
- **Screenshot reviewed** (`out/gui_mock.png`, offscreen grab after a real mock run):
  toolbar, MOCK ENGINES badge, all control groups with the configured values, the sample
  page rendered with correct Japanese, caption `sample_page.png · 900×1300 px · sha256
  b35b8283c659`, output table with the 5 mock blocks (JA/KO text intact) and the status line
  `5 block(s) · detect 0 ms · ocr 4 ms`.

**D. Notes for Part 6**
- The output area is deliberately a summary + block table (`set_result` / `clear_result`);
  `ResultCanvas` and the JSON pane replace exactly that pair, and `result_ready` is already
  the signal to draw from.
- `MainWindow._show_preview()` sets the label to the pixmap's natural size, so the central
  `QScrollArea` already scrolls both ways at 100% — the canvas must keep that behaviour.
- The blocks carry the **effective** box (D38), which is what the overlay should draw.

### Turn 9 - 2026-11-21 - Part 6 (output visualization: annotated canvas + JSON pane) - COMPLETE

**A. New code**
- `core/annotation.py` (new, ~350 lines, **imports no Qt**) — the single drawing recipe:
  `OverlayStyle` (+ `from_config`), `label_text` / `hover_text` / `one_line`, `chip_geometry`,
  `pick_block` (nested-box winner), `draw_order`, `render_overlay`, `annotated_png`,
  `save_annotated`. Qt-free on purpose: the P7 `render_annotated_image` tool and
  `detect_demo --annotate` now share the GUI's recipe instead of each keeping a copy.
- `gui/canvas.py` (new, ~520 lines) — `ResultCanvas` = a `QGraphicsView` plus its command row
  (Fit / 100 % / Save annotated PNG / zoom read-out). The overlay is painted in
  `drawForeground` with the painter switched to **viewport pixels**: geometry stays in scene
  coordinates (so it scrolls and scales with the page) while pen and chip sizes are
  zoom-compensated and clamped (D51), which is what keeps a fit-zoomed 3000 px scan readable.
  Hover = a child `QLabel` of the viewport (word-wrapped CJK, `gui.hover_opacity`, clamped
  inside the viewport); click = select; Ctrl+wheel walks `ZOOM_STEPS` (10…800 %).
- `gui/json_pane.py` (new, ~280 lines) — read-only view whose text is **byte-identical** to
  `PipelineResult.to_json()` (Copy All puts exactly that on the clipboard), light JSON
  highlighter, a block-id → line-range index derived from the document's own indentation, and
  Copy all / Copy block text / Copy block JSON / Save As… on the strip *and* the context menu.
- `gui/controls.py` — a **View** group (`overlay_check`, `labels_check`, fill slider+spin,
  zoom combo+spin) folded into the *existing* `apply_config()` / `collect_into()` pair, so
  `gui.*` round-trips through Export/Import like every other setting; one `view_changed`
  signal out, one `set_view_zoom()` mirror in. Typing a percentage flips the mode combo, so
  the two widgets can never disagree (D52).
- `gui/main_window.py` — the output group is now `canvas | (block table over JSON pane)`;
  `set_result()` feeds all three, `clear_result()` empties them, and **loading an image
  clears the previous result** (boxes from page A could previously float over page B).
  Selection is synced canvas ⇄ table ⇄ JSON behind a re-entrancy guard.
- `tools/detect_demo.py` — its private `annotate()` and font list are gone; it delegates to
  `core.annotation.save_annotated`.

**B. Defects found and fixed** (all by offscreen pixel probes; the two geometry ones were
invisible in a screenshot *because the on-screen and exported pictures disagreed*, and the
clipboard/config ones never show in a test that passes a path)
1. **`QRectF(*xyxy)`** — Qt's constructor is `(x, y, w, h)`. Every canvas box was drawn `x1`
   wide and `y1` tall (a 148 px bubble became 704 px). The PNG export was always right, so
   the canvas silently contradicted it. Now `_qrectf()`.
2. **`QRect(*xyxy)`** — same trap again for the label chip: a 122×19 chip painted as 281×90,
   blanketing the box under it. `_qrect()` uses the corner-to-corner form (right/bottom are
   inclusive there, hence the `-1`).
3. **`clicked` delivered `False` as `path`** — `QPushButton.clicked` carries a `checked` bool,
   and PySide6 hands it to any slot with an optional first parameter. P5's toolbar
   Export/Import were connected straight to `export_config(path=None)` / `import_config(path=None)`:
   **a real click wrote a file named `False`** instead of opening a dialog. The same latent bug
   landed in the two new Save buttons. Four explicit no-arg slots now, plus tests that *click*
   the buttons with the dialog recorded.
4. Chips could be drawn past the right edge (a long label on a page-wide band box):
   `_draw_chip` called `chip_geometry` without `image_size`.
5. `QSyntaxHighlighter.highlightBlock` raised (`re.Match` has no `.length()`) — an exception
   inside a Qt override is nearly silent, so the pane would have painted unstyled at best.
6. Chips were buried: one pass per box let a later (smaller) box's border cut an earlier box's
   label. Both painters now draw fills+borders first and **all** chips in a second pass.
7. `view_changed` was emitted but never connected — a tick in the View group changed the
   config and not the canvas.

**C. Verification (all re-run at the end of the turn)**
- `pytest -q` → **561 passed, 15 deselected** (+80: `tests/test_annotation.py` 41 new,
  `tests/test_gui.py` 39 new on top of P5's 23) · `pytest -m models` → **8 passed, 7 skipped**
  · `compileall` clean · `--headless --mock --run` unchanged · `--gui-smoke` now also reports
  the canvas (`… engines=mock, canvas+json=ready, zoom=…`).
- Pixel probes as real assertions: the fill lets the page through (compared against the
  *source image's* pixel, not against "white" — the sample page is `#fcfcfa`), the border is
  the block's own colour, unticking **Show boxes** restores the page byte-for-byte, the export
  contains every block's colour, the hover panel stays inside the viewport, its opacity comes
  from `gui.hover_opacity`, and the clipboard content is byte-exact with `to_json()`.
- **Screenshots reviewed** (`out/p6_probe/`): mock run at 100 % with the hover panel, the table
  row and the JSON pane all on block #2; **real offline run** (yolo12m + Hayai, 12 blocks,
  detect 3324 ms / ocr 3042 ms, `REAL ENGINES OFFLINE` badges) at fit; the full-resolution
  annotated PNG showing the nested band box #7 with its label clamped inside the page.

**D. For P7**
- `MainWindow.set_result()` / `clear_result()` are still the only doors a result enters by,
  and `result_ready` is the signal to hook; `canvas`, `json_pane`, `controls.set_mcp_status()`
  are the seams the bridge needs. `canvas.save_annotated(path)` and
  `core.annotation.annotated_png(image, blocks, style)` are the two ways P7 can produce the
  MCP image tool's bytes — the second one works headless, without a window.

## Constraints & blockers to respect

- **Sandbox file-size limit ~100 MB (user-reported, Turn 4)** — it applies to *downloads*,
  not to files the user places by hand. Detection: n 10.5 / s 37.2 / m 80.7 MB ONNX are
  bundled; `l` 105.8 and `x` 236.7 MB are still absent (optional, see Next up). OCR: the
  666.8 MB snapshot **is** on premises (user-copied, Turn 6) and runs offline — never
  re-download it, and never let a default test depend on fetching it.
- Detection ONNX metadata: `imgsz = (640, 640)` fixed square → non-`allow_dynamic`
  inference squashes tall pages (measured on a real page: 18 boxes vs 12 with
  `allow_dynamic`). Document the trade-off; the engine and demo already expose the toggle.

## Noted, deliberately not fixed yet (backlog for P7/P8)

> Status as of T19 (P8 close): **1 and 2 shipped (8.2)**, **3/4/5 are documented behaviour**
> (README Troubleshooting), **6 stands as designed**, **9's label de-collision shipped (8.6)**
> and its containment dedup is closed by `contained_in` + D58 (flag, don't drop). Still open:
> **7** (bundled-open retry), **8** (fallback badge), **10** (`reocr_blocks`) — listed as
> FOLLOW-UPS 1–3 in `tests/test_p8_qa.py`, not as unmet requirements.

1. ✅ **done in 8.2** — `export_config` writes `mcp.auth_token` in cleartext (PLAN wants fully
   exportable MCP settings). Add `--redact-secrets` (or redact-by-default + explicit opt-in) in P8.
2. ✅ **done in 8.2** — warn at startup when `mcp.host` is non-loopback (e.g. `0.0.0.0`, as in
   `examples/config.custom.json`) while `auth_token` is null or `"change-me"`.
3. imgutils 0.19.0 `_safe_eval_names_str` uses deprecated `ast.Str`/`ast.Num` → warnings
   now, breaks on Python 3.14. Recommend Python ≤ 3.13; revisit upstream or vendor it.
4. `dghs-imgutils` pins `numpy<2` (no cp313 wheels) → install with `--no-deps` on 3.13
   (verified working set in Environment notes).
5. imgutils auto-`pip install`s onnxruntime at import when missing (silent network use at
   import time). Document; the GUI should pre-flight instead.
6. `PipelineResult.from_dict` still requires `created_at`/`schema_version` (so does the
   schema). Fine for v1; relax via a versioned migration if harnesses need it.
7. A bundled `model.onnx` that is *large but corrupt* (e.g. a copy that filled the disk
   and stopped) still counts as bundled (the threshold is 1 MB) and fails at `load()`
   with an ONNX parse error. `--verify` catches it via sha256, but `load()` does not
   consult the manifest. Nice-to-have: on a bundled-open failure, retry via the Hub once
   and record it in `status().detail` (P8).
8. *(half closed in T8)* The GUI now disables non-bundled variants while offline and puts
   the `scripts/fetch_models.py --variants …` fix in the item tooltip (D49). Still open: a
   visible badge for `model_list_is_fallback` ("model list not from the repo") when the Hub
   listing timed out, and pushing a live `list_models()` result into the combo (P7).
9. ✅ **closed (8.6 + 8.2b)** — **Nested boxes are real output** (T7): the detector returns a title band containing the
   per-column boxes, and a page-wide box containing a smaller one; NMS keeps them (low
   IoU). *Half closed in T9:* the canvas draws big-first (D50's `draw_order`), hover picks
   the **smallest** containing box (`pick_block`), chips are a second pass, and the hover
   panel carries the full text — so the page stays usable. Still open for P8: an opt-in
   containment dedup (`--drop-contained F`) rather than changing the default document, and
   **label de-collision** — with 12 boxes at fit zoom the chips overlap (visible in
   `out/p6_probe/real_fit.png`); a simple "skip a chip that would cover an already-drawn
   one, show it on hover instead" rule would fix it without touching the export.
10. ⬜ **open, not a requirement** — `Pipeline` currently OCRs every kept box. Once P6 shows per-block text, an obvious win
    is re-OCR of a single block (crop → `ocr_crops([crop], …)`) without re-detecting — the
    engine contract already supports it; the pipeline just needs a `reocr_blocks()` entry
    point (P8).

## Turn 10 — P7: MCP server + GUI bridge (in progress, checkpoint)

**Built (all compile, imports clean, `python3 -m animetext_studio --dump-mcp-tools` works):**
* `mcp/bridge.py` — `ActivityLog`/`ActivityEvent` (ring buffer, newest-first `events()`,
  `summary()`, listeners never break a call), `redact_args` (pixels/secrets → shape+length),
  `Dispatcher` / `InlineDispatcher` (headless) / `QtDispatcher` (queued custom `QEvent` to a
  receiver QObject; runs inline if already on the main thread; `pump()`, `close()`),
  `DispatcherTimeoutError` → `StudioError(INTERNAL)`.
* `mcp/schemas.py` — pydantic output models (extra=forbid) for all 19 tools + `ErrorResult`.
* `mcp/service.py` — `StudioService`: the whole tool behaviour, one implementation, GUI-optional.
  `attach_window()` swaps the dispatcher to Qt; window is the state owner when attached.
  Public methods: snapshot/models/config_snapshot/apply_config_patch/export_config/import_config/
  load_image/describe_image/apply_run_overrides/start_job/wait_for_job/job_result/result_payload/
  save_result/annotated/set_overlay/set_zoom/highlight/reset/cancel/unload + the 6 resource
  documents + `execute()` (dispatch + one activity entry) and `call/acall` (no logging).
* `mcp/tools.py` — 19 tools (§4's 18 + `unload_engines`), `ok()`/`failure()` (structured
  `structuredContent` on success AND on `isError`), `_run_job()` shared by detect/run_pipeline
  (progress → `ctx.report_progress`), `TOOL_GUI_EFFECT` (per-tool visual confirmation),
  `tool_catalog()`.
* `mcp/resources.py` — the 6 resources. `mcp/server.py` — `build_mcp_server`, `HttpGateway`
  (bearer + `/health` + `rejected` counter), `StudioServer` (uvicorn/stdio in a thread,
  `_preflight_bind`, `port`/`url`, `stop()` idempotent), `catalog_document`/`write_catalog`.
* `gui/activity.py` — `ActivityDock` (table + summary + status bar), `flash_widget`
  (`#objectName` selector + `WA_StyledBackground`, restores both), `TOOL_TARGETS`.
* `gui/app.py` — `build_service()` / `start_mcp()`; `run_gui` wires service+dock+server, badge,
  smoke line now reports `mcp=…`.
* `MainWindow` seams: `image`/`image_info`/`config` properties, `set_config()` (the one door),
  `apply_view_config()`, `clear_image()`, `set_closing_hook()`, named `export_button`/
  `import_button`. `load_path(..., source=)` so MCP loads record `mcp-path`.
* `core`: `JobManager.clear_result()`, `Pipeline.run(ocr_enabled=)` + kwargs forwarding through
  `run_file`/`run_bytes` (stage-1 `detect` needs it). CLI: `--serve`, `--dump-mcp-tools`.
* `schemas/mcp-tools.json` generated (19 tools, 6 resources, error taxonomy, guiEffect+flash).
* `tools/mcp_client_demo.py` — deterministic SDK session (19 steps incl. 3 negative checks),
  `--no-reset`, `--expect-connect-failure`, `--json` transcript, exits 0.

**Verified by hand:** headless HTTP session via the demo client → exit 0 (17 calls, 8 progress
notifications, annotated PNG, saved JSON, 401 rejection counted). GUI (offscreen) driven by the
same client: 15 dock rows, canvas 5 blocks, JSON pane 2175 chars, zoom 110 %, score spin 0.20
from `set_config`, Import button flashing red, badge `MCP: http://127.0.0.1:PORT/mcp [auth]`.
Screenshots: `out/p7_gui/01_before.png`, `02_after.png`, `03_canvas.png`.

**Tests:** `test_mcp_bridge.py` 12 passed · `test_mcp_service.py` 31 passed ·
`test_mcp_server.py` 10 passed (catalog-vs-shipped-file, gateway auth/health, lifecycle,
port-in-use). `test_mcp_http.py` written (12 tests, every payload validated against the shipped
catalog with jsonschema) — **needs the last fixture wiring**: conftest now has `free_port`,
`mock_service`, `mcp_http` (dict: service/server/info/url) and `slow_gate` (gates
`manager.pipeline.detection.detect`, yields `(release, entered)`); the BUSY and cancel tests
were switched to `slow_gate` but still contain a leftover `_SlowDetection` block to delete.
Still to do: run the full suite (expect 561 + ~75 new), stdio handshake test, GUI bridge test
(`tests/test_mcp_gui.py`), `docs/MCP.md` with the curl transcript, `examples/` session transcript,
then the P7 acceptance write-up + P8 backlog items (non-loopback warning, label de-collision…).

**Traps learned in P7 (do not re-learn):** mcp 2.x — `MCPServer` (not FastMCP), snake_case
attributes (`structured_content`, `input_schema`), a tool may return `CallToolResult` directly
(validation of `structuredContent` is skipped when `isError`), `ctx` must be annotated `Context`
(a `Context` param typed `Any` becomes a *required tool argument*), `streamable_http_client` is
2-tuple, use one `AsyncExitStack` (hand-entered context managers get cancelled from a generator
finaliser), `logging.disable(logging.CRITICAL)` or the rich tracebacks swallow your own output.
`QWidget.application()` does not exist. A stylesheet on a parent with `*{}` tints every child.

## Turn 11 — P7: MCP server + GUI bridge ✅ (finished)

Picked up from the Turn 10 checkpoint. **`python -m pytest -q` → 639 passed, 0 failed**
(561 at the end of P6, +78 from P7: bridge 12 · service 34 · server 10 · http 7 · stdio 3 ·
gui 12). No model weights, no network, no display needed for any of it.

**Closed out from the checkpoint:**
* `test_mcp_http.py` finished (7 tests): the `slow_gate` fixture wiring replaced the leftover
  `_SlowDetection` block (BUSY + cancel now wait for `entered`, so they race a *running* job);
  every payload validated against the shipped catalog with jsonschema; `PNG_MAGIC` added;
  `_call()` given a default `arguments` so zero-arg tools work.
* **`tests/test_mcp_stdio.py` (new, 3 tests)** — spawns `python -m animetext_studio --mock
  --serve --mcp-transport stdio` and talks down the pipe: `tools/list` + `resources/list`
  match the shipped catalog *field for field* (`tool.input_schema == entry["inputSchema"]`),
  a tool call + resource read + structured `NO_IMAGE` error work, and — the part only a
  subprocess can prove — **stdout carries protocol bytes only** (a hand-written `initialize`
  parses; the banner is asserted to be on stderr) and the child exits when stdin closes.
* **`tests/test_mcp_gui.py` (new, 12 tests)** — the acceptance criterion itself: a real
  `MainWindow` + `build_service()` + `start_mcp()`, driven by the SDK client **from a worker
  thread** while the main thread pumps Qt events (`drive()`). Asserts: badge text; `gui: true`
  in `get_status`; load+run land on the canvas / JSON pane / progress bar; `set_config` moves
  the score spin; `set_zoom`/`set_overlay_visible`/`highlight_block` move the canvas;
  `reset` empties the window; `save_result`/`export_config` write real files; three calls →
  three dock rows (newest first, error row carries its code) + status-bar line; base64 pixels
  never reach the log; closing the window detaches the service; a failed call leaves the
  window usable.
* **`docs/MCP.md` (new, 350 lines)** — transports, config, auth, Claude Desktop/Code snippets,
  all 19 tools grouped with their GUI effect, the 6 resources, the error taxonomy with CLI
  exit codes, progress, what the human sees per call, **a captured raw JSON-RPC transcript**
  (not hand-typed), the curl equivalent, the catalog, troubleshooting, limits, test commands.
* **`scripts/mcp_curl_demo.sh`** (the session in curl, prints each request as it runs) and
  **`scripts/mcp_http_transcript.py`** (the same session in httpx, emitting the transcript the
  docs embed — needed because the sandbox blocks the `curl` binary). Artifacts:
  `examples/mcp_raw_transcript.txt` (11 steps, exit 0) and `examples/mcp_session_demo.json`
  (the SDK demo client's 19-step transcript, exit 0).
* **Inspector compatibility verified for real**, not assumed: `npx -y
  @modelcontextprotocol/inspector --cli --transport http --server-url …/mcp --method
  tools/list` → 19 tools; `--method tools/call --tool-name load_image` → `isError: false`.

**Two real fixes found by the new tests (not test-only changes):**
1. `highlight_block` scheduled `QTimer.singleShot(ms, lambda: canvas.set_hover(-1))`; quitting
   during the flash fired into a deleted C++ object (`RuntimeError: … QLabel already deleted`,
   printed from inside the Qt loop). Now `ResultCanvas.flash_hover()` owns a **child** `QTimer`
   (`gui/canvas.py`), which dies with the canvas. Regression test:
   `test_closing_the_window_mid_flash_is_silent`.
2. Every tool call used to log an empty message unless the tool passed `summarize=`, so the
   dock said "ok". Added `default_summary()` in `mcp/service.py` (one place, forgiving: any
   unknown shape → ""), and timing to the two job messages: the dock/status bar now read
   `run_pipeline: 5 block(s) in 4 ms`, `load_image: 900x1300 sample_page.png`,
   `set_config: changed: detection.score_threshold`, `set_zoom: zoom percent 150 %`,
   `save_result: /path/r.json (2550 bytes)`. Tested in `test_mcp_service.py::TestDefaultSummary`.

**Two stale P1 tests updated (behaviour legitimately moved):** `--headless` with no action is
now `INVALID_ARGUMENT` (exit 2) naming `--run` and `--serve`, not the old `NOT_IMPLEMENTED`
stub (exit 20) — in `test_cli.py` and `test_pipeline.py`.

**Visual proof (offscreen, `scripts/_p7_final_screenshots.py` → `out/p7_final/`):** one window
after an 8-call remote session — badge `MCP: http://127.0.0.1:PORT/mcp`, score spin at 0.20
(from `set_config`), zoom 110 % percent, block 2 selected with its hover panel (`決戦の前夜`,
`(628,713) → (812,1146) 184×433 px`), Labels unchecked from `set_overlay_visible`, JSON pane
filled (2177 chars), and the activity dock reading:

```text
03:43:23 | get_result          | ok |  8 | 5 blocks in 4 ms
03:43:23 | set_overlay_visible | ok |  8 | overlay on, labels off
03:43:23 | get_annotated_image | ok | 30 | 900x1300 PNG
03:43:23 | highlight_block     | ok |  8 | block 2: 決戦の前夜
03:43:23 | set_zoom            | ok |  4 | zoom percent 110 %
03:43:23 | set_config          | ok |  2 | changed: detection.score_threshold
03:43:23 | run_pipeline        | ok |219 | 5 block(s) in 4 ms
03:43:23 | load_image          | ok | 14 | 900x1300 sample_page.png
```

Two polish changes came out of looking at that screenshot: the dock's time column now shows
`HH:MM:SS` instead of `HH:MM:SS+01:00` (`log://mcp` keeps the full ISO stamp), and
`highlight_block` says `block 2: 決戦の前夜` instead of echoing its own arguments.

**Housekeeping:** `WIP.md` deleted (its own instruction: remove once P7 is in tracking.md).
`tracking.md` is the single record. Verified again after all of this: **639 passed** default,
`-m models` → **8 passed / 7 skipped**.

**Traps learned in Turn 11 (do not re-learn):**
* mcp 2.x `Tool` objects use **snake_case** (`tool.input_schema`, `tool.output_schema`) while
  the wire JSON is camelCase (`inputSchema`) — comparing a catalog to a live `tools/list`
  means mixing both spellings on purpose.
* `resources/list` returns declaration order, not sorted; sort before comparing.
* pytest-qt reports an exception raised inside the Qt event loop as a **teardown error** on
  whatever test was running — that is how the flash-timer bug surfaced. Keep it that way.
* The activity log is synchronous; the dock follows via a queued signal, so a test that stops
  pumping events right after a call sees N-1 rows. `qtbot.waitUntil(...)` is the fix, not a
  longer sleep.
* A helper named `payload(session, tool, **kw)` cannot also take a positional dict — accept
  `arguments=None, **kwargs` from the start.
* `detect` publishes a stage-1 result, so `get_result` after `detect` **succeeds** (blocks
  with empty text). The result document has no `stage`/`ocr_ran` marker — P8 item.
* `set_config` with an unknown `detection.model` is a **warning**, not an error (custom models
  are allowed; it fails at load). Use an out-of-range number for a negative-path example.
* The sandbox blocks the `curl` binary and `rm`; use httpx and `Path.unlink()`.

## Turn 12 — 2026-11-22 — P8.2 complete + bundle refresh (new Hayai weights, all 5 variants)

**Context:** the user supplied a working HF token, re-downloaded the big weights (Hayai was
updated upstream with merged, edge-case fine-tuned weights) and vendored `yolo12l`/`yolo12x`
by hand. Two workstreams, both closed.

### On-premises bundle refresh (D55)

* **Verified before touching anything**: the six big files the user copied by hand are
  byte-identical to the current HEAD of their repos (sha256 vs the Hub LFS oids) — Hayai
  `model.safetensors` `3d8c1971cfb7…`, `yolo26m-text.pt`, and `model.onnx`/`model.pt` for
  `l`/`x`. Nothing was re-downloaded; only the 6 missing accessory files (68 bytes total)
  and the updated Hayai `README.md` were fetched.
* `OCR_REVISION` **`f5f0f617…` → `b00317ed69fccee5f787187d913fcf8a96f37f65`** (repo HEAD,
  updated 2026-11-22 09:53 UTC). The snapshot's Python + tokenizer files are byte-identical
  to the previous revision, so the Part 3 engine API is unaffected — the model card confirms
  the change is weights only ("Stage 4 — Long-Text Edgecase Fine-Tuning and WiSE-FT Merge").
* **All five detection variants are now on premises** (n 10.0 / s 35.5 / m 77.0 / l 100.9 /
  x 225.7 MB) with `threshold.json` + `labels.json` + `model_type.json` for each; the fetched
  `l`/`x` metadata matches `MODEL_PRESETS` exactly (0.426 / f1 0.89, 0.425 / f1 0.90), which
  the `-m models` metadata test re-checks against the Hub.
* `fetch_models.py` gained `DETECTION_EXTRA_FILES`: a `model.pt` that happens to be in a
  variant folder is **recorded** in `MANIFEST.json` (provenance + `--verify` coverage) but
  never downloaded — the runtime only reads ONNX. Manifest now **33 files, hashes verified**;
  `--verify` → `RESULT: usable offline`; bundle **1.26 GB**.
* `python3 -m pytest -m models` went from **8 passed / 7 skipped** to **14 passed / 1 skipped**
  once a token was available (the one skip is now vacuous: "every variant is vendored").

### Token plumbing (D54)

`model_store.discover_hf_token()`: `--hf-token` > `$HF_TOKEN` > `HF_token.txt` / `.hf_token`
(cwd, project root, models root) > `$ANIMETEXT_TOKEN_FILE`. Used by `DetectionEngine._token()`
and `fetch_models.py`. Read-only, never printed, never written to a config; `.gitignore` now
covers `HF_token.txt` / `.hf_token` / `*token*.txt`. Deleting the file removes the capability
completely — the app has no other place it stores a token.

### P8.2 — security & config hardening: ✅ (D56, D57)

Shipped this turn (on top of the log-capacity/redaction/warning work in `P8.md`):
`--redact-secrets` on the CLI (`--export-config` → `save_config_file(redact=)`, and the output
says "(secrets redacted)"), and the **engines badge**: `engines_badge_text()` in
`engines/base.py` (Qt-free) + `ControlPanel.engines_badge` + `MainWindow.refresh_engine_badge()`
refreshed on finish/fail/cancel and by `StudioService.unload()` — closing the P7 backlog item
"`unload_engines` has no GUI effect beyond a log row".
New tests: `TestEnginesBadge` (6), `TestEngineBadge` (4, offscreen), `TestCliRedaction` (3),
`TestTokenDiscovery` (7), `TestUnloadBadge` (2) → **701 passed, 15 deselected**; `-m models`
**14 passed, 1 skipped**.

### Real-model comparison (new weights, same settings, CPU, `examples/sample_page.png`)

| Run | Blocks | detect | ocr | total |
|---|---|---|---|---|
| `yolo12m` @ `f5f0f617` (T7 evidence, `out/pipeline_real.json`) | 12 | 2289 ms | 4580 ms | 6869 ms |
| `yolo12m` @ `b00317ed` (`out/p8_newweights_m.json`) | 12 | 2008 ms | **2031 ms** | 4040 ms |
| `yolo12x` @ `b00317ed` (`out/p8_newweights_x.json`) | 13 | 2566 ms | 2358 ms | 4925 ms |

All 12 strings from the `m` run are **identical** to the previous weights (no regression on
this page) and OCR is ~2.2× faster on CPU. Two observations that feed the addendum:
the page-wide narration slab is read as one run-on string
(`決戦の前夜風が止んだ第4話その日、空は赤かった` — four lines, no separators), and `yolo12x`,
which finds more **nested** boxes, OCR'd one nested crop as the fragment `った` — a tail of a
line whose rest lives in the parent box. Both are geometry problems, not decoder problems.

### Also

* New [`ADDENDUM.md`](ADDENDUM.md): the language-policy / sub-crop / dedup ideas evaluated
  against the vendored model card and today's measurements, with verdicts. Nothing there is
  scheduled; it is deliberately outside the P1–P8 plan.
* `STATE.md` refreshed; `P8.md` now shows 8.1 ✅ 8.2 ✅ and 8.3 as the next subpart.

## Turn 13 — 2026-11-22 — P8.2b: cheap correctness flags (result schema 1.2) + addendum decisions

**User decisions taken first, and they shaped the subpart:** addendum items **A** (language/script
policy) and **B** (sub-crop splitting) are **out of scope for this app** — that discrimination
belongs to the caller driving the MCP, not to us (D60). `ADDENDUM.md` was rewritten to say so: the
§1 measurements stay, but as *consumer* documentation, and both items are marked "not to be
re-proposed as app features". What the app owes the caller is enough truth to decide with, which is
exactly what item **D** — the cheap flags — supplies. That is what shipped.

**Shipped (result schema 1.1 → 1.2, D58/D59):**
* `blocks[].flags` — `"possible_truncation"` when `len(text) >= 0.9 * ocr.max_new_tokens`, the
  model author's own cap-hitting ratio (`processing_hayai.check_escalation`). A list, not a
  boolean, so the next observation does not need another schema bump.
* `blocks[].contained_in` — id of the tightest **strictly larger** box covering >= 80 % of this
  one, else `null`. Pure geometry: the `った` case (a fragment inside a narration slab read as its
  own line) becomes visible to a consumer without us dropping or merging anything.
* `ocr.max_patches` now accepts **512** everywhere (config validation, both JSON schemas, GUI
  combo, engine `MAX_PATCH_CHOICES`). Our ceiling, not the model's: the vendored processor
  implements the 512 tier and its escalation targets it.
* Both annotations are emitted for `stage: "detect"` documents as well — containment needs no OCR.
* `schemas/result-1.1.schema.json` frozen next to `result-1.0.schema.json`; `from_dict` accepts
  1.0/1.1/1.2 and reads old documents as unflagged (`flags: []`, `contained_in: null`).
* Implementation lives in `core/pipeline.py` (`annotate_blocks`, `contained_in_map`,
  `is_possibly_truncated`) and `core/types.py` (`BBox.contained_ratio`, the two new fields) — all
  pure, Qt-free, deterministic (ties break by id).

**Found while testing, and left alone deliberately:** a `stage: "detect"` document carries the
**raw** detection boxes while a `full` document carries the padded *effective* boxes (D38),
because a stage-1 run never reaches the crop step. It is consistent with what each run actually
did, and the difference is `ocr.crop_padding_px` (2 px by default) — but it means "same page,
two documents" boxes differ by that much. Noted in P8.md; changing it would move every stage-1
box for a 2 px cosmetic gain.

**Tests:** default suite **707 → 727 passed, 15 deselected**; `-m models` → **14 passed, 1 skipped**
with the refreshed weights on disk. New: `tests/test_pipeline.py::TestDocumentAnnotations`
(13), `tests/test_types.py::TestBlockFlags` (5), `tests/test_schemas.py::TestResultSchemaVersions`
(the 1.0/1.1 frozen-reader tests rewritten around a `_downgrade` helper that drops exactly the keys
a version never had, plus 4 tests for the new fields), and the 512 budget added to the existing
config/engine budget loops.

**Docs:** `docs/MCP.md` gains a flags/`contained_in` section (what each means, why each can be
trusted, and the explicit note that acting on them happens outside the app); `ADDENDUM.md` item D
marked shipped with the "deliberately not done" list (`dense_slab`, per-block auto re-runs, GUI
surfacing — the JSON pane shows them, the block table can pick them up in 8.6).

## Turn 14 — 2026-11-22 — P8.3: warm-up, engine warmth, measured memory release

**What `--preload` became.** `core/warmup.py` (Qt-free) loads each engine and then runs one
throwaway inference on a 64×64 probe, because loading is only half of what the first page pays for.
It runs as a job — `JobManager.start_warmup()`, `kind: "warmup"` — so it is single-flight,
progress-reporting, cancellable, and produces no result document: the last real page survives
`has_result`, `result://latest` and the canvas. Entry points: `--preload` in the GUI (after the
window is up, never blocking; skipped with one log line under `--mock`), `--headless --serve
--preload` (a thread, so the port opens first), and the new MCP tool `warm_engines`, which answers
with a job handle rather than blocking the call. That tool exists because `unload_engines` had no
counterpart: an agent that lent its GPU away could only come back by running a page it did not want
(20 tools now, not 19).

**Warmth is now something you can check.** `EngineStatus` carries `inferences` and a derived
`warm`, both surfaced in `get_status`/`snapshot` as `state`/`loaded`/`warm`/`inferences`. `unload`
resets them. `probe: false` loads without inferring, which from cold leaves `loaded: true,
warm: false` — the distinction is the feature.

**Two defects found by measuring instead of assuming:**
* `engines[].loaded` had answered `false` since P7, for every engine in every state: `str()` on a
  `(str, Enum)` member is `"EngineState.READY"`, and the comparison wanted `"ready"`. No test
  covered it because no caller had a reason to read it until warmth made it interesting (D63).
* The unload/RSS test I wrote for this subpart **failed**: with onnxruntime 1.32 and
  `yolo12m_animetext` on CPU, load + one inference cost +975 MB and unload returned 191 MB (244 MB
  after an explicit `malloc_trim(0)`) — while `live_sessions()` dropped to zero and a second load
  reused the space. So the promise is now written honestly: objects released, memory reusable,
  `warm`/`loaded` cleared, and the residual belongs to ONNX Runtime's process-wide state until the
  process exits (D62). `core/memory.py` adds `process_rss_mb`, `trim_heap` (glibc only, called from
  both engines' teardown) and `live_sessions`, which is the assertion that actually means
  something.

**Tests:** default suite **727 → 753 passed, 16 deselected**; `-m models` → **15 passed, 1
skipped**. New `tests/test_p8_lifecycle.py` (29): warm-up outcomes and ordering, `probe=False`
semantics, cancellation through a blocked load, single-flight both directions, no-result-and-
previous-page-survives, progress stage, report shape and `raise_for_failures`, the MCP tool
surface, engine counters, RSS release/reuse with real weights, and the GUI wiring (the bridge must
not mistake a warm-up for a page — that would repaint the canvas with a stale result).

**Docs:** `docs/MCP.md` gains an "Engine warmth" section (the three fields, what warm-up is, the
measured unload numbers), the `warm_engines` row, the honest `unload_engines` row, and 19 → 20
in the two places that counted. Catalog regenerated.

## Turn 15 — 2026-11-23 — P8.4: the error-path matrix (84 faults, 8 doors, 11 real fixes)

**What was built.** `tests/test_p8_errors.py` (215 tests) drives a `MATRIX` of **84 named faults
across 8 entry points** — imaging, pipeline, job machine, config, MCP service, MCP wire (real
HTTP + the official client), CLI, GUI — and three sweeps over it: no fault escapes as a bare
exception, every row carries the code the matrix promises, and the app still processes a page
after the whole table has been thrown at it. `docs/ERROR-MATRIX.md` is the same data in prose —
taxonomy, per-door tables, what each tool can answer with, what each resource can answer with,
the five invariants — generated by `scripts/gen_error_matrix.py` (`--check` for CI) so the
documentation cannot drift from the tests (D70).

**Eleven defects the matrix found** (none of them found by reading the code):
1. `load_image(None)`, `load_image(b"…")`, `load_image("")` → bare `TypeError` / a confusing
   `"Cannot read image: ."` (D64).
2. `import_config(path=…)` on a UTF-16 file → bare `UnicodeDecodeError` (`UnicodeDecodeError` is a
   `ValueError`; the `except OSError` never had a chance). A UTF-8 BOM now also loads (D64).
3. `get_result(block_ids="nope")` → `ValueError: invalid literal for int()`; `[1.7]` silently
   became block 1 (D64).
4. An engine returning `None` or dicts instead of detections → `TypeError: 'NoneType' object is
   not iterable` reported as `INTERNAL`; a short or `None`-filled OCR answer produced blocks with
   **`text: null` and no `error`**, breaking the invariant D39 had promised (D65).
5. base64 with MIME line breaks — which is what `base64 file | pbcopy` and most clients produce —
   was rejected outright (D66).
6. The 64 MB base64 cap admitted ~96 MB (`* 2` instead of `4/3`), and because it raised *outside*
   the dispatch path the caller got no `{code,…}` and the activity log got no row (D66).
7. `get_annotated_image(scale=100)` was clamped to 8.0 and then allocated 8× the page — on a large
   page, a single tool call could exhaust RAM (D67).
8. **Every argument the MCP SDK rejected arrived as `isError` with a sentence and no code** —
   `structuredContent: null` — so a harness branching on `code` had nothing to branch on, and the
   GUI log showed *nothing at all* for a client hammering the server with garbage. Same for an
   unknown tool name. Both now answer in the §2.3 shape and log one row each (D68).
9. **Resource failures lost the taxonomy completely**: `result://latest` with no result reached the
   client as `{-32603, "Error reading resource result://latest"}` while the published catalog
   promised `NO_IMAGE` (D69).
10. `StudioError.from_dict` — the reader for error documents, i.e. the one place a *foreign* code
    can arrive — raised `ValueError`/`KeyError` on an unknown code or a missing message (D64).
11. The P8.3 unload/RSS test was intermittently red in a full run: an honest claim about the
    allocator's behaviour, in the wrong process (D71).

**Behaviour changes worth naming** (all pinned by tests, all documented):
`annotated(scale=…)` refuses instead of clamping, and bounds the rendered area; `block_ids` must be
a list of integers; a wrong-count engine answer fails the run (`INTERNAL`, `details.contract`) while
a *missing* answer degrades to per-block errors; `load_image` distinguishes "you gave me nothing"
from "the source you gave me was empty"; unknown *tool arguments* are ignored by the SDK (pinned,
not changed).

**Tests:** default suite **753 → 970 passed, 16 deselected** (3 clean runs in a row); `-m models`
→ **15 passed, 1 skipped** (the OCR memory measurement is now taken in a clean subprocess, so the
strict claim stays).

**Docs:** new [`docs/ERROR-MATRIX.md`](ERROR-MATRIX.md); `docs/MCP.md` gains the two SDK-decided
cases, the resource `error.data` shape, the corrected `get_annotated_image` scale range, and a
fixed error table (it had lost `INVALID_CONFIG` and `NOT_IMPLEMENTED` and still promised a 30 s
dispatch timeout — now checked against `EXIT_CODES` by a test).

## Turn 16 — 2026-11-23 — P8.5: the real-model end-to-end suite (25 tests, every door, one latent defect closed)

**What was built.** `tests/test_p8_e2e_models.py` — 25 opt-in `-m models` tests, ~25 s on CPU, no
token and no network: the first suite that runs the vendored weights through the **product**
instead of through a unit. One canonical run of `examples/sample_page.png` on
`yolo12m_animetext` + the pinned Hayai snapshot feeds the document tests; each door does its own
run; `sample_scans/` are used as opaque inputs (geometry and validity only — the standing rule
that their content is not inspected is written into the module docstring).

Five groups, because "it works with real weights" is five separate claims:

* **the weights** — the bundle is what runs, at `OCR_REVISION`, and a full run opens **zero
  sockets**: `socket.connect` is wrapped for the duration of the run, which is the first time
  "a copy of this folder is a complete offline install" (D26) has been an asserted fact rather
  than an intention;
* **the document** — schema-valid, and every invariant re-derived from the JSON alone (ids,
  reading order, palette colour, boxes inside the page, the D39 "null text carries an error"
  rule, `contained_in` geometry against `CONTAINMENT_RATIO`, flag against `len(text)` vs the
  token cap); a content canary against the page generator's own lines, tolerant on purpose
  because models get re-released; byte-determinism modulo `created_at`/`timing_ms`; a round trip
  through `from_dict`; a stage-1 run that reads nothing and still validates;
* **the flags on real data** — `possible_truncation` fires under a real 8-token cap and matches
  the token count on every block; `max_patches=512` runs against Hayai (measured: the same text
  as `auto` on a page this size — the tier is about reach, not about changing the reading);
* **the doors** — `--headless --offline --run` in a real subprocess (stdout is *only* the
  document, and a cold process reproduces the warm one byte-for-byte), the MCP service door, and
  the MCP wire over HTTP with the official client validated against the **published** tool
  schema as well as `result.schema.json`, with progress events, the overlay PNG and the log;
* **the lifecycle** — a real warm-up (3.4 s detection + 1.4 s OCR here) as a job that leaves the
  page the user is looking at alone, and a cancellation gated *inside* the OCR call, so the
  assertion is that Hayai's own chunk loop honours `check_cancel` rather than that the test won
  a race.

**One defect closed (D72).** `mcp/tools.py` dispatched `wait_for_job` through `acall()`. Today
that is harmless — the progress pump leaves its loop only after the job settles, so the join
returns immediately — but it contradicted the method's own docstring ("never on Qt's"), and a
dispatched *wait* is precisely the call that must never reach the app thread: with a window
attached it holds the GUI thread for the length of the job (frozen window, dead progress bar,
unresponsive cancel) and then answers `INTERNAL: The application did not answer within 20 s` for
a job that is merely slow and goes on to finish. Mock jobs cost milliseconds, which is why nine
suites' worth of MCP tests never found it. The wait now runs on the caller's thread
(`JobManager.wait` is thread-safe by design), `StudioService.wait_for_job` documents why it must
never be dispatched, and
`test_mcp_service.py::test_the_job_wait_is_never_dispatched_onto_the_app_thread` pins the shape —
confirmed to fail when the dispatch is put back.

**Found while probing, not a defect:** the same probe initially *appeared* to freeze the GUI on
every MCP run; it was the probe's own event pump that had stopped, not the app's. Recorded here
because the difference is the whole bug class, and because the corrected pump is what made the
answer trustworthy.

**Also corrected:** the `models` marker and `test_models.py`'s docstring still described a suite
that downloads and needs a token. Since D26 it needs neither — only `TestRealDetection` touches
the Hub, and it skips without a token.

**Tests:** `-m models` **15 passed / 1 skipped → 40 passed / 1 skipped** (~36 s); default suite
**970 passed, 41 deselected** (the +1 deselected is the new marker wording; the +1 test is the
dispatch pin, which runs by default).

## Turn 17 — 2026-11-24 — P8.6: labels that do not sit on each other, and the 1.2 flags in view

**What was built.** `tests/test_p8_annotation.py` (43 tests) plus the rule it tests:
`core/annotation.py` gained `place_chips()` / `chip_slots()` / `chip_variants()` /
`ChipOffer` / `Chip` / `ChipPlan` / `overlaps()`, and `flag_marks()` / `flag_note()` /
`flagged_count()`. `render_overlay` and `canvas.paint_overlay` now both call `place_chips` —
**one rule, two painters** — and neither paints chip geometry of its own any more.

The rule is pure geometry: the painter measures its own texts (`ImageDraw.textbbox`,
`QFontMetrics`) and hands over `(text, w, h)` sizes, because the two engines disagree by a pixel
and the *rule* must not. A chip is claimed **smallest box first** — the same rule that decides
which box a click lands on (`pick_block`) and the same one `contained_in` describes, so a
panel-sized false positive never out-claims the bubble inside it. Then: next edge (above →
inside-top → inside-bottom → below, clamped onto the page, duplicates dropped) → **keep the
number, drop the words** (`#7`, far narrower, and the id is the one thing not recoverable
elsewhere: it keys the table, the JSON and `focus_block`) → dropped, and listed in
`ChipPlan.hidden`. Words beat position: a full label below the box beats a bare `#id` above it.

Two properties were treated as more important than any individual case. **The layout ignores
hover**, so chips cannot jump while the mouse moves — the hovered/selected block is painted over
the top instead (that is what the backlog's "show it on hover instead" means, and the export
carries it too). And **the exported document is untouched**: pinned by hashing
`result.to_dict()` around two renders.

The first property earned its keep twice over. Measured, `place_chips` costs 0.1 ms on this page's
12 boxes but 8.9 ms on 300 — and the canvas repaints on *every mouse move*. Because the layout is
by construction a pure function of the blocks, the style and the view (hover is not an input), the
canvas caches it against `(blocks revision, style, zoom+scroll matrix, viewport, font pixel size)`
and hover repaints pay nothing; a test counts the calls across four hover changes (no new call)
and one zoom change (a new call). Same reason, two promises: **stability and cost both follow from
not reading the mouse**.

A 200-page seeded random sweep asserts the invariant exhaustively (no two placed chips ever
cross, every chip stays on the page, every block is either painted or reported) instead of
trusting hand-checked geometry; `style.decollide_labels = False` restores the pre-8.6 picture
exactly and doubles as the control in the before/after measurements.

**Deliberately not a config key.** The config document stays at 1.2: a paint detail does not earn
a version bump, and `OverlayStyle.decollide_labels` is the seam a future `gui.labels_decollide`
would use. The old "opt-in containment dedup" from the 8.6 brief stays closed by 8.2b's
`contained_in` (D58: flag, don't drop).

**The flags reach the eye** (the second half of the brief): a `flags` column in the block table
(`⚠` = `possible_truncation`, `⊂n` = contained in block n), a header tooltip that spells the
alphabet out, a per-row sentence naming the knob to turn (`max_new_tokens`) and the container to
read first, `· 2 flagged` on the summary line only when there is something to say, the marks in
the status line when a block is picked, and both notes at the bottom of the hover panel. One
spelling for all five places (`FLAG_MARKS` / `flag_note`), and an unknown flag renders as its own
name — a flag the viewer cannot show is a flag lost. The JSON pane already carried both keys;
`hover_text` gained the notes, which is also where a chip that shrank to `#id` still reaches its
full text.

**One defect closed (D73), found by looking at the pictures.** `canvas.paint_overlay` built a
`QFont` + `QFontMetrics` to size every chip and never called `painter.setFont(font)`, so
`drawText` drew with the widget default font: every label was wider than its chip and Qt clipped
both ends — `#0·0.80·この攻撃で決まりだ！` came out as `.80 この攻撃で決ま`, in the window *and* in
every exported PNG, since P6. It is visible in the Turn 16 screenshot and no test caught it,
because nothing compared the drawn text with the measured text. It also made the new de-collision
maths lie, since `place_chips` reserves the space the *measured* text needs. Pinned by a probe
that walks up a chip's padding and refuses a glyph outside it — confirmed to fail when the line is
removed.

**Measured**, on `examples/sample_page.png` (12 blocks, real bundled `yolo12m_animetext` + pinned
Hayai, CPU): nothing dropped at fit zoom (`hidden_labels == ()`), no two chips crossing, and the
window paints strictly more label area than the old rule — the assertion is the measurement
(`on > off`), not a screenshot promise. Before/after pictures: `out/p86/canvas_{on,off}.png`,
`out/p86/export_{on,off}.png`.

**Docs:** `docs/MCP.md` gained the de-collision rule under View (why a label may be short or
missing, and why zooming in brings it back) and the flags paragraph under Result out.

**Tests:** default suite **970 → 1014 passed, 41 deselected** (~35 s); `-m models` unchanged at
**40 passed / 1 skipped**. `scripts/gen_error_matrix.py --check` still clean after the `docs/MCP.md`
additions.

## Standing rules (each one was earned by a real bug)

* Never re-download weights — everything needed is in `animetext_studio/models/` (D26);
  `fetch_models.py --verify` is the health check, `--offline` is a promise the app keeps.
* Default tests never touch the network, the weights, or a display; real-model tests are
  `-m models` (offline and tokenless now — the weights are on premises).
* PySide6 must not be imported at module level from `core/`, `engines/` or `mcp/`; headless-safe
  code lives in `core/`. `tests/test_p8_qa.py` proves it by importing every non-GUI module with
  `PySide6` blocked.
* Never connect a slot with an optional first parameter to a signal that passes args (D53);
  never build a `QRect`/`QRectF` from xyxy (use `canvas._qrect`/`_qrectf`).
* Documents are versioned and additive: readers accept older ones (result 1.0 → `stage="full"`,
  config 1.0/1.1 → 1.2 defaults). Bump only when the shape changes; freeze the old shape as
  `schemas/result-1.x.schema.json`.
* Report, don't act: observations (`flags`, `contained_in`, `restart_required`) never rewrite a
  page, a budget or a socket (D58, D60, D78).
* `sample_scans/` are opaque inputs — I/O and geometry only, content never inspected, no
  annotated renders of them written into the workspace.
* Generated files are generated: `docs/QA.md`, `docs/ERROR-MATRIX.md`,
  `animetext_studio/schemas/mcp-tools.json` — edit the source table or the server, then run the
  script (`--check` in CI).
* A doc claim needs an owner in code. If a statement exists only in prose, it is a test row or it
  is a future lie (D79).
* The licence is Apache-2.0 for the code and nobody else's for the weights; `LICENSE` and `NOTICE`
  ship with any redistribution, and a new vendored component gets a NOTICE line the day it lands
  (D77).
* The HF token is the user's: `HF_token.txt`, gitignored, never copied into a config export, a
  fixture, a log or a commit. The QA secret scan is the guard.

## Next up

**Nothing is scheduled.** All eight parts of `PLAN.md` are complete and verified, and so is the
ninth: P9 closed on 2026-11-26 with the five approved enhancements and the trail that goes with
them (T20). The project is usable as shipped: `python -m animetext_studio` for the
window, `--headless --run` for the document, `--serve` for the MCP door, and a copy of the
folder is a complete offline installation (D26).

What a next session would pick up, in the order I would do it — all of them **enhancements, not
unfinished work** (each is also in `FOLLOW_UPS` in `tests/test_p8_qa.py`, which prints them into
`docs/QA.md`). **All five were approved on 2026-11-25 and are written up, with their blast
radius, in [`enhancements_proposition.md`](enhancements_proposition.md); the work was P9.**:

> **Status (2026-11-26): closed.** 1–5 shipped as subparts 9.1–9.5 of
> [`P9.md`](P9.md), and 9.6 — the trail — shipped the paperwork with them: the matrix is 91
> faults, the checklist is 131 requirements, the floors and the sign-off are measured. Running
> the doors for real turned up two defects (D93: an option given `""` opened the window; D94:
> `--dump-mcp-tools PATH` ignored PATH and rewrote the shipped catalog). The list below is left
> as written because it is the order the work was agreed in; what is genuinely still open is in
> `FOLLOW_UPS`, and the two biggest items there are that the upstream issues are **drafted but
> not filed** (D92) and that `scripts/bench_models.py` (ADDENDUM item E) is still unbuilt.

1. **`reocr_blocks()`** — re-read one block without re-detecting (backlog 10). The engine
   contract already supports it; it is a pipeline entry point plus a tool.
2. **Bundled-open retry** — a large-but-corrupt `model.onnx` counts as bundled and fails at
   `load()`; retry once via the Hub and record it in `status().detail` (backlog 7).
3. **`model_list_is_fallback` badge** — say so in the window when the model list came from the
   compiled presets rather than the repo (backlog 8, half done).
4. **8.3b** — per-device first-inference latency in `warmup_report`, and `--preload-off` for
   benchmarking. Deliberately deferred when 8.3 shipped.
5. Upstream nudges, not our code: imgutils 0.19's `ast.Str`/`ast.Num` (Python 3.14) and its
   silent `pip install onnxruntime` at import; both are in README Troubleshooting.

Housekeeping that is already true and should stay true: `HF_token.txt` is the user's to delete
(gitignored, mode 600, never copied anywhere — the QA secret scan is the guard); `out/`,
`p8t.txt` and `unsloth-tmp/` are scratch and deletable; the generated documents
(`docs/QA.md`, `docs/ERROR-MATRIX.md`, `schemas/mcp-tools.json`) are regenerated by their
scripts, never hand-edited.

## Turn 18 — 2026-11-24 — P8.7: the README, and the packaging review that turned up four defects

**What was built.** `README.md` (574 lines) — the front door that did not exist: install (CPU and
CUDA, the imgutils/`numpy` situation, the Blackwell wheel index), the gated-repo token setup with
its discovery order and its "a token in an exported config is a secret in a file" warning, a GUI
tour (badges, every panel group, canvas, table with the `⚠`/`⊂n` legend, JSON pane, MCP dock), the
full CLI table with all fourteen exit codes, an MCP summary with all 20 tools and 6 resources
linking to `docs/MCP.md`, the configuration reference (every key of the 1.2 document with its
range), the result-document reference including the 1.2 flags and the determinism contract, the
model table, the measured CPU numbers, the memory floor, a 20-row troubleshooting table, the
licensing picture, and the deliberate non-goals (D58, D60) so nobody files the language policy and
sub-crop splitting as bugs.

**Docs that are not pinned rot, so they are pinned.** `tests/test_p8_packaging.py` (22 tests)
compares the README against every fact that also lives in a machine-readable place: every long flag
in `build_parser()`, every `ErrorCode` with its **number** (matched by row shape, not by proximity
— the first version passed by reading a prose row and would have kept passing after a rewording
made it a lie), every tool in `schemas/mcp-tools.json`, every leaf key of `AppConfig.default()`,
every model preset and F1, every relative link, `pyproject` version against `__version__`, the
console-script target resolving to a callable, and the `all` extra against the four it is the union
of. It also pins the packaging contract itself, below.

**Four defects, all found by reading the packaging instead of trusting it.**

* **D75 — `.origin.json` was never shipped.** `package-data` said `models/*/*.json`; setuptools globs
  with stdlib `glob`, and `*` does not match a leading dot. A wheel therefore dropped the OCR
  snapshot's provenance file, and `ModelStore.ocr_status()` fell back to the compiled
  `OCR_REVISION` — the right answer today, silently wrong the first time someone re-vendors the
  bundle without editing code. Pattern added; the test now diffs the shipped set against every
  non-`.py` file under the package, so the next file nobody thought about fails instead of vanishing.
* **D76 — `MANIFEST.json` recorded absolute paths, so the bundle's passport expired when the bundle
  moved.** `fetch_models.py --verify` iterated the recorded paths and called every one of the 33
  files RECORDED BUT ABSENT on any other machine, any other prefix, any `$ANIMETEXT_MODELS_DIR` —
  `RESULT: incomplete`, exit 1 — on a bundle the app was loading perfectly well. That is the exact
  scenario D26 exists for. Entries now carry `relpath`; `manifest_entry_path()` resolves it, and for
  a manifest that predates the fix it re-bases the absolute path onto the recorded `models_root`
  *only when the re-based file exists*, so a genuinely missing file is still reported missing
  instead of being explained away. `write_manifest` keys its merge on the relative path (a merge
  keyed on the absolute one would have written a second entry for the same file after a move),
  `--verify` and `_already_there` go through the resolver, and the shipped manifest was migrated in
  place — hashes, revisions and the `model.pt` records untouched. `--verify` still prints
  `RESULT: usable offline`.
* **`mcp>=1.9` was a floor the code cannot run on.** `mcp.server.mcpserver` is the 2.x import path,
  and the app's own `ENGINE_LOAD_FAILED` hint had been saying `pip install 'mcp>=2.3'` since P7.
  The extra now says the same thing the error does.
* **`all` was a hand-copied list** of the other four extras and had already fallen a version
  behind. Now the PEP 621 self-reference `animetext-studio[gui,detect,ocr,mcp]`, which cannot drift.

Also: `readme`/`keywords`/`classifiers` added to `pyproject.toml`; `examples/config.default.json`
regenerated (it was a 1.0 document referenced by `docs/MCP.md` as the current default), while
`examples/config.custom.json` stays at 1.0 on purpose and the README says why.

**Measured, not assumed:** metadata was built with `prepare_metadata_for_build_wheel` and the
generated `METADATA` read back (extras, keywords, classifiers, the self-reference normalising to
`animetext-studio[detect,gui,mcp,ocr]`); the shipped file list came from setuptools' own
`build_py.find_data_files`, not from my own glob. **A full wheel build is impossible in this
sandbox** — `RLIMIT_FSIZE` is 100 MiB, so the 651 MB `model.safetensors` cannot be copied into
`build/`; the review says so rather than claiming a build it did not run.

**Deliberately not done:** no `license` field (**open question for the user** — the vendored weights
are GPL-3.0 and Apache-2.0, the README says so, but the licence for *this* code is a distributor's
decision and a packaging commit is the wrong place to make it); no `MANIFEST.in` (the wheel is the
deliverable, and an sdist would be a second 1.3 GB artifact nobody ships); no `requires-python`
ceiling (imgutils is what breaks on 3.14, not the app — that went into Troubleshooting instead of a
cap that would punish a GUI-only install); no `py.typed` (nothing here has been through a type
checker, and the marker advertises conformance we have not proven).

**Suite:** 1036 passed, 41 deselected (~35 s) · `-m models` 40 passed, 1 skipped ·
`scripts/gen_error_matrix.py --check` clean. Next: **8.8 final QA**.

## Turn 19 — 2026-11-24 — P8.8: the licence, the final QA pass, and the checklist that checks itself

**Two things happened, in order.** The distributor answered the one question 8.7 left open — the
code is **Apache-2.0** — and then 8.8 ran the plan's last item: a QA pass against every original
requirement, with the P8 entry written here and `P8.md`/`STATE.md` folded in and deleted.

### The licence (D77)

* `LICENSE` — the canonical Apache-2.0 text, appendix copyright filled in. Pinned by the sha256
  of **everything before the appendix**, so the terms cannot be edited while the copyright line
  stays personalisable.
* `NOTICE` — the inventory of what is not ours: both model repositories with their revisions and
  licences, the runtime dependencies, the fonts, and the scans that belong to the user.
* `pyproject.toml` — `license = "Apache-2.0"` + `license-files = ["LICENSE", "NOTICE"]`, PEP 639
  style, **no `License ::` classifier** (setuptools deprecated them once the SPDX field existed;
  two spellings of one fact is two chances to disagree). Verified through setuptools' own
  `prepare_metadata_for_build_wheel`: `Metadata-Version: 2.4`, `License-Expression: Apache-2.0`,
  `License-File: LICENSE`, `License-File: NOTICE`, both files under `dist-info/licenses/`.
* `animetext_studio/models/animetext-yolo/LICENSE.GPL-3.0.txt` — the GPL text beside the GPL
  weights, which were being redistributed without it. The Hayai snapshot needed nothing: it is
  Apache-2.0 too, and its `.origin.json` already records that.
* README §Licensing rewritten: what Apache covers, what it does not, the four practical cases
  (run / copy / ship-the-weights / ship-the-slim-wheel), and the non-goal that bundling is not a
  relicensing.
* **A memory of mine was wrong and the test caught it**: I wrote `dghs-imgutils — Apache-2.0` in
  NOTICE. It is **MIT**. The fix was not to correct the string but to stop trusting memory: the
  test now reads every dependency's own metadata and fails when NOTICE disagrees with the wheel.
* Ten licence tests in `tests/test_p8_packaging.py`, each mutation-checked (pyproject says MIT,
  licence body edited one word, placeholder copyright restored, NOTICE mislabelling a
  dependency, GPL text tampered, README dropping the NOTICE link) — all six fail as they should,
  and the clean re-run is green.

### The QA pass (D79)

`tests/test_p8_qa.py` is the checklist: **115 requirements** in 17 groups, each with a source, a
status and evidence that must resolve — a test node id looked up in the suite's ASTs, a command
whose flags are checked against the real argument parser, or a check in that file. Four rules: no
row without evidence; no evidence that does not exist; nothing marked unmet (what is not done is
`FOLLOW_UPS`, labelled *not a requirement*, with the reason); `docs/QA.md` generated from the
table with `--check`, like the error matrix. Counts are floors, so a suite that shrinks fails.

19 executable checks cover what no single test owned before: PLAN §5's CLI surface against the
parser (22/22 present), PLAN §4's 18 tools against the catalog (all ship, plus
`unload_engines`/`warm_engines`), §2.1/§2.2/§2.3 against the live documents, **nothing outside
`gui/` needs Qt** (a subprocess imports all 27 non-GUI modules with `PySide6` blocked), the three
doors produce the same document (`image.source` is the only difference, and it is the documented
provenance marker), determinism, the taxonomy in four places, loopback-and-no-auth defaults, the
MCP doc against the catalog, README commands against the parser, extras and the Python floor, the
CJK font probe, no URL in a default test, no secret in the tree, generated docs current, floors
held.

### What the QA pass found (both real, both fixed)

1. **`mcp.host`/`port`/`transport`/`enabled`/`auth_token` do not take effect at runtime, and
   nothing said so** (D78). PLAN §2.2 promised a server-thread restart that was never built;
   `StudioServer` is constructed once from `config.mcp` and the gateway captures the bind address
   and bearer token at that moment. A client that patched `mcp.port` got a stored value, an
   unchanged socket and no signal — the worst kind of defect, where the truth is available and
   everyone is politely not looking. Fixed by telling the truth everywhere rather than by
   building the risky thing: `RESTART_REQUIRED_KEYS` + `restart_required()` in `core/config.py`,
   the field in the `set_config`/`import_config` replies (schemas + regenerated catalog + tool
   descriptions), the window says it on import, and README/docs/MCP.md name the five keys. The
   restart itself stays unbuilt on purpose: a client that could move the port it is connected to
   could orphan itself mid-session, and "report, don't act" is the line D58/D60 already drew.
   11 tests in `tests/test_p8_hardening.py::TestRestartRequired`, including the GUI one and the
   "ordinary keys must not warn" control.
2. **`docs/MCP.md` still advertised the config document as v1.1** — it has been 1.2 since 8.2.
   Fixed, and the reason D79 exists: a hand-maintained doc states the version it was written in.

Checked and *not* defective, worth recording because they were plausible suspects: `examples/
config.default.json` is byte-equal to `AppConfig.default()`; every PLAN §2.1/§2.2 key is present
in the live documents; the MCP doc names all 20 tools and no invented ones; `gen_error_matrix.py
--check` is clean; the live MCP door answers all 20 tools and 6 resources with progress
notifications and a schema-valid save; `--gui-smoke` exits 0; no HF-token lookalike anywhere in
the tree outside `HF_token.txt`, which is gitignored.

### Measured, at sign-off

`QT_QPA_PLATFORM=offscreen python3 -m pytest -q` → **1092 passed, 41 deselected** (~37 s).
`python3 -m pytest -q -m models` → **40 passed, 1 skipped** (~40 s; the skip is the one
Hub-listing test). `scripts/fetch_models.py --verify` → 33 files, `RESULT: usable offline`.
`gen_error_matrix.py --check` and `gen_qa_report.py --check` → both current.
Live session against `--mock --headless --serve` → 20 tools published, 6 resources, 8 progress
notifications on a run, `save_result` 2486 bytes, `reset` cleared, harness `OK` exit 0.

### Deliberately not done

* No server restart on a config change (D78), and no `restart` *action* over MCP either.
* No per-file SPDX headers: the `LICENSE` file governs, and ASF-style boilerplate is not ours to
  paste onto a non-ASF work.
* No `MANIFEST.in`, no `py.typed`, no `requires-python` ceiling — unchanged from 8.7.
* The three open follow-ups (`reocr_blocks`, bundled-open retry, the fallback badge) stay open;
  they are not requirements and P8 closes without them.

**Suite:** 1092 passed, 41 deselected · `-m models` 40 passed, 1 skipped · both generated docs
current. **Parts P1–P8 are complete**; `docs/QA.md` is the traceable proof and `## Next up` above
is what, if anything, comes next.

---

## Turn 20 — 2026-11-26 — P9.6: the trail, and the two defects a sign-off run walks into

Subplan **9.6** — "the trail: catalog, matrix, QA table, README, floors, sign-off". The last
subpart of P9, and the one that decides whether the previous five are discoverable a month from
now. It also paid for itself: running the doors the way a user would turned up two real defects.

### D93 — an option that was given `""` was treated as an option that was not given

`main()` dispatched its string-valued actions on truthiness:

```python
if getattr(args, "warm_devices", None):     # "" is falsy → the action never runs
```

So `--warm-devices ""` fell through to the default action and **opened a window**. Same for
`--export-config ""`, `--validate-config ""` and `--dump-mcp-tools ""`; `--out ""` was quieter and
worse — it wrote nothing and exited 0. D86 had promised *an empty device list is refused*; the
core function kept that promise and the door never called it. The sign-off command
`python3 -m animetext_studio --warm-devices ""` hung for the length of its timeout because the
thing it did was start Qt.

The fix is one guard before any dispatch, over the four options that decide an action or a
destination (`--warm-devices`, `--export-config`, `--validate-config`, `--out`, plus
`--dump-mcp-tools`): an empty value is `INVALID_ARGUMENT`, exit 2, `details.option` naming the
flag. `--config ""` stays "no config file" — a blank *source* genuinely means unset, the same
answer a blank environment variable gives (D30), and that exception is pinned by a test so nobody
"fixes" it by analogy. `--mcp-token ""` still legitimately clears the token.

Two matrix rows (`--warm-devices with no devices`, `--export-config with an empty path`) plus
seven tests in `TestEmptyOptionValues`, each of which monkeypatches `run_gui` to fail the test if
the window is reached. Before the fix six of the seven failed; four of them failed by opening a
window.

### D94 — `--dump-mcp-tools PATH` did not write to PATH

```python
path = write_catalog(None if target == "-" else target)   # write_catalog(service, path)
```

The path bound to `service`. Every invocation — including the one a maintainer runs to *check*
the catalog — rewrote `animetext_studio/schemas/mcp-tools.json`. And the test that "wrote a copy
to `tmp_path`" had the identical mistake, so it regenerated the shipped file, read it back and
compared it with itself: green, while repairing the exact drift
`test_catalog_is_the_file_that_ships_in_the_wheel` exists to catch. A check that fixes its own
subject is not a check.

`write_catalog(path=…)` is honoured now; the test writes to `tmp_path/nested/tools.json`, asserts
the copy is the shipped catalog byte for byte, and asserts the shipped file's bytes did not move.
A CLI-door test does the same through `main()`.

### The rest of the trail

* **Matrix 84 → 91.** Seven rows, each with a case the sweeps actually run: four re-OCR refusals
  at the service door (the `NO_IMAGE` / `INVALID_ARGUMENT` split between "no page" and "no
  document" is D81, now written where someone reading the matrix will find it), the corrupt
  bundled file at the offline door, and the two D93 refusals at the CLI. `REOCR_CASES` and
  `OFFLINE_CASES` joined `RUNNABLE`, so the three invariants — no bare exception, the promised
  code, the app still works afterwards — cover the new doors too.
* **QA table 115 → 131.** A `P9` group of sixteen rows; `FOLLOW_UPS` rewritten (the four items P9
  shipped are off the open list; block colours and the language policy stay; "drafted, not filed"
  and "no `bench_models.py`" were added); floors moved to measured numbers.
* **README.** Fifth badge documented; right-click → **Re-read this block** is now the "what to do"
  for `⚠`; **Refresh models** in Controls; `stage` documented as three values, not two; a
  Troubleshooting row for a bundled file that is present and corrupt; every count re-measured.
* **docs/MCP.md** finally documents `detail` / `model_list_is_fallback` (9.2's row had never been
  written). **docs/UPSTREAM.md** cited a test class that has never existed; it cites the real one.
* **The demo client** drives `reocr_blocks` now, so the worked example shows the newest door.

### Measured at sign-off

| | |
|---|---|
| offline suite | **1253 passed, 41 deselected**, ~41 s |
| `-m models` | **40 passed, 1 skipped**, offline, ~50 s |
| `fetch_models.py --verify` | 33 files hash-clean — *usable offline* |
| generators | `gen_error_matrix --check`, `gen_qa_report --check` both clean |
| `--doctor` | 15 checks: 14 ok, 1 warn (the known `numpy <2` pin), 0 fail — exit 0 |
| `--warm-devices cpu --mock` | two rows, exit 0 |
| live MCP session, real weights | 21 tools published / 21 in the catalog; the re-OCR step: **84 ms for one block against the page's 949 ms**, boxes unchanged, exit 0 |

**What I got wrong and corrected on the way:** `P9.md` claimed the device-sweep tests lived in
`tests/test_p9_warmup.py` — there is no such file (they are in `test_p8_lifecycle.py::TestDeviceSweep`);
corrected. The lesson is the same one D79 taught: a citation nobody resolves is a citation that
will rot, and the trail subpart is where rot is supposed to be caught — so the resolver should be
a test, not a memory.

**Next up:** nothing scheduled. Open by choice: file the two upstream issues (D92), and
`scripts/bench_models.py` (ADDENDUM item E) now has its measurement primitive in
`--warm-devices`.

---

## Turn 21 — 2026-11-26 — the repository, and the boundary D26 never drew

The account arrived: `paoloinverse`, and a public placeholder repository containing one commit
with a `LICENSE` whose copyright line GitHub's picker had left as `[yyyy] [name]`. The history
starts from that commit, so the picker's work is the parent of ours rather than something
rebased over, and the second commit fills the line in.

**The boundary (D95, amending D26).** `git add -A` on this folder would have published 1.3 GB
of weights, 245 MB of the MCP inspector's `node_modules`, and seven of the user's own scans.
Three of those are three different arguments, so they are three rules and not one:

* **the AnimeText weights are not ours to distribute.** They are GPL-3.0, and the question was
  never whether the licence permits a re-publication but whether its author intends one. It
  does not, so the answer is no before any legal analysis starts.
* **the sizes are not a preference.** `model.safetensors` is 604 MB and three more files sit
  over GitHub's 100 MB line: the push is *rejected*, not discouraged, so a rule here describes
  physics rather than taste.
* **the scans are nobody's to license but the owner's** — which README and NOTICE had been
  saying for eight turns while `.gitignore` said nothing about them.

D26 is amended, not deleted: a folder copy is still a complete offline installation, which is
what D26 was about. A *clone* is not a folder copy, and `scripts/fetch_models.py` — which
already rebuilt a damaged bundle — is now also the first command a clone runs. README, NOTICE
and the layout table say so; `TestWhatTheRepositoryPublishes` pins it by reading
`git ls-files`, because the published set is what matters and the folder is not it.

**What the sweep turned up.**

* `examples/mcp_session_demo.json` and `examples/mcp_raw_transcript.txt` recorded the machine
  they were recorded on: 45 absolute paths including the local account name. Now
  `/home/you/animetext-studio`, which reads the same everywhere and is obviously a placeholder.
  Pinned by `test_no_tracked_text_carries_a_machine_path`, so re-recording a session cannot
  publish a home directory again.
* The secret scan knew `hf_…` and nothing else. It knows `ghp_…` / `github_pat_…` now, skips
  both token files, and asserts both are in `.gitignore` — the rule that had been leaning on
  `*token*.txt` was case-sensitive, so `GITHUB_TOKEN.txt` would have walked straight through it.
* `README` said "1092 passed" when the suite was 1253, and the checklist count moved with the
  new `PUB-1` row: 131 → 132, floors 1253 → 1258, `docs/QA.md` regenerated.
* **A clone does not exist yet, so one was simulated**: `git archive HEAD` into an empty
  folder — 99 files, 2.2 MB, no `models/`, no `sample_scans/` — and the suite run inside it.
  Three tests failed, all of them D26 wearing a test coat:
  `test_the_ocr_provenance_file_ships`, `test_the_schemas_and_the_manifest_ship` and
  `test_the_gpl_weights_ship_their_own_text` each expanded `package-data` against the
  filesystem and asserted a bundle file was in the expansion — true of a built folder, false of
  a clone, and never written down as either. They now check the **declaration** always (the
  `models/*/.origin.json` line, the manifest pattern, `models/*/*.txt` for the GPL text — the
  three things a wheel would silently drop) and the **expansion** only when a bundle is there
  to expand, skipping with the fetch command in the message. A clone is green; a built folder
  is still checked file by file. The simulation was then made real — `git init` inside the
  archived folder, so `git ls-files` has something to answer — and the five publication checks
  ran and passed there: the published set has no weight, no scan, nothing over 100 MB and no
  machine path. Verified, not assumed.
* `docs/UPSTREAM.md` gained the thing a draft rots without: **upstream already fixes Issue 2**,
  in PR #178, opened 2026-06-07 and unmerged, which rewrites exactly the two `isinstance` lines
  and relaxes `numpy<2` while it is there. Issue 2 is therefore a release request, not a bug
  report — and a fine-grained token cannot file either way, because only a classic token may
  write to a repository that is not yours. That is now written where someone will find it.

**What I got wrong and corrected on the way.** `LICENSE` opens with a blank line, which looked
like a stray from a text editor, and I deleted it. `test_the_apache_text_is_the_canonical_one`
went red: the blank line is in the canonical apache.org text, and the pinned sha256 — which I
could now check against the live URL rather than against itself — proves it. The byte went back.
A pin that only ever agreed with the file it pins is a rubber stamp; this one disagreed with me
because somebody had hashed the real thing.

**Measured, this run.** 1258 passed / 41 deselected offline (~41 s) · both generators
`--check` clean · **the simulated clone: 1250 passed, 8 skipped, 39 deselected, 0 failed**
(3 skips are the bundle checks; the other 5 are the publication checks, which `git archive`
leaves nothing to read — with a `git init` run inside the clone they execute and pass,
45 passed / 3 skipped in `test_p8_packaging.py`; 39 deselected rather than 41 because the two
real-page cases have no scans to parametrise over. Every skip names the command that would
make it run) · 100 files committed, 1.9 MB, largest 0.14 MB ·
nothing over 100 MB, no weight, no scan, no machine path in the tracked set.

**The push, and what it took.** The first token reached GitHub as the right account and was
still refused — **403, "Permission to paoloinverse/animetext-yolo-MCP.git denied to
paoloinverse"** — because *pull requests read/write* is exactly that: it opens pull requests,
it does not move commits. A second token, with `contents:write`, pushed `424eb27..e95e8be` in
one go, and the remote was then read back rather than trusted: an **anonymous clone** of the
public repository gives 99 tracked files, 2.2 MB, zero matches for `models/`, `sample_scans/`,
`.onnx`, `.safetensors`, `.pt` or `token`, the filled-in copyright line in `LICENSE`, and
`test_p8_packaging.py` green against a real published set (45 passed, 3 skipped — the three
bundle checks, which a clone cannot run and says so about).

Naming the repository then became the last piece of metadata that was still missing: a wheel
that cannot say where it came from is the provenance question asked again three months later.
`[project.urls]` now carries Homepage / Repository / Documentation / Issues, the README names
the same place in its first screenful, and one more test holds the two together — parsed with
`urlsplit`, because the offline-suite rule forbids a URL literal in a test file and this test's
subject *is* a URL. Suite: **1259**.

**Next up:** the two upstream issues (D92 — Issue 2 as a release request against #178, and
only a *classic* token can file into someone else's repository); `scripts/bench_models.py`
(ADDENDUM item E), still waiting on its corpus, its two revisions and its verdict metric.

---

## Turn 22 — 2026-11-26 — the README read as a manual, and Python 3.14 measured rather than assumed

**The README was addressed to the wrong reader.** The user's words: *"none of my users are
interested in what decision D95 is."* Right — and the project's own rule about the trail (D79:
a citation nobody resolves is a citation that rots) cuts the other way for a manual: an id that
points at an internal log is noise to the person trying to install the app. Thirteen decision
ids were in the README; **twelve are gone**, and the one that stays is in the row that describes
the build log, where the id is the subject rather than the residue. The reasoning is not lost —
it is in `P9.md` and `tracking.md`, which is where a reader who wants it will look.

**The clone command was broken, and nobody had run it.** `git clone <this repo> && cd
animetext-studio`: a placeholder nobody filled in, and a folder name the clone never produces —
the repository is `animetext-yolo-MCP`, the package is `animetext_studio`, so the `cd` would
have failed on the first command anyone types into a terminal. Both fixed, and the reason this
survived eight turns of README review is the reason it needs a test: the README is documentation,
so no test ever executed it.

**`test_the_readme_reads_as_a_manual_not_a_worksheet`.** Decision ids outside the build-log row,
`<this repo>`-class placeholders and `TODO`/`FIXME` are all failures, and the clone line must
carry a real URL and the folder the clone actually makes. Angle-bracket *syntax* is allowed —
`<package>`, `<variant>`, `<token>` are how a command is written down — the rule is about
residue, not typography. Suite: **1260**.

**Python 3.14, examined instead of accepted.** The README's 3.10–3.13 floor had become a fact
people repeat; the user asked whether there is truly no alternative. Measured today, in
`enhancements_proposition.md` → *Second round*:

* `dghs-imgutils` 0.19.0 pins `numpy<2` and declares no `requires-python` at all; **numpy 1.x
  ships no wheels past cp312**, so the pin is the harder blocker, not the `ast` one.
* **We already run imgutils on numpy 2.5.5** — installed `--no-deps`, which is the recipe this
  project's own Troubleshooting table gives. The pin has been fiction in the only installation
  that has ever worked here.
* The 3.14 breakage is **two lines in one file** (`imgutils/generic/yolo.py:468,474`); a grep of
  the installed package finds `ast.Str`/`ast.Num` nowhere else. Upstream's unmerged PR #178 fixes
  both lines *and* relaxes the numpy pin.
* Our whole dependency on imgutils is **five symbols**, and the one that fails is
  `_safe_eval_names_str` — where `engines/detection.py` already keeps an `ast.literal_eval`
  fallback that can never fire on 3.14, because the import succeeds and the failure happens
  later, inside upstream's own model load.

Four options are written up with costs: wait for a release (no cost, no control); **ship our own
eight-line shim** (a day, unblocks 3.14 against the `--no-deps` install we already document,
reversible, pinned by the test that already reads the installed source); **stop depending on
imgutils for detection** (we own the ONNX path already; it would settle 3.14 *and* the
import-time `pip install`, at the price of owning letterbox/NMS and re-baselining the goldens);
or pin a fork (rejected on provenance grounds — this project's one careful claim is provenance).
One trap recorded so nobody re-invents it: setting `ast.Str = ast.Constant` on import mutates the
standard library for every other importer and quietly accepts `None`/`True` literals the old pair
rejected. Patch the function, not `ast`.

**A classic token arrived.** It authenticates as `paoloinverse` (verified with one read-only
`GET /user`) and it is the shape that *can* write to `deepghs/imgutils`. It was used for that one
call and nothing else, and nothing was filed: an issue under someone else's name, in someone
else's repository, is theirs to press — especially Issue 2, whose right form depends on the #178
answer the user has not given yet. Stored in `GH_token.txt` (mode 600, gitignored), rotation
recommended.

**Measured, this run.** 1260 passed / 41 deselected offline · both generators `--check` clean ·
`docs/QA.md` regenerated against the 1260 floor.

**Next up:** the three answers `scripts/bench_models.py` needs (corpus, revisions, metric); the
two upstream issues (D92); and, if the 3.14 examination is taken up, the shim as a P10 subpart.
