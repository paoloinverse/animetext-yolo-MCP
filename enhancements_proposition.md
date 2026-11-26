# Enhancement propositions — five things this app could do next

> **Status (2026-11-25): all five were approved and have shipped**, as P9.1–P9.5 in
> [`P9.md`](P9.md) — 2 → 9.1, 3 → 9.2, 1 → 9.3, 4 → 9.4, 5 → 9.5.
> **P9.6 closed the trail on 2026-11-26**: the catalog, the error matrix (84 → 91), the QA
> checklist (115 → 131), the README, `docs/MCP.md`, the floors and a measured sign-off — and
> running the doors for real turned up two defects on the way (D93: an option given `""`
> opened the window instead of refusing; D94: `--dump-mcp-tools PATH` ignored PATH and rewrote
> the catalog that ships in the wheel). The file is kept as written
> because each entry still carries the analysis behind the decision; where the shipped thing
> differs from the proposal (9.4's `--warm-devices` action instead of a `devices=` option, D85;
> the doctor's exit code predicting the run's, D89) the decision table in `P9.md` says so.

Written 2026-11-25, **after** P1–P8 closed (T19). Nothing here is scheduled, committed, or a
defect: the plan is complete and `docs/QA.md` traces every requirement to evidence (115 rows,
1092 offline tests + 40 `-m models`). These are the five ideas that came up while closing it and
were deliberately left out — each one is written up here so it can be picked up, changed or
rejected **without re-deriving the analysis**.

Authority: [`PLAN.md`](PLAN.md) (requirements) · [`tracking.md`](tracking.md) (decisions D1–D79,
turn log, and its `## Next up` list — the five below are those five, in that order) ·
[`docs/QA.md`](docs/QA.md) (*Still open, on purpose*, generated from `FOLLOW_UPS` in
`tests/test_p8_qa.py`) · [`ADDENDUM.md`](ADDENDUM.md) (post-plan ideas already evaluated; A/B/C
closed as out of scope, D shipped as 8.2b, E still open).

Four of the five are also rows in `docs/QA.md`; the fifth (upstream) is not, because it is not
our code — and `docs/QA.md`'s sixth row, block colours, is not repeated here as an enhancement
because it was never a defect. Both are in the closing table below. This file **does not** change
any of those documents: adopting an item is what changes them, and each section names the trail.

## Summary

| # | Enhancement | In one line | Cost | Risk | Worth doing when |
|---|---|---|---|---|---|
| 1 | `reocr_blocks()` | re-read one block without re-detecting the page | M | medium — it is the first *mutating* door | someone actually asks for it |
| 2 | Bundled-open retry | a truncated `model.onnx` today fails as if it were absent | S | low | any time; it is a lie we currently tell |
| 3 | Fallback badge | say out loud that the model list came from presets, not the repo | S | low | any time; half-built already |
| 4 | 8.3b — per-device warm-up latency | *which* device is warm, and how long the first inference took | M | low | when a GPU is in the loop |
| 5 | Upstream: `dghs-imgutils` + a doctor | two upstream behaviours break promises this app makes | S (ours) | low | before the next dependency bump |

S ≈ under an hour of work plus tests · M ≈ a day, including the documentation trail each one
owes: the matrix in `docs/ERROR-MATRIX.md` is generated from `MATRIX` in `tests/test_p8_errors.py`
and `docs/QA.md` from the requirement table in `tests/test_p8_qa.py` (both with `--check` for CI),
`schemas/mcp-tools.json` is regenerated from the live server, and README and `docs/MCP.md` are
hand-written but pinned against all three by `tests/test_p8_packaging.py` and
`test_docs_mcp_md_matches_the_catalog`. The trail is not optional — that is what makes an
enhancement here cheap to check and expensive to forget.

---

## 1. `reocr_blocks()` — re-read one block without re-detecting

**What.** Today every OCR of one box costs a whole page run: detect (≈2.0 s CPU, `yolo12m`),
crop, read every kept box. A caller who sees `flags: ["possible_truncation"]` on block 7 and
wants that one block re-read at `max_new_tokens: 256` has no way to ask for it.

**Why the machinery is already there.** `OcrEngine.ocr_crops()` (`engines/ocr.py:311`) takes an
arbitrary list of crops — the pipeline just happens to hand it *all* of them
(`Pipeline.run` → `_recognise`, `core/pipeline.py:261`/`367`). `get_result` already accepts
`block_ids` and already validates them, and `annotate_blocks()` (`core/pipeline.py:488`) is a
pure function that recomputes `flags` + `contained_in` from blocks alone. So the engine work is
zero; this is an entry point, not a feature.

**Shape.** `Pipeline.reocr_blocks(image, blocks, ids, *, overrides) -> PipelineResult`: crop the
selected boxes, OCR just those, splice the texts back onto the *existing* blocks (ids, colours
and reading order unchanged — D32), re-run `annotate_blocks`, re-time. Job kind `reocr` beside
`run` and `warmup` (`core/jobs.py`), so single-flight, progress and cancellation come free.
In the window: a right-click on a row in the block table (`block_tree` has selection wired,
no context menu).

**Blast radius — this is the expensive part, and it is paperwork.** A new MCP tool is the 21st,
and the count is *asserted* end to end: `tests/test_mcp_http.py:87` (`len(tools.tools) == 20`),
`tests/test_p8_qa.py:660` against `SIGNOFF["tools"]` (whose failure message literally reads
"update SIGNOFF and the README with a reason"), `docs/MCP.md` §2's prose, and README's tool list —
all of which move together, or the suite goes red. Plus: regenerate `schemas/mcp-tools.json`
(`--dump-mcp-tools`), give the tool its row in `docs/ERROR-MATRIX.md`'s per-tool code table, and
add its faults to `MATRIX` so the error-path suite covers the new door.
That trail is the reason it waited: a 21st tool needs a user before it needs a schema.

**Open question for you.** Does a re-OCR produce a **new document** (new `created_at`, new
`timing_ms`, everything else identical — honest, and the determinism tests keep passing) or a
**patch** (smaller, but then `result://latest` is no longer a whole document)? The first is
simpler and keeps the contract "one document describes one page"; the second is what an agent
loop probably wants. Pick one before building.

---

## 2. Bundled-open retry — a corrupt file is currently reported as a missing file

**What.** "Bundled" is decided by size, and the size asked for depends on who is asking.
`bundled_detection_variants()` wants ≥ 1 MB (`_MIN_WEIGHTS_BYTES`, `core/model_store.py:72`),
while `detection_file()` — the one `load()` and `detection_variant_status()` actually use — calls
a file bundled if it **exists and is non-empty** (`_file_ok(path)` with its default `minimum=1`,
`core/model_store.py:245`/`297`). So a copy that was interrupted — disk full, half a
`git clone`, a truncated USB transfer — reports `"bundled": true`, and the failure surfaces at
ONNX session-open time
(`engines/detection.py:321`/`646`) as `ENGINE_LOAD_FAILED` (exit 8) with an ONNX parse error that
mentions neither the bundle nor the fix.

**Why it is worth its hour.** Everything needed to say the truth already exists:
`MANIFEST.json` carries **sha256 for all 33 vendored files**, `ModelStore.sha256_file()` is a
streaming hasher, and `scripts/fetch_models.py --verify` already does the comparison and prints
`RESULT: usable offline`. `load()` is simply the one place that never consults any of it.

**Shape.** In `load()`'s failure path: hash the file we just failed to open, compare with the
manifest entry. Match → the weights are fine and the fault is elsewhere (ONNX build, provider):
say that. Mismatch → `ENGINE_LOAD_FAILED` with `details.expected`/`details.got`, the offending
path, and `steps: ["python scripts/fetch_models.py --verify", "… --variants <name>"]`. Only
*then*, and only when `paths.offline` is false, retry once via the Hub and record the recovery in
`status().detail` — the same pattern D27 already uses for a stalled listing: do the useful thing,
and say that you did it.

**Risk.** Hashing is I/O on a cold path only (the failure path), never on the hot status path —
which is exactly what `sha256_file`'s own docstring warns about. Offline mode is untouched: no
retry, better message.

---

## 3. `model_list_is_fallback` — a badge, and the MCP door that never got the news

**What.** The Hub listing is bounded by 10 s (D27) and skipped entirely offline; past that the
engine answers from the bundle plus the compiled presets. That is the right behaviour and it is
*recorded* (`model_list_is_fallback`, `engines/detection.py:613`, with the reason in
`status().detail`) — but only the GUI's disabled combo items show it (D49,
`gui/controls.py:283`–`310`). Nothing shouts.

**The gap nobody has mentioned.** `StudioService._engine_info()` (`mcp/service.py:386`) copies
`backend`/`device`/`state`/`loaded`/`models`/`inferences`/`warm` — and **not `detail`**. So the
reason the list degraded never reaches the MCP door at all: a remote agent cannot tell "the repo
has five variants" from "the repo did not answer and these are the presets we shipped". That is
the kind of silent asymmetry D78 exists to prevent.

**Shape.** (a) Put `detail` (and a boolean `model_list_is_fallback`) in `_engine_info` +
`EngineModel` — additive, so `extra="forbid"` keeps biting but nothing breaks; regenerate the
catalog; add the row to `docs/MCP.md`. (b) A fifth badge in `ControlPanel._badges()`
(`gui/controls.py:84`–`97`, next to MOCK / OFFLINE / MCP / engines) reading `models: presets`,
shown only when the flag is set, with `detail` in its tooltip — `engines_badge_text()` is already
the Qt-free, testable pattern to copy. (c) *Optional:* a "Refresh model list" action that does
what `_startup_catalog()` deliberately does not do (`gui/main_window.py:172` — no network at
startup, D27) — off the startup path, bounded by the same 10 s, and reported.

**Risk.** Low. The one thing to keep: startup must stay network-free, so a live refresh is a
user action, not a constructor side effect.

---

## 4. 8.3b — per-device first-inference latency in `warmup_report`, and `--preload-off`

**What.** Spun off 8.3 on purpose. `WarmupOutcome` already measures `load_ms`, `probe_ms`,
`device` per engine (`core/warmup.py:79`), and `--preload` exists for the GUI, headless and MCP
(D61). What it cannot answer is the question a GPU user actually has: *is CUDA warm, and what
did the first inference cost on this device?* — because warm-up warms whatever `ocr.device`
resolved to, once.

**Shape.** A `devices: ["cpu","cuda"]` option on the warm-up job that loads per device and
reports one `WarmupOutcome` per (engine, device), so `warmup_report` becomes a small table
instead of a line. `--preload-off` then gives a benchmark a cold baseline to compare against
(today `--preload` is opt-in, so "off" is already the default — the flag earns its keep the day
something makes warm-up automatic, and it is the knob a harness needs to name the baseline).
Its real home is **ADDENDUM item E**, the one idea there still open: a `scripts/bench_models.py`
that runs a fixed corpus over two revisions/variants and prints a per-block diff, with
`--ocr-revision` to pin a Hayai commit and a JSON baseline. Hayai revisions arrive every few
days, so "did the update help?" is the question this app gets asked most and the one it has no
answer for.

**Cost note.** `warmup_report` lives on `JobManager` (`core/jobs.py:117`): `--preload` prints its
one-line `summary()` to stderr (`__main__.py:116`) and the GUI receives it (`gui/workers.py:55`),
but it is not a field on any MCP answer. Exposing it is a decision, not an oversight:
a per-device latency table is a fine thing for an agent to read, but it is also machine-specific
noise in a document-shaped API.

---

## 5. Upstream `dghs-imgutils`, and the doctor that would have caught it

**What.** Two behaviours of the detection dependency (`dghs-imgutils` 0.19.0 — still the latest
on PyPI, checked 2026-11-25) contradict promises this app makes:

| Upstream behaviour | Where | Which promise it bends |
|---|---|---|
| `imgutils.utils.onnxruntime._ensure_onnxruntime()` **`pip install`s at import** (`hbutils.system.pip_install`) when onnxruntime is absent | `imgutils/utils/onnxruntime.py:17`–`30` | "no surprise network use" — a first import can reach the index in a machine that was told to be offline (README Troubleshooting, `README.md:518`) |
| `_safe_eval_names_str` uses `ast.Str`/`ast.Num`, removed in Python 3.14 | `imgutils/generic/yolo.py:468`,`474` | the Python floor: detection works on 3.10–3.13, not 3.14 (`README.md:516`) |
| (and its `numpy<2` pin, no cp313 wheels) | `METADATA: Requires-Dist: numpy <2` | the documented `--no-deps` install dance (`README.md:81`–`92`) |

**Our half is small, and it is not "wait for upstream".** A `--doctor` (or `--check-env`)
command that answers, in JSON and in plain text, before anyone presses Generate: Python version
vs the imgutils floor, whether `onnxruntime` imports **without** importing imgutils, numpy
version vs the pin, torch/CUDA presence, `fetch_models.py --verify` result, HF token discovery
outcome, CJK font present. Both hooks already exist — `ModelStore.status()`'s docstring names
"doctor" as a consumer (`core/model_store.py:411`) and `engines_badge_text()` says it is shared
with "any future doctor view" (`engines/base.py:81`); `scripts/check_hf_access.py` is the same
shape for the token half and can be reused rather than duplicated. It turns two upstream quirks
into one sentence the user can act on, and it gives the README's Troubleshooting table a command
instead of a diagnosis.

**Upstream half.** Two short issues on `deepghs/imgutils`: (i) an opt-out for the import-time
install (`IMGUTILS_NO_AUTO_INSTALL=1`, or just raise with the pip command), (ii) the
`ast.Constant` migration. Neither is our code to fix; both are worth filing, with the file:line
above, while we still remember where they are.

---

## Not on this list (decided, so they are not re-proposed)

| Idea | Where it was closed |
|---|---|
| Language / script policy, sub-crop decomposition, nested-box de-duplication | **out of scope** — D58/D60: the app reports, the caller decides. Evidence stays in `ADDENDUM.md` §1 for whoever writes the caller |
| Block colours that differ between two runs of the same page | identity, not correctness; chips and table key on `id`, which is stable (D32) |
| Restarting the MCP server thread on a config change | refused at 8.8 (D78): a client must not be able to move the socket it is talking to. The five `RESTART_REQUIRED_KEYS` are *reported* instead |
| OCR the full page without cropping, a language prompt, beam-search knobs | impossible or counter-indicated by the model card — `ADDENDUM.md` §F, "Things explicitly **not** to do" |

## How to review this file

For each of the five, one of: **adopt** (→ a `P9.md` subpart with an exit demo), **adopt
narrowly** (say which half), **park** (stays in `FOLLOW_UPS`, no change), or **reject** (say why,
and it moves into the table above so the next session does not resurface it).

The two I would take first are **2** and **3**: both are small, both are about the app telling the
truth rather than doing more, and neither needs a decision from anyone but you. **1** needs the
new-document-or-patch answer above. **4** needs a GPU to be worth doing. **5**'s doctor is the
only one that adds surface area, and it is the one that pays off the next time a dependency moves
underneath us — which, given Hayai's cadence, is not a long way off.
