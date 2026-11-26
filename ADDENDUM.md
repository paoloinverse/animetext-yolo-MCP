# ADDENDUM — evaluated post-plan improvements (TBD, not scheduled)

Written 2026-11-22, **after** P8 was defined. Nothing here is committed work: each item carries
a verdict (adopt / adopt-narrowly / needs-a-pilot / reject) and the evidence behind it, so a
later session can pick one up without re-deriving the analysis.

**Decisions taken the same day (user), and they are final:**

| Item | Status | Where it went |
|---|---|---|
| A — language / script policy | **out of scope for this app** | the caller (user or the agent driving the MCP) decides what languages are expected and what to do about surprises. The §1 evidence stays here as reference for that caller |
| B — sub-crop decomposition | **out of scope for this app** | same reason; a caller that wants line-level crops asks for them and reasons over the undeduplicated list itself |
| C — dedup / topology | out of scope (unchanged) | provenance instead — and `contained_in` now supplies the topology edge |
| D — cheap correctness flags | **shipped** | folded into P8 as **subpart 8.2b** (result schema 1.2): `flags`, `contained_in`, and the missing 512 patch budget |
| E — bench harness | open | not scheduled yet; still recommended |

What this means in practice: A and B are **not** to be re-proposed as app features. Their
evidence (§1) is now input for consumers of the result document, together with the flags 8.2b
emits — the app's job is to hand over enough truth (geometry, per-block provenance, warnings)
for a caller to make those calls, not to make them.

Authority: [`PLAN.md`](PLAN.md) (requirements) · [`tracking.md`](tracking.md) (decisions D1…D79,
turn log) · [`docs/QA.md`](docs/QA.md) (what was verified, row by row). This file **does not**
change any of them; items A/B/C are closed as out of scope (D60), D shipped as 8.2b, and E (the
benchmark harness) is the one idea still waiting.

---

## 1. Evidence base

### 1.1 What the model card says (vendored `models/hayai-ocr-v2.7-supernova/README.md`, rev `b00317ed`)

| Card statement | Consequence for this app |
|---|---|
| "**Crop-level input only:** Passing full comic pages without cropping text regions will not work." | Any "OCR the whole page" idea is out — the detector→crop pipeline is not an optimisation, it is a requirement of the model |
| "**Multilingual domain gap:** Chinese, Korean, and English recognition is trained primarily on synthetic data, so accuracy on noisy real-world non-Japanese crops may vary." | Japanese output is trustworthy; `zh`/`ko`/`en` output is a *hypothesis*. Language is therefore a confidence signal, not metadata |
| "**Short-text sensitivity:** Many manga crops contain only 1 to 4 characters, where a single misread glyph drops Exact Match immediately." | Tiny crops are noisy; do not judge the pipeline by per-block accuracy on short bubbles |
| "Keep `repetition_penalty=1.0`" + "Use `max_patches="auto"` for mixed workloads" | Do not add a repetition knob; `auto` stays the default budget |
| Stage 4 = "Long-Text Edgecase Fine-Tuning and WiSE-FT Merge" | The rev we vendored (`b00317ed`) is the one aimed at long/dense crops — the exact case below |

### 1.2 What the code actually offers (read from the vendored snapshot, not the card)

* Patch budget tiers are **128 / 192 / 256 / 384 / 512** (`processing_hayai._select_best_max_patches`),
  with `min_patches=64`, `patch_size=16`, `max_scale=4.0` (`preprocessor_config.json`).
* `model.ocr(..., escalate=False)`: escalation re-runs a crop at 384 (or 512) when the first pass
  returned dense text (`check_escalation`). **It only fires inside the batched `auto` path**
  (`modeling_hayai.ocr` → `plan_batches`), and this engine calls with `batch_size=None` so it
  never fires today.
* **Gap found:** `BaseOcrEngine.MAX_PATCH_CHOICES = (128, 192, 256, 384)` — the engine *refuses*
  512, the one tier the upstream escalation targets. One-line fix, untested against quality.
* `generate()` accepts `num_beams` / `repetition_penalty`, but `ocr()` does **not** forward them.
  Exposing them would mean forking the remote code — not worth it (and the card forbids the
  penalty anyway).
* There is **no language argument anywhere** in `ocr()`/`generate()`/the processor. The model is
  not promptable for language; the only levers are the patch budget, `max_new_tokens`, crop
  geometry and crop padding.

### 1.3 Measurements (2026-11-22, CPU, `examples/sample_page.png`, rev `b00317ed`)

| Run | Blocks | detect | ocr |
|---|---|---|---|
| `yolo12m` @ `f5f0f617` (T7) | 12 | 2289 ms | 4580 ms |
| `yolo12m` @ `b00317ed` | 12 (identical text) | 2008 ms | 2031 ms |
| `yolo12x` @ `b00317ed` | 13 | 2566 ms | 2358 ms |

Two failure modes, both observed and both relevant here:

1. **Run-on slab.** The narration block `(590,335)-(819,722)` (229×387 px, four vertical columns)
   OCRs as `決戦の前夜風が止んだ第4話その日、空は赤かった` — correct characters, no separators.
   * `max_patches` 256 → 384: **byte-identical output**. This is not a resolution problem.
   * Ink-gap segmentation found **no** blank column (a panel background makes every column dirty).
   * A naive **5-way equal vertical split** recovered the columns: `その日、空は赤かった` / `第4話`
     / `馬` (a hallucination on a nearly-empty strip) / `風が止んだ` / `決戦の前夜` — 5 crops in
     271 ms vs 185 ms for the slab.
2. **Nested-box fragment.** `yolo12x` also emits a box *inside* another box; the nested crop OCR'd
   as `った` — a tail of a line whose rest is in the parent. Detector geometry, not decoding.

---

## 2. Items

### A. Language policy: flag unexpected scripts, re-run only where a lever exists

**Proposal (user's).** Let the user/MCP declare expected languages (default `ja`); when something
else appears, run again with more suitable parameters.

**Verdict (2026-11-22): out of scope for this app — the caller does this.** My earlier
"adopt narrowly" asked for script flags in the document; the user's call is that even that
discrimination belongs above the app, and it is the better rule: this app has no opinion about
which languages *should* appear on a page, and one added would be an opinion the caller cannot
override without a config flag. Note also that "re-run with a better language setting" does not
exist in Hayai's API and must not be faked (§1.2) — a caller wanting a different result re-runs
with different geometry or budget, which it can already do through `run_pipeline`.

The script counts in §1.3 are still useful — as something a caller can compute from
`blocks[].text` itself, in ten lines, with its own thresholds.

* Script classification is deterministic, dependency-free and ~30 lines: Unicode ranges decide
  `kana` (3040–30FF + FF66–FF9F), `hangul` (AC00–D7AF, 1100–11FF), `latin` (0041–007A, 00C0–024F),
  `han` (4E00–9FFF + extensions). **Kana is decisive for Japanese; Hangul for Korean.** A
  `han`-only crop is genuinely ambiguous (`ja` kanji compound vs `zh`) and must be reported as
  `ambiguous`, not guessed.
* Put it in `core/language.py` (Qt-free, pure): `script_profile(text) -> dict[str, float]`.
* Config: `ocr.expected_languages: ["ja"]` (empty = accept anything) and
  `ocr.unexpected_language: "flag" | "retries" | "off"`.
* Result document (additive, so **result schema 1.2**): `blocks[].script` (dominant script),
  `blocks[].script_mix` (`{"kana":0.72,"han":0.24,"latin":0.04}`), `blocks[].flags`
  (`"unexpected_script"`, `"ambiguous_script"`, `"dense_slab"`, `"possible_truncation"`).
* **The honest "re-run" menu** — because there is no language prompt, the only thing a second
  pass can change is geometry/budget: (1) re-crop with more padding, (2) re-OCR at a different
  `max_patches`, (3) split the crop (item B). If all three produce the same unexpected script,
  the answer is not a parameter problem: report it and let the caller decide. A retry that
  cannot change the input cannot change the output, and pretending otherwise burns time.
* On the user's premise (Chinese/Korean in Japanese manga is almost always a false positive):
  **agreed, and flagging is the right response, not suppression.** Suppressing a `zh` reading
  would delete the one signal that the crop is either a real foreign-language insert or a
  misread — both of which a human/agent needs to see.

**Cost:** classification ≈ microseconds; the flags are document-only. Retries are the expensive
part and belong behind an explicit opt-in with a per-page retry budget.

### B. Sub-crop decomposition for dense slabs

**Proposal (user's).** If a large crop contains sub-crops (single lines?), also process the
sub-crops; return everything undeduplicated.

**Verdict (2026-11-22): out of scope for this app — the caller does this.** Same reasoning as A:
choosing to split a slab is choosing what a "block" means, and that is the caller's question, not
the app's. A caller that wants line-level crops crops the page itself (it has the image and the
boxes) and calls `run_pipeline` on what it cut. The pilot below stays as the warning label for
that caller, not as a spec.

* The pilot (§1.3) shows the mechanism works and is cheap (≈ +45 % OCR time for that block), and
  also shows the two traps: **blank strips hallucinate characters** (`馬`) and **naive ink-gap
  segmentation does not survive panel backgrounds**.
* Trigger must be conservative — a *slab* is a crop that is (say) both larger than
  `ocr.slab_min_area_px` **and** whose ink profile has ≥ 2 strong periodic bands, or whose first
  pass returns ≥ `ocr.slab_min_chars` characters. Bubbles must never be split.
* Splitter: fixed **N-way split along the long axis** with `N` chosen from the ink-profile peak
  count, plus a margin; skip strips whose ink ratio is below a floor (that is what produced the
  `馬` hallucination). Do **not** try to build a line detector here — that is a different project.
* Output contract (this is the important part): **the parent block stays in the document** and
  children are appended with provenance — `parent_id`, `source: "slab_split"`, `split_index`,
  `crop_bbox` in page coordinates. Nothing is merged, nothing is dropped, order is stable.
  That is exactly the "full list, let an external agent reason" model the user asked for, and it
  keeps the golden JSON of a non-split run byte-identical (the feature is off by default).
* Budget guard: `ocr.max_splits_per_page` (e.g. 4 slabs) and the existing per-block isolation
  (D37) so one bad split cannot lose the page. Progress accounting must count the extra crops,
  or the progress bar will stall at 90 %.

### C. Deduplication / topology reconstruction

**Verdict: reject *inside* this app — agreed with the user.** A merge step needs to know what the
caller intends (transcription? translation? search index?), and any heuristic that deletes text
is un-debuggable from the outside. What this app owes the consumer instead is **provenance**:
`parent_id`, `source`, `crop_bbox`, per-block engine settings, and the flags from item A. With
those, an external agent can reconstruct topology without guessing; without them, it cannot.

### D. Cheap correctness flags worth doing before A/B

**Verdict: adopt — and now SHIPPED as P8.2b (result schema 1.2, 2026-11-22).** What landed:

* `blocks[].flags` — `"possible_truncation"` when `len(text) >= 0.9 * ocr.max_new_tokens`.
* `blocks[].contained_in` — id of the tightest strictly larger box covering >= 80 % of this one.
* `ocr.max_patches` accepts **512** in config, schema, GUI and engine (it always worked at the
  processor level; our validation was the ceiling).

What deliberately did **not** land, so a later reader does not think it was overlooked: a
`dense_slab` flag (needs a definition of "slab" that is really a splitting policy — item B, out of
scope), per-block re-runs on flag (changes a whole page's cost without the caller asking), and
GUI surfacing of the flags (the JSON pane shows them; the block table can pick them up in 8.6).
Both flags are emitted for `stage: "detect"` documents too, and are optional on input, so 1.0
and 1.1 documents still read. Tests: `tests/test_pipeline.py::TestDocumentAnnotations`,
`tests/test_types.py::TestBlockFlags`, `tests/test_schemas.py::TestResultSchemaVersions`.

Original proposal, kept for the record (all three shipped as written):

1. **Possible truncation:** `len(text) >= max_new_tokens * 0.9` → `possible_truncation`. The
   upstream escalation heuristic uses exactly this ratio, so it is the model author's own signal.
2. **`MAX_PATCH_CHOICES` missing 512** (§1.2) — either add it or document why 384 is the ceiling.
3. **Containment report:** blocks whose box is ≥ 80 % inside another box get
   `contained_in: <id>` (geometry only, no text logic). This is the `った` case: today the reader
cannot tell a fragment from a line. (Overlaps the old backlog item "opt-in containment dedup" —
   flag instead of dropping.)

### E. Bench harness for model updates

**Verdict: adopt — this is the workflow the user is actually in.** Hayai revisions arrive every
few days (this turn: `f5f0f617` → `b00317ed`), and the current way to compare them is a manual
`--run` and a diff. A `scripts/bench_models.py` that runs a fixed corpus (the sample page + N
user-supplied pages, kept out of the repo) over two revisions/variants and prints a per-block
diff table would turn "did the update help?" into a 2-minute answer. Needs: a pinned-revision
override (`--ocr-revision`), a JSON baseline, and a `--only-changed` view. It must stay
`-m models`-opt-in so the default suite never touches weights.

### F. Things explicitly **not** to do

| Idea | Why not |
|---|---|
| OCR the full page directly | The card: "Passing full comic pages without cropping text regions will not work." |
| A language prompt / language argument | It does not exist in `ocr()`, `generate()` or the processor (checked in the vendored code) |
| Beam search / repetition penalty knobs | `ocr()` does not forward them; the card requires `repetition_penalty=1.0` |
| In-app deduplication / merge | Item C — needs caller intent; destroys evidence |
| Training or shipping a line-segmentation model | Out of scope for an OCR front-end; the detector already exists and is the right tool |

---

## 3. If any of this is picked up

1. Sequence **D → A → B → E**: flags first (they cost nothing and make everything else
   measurable), then language flagging, then splitting, then the bench.
2. Document changes are **additive** → result schema **1.1 → 1.2**, readers keep accepting 1.0/1.1
   (the rule used for config 1.0 → 1.1 → 1.2).
3. Every new engine behaviour must stay **off by default** so golden-JSON tests and the existing
   MCP contracts do not move, and must respect D37 (per-block isolation) and the cancellation
   contract.
4. New MCP surface, if any: `set_config` already covers new `ocr.*` keys; splitting needs no new
   tool, only config — which is the cheapest kind of growth this project can have.
