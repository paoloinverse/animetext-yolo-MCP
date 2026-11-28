# The model bench — `scripts/bench_models.py`

The question this exists for is one sentence long: **did the new Hayai snapshot read my pages
better than the one the app is pinned to?**

A benchmark answers it only if everything except the weights is held fixed. That constraint
shapes the whole script, and the places where it is *not* held fixed are the places where the
numbers mean something else — so they are written down here, next to the commands.

```bash
python scripts/bench_models.py profile --corpus sample_scans --jobs 4   # measure the corpus once
python scripts/bench_models.py panel   -n 12 --seed 1 --dry-run         # see the sample, run nothing
python scripts/bench_models.py run     --against latest --noise-floor   # the comparison
python scripts/bench_models.py run     --against 90fc6148 --against f5f0f617 --only-changed
```

## What it writes, and where

Everything goes under `bench/`, which is gitignored as a block. The profile describes someone's
pages and a report quotes the text read off them: scan content wearing a `.json` extension is
still scan content. `bench/snapshots/` holds candidate revisions, which is the weights argument
again. Nothing here is written by the app, and deleting the folder costs only the minutes it
took to measure the corpus.

| file | what it is |
|---|---|
| `bench/profile.json` | every page measured once with the pinned snapshot: the boxes, the text, the timings, the derived features |
| `bench/panel.json` | a chosen sample and the seed that produced it — a panel that was not saved is a panel that cannot be re-run |
| `bench/reports/<utc>-<candidate>.json` | one comparison: both revisions by full sha, the environment, the per-page diff, the per-block text of what moved |
| `bench/reports/<utc>-<candidate>.selftest.json` | the noise floor, kept apart so it cannot be mistaken for a comparison |

## The panel, not the corpus

A benchmark is only comparable if the *same* pages go in every time — a score that moves on a
different set of pages tells you nothing about the model. But a fixed sample of easy pages also
reports "no change" on the day a snapshot finally reads hard pages.

So the panel is drawn from the profile by two rules, in this order:

1. **every difficulty tier present gets one page before any tier gets a second.** A 12-page
   panel over a five-tier corpus is two pages per tier: the sample covers the range rather than
   the middle of it.
2. **inside a tier, layouts are spread before any layout repeats** — a panel of twelve near
   identical pages measures one page twelve times.

Difficulty is a **score**, not a label, and it is measured from the pinned side's own output:
six components, each a straight ramp between "costs nothing" and "costs everything", averaged —
the smallest box's side, the block count, the share of the page covered by text, boxes nested in
other boxes, the detector's weakest score, and the longest line against the token cap. The tiers
are bands of that score. The components are stored per page and printed as the `why` column of
`panel`, because a difficulty you cannot interrogate is a difficulty you cannot disagree with.

The strata are **recomputed from the stored features every time a profile is read**, never read
out of the file. The profile is the measurement; the strata are an opinion about it, and an
opinion revised in a month must not require ten minutes of re-measuring.

## What the columns mean

    page        tier     verdict  blocks  changed  CER    lost  ms pinned/cand   ratio

* **changed** — blocks whose text differs *character for character*, on the raw text. The
  normalised comparison is kept separate: a revision that starts emitting half-width periods
  changes every line and says nothing new, and that is a `normalisation_only` reason on the row,
  not a reason to celebrate.
* **CER** — character error rate over the changed blocks, measured against the pinned side. It
  answers "did one comma move, or did the page get re-read?" It is *not* an accuracy figure:
  neither side is ground truth.
* **lost** — blocks the pinned side read and the candidate returned empty. This is the only
  column that can name a regression without reading Japanese, and it forces `SUSPECT`.
* **ratio** — candidate seconds over pinned seconds. A model that reads better at three times
  the cost is a decision, not a win.

`verdict` is one of `SAME` / `CHANGED` / `SUSPECT`. **`SUSPECT` is not "worse"**: it means the
diff cannot be read as a text change — boxes moved, blocks appeared or vanished, or text was
lost. The detector and its weights are the same on both sides, so structural movement is a
canary that something upstream of the OCR moved with it.

**The report never says "upgrade".** `verdict` is `null` in every report file, by test
(`test_the_harness_never_declares_an_upgrade`): whether new Japanese is *better* is not
computable from a diff, and a harness that printed "IMPROVED" would be the most misleading
number in the folder.

## The self-test, and why `--noise-floor` is worth the minutes

Running a snapshot against its own sha must print all-SAME; anything else is a fact about the
harness or about run-to-run variance. The first self-test here printed three changed blocks on a
fourteen-block page — and it was right to, because the harness had been reading the pinned side
through one code path and the candidate through another. That defect is recorded in the open
list of `docs/QA.md`; what the harness learned from it is in the next section.

`--noise-floor` runs that self-test before the real comparison and stores the result in the
report as `noise_floor`. Every "N blocks changed" below it should be read against that number.
On this machine (CPU, one process, batch 16) the floor measured **0 of 113 blocks** — the
pipeline is reproducible — while the *timings* moved ±15%, which is why the summary quotes the
median ratio and not the mean.

## What it refuses to do

* **A candidate whose weights and code are byte-identical to the pinned snapshot is not run.**
  `--against latest` on 2026-11-27 resolved to a revision whose only difference from the pin was
  `README.md`; the script says so, names the file, and asks for `--force`. Without that rule the
  answer to "did the update help?" would eventually be forty minutes of scheduler noise.
* **It does not invent a loading path for the candidate.** A candidate is staged as a
  bundle-shaped directory (`bench/snapshots/<sha>/hayai-ocr-v2.7-supernova/`, with an
  `.origin.json` naming the sha) and loaded by the unmodified `OcrEngine` with `offline=True`. A
  bench with its own loader benchmarks the bench.
* **It does not re-download bytes it already holds.** Staging first asks the bundle's
  `.origin.json` whether that sha is what is already vendored; failing that, it asks the Hub
  what hash each file of the candidate has and looks for a file with that hash among the bundle
  and the snapshots already staged — a match is **hard-linked**, not copied, so a candidate that
  changed only its README costs 0 bytes and 0 MB of disk. `--no-reuse` turns that off.
* **Every Hub question is asked before the first engine loads.** Loading the OCR engine puts the
  process into HF offline mode (the app's own behaviour, D26), and an API call made after that
  fails with "offline mode is enabled" — halfway through a run.

## Long runs, and the limits that end them

`profile` is the expensive command and it is resumable: `--resume` keeps every page whose
*content hash* was already measured at the same baseline revision, and re-measures the rest. A
page renamed is a page re-measured; the same page copied elsewhere is not.

Workers are recycled every 10 pages (`PROFILE_MAX_TASKS_PER_CHILD`) and a dead pool writes what
it measured before it exits, with a message to re-run with `--resume`. That is not paranoia:
sandboxes and CI runners set `RLIMIT_CPU`, and a worker that outlives its CPU budget takes the
whole pool down as `BrokenProcessPool` — a confusing way to lose ten minutes of work, which this
script did to itself once before it was taught to save its own results.

## Reading a report

Read `lost` first, then `changed` against the noise floor, then `ratio`. Read the
`--only-changed` rows before believing any of it: the diff of a page that changed is two minutes
of reading and is the only part that says whether the change is worth anything. The JSON keeps
the pinned and candidate text of every changed block, so the reading can happen afterwards, in a
text editor, without running the model again — or with `--no-text`, if the report has to leave
the machine, in which case those lines are sha256 and the judgement has to happen on the screen.
