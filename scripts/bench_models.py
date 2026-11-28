#!/usr/bin/env python3
"""Compare two Hayai OCR snapshots over a corpus of real pages, and show the diff.

The question this exists for (ADDENDUM item E): the OCR repo gets a new snapshot every few
days, and "did the update help?" currently costs a manual `--run` plus reading two JSON
files. This turns it into one command that reads the *same crops* with both snapshots and
prints a per-page table.

    python scripts/bench_models.py profile                     # measure the corpus once
    python scripts/bench_models.py panel -n 12 --seed 7        # what the sample would be
    python scripts/bench_models.py run --against latest        # pinned vs current Hub HEAD
    python scripts/bench_models.py run --against f5f0f617 --only-changed
    python scripts/bench_models.py run --against latest --against f5f0f617 -n 20 --seed 3

Three rules the whole script is built around
--------------------------------------------
**Only one variable moves.** Detection runs *once* per page; the candidate side re-reads the
boxes the pinned side produced (`Pipeline.reocr_blocks`, P9.3), so a text difference can only
come from the OCR snapshot. It is also cheaper: one detection pass, two OCR passes. The cost
of this design is that block count and box positions are equal by construction — so the
report's structural columns are a *canary* (non-zero means the harness or the app is lying),
not a score.

**Both sides are named by full commit sha.** "latest" is resolved to a 40-character sha on
the Hub before anything runs, and the sha of each side — plus the sha256 of the weights file
that was actually loaded — is written into every report. A report from March still says what
it compared, which is the only way a stored baseline stays meaningful after the Hub moved.

**The tool never says "upgrade".** It says SAME / CHANGED / SUSPECT per page and prints
numbers. Whether new text is *better* Japanese is a judgement about Japanese, not about a
diff, and it is the operator's. The report document carries `"verdict": null` on purpose.

What stays on this machine
--------------------------
`bench/` is gitignored and holds the profile, the panel, the staged candidate snapshots and
the reports. A report contains the recognised text of your pages, which is scan content:
`sample_scans/` is out of the repository (decision D95) and so is everything derived from it
that quotes it. Pass `--no-text` to write a report with per-block text replaced by sha256
prefixes — still enough to re-run the SAME/CHANGED comparison later, not enough to read.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run from a checkout

from animetext_studio import __version__ as APP_VERSION                       # noqa: E402
from animetext_studio.core.config import AppConfig                             # noqa: E402
from animetext_studio.core.model_store import (                                # noqa: E402
    OCR_DIRNAME, ModelStore, discover_hf_token, resolve_models_dir, sha256_file,
)
from animetext_studio.core.types import OCR_REPO, OCR_REVISION                 # noqa: E402

# ---------------------------------------------------------------------------
# Where things go, and what the documents are called
# ---------------------------------------------------------------------------

#: Everything the bench writes lives here, and `bench/` is gitignored: a profile is a
#: description of someone's scans, and a report quotes their text.
BENCH_DIR = Path("bench")
PROFILE_PATH = BENCH_DIR / "profile.json"
PANEL_PATH = BENCH_DIR / "panel.json"
SNAPSHOT_ROOT = BENCH_DIR / "snapshots"
REPORT_DIR = BENCH_DIR / "reports"

PROFILE_KIND = "animetext-bench-profile/1.0"
PANEL_KIND = "animetext-bench-panel/1.0"
REPORT_KIND = "animetext-bench-report/1.0"

#: Default corpus. The README's position is that nothing here licenses your scans, so the
#: bench reads them where they already are and copies nothing anywhere else.
DEFAULT_CORPUS = Path("sample_scans")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

#: The snapshot files a candidate side needs. `.gitattributes` and the repo's own
#: `yolo26m-text.pt` (47 MB, the model author's *demo* detector — this app detects with
#: AnimeText YOLO, never with this) are skipped: 47 MB and 1.5 kB of a 680 MB download
#: that no code path in the app reads.
SNAPSHOT_ALLOW = (
    "config.json", "configuration_hayai.py", "modeling_hayai.py", "processing_hayai.py",
    "preprocessor_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab*txt", "merges*txt",
    "*.safetensors",
)


# ---------------------------------------------------------------------------
# Text comparison (pure, no torch, no app engines — this part is unit-testable)
# ---------------------------------------------------------------------------

def norm_text(text: Optional[str]) -> str:
    """Normalise a recognised string for *measuring* difference.

    NFKC plus whitespace collapse. Used for the CER column only, never for the SAME/CHANGED
    decision: `changed_blocks` counts raw inequality, so a snapshot that starts emitting
    half-width tildes shows up as changed — and `normalisation_only` then says the change was
    only ever normalisation (raw differs, CER 0). Collapsing the two would hide a real,
    user-visible difference behind a 0.000.
    """
    if text is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, two rows. Small strings (one block of manga text), so O(nm) is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def block_cer(ref: Optional[str], hyp: Optional[str]) -> float:
    """How much one block moved between the two snapshots, on the pinned side's scale.

    The denominator is the *pinned* text's length, because the question is "how much of what
    we had turns out to be different". Three cases worth spelling out:

    * both empty → 0.0. Nothing moved, and this must not be a division by zero.
    * pinned empty, candidate not → 1.0 by convention: a whole block appeared, which is as
      much difference as a block can hold. Not infinity, and not 0 because "no text" is short.
    * candidate much longer than pinned → **greater than 1.0, on purpose.** A block that was
      two characters and is now forty has changed more than a block whose forty characters
      were re-worded. Capping it at 1 would make the two look the same.

    It is a scale for "how much moved", never an accuracy claim: neither side is ground
    truth, and the report says so in as many words.
    """
    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


# ---------------------------------------------------------------------------
# Page features, layout archetype, difficulty tier
# ---------------------------------------------------------------------------

#: Ramp endpoints for the difficulty components below. Each component is a straight line
#: between a value that costs nothing and a value that costs everything, clipped to 0..1 —
#: chosen over an absolute cut because "24 blocks" is not a difficulty, "many blocks" is, and
#: the ramp says how many is many. They sort pages into strata; they gate nothing in the app.
TINY_SIDE_PX = (30, 12)        #: smallest box side: 30 px is comfortable, 12 px is a few px/glyph
CROWDED_BLOCKS = (8, 36)       #: 8 boxes is a page of bubbles, 36 is a wall of them
DENSE_AREA_FRAC = (0.05, 0.35) #: share of the page covered by text boxes
NESTED_BLOCKS = (0, 10)        #: boxes inside other boxes — fragments read out of context
MARGINAL_SCORE = (0.25, 0.05)  #: the detector's own weakest box: 0.25 is sure, 0.05 is a guess
LONG_LINE_CHARS = (40, 120)    #: longest line: past `max_new_tokens` territory, truncation risk
#: Layout labels, in the order `layout_of` asks them, and the cuts it uses there.
SPARSE_BLOCKS = 5
CROWDED_LAYOUT_BLOCKS = 26
DENSE_LAYOUT_AREA = 0.25
MIXED_ORIENT_FRAC = 0.90
HORIZONTAL_ORIENT_FRAC = 0.34


def page_features(width: int, height: int, blocks: Sequence[dict]) -> dict[str, Any]:
    """Structural + outcome features of one page, from the result document alone.

    Deliberately derived, never measured twice: everything here is already in the document,
    so the profile cannot disagree with the run that produced it.
    """
    page_area = max(1, width * height)
    sides, areas, scores, chars = [], [], [], []
    with_text = empty = skipped = errored = flagged = contained = vertical = trunc = 0
    for b in blocks:
        box = b["bbox"]
        w = max(1, box["x1"] - box["x0"])
        h = max(1, box["y1"] - box["y0"])
        sides.append(min(w, h))
        areas.append(w * h)
        scores.append(float(b.get("score") or 0.0))
        if h > w:
            vertical += 1
        if b.get("error") and b["error"].get("code", "").startswith("SKIPPED"):
            skipped += 1
        elif b.get("error"):
            errored += 1
        if b.get("text"):
            with_text += 1
            chars.append(len(b["text"]))
        elif b.get("error") is None:
            empty += 1
        if b.get("flags"):
            flagged += 1
            trunc += sum(1 for f in b["flags"] if f == "possible_truncation")
        if b.get("contained_in") is not None:
            contained += 1
    n = len(blocks)
    total_chars = sum(chars)
    return {
        "megapixels": round(page_area / 1e6, 3),
        "aspect": round(width / max(1, height), 3),
        "blocks": n,
        "blocks_with_text": with_text,
        "blocks_empty": empty,
        "blocks_skipped": skipped,
        "blocks_error": errored,
        "blocks_flagged": flagged,
        "blocks_truncated": trunc,
        "blocks_contained": contained,
        "vertical_blocks": vertical,
        "vertical_frac": round(vertical / n, 3) if n else 0.0,
        "empty_frac": round(empty / n, 3) if n else 0.0,
        "text_area_frac": round(sum(areas) / page_area, 4),
        "min_side_px": min(sides) if sides else 0,
        "median_side_px": int(statistics.median(sides)) if sides else 0,
        "mean_score": round(sum(scores) / n, 4) if n else 0.0,
        "min_score": round(min(scores), 4) if n else 0.0,
        "chars": total_chars,
        "median_chars": int(statistics.median(chars)) if chars else 0,
        "max_chars": max(chars) if chars else 0,
    }


LAYOUTS = ("none", "sparse", "horizontal", "mixed", "dense", "crowded", "columns")
TIERS = ("trivial", "easy", "typical", "hard", "brutal", "unread")
#: Score bands. Absolute, not percentiles of the corpus in front of you: a tier has to mean
#: the same thing in March as it means today, and "this page is brutal" should not depend on
#: whether the rest of the folder happens to be easy. The profile prints the distribution, so
#: a corpus that lands entirely in one band is visible rather than hidden by a relabelling.
TIER_BANDS = ((0.12, "trivial"), (0.24, "easy"), (0.37, "typical"), (0.50, "hard"))


def _ramp(value: float, low: float, high: float) -> float:
    """0 at `low`, 1 at `high`, straight line between, clipped outside. Works in both
    directions, so a "bigger is worse" ramp and a "smaller is worse" ramp are the same call."""
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def layout_of(f: dict[str, Any]) -> str:
    """Which *kind* of page this is. First match wins, and the order is the argument:

    no boxes is not a layout; a handful of boxes is a handful whatever their shape; the
    orientation mix comes next because a page with horizontal Latin in it is a different
    reading problem from an all-vertical one, and that fact is lost if area and count are
    asked first; only then the two volume axes, area before count, because one slab covering a
    quarter of the page is a harder read than 26 bubbles. `columns` is what is left: the
    ordinary all-vertical manga page, which is the majority and deserves its own name.
    """
    blocks = f["blocks"]
    if blocks == 0:
        return "none"
    if blocks <= SPARSE_BLOCKS:
        return "sparse"
    if f["vertical_frac"] < HORIZONTAL_ORIENT_FRAC:
        return "horizontal"
    if f["vertical_frac"] < MIXED_ORIENT_FRAC:
        return "mixed"
    if f["text_area_frac"] >= DENSE_LAYOUT_AREA:
        return "dense"
    if blocks >= CROWDED_LAYOUT_BLOCKS:
        return "crowded"
    return "columns"


def difficulty_parts(f: dict[str, Any]) -> dict[str, float]:
    """The six measured signals that make a page hard, each 0..1, plus the page's score.

    Every component is something the run *observed* — box size, box count, coverage, nesting,
    the detector's own weakest score, the longest line — not a guess about the artwork. They
    are kept separately (in the report, and in `--verbose` output) so a page called brutal can
    be argued with: "brutal because the boxes are tiny" is actionable, "score 0.71" is not.

    Deliberately absent: `blocks_empty` and error counts. Those measure how the *pinned model*
    did, and a difficulty stratum has to be a property of the page — otherwise a model update
    silently re-sorts the panel and next month's run is not this month's panel.
    """
    if f["blocks"] == 0:
        return {}
    return {"tiny": _ramp(f["min_side_px"], *TINY_SIDE_PX),
            "crowd": _ramp(f["blocks"], *CROWDED_BLOCKS),
            "area": _ramp(f["text_area_frac"], *DENSE_AREA_FRAC),
            "nested": _ramp(f["blocks_contained"], *NESTED_BLOCKS),
            "marginal": _ramp(f["min_score"], *MARGINAL_SCORE),
            "longest": _ramp(f["max_chars"], *LONG_LINE_CHARS)}


def difficulty_of(f: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """(score 0..1, components). Equal weights: no measurement here says crowding hurts more
    than tiny type, and inventing a weighting would be a guess dressed as a number."""
    parts = difficulty_parts(f)
    if not parts:
        return 0.0, {}
    return sum(parts.values()) / len(parts), parts


def tier_of(f: dict[str, Any]) -> str:
    """The difficulty band of a page, from the measured score — `unread` excepted.

    `unread` is not the top of the scale, it is a different axis: no boxes at all, or boxes
    the current weights read no text from, is a fact about this model on this page, and it is
    the single most interesting page in a corpus for exactly that reason. It gets its own
    stratum so a panel always contains it instead of averaging it away.
    """
    if f["blocks"] == 0 or f["blocks_with_text"] == 0:
        return "unread"
    score, _ = difficulty_of(f)
    for cut, name in TIER_BANDS:
        if score < cut:
            return name
    return "brutal"


def strata_of(f: dict[str, Any]) -> dict[str, Any]:
    """layout + tier + score + components, computed from features. Recomputed on every read,
    which is what lets the bands above be retuned without re-measuring a corpus."""
    score, parts = difficulty_of(f)
    return {"layout": layout_of(f), "tier": tier_of(f), "score": round(score, 3),
            "parts": {k: round(v, 3) for k, v in parts.items()}}


# ---------------------------------------------------------------------------
# The panel: a seeded, stratified sample
# ---------------------------------------------------------------------------

def pick_panel(pages: Sequence[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Choose `size` pages covering every difficulty tier present, deterministically.

    Round-robin over the tiers (shuffled with `seed`), and within a tier over the layouts not
    yet used in that tier. The reason for round-robin instead of proportional sampling is the
    point of the panel: a proportional draw from a real corpus is 80% ordinary pages and
    never shows the rare ones, while "trivial to near-impossible" is only testable if the
    rare strata are *in* the sample. When a tier has fewer pages than its share, the leftover
    slots go back to the round-robin, so `size` is honoured unless the corpus is smaller.

    Deterministic for a given (profile content, size, seed): the pages are ordered by name
    before any randomness touches them, so re-running with the same seed picks the same pages.
    """
    eligible = [p for p in pages if p.get("status") == "ok"]
    if size <= 0 or len(eligible) <= size:
        return sorted(eligible, key=lambda p: p["file"])
    rng = random.Random(f"{seed}")
    by_tier: dict[str, list[dict[str, Any]]] = {}
    for page in sorted(eligible, key=lambda p: p["file"]):
        by_tier.setdefault(page["tier"], []).append(page)
    tiers = list(TIERS) + [t for t in by_tier if t not in TIERS]
    order = [t for t in tiers if t in by_tier]
    rng.shuffle(order)
    for tier in order:
        rng.shuffle(by_tier[tier])
    # Inside a tier: round-robin over that tier's layouts, so the first page of each layout
    # comes before the second page of any one layout. Two loops, no conditional skipping:
    # a page is always available, so the loop below cannot stall or drop a slot.
    ordered: dict[str, list[dict[str, Any]]] = {}
    for tier, pool in by_tier.items():
        by_layout: dict[str, list[dict[str, Any]]] = {}
        for page in pool:
            by_layout.setdefault(page["layout"], []).append(page)
        layout_names = list(by_layout)
        rng.shuffle(layout_names)
        for name in layout_names:
            rng.shuffle(by_layout[name])
        queue: list[dict[str, Any]] = []
        while any(by_layout[n] for n in layout_names):
            for name in layout_names:
                if by_layout[name]:
                    queue.append(by_layout[name].pop(0))
        ordered[tier] = queue

    chosen: list[dict[str, Any]] = []
    cursors = {t: 0 for t in by_tier}
    while len(chosen) < size:
        progressed = False
        for tier in order:
            queue = ordered[tier]
            if cursors[tier] >= len(queue):
                continue
            chosen.append(queue[cursors[tier]])
            cursors[tier] += 1
            progressed = True
            if len(chosen) >= size:
                break
        if not progressed:
            break
    return sorted(chosen, key=lambda p: p["file"])


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------

#: Verdicts, weakest first. `SUSPECT` outranks `CHANGED` because it means the diff cannot be
#: read as "more or fewer characters" — something structural moved, or text disappeared.
VERDICTS = ("SAME", "CHANGED", "SUSPECT")


@dataclass
class PageDiff:
    """One row of the table: what the two snapshots did differently on one page."""

    file: str
    tier: str = ""
    layout: str = ""
    score: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    blocks: int = 0
    changed: int = 0
    cer: float = 0.0
    lost: int = 0
    gained: int = 0
    new_errors: int = 0
    geometry_moved: int = 0
    normalisation_only: int = 0
    base_ms: int = 0
    cand_ms: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.geometry_moved or self.lost or self.new_errors or self.blocks < 0:
            return "SUSPECT"
        return "CHANGED" if self.changed else "SAME"

    @property
    def time_ratio(self) -> Optional[float]:
        if not self.base_ms:
            return None
        return self.cand_ms / self.base_ms

    def to_dict(self) -> dict[str, Any]:
        d = {"file": self.file, "tier": self.tier, "layout": self.layout,
             "difficulty": self.score, "difficulty_parts": self.parts,
             "blocks": abs(self.blocks), "changed_blocks": self.changed,
             "cer": round(self.cer, 4), "text_lost": self.lost, "text_gained": self.gained,
             "new_errors": self.new_errors, "geometry_moved": self.geometry_moved,
             "normalisation_only": self.normalisation_only,
             "verdict": self.verdict, "reasons": self.reasons,
             "timing_ms": {"pinned": self.base_ms, "candidate": self.cand_ms}}
        ratio = self.time_ratio
        d["time_ratio"] = round(ratio, 3) if ratio is not None else None
        return d


def diff_blocks(base: Sequence[dict], cand: Sequence[dict]) -> PageDiff:
    """Compare two block lists for one page (same ids expected; anything else is SUSPECT).

    `changed` is raw text inequality. A block whose text vanished counts as `lost`, which is
    a different event from "rewritten": a snapshot that drops a line is not "different", it
    is broken, and the operator should not have to notice it in a count of changed blocks.
    """
    by_id_base = {b["id"]: b for b in base}
    by_id_cand = {b["id"]: b for b in cand}
    out = PageDiff(file="", blocks=len(base))
    if len(base) != len(cand) or set(by_id_base) != set(by_id_cand):
        out.reasons.append("block set differs")
        out.geometry_moved = abs(len(base) - len(cand)) or 1
    for bid in sorted(set(by_id_base) & set(by_id_cand)):
        b, c = by_id_base[bid], by_id_cand[bid]
        if b["bbox"] != c["bbox"]:
            out.geometry_moved += 1
        bt, ct = b.get("text"), c.get("text")
        b_err = (b.get("error") or {}).get("code")
        c_err = (c.get("error") or {}).get("code")
        if b_err is None and c_err is not None and not str(c_err).startswith("SKIPPED"):
            out.new_errors += 1
        if bt and not ct:
            out.lost += 1
        elif ct and not bt:
            out.gained += 1
        if bt == ct:
            continue
        out.changed += 1
        out.cer += block_cer(bt, ct)
        if norm_text(bt) == norm_text(ct):
            out.normalisation_only += 1
    if out.changed:
        out.cer /= out.changed
    if out.geometry_moved:
        out.reasons.append("box positions moved between sides")
    if out.lost:
        out.reasons.append(f"{out.lost} block(s) lost their text")
    if out.new_errors:
        out.reasons.append(f"{out.new_errors} block(s) gained an OCR error")
    if out.gained:
        out.reasons.append(f"{out.gained} block(s) gained text")
    if out.normalisation_only and out.normalisation_only == out.changed:
        out.reasons.append("every change is width/spacing normalisation only")
    return out


# ---------------------------------------------------------------------------
# Identity: what produced this number
# ---------------------------------------------------------------------------

def app_identity(root: Path = Path(".")) -> dict[str, Any]:
    """Which build of the app ran the bench: git commit if this is a checkout, else a hash.

    A bench number without a build identity is unrepeatable, so the fallback is a content
    hash of the package rather than a blank — and it says which of the two it is.
    """
    def git(*args: str) -> Optional[str]:
        try:
            r = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                               text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    if head:
        return {"kind": "git", "version": APP_VERSION, "commit": head,
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    h = hashlib.sha256()
    for path in sorted((root / "animetext_studio").rglob("*.py")):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return {"kind": "content-hash", "version": APP_VERSION, "commit": None,
            "content_sha256": h.hexdigest(),
            "note": "not a git checkout: the app is identified by a hash of its own source"}


def environment() -> dict[str, Any]:
    """Machine facts that change a timing. Versions, device, thread counts — not the CPU model."""
    env: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "onnxruntime", "numpy", "PIL"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:                                        # noqa: BLE001
            env[mod] = None
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ONNX_MODE", "HAYAI_DISABLE_COMPILE"):
        if os.environ.get(key):
            env[f"env:{key}"] = os.environ[key]
    return env


def page_identity(path: Path) -> dict[str, Any]:
    from PIL import Image                                        # lazy: only the runner needs it

    with Image.open(path) as im:
        w, h = im.size
    return {"file": path.name, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "width": w, "height": h}


# ---------------------------------------------------------------------------
# Hub: resolve a revision, see what actually differs, stage it for loading
# ---------------------------------------------------------------------------

def quiet_bar_class():
    """A disabled `tqdm`, for `snapshot_download(tqdm_class=...)` under --quiet.

    Subclassing is the supported way to silence the download bar: `quiet=` is a newer keyword
    and is not in every installed `huggingface_hub`, and a script that dies on an unexpected
    keyword in front of a 630 MB download is a script that wastes the download.
    """
    try:
        from tqdm.auto import tqdm
    except Exception:                                            # noqa: BLE001
        return None

    class _Silent(tqdm):                                          # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _Silent


def hub_api(token: Optional[str]):
    from huggingface_hub import HfApi                            # lazy: offline paths skip it

    return HfApi(token=token)


def resolve_revision(api: Any, spec: str) -> dict[str, Any]:
    """Turn "latest" / a short sha / a branch into a full 40-character commit sha.

    Refusing to run against an ambiguous ref is the whole point: `main` moves, and a report
    that recorded "main" would be unreadable the next time the author uploads.
    """
    wanted = spec.strip()
    head = False
    if wanted.lower() in ("latest", "head", "main", "master"):
        wanted, head = "main", True
    commits = api.list_repo_commits(OCR_REPO, revision="main")
    if head:
        return {"spec": spec, "sha": commits[0].commit_id, "is_head": True,
                "title": commits[0].title, "date": str(commits[0].created_at)}
    if re.fullmatch(r"[0-9a-f]{40}", wanted):
        match = next((c for c in commits if c.commit_id == wanted), None)
        return {"spec": spec, "sha": wanted, "is_head": False,
                "title": match.title if match else None,
                "date": str(match.created_at) if match else None}
    matches = [c for c in commits if c.commit_id.startswith(wanted)]
    if not matches:
        raise SystemExit(f"revision {spec!r} is not a commit of {OCR_REPO} "
                         f"(tried {len(commits)} commits of 'main')")
    if len(matches) > 1:
        raise SystemExit(f"revision {spec!r} is ambiguous: "
                         + ", ".join(m.commit_id[:12] for m in matches))
    c = matches[0]
    return {"spec": spec, "sha": c.commit_id, "is_head": False, "title": c.title,
            "date": str(c.created_at)}


def file_fingerprint(api: Any, sha: str) -> dict[str, tuple[Optional[int], Optional[str]]]:
    """name -> (bytes, content hash) for one revision, without downloading anything.

    LFS files report their lfs sha256, ordinary files their git blob id. The two kinds are
    not comparable to each other, but they are comparable *across revisions* of the same
    file, which is the only question asked here.
    """
    info = api.repo_info(OCR_REPO, files_metadata=True, revision=sha)
    out = {}
    for s in info.siblings or ():
        out[s.rfilename] = (s.size, s.lfs.sha256 if s.lfs else s.blob_id)
    return out


def revision_difference(a: dict, b: dict) -> dict[str, Any]:
    """Which files differ between two revisions, and whether the engine could tell them apart.

    `model.safetensors` and `modeling_hayai.py` are what the OCR engine *is*; everything else
    (README, .gitattributes) is bookkeeping. When the engine files are identical the run
    would measure scheduler noise, and the script says that instead of spending ten minutes
    proving it.
    """
    names = sorted(set(a) | set(b))
    differing = [n for n in names if a.get(n) != b.get(n)]
    engine = [n for n in differing if n not in ("README.md", ".gitattributes")]
    return {"differing_files": differing, "engine_files_differ": engine,
            "weights_identical": a.get("model.safetensors") == b.get("model.safetensors")}


def wanted_at(revision_files: dict[str, Any]) -> dict[str, tuple[int, Optional[str]]]:
    """Which files of a revision belong in a staged snapshot, with their recorded hashes.

    `SNAPSHOT_ALLOW` is a set of globs because the repo's file list is the model author's to
    change; the fingerprint is filtered through it rather than assumed to match it. The hash
    that comes back from the Hub is the LFS sha256 for the weights and the blob id for the
    small files — which is exactly what makes "we already have this byte-for-byte" checkable.
    """
    from fnmatch import fnmatch

    out: dict[str, tuple[int, Optional[str]]] = {}
    for name, (size, digest) in revision_files.items():
        if any(fnmatch(name, pattern) for pattern in SNAPSHOT_ALLOW):
            out[name] = (int(size or 0), digest)
    return out


def reuse_sources(extra: Optional[Path] = None) -> list[Path]:
    """Directories worth searching before anything is downloaded: the bundle, then every
    snapshot already staged. The bundle first because it is the one this project guarantees."""
    out: list[Path] = []
    bundle = resolve_models_dir() / OCR_DIRNAME
    if bundle.is_dir():
        out.append(bundle)
    if SNAPSHOT_ROOT.is_dir():
        out.extend(sorted(p.parent.parent for p in SNAPSHOT_ROOT.glob(f"*/{OCR_DIRNAME}")))
    if extra is not None and extra.parent != SNAPSHOT_ROOT:
        out.append(extra)
    return out


def find_reusable(name: str, size: int, digest: Optional[str],
                  sources: Sequence[Path]) -> Optional[Path]:
    """The first source holding this exact file, or None.

    Identity is the recorded hash, never the filename: two revisions can carry a file of the
    same name whose bytes differ, and loading the wrong one under a name that matches is the
    kind of error a benchmark would report as a result. The size is checked first because it
    costs nothing and rejects almost every miss before a 633 MB hash is computed.
    """
    if not digest:
        return None
    for source in sources:
        candidate = source / name
        try:
            if not candidate.is_file():
                continue
            if size and candidate.stat().st_size != size:
                continue
        except OSError:
            continue
        if sha256_file(candidate) == digest:
            return candidate
    return None


def link_or_copy(src: Path, dst: Path) -> str:
    """Put a verified-identical file in place without copying its bytes if the FS can help it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)                                  # same filesystem: no bytes moved
        return "linked"
    except OSError:                                        # different device: copy it
        import shutil

        shutil.copy2(src, dst)
        return "copied"


def stage_snapshot(sha: str, token: Optional[str], *, quiet: bool = False,
                   use_bundle: bool = True, reuse: bool = True,
                   revision_files: Optional[dict[str, Any]] = None) -> Path:
    """Make one revision available as a bundle-shaped directory, and say where it is.

    Four ways, in the order they are tried, because the difference between them is 680 MB of
    network and minutes of the operator's time:

    1. **it is already staged** under `bench/snapshots/<sha>/`;
    2. **it is the bundle** — the revision the app already vendored, which `.origin.json` says
       by name. This is what makes `run --against <the pinned sha>` (the self-test) and a
       re-run after a failed session cost nothing;
    3. **its files are already on this machine**, file by file, *verified by the hash the Hub
       records for them*: a revision that changed only its README has the same weights as the
       one already vendored, and re-downloading 633 MB to discover that is rude to the
       operator's time. Matches are hard-linked, so they cost no extra disk either;
    4. **download what is left**.

    The staged directory *is* the app's bundle layout, which is what lets the candidate side
    be loaded by the unmodified `OcrEngine` with `models_dir=` and `offline=True`: the bench
    never invents its own loading path, so what it measures is what the app does. The
    `.origin.json` it writes carries the resolved sha, so `--doctor` and `model-status` would
    read that folder truthfully too.
    """
    target = SNAPSHOT_ROOT / sha / OCR_DIRNAME
    origin = target / ".origin.json"
    from animetext_studio.core.model_store import OCR_REQUIRED_FILES

    if origin.is_file() and not any(
            not (target / f).is_file() for f in OCR_REQUIRED_FILES):
        return target.parent
    bundle = resolve_models_dir()
    if use_bundle and bundled_revision(bundle) == sha and ModelStore(bundle).has_ocr_snapshot():
        if not quiet:
            print(f"  {sha[:12]} is the vendored bundle — no download")
        return bundle
    from huggingface_hub import snapshot_download                # lazy

    # Reuse is only safe because the Hub tells us each file's hash, so "we have this" is a
    # verified statement and not a filename match. Without a fingerprint we download.
    reused: list[str] = []
    if reuse and revision_files:
        for name, (size, digest) in wanted_at(revision_files).items():
            dest = target / name
            if dest.is_file() and (not size or dest.stat().st_size == size):
                continue                                   # already sitting in the target
            found = find_reusable(name, size, digest, reuse_sources())
            if found is not None:
                how = link_or_copy(found, dest)
                reused.append(f"{name} {how} from {found.parent.name} (hash verified)")
    if revision_files:
        needed = [n for n in sorted(wanted_at(revision_files)) if not (target / n).is_file()]
    else:
        needed = sorted(SNAPSHOT_ALLOW)
    if not quiet:
        for line in reused:
            print(f"  reusing {line}", flush=True)
        if needed:
            print(f"  staging {sha[:12]} from {OCR_REPO}: "
                  f"{len(needed)} file(s) still to fetch ...", flush=True)
    # `local_dir` is asked for explicitly: the HF cache would deduplicate against the pinned
    # snapshot, but the engine loads a *directory*, and this one has to hold exactly one
    # revision. `tqdm_class` silences the progress bar without touching a keyword the
    # installed hub version may not have (`quiet` was added later and is not universal).
    bar = quiet_bar_class() if quiet else None
    kwargs = {"tqdm_class": bar} if bar is not None else {}
    try:
        if needed:
            snapshot_download(repo_id=OCR_REPO, revision=sha, allow_patterns=needed,
                              local_dir=str(target), token=token, **kwargs)
    except (OSError, RuntimeError) as exc:
        # Disk full, a per-file size limit, or a read-only mount. The plain HTTP path raises an
        # OSError (`File too large`); the xet transfer backend wraps the same IO failure in a
        # RuntimeError, so both are caught here rather than leaving the operator with a
        # traceback about a library they did not choose. None of the three is an internal error,
        # and the files that did arrive stay put: the next attempt resumes from them.
        raise SystemExit(
            f"cannot stage {sha[:12]}: {exc}\n"
            f"  target: {target}\n"
            f"  the snapshot needs ~680 MB free, and no per-file write limit below that "
            f"(`ulimit -f`). Files that did download are kept, so a retry resumes; or bench a "
            f"revision the bundle already carries, which downloads nothing."
        ) from None
    origin.write_text(json.dumps({
        "repo_id": OCR_REPO, "revision": sha, "source": "huggingface.co",
        "staged_by": "scripts/bench_models.py",
        "note": "a benchmark side, not the app bundle. Re-created on demand; delete freely.",
    }, indent=2) + "\n", encoding="utf-8")
    return target.parent


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def config_for_bench(cfg_dict: Optional[dict], device: str, provider: Optional[str],
                     models_dir: Optional[str]) -> AppConfig:
    """The app's own defaults, with only the bench's three overrides applied."""
    cfg = AppConfig.from_dict(cfg_dict) if cfg_dict else AppConfig.default()
    data = cfg.to_dict()
    data["ocr"] = {**data["ocr"], "device": device}
    data["paths"] = {**data["paths"], "offline": True,
                     "models_dir": models_dir or str(resolve_models_dir())}
    cfg = AppConfig.from_dict(data)
    return cfg


def bench_settings(cfg: AppConfig) -> dict[str, Any]:
    """The reading settings a run was made under — the things that change boxes or crops.

    Written into both the profile and the report, and compared between them, because a diff
    between a run at `crop_padding_px: 2` and a run at `4` is not a model comparison.
    """
    return {"ocr": asdict(cfg.ocr),
            "detection": asdict(cfg.detection),
            "reading_order": cfg.pipeline.reading_order,
            "min_box_area_px": cfg.pipeline.min_box_area_px}


def build_pair(cfg: AppConfig, ocr_models_dir: Optional[str], provider: Optional[str],
               revision: str = OCR_REVISION):
    """(DetectionEngine, OcrEngine): detection from the bundle, OCR from wherever is asked.

    `revision` is recorded on the engine (it reports itself in `status()` and in every
    document it produces). The *weights* come from `ocr_models_dir`, which is the app bundle
    for the pinned side and the staged snapshot for the candidate: `OcrEngine` loads a local
    folder when it has one, so no app code needs a revision argument to bench a revision.
    """
    from animetext_studio.engines.detection import DetectionEngine
    from animetext_studio.engines.ocr import OcrEngine

    bundle = str(resolve_models_dir())
    detection = DetectionEngine(hf_token=discover_hf_token(),
                                hf_cache_dir=cfg.paths.hf_cache_dir,
                                provider=provider or "auto", models_dir=bundle, offline=True)
    ocr = OcrEngine(device=cfg.ocr.device, models_dir=ocr_models_dir or bundle,
                    offline=True, revision=revision)
    return detection, ocr


def ocr_side_record(sha: Optional[str], models_dir: Path,
                    device_seen: Optional[str] = None) -> dict[str, Any]:
    """What an OCR side really was: resolved sha, the folder it loaded, the weights' sha256.

    The weights hash is the part that survives a Hub rewrite: if two reports name different
    revisions but the same `weights_sha256`, they measured the same model, and the sha256 is
    how you find that out six months later without re-downloading anything.
    """
    weights = Path(models_dir) / OCR_DIRNAME / "model.safetensors"
    return {
        "repo": OCR_REPO,
        "revision": sha,
        "models_dir": str(models_dir),
        "weights_sha256": sha256_file(weights) if weights.is_file() else None,
        "weights_bytes": weights.stat().st_size if weights.is_file() else None,
        "device_seen": device_seen,
    }


# ---------------------------------------------------------------------------
# `profile` — one pass over the corpus with the pinned weights
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}

#: Pages one worker process reads before it is replaced. Two reasons, one of them universal:
#: a long-lived worker accumulates allocator state that never comes back (decision D62 measured
#: that the app's own memory has a floor it will not give up), and a corpus of thousands of
#: pages should not be one process's lifetime job. The other is environmental: sandboxes and
#: CI runners set `RLIMIT_CPU`, and a worker that outlives its CPU budget takes the whole pool
#: down with `BrokenProcessPool` — a confusing way to lose ten minutes of work.
PROFILE_MAX_TASKS_PER_CHILD = 10


def _worker_init(cfg_dict: dict, provider: Optional[str], device: str,
                 ocr_dir: Optional[str], revision: str) -> None:
    """Per-process engine pair. Spawn, not fork: torch and ONNX Runtime both keep
    process-wide state that does not survive a fork cleanly, and a worker that quietly
    shares a half-initialised session would produce timings and text worth nothing."""
    cfg = config_for_bench(cfg_dict, device, provider, None)
    detection, ocr = build_pair(cfg, ocr_dir, provider, revision)
    from animetext_studio.core.pipeline import Pipeline

    _WORKER["pipeline"] = Pipeline(detection, ocr, cfg)
    _WORKER["config"] = cfg


def _profile_one(path_str: str) -> dict[str, Any]:
    """Detect + OCR one page with the pinned weights. Runs in a worker process.

    The whole result document is kept, not a summary of it: the profile doubles as the
    pinned side of a later comparison, and `reocr_blocks` needs a real document to take
    boxes from. Reconstructing a plausible-looking one from a few fields would mean the
    bench measured against a document the app never wrote, which is exactly the kind of
    approximation this script is supposed to remove.
    """
    from animetext_studio.core.pipeline import load_image

    path = Path(path_str)
    started = datetime.now().isoformat(timespec="seconds")
    try:
        image, info = load_image(path)
    except Exception as exc:                                     # noqa: BLE001
        return {"file": path.name, "path": str(path), "status": "unreadable",
                "error": f"{type(exc).__name__}: {exc}", "started_at": started}
    pipeline = _WORKER["pipeline"]
    t0 = time.monotonic()
    try:
        result = pipeline.run(image, info=info)
    except Exception as exc:                                     # noqa: BLE001
        return {"file": path.name, "path": str(path), "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "started_at": started}
    doc = result.to_dict()
    rec = page_identity(path)
    rec.update({
        "status": "ok",
        "started_at": started,
        "wall_ms": int((time.monotonic() - t0) * 1000),
        "timing_ms": doc["timing_ms"],
        "document": doc,
    })
    rec["features"] = page_features(rec["width"], rec["height"], doc["blocks"])
    rec["strata"] = strata_of(rec["features"])
    rec["layout"], rec["tier"] = rec["strata"]["layout"], rec["strata"]["tier"]
    return rec


def cmd_profile(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        raise SystemExit(f"corpus directory not found: {corpus}")
    pages = sorted(p for p in corpus.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if args.limit:
        pages = pages[args.offset:args.offset + args.limit]
    elif args.offset:
        pages = pages[args.offset:]
    if not pages:
        raise SystemExit(f"no images found in {corpus}")

    cfg = config_for_bench(None, args.device, args.provider, None)
    if args.pinned == "pinned":
        pinned = {"spec": "pinned", "sha": OCR_REVISION, "is_head": False,
                  "title": "the revision the app pins (core/types.py)", "date": None}
        if args.resolve:
            pinned = resolve_revision(hub_api(args.token), OCR_REVISION)
    else:
        pinned = resolve_revision(hub_api(args.token), args.pinned)
    bundle = resolve_models_dir()
    # Which snapshot the baseline side really loads. "pinned" means the bundle that is on disk,
    # so the bundle's own .origin.json is asked what revision it is — and if that disagrees
    # with the constant in core/types.py, the profile says so instead of labelling the run
    # with a revision it never touched.
    ocr_dir: Optional[str] = None
    if args.pinned == "pinned":
        on_disk = bundled_revision(bundle)
        if on_disk and on_disk != pinned["sha"]:
            print(f"note: the bundle on disk is {on_disk[:12]}, core/types.py pins "
                  f"{pinned['sha'][:12]}. The profile records the bundle's revision.")
            pinned = {**pinned, "sha": on_disk}
    else:
        ocr_dir = str(stage_snapshot(pinned["sha"], args.token, quiet=args.quiet))
    print(f"corpus   : {corpus}  ({len(pages)} pages)")
    print(f"baseline : {pinned['sha']}  ({pinned.get('title') or 'no commit title'})")
    where = f"staged snapshot {ocr_dir}" if ocr_dir else f"the app bundle: {bundle}"
    print(f"           {where}")
    print(f"detector : {cfg.detection.model} @ {bundle}")
    print(f"device   : ocr={args.device}  onnx={args.provider or 'auto'}  jobs={args.jobs}")

    # Resume: a 239-page pass is ten minutes of CPU, and a ten-minute job gets interrupted.
    # A page counts as done when the same bytes were read at the same baseline revision with
    # the same reading settings — matched on content hash, so a renamed or re-saved page is
    # measured again and a copy of the corpus somewhere else is not.
    keep: list[dict[str, Any]] = []
    out = Path(args.out)
    if args.resume and out.is_file():
        earlier = load_profile(out)
        if (earlier.get("pinned") or {}).get("revision") != pinned["sha"]:
            raise SystemExit(f"{out} was profiled at baseline "
                             f"{(earlier.get('pinned') or {}).get('revision')}, not "
                             f"{pinned['sha']}. Resume only works against the same baseline.")
        if earlier.get("settings") != bench_settings(cfg):
            print("note: the reading settings differ from the stored profile — resumed pages "
                  "keep their old numbers, new pages get the new ones. Re-profile without "
                  "--resume if you want one coherent baseline.")
        done = {p["sha256"]: p for p in earlier.get("pages", []) if p.get("status") == "ok"}
        fresh: list[Path] = []
        for page in pages:
            record = done.get(sha256_file(page))
            if record is None:
                fresh.append(page)
            else:
                keep.append(record)
        pages = fresh
        print(f"resume   : {len(keep)} page(s) already measured, {len(pages)} to go")
        if not pages:
            write_profile(out, corpus, keep, pinned, ocr_dir, cfg)
            return 0

    records: list[dict[str, Any]] = []
    if args.jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        kwargs: dict[str, Any] = {}
        if sys.version_info >= (3, 11):        # `max_tasks_per_child` is 3.11+
            kwargs["max_tasks_per_child"] = PROFILE_MAX_TASKS_PER_CHILD
            # A worker can still die (OOM, a CPU-time cap, a segfault in a backend). The pages
            # already read are real measurements and must not go with it, so the interrupted
            # run writes its profile and tells the operator to continue with --resume.
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.jobs, mp_context=ctx, initializer=_worker_init,
                    initargs=(cfg.to_dict(), args.provider, args.device, ocr_dir,
                              pinned["sha"]), **kwargs) as pool:
                try:
                    for i, rec in enumerate(pool.map(_profile_one, [str(p) for p in pages]), 1):
                        records.append(rec)
                        _progress(i, len(pages), rec)
                except concurrent.futures.process.BrokenProcessPool as exc:
                    write_profile(out, corpus, keep + records, pinned, ocr_dir, cfg)
                    raise SystemExit(
                        f"\nthe worker pool died ({exc}) after {len(records)} of {len(pages)} "
                        f"pages.\n  what was measured is saved in {out}\n"
                        f"  continue with the same command plus --resume"
                    ) from None
    else:
        _worker_init(cfg.to_dict(), args.provider, args.device, ocr_dir, pinned["sha"])
        for i, page in enumerate(pages, 1):
            rec = _profile_one(str(page))
            records.append(rec)
            _progress(i, len(pages), rec)

    write_profile(out, corpus, keep + records, pinned, ocr_dir, cfg)
    return 0


def write_profile(out: Path, corpus: Path, records: list[dict[str, Any]], pinned: dict,
                  ocr_dir: Optional[str], cfg: AppConfig) -> None:
    """Write the profile document. Split out so `--resume` can finish a run without re-measuring."""
    records = sorted(records, key=lambda r: r["file"])
    profile = {
        "kind": PROFILE_KIND,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": str(corpus),
        "app": app_identity(),
        "environment": environment(),
        "settings": bench_settings(cfg),
        "pinned": {**ocr_side_record(pinned["sha"], Path(ocr_dir or resolve_models_dir())),
                   "requested_as": pinned["spec"],
                   "commit_title": pinned.get("title"), "commit_date": pinned.get("date")},
        "counts": {
            "pages": len(records),
            "ok": sum(1 for r in records if r["status"] == "ok"),
            "failed": sum(1 for r in records if r["status"] != "ok"),
            "layouts": _tally(r.get("layout") for r in records),
            "tiers": _tally(r.get("tier") for r in records),
        },
        "pages": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}  ({profile['counts']['ok']} ok, {profile['counts']['failed']} failed)")
    print("tiers   :", json.dumps(profile["counts"]["tiers"]))
    print("layouts :", json.dumps(profile["counts"]["layouts"]))


def bundled_revision(bundle: Path) -> Optional[str]:
    """The revision recorded in the bundle's own `.origin.json`, or None if it does not say."""
    origin = bundle / OCR_DIRNAME / ".origin.json"
    if not origin.is_file():
        return None
    try:
        return json.loads(origin.read_text(encoding="utf-8")).get("revision")
    except (OSError, ValueError):
        return None


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _progress(i: int, total: int, rec: dict[str, Any]) -> None:
    if rec["status"] != "ok":
        print(f"[{i}/{total}] {rec['file']:<24} {rec['status']}: {rec.get('error','')}",
              flush=True)
        return
    f = rec["features"]
    print(f"[{i}/{total}] {rec['file']:<24} {f['blocks']:>3} blk "
          f"{f['chars']:>5} ch  {rec['timing_ms']['total']:>6} ms  "
          f"{rec['layout']:<8} {rec['tier']}", flush=True)


# ---------------------------------------------------------------------------
# `panel` — show the sample without running anything
# ---------------------------------------------------------------------------

def load_profile(path: Path) -> dict[str, Any]:
    """Read a profile, and re-derive every page's strata from its stored features.

    The profile is a measurement; the layout label and the difficulty tier are an
    interpretation of it. Recomputing the interpretation on read is what lets the bands and
    ramp endpoints above be retuned without asking anyone to spend ten minutes re-measuring a
    corpus they already measured — and it means a profile written by an older version of this
    script is still a valid input, as long as the features it stored are the ones we read.
    """
    if not path.is_file():
        raise SystemExit(f"no profile at {path}. Run: python scripts/bench_models.py profile")
    profile = json.loads(path.read_text(encoding="utf-8"))
    for page in profile.get("pages", []):
        features = page.get("features")
        if not features:
            continue
        page["strata"] = strata_of(features)
        page["layout"], page["tier"] = page["strata"]["layout"], page["strata"]["tier"]
    counts = profile.get("counts")
    if isinstance(counts, dict):
        counts["layouts"] = _tally(p.get("layout") for p in profile.get("pages", []))
        counts["tiers"] = _tally(p.get("tier") for p in profile.get("pages", []))
    return profile


def cmd_panel(args: argparse.Namespace) -> int:
    profile = load_profile(Path(args.profile))
    pages = profile["pages"]
    chosen = pick_panel(pages, args.size, args.seed)
    panel = {
        "kind": PANEL_KIND,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": str(Path(args.profile)),
        "profile_created_at": profile.get("created_at"),
        "corpus": profile.get("corpus"),
        "seed": args.seed,
        "size": args.size,
        "rule": "round-robin over difficulty tiers, layout-diverse inside each tier",
        "pages": [{"file": p["file"], "sha256": p["sha256"], "tier": p["tier"],
                   "layout": p["layout"], "difficulty": (p.get("strata") or {}).get("score"),
                   "blocks": p["features"]["blocks"]} for p in chosen],
    }
    print(f"{'file':<26} {'tier':<9} {'layout':<11} {'blk':>4} {'chars':>6} "
          f"{'min px':>6} {'score':>6}  why")
    for p in chosen:
        f, s = p["features"], p.get("strata", {})
        worst = sorted((s.get("parts") or {}).items(), key=lambda kv: -kv[1])[:2]
        why = ", ".join(f"{k}={v:.2f}" for k, v in worst if v > 0) or "-"
        print(f"{p['file']:<26} {p['tier']:<9} {p['layout']:<11} {f['blocks']:>4} "
              f"{f['chars']:>6} {f['min_side_px']:>6} {s.get('score', 0):>6.3f}  {why}")
    print(f"\n{len(chosen)} of {sum(1 for p in pages if p.get('status') == 'ok')} pages, "
          f"seed={args.seed}, tiers={_tally(p['tier'] for p in chosen)}")
    print(f"layouts: {json.dumps(_tally(p['layout'] for p in chosen))}")
    if not args.dry_run:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(panel, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# `run` — the comparison
# ---------------------------------------------------------------------------

def _panel_pages(profile: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    ok = [p for p in profile["pages"] if p.get("status") == "ok"]
    if args.pages:
        wanted = set(args.pages)
        found = [p for p in ok if p["file"] in wanted]
        missing = sorted(wanted - {p["file"] for p in found})
        if missing:
            raise SystemExit(f"not in the profile: {', '.join(missing)}"
                             "  (profile the corpus again, or check the name)")
        return sorted(found, key=lambda p: p["file"])
    if args.all:
        return sorted(ok, key=lambda p: p["file"])
    return pick_panel(ok, args.size, args.seed)


def _blocks_of(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"id": b["id"], "bbox": b["bbox"], "score": b["score"], "text": b["text"],
             "error": b["error"], "flags": b["flags"], "contained_in": b["contained_in"]}
            for b in doc["blocks"]]


def cmd_run(args: argparse.Namespace) -> int:
    profile = load_profile(Path(args.profile))
    panel = _panel_pages(profile, args)
    if not panel:
        raise SystemExit("the panel is empty")
    pinned_sha = profile["pinned"]["revision"] or OCR_REVISION
    if pinned_sha != OCR_REVISION:
        print(f"note: the profile was built at {pinned_sha[:12]}, the app now pins "
              f"{OCR_REVISION[:12]} — the pinned side of this report is the profile's.")

    # Every Hub question is asked here, before a single engine is built. Loading the OCR
    # engine puts the process into HF offline mode (that is the app's own doing, decision D26,
    # and it is correct for the app), so an API call made after the first page is read fails
    # with "offline mode is enabled" — a confusing way to lose a run that is halfway done.
    api = hub_api(args.token)
    candidates = [resolve_revision(api, spec) for spec in args.against]
    pinned_fp = file_fingerprint(api, pinned_sha)
    candidate_fps = {c["sha"]: file_fingerprint(api, c["sha"]) for c in candidates}
    file_diffs = {sha: revision_difference(pinned_fp, fp)
                  for sha, fp in candidate_fps.items()}
    for c in candidates:
        # The fingerprint travels with the candidate: staging uses those recorded hashes to
        # recognise files this machine already holds, which is the difference between a
        # README-only bump costing 0 bytes and costing 633 MB.
        c["files"] = candidate_fps[c["sha"]]

    cfg = config_for_bench(None, args.device, args.provider, None)
    print(f"panel    : {len(panel)} page(s)"
          + (f"  seed={args.seed} size={args.size}" if not args.pages and not args.all else ""))
    print(f"pinned   : {pinned_sha}")
    for c in candidates:
        print(f"candidate: {c['sha']}  ({c.get('title') or 'no commit title'}, {c.get('date')})")

    noise: Optional[dict[str, Any]] = None
    if args.noise_floor:
        # Read the panel twice with the *same* weights. Whatever moves is not a model change:
        # it is the harness, the threads, or the model's own run-to-run variance. Every number
        # in the report below is only meaningful against that floor, which is why it is printed
        # first and stored in every report rather than left as an exercise.
        print("\nnoise floor: reading every page twice with the pinned snapshot")
        self_report = run_one(args, profile, panel, cfg, pinned_sha,
                              {"spec": "pinned", "sha": pinned_sha,
                               "title": "the pinned snapshot, against itself", "date": None},
                              {"differing_files": [], "engine_files_differ": [],
                               "weights_identical": True}, noise=None,
                              # Its own file: the self-test is evidence, and it would otherwise
                              # be overwritten a moment later by the run it qualifies.
                              out=str(Path(args.out).with_suffix(".selftest.json"))
                              if args.out else None)
        noise = dict(self_report["summary"])
        noise["reading"] = (f"{noise['changed_blocks']} of {noise['blocks_compared']} blocks "
                            f"moved on the same weights; a candidate change smaller than that "
                            f"is not a finding")
        print(f"noise floor: {noise['reading']}")

    reports = []
    for cand in candidates:
        diff = file_diffs[cand["sha"]]
        if cand["sha"] == pinned_sha:
            print(f"\ncandidate {cand['sha'][:12]} IS the pinned snapshot: this run is a "
                  f"self-test, and every row it prints is a bug in this harness or a fact "
                  f"about run-to-run variance.")
        elif not diff["engine_files_differ"] and not args.force:
            print(f"\ncandidate {cand['sha'][:12]} differs from the pinned snapshot only in "
                  f"{', '.join(diff['differing_files']) or 'nothing'}.")
            print("The weights and the code are byte-identical, so a run would measure "
                  "scheduler noise. Pass --force to run anyway.")
            continue
        reports.append(run_one(args, profile, panel, cfg, pinned_sha, cand, diff, noise=noise))
    # Declining to run a comparison that cannot show anything is a successful answer to the
    # question, not a failure: the operator asked "did the update help?" and was told "there
    # is no update to measure". Exit 0, so a scheduled check does not light up red.
    if not reports:
        print("\nnothing to compare: every candidate is the engine you already have.")
    return 0


def run_one(args: argparse.Namespace, profile: dict[str, Any], panel: list[dict],
            cfg: AppConfig, pinned_sha: str, cand: dict[str, Any],
            file_diff: dict[str, Any], *, noise: Optional[dict[str, Any]] = None,
            out: Optional[str] = None) -> dict[str, Any]:
    """Read every page of the panel twice, once per snapshot, and diff the two documents.

    **Both sides go through the same door**: a full `Pipeline.run()` each, on one shared
    detection engine. An earlier revision of this harness read the pinned side from the
    profile and the candidate side with `Pipeline.reocr_blocks`, on the theory that re-reading
    the pinned boxes is cheaper; it was cheaper, and it was wrong. `reocr_blocks` re-crops from
    the stored bbox, which the pipeline has already padding-adjusted, so its crops come out
    `2 * crop_padding_px` larger than the crops the original run fed to the model — and on the
    first self-test (a snapshot compared against itself, which must be all-SAME by
    construction) that produced three changed blocks on a 14-block page. A harness whose
    baseline is a different code path from its candidate measures the harness. So the boxes
    here are detected twice, on the same session, and any structural difference that shows up
    in the columns is one the app really produced.
    """
    from animetext_studio.core.pipeline import Pipeline, load_image

    cand_dir = stage_snapshot(cand["sha"], args.token, quiet=args.quiet,
                              reuse=not args.no_reuse, revision_files=cand.get("files"))
    bundle = resolve_models_dir()
    print(f"\nloading engines (pinned {pinned_sha[:12]} | candidate {cand['sha'][:12]})")
    t_load = time.monotonic()
    detection, ocr_pinned = build_pair(cfg, str(bundle), args.provider, pinned_sha)
    _, ocr_cand = build_pair(cfg, str(cand_dir), args.provider, cand["sha"])
    pinned_pipeline = Pipeline(detection, ocr_pinned, cfg)
    cand_pipeline = Pipeline(detection, ocr_cand, cfg)
    load_ms = int((time.monotonic() - t_load) * 1000)

    use_cache = (args.reuse_pinned
                 and profile["pinned"].get("revision") == pinned_sha
                 and _settings_match(profile, cfg))
    cached = {p["file"]: p for p in profile["pages"]} if use_cache else {}
    print("pinned side: "
          + ("the profile's results — same code path, but a different session, so the timing "
             "columns are not comparable (--reuse-pinned)" if use_cache else
             "read again in this session, page by page, next to the candidate"))

    diffs: list[PageDiff] = []
    detail: list[dict[str, Any]] = []
    device_seen: set[str] = set()
    for i, page in enumerate(panel, 1):
        image, info = load_image(page["path"])
        if page["file"] in cached:
            base_doc = cached[page["file"]]["document"]
        else:
            base_doc = pinned_pipeline.run(image, info=info).to_dict()
        cand_doc = cand_pipeline.run(image, info=info).to_dict()
        device_seen.add(str(cand_doc["ocr"].get("device")))
        d = diff_blocks(_blocks_of(base_doc), _blocks_of(cand_doc))
        d.file, d.tier, d.layout = page["file"], page.get("tier", ""), page.get("layout", "")
        strata = page.get("strata") or {}
        d.score = float(strata.get("score") or 0.0)
        d.parts = dict(strata.get("parts") or {})
        d.base_ms, d.cand_ms = int(base_doc["timing_ms"].get("ocr") or 0), \
            int(cand_doc["timing_ms"].get("ocr") or 0)
        diffs.append(d)
        detail.append(_page_detail(page, _blocks_of(base_doc), _blocks_of(cand_doc),
                                   d, args.no_text))
        if not args.quiet:
            print(f"[{i}/{len(panel)}] {_row(d, args.width)}", flush=True)

    devices = ",".join(sorted(d for d in device_seen if d and d != "None")) or None
    report = build_report(args, profile, panel, cfg, pinned_sha, cand, file_diff,
                          diffs, detail, load_ms, use_cache,
                          ocr_side_record(pinned_sha, bundle, devices),
                          ocr_side_record(cand["sha"], cand_dir, devices), noise)
    out = Path(out) if out else (Path(args.out) if args.out else default_report_path(cand["sha"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print_table(diffs, args.only_changed, args.width)
    print(f"\nwrote {out}")
    return report


def _settings_match(profile: dict[str, Any], cfg: AppConfig) -> bool:
    """Can the profile's pinned results be compared against a run made today at all?

    Same boxes only means the same *reading settings* too: a profile made with a different
    crop padding, a different detector or a different `max_new_tokens` is a different
    experiment, and silently mixing the two would produce a diff that means nothing. The
    comparison is over everything in `bench_settings` except the OCR device, which changes
    speed but not text.
    """
    stored, now = profile.get("settings") or {}, bench_settings(cfg)

    def strip(section: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in (section or {}).items() if k != "device"}

    for key in ("detection", "min_box_area_px", "reading_order"):
        if stored.get(key) != now.get(key):
            return False
    return strip(stored.get("ocr")) == strip(now.get("ocr"))


def _page_detail(page: dict, base: list[dict], cand: list[dict], d: PageDiff,
                 no_text: bool) -> dict[str, Any]:
    """Per-block text for the report — the part a human actually reads, when they asked for it."""
    by_cand = {b["id"]: b for b in cand}
    rows = []
    for b in base:
        c = by_cand.get(b["id"], {})
        if b.get("text") == c.get("text"):
            continue
        row = {"id": b["id"], "bbox": b["bbox"],
               "pinned": _text_or_hash(b.get("text"), keep=not no_text),
               "candidate": _text_or_hash(c.get("text"), keep=not no_text),
               "cer": round(block_cer(b.get("text"), c.get("text")), 4)}
        rows.append(row)
    return {"file": page["file"], "tier": page.get("tier"), "layout": page.get("layout"),
            "diff": d.to_dict(), "changed_blocks": rows}


def _text_or_hash(text: Optional[str], *, keep: bool) -> Optional[str]:
    if text is None:
        return None
    if keep:
        return text
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_report(args, profile, panel, cfg, pinned_sha, cand, file_diff, diffs, detail,
                 load_ms, used_cache, pinned_record, cand_record, noise=None) -> dict[str, Any]:
    changed = [d for d in diffs if d.verdict == "CHANGED"]
    suspect = [d for d in diffs if d.verdict == "SUSPECT"]
    same = [d for d in diffs if d.verdict == "SAME"]
    total_changed_blocks = sum(d.changed for d in diffs)
    ratios = [d.time_ratio for d in diffs if d.time_ratio]
    return {
        "kind": REPORT_KIND,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "app": app_identity(),
        "environment": environment(),
        "settings": bench_settings(cfg),
        "sides": {
            "pinned": {**pinned_record, "label": "pinned",
                       "results_from": "profile cache" if used_cache else "this session"},
            "candidate": {**cand_record, "label": "candidate",
                          "requested_as": cand["spec"], "commit_title": cand.get("title"),
                          "commit_date": cand.get("date")},
        },
        "revision_file_diff": file_diff,
        "method": {
            "detection": "run per page, per side, on one shared detection engine in one "
                         "session: both sides enter through Pipeline.run(), because a baseline "
                         "taken through a different code path measures that path (see run_one)",
            "structural_columns": "a canary, not a score: the detector and its weights are the "
                                  "same on both sides, so a moved box means something upstream "
                                  "of the OCR moved with it",
            "changed_blocks": "raw text inequality per block",
            "cer": "normalised (NFKC, whitespace-collapsed) edit distance over the pinned text, "
                   "averaged across the blocks that changed. Neither side is ground truth.",
            "timing": "OCR-stage wall time per page"
                      + ("; the pinned column comes from the profile session, so ratios are "
                         "cross-session and only good for an order of magnitude"
                         if used_cache else "; both sides timed in this session, page by page"),
            "verdict_rule": "SUSPECT if boxes moved, text vanished, or a block gained an error; "
                            "else CHANGED if any block's text differs; else SAME",
            "load_ms": load_ms,
        },
        "noise_floor": noise,
        "panel": {"size": len(panel), "seed": None if (args.pages or args.all) else args.seed,
                  "tiers": _tally(d.tier for d in diffs),
                  "layouts": _tally(d.layout for d in diffs),
                  "pages": [p["file"] for p in panel]},
        "summary": {
            "pages": len(diffs),
            "same": len(same), "changed": len(changed), "suspect": len(suspect),
            "blocks_compared": sum(d.blocks for d in diffs),
            "changed_blocks": total_changed_blocks,
            "blocks_lost": sum(d.lost for d in diffs),
            "blocks_gained": sum(d.gained for d in diffs),
            "geometry_moved": sum(d.geometry_moved for d in diffs),
            "cer_mean_changed": round(statistics.fmean([d.cer for d in diffs if d.changed]), 4)
            if total_changed_blocks else 0.0,
            "time_ratio_median": round(statistics.median(ratios), 3) if ratios else None,
            "total_ms": {"pinned": sum(d.base_ms for d in diffs),
                         "candidate": sum(d.cand_ms for d in diffs)},
        },
        "verdict": None,
        "verdict_note": "No upgrade is asserted here. This harness can say that text changed, "
                        "how much, and whether anything structural broke; whether the new text "
                        "is better Japanese is not computable from a diff and is left to the "
                        "person who reads the language.",
        "pages": detail,
    }


def default_report_path(sha: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPORT_DIR / f"{stamp}-{sha[:12]}.json"


def _row(d: PageDiff, width: int) -> str:
    ratio = f"{d.time_ratio:.2f}x" if d.time_ratio else "  -  "
    return (f"{d.file:<{width}} {d.tier:<8} {d.verdict:<8} blk={d.blocks:<3} "
            f"chg={d.changed:<3} cer={d.cer:.3f} lost={d.lost} "
            f"t={d.base_ms:>6}/{d.cand_ms:<6} {ratio}")


def print_table(diffs: Sequence[PageDiff], only_changed: bool, width: int) -> None:
    rows = [d for d in diffs if not only_changed or d.verdict != "SAME"]
    print()
    print(f"{'page':<{width}} {'tier':<8} {'verdict':<8} {'blocks':<7} {'changed':<8} "
          f"{'CER':<6} {'lost':<5} {'ms pinned/cand':<16} {'ratio':<6}")
    print("-" * (width + 72))
    for d in rows:
        print(_row(d, width))
    if not rows:
        print("(no rows: every page read identically by both snapshots)")
    total_changed = sum(d.changed for d in diffs)
    print("-" * (width + 72))
    print(f"{len(diffs)} page(s): "
          f"{sum(1 for d in diffs if d.verdict == 'SAME')} SAME, "
          f"{sum(1 for d in diffs if d.verdict == 'CHANGED')} CHANGED, "
          f"{sum(1 for d in diffs if d.verdict == 'SUSPECT')} SUSPECT; "
          f"{total_changed} of {sum(d.blocks for d in diffs)} blocks differ in text")
    for d in diffs:
        for reason in d.reasons:
            print(f"  ! {d.file}: {reason}")
    print("\nThis is a diff, not a verdict: whether the changed text is better is a "
          "judgement about Japanese, and it stays yours.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bench_models.py", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="auto", help="OCR device (auto|cpu|cuda[:N])")
        p.add_argument("--provider", default=None,
                       help="ONNX Runtime provider for detection (auto|cpu|cuda)")
        p.add_argument("--hf-token", dest="token", default=None,
                       help="HF token for the candidate download "
                            "(default: $HF_TOKEN / cached login)")
        p.add_argument("--quiet", action="store_true", help="no per-page lines")

    prof = sub.add_parser("profile", help="measure the corpus once with the pinned weights")
    prof.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    prof.add_argument("--out", default=str(PROFILE_PATH))
    prof.add_argument("--jobs", type=int, default=1,
                      help="worker processes (each loads its own weights: ~3 GB and a core set); "
                           "2-4 on a normal machine, 1 if you want to watch it page by page")
    prof.add_argument("--limit", type=int, default=0)
    prof.add_argument("--offset", type=int, default=0)
    prof.add_argument("--pinned", default="pinned",
                      help="'pinned' (the app's OCR_REVISION) or a revision to treat as pinned")
    prof.add_argument("--resolve", action="store_true",
                      help="ask the Hub what the pinned revision is (needs network)")
    prof.add_argument("--resume", action="store_true",
                      help="keep the pages already in --out (same bytes, same baseline, same "
                           "settings) and measure only the rest")
    common(prof)
    prof.set_defaults(func=cmd_profile)

    pan = sub.add_parser("panel", help="show the sample a run would use")
    pan.add_argument("--profile", default=str(PROFILE_PATH))
    pan.add_argument("-n", "--size", type=int, default=12)
    pan.add_argument("--seed", type=int, default=1)
    pan.add_argument("--out", default=str(PANEL_PATH))
    pan.add_argument("--dry-run", action="store_true", help="print only, write nothing")
    pan.set_defaults(func=cmd_panel)

    run = sub.add_parser("run", help="compare the pinned snapshot against one or more revisions")
    run.add_argument("--profile", default=str(PROFILE_PATH))
    run.add_argument("--against", action="append", required=True, metavar="REV",
                     help="revision to compare against pinned; repeatable. "
                          "'latest' resolves to the current Hub HEAD and is recorded as a full sha")
    run.add_argument("-n", "--size", type=int, default=12, help="panel size (default 12)")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--pages", nargs="*", default=None, help="exact pages instead of a sample")
    run.add_argument("--all", action="store_true", help="every profiled page")
    run.add_argument("--out", default=None, help=f"report path (default {REPORT_DIR}/...)")
    run.add_argument("--only-changed", action="store_true", help="hide the SAME rows")
    run.add_argument("--no-text", action="store_true",
                     help="write text hashes instead of text into the report")
    run.add_argument("--reuse-pinned", action="store_true",
                     help="take the pinned side from the profile instead of reading it again "
                         "now: same code path, different session, so the timing columns stop "
                         "being comparable (the text columns stay valid)")
    run.add_argument("--noise-floor", action="store_true",
                     help="first read every page twice with the PINNED snapshot and report how "
                         "much it disagreed with itself: that number is the resolution limit "
                         "of everything below it")
    run.add_argument("--force", action="store_true",
                     help="run even when the two revisions' engine files are byte-identical")
    run.add_argument("--no-reuse", action="store_true",
                     help="download the candidate from scratch instead of hard-linking files "
                          "this machine already holds at the same recorded hash")
    run.add_argument("--width", type=int, default=24, help="page-name column width")
    common(run)
    run.set_defaults(func=cmd_run)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if getattr(args, "token", None) is None:
        args.token = discover_hf_token()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
