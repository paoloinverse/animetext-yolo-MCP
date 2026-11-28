
"""P9.6 — `scripts/bench_models.py`: the deciding parts, tested without a model.

The harness has two halves. One touches torch, ONNX Runtime, the Hub and 600 MB of
weights; it can only be exercised with `-m models` on a machine that has them. The other
half is arithmetic that decides what an operator is about to read: what counts as a
changed block, what makes a page *hard*, which pages a seed picks, when a row is SUSPECT
instead of CHANGED, and what ends up in the JSON that outlives the session. That half is
pure, it is the half that can be wrong in a way that misleads a human, and it is all
testable with no weights, no network and no torch — so it runs in the default suite.

The rule the tests hold the script to is its own: **the harness may describe and must not
judge.** Anything that would call a candidate better, or quietly fold two different
measurements into one number, fails here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))        # the script under test is not a package

import bench_models as bench                          # noqa: E402
from animetext_studio.core.config import AppConfig    # noqa: E402


# -----------------------------------------------------------------------
# helpers: the smallest thing that is still a result document
# -----------------------------------------------------------------------

def block(bid=0, x0=0, y0=0, x1=40, y1=40, text="hi", score=0.8, error=None,
          flags=(), contained=None):
    return {"id": bid, "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "label": "text_block",
            "score": score, "text": text, "reading_order": bid, "color": "#FF0000",
            "error": error, "flags": list(flags), "contained_in": contained}


def page(name="001.jpg", status="ok", **over):
    rec = {"file": name, "path": f"corpus/{name}", "sha256": name * 4, "status": status,
           "width": 1000, "height": 1500, "features": dict(bench.page_features(
               1000, 1500, over.pop("blocks", [block()]))), }
    rec.update(over)
    strata = bench.strata_of(rec["features"])
    # An explicit tier/layout in `over` wins: a selection test should be about selection, and
    # not about whether synthetic boxes happen to land on the other side of a ramp.
    strata.update({k: over[k] for k in ("tier", "layout") if k in over})
    rec["strata"] = strata
    rec["layout"], rec["tier"] = strata["layout"], strata["tier"]
    return rec


def corpus(*names, tier=None, layout=None):
    """Pages of one stratum (or as close as synthetic boxes get)."""
    out = []
    for i, name in enumerate(names):
        rec = page(name)
        if tier or layout:
            # The panel reads the labels, not the reasoning behind them; overriding them
            # here keeps a strata test about selection rather than about the ramps.
            rec["tier"] = tier or rec["tier"]
            rec["layout"] = layout or rec["layout"]
        out.append(rec)
    return out


# -----------------------------------------------------------------------
# measuring text
# -----------------------------------------------------------------------

class TestMeasuringText:
    def test_normalisation_is_only_width_and_spacing(self):
        assert bench.norm_text("ＡＢＣ  １２３") == "ABC 123"
        assert bench.norm_text(None) == ""
        assert bench.norm_text("  a\tb\n") == "a b"

    def test_edit_distance_is_the_number_of_edits(self):
        assert bench.edit_distance("", "") == 0
        assert bench.edit_distance("abc", "abc") == 0
        assert bench.edit_distance("abc", "abd") == 1
        assert bench.edit_distance("", "abcd") == 4
        assert bench.edit_distance("ジャグシーバス", "ジャクシーバス") == 1

    def test_cer_of_two_empties_is_zero_not_an_error(self):
        # A block both sides read as nothing contributes nothing. A ZeroDivisionError here
        # would be a crash dressed as an edge case.
        assert bench.block_cer("", None) == 0.0

    def test_cer_is_measured_against_the_pinned_side(self):
        assert bench.block_cer("a", "a") == 0.0
        assert bench.block_cer("aaaa", "aaab") == pytest.approx(0.25)
        # Nothing -> something is a whole block of difference, and stays inside 0..1.
        assert bench.block_cer("", "hello") == 1.0

    def test_a_normalisation_only_change_measures_zero_but_still_counts(self):
        changed = ("ＡＢ", "AB")
        assert bench.block_cer(*changed) == 0.0
        d = bench.diff_blocks([block(text=changed[0])], [block(text=changed[1])])
        assert d.changed == 1 and d.cer == 0.0
        assert d.verdict == "CHANGED"
        assert any("normalisation" in r for r in d.reasons)


# -----------------------------------------------------------------------
# strata: what kind of page, and how hard
# -----------------------------------------------------------------------

class TestStrata:
    def test_layout_asks_the_questions_in_the_stated_order(self):
        assert bench.layout_of({"blocks": 0, "vertical_frac": 1, "text_area_frac": 0.9}) == "none"
        # three boxes are sparse whatever shape they are
        assert bench.layout_of({"blocks": 3, "vertical_frac": 1,
                                "text_area_frac": 0.9}) == "sparse"
        # orientation is asked before volume
        assert bench.layout_of({"blocks": 30, "vertical_frac": 0.5,
                                "text_area_frac": 0.9}) == "mixed"
        assert bench.layout_of({"blocks": 30, "vertical_frac": 0.0,
                                "text_area_frac": 0.05}) == "horizontal"
        # area before count: one slab is denser than it is crowded
        assert bench.layout_of({"blocks": 30, "vertical_frac": 1.0,
                                "text_area_frac": 0.5}) == "dense"
        assert bench.layout_of({"blocks": 30, "vertical_frac": 1.0,
                                "text_area_frac": 0.1}) == "crowded"
        assert bench.layout_of({"blocks": 12, "vertical_frac": 1.0,
                                "text_area_frac": 0.1}) == "columns"

    def test_every_layout_label_the_rules_can_produce_is_a_declared_one(self):
        for blocks in range(0, 40):
            for vf in (0.0, 0.5, 0.95, 1.0):
                for area in (0.0, 0.1, 0.3, 0.6):
                    got = bench.layout_of({"blocks": blocks, "vertical_frac": vf,
                                           "text_area_frac": area})
                    assert got in bench.LAYOUTS, got

    def test_difficulty_is_a_property_of_the_page_not_of_the_models_performance(self):
        """How well the pinned weights did must not move the difficulty score.

        If it did, a model update would re-sort the panel, and next month's seed would
        not be this month's panel — the one thing a regression harness cannot afford.
        """
        read = {"blocks": 20, "min_side_px": 10, "text_area_frac": 0.2,
                "blocks_contained": 0, "min_score": 0.1, "max_chars": 30,
                "blocks_with_text": 20, "blocks_empty": 0}
        silent = dict(read, blocks_with_text=0, blocks_empty=20)
        assert bench.difficulty_of(read) == bench.difficulty_of(silent)
        assert bench.tier_of(read) != bench.tier_of(silent)   # but it *is* `unread`
        assert bench.tier_of(silent) == "unread"

    def test_the_components_point_the_way_a_reader_would_guess(self):
        tiny = bench.difficulty_parts({"blocks": 4, "min_side_px": 8, "text_area_frac": 0.1,
                                       "blocks_contained": 0, "min_score": 0.9,
                                       "max_chars": 10})
        assert tiny["tiny"] == 1.0 and tiny["crowd"] == 0.0
        assert tiny["marginal"] == 0.0 and tiny["longest"] == 0.0
        wall = bench.difficulty_parts({"blocks": 40, "min_side_px": 60,
                                       "text_area_frac": 0.4, "blocks_contained": 12,
                                       "min_score": 0.01, "max_chars": 200})
        assert wall["crowd"] == 1.0 and wall["nested"] == 1.0 and wall["marginal"] == 1.0
        assert wall["tiny"] == 0.0

    def test_the_score_stays_in_its_range_for_any_page(self):
        for blocks in range(0, 60):
            for side in (0, 5, 40, 200):
                score, parts = bench.difficulty_of(
                    {"blocks": blocks, "min_side_px": side, "text_area_frac": side / 200,
                     "blocks_contained": side, "min_score": side / 100, "max_chars": side})
                assert 0.0 <= score <= 1.0
                assert all(0.0 <= v <= 1.0 for v in parts.values())

    def test_the_bands_are_ordered_and_exhaustive(self):
        cuts = [cut for cut, _ in bench.TIER_BANDS]
        assert cuts == sorted(cuts)
        names = [name for _, name in bench.TIER_BANDS] + ["brutal"]
        assert names == [t for t in bench.TIERS if t != "unread"]


# -----------------------------------------------------------------------
# the panel
# -----------------------------------------------------------------------

class TestPanel:
    def test_the_same_seed_picks_the_same_pages(self):
        pages = [page(f"{i:03d}.jpg") for i in range(60)]
        assert bench.pick_panel(pages, 12, 7) == bench.pick_panel(pages, 12, 7)

    def test_a_different_seed_picks_different_pages(self):
        pages = [page(f"{i:03d}.jpg") for i in range(60)]
        one = {p["file"] for p in bench.pick_panel(pages, 12, 1)}
        two = {p["file"] for p in bench.pick_panel(pages, 12, 2)}
        assert one != two

    def test_the_selection_does_not_depend_on_the_order_it_was_handed(self):
        pages = [page(f"{i:03d}.jpg") for i in range(40)]
        forward = [p["file"] for p in bench.pick_panel(pages, 10, 3)]
        back = [p["file"] for p in bench.pick_panel(list(reversed(pages)), 10, 3)]
        assert forward == back

    def test_every_tier_present_gets_a_slot_before_any_tier_gets_a_second(self):
        pages = []
        for tier in bench.TIERS:
            pages += [page(f"{tier}-{i:02d}.jpg", tier=tier,
                           layout=bench.LAYOUTS[i % len(bench.LAYOUTS)]) for i in range(8)]
        chosen = bench.pick_panel(pages, 6, 5)
        assert {p["tier"] for p in chosen} == set(bench.TIERS)

    def test_a_short_tier_does_not_eat_the_rest_of_the_panel(self):
        pages = [page("rare.jpg", tier="brutal")] + [page(f"e{i:02d}.jpg")
                                                     for i in range(40)]
        chosen = bench.pick_panel(pages, 10, 1)
        assert len(chosen) == 10
        assert any(p["tier"] == "brutal" for p in chosen)

    def test_a_corpus_smaller_than_the_panel_is_taken_whole(self):
        pages = corpus("a.jpg", "b.jpg")
        assert len(bench.pick_panel(pages, 12, 1)) == 2

    def test_pages_that_failed_to_measure_are_never_selected(self):
        pages = corpus("ok.jpg") + [page("broken.jpg", status="error")]
        assert [p["file"] for p in bench.pick_panel(pages, 12, 1)] == ["ok.jpg"]

    def test_size_zero_means_the_whole_corpus_not_nothing(self):
        pages = corpus("a.jpg", "b.jpg", "c.jpg")
        assert len(bench.pick_panel(pages, 0, 1)) == 3

    def test_layouts_are_spread_inside_a_tier(self):
        pages = []
        for layout in ("columns", "crowded", "dense", "mixed"):
            pages += [page(f"{layout}-{i:02d}.jpg", tier="hard", layout=layout)
                      for i in range(10)]
        chosen = bench.pick_panel(pages, 4, 1)
        assert len({p["layout"] for p in chosen}) == 4


# -----------------------------------------------------------------------
# the diff
# -----------------------------------------------------------------------

class TestDiff:
    def test_identical_pages_are_same_and_say_nothing_more(self):
        base = [block(0, text="a"), block(1, x0=50, text="b")]
        d = bench.diff_blocks(base, [dict(b) for b in base])
        assert (d.verdict, d.changed, d.reasons) == ("SAME", 0, [])

    def test_rewritten_text_is_changed_and_measured(self):
        d = bench.diff_blocks([block(0, text="ジャグシーバス")],
                              [block(0, text="ジャクシーバス")])
        assert d.verdict == "CHANGED" and d.changed == 1
        assert 0 < d.cer < 1

    def test_vanished_text_is_suspect_not_merely_different(self):
        d = bench.diff_blocks([block(0, text="something"), block(1, x0=50, text="else")],
                              [block(0, text=""), block(1, x0=50, text="else")])
        assert d.verdict == "SUSPECT" and d.lost == 1
        assert any("lost their text" in r for r in d.reasons)

    def test_text_that_appeared_is_reported_but_is_not_suspect(self):
        d = bench.diff_blocks([block(0, text="")], [block(0, text="new")])
        assert d.verdict == "CHANGED" and d.gained == 1

    def test_a_new_engine_error_is_suspect(self):
        ok = block(0, text="x")
        bad = block(0, text=None, error={"code": "OCR_FAILED", "message": "boom"})
        assert bench.diff_blocks([ok], [bad]).verdict == "SUSPECT"

    def test_a_block_the_crop_floor_refused_is_not_called_an_error(self):
        """SKIPPED_TOO_SMALL on both sides is a fact about the box, not a regression."""
        small = block(0, text=None, error={"code": "SKIPPED_TOO_SMALL", "message": "tiny"})
        assert bench.diff_blocks([small], [dict(small)]).verdict == "SAME"

    def test_a_moved_box_is_suspect_however_nice_the_text(self):
        d = bench.diff_blocks([block(0, x0=0, text="same")],
                              [block(0, x0=6, text="same")])
        assert d.verdict == "SUSPECT" and d.geometry_moved == 1

    def test_a_different_number_of_blocks_is_suspect(self):
        d = bench.diff_blocks([block(0), block(1, x0=50)], [block(0)])
        assert d.verdict == "SUSPECT"
        assert any("block set differs" in r for r in d.reasons)

    def test_suspect_outranks_changed_because_it_changes_the_question(self):
        d = bench.diff_blocks([block(0, text="a"), block(1, x0=50, text="b")],
                              [block(0, text="changed"), block(1, x0=50, text="")])
        assert d.verdict == "SUSPECT" and d.changed == 2

    def test_the_cer_column_averages_only_the_blocks_that_moved(self):
        base = [block(0, text="aaaa"), block(1, x0=50, text="untouched")]
        cand = [block(0, text="aaab"), block(1, x0=50, text="untouched")]
        d = bench.diff_blocks(base, cand)
        assert d.changed == 1 and d.cer == pytest.approx(0.25)

    def test_a_page_with_no_blocks_is_reported_without_dividing_by_anything(self):
        d = bench.diff_blocks([], [])
        assert (d.verdict, d.blocks, d.cer) == ("SAME", 0, 0.0)

    def test_the_row_names_the_columns_the_table_prints(self):
        d = bench.diff_blocks([block(0, text="a")], [block(0, text="b")])
        d.file, d.base_ms, d.cand_ms = "x.jpg", 100, 150
        row = bench._row(d, 12)
        for column in ("x.jpg", "CHANGED", "blk=1", "chg=1", "1.50x"):
            assert column in row, column
        assert d.to_dict()["time_ratio"] == 1.5


# -----------------------------------------------------------------------
# what a report is made of
# -----------------------------------------------------------------------

def args(**over):
    base = dict(pages=None, all=False, size=12, seed=1, only_changed=False, no_text=False)
    base.update(over)
    return argparse_namespace(base)


def argparse_namespace(mapping):
    import argparse
    return argparse.Namespace(**mapping)


def profile_document():
    return {"kind": bench.PROFILE_KIND, "created_at": "2026-11-27T00:00:00",
            "corpus": "sample_scans", "settings": bench.bench_settings(AppConfig.default()),
            "pinned": {"revision": "b" * 40, "models_dir": "animetext_studio/models"},
            "pages": [page("001.jpg")]}


class TestReport:
    def _report(self, no_text=False):
        page_recs = [page("001.jpg")]
        base = [block(0, text="secret words")]
        cand = [block(0, text="other words")]
        d = bench.diff_blocks(base, cand)
        d.file, d.tier, d.layout = "001.jpg", "hard", "columns"
        detail = [bench._page_detail(page_recs[0], base, cand, d, no_text)]
        return bench.build_report(args(no_text=no_text), profile_document(), page_recs,
                                  AppConfig.default(), "b" * 40,
                                  {"spec": "latest", "sha": "c" * 40, "title": "t",
                                   "date": "today"},
                                  {"differing_files": [], "engine_files_differ": [],
                                   "weights_identical": False},
                                  [d], detail, 1234, False,
                                  {"revision": "b" * 40}, {"revision": "c" * 40}, None)

    def test_the_harness_never_declares_an_upgrade(self):
        report = self._report()
        assert report["verdict"] is None
        assert "left to the person who reads the language" in report["verdict_note"]
        assert "upgrade" not in json.dumps(
            {k: v for k, v in report.items() if k != "verdict_note"}).lower()

    def test_the_summary_counts_match_the_rows(self):
        report = self._report()
        assert report["summary"]["pages"] == 1
        assert report["summary"]["changed"] == 1
        assert report["summary"]["changed_blocks"] == 1
        assert report["summary"]["suspect"] == 0

    def test_both_sides_are_named_by_full_sha_and_weights_hash(self):
        report = self._report()
        for side in report["sides"].values():
            assert side["revision"] and len(side["revision"]) == 40

    def test_no_text_replaces_words_with_a_hash_that_cannot_be_read_back(self):
        report = self._report(no_text=True)
        row = report["pages"][0]["changed_blocks"][0]
        assert row["pinned"].startswith("sha256:") and "secret" not in json.dumps(report)
        assert len(row["pinned"]) == len("sha256:") + 16

    def test_the_text_a_report_quotes_is_only_the_text_that_changed(self):
        report = self._report()
        rows = report["pages"][0]["changed_blocks"]
        assert len(rows) == 1 and rows[0]["id"] == 0

    def test_no_token_of_any_kind_reaches_the_report(self):
        report = self._report()
        blob = json.dumps(report)
        # `max_new_tokens` is a reading setting and belongs in the report; a credential does
        # not. So what is checked is the shape of a token, and any key named like a secret.
        for prefix in ("hf_", "ghp_", "github_pat_"):
            assert prefix not in blob
        for key in ("token", "hf_token", "api_key", "password", "secret"):
            assert f'"{key}"' not in blob

    def test_the_method_section_says_what_the_columns_mean(self):
        method = self._report()["method"]
        assert "reocr" not in method["detection"]        # both sides use Pipeline.run()
        for key in ("changed_blocks", "cer", "verdict_rule", "structural_columns"):
            assert method[key]


class TestComparabilityGuard:
    def test_the_same_settings_match_themselves(self):
        assert bench._settings_match(profile_document(), AppConfig.default())

    def test_a_different_crop_padding_is_a_different_experiment(self):
        stored = profile_document()
        stored["settings"]["ocr"]["crop_padding_px"] = 8
        assert not bench._settings_match(stored, AppConfig.default())

    def test_a_different_detector_is_a_different_experiment(self):
        stored = profile_document()
        stored["settings"]["detection"]["model"] = "yolo12n_animetext"
        assert not bench._settings_match(stored, AppConfig.default())

    def test_the_device_may_differ_because_it_changes_speed_not_text(self):
        stored = profile_document()
        stored["settings"]["ocr"]["device"] = "cuda:0"
        assert bench._settings_match(stored, AppConfig.default())


class TestRevisionArithmetic:
    def test_identical_fingerprints_differ_in_nothing(self):
        fp = {"model.safetensors": (10, "aa"), "README.md": (2, "bb")}
        out = bench.revision_difference(fp, dict(fp))
        assert out["differing_files"] == [] and out["weights_identical"] is True
        assert out["engine_files_differ"] == []

    def test_a_readme_only_commit_is_called_what_it_is(self):
        a = {"model.safetensors": (10, "aa"), "README.md": (2, "bb")}
        b = {"model.safetensors": (10, "aa"), "README.md": (3, "cc")}
        out = bench.revision_difference(a, b)
        assert out["differing_files"] == ["README.md"]
        assert out["engine_files_differ"] == []
        assert out["weights_identical"] is True

    def test_new_weights_are_never_mistaken_for_bookkeeping(self):
        a = {"model.safetensors": (10, "aa"), "modeling_hayai.py": (5, "dd")}
        b = {"model.safetensors": (10, "zz"), "modeling_hayai.py": (5, "dd")}
        out = bench.revision_difference(a, b)
        assert out["weights_identical"] is False
        assert out["engine_files_differ"] == ["model.safetensors"]

    def test_a_file_appearing_in_one_revision_is_a_difference(self):
        out = bench.revision_difference({"a.py": (1, "x")},
                                        {"a.py": (1, "x"), "b.py": (2, "y")})
        assert out["differing_files"] == ["b.py"]


class TestStagingReusesBytesItAlreadyHas:
    """A candidate that changed only its README should not cost a 633 MB download.

    Staging is where a bench can quietly become rude: the Hub records a hash per file, so a
    file this machine already holds at that hash is *known* identical and gets hard-linked. The
    tests are about the three ways that can go wrong — a name that is not part of a snapshot,
    a same-named file with different bytes, and a filesystem that refuses the link.
    """

    FILES = {"model.safetensors": (633564960, "aaa"), "README.md": (28443, "bbb"),
             ".gitattributes": (1519, "ccc"), "processing_hayai.py": (18135, "ddd")}

    def test_only_snapshot_files_are_candidates(self):
        wanted = bench.wanted_at(self.FILES)
        assert "model.safetensors" in wanted and "processing_hayai.py" in wanted
        assert "README.md" not in wanted and ".gitattributes" not in wanted
        assert wanted["model.safetensors"] == (633564960, "aaa")

    def test_a_file_is_found_by_its_hash_and_not_its_name(self, tmp_path):
        bundle = tmp_path / "bundle" / bench.OCR_DIRNAME
        bundle.mkdir(parents=True)
        weights = bundle / "model.safetensors"
        weights.write_bytes(b"the real weights")
        digest = bench.sha256_file(weights)
        size = weights.stat().st_size
        assert bench.find_reusable("model.safetensors", size, digest, [bundle]) == weights
        # Same name, same size, different bytes: not a match, at any price.
        other = tmp_path / "other" / bench.OCR_DIRNAME
        other.mkdir(parents=True)
        (other / "model.safetensors").write_bytes(b"other  weightz")   # same length!
        assert bench.find_reusable("model.safetensors", size, digest, [other]) is None
        # A recorded digest of None means the Hub told us nothing: nothing may be reused.
        assert bench.find_reusable("model.safetensors", size, None, [bundle]) is None
        # Wrong size is rejected before a 633 MB file is hashed.
        assert bench.find_reusable("model.safetensors", size + 1, digest, [bundle]) is None

    def test_a_reuse_is_a_hard_link_so_it_costs_no_disk(self, tmp_path):
        src = tmp_path / "bundle" / bench.OCR_DIRNAME / "model.safetensors"
        src.parent.mkdir(parents=True)
        src.write_bytes(b"weights")
        dst = tmp_path / "staged" / bench.OCR_DIRNAME / "model.safetensors"
        assert bench.link_or_copy(src, dst) == "linked"
        assert dst.read_bytes() == b"weights"
        assert dst.stat().st_ino == src.stat().st_ino      # the same inode, not a copy

    def test_the_bundle_is_searched_before_the_staged_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bench, "resolve_models_dir", lambda: tmp_path / "bundle")
        monkeypatch.setattr(bench, "SNAPSHOT_ROOT", tmp_path / "bench" / "snapshots")
        (tmp_path / "bundle" / bench.OCR_DIRNAME).mkdir(parents=True)
        (tmp_path / "bench" / "snapshots" / "abc" / bench.OCR_DIRNAME).mkdir(parents=True)
        sources = bench.reuse_sources()
        assert sources[0] == tmp_path / "bundle" / bench.OCR_DIRNAME
        assert len(sources) == 2


class TestIdentity:
    def test_a_non_checkout_is_identified_by_a_hash_and_says_so(self, tmp_path, monkeypatch):
        # "not a checkout" is simulated, not travelled to: this sandbox's temp tree sits inside
        # a real working copy, so a `git rev-parse` run from there answers with the enclosing
        # repository — git behaving correctly, not the test finding a bug.
        (tmp_path / "animetext_studio").mkdir()
        (tmp_path / "animetext_studio" / "a.py").write_text("print(1)\n", encoding="utf-8")

        def no_git(*args, **kwargs):
            raise OSError("no git here")

        monkeypatch.setattr(bench.subprocess, "run", no_git)
        ident = bench.app_identity(tmp_path)
        assert ident["kind"] == "content-hash" and ident["commit"] is None
        assert len(ident["content_sha256"]) == 64
        assert "hash" in ident["note"]

    def test_the_same_source_hashes_to_the_same_identity(self, tmp_path, monkeypatch):
        for name in ("a.py", "b.py"):
            (tmp_path / "animetext_studio").mkdir(exist_ok=True)
            (tmp_path / "animetext_studio" / name).write_text("x = 1\n", encoding="utf-8")

        def no_git(*args, **kwargs):
            raise OSError("no git here")

        monkeypatch.setattr(bench.subprocess, "run", no_git)
        first = bench.app_identity(tmp_path)["content_sha256"]
        second = bench.app_identity(tmp_path)["content_sha256"]
        assert first == second
        (tmp_path / "animetext_studio" / "c.py").write_text("x = 2\n", encoding="utf-8")
        assert bench.app_identity(tmp_path)["content_sha256"] != first

    def test_the_environment_names_the_versions_a_timing_depends_on(self):
        env = bench.environment()
        assert env["python"] and "platform" in env
        for key in ("torch", "transformers", "onnxruntime"):
            assert key in env            # present-or-None is the fact; absence is not


class TestNothingLeaks:
    def test_the_bench_directory_is_ignored(self):
        """"A report quotes someone's scan" is a promise, and .gitignore is where it is kept."""
        rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
        lines = [line.strip() for line in rules.splitlines()]
        assert "bench/" in lines

    def test_the_default_paths_all_live_under_the_ignored_directory(self):
        for path in (bench.PROFILE_PATH, bench.PANEL_PATH, bench.SNAPSHOT_ROOT,
                     bench.REPORT_DIR):
            assert str(path).startswith("bench/")

    def test_a_staged_snapshot_never_carries_the_repos_own_detector_weights(self):
        # `yolo26m-text.pt` is the model author's demo detector; this app detects with
        # AnimeText YOLO, so downloading it would be 47 MB of nothing.
        assert not any("yolo26m" in p for p in bench.SNAPSHOT_ALLOW)


class TestCommandLine:
    def test_run_refuses_to_guess_what_to_compare_against(self):
        with pytest.raises(SystemExit):
            bench.parser().parse_args(["run"])

    def test_several_candidates_can_be_asked_for_at_once(self):
        parsed = bench.parser().parse_args(
            ["run", "--against", "latest", "--against", "f5f0f617", "-n", "20", "--seed", "9"])
        assert parsed.against == ["latest", "f5f0f617"]
        assert (parsed.size, parsed.seed) == (20, 9)

    def test_the_panel_defaults_are_a_panel_not_the_whole_corpus(self):
        parsed = bench.parser().parse_args(["panel"])
        assert (parsed.size, parsed.seed) == (12, 1)

    def test_a_short_sha_is_refused_unless_it_is_unique(self):
        class Api:
            @staticmethod
            def list_repo_commits(repo, revision):
                class C:
                    commit_id = ""
                    title = ""
                    created_at = ""
                one, two = C(), C()
                one.commit_id = "f5f0f617" + "0" * 32
                two.commit_id = "f5f0f617" + "1" * 32
                one.created_at = two.created_at = "now"
                return [one, two]

        with pytest.raises(SystemExit):
            bench.resolve_revision(Api(), "f5f0f617")

    def test_a_short_sha_that_matches_one_commit_resolves_to_the_full_one(self):
        class Api:
            @staticmethod
            def list_repo_commits(repo, revision):
                class C:
                    commit_id = "f5f0f617" + "0" * 32
                    title = "Upload folder using huggingface_hub"
                    created_at = "2026-11-19 20:11:52+00:00"
                return [C()]

        got = bench.resolve_revision(Api(), "f5f0f617")
        assert got["sha"] == "f5f0f617" + "0" * 32 and got["is_head"] is False

    def test_latest_is_recorded_as_a_sha_not_as_the_word_latest(self):
        class Api:
            @staticmethod
            def list_repo_commits(repo, revision):
                class C:
                    commit_id = "a" * 40
                    title = "Update README.md"
                    created_at = "2026-11-22 16:41:07+00:00"
                return [C()]

        got = bench.resolve_revision(Api(), "latest")
        assert got["sha"] == "a" * 40 and got["is_head"] is True

    def test_an_unknown_revision_is_named_not_guessed_at(self):
        class Api:
            @staticmethod
            def list_repo_commits(repo, revision):
                return []

        with pytest.raises(SystemExit):
            bench.resolve_revision(Api(), "nope")
