"""Part 8.6 tests — label chips that stop sitting on each other, and the 1.2 flags in view.

Two promises, both new here:

* **`place_chips` is the one de-collision rule**, in `core/annotation.py`, that the canvas and
  the PNG export both call — so the window and the saved file cannot disagree about where a
  label went. It is pure geometry: no font, no Qt, no image, which is why most of this file
  needs neither.
* **`possible_truncation` and `contained_in` reach the eye**: a mark in the block table, a
  sentence in the tooltip and the hover panel, a count in the summary line.

The standing rule for 8.6 is that none of this touches the exported *document* — the last
group pins that too, by hashing a document before and after painting it.
"""
from __future__ import annotations

import os
import random

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed (pip install animetext-studio[gui])")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image                                           # noqa: E402

from animetext_studio.core import annotation as A              # noqa: E402
from animetext_studio.core.config import AppConfig             # noqa: E402
from animetext_studio.core.types import BBox, TextBlock        # noqa: E402


def block(bid, box, text="some text", *, color="#FF0000", score=0.5,
          flags=(), contained_in=None):
    return TextBlock(id=bid, bbox=BBox(*box), label="speech", score=score, text=text,
                     reading_order=bid, color=color, flags=tuple(flags),
                     contained_in=contained_in)


def offer(bid, box, full_w, variants=2, text="#1 " + "x" * 20, h=14):
    """A hand-made chip offer: `box`, a `full_w`-pixel-wide label, and the `#id` fallback.

    Sizes are passed in rather than measured because the rule under test is geometry: the
    font belongs to the painter, and both painters are tested separately below.
    """
    texts = [text] if variants == 1 else [text, f"#{bid}"]
    sizes = [(t, full_w if i == 0 else 6 * len(t), h) for i, t in enumerate(texts)]
    return A.ChipOffer(block=block(bid, box), box=tuple(box), variants=tuple(sizes))


def rects(plan):
    return [chip.rect for chip in plan.placed]


# ---------------------------------------------------------------------------
# The slots a chip will accept
# ---------------------------------------------------------------------------

class TestChipSlots:
    def test_the_first_slot_is_the_one_painters_already_used(self):
        """Nothing about a roomy page changes: slot 0 *is* `chip_geometry`."""
        box, w, h = (100, 200, 300, 300), 80, 14
        slot, rect = A.chip_slots(box, w, h)[0]
        assert slot == "above"
        assert rect == A.chip_geometry(box, w, h)

    def test_the_fallbacks_are_the_other_three_edges(self):
        assert [slot for slot, _ in A.chip_slots((100, 200, 300, 300), 80, 14)] == [
            "above", "inside-top", "inside-bottom", "below"]

    def test_a_slot_that_clamps_onto_another_is_not_offered_twice(self):
        """A chip taller than its box has nothing to choose between the inside edges."""
        slots = A.chip_slots((100, 2, 300, 10), 80, 14, image_size=(900, 1300))
        assert len({rect for _slot, rect in slots}) == len(slots)

    def test_every_slot_stays_on_the_page(self):
        for _slot, rect in A.chip_slots((880, 1290, 899, 1299), 80, 14, image_size=(900, 1300)):
            assert rect[0] >= 0 and rect[1] >= 0
            assert rect[2] <= 900 and rect[3] <= 1300

    def test_the_id_variant_is_offered_alongside_the_label(self):
        assert A.chip_variants(block(7, (0, 0, 10, 10), "hello")) == ("#7\u00b70.50\u00b7hello", "#7")

    def test_a_chip_that_is_already_only_an_id_is_not_offered_twice(self):
        """`chip_geometry` of the same text twice would make a lost chip look like a fit."""
        assert A.chip_variants(block(7, (0, 0, 10, 10), "7"))[-1] == "#7"
        assert len(A.chip_variants(block(7, (0, 0, 10, 10)))) == 2

    def test_touching_is_not_colliding(self):
        """Boxes share borders on a manga page; a graze must not empty the page of labels."""
        assert not A.overlaps((0, 0, 10, 10), (10, 0, 20, 10))
        assert not A.overlaps((0, 0, 10, 10), (9, 0, 20, 10)), "a 1 px graze"
        assert A.overlaps((0, 0, 10, 10), (5, 5, 20, 20))


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

class TestPlaceChips:
    def test_a_chip_with_room_is_left_exactly_where_it_was(self):
        plan = A.place_chips([offer(1, (100, 200, 300, 300), 80)])
        assert plan.skipped == ()
        assert plan.placed[0].slot == "above"
        assert plan.placed[0].rect == A.chip_geometry((100, 200, 300, 300), 80, 14, pad=3)

    def test_the_specific_box_claims_the_space_first(self):
        """A panel-sized box never out-claims the bubble inside it — same rule as a click."""
        big, small = (100, 100, 900, 500), (100, 100, 200, 140)
        plan = A.place_chips([offer(1, big, 200), offer(2, small, 200)])
        assert plan.wanted(2).slot == "above", "the small box keeps the spot it wanted"
        assert plan.wanted(1).slot == "inside-top", "the container moves aside"
        assert not A.overlaps(*rects(plan))

    def test_a_moved_chip_keeps_its_words(self):
        """Words beat position: a full label *below* beats a bare `#id` above."""
        wide, wall = (100, 100, 300, 110), (100, 110, 140, 120)
        plan = A.place_chips([offer(1, wide, 180), offer(2, wall, 200)])
        chip = plan.wanted(1)
        assert chip.variant == 0 and chip.slot == "below"
        assert chip.text.startswith("#1"), "it moved, it did not shrink"

    def test_a_chip_with_no_room_for_its_words_keeps_its_number(self):
        """Four neighbours wall off every side of the wide chip. `#1` still fits above.

        The number is the part that cannot be recovered anywhere else — it is the key into
        the table, the JSON and `focus_block` — so it is the last thing to go.
        """
        offers = [offer(1, (10, 50, 200, 80), 220)]
        for bid, y in ((2, 50), (3, 70), (4, 80), (5, 90)):
            offers.append(offer(bid, (200, y, 240, y + 10), 94, variants=1, text="wall"))
        plan = A.place_chips(offers)
        chip = plan.wanted(1)
        assert chip.variant == 1 and chip.text == "#1"
        assert plan.painted(1) and 1 not in plan.hidden

    def test_a_chip_with_no_room_at_all_is_dropped_and_reported(self):
        """One wall, tall enough to cover every side of the box: nothing fits, and it says so.

        Dropping is the last resort and it is never silent — the block lands in `hidden`,
        and the plan remembers the label it lost so the hover path can still paint it.
        """
        offers = [offer(1, (10, 20, 200, 28), 220),
                  offer(2, (0, 110, 10, 120), 300, variants=1, text="wall", h=100)]
        plan = A.place_chips(offers, image_size=(300, 130))
        assert not plan.painted(1)
        assert plan.hidden == (1,)
        assert plan.wanted(1).text == offers[0].variants[0][0], "it remembers what it lost"
        assert plan.wanted(1).slot == "above", "and where it wanted to be"

    def test_the_plan_accounts_for_every_block_it_was_given(self):
        offers = [offer(i, (i * 30, i * 20, i * 30 + 25, i * 20 + 15), 90) for i in range(1, 9)]
        plan = A.place_chips(offers)
        assert len(plan.placed) + len(plan.skipped) == len(offers)
        assert set(plan.hidden) == {id_of(c) for c in plan.skipped}

    def test_the_layout_does_not_depend_on_the_order_the_blocks_arrive_in(self):
        """Reading order, click order, MCP order: the page must look the same either way."""
        offers = [offer(i, (40 + i * 17 % 200, 30 + i * 29 % 150, 120 + i * 17 % 180,
                            90 + i * 29 % 120), 70 + 9 * i) for i in range(1, 10)]
        first = {id_of(c): c.rect for c in A.place_chips(offers).placed}
        second = {id_of(c): c.rect for c in A.place_chips(list(reversed(offers))).placed}
        assert first == second

    def test_paint_order_puts_the_specific_chip_on_top(self):
        big, small = (0, 0, 400, 400), (300, 300, 340, 340)
        plan = A.place_chips([offer(1, big, 60), offer(2, small, 60)])
        assert [id_of(c) for c in plan.paintable] == [1, 2]

    def test_turning_the_rule_off_restores_the_old_picture_exactly(self):
        """`decollide_labels=False` is an escape hatch, not a second behaviour: it is 8.5."""
        big, small = (100, 100, 900, 500), (100, 100, 200, 140)
        style = A.OverlayStyle(decollide_labels=False)
        plan = A.place_chips([offer(1, big, 200), offer(2, small, 200)], style=style)
        assert len(plan.placed) == 2 and plan.skipped == ()
        assert rects(plan)[0] == rects(plan)[1], "both in their favourite spot, overlapping"
        assert all(c.slot == "above" for c in plan.placed)

    def test_no_offers_no_plan(self):
        assert A.place_chips([]).placed == ()
        assert A.place_chips(()).hidden == ()

    def test_two_hundred_random_pages_never_overlap(self):
        """The invariant, said once and checked exhaustively instead of case by case."""
        rnd = random.Random(20261124)
        page = (600, 400)
        for _case in range(200):
            offers = []
            for i in range(1, rnd.randint(3, 14)):
                x0, y0 = rnd.randint(0, 560), rnd.randint(0, 380)
                box = (x0, y0, min(600, x0 + rnd.randint(4, 140)), min(400, y0 + rnd.randint(4, 90)))
                offers.append(offer(i, box, rnd.randint(20, 260)))
            plan = A.place_chips(offers, image_size=page)
            placed = list(plan.placed)
            for i in range(len(placed)):
                for j in range(i + 1, len(placed)):
                    assert not A.overlaps(placed[i].rect, placed[j].rect), (
                        f"chips {id_of(placed[i])} and {id_of(placed[j])} sit on each other")
            for chip in placed:
                assert chip.rect[0] >= 0 and chip.rect[1] >= 0
                assert chip.rect[2] <= page[0] and chip.rect[3] <= page[1]
            assert len(placed) + len(plan.skipped) == len(offers)


def id_of(chip):
    return int(chip.block.id)


# ---------------------------------------------------------------------------
# The 1.2 observations, in words and in marks
# ---------------------------------------------------------------------------

class TestFlagSurfacing:
    def test_the_marks_are_short_enough_for_a_table_cell(self):
        b = block(4, (0, 0, 9, 9), "x", flags=("possible_truncation",), contained_in=1)
        assert A.flag_marks(b) == ("\u26a0", "\u22821")

    def test_a_clean_block_is_marked_with_nothing(self):
        assert A.flag_marks(block(4, (0, 0, 9, 9), "x")) == ()
        assert A.flag_note(block(4, (0, 0, 9, 9), "x")) == ""

    def test_an_unknown_flag_shows_its_name_instead_of_vanishing(self):
        """A flag the viewer does not know is a flag lost; a name is never silently dropped."""
        assert A.flag_marks(block(4, (0, 0, 9, 9), "x", flags=("blurry",))) == ("blurry",)
        assert "blurry" in A.flag_note(block(4, (0, 0, 9, 9), "x", flags=("blurry",)))

    def test_the_note_says_what_to_do_not_only_what_is_wrong(self):
        note = A.flag_note(block(4, (0, 0, 9, 9), "x", flags=("possible_truncation",)))
        assert "possible_truncation" in note, "spelled as the JSON spells it"
        assert "max_new_tokens" in note, "and with the knob that would fix it"

    def test_the_container_is_named_by_id(self):
        note = A.flag_note(block(4, (0, 0, 9, 9), "x", contained_in=2))
        assert "contained_in #2" in note

    def test_the_hover_panel_carries_the_note(self):
        panel = A.hover_text(block(4, (0, 0, 9, 9), "hello",
                                   flags=("possible_truncation",), contained_in=2))
        assert "possible_truncation" in panel and "contained_in #2" in panel
        assert "hello" in panel, "and still the full text"

    def test_the_hover_panel_says_nothing_when_there_is_nothing_to_say(self):
        assert "possible_truncation" not in A.hover_text(block(4, (0, 0, 9, 9), "hello"))

    def test_the_count_covers_both_observations(self):
        blocks = [block(1, (0, 0, 9, 9)), block(2, (0, 0, 9, 9), flags=("possible_truncation",)),
                  block(3, (0, 0, 9, 9), contained_in=1), block(4, (0, 0, 9, 9))]
        assert A.flagged_count(blocks) == 2
        assert A.flagged_count([]) == 0


# ---------------------------------------------------------------------------
# The Pillow painter honours the plan
# ---------------------------------------------------------------------------

PAGE = (400, 300)


def painted(image, color):
    """How many pixels carry this exact colour — a proxy for "how much label is visible"."""
    rgb = A.rgb_of(color)
    return sum(1 for px in image.getdata() if px[:3] == rgb)


def two_blocks_on_one_box():
    box = (50, 100, 250, 200)
    return [block(1, box, "hello", color="#FF0000"),
            block(2, box, "hello", color="#00FF00", contained_in=1)]


class TestThePainterSharesTheRule:
    def test_two_labels_on_one_box_both_reach_the_picture(self):
        """Before 8.6 the second chip was painted over the first and one label did not exist."""
        image = Image.new("RGB", PAGE, "white")
        on = A.render_overlay(image, two_blocks_on_one_box())
        off = A.render_overlay(image, two_blocks_on_one_box(),
                               A.OverlayStyle(decollide_labels=False))
        assert painted(on, "#FF0000") + painted(on, "#00FF00") > \
            painted(off, "#FF0000") + painted(off, "#00FF00"), \
            "de-collision must put more label on the page, not less"

    def test_the_block_you_are_looking_at_always_keeps_its_label(self, monkeypatch):
        """Even when the rule drops it: what was on screen is what gets exported."""
        real = A.place_chips
        monkeypatch.setattr(A, "place_chips", lambda offers, **kw:
                            A.ChipPlan(placed=(), skipped=real(offers, **kw).placed))
        image = Image.new("RGB", PAGE, "white")
        plain = A.render_overlay(image, two_blocks_on_one_box())
        lit = A.render_overlay(image, two_blocks_on_one_box(), highlight=2)
        assert painted(lit, "#00FF00") > painted(plain, "#00FF00")

    def test_the_painter_offers_every_block_and_both_variants(self, monkeypatch):
        seen = {}
        real = A.place_chips

        def spy(offers, **kw):
            seen["offers"] = list(offers)
            seen["image_size"] = kw.get("image_size")
            return real(offers, **kw)

        monkeypatch.setattr(A, "place_chips", spy)
        A.render_overlay(Image.new("RGB", PAGE, "white"), two_blocks_on_one_box())
        assert [id_of(o) for o in seen["offers"]] == [1, 2]
        assert all(len(o.variants) == 2 for o in seen["offers"]), "label and #id"
        assert seen["image_size"] == PAGE

    def test_the_document_is_untouched_by_any_of_this(self, sample_result):
        """The standing rule for 8.6: the paint recipe changed, the exported bytes did not."""
        import hashlib
        import json

        before = hashlib.sha256(
            json.dumps(sample_result.to_dict(), sort_keys=True).encode()).hexdigest()
        A.render_overlay(Image.new("RGB", PAGE, "white"), sample_result.blocks)
        A.render_overlay(Image.new("RGB", PAGE, "white"), sample_result.blocks, highlight=1)
        after = hashlib.sha256(
            json.dumps(sample_result.to_dict(), sort_keys=True).encode()).hexdigest()
        assert before == after


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

from animetext_studio.core.jobs import JobManager              # noqa: E402
from animetext_studio.core.pipeline import (                   # noqa: E402
    Pipeline, engines_from_config,
)
from animetext_studio.core.types import (                      # noqa: E402
    DETECTION_REPO, OCR_REPO, TEXT_LABEL, DetectionInfo, ImageInfo, OcrInfo, PipelineResult,
    TimingMs,
)
from animetext_studio.gui.main_window import MainWindow        # noqa: E402


def result_of(blocks):
    return PipelineResult(
        image=ImageInfo(source="file", file_name="page.png", width=PAGE[0], height=PAGE[1],
                        sha256="0" * 64),
        detection=DetectionInfo(repo=DETECTION_REPO, model="stub", iou_threshold=0.35,
                                score_threshold=0.05, allow_dynamic=False),
        ocr=OcrInfo(repo=OCR_REPO, device="cpu", max_patches="auto", batch_size=16,
                    max_new_tokens=128, crop_padding_px=2, normalize_nfkc=False),
        timing_ms=TimingMs(detect=1, ocr=1, total=2),
        blocks=tuple(blocks),
        created_at="2026-11-24T00:00:00+00:00",
    )


@pytest.fixture
def page_file(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", PAGE, "white").save(path)
    return str(path)


def shown_window(qtbot, page_file, blocks=None):
    """A visible window with a page loaded and hand-made blocks drawn on it."""
    win = build_window()
    qtbot.addWidget(win)
    win.resize(420, 320)
    win.show()
    win.load_path(page_file)
    win.set_result(result_of(blocks if blocks is not None else two_blocks_on_one_box()))
    win.canvas.view.viewport().grab()            # force the paint the assertions read
    return win


def build_window(config=None):
    config = config or AppConfig.default()
    engines = engines_from_config(config, mock=True)
    return MainWindow(JobManager(Pipeline(engines.detection, engines.ocr, config)),
                      config, mock=True)


@pytest.fixture
def window_with_flags(qtbot):
    win = build_window()
    qtbot.addWidget(win)
    win.show()
    win.set_result(result_of([
        block(1, (10, 10, 200, 90), "the panel"),
        block(2, (20, 20, 90, 60), "a bubble inside it", contained_in=1, color="#00FF00"),
        block(3, (10, 120, 200, 190), "cut off mid sentence",
              flags=("possible_truncation",), color="#0000FF"),
    ]))
    return win


class TestTheWindowShowsTheObservations:
    def test_the_table_has_a_flags_column(self, window_with_flags):
        assert window_with_flags.block_tree.columnCount() == 4
        assert window_with_flags.block_tree.headerItem().text(3) == "flags"

    def test_the_nested_block_and_the_truncated_one_are_marked(self, window_with_flags):
        tree = window_with_flags.block_tree
        rows = {tree.topLevelItem(i).text(0): tree.topLevelItem(i)
                for i in range(tree.topLevelItemCount())}
        assert rows["2"].text(3) == "\u22821"
        assert rows["3"].text(3) == "\u26a0"
        assert rows["1"].text(3) == "", "the container itself carries nothing"

    def test_a_mark_that_means_something_comes_with_a_sentence(self, window_with_flags):
        tree = window_with_flags.block_tree
        rows = {tree.topLevelItem(i).text(0): tree.topLevelItem(i)
                for i in range(tree.topLevelItemCount())}
        assert "contained_in #1" in rows["2"].toolTip(3)
        assert "max_new_tokens" in rows["3"].toolTip(3)

    def test_the_header_explains_the_alphabet(self, window_with_flags):
        tip = window_with_flags.block_tree.headerItem().toolTip(3)
        assert "possible_truncation" in tip and "contained_in" in tip

    def test_the_summary_counts_them(self, window_with_flags):
        assert "3 block(s)" in window_with_flags.summary_label.text()
        assert "2 flagged" in window_with_flags.summary_label.text()

    def test_a_clean_run_does_not_shout(self, qtbot):
        win = build_window()
        qtbot.addWidget(win)
        win.set_result(result_of([block(1, (10, 10, 200, 90), "quiet")]))
        assert "flagged" not in win.summary_label.text()

    def test_the_marks_are_said_out_loud_when_a_block_is_picked(self, window_with_flags):
        """The status line is where a screen reader and a quick glance both land."""
        window_with_flags.canvas.select_block(3, emit=True)   # the path a canvas click takes
        message = window_with_flags.status_label.text()
        assert "Block #3" in message and "\u26a0" in message

    def test_the_hover_panel_of_a_real_window_carries_the_note(self, window_with_flags):
        window_with_flags.canvas.set_hover(3)
        text = window_with_flags.canvas.hover_panel.text()
        assert "possible_truncation" in text and "cut off mid sentence" in text

    def test_the_canvas_uses_the_same_rule_as_the_export(self, monkeypatch, qtbot, page_file):
        """One rule, two painters: the canvas calls `place_chips`, not its own geometry."""
        from animetext_studio.gui import canvas as C

        calls = []
        real = C.place_chips

        def spy(offers, **kw):
            calls.append((len(offers), kw.get("image_size")))
            return real(offers, **kw)

        monkeypatch.setattr(C, "place_chips", spy)
        win = shown_window(qtbot, page_file)
        win.canvas.view.viewport().grab()
        assert calls and calls[-1][0] == 2, "both blocks were offered to the rule"
        assert calls[-1][1] and calls[-1][1][0] > 0, "measured against the viewport, not the page"

    def test_the_window_paints_more_label_than_the_old_rule_did(self, qtbot, page_file):
        """The point of 8.6, measured in pixels: two labels where there used to be one."""
        from dataclasses import replace

        win = shown_window(qtbot, page_file)

        def painted():
            image = win.canvas.view.viewport().grab().toImage()
            want = [A.rgb_of("#FF0000"), A.rgb_of("#00FF00")]
            hits = 0
            for y in range(image.height()):
                for x in range(image.width()):
                    if image.pixelColor(x, y).getRgb()[:3] in want:
                        hits += 1
            return hits

        on = painted()
        win.canvas._style = replace(win.canvas._style, decollide_labels=False)
        win.canvas.view.viewport().update()
        qtbot.wait(80)
        assert on > painted(), "de-collision must put more label on the page, not less"

    def test_a_chip_is_wide_enough_for_the_text_it_draws(self, qtbot, page_file):
        """The font that measured the chip is the font that draws it.

        The chip is a solid rect and the text is centred in it, so a painter left on the
        widget's default font draws a wider string than the rect it was measured for and Qt
        clips both ends \u2014 the label came out as `.80 \u3053\u306e\u653b\u6483\u3067\u6c7a\u307e`. Latin text
        keeps the probe independent of which CJK face the machine happens to have.
        """
        from PySide6.QtCore import QPointF

        win = shown_window(qtbot, page_file, blocks=[
            block(1, (120, 120, 300, 180), "the final attack ends it here", color="#FF0000")])
        image = win.canvas.view.viewport().grab().toImage()
        red, white = A.rgb_of("#FF0000"), (255, 255, 255)
        point = win.canvas.view.mapFromScene(QPointF(120.0, 120.0))
        left, top = int(round(point.x())), int(round(point.y()))

        def px(x, y):
            return image.pixelColor(x, y).getRgb()[:3]

        y = top - 2                       # walk up the chip's left padding to its top edge
        while y > 0 and px(left + 2, y) == red:
            y -= 1
        assert y + 1 < top, "there is a chip above the box to measure"
        middle = (y + 1 + top) // 2
        assert px(left - 4, middle) == white, "no glyph sticks out past the chip"
        assert any(px(x, middle) not in (red, white) for x in range(left, left + 80)), \
            "and the chip is not empty either"

    def test_moving_the_mouse_does_not_make_anybody_else_relayout(self, monkeypatch, qtbot,
                                                                   page_file):
        """The layout ignores hover, and it is not even recomputed on hover.

        Two halves of one promise: chips cannot jump while the pointer moves, and a repaint
        per mouse move does not pay for the layout again (the reason it may be cached is the
        same reason it is stable).
        """
        from animetext_studio.gui import canvas as C

        calls = []
        real = C.place_chips

        def spy(offers, **kw):
            calls.append(len(offers))
            return real(offers, **kw)

        monkeypatch.setattr(C, "place_chips", spy)
        win = shown_window(qtbot, page_file)
        after_paint = len(calls)
        assert after_paint >= 1
        for block_id in (1, 2, 1, C.NO_BLOCK):
            win.canvas.set_hover(block_id)
            win.canvas.view.viewport().grab()
        assert len(calls) == after_paint, "hover changed the picture, not the layout"
        win.canvas.set_zoom(percent=200)        # geometry changed: it must run again
        win.canvas.view.viewport().grab()
        assert len(calls) > after_paint

    def test_the_canvas_reports_which_labels_it_had_to_hide(self, qtbot, page_file):
        win = shown_window(qtbot, page_file, blocks=[
            block(1, (5, 5, 395, 295), "the whole page", color="#FF0000"),
            block(2, (10, 10, 390, 290), "and again, almost the same box", color="#00FF00"),
            block(3, (15, 15, 385, 285), "and once more", color="#0000FF"),
        ])
        win.canvas.view.viewport().grab()
        hidden = win.canvas.hidden_labels
        assert isinstance(hidden, tuple)
        assert set(hidden) <= {1, 2, 3}
