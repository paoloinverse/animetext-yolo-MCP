"""Part 6 tests — the shared drawing recipe (`core/annotation.py`), no Qt needed.

Everything a painter needs to agree on — colour, chip text, chip placement, the nested-box
draw order, click picking, and the full-resolution export — is here, so it is tested with
plain Pillow and no display. The Qt side (`tests/test_gui.py`) checks that the canvas
*uses* these rules, not that it re-implements them.
"""
from __future__ import annotations

import pytest

from animetext_studio.core import annotation as A
from animetext_studio.core.config import GuiConfig
from animetext_studio.core.types import BBox, ErrorCode, StudioError, TextBlock
from PIL import Image


def block(bid: int, box, *, text="hello", score=0.5, color="#E6194B", error=None) -> TextBlock:
    return TextBlock(id=bid, bbox=BBox(*box), label="speech", score=score, text=text,
                     reading_order=bid, color=color, error=error)


# ---------------------------------------------------------------------------
# Style and colours
# ---------------------------------------------------------------------------

class TestStyle:
    def test_defaults_match_the_plan(self):
        style = A.OverlayStyle()
        assert (style.overlay, style.labels) == (True, True)
        assert style.fill_opacity == pytest.approx(0.18)

    def test_from_config_reads_the_gui_section(self):
        style = A.OverlayStyle.from_config(GuiConfig(overlay_visible=False,
                                                     labels_visible=False,
                                                     fill_opacity=0.4))
        assert (style.overlay, style.labels, style.fill_opacity) == (False, False, 0.4)

    def test_from_config_none_is_the_default(self):
        assert A.OverlayStyle.from_config(None) == A.OverlayStyle()

    @pytest.mark.parametrize("color,expected", [
        ("#E6194B", (230, 25, 75)),
        ("e6194b", (230, 25, 75)),
        ("#abc", (170, 187, 204)),
    ])
    def test_rgb_of(self, color, expected):
        assert A.rgb_of(color) == expected

    def test_a_broken_colour_falls_back_to_the_palette(self):
        from animetext_studio.core.types import color_for_index

        assert A.rgb_of("not-a-colour", fallback_index=3) == A.rgb_of(color_for_index(3))
        assert A.rgb_of("", fallback_index=0) == A.rgb_of("#E6194B")

    def test_chip_text_contrasts_with_its_background(self):
        assert A.contrast_text("#FFE119") == (0, 0, 0)        # yellow: black text
        assert A.contrast_text("#000075") == (255, 255, 255)  # navy: white text


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

class TestText:
    def test_the_chip_is_id_score_and_the_first_words(self):
        assert A.label_text(block(3, (0, 0, 10, 10), text="この攻撃で決まりだ",
                                  score=0.87)) == "#3\u00b70.87\u00b7この攻撃で決まりだ"

    def test_truncation_counts_characters_not_bytes(self):
        text = "絶対に負けない" * 10
        chip = A.label_text(block(1, (0, 0, 10, 10), text=text), max_chars=10)
        assert chip.endswith("\u2026")
        assert len(chip) == 10 + len("#1\u00b70.50\u00b7")
        chip.encode("utf-8").decode("utf-8")          # would raise on a sliced glyph

    def test_whitespace_and_newlines_collapse(self):
        assert A.one_line("  a\n\tb   c  ", 40) == "a b c"

    def test_an_empty_block_says_why_instead_of_nothing(self):
        chip = A.label_text(block(7, (0, 0, 5, 5), text=None, score=0.1,
                                  error={"code": "SKIPPED_TOO_SMALL"}))
        assert chip == "#7\u00b70.10\u00b7SKIPPED_TOO_SMALL"

    def test_the_hover_panel_carries_the_whole_text(self):
        text = "その日、空は赤かった" * 4
        panel = A.hover_text(block(2, (10, 20, 30, 40), text=text, score=0.912345))
        assert text in panel, "the chip truncates, the panel must not"
        assert "#2" in panel and "0.912" in panel
        assert "(10, 20)" in panel and "20\u00d720 px" in panel

    def test_the_hover_panel_names_the_error_code(self):
        panel = A.hover_text(block(9, (0, 0, 4, 4), text=None,
                                   error={"code": "OCR_FAILED"}))
        assert "OCR_FAILED" in panel


# ---------------------------------------------------------------------------
# Geometry: chips and nested boxes
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_the_chip_sits_above_the_box(self):
        assert A.chip_geometry((100, 200, 300, 300), 80, 14, pad=3) == (100, 180, 186, 200)

    def test_a_box_at_the_top_edge_gets_its_label_inside(self):
        x0, y0, x1, y1 = A.chip_geometry((100, 2, 300, 10), 80, 14, pad=3)
        assert y0 >= 0 and y1 >= 10, "the label must not go off the top of the image"

    def test_the_chip_never_leaves_the_image(self):
        chip = A.chip_geometry((880, 1290, 899, 1299), 80, 14, pad=3, image_size=(900, 1300))
        assert chip[0] >= 0 and chip[1] >= 0
        assert chip[2] <= 900 and chip[3] <= 1300

    def test_bigger_boxes_are_drawn_first(self):
        blocks = [block(1, (0, 0, 50, 50)), block(2, (0, 0, 900, 900)), block(3, (0, 0, 5, 5))]
        assert [b.id for b in A.draw_order(blocks)] == [2, 1, 3]

    def test_a_click_inside_nested_boxes_picks_the_smallest(self):
        page = block(0, (0, 0, 900, 1300))
        band = block(1, (10, 10, 300, 60))
        bubble = block(2, (20, 20, 120, 55))
        blocks = [page, band, bubble]
        assert A.pick_block(blocks, 30, 30).id == 2
        assert A.pick_block(blocks, 200, 30).id == 1, "outside the bubble: the band wins"
        assert A.pick_block(blocks, 800, 900).id == 0, "outside both: the page"

    def test_a_point_outside_everything_picks_nothing(self):
        assert A.pick_block([block(0, (0, 0, 10, 10))], 50, 50) is None

    def test_picking_is_deterministic_when_two_boxes_tie(self):
        left = block(5, (0, 0, 100, 100))
        right = block(4, (0, 0, 100, 100))
        assert A.pick_block([left, right], 50, 50).id == A.pick_block([right, left], 50, 50).id

    def test_the_right_and_bottom_edge_belong_to_the_next_box(self):
        assert A.contains((0, 0, 10, 10), 9.5, 9.5) is True
        assert A.contains((0, 0, 10, 10), 10, 5) is False


# ---------------------------------------------------------------------------
# The full-resolution renderer
# ---------------------------------------------------------------------------

@pytest.fixture
def page():
    return Image.new("RGB", (400, 300), (255, 255, 255))


@pytest.fixture
def blocks():
    return [block(0, (20, 20, 180, 120), text="この攻撃で決まりだ", score=0.91),
            block(1, (200, 150, 380, 280), text=None, score=0.44, color="#4363D8",
                  error={"code": "OCR_FAILED"})]


class TestRender:
    def test_the_input_is_never_touched(self, page, blocks):
        before = page.tobytes()
        A.render_overlay(page, blocks)
        assert page.tobytes() == before

    def test_size_is_kept_and_the_mode_is_rgb_or_rgba(self, page, blocks):
        out = A.render_overlay(page, blocks)
        assert out.size == page.size
        assert out.mode == "RGB"
        assert A.render_overlay(page.convert("RGBA"), blocks).mode == "RGBA"
        # A greyscale page comes back as RGB on purpose: the boxes are coloured, and
        # converting back to "L" would wash several of the palette colours into each other.
        assert A.render_overlay(page.convert("L"), blocks).mode == "RGB"
        assert A.render_overlay(page.convert("P"), blocks).mode == "RGB"

    def test_the_fill_is_translucent_and_the_border_is_not(self, page, blocks):
        out = A.render_overlay(page, blocks)
        border = out.getpixel((20, 70))
        inside = out.getpixel((150, 110))
        assert border == (230, 25, 75), "the border is the block colour, opaque"
        assert inside != (255, 255, 255)
        assert inside != (230, 25, 75), "the fill must blend with the page, not cover it"

    def test_a_pixel_outside_every_box_is_untouched(self, page, blocks):
        assert A.render_overlay(page, blocks).getpixel((395, 5)) == (255, 255, 255)

    def test_opacity_zero_is_the_page_with_borders(self, page, blocks):
        out = A.render_overlay(page, blocks, A.OverlayStyle(fill_opacity=0.0))
        assert out.getpixel((150, 110)) == (255, 255, 255)
        assert out.getpixel((20, 70)) == (230, 25, 75)

    def test_turning_the_overlay_off_returns_the_page(self, page, blocks):
        out = A.render_overlay(page, blocks, A.OverlayStyle(overlay=False))
        assert out.tobytes() == page.tobytes()

    def test_labels_can_be_switched_off(self, page, blocks):
        with_labels = A.render_overlay(page, blocks).tobytes()
        without = A.render_overlay(page, blocks, A.OverlayStyle(labels=False)).tobytes()
        assert with_labels != without
        plain = Image.new("RGB", (400, 300), (255, 255, 255))
        assert A.render_overlay(plain, blocks, A.OverlayStyle(labels=False)).getpixel(
            (200, 5)) == (255, 255, 255)

    def test_a_highlighted_block_is_stronger_than_its_neighbours(self, page, blocks):
        plain = A.render_overlay(page, blocks)
        lit = A.render_overlay(page, blocks, highlight=0)
        assert lit.tobytes() != plain.tobytes()
        assert sum(lit.getpixel((150, 110))) < sum(plain.getpixel((150, 110))), \
            "more colour, not less"
        assert lit.getpixel((395, 5)) == plain.getpixel((395, 5)), "only that block changed"

    def test_chips_are_drawn_above_every_box(self, page):
        """Boxes big-to-small, then every chip: a later box's border must not cut a label."""
        from PIL import ImageDraw

        big = block(0, (0, 40, 400, 300), text="page", color="#808080")
        drawer = ImageDraw.Draw(Image.new("RGB", page.size))
        box = drawer.textbbox((0, 0), A.label_text(big), font=A.load_font(
            A.font_size_for(page.size)))
        chip = A.chip_geometry((0, 40, 400, 300), box[2] - box[0], box[3] - box[1], pad=3)
        column = chip[2] - 1                 # the chip's right padding: solid colour, no glyph
        strip = block(1, (column, 0, column + 1, 300), text="x", color="#000000")
        assert strip.bbox.area < big.bbox.area, "the strip must be drawn after the big box"
        out = A.render_overlay(page, [big, strip], A.OverlayStyle(fill_opacity=0.0))
        row = (chip[1] + chip[3]) // 2
        assert out.getpixel((column, row)) == (128, 128, 128), \
            "the label chip has to cover the border that crosses it"

    def test_no_blocks_is_the_page(self, page):
        assert A.render_overlay(page, []).tobytes() == page.tobytes()

    def test_a_non_image_is_a_structured_error(self, blocks):
        with pytest.raises(StudioError) as excinfo:
            A.render_overlay("not an image", blocks)
        assert excinfo.value.code is ErrorCode.INVALID_IMAGE

    def test_duck_typed_blocks_work(self, page):
        from types import SimpleNamespace

        loose = [SimpleNamespace(id=1, bbox=(10, 10, 60, 40), score=0.5, text="ok",
                                 error=None, color="#3CB44B")]
        out = A.render_overlay(page, loose)
        assert out.getpixel((10, 25)) == (60, 180, 75)

    def test_a_tuple_bbox_is_accepted(self, page):
        from types import SimpleNamespace

        loose = [SimpleNamespace(id=1, bbox=(10, 10, 60, 40), score=0.5, text="ok",
                                 error=None, color="#3CB44B")]
        assert A.render_overlay(page, loose).getpixel((35, 25)) != (255, 255, 255)


class TestExport:
    def test_annotated_png_is_a_png(self, page, blocks):
        data = A.annotated_png(page, blocks)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert data != A.annotated_png(page, [], A.OverlayStyle(overlay=False))

    def test_save_writes_full_resolution_and_makes_the_folder(self, page, blocks, tmp_path):
        target = tmp_path / "deep" / "er" / "annotated.png"
        written = A.save_annotated(page, blocks, target)
        assert written == target and target.exists()
        with Image.open(target) as saved:
            assert saved.size == page.size

    def test_save_accepts_a_str_path(self, page, blocks, tmp_path):
        assert A.save_annotated(page, blocks, str(tmp_path / "x.png")).exists()

    def test_an_unwritable_path_is_an_io_error(self, page, blocks):
        with pytest.raises(StudioError) as excinfo:
            A.save_annotated(page, blocks, "/proc/definitely-not-writable/x.png")
        assert excinfo.value.code is ErrorCode.IO_ERROR


class TestChipStaysOnThePage:
    def test_the_renderer_tells_chip_geometry_the_page_size(self, monkeypatch):
        """A label on a box that reaches the right edge must be pulled inside the page,
        not written past it (seen on a full-page band box in T9)."""
        seen = []
        real = A.chip_geometry

        def spy(*args, **kwargs):
            seen.append(kwargs.get("image_size"))
            return real(*args, **kwargs)

        monkeypatch.setattr(A, "chip_geometry", spy)
        page = Image.new("RGB", (400, 300), "white")
        A.render_overlay(page, [block(0, (320, 10, 399, 290), text="a long label here")])
        assert seen, "no chip was drawn"
        assert all(size == (400, 300) for size in seen)

    def test_a_wide_label_on_a_narrow_box_is_clamped(self):
        # The chip would be ~200 px wide starting at x=320 on a 400 px page.
        page = Image.new("RGB", (400, 300), "white")
        out = A.render_overlay(page, [block(0, (320, 40, 399, 290),
                                           text="a long label here", color="#E6194B")])
        # the page's last column is chip colour, i.e. the label was pulled back to fit
        assert out.getpixel((399, 30)) == (230, 25, 75)
        assert out.getpixel((0, 30)) == (255, 255, 255), "it must not wrap to the left edge"
