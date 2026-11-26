"""The block-drawing recipe, in one place and free of Qt (PLAN Part 6).

Two painters use this module:

* `gui/canvas.py` paints interactively with QPainter (zoom, hover, selection);
* :func:`render_overlay` bakes the same picture at **full resolution** into a PNG — the
  GUI's "Save annotated PNG" here, and the MCP `render_annotated_image` tool in P7, which
  runs with no display at all.

Only the *brush* is duplicated. The *recipe* — which colour, what the label chip says,
where the chip goes, which box wins a click when boxes are nested — lives here, so the
screenshot and the exported file cannot drift apart, and both are testable without Qt.

Nested boxes are real detector output (T7 / backlog 9): a title band contains its column
boxes. Two rules follow from that and are shared by both painters:

* **draw order**: biggest area first, so the small (usually the actual bubble) boxes and
  their chips end up on top;
* **picking**: a click goes to the *smallest* box under the cursor — the most specific
  one — and is deterministic when areas tie.
"""
from __future__ import annotations

from collections.abc import Sequence          # isinstance-safe (not typing.Sequence)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from PIL import Image, ImageDraw

from .types import BBox, ErrorCode, StudioError, color_for_index

__all__ = [
    "OverlayStyle", "DEFAULT_STYLE", "label_text", "hover_text", "chip_geometry",
    "chip_slots", "chip_variants", "id_text", "place_chips", "Chip", "ChipOffer",
    "ChipPlan", "overlaps", "pick_block", "draw_order", "render_overlay",
    "annotated_png", "save_annotated", "rgb_of", "contrast_text", "load_font",
    "font_size_for", "border_width_for", "one_line", "flag_marks", "flag_note",
    "flagged_count",
]

#: CJK-capable faces first: a label that renders as tofu boxes is worse than no label.
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

ELLIPSIS = "…"
MIDDOT = "·"
ARROW = "→"
TIMES = "×"

#: Marks for the result-1.2 observations, so the table, the tooltip and the hover panel
#: cannot invent three different spellings for one flag. An unknown flag renders as its
#: own name rather than disappearing: a flag the viewer cannot show is a flag lost.
FLAG_MARKS = {"possible_truncation": "\u26a0"}          # ⚠
CONTAINMENT_MARK = "\u2282"                             # ⊂

#: Two chips may touch; they must not *cross* by more than this many pixels on both axes
#: before one of them has to move (`place_chips`, P8.6).
CHIP_GAP_PX = 1


@dataclass(frozen=True)
class OverlayStyle:
    """Everything that affects how the overlay looks, as data (driven by `AppConfig.gui`).

    Sizes are divisors of the image's shorter side, so a 900 px page and a 3000 px scan
    get proportionally readable chips instead of the same 12 px label on both.
    """

    overlay: bool = True
    labels: bool = True
    fill_opacity: float = 0.18
    #: hover/selection multiply the fill (and thicken the border) so the picked box is obvious
    highlight_fill_boost: float = 2.0
    border_ratio: float = 1.0 / 450.0
    min_border_px: int = 2
    font_ratio: float = 1.0 / 70.0
    min_font_px: int = 12
    label_max_chars: int = 28
    chip_padding_px: int = 3
    #: P8.6: chips that would land on top of another chip move, shrink to `#id`, or step
    #: aside — see :func:`place_chips`. On by default: overlapping chips are unreadable,
    #: and the words are never *lost*, only deferred to the hover panel and the table.
    #: Deliberately **not** an `AppConfig.gui` key — the config document is frozen at 1.2
    #: and a paint detail does not justify a version bump (P8 rule: bump only when the
    #: shape changes). This field is the seam a future `gui.labels_decollide` would use.
    decollide_labels: bool = True
    chip_gap_px: int = CHIP_GAP_PX

    @classmethod
    def from_config(cls, gui_config: Any) -> "OverlayStyle":
        """From `AppConfig.gui` (duck-typed: this module must not import the GUI)."""
        if gui_config is None:
            return cls()
        return cls(overlay=bool(getattr(gui_config, "overlay_visible", True)),
                   labels=bool(getattr(gui_config, "labels_visible", True)),
                   fill_opacity=float(getattr(gui_config, "fill_opacity", 0.18)))


DEFAULT_STYLE = OverlayStyle()


# ---------------------------------------------------------------------------
# Colour, font, text
# ---------------------------------------------------------------------------

def _hex_to_tuple(color: str) -> tuple[int, int, int]:
    text = str(color).lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def rgb_of(color: str, *, fallback_index: int = 0) -> tuple[int, int, int]:
    """`"#4363D8"` -> `(67, 99, 216)`. A malformed value falls back to the palette."""
    text = str(color or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) == 6:
        try:
            return _hex_to_tuple(text)
        except ValueError:
            pass
    return _hex_to_tuple(color_for_index(fallback_index))


def contrast_text(color: str, *, fallback_index: int = 0) -> tuple[int, int, int]:
    """Black or white, whichever is readable on `color` (ITU-R BT.601 luma)."""
    r, g, b = rgb_of(color, fallback_index=fallback_index)
    luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return (0, 0, 0) if luma > 0.6 else (255, 255, 255)


def load_font(size: int):
    """First truetype face that opens, else Pillow's bitmap default (never raises)."""
    from PIL import ImageFont

    size = max(8, int(size))
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def font_size_for(image_size: Sequence[int], style: OverlayStyle = DEFAULT_STYLE) -> int:
    short = min(int(image_size[0]), int(image_size[1]))
    return max(style.min_font_px, int(round(short * style.font_ratio)))


def border_width_for(image_size: Sequence[int], style: OverlayStyle = DEFAULT_STYLE) -> int:
    short = min(int(image_size[0]), int(image_size[1]))
    return max(style.min_border_px, int(round(short * style.border_ratio)))


def one_line(text: Any, max_chars: int) -> str:
    """Collapse whitespace and truncate on *characters* (CJK-safe: never slices a glyph)."""
    flat = " ".join(str(text).split())
    if max_chars <= 0 or len(flat) <= max_chars:
        return flat
    return flat[: max(1, max_chars - 1)].rstrip() + ELLIPSIS


def label_text(block: Any, *, max_chars: int = DEFAULT_STYLE.label_max_chars) -> str:
    """The chip: `#3` · `0.87` · the first words of the text.

    A block with no text still gets a chip naming *why* (`SKIPPED_TOO_SMALL`,
    `OCR_FAILED`): an unlabelled empty box reads as a bug in the viewer.
    """
    what = one_line(block.text, max_chars) if getattr(block, "text", None) else ""
    if not what:
        error = getattr(block, "error", None) or {}
        what = str(error.get("code", "no text"))
    score = float(getattr(block, "score", 0.0))
    return f"#{getattr(block, 'id', '?')}{MIDDOT}{score:.2f}{MIDDOT}{what}"


def hover_text(block: Any) -> str:
    """The hover panel: identity line, the **full** text, the box, then the 1.2 notes.

    Deliberately untruncated \u2014 the chip is the index, the panel is the reading view. The
    observations get the last lines because they are the part you act on, and because a chip
    that had to shrink to `#id` (P8.6) still reaches its full text here: this panel is what
    "shown on hover instead" means.
    """
    score = float(getattr(block, "score", 0.0))
    head = f"#{getattr(block, 'id', '?')}  {score:.3f}"
    label = getattr(block, "label", "")
    if label:
        head += f"  {label}"
    lines = [head]
    text = getattr(block, "text", None)
    if text:
        lines.append(str(text))
    else:
        error = getattr(block, "error", None) or {}
        lines.append(f"[{error.get('code', 'no text')}]")
    bbox = getattr(block, "bbox", None)
    if bbox is not None:
        lines.append(f"({bbox.x0}, {bbox.y0}) {ARROW} ({bbox.x1}, {bbox.y1})  "
                     f"{bbox.width}{TIMES}{bbox.height} px")
    note = flag_note(block)
    if note:
        lines.append(note)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometry (pure numbers: shared by QPainter and Pillow, testable with neither)
# ---------------------------------------------------------------------------

def chip_geometry(bbox: Sequence[int], text_w: int, text_h: int, *,
                  pad: int = DEFAULT_STYLE.chip_padding_px,
                  image_size: Optional[Sequence[int]] = None) -> tuple[int, int, int, int]:
    """Where the label chip goes: above the box, or inside it when there is no room.

    `bbox` is `(x0, y0, x1, y1)`; the return is the chip's `(x0, y0, x1, y1)`. The chip is
    left-aligned with the box and never leaves the image, so a block touching the top edge
    gets its label *inside* the box instead of off-screen.
    """
    x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    w = int(text_w) + 2 * pad
    h = int(text_h) + 2 * pad
    top = y0 - h
    if top < 0:
        top = y0                      # inside, along the top edge
        if y1 - y0 < h:               # the box itself is too short: sit on its lower edge
            top = max(0, y1 - h)
    left = x0
    if image_size is not None:
        img_w, img_h = int(image_size[0]), int(image_size[1])
        left = max(0, min(left, max(0, img_w - w)))
        top = max(0, min(top, max(0, img_h - h)))
    return (left, top, left + w, top + h)


def draw_order(blocks: Iterable[Any]) -> list[Any]:
    """Largest area first, so the specific (small) boxes and chips end up on top."""
    return sorted(blocks, key=lambda b: (-_area(b), getattr(b, "id", 0)))


def _area(block: Any) -> int:
    """Box area, whether the block carries a `BBox`, a plain tuple, or nothing at all."""
    x0, y0, x1, y1 = _rect_of(getattr(block, "bbox", (0, 0, 0, 0)))
    return max(0, x1 - x0) * max(0, y1 - y0)


def contains(bbox: Any, x: float, y: float) -> bool:
    """Half-open containment: a click on the right/bottom edge belongs to the *next* box."""
    if isinstance(bbox, Sequence):
        x0, y0, x1, y1 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    else:
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    return x0 <= x < x1 and y0 <= y < y1


def _sized(text_w: int, text_h: int, pad: int) -> tuple[int, int]:
    """The chip's own size around a piece of measured text."""
    return int(text_w) + 2 * int(pad), int(text_h) + 2 * int(pad)


def _clamped(
    left: int, top: int, w: int, h: int, image_size: Optional[tuple[int, int]]
) -> tuple[int, int, int, int]:
    """A `w`×`h` chip with its top-left at (`left`, `top`), pushed back onto the page."""
    left, top = int(left), int(top)
    if image_size is not None:
        img_w, img_h = int(image_size[0]), int(image_size[1])
        left = max(0, min(left, max(0, img_w - w)))
        top = max(0, min(top, max(0, img_h - h)))
    return (left, top, left + w, top + h)


def chip_slots(
    bbox: Sequence[float],
    text_w: int,
    text_h: int,
    *,
    pad: int = DEFAULT_STYLE.chip_padding_px,
    image_size: Optional[tuple[int, int]] = None,
) -> tuple[tuple[str, tuple[int, int, int, int]], ...]:
    """Every home this chip could take, best first, as `(slot, rect)` pairs.

    ``"above"`` is :func:`chip_geometry` — the preference every painter already had — so a
    page with room to spare is painted exactly as it was before P8.6. The rest are the
    fallbacks a colliding chip tries before it gives up: tucked inside along the top edge,
    inside along the bottom edge, then just below the box. Slots that clamp to the same
    rectangle are listed once, so a chip against a page edge is not asked to try one spot
    four times.
    """
    x0, y0, x1, y1 = (int(bbox[i]) for i in range(4))
    w, h = _sized(text_w, text_h, pad)
    first = chip_geometry(bbox, text_w, text_h, pad=pad, image_size=image_size)
    out: list[tuple[str, tuple[int, int, int, int]]] = [("above", first)]
    seen = {first}
    for slot, top in (("inside-top", y0), ("inside-bottom", y1 - h), ("below", y1)):
        rect = _clamped(x0, top, w, h, image_size)
        if rect in seen:
            continue
        seen.add(rect)
        out.append((slot, rect))
    return tuple(out)


def id_text(block: Any) -> str:
    """The chip with the words taken off: ``#7``.

    The id is the part of a chip you cannot get any other way — it is the key into the block
    table, the JSON document and `focus_block`. When a chip has no room for its sentence, it
    keeps at least this.
    """
    return f"#{_block_id(block)}"


def chip_variants(block: Any, *, max_chars: int = DEFAULT_STYLE.label_max_chars) -> tuple[str, ...]:
    """The texts this block's chip is willing to wear, best first: the label, then `#id`.

    De-duplicated: a label that is already nothing but an id has one variant, and offering
    the same rectangle twice would make a chip look like it fitted on a second attempt.
    """
    full = label_text(block, max_chars=max_chars)
    short = id_text(block)
    return (full,) if short == full else (full, short)


def overlaps(a: Sequence[float], b: Sequence[float], *, gap: int = CHIP_GAP_PX) -> bool:
    """True when two rectangles cross by more than `gap` pixels on **both** axes.

    Touching edges and one-pixel grazes do not count: boxes that share a border are the
    normal case on a manga page, and calling a graze a collision would empty the page of
    labels.
    """
    return (
        min(a[2], b[2]) - max(a[0], b[0]) > gap
        and min(a[3], b[3]) - max(a[1], b[1]) > gap
    )


def _rect_area(rect: Sequence[float]) -> int:
    return max(0, int(rect[2]) - int(rect[0])) * max(0, int(rect[3]) - int(rect[1]))


@dataclass(frozen=True)
class ChipOffer:
    """One block's chip, as measured by the painter that offered it.

    Measuring is the painter's business — Pillow has `ImageDraw.textbbox`, the canvas has
    `QFontMetrics`, and they disagree by a pixel or two — so the shared rule below takes
    sizes rather than fonts. `variants` are `(text, width, height)` triples in preference
    order; build the texts with :func:`chip_variants` and measure each one.
    """

    block: Any
    box: tuple[int, int, int, int]
    variants: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class Chip:
    """A chip that has been given a place to live.

    `box` is the block's own box and `rect` where the chip ended up; keeping both means a
    painter never has to look the block up again, and `ChipPlan.paintable` can reproduce
    :func:`draw_order` exactly.
    """

    block: Any
    box: tuple[int, int, int, int]
    text: str
    rect: tuple[int, int, int, int]
    slot: str = "above"
    variant: int = 0


@dataclass(frozen=True)
class ChipPlan:
    """What de-collision decided: which chips are drawn, where, and which had no room.

    `skipped` keeps each rejected chip's *first* choice — the label it would have worn in
    its preferred slot — so a caller can still paint the block the user is pointing at (see
    :meth:`wanted`) without measuring anything again.
    """

    placed: tuple[Chip, ...] = ()
    skipped: tuple[Chip, ...] = ()

    @property
    def paintable(self) -> tuple[Chip, ...]:
        """The chips to draw, in paint order: the generic box first, the specific on top.

        The same order as :func:`draw_order`, so the chips that *do* fit land in the same
        stacking order as the boxes they belong to.
        """
        return tuple(sorted(self.placed, key=lambda c: (-_rect_area(c.box), _block_id(c.block))))

    @property
    def hidden(self) -> tuple[int, ...]:
        """Ids of the blocks whose words found no room anywhere."""
        return tuple(_block_id(c.block) for c in self.skipped)

    def wanted(self, block_id: int) -> Optional[Chip]:
        """This block's chip: the place it won, or the place it wanted and lost."""
        for chip in self.placed:
            if _block_id(chip.block) == block_id:
                return chip
        for chip in self.skipped:
            if _block_id(chip.block) == block_id:
                return chip
        return None

    def painted(self, block_id: int) -> bool:
        """Whether this block keeps a chip on the canvas at all."""
        return any(_block_id(c.block) == block_id for c in self.placed)


def place_chips(
    offers: Sequence[ChipOffer],
    *,
    style: OverlayStyle = DEFAULT_STYLE,
    image_size: Optional[tuple[int, int]] = None,
) -> ChipPlan:
    """Hand out chip positions so no two chips sit on each other (P8.6).

    The rule, in the order it is applied:

    1. **the specific box claims the space first** — chips are claimed from the smallest box
       upwards, the same rule that decides which box a click lands on (`pick_block`) and
       which box `contained_in` names as the container. A panel-sized false positive never
       out-claims the bubble inside it;
    2. **a chip that cannot have its favourite spot takes the next one** — above, inside
       along the top, inside along the bottom, below (:func:`chip_slots`);
    3. **a chip with no room for its words keeps its number** — the `#id` variant is far
       narrower, and the number is the part that cannot be recovered anywhere else;
    4. **only then is a chip dropped**, and the block is listed in `ChipPlan.hidden`. It is
       not lost: it keeps its box, its row in the table, its entry in the JSON, and its full
       text in the hover panel.

    Words beat position: a full label *below* the box is preferred over a bare `#id` above
    it. The layout is a pure function of the boxes offered, so it does not shift while the
    mouse moves — the block under the cursor is painted over the top by the caller instead
    (that is what "show it on hover instead" means here).

    `style.decollide_labels = False` restores the pre-8.6 behaviour exactly: every chip in
    its favourite spot, overlapping whatever is already there.
    """
    given = [offer for offer in (offers or ()) if offer.variants]
    if not given:
        return ChipPlan()
    if not style.decollide_labels:
        return ChipPlan(tuple(_first_choice(o, style, image_size) for o in given), ())

    taken: list[tuple[int, int, int, int]] = []
    placed: list[Chip] = []
    skipped: list[Chip] = []
    for offer in sorted(given, key=lambda o: (_rect_area(o.box), _block_id(o.block))):
        chip = _best_fit(offer, style, image_size, taken)
        if chip is None:
            skipped.append(_first_choice(offer, style, image_size))
            continue
        taken.append(chip.rect)
        placed.append(chip)
    return ChipPlan(tuple(placed), tuple(skipped))


def _first_choice(offer: ChipOffer, style: OverlayStyle, image_size) -> Chip:
    text, w, h = offer.variants[0]
    slot, rect = chip_slots(offer.box, w, h, pad=style.chip_padding_px, image_size=image_size)[0]
    return Chip(block=offer.block, box=offer.box, text=text, rect=rect, slot=slot, variant=0)


def _best_fit(
    offer: ChipOffer, style: OverlayStyle, image_size, taken: Sequence[tuple[int, int, int, int]]
) -> Optional[Chip]:
    for index, (text, w, h) in enumerate(offer.variants):
        for slot, rect in chip_slots(
            offer.box, w, h, pad=style.chip_padding_px, image_size=image_size
        ):
            if not any(overlaps(rect, other, gap=style.chip_gap_px) for other in taken):
                return Chip(block=offer.block, box=offer.box, text=text, rect=rect,
                            slot=slot, variant=index)
    return None


def flag_marks(block: Any) -> tuple[str, ...]:
    """Short marks for a block's 1.2 observations: `("⚠", "⊂3")`.

    A table cell and a status line have no room for sentences; `FLAG_MARKS` keeps this
    spelling, :func:`flag_note` and the tooltip the same.
    """
    marks: list[str] = []
    for flag in getattr(block, "flags", ()) or ():
        marks.append(FLAG_MARKS.get(flag, str(flag)))
    container = getattr(block, "contained_in", None)
    if container is not None:
        marks.append(f"{CONTAINMENT_MARK}{container}")
    return tuple(marks)


def flag_note(block: Any) -> str:
    """One readable line per observation, `""` when the block has nothing to confess.

    Says what the observation *means*, not only its name — a flag the user cannot act on is
    decoration. The key names are spelled as the JSON spells them, so the hover panel and
    the JSON pane describe one thing with one vocabulary.
    """
    notes: list[str] = []
    flags = tuple(getattr(block, "flags", ()) or ())
    if "possible_truncation" in flags:
        notes.append(
            "possible_truncation \u2014 the text may stop at max_new_tokens; "
            "raise the cap and read this box again"
        )
    for flag in flags:
        if flag != "possible_truncation":
            notes.append(f"{flag} \u2014 reported by the reader")
    container = getattr(block, "contained_in", None)
    if container is not None:
        notes.append(
            f"contained_in #{container} \u2014 a larger box holds this one; read #{container} first"
        )
    return "\n".join(notes)


def flagged_count(blocks: Sequence[Any]) -> int:
    """How many blocks carry a flag or a container, for the summary line."""
    return sum(
        1
        for b in (blocks or ())
        if (getattr(b, "flags", ()) or ()) or getattr(b, "contained_in", None) is not None
    )


def pick_block(blocks: Iterable[Any], x: float, y: float) -> Optional[Any]:
    """The block under `(x, y)`: smallest containing box wins (nested boxes, backlog 9)."""
    best = None
    for block in blocks:
        if not contains(block.bbox, x, y):
            continue
        if best is None or (_area(block), -_block_id(block)) < (_area(best), -_block_id(best)):
            best = block
    return best


def _block_id(block: Any) -> int:
    try:
        return int(getattr(block, "id", 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# The full-resolution painter (Save annotated PNG / MCP render_annotated_image)
# ---------------------------------------------------------------------------

def render_overlay(image: Image.Image,
                   blocks: Sequence[Any],
                   style: OverlayStyle = DEFAULT_STYLE,
                   *,
                   highlight: Optional[int] = None,
                   selected: Optional[int] = None,
                   ) -> Image.Image:
    """Return a **copy** of `image` with the overlay baked in. Never mutates the input.

    `highlight` / `selected` are block ids (hover and click state); both are drawn
    stronger so an annotated export can show what the user was looking at.

    The result is RGB, or RGBA when the input already was: boxes are coloured, so
    returning a greyscale page "unchanged" mode would silently flatten the palette.
    """
    if not isinstance(image, Image.Image):
        raise StudioError(ErrorCode.INVALID_IMAGE,
                          "render_overlay needs a PIL.Image.Image.",
                          {"got": type(image).__name__})
    blocks = list(blocks or [])
    was_rgba = image.mode == "RGBA"
    canvas = image.copy() if was_rgba else image.convert("RGBA")

    if not (style.overlay and blocks):
        return canvas.convert("RGB") if not was_rgba else canvas

    line = border_width_for(canvas.size, style)
    font = load_font(font_size_for(canvas.size, style))
    active = {i for i in (highlight, selected) if i is not None}

    # Translucent fills go on their own layer: alpha_composite is the only way to get
    # real translucency in Pillow (a filled rectangle with alpha painted straight onto
    # the canvas replaces the pixels instead of blending them).
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    fills = ImageDraw.Draw(layer)
    for block in draw_order(blocks):
        color = rgb_of(getattr(block, "color", ""), fallback_index=_block_id(block))
        boost = style.highlight_fill_boost if _block_id(block) in active else 1.0
        alpha = int(round(255.0 * max(0.0, min(1.0, style.fill_opacity)) * boost))
        box = _rect_of(block.bbox)
        fills.rectangle(box, fill=(color[0], color[1], color[2], min(255, alpha)))
    canvas = Image.alpha_composite(canvas, layer)

    draw = ImageDraw.Draw(canvas)
    boxed = []
    for block in draw_order(blocks):
        color = rgb_of(getattr(block, "color", ""), fallback_index=_block_id(block))
        box = _rect_of(block.bbox)
        bold = _block_id(block) in active
        draw.rectangle(box, outline=color, width=line * (2 if bold else 1))
        boxed.append((block, color, box))
    if style.labels:
        # Chips in a pass of their own, above every box, laid out by the same rule the
        # canvas uses (`place_chips`) — so the saved PNG and the window show the same page
        # (P8.6). The exported *document* is untouched; only the paint recipe changed.
        colors = {_block_id(block): color for block, color, _box in boxed}
        plan = place_chips(
            [_pillow_offer(draw, font, block, box, style) for block, _c, box in boxed],
            style=style,
            image_size=canvas.size,
        )
        for chip in plan.paintable:
            _draw_chip(draw, chip, colors[_block_id(chip.block)], font, style)
        # The block the user was looking at always keeps its label, drawn last and over
        # anybody in the way: an export carries what was on screen, not what fitted.
        for block, color, _box in boxed:
            bid = _block_id(block)
            if bid in active and not plan.painted(bid):
                wanted = plan.wanted(bid)
                if wanted is not None:
                    _draw_chip(draw, wanted, color, font, style)
    return canvas.convert("RGB") if not was_rgba else canvas


def _rect_of(bbox: Any) -> tuple[int, int, int, int]:
    """`BBox` | `(x0, y0, x1, y1)` | any object with x0..y1 -> an int tuple."""
    if isinstance(bbox, BBox):
        return bbox.as_tuple()
    if isinstance(bbox, Sequence):
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    return (int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1))


def _pillow_offer(draw: ImageDraw.ImageDraw, font: Any, block: Any,
                  box: tuple[int, int, int, int], style: OverlayStyle) -> ChipOffer:
    """Measure this block's chip texts with Pillow, for :func:`place_chips`."""
    variants = []
    for text in chip_variants(block, max_chars=style.label_max_chars):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        variants.append((text, x1 - x0, y1 - y0))
    return ChipOffer(block=block, box=box, variants=tuple(variants))


def _draw_chip(draw: ImageDraw.ImageDraw, chip: Chip, color: tuple[int, int, int],
               font: Any, style: OverlayStyle) -> None:
    """One label chip, at the place :func:`place_chips` gave it.

    The rectangle was chosen elsewhere so the canvas and this painter cannot disagree; the
    text is re-measured here only to line the glyphs up inside the chip's padding.
    """
    x0, y0, x1, y1 = draw.textbbox((0, 0), chip.text, font=font)
    draw.rectangle(chip.rect, fill=color)
    draw.text((chip.rect[0] + style.chip_padding_px - x0,
               chip.rect[1] + style.chip_padding_px - y0),
              chip.text, fill=contrast_text("#%02X%02X%02X" % color), font=font)


def annotated_png(image: Image.Image, blocks: Sequence[Any],
                  style: OverlayStyle = DEFAULT_STYLE, *,
                  highlight: Optional[int] = None,
                  selected: Optional[int] = None) -> bytes:
    """The overlay baked at full resolution, as PNG bytes (MCP returns these directly)."""
    from .imaging import to_png_bytes

    return to_png_bytes(render_overlay(image, blocks, style,
                                       highlight=highlight, selected=selected))


def save_annotated(image: Image.Image, blocks: Sequence[Any],
                   path: Union[str, Path],
                   style: OverlayStyle = DEFAULT_STYLE, *,
                   highlight: Optional[int] = None,
                   selected: Optional[int] = None) -> Path:
    """Write :func:`render_overlay` to `path` (parent dirs created). Full resolution."""
    target = Path(str(path))
    rendered = render_overlay(image, blocks, style, highlight=highlight, selected=selected)
    try:
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(target)
    except OSError as exc:
        raise StudioError(ErrorCode.IO_ERROR, "Cannot write annotated image.",
                          {"path": str(target), "reason": str(exc)}) from exc
    return target
