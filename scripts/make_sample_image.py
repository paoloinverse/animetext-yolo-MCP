#!/usr/bin/env python3
"""Generate `examples/sample_page.png` — a synthetic manga-like page for demos/tests.

No network needed; uses a locally installed CJK font (falls back to Latin text).
Useful for exercising the detection demo, the OCR demo and the GUI without shipping
copyrighted artwork in the repo.

    python scripts/make_sample_image.py [--out examples/sample_page.png] [--seed 7]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

JA_LINES = [
    "この攻撃で決まりだ！",
    "まだだよ、私は",
    "絶対に負けない",
    "何!? その力は",
    "明日、学校で会おう",
    "真相はいつも闇の中",
]
JA_VERTICAL = ["決戦の前夜", "風が止んだ", "第４話", "その日、空は赤かった"]
EN_LINES = [
    "The final attack ends it here!",
    "I will not give up, not ever.",
    "What?! Where does that power come from?",
    "See you at school tomorrow.",
]


def pick_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()  # pragma: no cover


def cjk_ok(font: ImageFont.FreeTypeFont) -> bool:
    try:
        return font.getbbox("あ")[2] > 0 and "dejavu" not in font.path.lower()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="examples/sample_page.png")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img = Image.new("RGB", (args.width, args.height), (252, 252, 250))
    draw = ImageDraw.Draw(img)

    # panel frames
    for (x0, y0, x1, y1) in ((40, 40, 860, 600), (40, 630, 420, 1250), (450, 630, 860, 1250)):
        draw.rectangle([x0, y0, x1, y1], outline=(20, 20, 20), width=4)

    lines = JA_LINES if cjk_ok(pick_font(30)) else EN_LINES

    # speech bubbles with wrapped text
    bubbles = [(120, 110, 380, 250), (480, 150, 800, 300), (150, 760, 380, 900),
               (520, 900, 820, 1040)]
    for i, (x0, y0, x1, y1) in enumerate(bubbles):
        draw.rounded_rectangle([x0, y0, x1, y1], radius=40, outline=(15, 15, 15), width=3)
        font = pick_font(26)
        text = lines[i % len(lines)]
        words = [text] if len(text) < 14 else text.split(" ")
        y = y0 + 18
        chunk = ""
        for w in words:
            trial = f"{chunk} {w}".strip()
            if draw.textbbox((0, 0), trial, font=font)[2] > (x1 - x0 - 30) and chunk:
                draw.text((x0 + 15, y), chunk, fill=(10, 10, 10), font=font)
                y += 32
                chunk = w
            else:
                chunk = trial
        if chunk:
            draw.text((x0 + 15, y), chunk, fill=(10, 10, 10), font=font)

    # vertical text columns (tategaki), right to left
    vfont = pick_font(30)
    for col, text in enumerate(JA_VERTICAL if cjk_ok(vfont) else ["chapter four", "the wind stops"]):
        x = args.width - 120 - col * 60
        y = 340 + rng.randint(0, 40)
        for ch in text:
            if ch == " ":
                y += 18
                continue
            draw.text((x, y), ch, fill=(12, 12, 12), font=vfont)
            y += 36

    # narration band
    draw.rectangle([60, 1150, 840, 1230], fill=(235, 235, 235), outline=(20, 20, 20), width=3)
    draw.text((80, 1170), lines[-1], fill=(10, 10, 10), font=pick_font(28))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"Wrote {out} ({img.width}x{img.height}); CJK font in use: {cjk_ok(pick_font(30))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
