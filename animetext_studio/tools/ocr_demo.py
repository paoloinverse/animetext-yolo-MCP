"""Stage-2 demo: crop-level OCR (PLAN.md Part 3 exit demo).

Examples
--------
    # deterministic, no torch, no downloads:
    python -m animetext_studio.tools.ocr_demo crop.png --mock

    # real model, on-premises weights, no network at all:
    python -m animetext_studio.tools.ocr_demo crop.png --offline

    # several crops from one page (boxes in original-image pixels):
    python -m animetext_studio.tools.ocr_demo page.png --bbox 812,96,918,402 \
        --bbox 100,500,220,860 --padding 2 --max-patches auto

    # batch a directory of crops through the engine in one call:
    python -m animetext_studio.tools.ocr_demo crops/*.png --batch-size 16

Prints a JSON document on stdout: engine status, effective OCR settings and one
entry per crop (source/bbox/crop size/text). Exit code 0 on success, else the
StudioError exit code (9 = OCR_FAILED, 8 = ENGINE_LOAD_FAILED, 12 = MODEL_NOT_FOUND,
11 = INVALID_IMAGE, 5 = CANCELLED, ...).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):     # `python animetext_studio/tools/ocr_demo.py crop.png`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "animetext_studio.tools"

from .. import APP_NAME, __version__
from ..core.imaging import crop_pair, load_image
from ..core.types import (
    BBox, SCHEMA_VERSION, ErrorCode, StudioError, now_iso,
)
from ..engines.base import BaseOcrEngine, FakeOcrEngine, ProgressCallback
from ..engines.ocr import DTYPE_CHOICES, OcrEngine


def _patches_arg(text: str) -> Any:
    if text == "auto":
        return "auto"
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"max-patches must be 'auto' or one of {BaseOcrEngine.MAX_PATCH_CHOICES}, got {text!r}")
    if value not in BaseOcrEngine.MAX_PATCH_CHOICES:
        raise argparse.ArgumentTypeError(
            f"max-patches must be 'auto' or one of {BaseOcrEngine.MAX_PATCH_CHOICES}, got {value}")
    return value


def _bbox_arg(text: str) -> BBox:
    parts = text.replace(":", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"bbox must be x0,y0,x1,y1, got {text!r}")
    try:
        box = BBox(*[int(float(p)) for p in parts])
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"bbox coordinates must be integers, got {text!r}")
    if not box.is_valid():
        raise argparse.ArgumentTypeError(f"bbox is empty (x1<=x0 or y1<=y0): {text!r}")
    return box


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr_demo",
        description="Run stage 2 (Hayai crop OCR) on crop images or boxed regions of a page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="*", metavar="IMAGE",
                        help="crop image(s); with --bbox, one page image instead")
    parser.add_argument("--bbox", action="append", type=_bbox_arg, default=None, metavar="X0,Y0,X1,Y1",
                        help="OCR this region of the (single) page image; repeatable")
    parser.add_argument("--padding", type=int, default=2, metavar="PX",
                        help="crop padding in px (default 2, same as ocr.crop_padding_px)")
    parser.add_argument("--mock", action="store_true",
                        help="use the deterministic fake engine (no torch, no weights)")
    parser.add_argument("--device", default="auto",
                        help="OCR device: auto|cpu|cuda|cuda:N (default auto)")
    parser.add_argument("--dtype", choices=DTYPE_CHOICES, default="auto",
                        help="inference dtype override (default: the model decides per device)")
    parser.add_argument("--cuda-graphs", dest="cuda_graphs", action="store_true", default=None,
                        help="force CUDA graphs on (auto: on for CUDA, off elsewhere)")
    parser.add_argument("--no-cuda-graphs", dest="cuda_graphs", action="store_false",
                        help="force CUDA graphs off")
    parser.add_argument("--max-patches", type=_patches_arg, default="auto",
                        help=f"'auto' or one of {BaseOcrEngine.MAX_PATCH_CHOICES} (default auto)")
    parser.add_argument("--batch-size", type=int, default=16, metavar="N",
                        help="crops per engine batch/chunk (default 16)")
    parser.add_argument("--max-new-tokens", type=int, default=128, metavar="N",
                        help="decoder budget per crop (default 128)")
    parser.add_argument("--normalize-nfkc", action="store_true",
                        help="apply NFKC normalization + whitespace collapse to the output")
    parser.add_argument("--models-dir", metavar="PATH",
                        help="on-premises bundle (default: $ANIMETEXT_MODELS_DIR or <package>/models)")
    parser.add_argument("--offline", action="store_true",
                        help="never touch the network: bundled snapshot only (D26)")
    parser.add_argument("--out", metavar="PATH", help="also write the JSON document here")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    return parser


def _make_engine(args: argparse.Namespace):
    if args.mock:
        return FakeOcrEngine()
    return OcrEngine(args.device, models_dir=args.models_dir, offline=args.offline,
                     dtype=args.dtype, use_cuda_graph=args.cuda_graphs)


def collect_crops(args: argparse.Namespace) -> tuple[list[Any], list[dict[str, Any]]]:
    """-> (crops, per-crop source records). Two input shapes, one output shape."""
    if args.bbox:
        if len(args.images) != 1:
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              "--bbox needs exactly one page image.",
                              {"got": args.images})
        page, info = load_image(Path(args.images[0]))
        crops: list[Any] = []
        records: list[dict[str, Any]] = []
        for box in args.bbox:
            effective, crop_box = crop_pair(page, box, padding=args.padding)
            crops.append(effective)
            records.append({
                "source": info.file_name, "source_size": [info.width, info.height],
                "bbox": crop_box.to_dict(), "crop_size": [effective.width, effective.height],
            })
        return crops, records

    if not args.images:
        raise StudioError(ErrorCode.INVALID_ARGUMENT,
                          "Give at least one crop image (or one page + --bbox).")
    crops, records = [], []
    for name in args.images:
        image, info = load_image(Path(name))
        crops.append(image)
        records.append({"source": info.file_name, "source_size": [info.width, info.height],
                        "bbox": None, "crop_size": [image.width, image.height]})
    return crops, records


def run(args: argparse.Namespace, progress: Optional[ProgressCallback] = None) -> dict[str, Any]:
    engine = _make_engine(args)
    crops, records = collect_crops(args)

    started = time.perf_counter()
    texts = engine.ocr_crops(
        crops,
        max_patches=args.max_patches,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        normalize_nfkc=args.normalize_nfkc or None,
        progress=progress,
    )
    elapsed = int((time.perf_counter() - started) * 1000)

    results = []
    for index, (record, text) in enumerate(zip(records, texts)):
        results.append({
            "index": index,
            **record,
            "text": text,
            "chars": len(text),
            "empty": not text,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "ocr",
        "created_at": now_iso(),
        "engine": engine.status().to_dict(),
        "ocr": {
            "repo": "fake" if args.mock else engine.plan()["repo_id"],
            "weights_source": getattr(engine, "weights_source", None),
            "device": engine.status().device,
            "max_patches": args.max_patches,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "crop_padding_px": args.padding,
            "normalize_nfkc": bool(args.normalize_nfkc),
        },
        "timing_ms": {"ocr": elapsed},
        "crop_count": len(crops),
        "empty_count": sum(1 for t in texts if not t),
        "results": results,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(stage: str, percent: float, message: str) -> None:
        print(f"[{stage}] {percent:5.1f}% {message}", file=sys.stderr)

    try:
        document = run(args, progress=None if args.quiet else progress)
    except StudioError as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    payload = json.dumps(document, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
