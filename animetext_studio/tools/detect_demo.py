"""Stage-1 demo: text-block detection only (PLAN.md Part 2 exit demo).

Examples
--------
    # deterministic, no downloads / no token:
    python -m animetext_studio.tools.detect_demo page.png --mock

    # real model (needs HF_TOKEN for the gated deepghs/AnimeText_yolo repo):
    python -m animetext_studio.tools.detect_demo page.png --model yolo12n_animetext \\
        --iou 0.35 --score 0.05 --annotate out_boxes.png

    # repo-recommended F1-optimal threshold instead of 0.05:
    python -m animetext_studio.tools.detect_demo page.png --score recommended

    # what does the repo actually offer (offline -> the 5 known names):
    python -m animetext_studio.tools.detect_demo --list-models

Prints a JSON document on stdout: image info, effective detection settings, the
detection array (bbox/label/score) and timings. Exit code is 0 on success, else the
StudioError exit code (6 = HF_AUTH_REQUIRED, 7 = DOWNLOAD_FAILED, 12 = MODEL_NOT_FOUND,
8 = ENGINE_LOAD_FAILED, 11 = INVALID_IMAGE, ...).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):     # `python animetext_studio/tools/detect_demo.py page.png`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "animetext_studio.tools"

from .. import APP_NAME
from ..core.config import KNOWN_MODELS, MODEL_PRESETS
from ..core.types import (
    DETECTION_REPO, SCHEMA_VERSION, ErrorCode, ImageInfo, StudioError,
    color_for_index, now_iso, sha256_bytes,
)
from ..engines.base import FakeDetectionEngine, ProgressCallback
from ..engines.detection import DetectionEngine, PROVIDER_CHOICES

def _score_arg(text: str) -> Optional[float]:
    """--score accepts a number or 'recommended'/'default' (-> repo threshold.json)."""
    if text.lower() in ("recommended", "default", "repo", "none"):
        return None
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number in [0, 1] or 'recommended', got {text!r}")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"threshold must be in [0, 1], got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{APP_NAME}-detect-demo",
        description="Run stage 1 (AnimeText YOLO text-block detection) on one image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", metavar="IMAGE", help="input image (png/jpg/...)")
    parser.add_argument("--model", default="yolo12m_animetext",
                        help=f"detection model (default: yolo12m_animetext; known: {', '.join(KNOWN_MODELS)})")
    parser.add_argument("--iou", type=float, default=0.35, help="IOU threshold [0..1] (default 0.35, the Space demo value)")
    parser.add_argument("--score", type=_score_arg, default=0.05,
                        help="score threshold [0..1], or 'recommended' for the repo's F1-optimal value (default 0.05)")
    parser.add_argument("--allow-dynamic", action="store_true",
                        help="preserve aspect ratio when resizing (better for tall manga pages)")
    parser.add_argument("--mock", action="store_true", help="use the deterministic fake engine (no network/token)")
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, default=None,
                        help="ONNX Runtime provider for the real engine (default: auto)")
    parser.add_argument("--hf-token", metavar="TOKEN", help="HF token (else HF_TOKEN env); never persisted")
    parser.add_argument("--hf-cache-dir", metavar="PATH", help="HF cache directory for the model files")
    parser.add_argument("--models-dir", metavar="PATH",
                        help="on-premises bundle to load weights from "
                             "(default: $ANIMETEXT_MODELS_DIR or <package>/models)")
    parser.add_argument("--offline", action="store_true",
                        help="never touch the network: bundled weights only (D26)")
    parser.add_argument("--list-models", action="store_true", help="print the model catalog and exit")
    parser.add_argument("--annotate", metavar="PATH", help="also write a PNG with the boxes drawn")
    parser.add_argument("--out", metavar="PATH", help="also write the JSON document to this file")
    parser.add_argument("--limit", type=int, default=0, help="print at most N detections (0 = all)")
    parser.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    return parser


# ---------------------------------------------------------------------------
# image loading (core/imaging.py arrives in Part 3; the demo keeps a private
# minimal loader so Part 2 stays independently reviewable)
# ---------------------------------------------------------------------------

def load_image(path: Path) -> tuple[Any, ImageInfo]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover
        raise StudioError(ErrorCode.ENGINE_LOAD_FAILED,
                          "Pillow is required (pip install animetext-studio[detect]).",
                          {"reason": str(exc)}) from exc
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StudioError(ErrorCode.IO_ERROR, f"Cannot read image: {path}",
                          {"reason": str(exc)}) from exc
    try:
        image = Image.open(Path(path))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise StudioError(ErrorCode.INVALID_IMAGE, f"Not a readable image: {path}",
                          {"reason": f"{type(exc).__name__}: {exc}"}) from exc
    info = ImageInfo(
        source="file", file_name=path.name,
        width=image.width, height=image.height, sha256=sha256_bytes(raw),
    )
    return image, info


def annotate(image: Any, detections: list, path: Path) -> Path:
    """Bake the boxes (colour + label chip) into a PNG copy of the input.

    Delegates to `core/annotation.py` (Part 6): one drawing recipe for the demo, the GUI
    canvas and the MCP image tool, so a preview and a real result cannot look different.
    A bare `Detection` has no text, so the chip carries its label instead.
    """
    from types import SimpleNamespace

    from ..core.annotation import OverlayStyle, save_annotated

    blocks = [SimpleNamespace(id=i, bbox=det.bbox, score=det.score, text=det.label,
                              error=None, color=color_for_index(i))
              for i, det in enumerate(detections)]
    return save_annotated(image, blocks, path,
                          OverlayStyle(overlay=True, labels=True, label_max_chars=16))


def _make_engine(args: argparse.Namespace):
    if args.mock:
        return FakeDetectionEngine()
    return DetectionEngine(hf_token=args.hf_token, hf_cache_dir=args.hf_cache_dir,
                           provider=args.provider, models_dir=args.models_dir,
                           offline=args.offline)


def run(args: argparse.Namespace, progress: Optional[ProgressCallback] = None) -> dict[str, Any]:
    engine = _make_engine(args)

    if args.list_models:
        # List first, then read the status: `list_models()` is what decides whether the
        # repo answered, and `engine.detail` must reflect that (dict literals evaluate
        # in order, so putting status() first used to report a stale, empty detail).
        models = engine.list_models()
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "list_models",
            "repo": DETECTION_REPO if not args.mock else "fake",
            "engine": engine.status().to_dict(),
            "models": models,
            "presets": MODEL_PRESETS,
        }

    if not args.image:
        raise StudioError(ErrorCode.INVALID_ARGUMENT,
                          "An IMAGE argument is required (or use --list-models).")
    image, info = load_image(Path(args.image))

    started = time.perf_counter()
    detections = engine.detect(
        image, args.model,
        iou_threshold=args.iou,
        score_threshold=args.score,
        allow_dynamic=args.allow_dynamic,
        progress=progress,
    )
    detect_ms = int(round((time.perf_counter() - started) * 1000))

    # What will the inference actually have used? Explicit value, else the repo's
    # threshold.json (real engine) / the fake's constant (mock).
    effective_score = engine.resolved_score_threshold(args.model, args.score)
    recommended = engine.recommended_threshold(args.model)
    document = {
        "schema_version": SCHEMA_VERSION,
        "stage": "detect",
        "created_at": now_iso(),
        "image": info.to_dict(),
        "detection": {
            "repo": "fake" if args.mock else DETECTION_REPO,
            "model": args.model,
            "iou_threshold": args.iou,
            "score_threshold": effective_score,
            "score_threshold_requested": args.score,
            "score_threshold_mode": "explicit" if args.score is not None else "engine_default",
            "recommended_score_threshold": recommended,
            "allow_dynamic": args.allow_dynamic,
            "backend": engine.status().backend,
        },
        "engine": engine.status().to_dict(),
        "timing_ms": {"detect": detect_ms},
        "detection_count": len(detections),
        "detections": [
            {"index": i, **det.to_dict()} for i, det in enumerate(detections)
        ],
    }
    if args.annotate:
        document["annotated_image"] = str(annotate(image, detections, Path(args.annotate)))
    if args.limit:
        document["detections"] = document["detections"][: args.limit]
    return document


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(stage: str, percent: float, message: str) -> None:
        if not args.quiet:
            print(f"[{stage}] {percent:5.1f}%  {message}", file=sys.stderr, flush=True)

    try:
        document = run(args, progress=progress if not args.quiet else None)
    except StudioError as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover
        err = StudioError(ErrorCode.CANCELLED, "Interrupted.")
        print(json.dumps({"error": err.to_dict()}), file=sys.stderr)
        return err.exit_code

    text = json.dumps(document, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        try:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            err = StudioError(ErrorCode.IO_ERROR, f"Cannot write {args.out}",
                              {"reason": str(exc)})
            print(json.dumps({"error": err.to_dict()}), file=sys.stderr)
            return err.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
