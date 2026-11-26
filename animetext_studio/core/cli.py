"""Command-line interface (PLAN.md §5).

All value options default to None so that only explicitly-set flags participate
in the override layering (CLI > config file > defaults). Flags added in P1 for
foundation utility: --validate-config, --export-config (decision D9).
"""
from __future__ import annotations

import argparse
import re
from typing import Optional, Sequence

from .. import APP_NAME, __version__
from .config import MCP_TRANSPORTS

_DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:[0-9]+)?)$")


def _threshold_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}")
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"threshold must be in [0, 1], got {value}")
    return value


def _port_arg(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer port, got {text!r}")
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"port must be in [1, 65535], got {value}")
    return value


def _device_arg(text: str) -> str:
    if not _DEVICE_RE.match(text):
        raise argparse.ArgumentTypeError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:N', got {text!r}")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=(
            "AnimeText OCR Studio: detect anime/manga text blocks (deepghs/AnimeText_yolo), "
            "OCR each crop (hayai-ocr-v2.7-supernova), output box→text JSON. "
            "PySide6 GUI + MCP remote control."
        ),
        epilog=(
            "examples:\n"
            "  animetext-studio --dump-default-config\n"
            "  animetext-studio --config my.json --model yolo12x_animetext --score 0.425\n"
            "  animetext-studio --mcp-host 0.0.0.0 --mcp-port 9000 --mcp-token secret\n"
            "  python scripts/fetch_models.py --all          # vendor weights into <package>/models\n"
            "  animetext-studio --model-status               # what is installed on premises\n"
            "  animetext-studio --doctor                     # why this machine cannot run it\n"
            "  animetext-studio --offline --model yolo12m_animetext ...   # no network at all\n"
            "  animetext-studio --headless --image page.png --run --out result.json   (Part 4)\n"
            "  animetext-studio --headless --serve --mock                 (MCP over HTTP)\n"
            "  animetext-studio --headless --mcp-transport stdio --mock   (MCP over stdio)\n"
            "  animetext-studio --dump-mcp-tools animetext_studio/schemas/mcp-tools.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")

    gui = parser.add_argument_group("GUI / app")
    gui.add_argument("--config", metavar="PATH", help="load configuration JSON file (CLI flags override it)")
    gui.add_argument("--image", metavar="PATH", help="image to load at startup")
    gui.add_argument("--model", metavar="NAME", help="detection model (e.g. yolo12m_animetext)")
    gui.add_argument("--iou", metavar="F", type=_threshold_arg, help="IOU threshold [0..1] (default 0.35)")
    gui.add_argument("--score", metavar="F", type=_threshold_arg, help="score/confidence threshold [0..1] (default 0.05)")
    gui.add_argument("--device", metavar="DEV", type=_device_arg, help="OCR device: auto|cpu|cuda|cuda:N")
    gui.add_argument("--mock", action="store_true", help="use fake engines (no downloads; development/demo)")
    # Mutually exclusive, not last-one-wins: a harness that passes both is asking two different
    # questions, and silently believing the second one is how a benchmark lies.
    warm = gui.add_mutually_exclusive_group()
    warm.add_argument("--preload", action="store_true",
                      help="warm up the engines at startup: load the weights AND run one "
                           "throwaway inference, so the first real run is not 2-3x slower. "
                           "Runs as a cancellable 'warmup' job after the window is up; "
                           "skipped under --mock; also honoured by --serve (headless)")
    warm.add_argument("--preload-off", action="store_true",
                      help="name the cold baseline out loud: no warm-up anywhere in this run, "
                           "whatever else defaults to it. Off by default, so this is mostly a "
                           "flag a benchmark passes to say what it measured against (P9.4)")
    gui.add_argument("--warm-devices", metavar="DEVICES",
                     help="benchmark mode: load + one first inference per device (comma-"
                          "separated, e.g. 'cpu,cuda'; 'all' = cpu,cuda) and print a per-device "
                          "table, then exit. Each device gets its own engine pair, warmed cold "
                          "and released, so the numbers are comparable — and each row says which "
                          "device was asked for and which one the engine actually got")
    gui.add_argument("--headless", action="store_true", help="run without GUI")

    headless = parser.add_argument_group("headless")
    headless.add_argument("--run", action="store_true", help="with --headless: run the pipeline and print result JSON (Part 4)")
    headless.add_argument("--out", metavar="PATH", help="with --run: also write result JSON to this file")

    mcp = parser.add_argument_group("MCP server")
    mcp.add_argument("--serve", action="store_true",
                     help="start the MCP server and stay up (headless serving; implied by "
                          "--mcp-transport stdio)")
    mcp.add_argument("--mcp-transport", choices=MCP_TRANSPORTS, help="MCP transport (default streamable-http; 'none' disables)")
    mcp.add_argument("--mcp-host", metavar="ADDR", help="MCP bind address (default 127.0.0.1)")
    mcp.add_argument("--mcp-port", metavar="N", type=_port_arg, help="MCP bind port (default 8765)")
    mcp.add_argument("--mcp-token", metavar="TOKEN", help="optional bearer token required from MCP clients")
    mcp.add_argument("--dump-mcp-tools", nargs="?", const="-", metavar="PATH",
                     help="write the machine-readable MCP tool catalog (JSON) to PATH, "
                          "or to stdout with '-', and exit")

    hf = parser.add_argument_group("Hugging Face / models")
    hf.add_argument("--hf-token", metavar="TOKEN", help="HF token for the gated detection repo (never written to config files)")
    hf.add_argument("--hf-cache-dir", metavar="PATH", help="HF cache directory (default: standard HF cache)")
    hf.add_argument("--models-dir", metavar="PATH",
                    help="on-premises model bundle (default: $ANIMETEXT_MODELS_DIR or <package>/models)")
    hf.add_argument("--offline", action="store_true",
                    help="never use the network: only bundled/cached weights may load")
    hf.add_argument("--model-status", action="store_true",
                    help="print a JSON report of the on-premises model bundle and exit")
    hf.add_argument("--doctor", action="store_true",
                    help="read-only environment report — the Python floor, the imgutils "
                         "import-time pip install, the numpy pin, torch/CUDA, the bundle, the "
                         "token, a CJK font — and exit. Never installs, never downloads, never "
                         "writes; --verbose adds the JSON")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--dump-default-config", action="store_true", help="print the default configuration JSON and exit")
    misc.add_argument("--validate-config", metavar="PATH", help="validate a configuration file, print a JSON report and exit")
    misc.add_argument("--export-config", metavar="PATH", help="write the resolved configuration (defaults+file+CLI) and exit")
    misc.add_argument("--redact-secrets", action="store_true",
                      help="with --export-config: write <redacted> instead of secret values "
                           "(mcp.auth_token) so the file is safe to share or attach to a bug "
                           "report; importing such a file keeps the live value (P8)")
    misc.add_argument("--gui-smoke", action="store_true",
                      help="build the main window, lay it out, print a one-line summary and "
                           "exit 0 (packaging / CI check; combine with --mock)")
    misc.add_argument("--verbose", action="store_true", help="verbose logging")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    # `--preload-off` is the same switch seen from the other side. Normalise it here rather than
    # at each consumer, so the window, `--serve` and the sweep all read one answer about whether
    # this process intends to warm up — and a harness can name the cold baseline it measured
    # against without trusting that nothing else changed the default (P9.4).
    if getattr(args, "preload_off", False):
        args.preload = False
    return args
