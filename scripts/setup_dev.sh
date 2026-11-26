#!/usr/bin/env bash
# Dev environment bootstrap.
#
#   bash scripts/setup_dev.sh              # test-only deps (pytest, jsonschema, timeout)
#   bash scripts/setup_dev.sh --detect     # + real detection (dghs-imgutils, onnxruntime)
#   bash scripts/setup_dev.sh --detect --gui   # + PySide6 (Part 5)
#   bash scripts/setup_dev.sh --venv --detect  # force an isolated .venv instead
#
# Two supported environments:
#  1. Rich interpreter already carrying pytest/pillow/numpy (e.g. the Unsloth Studio
#     sandbox, where pip installs land in $PIP_PREFIX = ./.unsloth-packages, already on
#     PYTHONPATH). Default mode: install only what is missing, no venv.
#  2. Bare system python -> pass --venv to create ./.venv (NOT persisted across
#     sandbox snapshots; re-run this script at the start of a session).
#
# NOTE on dghs-imgutils: it pins numpy<2, which has no wheels for Python 3.13 and
# would downgrade numpy under torch. We therefore install it with --no-deps and add
# the handful of modules the YOLO path actually imports. Verified working set:
#   dghs-imgutils hfutils hbutils onnxruntime opencv-contrib-python pyrfc6266
#   piexif pyclipper shapely emoji pilmoji
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=0; WANT_DETECT=0; WANT_GUI=0; WANT_OCR=0
for arg in "$@"; do
  case "$arg" in
    --venv) VENV=1 ;;
    --detect) WANT_DETECT=1 ;;
    --gui) WANT_GUI=1 ;;
    --ocr) WANT_OCR=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY=python3
if [ "$VENV" = 1 ] || ! $PY -c "import pytest" >/dev/null 2>&1; then
  echo "Creating .venv ..."
  $PY -m venv .venv
  PY=".venv/bin/python"
  "$PY" -m pip install --quiet --upgrade pip
fi
echo "Interpreter: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"

install() { echo "+ $*"; "$PY" -m pip install --quiet "$@"; }

install pytest jsonschema pytest-timeout

if [ "$WANT_DETECT" = 1 ]; then
  install --no-deps dghs-imgutils
  install hfutils hbutils onnxruntime
  # imgutils.generic.__init__ imports its siblings eagerly -> these are needed to
  # import the package at all, even though the YOLO module itself does not use them.
  install --no-deps opencv-contrib-python pyrfc6266 piexif pyclipper shapely "emoji>=2.5,<2.12" pilmoji
fi

[ "$WANT_GUI" = 1 ] && install --no-deps PySide6
[ "$WANT_OCR" = 1 ] && install torch transformers accelerate  # Blackwell GPUs: add --index-url .../whl/cu128

"$PY" - <<'EOF'
import importlib
for mod in ("pytest", "jsonschema", "PIL", "numpy", "huggingface_hub",
            "imgutils.generic.yolo", "onnxruntime", "PySide6.QtCore", "torch", "transformers"):
    try:
        importlib.import_module(mod)
        print(f"  ok      {mod}")
    except Exception:
        print(f"  missing {mod}")
EOF

echo
echo "Run tests:   $PY -m pytest                 # offline unit tests (fakes)"
echo "Real models: $PY -m pytest -m models -s    # needs HF_TOKEN (gated detection repo)"
echo "Preflight:   $PY scripts/check_hf_access.py"
echo "Demo:        $PY -m animetext_studio.tools.detect_demo examples/sample_page.png --mock"
echo "             $PY -m animetext_studio.tools.detect_demo examples/sample_page.png --model yolo12n_animetext --annotate out/boxes.png"
