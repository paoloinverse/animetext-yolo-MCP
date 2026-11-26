"""Image IO, crops and Qt interop (PLAN Part 3).

Pure Pillow at import time: PySide6 is imported **lazily** inside the two QImage
helpers, so the pipeline, the CLI and the MCP server work without the GUI extra (D7).
Everything here maps failures to ``StudioError`` — the GUI, headless run and MCP all
surface the same {code, message} contract (PLAN 2.3).

Crops matter for OCR quality: Hayai resizes every crop to a patch grid, so a box that
clips a stroke or lands on the bubble border changes the answer. ``crop()`` therefore
pads (clamped to the image) and guarantees a minimum side length (D31).
"""
from __future__ import annotations

import io
import unicodedata
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image, ImageOps

from .types import (
    BBox, ErrorCode, ImageInfo, StudioError, color_for_index, sha256_bytes,
)

__all__ = [
    "MIN_CROP_SIDE", "MAX_IMAGE_PIXELS", "load_image", "image_from_bytes",
    "image_from_base64", "describe", "crop", "crop_of", "is_ocr_worthy",
    "normalize_text", "to_png_bytes", "thumbnail", "pil_to_qimage", "qimage_to_pil",
    "color_for_index",
]

#: Crops below this side length carry no legible text; they are upscaled rather than
#: rejected (Hayai resamples to a 16 px patch grid, and a 1 px image is both useless and
#: a crash-bait for some resize paths). Detection filters real junk via `pipeline.min_box_area_px`.
MIN_CROP_SIDE = 8

#: Area floor used by :func:`is_ocr_worthy`; deliberately the same number as the
#: pipeline's configurable `pipeline.min_box_area_px` default, so the two agree out of
#: the box and P4 can override both with one value.
MIN_AREA_CROP = 16

#: Guard against decompression bombs / absurd inputs. Pillow's own limit (~178 MP) only
#: raises; ours produces a structured INVALID_IMAGE with the actual size.
MAX_IMAGE_PIXELS = 120_000_000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _open(data: bytes, origin: str) -> Image.Image:
    """Decode bytes -> RGB PIL image (EXIF orientation applied), with structured errors."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Image.DecompressionBombError as exc:      # pragma: no cover - needs a huge file
        raise StudioError(
            ErrorCode.INVALID_IMAGE,
            f"Image is too large to decode safely: {origin}",
            {"reason": str(exc), "max_pixels": MAX_IMAGE_PIXELS},
        ) from exc
    except Exception as exc:                          # noqa: BLE001
        raise StudioError(
            ErrorCode.INVALID_IMAGE, f"Not a readable image: {origin}",
            {"reason": f"{type(exc).__name__}: {exc}", "origin": origin},
        ) from exc

    pixels = image.width * image.height
    if pixels > MAX_IMAGE_PIXELS:
        raise StudioError(
            ErrorCode.INVALID_IMAGE,
            f"Image is too large: {image.width}x{image.height} "
            f"({pixels:,} px > {MAX_IMAGE_PIXELS:,} px)",
            {"width": image.width, "height": image.height, "max_pixels": MAX_IMAGE_PIXELS},
        )
    try:
        image = ImageOps.exif_transpose(image)        # keeps orientation as shot
    except Exception:                                 # noqa: BLE001 - orientation is best-effort
        pass
    if image.mode != "RGB":
        image = image.convert("RGB")                  # P/LA/CMYK/I16/16-bit -> RGB
    return image


def describe(image: Image.Image, *, source: str = "file",
             file_name: Optional[str] = None, digest: Optional[str] = None) -> ImageInfo:
    """ImageInfo for an in-memory image (``digest`` = hash of the original bytes)."""
    return ImageInfo(
        source=source, file_name=file_name,
        width=int(image.width), height=int(image.height),
        sha256=digest if digest is not None else sha256_bytes(to_png_bytes(image)),
    )


def _as_path(path: Any, *, what: str = "path") -> Path:
    """Coerce a caller-supplied path, or say precisely why it is not one (P8.4).

    A `None` or a `bytes` reaching `Path()` raises a bare `TypeError` out of `pathlib`, which
    the job runner would report as INTERNAL with a pathlib sentence in it. `load_image` is
    public API (`JobManager.start_file`, the CLI, `mcp load_image(path=...)`), so the wrong
    type is the caller's mistake and has to come back as INVALID_ARGUMENT.
    """
    if isinstance(path, Path):
        candidate = path
    elif isinstance(path, str):
        if not path.strip():
            raise StudioError(ErrorCode.INVALID_ARGUMENT,
                              f"{what} is empty.", {"hint": "pass an existing image file"})
        candidate = Path(path)
    else:
        fspath = getattr(path, "__fspath__", None)
        if callable(fspath):                      # os.PathLike of some other flavour
            return _as_path(fspath(), what=what)
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"{what} must be a string or a path, got {type(path).__name__}.",
            {"got": type(path).__name__},
        )
    return candidate


def load_image(path: Union[str, Path], *, source: str = "file") -> tuple[Image.Image, ImageInfo]:
    """Read a file into an RGB image + its ImageInfo (sha256 of the *file* bytes)."""
    path = _as_path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise StudioError(ErrorCode.IO_ERROR, f"Cannot read image: {path}",
                          {"reason": str(exc)}) from exc
    if not data:
        raise StudioError(ErrorCode.INVALID_IMAGE, f"Image file is empty: {path}")
    image = _open(data, str(path))
    return image, describe(image, source=source, file_name=path.name,
                           digest=sha256_bytes(data))


def image_from_bytes(data: bytes, *, file_name: Optional[str] = None,
                     source: str = "mcp-base64") -> tuple[Image.Image, ImageInfo]:
    """Decode in-memory bytes (clipboard, MCP payload, upload) into an RGB image."""
    if not isinstance(data, (bytes, bytearray)):
        raise StudioError(ErrorCode.INVALID_ARGUMENT,
                          f"Expected bytes, got {type(data).__name__}.")
    data = bytes(data)
    if not data:
        raise StudioError(ErrorCode.INVALID_IMAGE, "Image payload is empty.")
    image = _open(data, file_name or "<bytes>")
    return image, describe(image, source=source, file_name=file_name,
                           digest=sha256_bytes(data))


def image_from_base64(text: str, *, file_name: Optional[str] = None,
                      source: str = "mcp-base64") -> tuple[Image.Image, ImageInfo]:
    """Decode a base64 image (optionally a ``data:`` URL) — the MCP image input path.

    Whitespace *between* characters is ignored (P8.4): the `base64` command-line tool, most
    MIME encoders and several MCP clients wrap at 76 columns, and rejecting a perfectly good
    page because it has newlines in it is our bug, not the caller's. Anything that is not
    base64 alphabet at all is still rejected — `validate=True` keeps that guarantee.
    """
    import base64

    if not isinstance(text, str) or not text.strip():
        raise StudioError(ErrorCode.INVALID_ARGUMENT, "Expected a non-empty base64 string.")
    payload = text.strip()
    if payload.startswith("data:"):
        if "," not in payload:
            raise StudioError(
                ErrorCode.INVALID_ARGUMENT,
                "`data:` URL has no base64 payload after the comma.",
                {"hint": "expected data:image/png;base64,<payload>",
                 "got": payload[:48] + ("…" if len(payload) > 48 else "")},
            )
        payload = payload.split(",", 1)[1]
    payload = "".join(payload.split())          # drop newlines/spaces, keep the alphabet check
    if not payload:
        raise StudioError(ErrorCode.INVALID_ARGUMENT,
                          "Expected a non-empty base64 string (only whitespace was given).")
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:                          # noqa: BLE001
        raise StudioError(ErrorCode.INVALID_ARGUMENT, "Not valid base64 data.",
                          {"reason": f"{type(exc).__name__}: {exc}",
                           "chars": len(payload)}) from exc
    return image_from_bytes(raw, file_name=file_name, source=source)


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------

def crop_of(image: Image.Image, bbox: BBox, padding: int = 0) -> BBox:
    """The effective crop box for ``bbox``: padded by ``padding`` and clamped."""
    bbox = bbox.clamped(image.width, image.height)
    if padding:
        bbox = bbox.expanded(int(padding), image.width, image.height)
    return bbox


def crop(image: Image.Image, bbox: BBox, *, padding: int = 0,
         min_side: int = MIN_CROP_SIDE) -> Image.Image:
    """RGB crop of ``bbox`` (padded, clamped), upscaled so both sides >= ``min_side``.

    Never raises for a box that is merely small or partly outside the image — that is
    normal detector output. An *empty* box (no overlap) is an argument error, because a
    silently empty crop would be OCR'd into a hallucinated string.
    """
    if not isinstance(image, Image.Image):
        raise StudioError(ErrorCode.INVALID_IMAGE, "Expected a PIL.Image.Image.")
    box = crop_of(image, bbox, padding)
    if not box.is_valid():
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Crop box {box.as_tuple()} has no pixels inside the image "
            f"({image.width}x{image.height}).",
            {"bbox": box.to_dict(), "image": {"width": image.width, "height": image.height}},
        )
    region, effective = crop_pair(image, bbox, padding=padding, min_side=min_side)
    return region


def crop_pair(image: Image.Image, bbox: BBox, *, padding: int = 0,
              min_side: int = MIN_CROP_SIDE) -> tuple[Image.Image, BBox]:
    """Like :func:`crop`, but also returns the **effective** box.

    The effective box is what the pixels actually cover after padding and clamping, so
    a result document can record where the text really came from instead of the
    requested box (P4 records it, P6 uses it for the hover overlay on edge boxes).
    """
    if not isinstance(image, Image.Image):
        raise StudioError(ErrorCode.INVALID_IMAGE, "Expected a PIL.Image.Image.")
    box = crop_of(image, bbox, padding)
    if not box.is_valid():
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Crop box {bbox.as_tuple()} has no pixels inside the image "
            f"({image.width}x{image.height}).",
            {"bbox": box.to_dict(), "image": {"width": image.width, "height": image.height}},
        )
    region = image.crop(box.as_tuple())
    if region.mode != "RGB":
        region = region.convert("RGB")
    w, h = region.size
    if min_side and (w < min_side or h < min_side):
        scale = max(min_side / max(1, w), min_side / max(1, h))
        region = region.resize((max(min_side, int(round(w * scale))),
                                max(min_side, int(round(h * scale)))), Image.Resampling.LANCZOS)
    return region, box


def is_ocr_worthy(bbox: BBox, *, min_side: int = MIN_CROP_SIDE,
                  min_area: int = MIN_AREA_CROP) -> bool:
    """False for boxes too small to contain legible text (P4 skips OCR for them).

    The default matches the pipeline's own floor (``pipeline.min_box_area_px`` = 16); P4
    passes the configured value so exactly one threshold is ever in effect.
    """
    if not bbox.is_valid():
        return False
    if min_side and (bbox.width < min_side or bbox.height < min_side):
        return False
    return bbox.area >= int(min_area)


def normalize_text(text: Optional[str], *, nfkc: bool = True,
                   collapse_lines: bool = False) -> Optional[str]:
    """Optional NFKC normalisation + whitespace tidy (config ``ocr.normalize_nfkc``).

    NFKC folds full-width forms (ＡＢＣ → ABC, １２３ → 123) and compatibility
    punctuation. It is opt-in because it also rewrites some vertical-text and kana
    presentation forms the user may want preserved.

    Whitespace is tidied **per line** (spaces/tabs collapsed, lines stripped, blank lines
    dropped) but line breaks are kept: a multi-line bubble is a multi-line result and the
    GUI renders it that way. Pass ``collapse_lines=True`` to flatten to one line.
    """
    if text is None:
        return None
    out = unicodedata.normalize("NFKC", text) if nfkc else text
    out = out.replace("\u3000", " ").replace("\u00a0", " ")
    lines = [" ".join(line.split()) for line in out.splitlines()]
    lines = [line for line in lines if line]
    return " ".join(lines) if collapse_lines else "\n".join(lines)


# ---------------------------------------------------------------------------
# Small conversions
# ---------------------------------------------------------------------------

def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def thumbnail(image: Image.Image, max_side: int) -> Image.Image:
    """Proportional copy fitted into ``max_side`` (never upscales)."""
    if max_side <= 0:
        raise StudioError(ErrorCode.INVALID_ARGUMENT, "max_side must be > 0.")
    w, h = image.size
    if max(w, h) <= max_side:
        return image.copy()
    ratio = max_side / float(max(w, h))
    size = (max(1, int(round(w * ratio))), max(1, int(round(h * ratio))))
    return image.resize(size, Image.Resampling.LANCZOS)


_QT_IMPORT_HINT = "pip install animetext-studio[gui]  (PySide6)"


def pil_to_qimage(image: Image.Image):
    """PIL -> ``QImage`` (deep copy, no lifetime coupling). GUI only (P5/P6)."""
    if not isinstance(image, Image.Image):
        raise StudioError(ErrorCode.INVALID_IMAGE, "Expected a PIL.Image.Image.")
    try:
        from PySide6.QtGui import QImage
    except ImportError as exc:
        raise StudioError(ErrorCode.ENGINE_LOAD_FAILED,
                          "PySide6 is required for image display.",
                          {"hint": _QT_IMPORT_HINT, "reason": str(exc)}) from exc
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888)
    return qimage.copy()          # detach from `data`'s lifetime


#: The methods we actually call on a ``QImage``, in the order they are needed. Probed in
#: two cheap passes so a wrong object gets a message naming *every* missing method, while
#: an object that is a QImage but in the wrong format gets the *format* error (the caller
#: can fix that without reading our code).
_QIMAGE_HEAD = ("isNull", "format")
_QIMAGE_BUFFER = ("width", "height", "bytesPerLine", "constBits")
_QIMAGE_INTERFACE = _QIMAGE_HEAD + _QIMAGE_BUFFER


def _qformat_name(fmt: Any) -> str:
    """``QImage.Format`` enum -> readable name (``Format_Grayscale8``), stub-safe."""
    name = getattr(fmt, "name", None)
    if isinstance(name, str):               # real Qt enum: "Format.Format_Grayscale8"
        return name.rsplit(".", 1)[-1]
    return str(fmt)                          # tests stub the format with a plain string


def qimage_to_pil(qimage: Any) -> Image.Image:
    """``QImage`` (``Format_RGB888``) -> PIL RGB, without copying the buffer twice.

    Duck-typed on purpose (see :data:`_QIMAGE_INTERFACE`): a strict ``isinstance`` wall
    would make this untestable without Qt installed, and clipboard screenshots arrive as
    subclasses/other bindings' objects just as often.

    Only ``Format_RGB888`` is accepted. We deliberately do **not** call
    ``convertToFormat()`` behind the caller's back: the conversion that greyscale or
    premultiplied-ARGB input needs is a lossy, intent-dependent choice, so it belongs in
    the GUI layer that knows what it pasted — and silently guessing here would turn a
    "wire the format wrong" bug into garbled OCR input. Anything else is reported by name
    with the conversion the caller has to do.
    """
    try:
        from PySide6.QtGui import QImage
    except ImportError as exc:
        raise StudioError(ErrorCode.ENGINE_LOAD_FAILED,
                          "PySide6 is required for image input.",
                          {"hint": _QT_IMPORT_HINT, "reason": str(exc)}) from exc
    def _missing(names: tuple[str, ...]) -> list[str]:
        return [name for name in names if not callable(getattr(qimage, name, None))]

    if _missing(_QIMAGE_HEAD):
        absent = _missing(_QIMAGE_INTERFACE)
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            "Object does not look like a QImage (missing: " + ", ".join(absent) + ").",
            {"got": type(qimage).__name__, "missing": absent,
             "expected": list(_QIMAGE_INTERFACE)},
        )
    if qimage.isNull():
        raise StudioError(ErrorCode.INVALID_IMAGE, "The QImage is null.")
    fmt = qimage.format()
    rgb_format = QImage.Format.Format_RGB888
    if fmt != rgb_format:
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Unsupported QImage format: {_qformat_name(fmt)}.",
            {"got": type(qimage).__name__, "format": _qformat_name(fmt),
             "supported": [_qformat_name(rgb_format)],
             "hint": "pass a Format_RGB888 image: qimage.convertToFormat("
                     "QImage.Format.Format_RGB888)"},
        )
    if _missing(_QIMAGE_BUFFER):
        absent = _missing(_QIMAGE_BUFFER)
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            "QImage is missing the pixel-access methods (" + ", ".join(absent) + ").",
            {"got": type(qimage).__name__, "missing": absent},
        )
    img = qimage
    width, height = int(img.width()), int(img.height())
    bytes_per_line = int(img.bytesPerLine())
    buf = img.constBits()
    # constBits() is a view of length width*height*bpp; Qt pads lines to 4 bytes, so a
    # narrow image needs its padding sliced out before PIL sees it.
    raw = bytes(buf)[:bytes_per_line * height]
    if bytes_per_line != width * 3:
        raw = b"".join(raw[i * bytes_per_line:i * bytes_per_line + width * 3]
                       for i in range(height))
    return Image.frombytes("RGB", (width, height), raw)
