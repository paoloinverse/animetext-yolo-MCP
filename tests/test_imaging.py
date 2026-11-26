"""Tests for core.imaging (Part 3): IO, EXIF, crops, normalization, Qt seams.

Qt is optional here: the QImage helpers are tested against a stub module so the suite
stays green on a machine without PySide6 (and proves the call shape when it is there).
"""
from __future__ import annotations

import io
import struct
import sys
import types

import pytest
from PIL import Image

from animetext_studio.core import imaging as im
from animetext_studio.core.types import BBox, ErrorCode, StudioError


@pytest.fixture
def page():
    return Image.new("RGB", (200, 100), "white")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class TestLoad:
    def test_roundtrip_from_file(self, tmp_path):
        src = Image.new("RGB", (37, 21), (10, 20, 30))
        path = tmp_path / "a.png"
        src.save(path)
        image, info = im.load_image(path)
        assert image.size == (37, 21) and image.mode == "RGB"
        assert info.source == "file" and info.file_name == "a.png"
        assert (info.width, info.height) == (37, 21)
        assert info.sha256 == im.sha256_bytes(path.read_bytes())

    def test_missing_file_is_io_error(self, tmp_path):
        with pytest.raises(StudioError) as ei:
            im.load_image(tmp_path / "nope.png")
        assert ei.value.code is ErrorCode.IO_ERROR

    def test_empty_file_is_invalid_image(self, tmp_path):
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        with pytest.raises(StudioError) as ei:
            im.load_image(path)
        assert ei.value.code is ErrorCode.INVALID_IMAGE

    def test_garbage_is_invalid_image(self, tmp_path):
        path = tmp_path / "garbage.png"
        path.write_bytes(b"not an image at all")
        with pytest.raises(StudioError) as ei:
            im.load_image(path)
        assert ei.value.code is ErrorCode.INVALID_IMAGE
        assert "reason" in ei.value.details

    def test_palette_and_gray_and_rgba_become_rgb(self, tmp_path):
        for mode, fmt in (("L", "PNG"), ("RGBA", "PNG"), ("P", "PNG"), ("CMYK", "JPEG")):
            path = tmp_path / f"{mode}.img"
            Image.new(mode, (8, 6)).save(path, format=fmt)
            image, _ = im.load_image(path)
            assert image.mode == "RGB", mode

    def test_exif_orientation_is_applied(self, tmp_path):
        """A 20x10 image tagged 'rotate 90' must load as 10x20 (what the user sees)."""
        base = Image.new("RGB", (20, 10), "white")
        exif = base.getexif()
        exif[0x0112] = 6                      # Orientation = rotate 90 CW
        path = tmp_path / "rot.jpg"
        base.save(path, exif=exif)
        image, info = im.load_image(path)
        assert image.size == (10, 20)
        assert (info.width, info.height) == (10, 20)

    def test_from_bytes_records_source_and_digest(self):
        buffer = io.BytesIO()
        Image.new("RGB", (5, 4), "black").save(buffer, format="PNG")
        data = buffer.getvalue()
        image, info = im.image_from_bytes(data, source="mcp-base64", file_name="x.png")
        assert image.size == (5, 4)
        assert info.source == "mcp-base64" and info.file_name == "x.png"
        assert info.sha256 == im.sha256_bytes(data)

    def test_from_bytes_rejects_empty_and_garbage(self):
        for payload in (b"", bytearray()):
            with pytest.raises(StudioError) as ei:
                im.image_from_bytes(payload)
            assert ei.value.code is ErrorCode.INVALID_IMAGE
        with pytest.raises(StudioError) as ei:
            im.image_from_bytes(b"xxxx", source="mcp-path")
        assert ei.value.code is ErrorCode.INVALID_IMAGE
        assert ei.value.details["origin"]      # where the bytes came from, for the report

    def test_desompression_bomb_is_reported_not_crashing(self, monkeypatch):
        monkeypatch.setattr(im, "MAX_IMAGE_PIXELS", 100)
        with pytest.raises(StudioError) as ei:
            im.image_from_bytes(_png_of(Image.new("RGB", (400, 400))))
        assert ei.value.code is ErrorCode.INVALID_IMAGE
        assert "too large" in ei.value.message.lower()
        assert ei.value.details["max_pixels"] == 100


def _png_of(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_none_is_none(self):
        assert im.normalize_text(None) is None

    def test_nfkc_fullwidth_and_ligature(self):
        assert im.normalize_text("ＡＢﬁ") == "ABfi"

    def test_whitespace_is_tidied_per_line_and_breaks_are_kept(self):
        assert im.normalize_text("  a\u3000\n b\tc ") == "a\nb c"

    def test_blank_lines_are_dropped(self):
        assert im.normalize_text("one\n\n\ntwo\n") == "one\ntwo"

    def test_cjk_text_is_untouched_apart_from_spaces(self):
        assert im.normalize_text("その日、空は赤かった") == "その日、空は赤かった"

    def test_collapse_lines_flattens_when_asked(self):
        assert im.normalize_text("a\nb", collapse_lines=True) == "a b"

    def test_nfkc_can_be_disabled(self):
        assert im.normalize_text("Ａ", nfkc=False) == "Ａ"


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------

class TestCrop:
    def test_plain_crop(self, page):
        out = im.crop(page, BBox(10, 10, 30, 40))
        assert out.size == (20, 30) and out.mode == "RGB"

    def test_padding_is_clamped_to_the_image(self, page):
        out, box = im.crop_pair(page, BBox(0, 0, 10, 10), padding=5)
        assert box == BBox(0, 0, 15, 15)          # never negative, never past the edge
        assert out.size == (15, 15)

    def test_out_of_range_box_is_clamped_then_cropped(self, page):
        out, box = im.crop_pair(page, BBox(150, 80, 999, 999), padding=4)
        assert box == BBox(146, 76, 200, 100)
        assert out.size == (54, 24)

    def test_no_overlap_is_an_argument_error(self, page):
        with pytest.raises(StudioError) as ei:
            im.crop(page, BBox(500, 500, 600, 600))
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert ei.value.details["image"] == {"width": 200, "height": 100}

    def test_tiny_crop_is_upscaled_to_min_side(self, page):
        out = im.crop(page, BBox(3, 3, 5, 4), min_side=16)
        assert out.width >= 16 and out.height >= 16

    def test_min_side_zero_keeps_the_raw_pixels(self, page):
        out = im.crop(page, BBox(3, 3, 5, 4), min_side=0)
        assert out.size == (2, 1)

    def test_non_image_rejected(self):
        with pytest.raises(StudioError) as ei:
            im.crop("not-an-image", BBox(0, 0, 1, 1))
        assert ei.value.code is ErrorCode.INVALID_IMAGE

    def test_crop_of_reports_the_effective_box(self, page):
        assert im.crop_of(page, BBox(-5, -5, 300, 300), padding=0) == BBox(0, 0, 200, 100)

    def test_is_ocr_worthy(self):
        assert im.is_ocr_worthy(BBox(0, 0, 12, 12)) is True
        assert im.is_ocr_worthy(BBox(0, 0, 3, 40)) is False        # side below min
        assert im.is_ocr_worthy(BBox(0, 0, 8, 1), min_side=0) is False    # area below min
        assert im.is_ocr_worthy(BBox(0, 0, 8, 1), min_side=0, min_area=0) is True


# ---------------------------------------------------------------------------
# encodings + Qt seams
# ---------------------------------------------------------------------------

class TestEncodeAndQt:
    def test_to_png_bytes_is_a_png(self, page):
        assert im.to_png_bytes(page)[:8] == b"\x89PNG\r\n\x1a\n"

    def test_thumbnail_never_enlarges(self, page):
        assert im.thumbnail(page, 500).size == (200, 100)
        assert max(im.thumbnail(page, 50).size) == 50

    def test_missing_pyside6_gives_a_structured_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "PySide6", None)
        monkeypatch.setitem(sys.modules, "PySide6.QtGui", None)
        with pytest.raises(StudioError) as ei:
            im.pil_to_qimage(Image.new("RGB", (4, 4)))
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED
        assert "PySide6" in ei.value.details["hint"]

    def test_pil_to_qimage_call_shape(self, monkeypatch):
        """Stub PySide6: proves format, bytes and that no copy is silently dropped."""
        captured = {}

        class FakeImage:
            class Format:
                Format_RGB888 = "RGB888"

            def __init__(self, data, width, height, bytes_per_line, fmt):
                captured.update(width=width, height=height, bytes_per_line=bytes_per_line,
                                fmt=fmt, size=len(bytes(data)))

            def copy(self):
                return "qimage"          # the copy is what the caller must get back

        module = types.ModuleType("PySide6.QtGui")
        module.QImage = FakeImage
        package = types.ModuleType("PySide6")
        package.QtGui = module
        monkeypatch.setitem(sys.modules, "PySide6", package)
        monkeypatch.setitem(sys.modules, "PySide6.QtGui", module)

        assert im.pil_to_qimage(Image.new("RGB", (2, 2))) == "qimage"
        assert captured == {"width": 2, "height": 2, "bytes_per_line": 6,
                            "fmt": "RGB888", "size": 12}   # 2x2 RGB, tightly packed

    def _stub_qt(self, monkeypatch, fmt):
        module = types.ModuleType("PySide6.QtGui")

        class Format:
            Format_RGB888 = "RGB888"
            Format_Grayscale8 = "Grayscale8"

        module.QImage = types.SimpleNamespace(Format=Format)
        package = types.ModuleType("PySide6")
        package.QtGui = module
        monkeypatch.setitem(sys.modules, "PySide6", package)
        monkeypatch.setitem(sys.modules, "PySide6.QtGui", module)
        return fmt

    def test_qimage_to_pil_roundtrip_with_the_stub(self, monkeypatch):
        self._stub_qt(monkeypatch, "RGB888")
        raw = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        class FakeImage:
            def __init__(self, data):
                self._data = data

            def format(self):
                return "RGB888"

            def isNull(self):
                return False

            def convertToFormat(self, fmt):
                return self

            def constBits(self):
                return self._data

            def width(self):
                return 2

            def height(self):
                return 2

            def bytesPerLine(self):
                return 6

        image = im.qimage_to_pil(FakeImage(raw))
        assert isinstance(image, Image.Image) and image.size == (2, 2)
        assert list(image.getdata())[0] == (1, 2, 3)

    def test_qimage_without_the_right_format_is_rejected(self, monkeypatch):
        self._stub_qt(monkeypatch, "Grayscale8")

        class WrongFormat:
            def format(self):
                return "Grayscale8"

            def isNull(self):
                return False

            def convertToFormat(self, fmt):
                raise AssertionError("conversion must be reported, not guessed")

            def width(self):
                return 1

            def height(self):
                return 1

        with pytest.raises(StudioError) as ei:
            im.qimage_to_pil(WrongFormat())
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert "Grayscale8" in str(ei.value.details)

    def test_a_non_qimage_object_says_what_is_missing(self, monkeypatch):
        self._stub_qt(monkeypatch, "RGB888")
        with pytest.raises(StudioError) as ei:
            im.qimage_to_pil(object())
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert "width" in ei.value.message
