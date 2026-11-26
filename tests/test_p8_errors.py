"""P8.4 - the error-path matrix.

One table, every way this app can be told to fail, and what it answers with. The rows are the
entry points (imaging, the pipeline, `JobManager`, `StudioService`, the MCP wire, the CLI); the
columns are the fault classes the plan named: corrupt / empty / huge input, cancel storms,
config-import garbage, malformed MCP arguments, offline and no-token.

Three things are asserted for every cell, and the third is the one that matters most:

1. **shape** - a `StudioError` with the expected `ErrorCode`, never a bare `TypeError`,
   `ValueError` or `KeyError`. A bare exception becomes `INTERNAL` at the job boundary, which
   tells the caller nothing and hides a defect in someone else's traceback.
2. **state** - after the failure the app still answers `get_status`, still accepts a good
   request, and still has the config it had before. An error that leaves the app wedged is two
   bugs, not one.
3. **no leak** - nothing a client sends (a token, a base64 page) reaches a log line, an error
   message or an exported document on the way out.

Everything here runs on the fake engines with no network, no weights and no display. The
`-m models` suite (`tests/test_models.py`) is where real weights are involved; this file is
about the contract, and the contract has to hold with a stub too.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the MCP wire cases need the SDK (pip install 'animetext-studio[mcp]')")

from animetext_studio.core import imaging                                        # noqa: E402
from animetext_studio.core.config import AppConfig                                # noqa: E402
from animetext_studio.core.jobs import JobManager                                 # noqa: E402
from animetext_studio.core.pipeline import Pipeline                               # noqa: E402
from animetext_studio.core.types import (                                         # noqa: E402
    EXIT_CODES, ErrorCode, JobState, StudioError,
)
from animetext_studio.engines.base import (                                       # noqa: E402
    BaseOcrEngine, EngineState, FakeDetectionEngine, FakeOcrEngine,
)

from conftest import SCHEMA_DIR                                                   # noqa: E402
from jsonschema import Draft202012Validator                                       # noqa: E402

from animetext_studio.core.model_store import MODELS_DIR_ENV                      # noqa: E402
from animetext_studio.mcp import tools as mcp_tools                               # noqa: E402
from animetext_studio.mcp.service import StudioService                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = str(ROOT / "examples" / "sample_page.png")
DOC = ROOT / "docs" / "ERROR-MATRIX.md"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Small builders
# ---------------------------------------------------------------------------

def png_bytes(w: int = 64, h: int = 48) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def wrapped_b64(data: bytes, width: int = 64) -> str:
    """base64 with MIME line breaks, the way `base64(1)` and many clients emit it."""
    text = b64(data)
    return "\n".join(text[i:i + width] for i in range(0, len(text), width)) + "\n"


def write(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def service_with_fakes(**kwargs) -> StudioService:
    return StudioService(JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine())),
                         AppConfig.default(), mock=True, **kwargs)


def load_page(svc: StudioService) -> StudioService:
    svc.load_image(path=SAMPLE)
    return svc


def fresh_service() -> StudioService:
    """A headless service on the mock engines — the thing every door ultimately drives."""
    return StudioService(JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(),
                                           AppConfig.default())),
                         AppConfig.default(), mock=True)


def run_page(svc: StudioService):
    svc.load_image(path=SAMPLE)
    svc.start_job()
    svc.wait_for_job()
    return svc.manager.result


# ---------------------------------------------------------------------------
# The matrix itself
#
# (entry point, fault, expected ErrorCode). `docs/ERROR-MATRIX.md` carries the same rows in
# prose; `TestMatrixInvariants::test_every_row_of_the_matrix_is_documented` fails if the two
# drift, because a matrix nobody can read is a matrix nobody consults.
# ---------------------------------------------------------------------------

MATRIX: list[tuple[str, str, ErrorCode]] = [
    # --- imaging: the input door -------------------------------------------
    ("imaging", "missing file", ErrorCode.IO_ERROR),
    ("imaging", "path is a directory", ErrorCode.IO_ERROR),
    ("imaging", "path is empty", ErrorCode.INVALID_ARGUMENT),
    ("imaging", "path is not a path", ErrorCode.INVALID_ARGUMENT),
    ("imaging", "file is unreadable", ErrorCode.IO_ERROR),
    ("imaging", "file is empty", ErrorCode.INVALID_IMAGE),
    ("imaging", "file is not an image", ErrorCode.INVALID_IMAGE),
    ("imaging", "png truncated", ErrorCode.INVALID_IMAGE),
    ("imaging", "payload is empty bytes", ErrorCode.INVALID_IMAGE),
    ("imaging", "payload is a str, not bytes", ErrorCode.INVALID_ARGUMENT),
    ("imaging", "base64 is empty", ErrorCode.INVALID_ARGUMENT),
    ("imaging", "base64 is not base64", ErrorCode.INVALID_ARGUMENT),
    ("imaging", "base64 decodes to garbage", ErrorCode.INVALID_IMAGE),
    ("imaging", "data: URL without a payload", ErrorCode.INVALID_ARGUMENT),
    ("imaging", "decoder says decompression bomb", ErrorCode.INVALID_IMAGE),
    ("imaging", "over MAX_IMAGE_PIXELS", ErrorCode.INVALID_IMAGE),
    ("imaging", "crop outside the image", ErrorCode.INVALID_ARGUMENT),
    # --- pipeline: engine contract -----------------------------------------
    ("pipeline", "run without an ImageInfo", ErrorCode.INVALID_ARGUMENT),
    ("pipeline", "run with a non-image", ErrorCode.INVALID_IMAGE),
    ("pipeline", "engine returns None", ErrorCode.INTERNAL),
    ("pipeline", "engine returns dicts", ErrorCode.INTERNAL),
    ("pipeline", "ocr returns the wrong count", ErrorCode.INTERNAL),
    ("pipeline", "ocr returns a non-string", ErrorCode.INTERNAL),
    ("pipeline", "engine raises MODEL_NOT_FOUND", ErrorCode.MODEL_NOT_FOUND),
    ("pipeline", "engine raises HF_AUTH_REQUIRED", ErrorCode.HF_AUTH_REQUIRED),
    ("pipeline", "engine raises DOWNLOAD_FAILED", ErrorCode.DOWNLOAD_FAILED),
    # Not a per-block note: with one crop there is nothing left to isolate (D37), so the
    # engine's own code becomes the run's code - and `run_pipeline` answers with it.
    ("pipeline", "ocr fails on the only crop", ErrorCode.OCR_FAILED),
    # --- jobs: the single-flight machine -----------------------------------
    ("jobs", "second job while running", ErrorCode.BUSY),
    ("jobs", "reset while running", ErrorCode.BUSY),
    ("jobs", "cancel a running job", ErrorCode.CANCELLED),
    ("jobs", "cancel mid-warmup", ErrorCode.CANCELLED),
    ("jobs", "worker raises a bare exception", ErrorCode.INTERNAL),
    ("jobs", "clear_result while running", ErrorCode.BUSY),
    # --- config: import garbage --------------------------------------------
    ("config", "document is a list", ErrorCode.INVALID_CONFIG),
    ("config", "document is a string", ErrorCode.INVALID_CONFIG),
    ("config", "section is a list", ErrorCode.INVALID_CONFIG),
    ("config", "value has the wrong type", ErrorCode.INVALID_CONFIG),
    ("config", "value out of range", ErrorCode.INVALID_CONFIG),
    ("config", "unknown schema_version", ErrorCode.INVALID_CONFIG),
    ("config", "file is not JSON", ErrorCode.INVALID_CONFIG),
    ("config", "file is not UTF-8", ErrorCode.INVALID_CONFIG),
    ("config", "file does not exist", ErrorCode.IO_ERROR),
    ("config", "path is a directory", ErrorCode.IO_ERROR),
    ("config", "both json and path given", ErrorCode.INVALID_ARGUMENT),
    ("config", "neither json nor path given", ErrorCode.INVALID_ARGUMENT),
    # --- service: arguments and missing state ------------------------------
    ("service", "no image loaded", ErrorCode.NO_IMAGE),
    ("service", "no result yet", ErrorCode.NO_IMAGE),
    ("service", "two image sources", ErrorCode.INVALID_ARGUMENT),
    ("service", "empty image source", ErrorCode.INVALID_ARGUMENT),
    ("service", "from_clipboard without a GUI", ErrorCode.INVALID_ARGUMENT),
    ("service", "block_ids is not a list", ErrorCode.INVALID_ARGUMENT),
    ("service", "block_ids has a non-integer", ErrorCode.INVALID_ARGUMENT),
    ("service", "no block id matches", ErrorCode.INVALID_ARGUMENT),
    ("service", "highlight an unknown block", ErrorCode.INVALID_ARGUMENT),
    ("service", "save_result to an empty path", ErrorCode.INVALID_ARGUMENT),
    ("service", "save_result into an unwritable path", ErrorCode.IO_ERROR),
    ("service", "annotated scale out of range", ErrorCode.INVALID_ARGUMENT),
    ("service", "annotated scale would render too much", ErrorCode.INVALID_ARGUMENT),
    ("service", "overlay with nothing to set", ErrorCode.INVALID_ARGUMENT),
    ("service", "zoom mode unknown", ErrorCode.INVALID_ARGUMENT),
    ("service", "zoom percent out of range", ErrorCode.INVALID_ARGUMENT),
    ("service", "base64 over the size cap", ErrorCode.INVALID_ARGUMENT),
    # --- service: the re-OCR door (P9.3) ----------------------------------
    # A page and a document are two different preconditions, and the answers differ: no page is
    # `NO_IMAGE` (nothing to look at), a page with no document is `INVALID_ARGUMENT` (there is
    # nothing to re-read *of*). `get_result` answers `NO_IMAGE` to the second; a re-OCR asks for
    # a document, so it says so (D81).
    ("service", "re-OCR with an empty id list", ErrorCode.INVALID_ARGUMENT),
    ("service", "re-OCR an id that is not in the document", ErrorCode.INVALID_ARGUMENT),
    ("service", "re-OCR with no page loaded", ErrorCode.NO_IMAGE),
    ("service", "re-OCR a page that has no document yet", ErrorCode.INVALID_ARGUMENT),
    # --- mcp: what a client can actually cause -----------------------------
    ("mcp", "unknown tool", ErrorCode.INVALID_ARGUMENT),
    ("mcp", "missing required argument", ErrorCode.INVALID_ARGUMENT),
    ("mcp", "argument of the wrong type", ErrorCode.INVALID_ARGUMENT),
    ("mcp", "unknown argument name", ErrorCode.INVALID_ARGUMENT),
    ("mcp", "argument out of range", ErrorCode.INVALID_ARGUMENT),
    ("mcp", "config value rejected", ErrorCode.INVALID_CONFIG),
    ("mcp", "no image loaded", ErrorCode.NO_IMAGE),
    ("mcp", "job cancelled by another client", ErrorCode.CANCELLED),
    ("mcp", "second run while busy", ErrorCode.BUSY),
    ("mcp", "unknown model", ErrorCode.MODEL_NOT_FOUND),
    ("mcp", "resource read with no state", ErrorCode.NO_IMAGE),
    ("mcp", "unknown resource", ErrorCode.INVALID_ARGUMENT),
    # --- offline / no token -------------------------------------------------
    ("offline", "offline and the bundle is missing", ErrorCode.MODEL_NOT_FOUND),
    ("offline", "gated repo, no token", ErrorCode.HF_AUTH_REQUIRED),
    ("offline", "repo not found", ErrorCode.MODEL_NOT_FOUND),
    ("offline", "network unreachable", ErrorCode.DOWNLOAD_FAILED),
    ("offline", "backend not installed", ErrorCode.ENGINE_LOAD_FAILED),
    # Present and unusable is not absent (P9.1): the file is there, it is the wrong bytes, and
    # only the manifest knows. `MODEL_NOT_FOUND` here would send the user to install a bundle
    # they already have.
    ("offline", "bundled weights are corrupt", ErrorCode.ENGINE_LOAD_FAILED),
    # --- cli: exit codes ----------------------------------------------------
    ("cli", "--run without --image", ErrorCode.NO_IMAGE),
    ("cli", "--run on a corrupt image", ErrorCode.INVALID_IMAGE),
    ("cli", "--config pointing at garbage", ErrorCode.INVALID_CONFIG),
    ("cli", "--out into an unwritable path", ErrorCode.IO_ERROR),
    ("cli", "--headless with no action", ErrorCode.INVALID_ARGUMENT),
    # An option that chooses an action, given `""`, is a mistake about this command — not a
    # request to run the default one. These used to open the window (D93).
    ("cli", "--warm-devices with no devices", ErrorCode.INVALID_ARGUMENT),
    ("cli", "--export-config with an empty path", ErrorCode.INVALID_ARGUMENT),
]

CODE_OF = {(entry, fault): code for entry, fault, code in MATRIX}


def test_matrix_covers_every_error_code():
    """No code in the taxonomy may be untested (that is how a code stops meaning anything)."""
    #: in the taxonomy (and in the catalog) but produced by no code path since Part 7
    #: shipped — `--headless` with no action used to be it. Kept so old documents parse.
    unreachable = {ErrorCode.NOT_IMPLEMENTED}
    covered = {code for _entry, _fault, code in MATRIX}
    assert covered | unreachable == set(ErrorCode), (
        f"untested: {sorted(c.value for c in set(ErrorCode) - covered - unreachable)}; "
        f"unknown: {sorted(c.value for c in covered - set(ErrorCode))}")


def test_matrix_rows_are_unique():
    keys = [(entry, fault) for entry, fault, _code in MATRIX]
    assert len(keys) == len(set(keys)), "duplicate matrix row"


# ---------------------------------------------------------------------------
# Entry point 1 - imaging (the input door)
# ---------------------------------------------------------------------------

def _unreadable_file(tmp):
    path = write(tmp, "locked.png", png_bytes())
    os.chmod(path, 0o000)
    try:
        return imaging.load_image(path)
    finally:
        os.chmod(path, 0o644)


IMAGING_CASES: dict[str, callable] = {
    "missing file": lambda tmp: imaging.load_image(str(tmp / "nope.png")),
    "path is a directory": lambda tmp: imaging.load_image(str(tmp)),
    "path is empty": lambda tmp: imaging.load_image("   "),
    "path is not a path": lambda tmp: imaging.load_image(None),
    "file is unreadable": _unreadable_file,
    "file is empty": lambda tmp: imaging.load_image(write(tmp, "empty.png", b"")),
    "file is not an image": lambda tmp: imaging.load_image(
        write(tmp, "notes.png", b"not an image at all")),
    "png truncated": lambda tmp: imaging.load_image(
        write(tmp, "cut.png", png_bytes()[:24])),
    "payload is empty bytes": lambda tmp: imaging.image_from_bytes(b""),
    "payload is a str, not bytes": lambda tmp: imaging.image_from_bytes("i am not bytes"),
    "base64 is empty": lambda tmp: imaging.image_from_base64("   "),
    "base64 is not base64": lambda tmp: imaging.image_from_base64("!!!!not base64!!!!"),
    "base64 decodes to garbage": lambda tmp: imaging.image_from_base64(b64(b"hello world")),
    "data: URL without a payload": lambda tmp: imaging.image_from_base64(
        "data:image/png;base64"),
    "decoder says decompression bomb": lambda tmp: _with_pillow_limit(tmp),
    "over MAX_IMAGE_PIXELS": lambda tmp: _over_our_limit(tmp),
    "crop outside the image": lambda tmp: imaging.crop(_tiny_image(), imaging.BBox(500, 500, 600, 600)),
}


def _tiny_image():
    from PIL import Image

    return Image.new("RGB", (16, 16), "white")


def _with_pillow_limit(tmp):
    """Pillow's own bomb check firing on a small file (the limit is lowered, not the file)."""
    from PIL import Image

    saved = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 10
    try:
        return imaging.image_from_bytes(png_bytes(64, 48))
    finally:
        Image.MAX_IMAGE_PIXELS = saved


def _over_our_limit(tmp):
    """Our own ceiling (`MAX_IMAGE_PIXELS`) - same trick, our number, our error."""
    saved = imaging.MAX_IMAGE_PIXELS
    imaging.MAX_IMAGE_PIXELS = 100
    try:
        return imaging.image_from_bytes(png_bytes(64, 48))
    finally:
        imaging.MAX_IMAGE_PIXELS = saved


@pytest.mark.parametrize("fault", sorted(IMAGING_CASES))
def test_imaging_faults_answer_with_the_matrix_code(tmp_path, fault):
    expected = CODE_OF[("imaging", fault)]
    with pytest.raises(StudioError) as caught:
        IMAGING_CASES[fault](tmp_path)
    error = caught.value
    assert error.code is expected, f"{fault}: {error.code.value} - {error.message}"
    assert error.message.strip(), "an error must say something"
    assert error.exit_code == EXIT_CODES[expected]


def test_imaging_matrix_rows_all_have_a_case():
    documented = {fault for entry, fault, _ in MATRIX if entry == "imaging"}
    assert documented == set(IMAGING_CASES)


def test_a_huge_but_legal_page_is_not_an_error(tmp_path):
    """The ceiling is a ceiling, not a target: a big legal image must still load."""
    from PIL import Image

    path = tmp_path / "big.png"
    Image.new("RGB", (4000, 6000), "white").save(path, "PNG")
    image, info = imaging.load_image(path)
    assert (info.width, info.height) == (4000, 6000)
    os.remove(path)


def test_mime_wrapped_base64_is_accepted(tmp_path):
    """Newlines between base64 characters are the caller's encoder, not their mistake (P8.4)."""
    image, info = imaging.image_from_base64(wrapped_b64(png_bytes()))
    assert (info.width, info.height) == (64, 48)


def test_the_app_survives_every_imaging_fault(tmp_path):
    """Column 3 of the matrix: after all that, a good request still works."""
    for fault, case in IMAGING_CASES.items():
        try:
            case(tmp_path)
        except StudioError:
            pass
        except Exception as exc:                                  # noqa: BLE001
            pytest.fail(f"{fault} raised {type(exc).__name__}, not a StudioError: {exc}")
    image, info = imaging.load_image(SAMPLE)
    assert info.width > 0


# ---------------------------------------------------------------------------
# Entry point 2 - the pipeline (the engine contract)
# ---------------------------------------------------------------------------

class _BrokenDetection(FakeDetectionEngine):
    def __init__(self, value=None, exc=None):
        super().__init__()
        self.value, self.exc = value, exc

    def detect(self, image, model_name, **kwargs):
        if self.exc is not None:
            raise self.exc
        return self.value


class _BrokenOcr(FakeOcrEngine):
    def __init__(self, value=None, exc=None):
        super().__init__()
        self.value, self.exc = value, exc

    def ocr_crops(self, images, **kwargs):
        if self.exc is not None:
            raise self.exc
        return self.value(images) if callable(self.value) else self.value


PIPELINE_CASES: dict[str, callable] = {
    "run without an ImageInfo": lambda tmp: Pipeline(
        FakeDetectionEngine(), FakeOcrEngine()).run(_tiny_image()),
    "run with a non-image": lambda tmp: Pipeline(
        FakeDetectionEngine(), FakeOcrEngine()).run("not an image"),
    "engine returns None": lambda tmp: Pipeline(
        _BrokenDetection(value=None), FakeOcrEngine()).run_file(SAMPLE),
    "engine returns dicts": lambda tmp: Pipeline(
        _BrokenDetection(value=[{"bbox": [1, 2, 3, 4], "score": 0.5}]),
        FakeOcrEngine()).run_file(SAMPLE),
    "ocr returns the wrong count": lambda tmp: Pipeline(
        FakeDetectionEngine(), _BrokenOcr(value=["only one"])).run_file(SAMPLE),
    "ocr returns a non-string": lambda tmp: Pipeline(
        FakeDetectionEngine(), _BrokenOcr(value=lambda imgs: [3 for _ in imgs])).run_file(SAMPLE),
    "engine raises MODEL_NOT_FOUND": lambda tmp: Pipeline(
        _BrokenDetection(exc=StudioError(ErrorCode.MODEL_NOT_FOUND, "no such model")),
        FakeOcrEngine()).run_file(SAMPLE),
    "engine raises HF_AUTH_REQUIRED": lambda tmp: Pipeline(
        _BrokenDetection(exc=StudioError(ErrorCode.HF_AUTH_REQUIRED, "repo is gated")),
        FakeOcrEngine()).run_file(SAMPLE),
    "engine raises DOWNLOAD_FAILED": lambda tmp: Pipeline(
        _BrokenDetection(exc=StudioError(ErrorCode.DOWNLOAD_FAILED, "no network")),
        FakeOcrEngine()).run_file(SAMPLE),
    "ocr fails on the only crop": lambda tmp: Pipeline(
        FakeDetectionEngine(),
        _BrokenOcr(exc=StudioError(ErrorCode.OCR_FAILED, "CUDA out of memory")),
        _one_box_only()).run_file(SAMPLE),
}


def _one_box_only() -> AppConfig:
    """A config that keeps exactly one detection, so the per-crop fallback cannot apply.

    `Pipeline._recognise` retries crop by crop when a *batch* fails (D37) — with one crop there
    is nothing to isolate, and the engine's own code is what the run reports.
    """
    config = AppConfig.default()
    config.pipeline.min_box_area_px = 60_000        # one box of the fake layout survives
    return config


@pytest.mark.parametrize("fault", sorted(PIPELINE_CASES))
def test_pipeline_faults_answer_with_the_matrix_code(fault):
    expected = CODE_OF[("pipeline", fault)]
    with pytest.raises(StudioError) as caught:
        PIPELINE_CASES[fault](None)
    assert caught.value.code is expected, f"{fault}: {caught.value.to_dict()}"


def test_pipeline_matrix_rows_all_have_a_case():
    documented = {fault for entry, fault, _ in MATRIX if entry == "pipeline"}
    assert documented == set(PIPELINE_CASES)


def test_a_contract_violation_names_the_engine_not_a_stack_frame():
    """The point of the guard: the message has to be actionable without reading the code."""
    with pytest.raises(StudioError) as caught:
        Pipeline(_BrokenDetection(value=None), FakeOcrEngine()).run_file(SAMPLE)
    assert "detection" in caught.value.message
    assert caught.value.details["contract"] is True


def test_a_short_ocr_answer_fails_loudly_instead_of_answering_short():
    """D39, enforced at the door: `text: null` must always carry a reason.

    A batch answer with the wrong number of strings used to be `zip`ped into the crops, so the
    crops past the end came out `text: null` with no `error` - a document that looks like "the
    page was empty there" when the truth is "the engine lost track". It is now a contract
    violation: one clear INTERNAL, and no half-answered document on disk. Retrying crop by crop
    would only run the same broken engine N times, so it is deliberately not retried.
    """
    class PartialOcr(FakeOcrEngine):
        def ocr_crops(self, images, **kwargs):
            return ["read"]                     # one answer, however many crops

    with pytest.raises(StudioError) as caught:
        Pipeline(FakeDetectionEngine(), PartialOcr()).run_file(SAMPLE)
    assert caught.value.code is ErrorCode.INTERNAL
    assert caught.value.details["contract"] is True
    assert "1 text(s) for" in caught.value.message


def test_a_none_answer_becomes_a_per_block_error():
    class NoneOcr(FakeOcrEngine):
        def ocr_crops(self, images, **kwargs):
            return [None for _ in images]

    result = Pipeline(FakeDetectionEngine(), NoneOcr()).run_file(SAMPLE)
    assert result.block_count >= 1
    for block in result.blocks:
        assert block.text is None and block.error
        assert block.error["code"] == ErrorCode.OCR_FAILED.value


def test_a_poisoned_crop_loses_only_itself():
    """D37, from the other side: one crop the engine chokes on must not lose the page."""
    class PoisonOcr(FakeOcrEngine):
        def ocr_crops(self, images, **kwargs):
            if len(images) > 1:
                raise StudioError(ErrorCode.OCR_FAILED, "CUDA out of memory")
            if images[0].width < 140:           # the narrow column crops (130 px wide)
                raise StudioError(ErrorCode.OCR_FAILED, "bad tensor")
            return ["ok"]

    result = Pipeline(FakeDetectionEngine(), PoisonOcr()).run_file(SAMPLE)
    read = [b for b in result.blocks if b.text]
    failed = [b for b in result.blocks if b.error and b.error["code"] == "OCR_FAILED"]
    assert read, "the whole page was lost to one crop"
    assert failed, "the poisoned crop is not recorded where it happened"


# ---------------------------------------------------------------------------
# Entry point 3 - JobManager (the single-flight machine)
# ---------------------------------------------------------------------------

class _HoldDetection(FakeDetectionEngine):
    """Blocks inside `detect` so a test can be certain it is racing a *running* job."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def detect(self, image, model_name, **kwargs):
        self.entered.set()
        self.release.wait(20)
        return super().detect(image, model_name, **kwargs)


def _held_manager():
    engine = _HoldDetection()
    manager = JobManager(Pipeline(engine, FakeOcrEngine(), AppConfig.default()))
    manager.start_file(SAMPLE)
    assert engine.entered.wait(10), "the job never reached detect"
    return manager, engine


#: every case takes `tmp_path` (mostly ignored) so the sweeps can call them uniformly
JOBS_CASES: dict[str, callable] = {
    "second job while running": lambda _tmp=None: _busy_second_start(),
    "reset while running": lambda _tmp=None: _reset_while_running(),
    "cancel a running job": lambda _tmp=None: _cancel_once(),
    "cancel mid-warmup": lambda _tmp=None: _cancel_warmup(),
    "worker raises a bare exception": lambda _tmp=None: _bare_worker_crash(),
    "clear_result while running": lambda _tmp=None: _clear_while_running(),
}


def _busy_second_start():
    manager, engine = _held_manager()
    try:
        manager.start_file(SAMPLE)
        raise AssertionError("a second job was allowed")
    finally:
        engine.release.set()
        manager.wait(10)


def _reset_while_running():
    manager, engine = _held_manager()
    try:
        manager.clear_result()
        raise AssertionError("clear_result was allowed during a job")
    finally:
        engine.release.set()
        manager.wait(10)


def _clear_while_running():
    manager, engine = _held_manager()
    try:
        service = StudioService(manager, AppConfig.default(), mock=True)
        service.reset()
        raise AssertionError("reset was allowed during a job")
    finally:
        engine.release.set()
        manager.wait(10)


def _cancel_once():
    manager, engine = _held_manager()
    assert manager.cancel() is True
    engine.release.set()
    manager.wait(10)
    assert manager.state is JobState.CANCELLED
    raise StudioError(ErrorCode.CANCELLED, "Job cancelled.")     # what `_await` would raise


def _cancel_warmup():
    hold = {"in": threading.Event(), "off": threading.Event()}

    class SlowLoad(FakeDetectionEngine):
        def load(self, model_name, progress=None):
            hold["in"].set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not hold["off"].is_set():
                time.sleep(0.005)
            if hold["off"].is_set():
                raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")
            return super().load(model_name, progress=progress)

    engine = SlowLoad()
    manager = JobManager(Pipeline(engine, FakeOcrEngine(), AppConfig.default()))
    manager.start_warmup()
    assert hold["in"].wait(10), "warm-up never reached the load"
    assert manager.cancel() is True
    hold["off"].set()
    manager.wait(10)
    assert manager.state is JobState.CANCELLED
    assert manager.kind == "warmup"
    raise StudioError(ErrorCode.CANCELLED, "Job cancelled.")


def _bare_worker_crash():
    manager = JobManager(Pipeline(_BrokenDetection(exc=RuntimeError("onnx exploded")),
                                  FakeOcrEngine(), AppConfig.default()))
    manager.start_file(SAMPLE)
    manager.wait(10)
    error = manager.error
    assert error is not None and error.code is ErrorCode.INTERNAL
    raise error


@pytest.mark.parametrize("fault", sorted(JOBS_CASES))
def test_job_faults_answer_with_the_matrix_code(fault):
    expected = CODE_OF[("jobs", fault)]
    with pytest.raises(StudioError) as caught:
        JOBS_CASES[fault]()
    assert caught.value.code is expected, f"{fault}: {caught.value.to_dict()}"


def test_jobs_matrix_rows_all_have_a_case():
    documented = {fault for entry, fault, _ in MATRIX if entry == "jobs"}
    assert documented == set(JOBS_CASES)


def test_a_cancel_storm_leaves_one_settled_job():
    """25 cancels, one job: idempotent, and the machine ends in exactly one state."""
    manager, engine = _held_manager()
    answers = [manager.cancel() for _ in range(25)]
    engine.release.set()
    manager.wait(10)
    assert answers[0] is True
    assert set(answers) == {True}, "a cancel during CANCELLING should still report True"
    assert manager.state is JobState.CANCELLED
    assert manager.status()["busy"] is False


def test_cancel_then_restart_is_busy_until_the_worker_settles():
    manager, engine = _held_manager()
    manager.cancel()
    with pytest.raises(StudioError) as caught:      # CANCELLING is still "occupied" (D14)
        manager.start_file(SAMPLE)
    assert caught.value.code is ErrorCode.BUSY
    engine.release.set()
    manager.wait(10)
    manager.start_file(SAMPLE)                      # and now it works
    manager.wait(10)
    assert manager.state is JobState.DONE


def test_cancel_after_the_job_settled_is_not_an_error():
    manager = JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), AppConfig.default()))
    manager.start_file(SAMPLE)
    manager.wait(10)
    assert manager.cancel() is False                # nothing to cancel: a plain answer
    assert manager.state is JobState.DONE


def test_a_broken_progress_listener_cannot_kill_a_job():
    manager = JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), AppConfig.default()))

    def explode(_progress):
        raise RuntimeError("the GUI slot is broken")

    manager.add_listener(explode)
    manager.start_file(SAMPLE)
    manager.wait(10)
    assert manager.state is JobState.DONE and manager.result is not None


def test_the_manager_is_still_usable_after_every_fault():
    for fault, case in JOBS_CASES.items():
        try:
            case()
        except StudioError:
            continue
        except Exception as exc:                    # noqa: BLE001
            pytest.fail(f"{fault}: {type(exc).__name__}: {exc}")
    manager = JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), AppConfig.default()))
    manager.start_file(SAMPLE)
    manager.wait(10)
    assert manager.state is JobState.DONE


# ---------------------------------------------------------------------------
# Entry point 4 - configuration (the document door)
# ---------------------------------------------------------------------------

#: The same garbage goes in through `set_config` (a patch) and through `import_config` (a
#: whole document). Both must answer the same way, so the table is written once.
GARBAGE: dict[str, str] = {
    "section is a list": '{"detection": []}',
    "section is a string": '{"ocr": "hello"}',
    "threshold is a word": '{"detection": {"score_threshold": "high"}}',
    "threshold out of range": '{"detection": {"iou_threshold": 42}}',
    "batch_size is zero": '{"ocr": {"batch_size": 0}}',
    "max_patches is not a tier": '{"ocr": {"max_patches": 100}}',
    "device is nonsense": '{"ocr": {"device": "tpu"}}',
    "port out of range": '{"mcp": {"port": 99999}}',
    "transport is unknown": '{"mcp": {"transport": "carrier-pigeon"}}',
    "schema_version from the future": '{"schema_version": "9.9"}',
    "schema_version is null": '{"schema_version": null}',
    "boolean where a number belongs": '{"ocr": {"batch_size": true}}',
    "deeply nested garbage": '{"ocr": {"device": {"nested": [1, 2, {"x": null}]}}}',
    "model is a number": '{"detection": {"model": 12}}',
    "log_capacity below its floor": '{"mcp": {"log_capacity": 2}}',
}

#: Accepted, but only after a warning: unknown keys are the caller's business (D10).
TOLERATED: dict[str, str] = {
    "unknown top-level key": '{"typo_section": {"a": 1}}',
    "unknown key in a known section": '{"ocr": {"teleport": true}}',
    "unknown model name": '{"detection": {"model": "yolo99z_bogus"}}',
    "key with a dot in it": '{"a.b": 1}',
}


def _manager() -> JobManager:
    return JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), AppConfig.default()))


def _patch(document):
    return StudioService(_manager(), AppConfig.default(), mock=True).apply_config_patch(document)


def _replace(document):
    return StudioService(_manager(), AppConfig.default(), mock=True).import_config(document)


@pytest.mark.parametrize("label", sorted(GARBAGE))
def test_garbage_config_is_refused_by_both_doors(label):
    """One fault, two doors, one code - and the message names the key that broke it."""
    document = json.loads(GARBAGE[label])
    for door, call in (("set_config", _patch), ("import_config", _replace)):
        with pytest.raises(StudioError) as caught:
            call(document)
        assert caught.value.code is ErrorCode.INVALID_CONFIG, f"{label} via {door}"
        issues = caught.value.details["issues"]
        assert issues and issues[0]["path"], "an INVALID_CONFIG must say which key, where"


@pytest.mark.parametrize("label", sorted(TOLERATED))
def test_unknown_config_keys_warn_instead_of_failing(label):
    """D10: an unknown key is a warning; refusing a config the user can see is not helpful."""
    assert _patch(json.loads(TOLERATED[label]))["warnings"], f"{label} should warn"


def _config_case(document):
    """A matrix case out of a patch document (a fresh service each time)."""
    def run(_tmp):
        svc = StudioService(_manager(), AppConfig.default(), mock=True)
        with pytest.raises(StudioError) as caught:
            svc.apply_config_patch(document)
        return caught.value
    return run


def _config_import_case(document):
    def run(_tmp):
        svc = StudioService(_manager(), AppConfig.default(), mock=True)
        with pytest.raises(StudioError) as caught:
            svc.import_config(document)
        return caught.value
    return run


def _config_file_case(name: str, data: bytes):
    """Read a config file the caller should not have been able to load."""
    def run(tmp):
        svc = StudioService(_manager(), AppConfig.default(), mock=True)
        path = tmp / name
        if data is not None:
            path.write_bytes(data)
        with pytest.raises(StudioError) as caught:
            svc.import_config(path=str(path))
        return caught.value
    return run


def _config_arguments_case(**kwargs):
    # the service method's parameter is `data`; `json` is the tool's wire name for it
    def run(_tmp):
        svc = StudioService(_manager(), AppConfig.default(), mock=True)
        with pytest.raises(StudioError) as caught:
            svc.import_config(**kwargs)
        return caught.value
    return run


#: the config faults, keyed by their MATRIX row, for the sweeps at the end of the file
CONFIG_CASES: dict[str, Any] = {
    "document is a list": _config_import_case([1, 2, 3]),
    "document is a string": _config_import_case("not a config"),
    "section is a list": _config_case(json.loads(GARBAGE["section is a list"])),
    "value has the wrong type": _config_case(json.loads(GARBAGE["threshold is a word"])),
    "value out of range": _config_case(json.loads(GARBAGE["threshold out of range"])),
    "unknown schema_version": _config_case(json.loads(GARBAGE["schema_version from the future"])),
    "file is not JSON": _config_file_case("garbage.json", b"{not json,,,"),
    "file is not UTF-8": _config_file_case("nonutf.json", b"\xff\xfe{\x00\x01"),
    "file does not exist": _config_file_case("absent.json", None),
    "path is a directory": _config_file_case(".", None),
    "both json and path given": _config_arguments_case(data={}, path="/tmp/x.json"),
    "neither json nor path given": _config_arguments_case(),
}


def test_a_refused_patch_leaves_the_live_config_alone():
    service = StudioService(_manager(), AppConfig.default(), mock=True)
    before = service.config.to_dict()
    with pytest.raises(StudioError):
        service.apply_config_patch({"ocr": {"batch_size": -5}})
    assert service.config.to_dict() == before


def test_a_refused_import_leaves_the_live_config_alone():
    service = StudioService(_manager(), AppConfig.default(), mock=True)
    before = service.config.to_dict()
    with pytest.raises(StudioError):
        service.import_config({"ocr": {"device": "tpu"}})
    assert service.config.to_dict() == before


def test_a_redacted_placeholder_never_becomes_a_literal_token():
    """P8.2: `<redacted>` arriving back means "keep what is set", not "set this string"."""
    service = StudioService(_manager(), AppConfig.default(), mock=True)
    service.apply_config_patch({"mcp": {"auth_token": "s3cret"}})
    answer = service.apply_config_patch({"mcp": {"auth_token": "<redacted>"}})
    assert answer["config"]["mcp"]["auth_token"] == "s3cret"
    # `set_config` reports it as a warning, `import_config` as `kept` (see SetConfigResult):
    # both say it out loud, neither writes the placeholder into the live config.
    assert any("mcp.auth_token" in warning for warning in answer["warnings"]), answer["warnings"]
    assert service.import_config({"mcp": {"auth_token": "<redacted>"}})["kept"] == ["mcp.auth_token"]


def test_an_empty_import_is_a_reset_and_says_so():
    """`import_config` replaces, so `{}` means "defaults, all of them" - on purpose.

    It is the one destructive config call, which is why the answer lists every key it changed
    (and why `set_config({})`, which would mean the same thing by accident, is refused).
    """
    service = StudioService(_manager(), AppConfig.default(), mock=True)
    service.apply_config_patch({"ocr": {"batch_size": 64}})
    answer = service.import_config({})
    assert answer["config"]["ocr"]["batch_size"] == AppConfig.default().ocr.batch_size
    assert "ocr.batch_size" in answer["changed"]


# ---------------------------------------------------------------------------
# Entry point 5 - the MCP wire (what a client actually receives)
# ---------------------------------------------------------------------------

pytest.importorskip("mcp", reason="the MCP SDK is not installed (pip install 'animetext-studio[mcp]')")

from mcp.shared.exceptions import MCPError                       # noqa: E402

#: (tool, arguments, expected code). The first block is what the SDK's own validator rejects -
#: the class of failure that used to arrive as `isError` with a sentence and no code at all.
WIRE_CASES: list[tuple[str, dict, ErrorCode]] = [
    ("set_config", {}, ErrorCode.INVALID_ARGUMENT),
    ("set_config", {"patch": "not-a-dict"}, ErrorCode.INVALID_ARGUMENT),
    ("set_config", {"patch": {"detection": {"iou_threshold": 42}}}, ErrorCode.INVALID_CONFIG),
    # an unknown *argument name* is dropped by the SDK, so the empty patch is what fails
    ("set_config", {"definitely_not_a_config_section": 1, "patch": {}},
     ErrorCode.INVALID_CONFIG),
    ("set_zoom", {"mode": 5}, ErrorCode.INVALID_ARGUMENT),
    ("set_zoom", {"mode": "percent", "percent": 5}, ErrorCode.INVALID_ARGUMENT),
    ("highlight_block", {"id": "abc"}, ErrorCode.INVALID_ARGUMENT),
    ("highlight_block", {"id": 1, "duration_ms": 999999}, ErrorCode.INVALID_ARGUMENT),
    ("get_result", {"block_ids": 5}, ErrorCode.INVALID_ARGUMENT),
    ("save_result", {}, ErrorCode.INVALID_ARGUMENT),
    ("load_image", {}, ErrorCode.INVALID_ARGUMENT),
    ("load_image", {"path": "/tmp/x.png", "base64_png_jpg": "YQ=="}, ErrorCode.INVALID_ARGUMENT),
    ("load_image", {"base64_png_jpg": "!!!not base64!!!"}, ErrorCode.INVALID_ARGUMENT),
    ("load_image", {"base64_png_jpg": b64(b"this is not a png")}, ErrorCode.INVALID_IMAGE),
    ("run_pipeline", {"ocr": "not-a-dict"}, ErrorCode.INVALID_ARGUMENT),
    ("run_pipeline", {"ocr": {"batch_size": -1}}, ErrorCode.INVALID_CONFIG),
    ("run_pipeline", {"reading_order": "sideways"}, ErrorCode.INVALID_CONFIG),
    ("detect", {"score_threshold": 42}, ErrorCode.INVALID_ARGUMENT),
    ("get_annotated_image", {"scale": 50}, ErrorCode.INVALID_ARGUMENT),
    ("set_overlay_visible", {}, ErrorCode.INVALID_ARGUMENT),
    ("warm_engines", {"probe": "not-a-bool"}, ErrorCode.INVALID_ARGUMENT),
    # P9.3 — the first door that acts on an existing answer rather than making a new one.
    ("reocr_blocks", {"block_ids": []}, ErrorCode.INVALID_ARGUMENT),
    ("reocr_blocks", {"block_ids": "1"}, ErrorCode.INVALID_ARGUMENT),
    # No page is loaded in this harness, so an id that cannot exist is answered at the door
    # *before* the page check only when the list itself is malformed; a well-formed ask for a
    # block on a page that was never loaded is, honestly, `NO_IMAGE`.
    ("reocr_blocks", {"block_ids": [999]}, ErrorCode.NO_IMAGE),
    ("reocr_blocks", {"block_ids": [0]}, ErrorCode.NO_IMAGE),
    ("reocr_blocks", {"block_ids": [0], "ocr": {"max_new_tokens": -8}}, ErrorCode.INVALID_CONFIG),
    ("import_config", {"json": {}, "path": "/nope.json"}, ErrorCode.INVALID_ARGUMENT),
    ("import_config", {"json": {"ocr": {"device": "tpu"}}}, ErrorCode.INVALID_CONFIG),
    ("no_such_tool_at_all", {}, ErrorCode.INVALID_ARGUMENT),
]


def _session_of(url: str):
    """A connected `ClientSession`, as an async context manager (same shape as test_mcp_http)."""
    import contextlib

    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    @contextlib.asynccontextmanager
    async def open_():
        async with contextlib.AsyncExitStack() as stack:
            streams = await stack.enter_async_context(streamable_http_client(url))
            session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
            await session.initialize()
            yield session

    return open_()


def _call(url: str, tool: str, arguments: dict):
    import asyncio

    async def go():
        async with _session_of(url) as session:
            return await session.call_tool(tool, arguments)

    return asyncio.run(go())


def _error_document(result) -> dict:
    """The §2.3 promise, checked: isError + {code,message} + the same object as text."""
    assert result.is_error, "the call should have failed"
    document = result.structured_content
    assert isinstance(document, dict), (
        f"a failed tool call must carry structured content, got {document!r}")
    assert set(document) >= {"code", "message"}, document
    assert document["code"] in {code.value for code in ErrorCode}, document
    assert document["message"].strip(), "an error must say something"
    text = result.content[0].text
    assert json.loads(text)["code"] == document["code"], "text and structuredContent must agree"
    return document


@pytest.mark.parametrize("index", range(len(WIRE_CASES)))
def test_every_rejected_tool_call_comes_back_with_a_code(mcp_http, index):
    """The headline invariant of P8.4.

    A harness is allowed to write `if result.structuredContent.code == ...` and never think
    again - including for the failures the SDK produces before our code runs (bad argument
    types, missing arguments, unknown tools). Before the error-shaping middleware those came
    back as `isError` with a pydantic sentence and no structured content at all.
    """
    tool, arguments, expected = WIRE_CASES[index]
    document = _error_document(_call(mcp_http["url"], tool, arguments))
    assert document["code"] == expected.value, f"{tool}({arguments}) -> {document}"


def test_a_rejected_call_is_recorded_for_the_user(mcp_http):
    """The GUI log is the user's witness. A client hammering the server with garbage used to
    leave no trace in it at all; now every rejection is one row with the same code."""
    service = mcp_http["service"]
    before = len(service.activity.events())
    for tool, arguments, _code in WIRE_CASES:
        _call(mcp_http["url"], tool, arguments)
    rows = service.activity.events()[:len(WIRE_CASES) + 5]
    assert len(rows) > before, "rejected calls must appear in the activity log"
    assert all(row.ok is False for row in rows[:len(WIRE_CASES)]), rows[0].to_dict()
    assert all(row.code for row in rows[:len(WIRE_CASES)]), "and each row carries its code"
    unknown = [row for row in rows if row.tool == "no_such_tool_at_all"]
    assert unknown and unknown[0].code == ErrorCode.INVALID_ARGUMENT.value


def test_the_server_is_fine_after_a_storm_of_garbage(mcp_http):
    """Nothing in the matrix may wedge the app: after all that, a real run still works."""
    for tool, arguments, _code in WIRE_CASES:
        _call(mcp_http["url"], tool, arguments)
    assert _call(mcp_http["url"], "get_status", {}).is_error is False
    _call(mcp_http["url"], "load_image", {"path": SAMPLE})
    result = _call(mcp_http["url"], "run_pipeline", {})
    assert not result.is_error, result.structured_content
    assert result.structured_content["block_count"] > 0


def test_an_unknown_tool_says_which_tools_exist(mcp_http):
    document = _error_document(_call(mcp_http["url"], "no_such_tool_at_all", {}))
    assert document["details"]["reason"] == "unknown_tool"
    assert "run_pipeline" in document["details"]["known_tools"]


def test_from_clipboard_headless_is_an_explanation_not_a_failure(mcp_http):
    """The one thing a headless server cannot do, said in a way the client can act on."""
    document = _error_document(_call(mcp_http["url"], "load_image", {"from_clipboard": True}))
    assert document["code"] == ErrorCode.INVALID_ARGUMENT.value
    assert "base64_png_jpg" in document["details"]["hint"]


def test_an_oversized_payload_is_refused_before_it_is_decoded(mcp_http, monkeypatch):
    """The cap is a *service* rule now (P8.4), so it is enforced on every door and logged."""
    from animetext_studio.mcp import service as service_mod

    monkeypatch.setattr(service_mod, "MAX_BASE64_CHARS", 4096)
    document = _error_document(
        _call(mcp_http["url"], "load_image", {"base64_png_jpg": "A" * 5000}))
    assert document["code"] == ErrorCode.INVALID_ARGUMENT.value
    assert document["details"]["max_chars"] == 4096
    row = mcp_http["service"].activity.last()
    assert row is not None and row.tool == "load_image" and not row.ok
    # ...and the pixels are never echoed into the log, however the call fails
    assert all("<" in str(value) or len(str(value)) <= 120
               for value in (row.args or {}).values()), row.args


def test_mime_wrapped_base64_loads_over_the_wire(mcp_http):
    """A client that wraps its base64 at 76 columns is doing nothing wrong (P8.4)."""
    result = _call(mcp_http["url"], "load_image",
                   {"base64_png_jpg": wrapped_b64(png_bytes(64, 48))})
    assert not result.is_error, result.structured_content
    assert result.structured_content["image"]["width"] == 64


# ---------------------------------------------------------------------------
# The command line: exit codes are the error contract there
# ---------------------------------------------------------------------------

CLI_CASES: list[tuple[list[str], ErrorCode, str]] = [
    (["--headless"], ErrorCode.INVALID_ARGUMENT, "needs an action"),
    (["--headless", "--run"], ErrorCode.NO_IMAGE, "--run needs an image"),
    (["--headless", "--mock", "--run", "--image", "/definitely/not/here.png"],
     ErrorCode.IO_ERROR, "not/here.png"),
    (["--headless", "--mock", "--run", "--image", "@corrupt@"],
     ErrorCode.INVALID_IMAGE, "not a readable image"),
    (["--config", "@garbage@"], ErrorCode.INVALID_CONFIG, "not valid json"),
    (["--export-config", "/proc/1/cannot/write.json"], ErrorCode.IO_ERROR, "Cannot write"),
    # D93: an option that chooses an action, given an empty value, is answered — it used to fall
    # through to the default action and open a window.
    (["--warm-devices", ""], ErrorCode.INVALID_ARGUMENT, "empty value"),
    (["--export-config", ""], ErrorCode.INVALID_ARGUMENT, "--export-config"),
]


@pytest.mark.parametrize("argv,code,needle", CLI_CASES,
                         ids=[c[1].value.lower() + "-" + str(i)
                              for i, c in enumerate(CLI_CASES)])
def test_the_cli_answers_with_the_matrix_code(argv, code, needle, tmp_path, capsys):
    from animetext_studio.__main__ import main

    substitutions = {
        "@corrupt@": str(write(tmp_path, "corrupt.png", b"not a png")),
        "@garbage@": str(write(tmp_path, "garbage.json", b"{not json,,,")),
    }
    argv = [substitutions.get(arg, arg) for arg in argv]
    status = main(argv)
    want = EXIT_CODES[code]
    assert status == want, f"{argv} -> exit {status}, expected {want}"
    err = json.loads(capsys.readouterr().err)
    assert err["error"]["code"] == code.value, err
    if needle:
        assert needle.lower() in json.dumps(err).lower()


def test_validate_config_reports_instead_of_raising(tmp_path, capsys):
    """`--validate-config` is the diagnostic door: a broken file is a *report*, not a crash."""
    from animetext_studio.__main__ import main

    path = write(tmp_path, "bad.json", b'{"detection": {"iou_threshold": 42}, "nope": 1}')
    assert main(["--validate-config", str(path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is False
    assert any(issue["path"] == "detection.iou_threshold" for issue in report["errors"])
    assert any(issue["path"] == "nope" for issue in report["warnings"])
    capsys.readouterr()


def test_a_bad_option_value_is_argparse_business_not_ours(capsys):
    """argparse exits 2 with its own message; that is the documented CLI behaviour."""
    from animetext_studio.__main__ import main

    with pytest.raises(SystemExit) as caught:
        main(["--iou", "not-a-number"])
    assert caught.value.code == 2
    assert "expected a number" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        main(["--mcp-port", "99999"])
    assert "port must be in" in capsys.readouterr().err


def test_a_headless_run_writes_the_document_and_reads_cleanly(tmp_path, capsys):
    """The success path of the same command, so the error cases above mean something."""
    from animetext_studio.__main__ import main

    out_file = tmp_path / "result.json"
    status = main(["--headless", "--mock", "--image", SAMPLE, "--run",
                   "--out", str(out_file)])
    assert status == 0
    document = json.loads(out_file.read_text(encoding="utf-8"))
    result_schema = json.loads((SCHEMA_DIR / "result.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(result_schema).validate(document)
    assert document["stage"] == "full" and document["block_count"] > 0


# ---------------------------------------------------------------------------
# Offline and no-token: the two ways this app meets a missing model
# ---------------------------------------------------------------------------

def test_offline_with_an_empty_bundle_says_model_not_found(tmp_path, monkeypatch):
    """No network, no weights: MODEL_NOT_FOUND naming the directory and the fix."""
    from animetext_studio.core.pipeline import engines_from_config

    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "empty-bundle"))
    config = AppConfig.default()
    config.paths.offline = True
    pair = engines_from_config(config)
    with pytest.raises(StudioError) as caught:
        Pipeline(pair.detection, pair.ocr, config).run_file(SAMPLE)
    assert caught.value.code is ErrorCode.MODEL_NOT_FOUND
    assert "offline" in json.dumps(caught.value.details or {}).lower()


def test_no_token_anywhere_is_answered_none_not_an_error(tmp_path, monkeypatch):
    """Token discovery is best-effort: nothing found is `None`, never an exception."""
    from animetext_studio.core import model_store

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(model_store, "discover_token_file", lambda *a, **k: None)
    assert model_store.discover_hf_token() is None
    assert model_store.discover_hf_token("  explicit  ") == "explicit"


def test_a_gated_repo_refusal_reaches_the_client_as_hf_auth_required(mcp_http):
    """The one error whose `details` are instructions, kept intact all the way to the wire."""
    service = mcp_http["service"]
    boom = StudioError(ErrorCode.HF_AUTH_REQUIRED,
                       "Hugging Face denied access to yolo12m_animetext. The repo is gated.",
                       {"steps": ["accept the terms", "create a token", "export HF_TOKEN"]})
    service.manager.pipeline.detection.detect = lambda *a, **k: (_ for _ in ()).throw(boom)
    service.load_image(path=SAMPLE)
    document = _error_document(_call(mcp_http["url"], "run_pipeline", {}))
    assert document["code"] == ErrorCode.HF_AUTH_REQUIRED.value
    assert document["details"]["steps"] == ["accept the terms", "create a token",
                                            "export HF_TOKEN"]


# ---------------------------------------------------------------------------
# The GUI door (offscreen): a failure says so in the window, and wedges nothing
# ---------------------------------------------------------------------------

pytest_gui = pytest.importorskip("PySide6", reason="PySide6 not installed")


def _window(qtbot):
    from animetext_studio.core.pipeline import engines_from_config
    from animetext_studio.gui.main_window import MainWindow

    config = AppConfig.default()
    engines = engines_from_config(config, mock=True)
    manager = JobManager(Pipeline(engines.detection, engines.ocr, config))
    win = MainWindow(manager, config, mock=True)
    qtbot.addWidget(win)
    win.show()
    return win, manager


def test_a_corrupt_file_is_said_in_the_status_bar_not_raised(qtbot, tmp_path):
    win, _manager = _window(qtbot)
    assert win.load_path(write(tmp_path, "junk.png", b"definitely not a png")) is False
    text = win.status_label.text()
    assert "INVALID_IMAGE" in text, text          # the code is in the human line, on purpose
    assert win.image_info is None


def test_the_mcp_side_reads_that_failure_back_as_the_same_code(qtbot, tmp_path):
    """`_window_failure` is the bridge from "the window said no" to a structured error."""
    from animetext_studio.mcp.service import _window_failure

    win, _manager = _window(qtbot)
    win.load_path(write(tmp_path, "junk.png", b"definitely not a png"))
    assert _window_failure(win).code is ErrorCode.INVALID_IMAGE


def test_a_failing_job_leaves_the_window_runnable(qtbot, tmp_path):
    from animetext_studio.engines.base import FakeOcrEngine

    class Failing(FakeOcrEngine):
        def ocr_crops(self, images, **kwargs):
            raise StudioError(ErrorCode.ENGINE_LOAD_FAILED, "torch is not installed")

    win, manager = _window(qtbot)
    manager.pipeline.ocr = Failing()
    assert win.load_path(SAMPLE)
    win.run_now()
    manager.wait(20)
    assert manager.state is JobState.ERROR
    # the status bar is written by a slot, so it lands one turn after the worker settles
    qtbot.waitUntil(lambda: "ENGINE_LOAD_FAILED" in win.status_label.text(), timeout=5000)
    # and the window is not wedged: a second attempt with a working engine runs
    manager.pipeline.ocr = FakeOcrEngine()
    win.run_now()
    manager.wait(20)
    assert manager.state is JobState.DONE


# ---------------------------------------------------------------------------
# The doors P9 opened, as cases
# ---------------------------------------------------------------------------

def _reocr_with_no_ids(_tmp=None):
    service = fresh_service()
    run_page(service)
    return service.start_reocr([])


def _reocr_an_id_that_is_not_there(_tmp=None):
    service = fresh_service()
    run_page(service)
    return service.start_reocr([999])


def _reocr_with_no_page(_tmp=None):
    return fresh_service().start_reocr([0])


def _reocr_a_page_with_no_document(_tmp=None):
    return load_page(fresh_service()).start_reocr([0])


#: The re-OCR door's refusals, in the order the door checks them: the shape of the id list,
#: then the page, then the document, then the ids against that document (P9.3). Swept by the
#: same three invariants as every other door, so a row cannot rot into a wish.
REOCR_CASES: dict[str, Callable] = {
    "re-OCR with an empty id list": _reocr_with_no_ids,
    "re-OCR an id that is not in the document": _reocr_an_id_that_is_not_there,
    "re-OCR with no page loaded": _reocr_with_no_page,
    "re-OCR a page that has no document yet": _reocr_a_page_with_no_document,
}


def _bundled_weights_are_corrupt(tmp):
    """P9.1 — the bundle is there and it is wrong, which is not the same fault as no bundle.

    The manifest recorded good bytes; the file on disk is now different ones. Only a hash can
    tell, and only on the failure path (the hot path stays hash-free), so this is the one row
    whose whole value is its `details`: the path, both hashes, and the command that fixes it.
    """
    from animetext_studio.core import model_store as ms
    from animetext_studio.engines.detection import DetectionEngine

    root = tmp / "bundle"
    variant = root / ms.DETECTION_DIRNAME / "yolo11m"
    variant.mkdir(parents=True)
    path = variant / "model.onnx"
    path.write_bytes(b"real weights" * 100_000)
    ms.write_manifest(root, [{"path": str(path), "sha256": ms.sha256_file(path),
                              "size_bytes": path.stat().st_size}], merge=False)
    path.write_bytes(b"truncated cop" * 87_500)     # right shape, wrong bytes: the real fault
    engine = DetectionEngine(models_dir=str(root), offline=True)
    # `_open_failed` is what `load()` hands a backend's exception to; offline it cannot retry,
    # so the diagnosis is the answer. The public path is pinned in tests/test_model_store.py.
    return engine._open_failed("yolo11m", path, RuntimeError("Protobuf parsing failed."))


#: The offline door's one new row. The older offline rows are asserted by the dedicated tests
#: in the offline section, where the fakes that imitate a hub live.
OFFLINE_CASES: dict[str, Callable] = {"bundled weights are corrupt": _bundled_weights_are_corrupt}


# ---------------------------------------------------------------------------
# Invariants over the whole matrix
# ---------------------------------------------------------------------------

RUNNABLE: dict[tuple[str, str], Callable] = {}
for _name, _table in (("imaging", IMAGING_CASES), ("pipeline", PIPELINE_CASES),
                      ("jobs", JOBS_CASES), ("config", CONFIG_CASES),
                      ("service", REOCR_CASES), ("offline", OFFLINE_CASES)):
    for _fault, _fn in _table.items():
        RUNNABLE[(_name, _fault)] = _fn


def _expect_studio_error(entry: str, fault: str, tmp_path) -> StudioError:
    """Run one matrix case and return the error it must produce (fails on a bare exception).

    Cases are written in two styles: they raise, or they assert internally and hand back the
    error they caught. Both are fine; a Python built-in exception is not, and neither is a case
    that quietly succeeds — the sweep is what stops either from hiding.
    """
    try:
        outcome = RUNNABLE[(entry, fault)](tmp_path)
    except StudioError as exc:
        return exc
    except Exception as exc:                                    # noqa: BLE001
        pytest.fail(f"{entry}/{fault} escaped as {type(exc).__name__}: {exc}")
    if isinstance(outcome, StudioError):
        return outcome
    pytest.fail(f"{entry}/{fault} produced no error — the row is a lie")


@pytest.mark.parametrize("entry,fault", sorted(RUNNABLE), ids=lambda v: str(v))
def test_no_matrix_fault_escapes_as_a_bare_exception(entry, fault, tmp_path):
    """The whole point of the matrix: a fault is a `StudioError`, never a Python surprise."""
    exc = _expect_studio_error(entry, fault, tmp_path)
    assert type(exc) is StudioError, f"{entry}/{fault} raised a subclass: {type(exc)}"
    assert exc.message.strip(), f"{entry}/{fault} has an empty message"
    assert exc.exit_code != 0


@pytest.mark.parametrize("entry,fault", sorted(RUNNABLE), ids=lambda v: str(v))
def test_each_matrix_row_carries_the_code_the_matrix_promises(entry, fault, tmp_path):
    exc = _expect_studio_error(entry, fault, tmp_path)
    assert exc.code is CODE_OF[(entry, fault)], (
        f"{entry}/{fault}: raised {exc.code.value}, matrix says "
        f"{CODE_OF[(entry, fault)].value} ({exc.message})")


def test_the_app_still_works_after_the_entire_matrix(tmp_path):
    """Throw everything at one process, then ask it to do its job. It must still do it."""
    for (entry, fault), fn in sorted(RUNNABLE.items()):
        try:
            fn(tmp_path)
        except StudioError:
            pass
        except Exception as exc:                                # noqa: BLE001
            pytest.fail(f"{entry}/{fault} raised {type(exc).__name__}: {exc}")
    service = fresh_service()
    run_page(service)
    assert service.manager.result.block_count > 0
    assert service.annotated()[1]["blocks"] > 0


def test_internal_is_reserved_for_the_engine_breaking_its_contract():
    """INTERNAL means "a component misbehaved", not "the caller made a typo".

    Pinned as a set so a new INVALID_* path cannot quietly be reported as INTERNAL (the old
    behaviour for a bad path, a short OCR answer and a rejected tool argument alike), and so a
    genuine contract violation cannot be downgraded into something the client is blamed for.
    """
    internal = {f"{entry}/{fault}" for (entry, fault), code in CODE_OF.items()
                if code is ErrorCode.INTERNAL}
    assert internal == {
        "pipeline/engine returns None",
        "pipeline/engine returns dicts",
        "pipeline/ocr returns the wrong count",
        "pipeline/ocr returns a non-string",
        "jobs/worker raises a bare exception",
    }, sorted(internal)


def test_every_code_has_an_exit_code_and_is_published():
    for code in ErrorCode:
        assert EXIT_CODES[code] != 0, f"{code} would exit 0 — silent success"
    catalog = json.loads((SCHEMA_DIR / "mcp-tools.json").read_text(encoding="utf-8"))
    assert set(catalog["errorCodes"]) == {c.value for c in ErrorCode}


def test_an_error_document_survives_a_round_trip_even_from_another_version():
    """`from_dict` is a reader: unknown codes and missing messages must not explode."""
    original = StudioError(ErrorCode.DOWNLOAD_FAILED, "no network", {"reason": "x"})
    assert StudioError.from_dict(original.to_dict()).to_dict() == original.to_dict()
    future = StudioError.from_dict({"code": "SOMETHING_NEW", "message": "later",
                                    "details": {"a": 1}})
    assert future.code is ErrorCode.INTERNAL
    assert future.details["unknown_code"] == "SOMETHING_NEW"
    assert future.details["a"] == 1
    assert StudioError.from_dict({"code": "NO_IMAGE"}).message
    with pytest.raises(StudioError) as caught:
        StudioError.from_dict(["not", "an", "object"])
    assert caught.value.code is ErrorCode.INVALID_ARGUMENT


def test_the_activity_log_records_every_service_failure_exactly_once():
    service = fresh_service()
    for patch in ({"detection": {"iou_threshold": 42}}, {"ocr": {"device": "tpu"}},
                  {"mcp": {"port": 0}}):
        with pytest.raises(StudioError):
            # `execute`, not `apply_config_patch`: the log is the record of *calls*, and it is
            # the dispatch wrapper that writes one entry per call, success or failure.
            service.execute("set_config", lambda p=patch: service.apply_config_patch(p),
                            args={"patch": patch})
    entries = [event for event in service.activity.events() if not event.ok]
    assert len(entries) == 3, [(e.tool, e.code) for e in entries]
    assert {e.code for e in entries} == {ErrorCode.INVALID_CONFIG.value}
    assert all(e.args for e in entries), "a logged failure must say what was asked for"


MCP_DOC = ROOT / "docs" / "MCP.md"


def test_the_mcp_doc_agrees_with_the_error_taxonomy():
    """Both documents' code tables must match `EXIT_CODES`, row for row.

    Writing the matrix meant reading the error sections again, and they had drifted: `MCP.md`
    was missing `INVALID_CONFIG` and `NOT_IMPLEMENTED` entirely and still quoted a 30 s dispatch
    timeout (it is 20), and the first draft of `ERROR-MATRIX.md` — written from memory rather
    than from `core/types.py` — got six exit codes wrong. That is exactly the failure a table
    like this invites, so the numbers are compared rather than trusted.
    """
    for document in (MCP_DOC, ROOT / "docs" / "ERROR-MATRIX.md"):
        rows = _exit_table(document)
        missing = sorted({c.value for c in ErrorCode} - set(rows))
        extra = sorted(set(rows) - {c.value for c in ErrorCode})
        assert not missing and not extra, f"{document.name}: missing {missing}, unknown {extra}"
        wrong = [(code, rows[code], EXIT_CODES[code]) for code in rows
                 if rows[code] != EXIT_CODES[code]]
        assert not wrong, f"{document.name} has the wrong CLI exit for: " + ", ".join(
            f"{code} (doc {got}, real {want})" for code, got, want in wrong)


def _exit_table(document) -> dict[str, int]:
    """Every `| `CODE` | … | <exit> |` row in a markdown document, whichever column it is in.

    The two documents lay the same facts out differently (three columns in MCP.md, four in
    ERROR-MATRIX.md), so the reader looks for a code in column 1 and a number in the next
    numeric cell rather than assuming a shape.
    """
    known = {c.value for c in ErrorCode}
    found: dict[str, int] = {}
    for line in document.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not (cells[0].startswith("`") and cells[0].endswith("`")):
            continue
        code = cells[0].strip("`")
        if code not in known:
            continue
        for cell in cells[1:]:
            if cell.isdigit():
                found[code] = int(cell)
                break
    return found


DOCS = ROOT / "docs" / "ERROR-MATRIX.md"


def test_every_row_of_the_matrix_is_documented():
    """`docs/ERROR-MATRIX.md` carries the same rows; a matrix nobody reads is not a contract."""
    assert DOCS.exists(), "docs/ERROR-MATRIX.md is missing"
    text = DOCS.read_text(encoding="utf-8")
    missing = [f"{entry} / {fault}" for entry, fault, _code in MATRIX
               if f"`{fault}`" not in text]
    assert not missing, "undocumented matrix rows: " + "; ".join(missing)


def test_the_documentation_names_every_error_code_and_every_tool():
    text = DOCS.read_text(encoding="utf-8")
    for code in ErrorCode:
        assert code.value in text, f"{code.value} is not explained in docs/ERROR-MATRIX.md"
    catalog = json.loads((SCHEMA_DIR / "mcp-tools.json").read_text(encoding="utf-8"))
    for tool in catalog["tools"]:
        assert f"`{tool['name']}`" in text, f"{tool['name']} is not in the docs"


class TestReocrDoor:
    """P9.3 — the `reocr_blocks` rows, on the door that owns them.

    The order of the checks is the contract: the shape of `block_ids` is answerable with no
    state at all, then the page, then the document, then the ids against that document. Each
    answer names the next thing the caller has to do.
    """

    def test_an_empty_id_list_is_refused_before_anything_runs(self):
        svc = fresh_service()
        run_page(svc)
        with pytest.raises(StudioError) as ei:
            svc.start_reocr([])
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT

    def test_a_non_list_is_refused(self):
        svc = fresh_service()
        run_page(svc)
        for bad in ("1", 1, {"ids": [0]}):
            with pytest.raises(StudioError) as ei:
                svc.start_reocr(bad)
            assert ei.value.code is ErrorCode.INVALID_ARGUMENT, bad

    def test_an_unknown_id_names_the_ones_that_exist(self):
        svc = fresh_service()
        run_page(svc)
        with pytest.raises(StudioError) as ei:
            svc.start_reocr([999])
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert ei.value.details["unknown"] == [999]
        assert ei.value.details["available"] == [0, 1, 2, 3, 4]

    def test_with_no_page_loaded_the_answer_is_no_image(self):
        svc = fresh_service()
        with pytest.raises(StudioError) as ei:
            svc.start_reocr([0])
        assert ei.value.code is ErrorCode.NO_IMAGE

    def test_a_page_without_a_result_says_there_is_nothing_to_re_read(self):
        svc = load_page(fresh_service())
        with pytest.raises(StudioError) as ei:
            svc.start_reocr([0])
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert "no result" in ei.value.message.lower()

    def test_a_bad_override_is_refused_before_the_job_starts(self):
        """The override patch runs first, so a bad `max_new_tokens` never becomes a running job."""
        svc = fresh_service()
        run_page(svc)
        with pytest.raises(StudioError) as ei:
            svc.apply_run_overrides({"ocr": {"max_new_tokens": -8}})
        assert ei.value.code is ErrorCode.INVALID_CONFIG
        assert svc.manager.is_busy is False
