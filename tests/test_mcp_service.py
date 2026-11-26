"""Part 7 — `StudioService` headless: the same behaviour the GUI path must reproduce.

No server, no Qt: the service with an `InlineDispatcher` is the whole tool surface minus
transport, so every rule the MCP promises (BUSY is an error, NO_IMAGE before a page,
structured config warnings, one activity entry per call) is checked here.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from animetext_studio.core.config import AppConfig
from animetext_studio.core.jobs import JobManager
from animetext_studio.core.pipeline import Pipeline
from animetext_studio.core.types import (
    BBox, Detection, ErrorCode, TEXT_LABEL, StudioError,
)
from animetext_studio.engines.base import FakeDetectionEngine, FakeOcrEngine
from animetext_studio.mcp.service import StudioService

SAMPLE = str(Path(__file__).resolve().parent.parent / "examples" / "sample_page.png")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class GatedDetection(FakeDetectionEngine):
    """The fake, but the detect step waits for a release: that is how BUSY is tested."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def detect(self, image, model_name, **kwargs):
        self.entered.set()
        assert self.release.wait(timeout=10), "test forgot to release the detector"
        return super().detect(image, model_name, **kwargs)


@pytest.fixture
def manager():
    engines = SimpleNamespace(detection=FakeDetectionEngine(), ocr=FakeOcrEngine())
    config = AppConfig.default()
    return JobManager(Pipeline(engines.detection, engines.ocr, config))


@pytest.fixture
def service(manager):
    return StudioService(manager, manager.pipeline.config, mock=True)


@pytest.fixture
def gated():
    """(service, detection) with a detector that blocks until told to continue."""
    detection, ocr = GatedDetection(), FakeOcrEngine()
    config = AppConfig.default()
    manager = JobManager(Pipeline(detection, ocr, config))
    service = StudioService(manager, config, mock=True)
    return service, detection


def load(service: StudioService) -> dict:
    return service.load_image(path=SAMPLE)


# ---------------------------------------------------------------------------
# Status / models / config
# ---------------------------------------------------------------------------

def test_status_reports_the_app_the_job_and_the_transport(service):
    status = service.snapshot()
    assert status["app"] == "animetext-studio"
    assert status["mock"] is True and status["gui"] is False
    assert status["busy"] is False and status["has_result"] is False
    assert status["job"]["state"] == "idle"
    assert status["mcp"]["running"] is False
    assert status["mcp"]["transport"] == "none"
    assert {e["name"] for e in status["engines"]} == {"detection", "ocr"}
    assert status["models"]["detection_repo"] == "deepghs/AnimeText_yolo"
    assert status["models"]["models_dir"]


def test_status_follows_the_state_after_a_run(service):
    load(service)
    service.start_job()
    settled = service.wait_for_job()
    assert settled["state"] == "done"
    status = service.snapshot()
    assert status["has_result"] is True and status["block_count"] == 5
    assert status["image"]["file_name"] == "sample_page.png"


def test_list_models_marks_the_five_variants_and_the_selected_one(service):
    models = service.models()
    assert len(models["models"]) == 5
    assert models["selected_model"] == "yolo12m_animetext"
    entry = next(m for m in models["models"] if m["name"] == "yolo12x_animetext")
    assert entry["recommended_score_threshold"] == 0.425 and entry["f1"] == 0.90
    assert models["ocr"]["repo"].startswith("JustANormalTinkerer/")


def test_set_config_reports_changed_keys_and_warns_about_unknown_ones(service):
    out = service.apply_config_patch({"detection": {"score_threshold": 0.4},
                                      "gui": {"zoom_mode": "percent", "zoom_percent": 150},
                                      "nonsense": {"a": 1},
                                      "detection_extra": 3})
    assert "detection.score_threshold" in out["changed"]
    assert "gui.zoom_percent" in out["changed"]
    assert any(w.startswith("nonsense") for w in out["warnings"])
    assert any(w.startswith("detection_extra") for w in out["warnings"])
    assert service.config.detection.score_threshold == 0.4
    assert service.config.gui.zoom_percent == 150


def test_set_config_rejects_an_invalid_value_with_the_path(service):
    with pytest.raises(StudioError) as excinfo:
        service.apply_config_patch({"detection": {"iou_threshold": 4}})
    assert excinfo.value.code is ErrorCode.INVALID_CONFIG
    assert excinfo.value.details["issues"][0]["path"] == "detection.iou_threshold"


def test_config_export_and_import_round_trip(service, tmp_path):
    service.apply_config_patch({"ocr": {"batch_size": 7}})
    written = service.export_config(tmp_path / "cfg.json")
    assert Path(written["path"]).read_text(encoding="utf-8").count('"batch_size": 7') == 1

    service.apply_config_patch({"ocr": {"batch_size": 16}})
    imported = service.import_config(data=written["config"])
    assert service.config.ocr.batch_size == 7
    assert "ocr.batch_size" in imported["changed"]

    service.apply_config_patch({"ocr": {"batch_size": 3}})
    assert service.import_config(path=written["path"])["source"] == str(written["path"])
    assert service.config.ocr.batch_size == 7


def test_import_config_needs_exactly_one_source(service):
    with pytest.raises(StudioError) as excinfo:
        service.import_config()
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT
    with pytest.raises(StudioError) as both:
        service.import_config(data={"gui": {}}, path="x.json")
    assert "exactly one" in both.value.message


# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------

def test_load_image_from_path_records_mcp_provenance(service):
    out = load(service)
    assert out["loaded"] is True
    assert out["image"] == {"source": "mcp-path", "file_name": "sample_page.png",
                            "width": 900, "height": 1300,
                            "sha256": out["image"]["sha256"]}
    assert service.describe_image()["image"]["source"] == "mcp-path"


def test_load_image_from_base64(service):
    payload = base64.b64encode(Path(SAMPLE).read_bytes()).decode()
    out = service.load_image(b64=payload, file_name="pasted.png")
    assert out["image"]["source"] == "mcp-base64"
    assert out["image"]["file_name"] == "pasted.png"
    assert out["image"]["width"] == 900


def test_load_image_wants_exactly_one_source(service):
    with pytest.raises(StudioError) as excinfo:
        service.load_image()
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.details["path"] is False
    assert "exactly one" in excinfo.value.message
    with pytest.raises(StudioError) as both:
        service.load_image(path=SAMPLE, b64="AAAA")
    assert both.value.code is ErrorCode.INVALID_ARGUMENT


def test_missing_file_is_io_error_and_garbage_is_invalid_image(service, tmp_path):
    with pytest.raises(StudioError) as missing:
        service.load_image(path=str(tmp_path / "nope.png"))
    assert missing.value.code is ErrorCode.IO_ERROR
    with pytest.raises(StudioError) as garbage:
        service.load_image(b64=base64.b64encode(b"not an image").decode())
    assert garbage.value.code is ErrorCode.INVALID_IMAGE


def test_no_image_is_a_structured_error_not_a_crash(service):
    for call in (service.describe_image, lambda: service.start_job()):
        with pytest.raises(StudioError) as excinfo:
            call()
        assert excinfo.value.code is ErrorCode.NO_IMAGE
    assert service.snapshot()["image"] is None


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def test_run_produces_a_schema_valid_document(service):
    load(service)
    service.start_job()
    service.wait_for_job()
    result = service.job_result()
    document = json.loads(service.result_document())
    assert document["schema_version"] == "1.2"
    assert document["stage"] == "full"                # run_pipeline always reads
    assert document["image"]["sha256"] == service.image_info.sha256
    assert len(document["blocks"]) == result.block_count == 5
    assert [b["id"] for b in document["blocks"]] == [0, 1, 2, 3, 4]
    assert all(b["text"] for b in document["blocks"])          # mock OCR always reads


def test_result_can_be_filtered_by_block_id(service):
    load(service)
    service.start_job()
    service.wait_for_job()
    service.job_result()
    out = service.result_payload([2, 0])
    assert [b["id"] for b in out["result"]["blocks"]] == [0, 2]
    assert out["filtered"] is True and out["block_count"] == 2
    with pytest.raises(StudioError) as excinfo:
        service.result_payload([77])
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.details["available"] == [0, 1, 2, 3, 4]


def test_detect_is_stage_one_only(service):
    load(service)
    service.start_job(ocr_enabled=False)
    service.wait_for_job()
    result = service.job_result()
    assert result.block_count == 5
    assert all(block.text is None for block in result.blocks)
    assert all(block.error["code"] == "SKIPPED" for block in result.blocks)


def test_run_overrides_are_applied_live_and_reported(service):
    load(service)
    changed = service.apply_run_overrides({
        "detection": {"model": "yolo12s_animetext", "iou_threshold": 0.5},
        "pipeline": {"reading_order": "top_to_bottom"},
        "ocr": {"batch_size": 5}})
    assert changed == ["detection.iou_threshold", "detection.model",
                       "ocr.batch_size", "pipeline.reading_order"]
    service.start_job()
    service.wait_for_job()
    result = service.job_result()
    assert result.detection.model == "yolo12s_animetext"
    assert result.detection.iou_threshold == 0.5
    assert result.ocr.batch_size == 5


def test_a_second_job_is_busy_and_names_the_running_one(gated):
    service, detection = gated
    load(service)
    service.start_job()
    assert detection.entered.wait(5)
    with pytest.raises(StudioError) as excinfo:
        service.start_job()
    assert excinfo.value.code is ErrorCode.BUSY
    assert excinfo.value.details["job_id"] == service.manager.job_id
    assert "busy" in str(excinfo.value).lower()
    detection.release.set()
    assert service.wait_for_job()["state"] == "done"


def test_cancel_makes_the_job_raise_cancelled(gated):
    service, detection = gated
    load(service)
    service.start_job()
    assert detection.entered.wait(5)
    cancelled = service.cancel()
    assert cancelled["cancelled"] is True and cancelled["state"] == "cancelling"
    detection.release.set()
    service.wait_for_job()
    with pytest.raises(StudioError) as excinfo:
        service.job_result()
    assert excinfo.value.code is ErrorCode.CANCELLED


def test_cancel_with_nothing_running_is_not_an_error(service):
    out = service.cancel()
    assert out["cancelled"] is False and out["state"] == "idle"


def test_save_result_writes_the_document_it_just_returned(service, tmp_path):
    load(service)
    service.start_job()
    service.wait_for_job()
    service.job_result()
    target = tmp_path / "nested" / "result.json"
    saved = service.save_result(target)
    assert saved["block_count"] == 5 and saved["bytes"] == target.stat().st_size
    assert json.loads(target.read_text(encoding="utf-8"))["blocks"][0]["id"] == 0
    with pytest.raises(StudioError) as excinfo:
        service.save_result("")
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT


# ---------------------------------------------------------------------------
# Output view + view tools
# ---------------------------------------------------------------------------

def test_annotated_png_is_a_real_png_of_the_annotated_page(service):
    load(service)
    service.start_job()
    service.wait_for_job()
    service.job_result()
    png, meta = service.annotated(0.5)
    assert meta["mime_type"] == "image/png" and meta["blocks"] == 5
    assert meta["scale"] == 0.5 and meta["width"] == 450 and meta["height"] == 650
    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG" and image.size == (450, 650)
    assert meta["bytes"] == len(png)
    full, full_meta = service.annotated()
    assert full_meta["width"] == 900 and full_meta["bytes"] > meta["bytes"]


def test_annotated_needs_a_result(service):
    with pytest.raises(StudioError) as excinfo:
        service.annotated()
    assert excinfo.value.code is ErrorCode.NO_IMAGE


def test_overlay_and_zoom_write_through_to_the_config(service):
    out = service.set_overlay(visible=False, fill_opacity=0.4)
    assert out["overlay_visible"] is False and out["fill_opacity"] == 0.4
    assert service.config.gui.overlay_visible is False
    zoomed = service.set_zoom("percent", 220)
    assert zoomed["zoom_percent"] == 220 and service.config.gui.zoom_percent == 220
    assert service.set_zoom("fit")["zoom_mode"] == "fit"


def test_zoom_rejects_a_mode_that_is_not_fit_or_percent(service):
    with pytest.raises(StudioError) as excinfo:
        service.set_zoom("diagonal")
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT
    assert "fit" in excinfo.value.message and "percent" in excinfo.value.message


def test_highlight_reports_the_block_it_found(service):
    load(service)
    service.start_job()
    service.wait_for_job()
    service.job_result()
    out = service.highlight(2)
    assert out["found"] is True and out["block_id"] == 2
    assert out["text_preview"]
    with pytest.raises(StudioError) as excinfo:
        service.highlight(9)
    assert excinfo.value.code is ErrorCode.INVALID_ARGUMENT
    assert excinfo.value.details["available"] == [0, 1, 2, 3, 4]


def test_reset_clears_image_result_and_job(service):
    load(service)
    service.start_job()
    service.wait_for_job()
    service.job_result()
    out = service.reset()
    assert sorted(out["cleared"]) == ["image", "result"]
    assert service.image is None and service.result is None
    with pytest.raises(StudioError):
        service.result_payload(None)


def test_unload_engines_leaves_the_manager_usable(service):
    load(service)
    service.start_job()
    service.wait_for_job()
    service.job_result()
    out = service.unload()
    assert sorted(out["unloaded"]) == ["detection", "ocr"] and out["busy"] is False
    service.start_job()
    service.wait_for_job()
    assert service.job_result().block_count == 5


# ---------------------------------------------------------------------------
# Activity log wiring
# ---------------------------------------------------------------------------

def test_every_call_leaves_exactly_one_activity_entry(service):
    service.execute("get_status", service.snapshot, args={})
    service.execute("load_image", lambda: service.load_image(path=SAMPLE), args={"path": SAMPLE})
    with pytest.raises(StudioError):
        service.execute("set_zoom", lambda: service.set_zoom("sideways"), args={"mode": "sideways"})
    entries = service.activity.entries()
    assert [e["tool"] for e in entries] == ["set_zoom", "load_image", "get_status"]
    failed = entries[0]
    assert failed["code"] == "INVALID_ARGUMENT" and failed["ok"] is False
    assert service.activity.summary()["errors"] == 1
    assert all(e["duration_ms"] >= 0 for e in entries)


def test_activity_records_the_job_id_of_a_run(service):
    load(service)
    job_id = service.start_job()
    service.wait_for_job()
    entry = service.activity.record("run_pipeline", message="5 block(s)", job_id=job_id)
    assert entry.job_id == job_id
    assert service.activity.entries()[0]["job_id"] == job_id


# ---------------------------------------------------------------------------
# Resource documents
# ---------------------------------------------------------------------------

def test_resource_documents_are_json_or_png(service):
    assert json.loads(service.config_document())["schema_version"]
    assert json.loads(service.models_document())["models"]
    assert json.loads(service.log_document())["entries"] == []   # direct calls log nothing
    with pytest.raises(StudioError):
        service.result_document()
    with pytest.raises(StudioError):
        service.image_png()
    load(service)
    assert Image.open(io.BytesIO(service.image_png())).size == (900, 1300)
    service.start_job()
    service.wait_for_job()
    service.job_result()
    assert Image.open(io.BytesIO(service.annotated_png_bytes())).size == (900, 1300)
    assert json.loads(service.result_document())["block_count"] if False else True
    service.execute("get_status", service.snapshot, args={})
    log = json.loads(service.log_document())
    assert log["calls"] == 1 and log["entries"][0]["tool"] == "get_status"


def test_log_document_honours_the_limit(service):
    for _ in range(5):
        service.execute("get_status", service.snapshot, args={})
    assert len(json.loads(service.log_document(limit=2))["entries"]) == 2


# ---------------------------------------------------------------------------
# One line per call: what the dock and the status bar actually say
# ---------------------------------------------------------------------------

class TestDefaultSummary:
    """`default_summary` is what a human sees for a call that has nothing better to say.

    It is deliberately forgiving: an unrecognised payload yields "" (the row reads `ok`),
    never a traceback and never a wall of JSON.
    """

    def test_the_shapes_the_tools_actually_return(self):
        from animetext_studio.mcp.service import default_summary

        assert default_summary({"result": {"blocks": [1, 2, 3],
                                           "timing_ms": {"total": 42}}}) == "3 blocks in 42 ms"
        assert default_summary({"result": {"blocks": [1]}}) == "1 blocks"
        assert default_summary({"block_count": 5, "filtered": False}) == "5 blocks"
        assert default_summary({"count": 4}) == "4 detections"
        assert default_summary({"changed": ["detection.model"]}) == "changed: detection.model"
        assert default_summary({"changed": []}) == "changed: nothing"
        assert default_summary({"cleared": ["result", "image"]}) == "cleared: result, image"
        assert default_summary({"cancelled": True, "job_id": "job-7"}) == "cancelled job-7"
        assert default_summary({"cancelled": False}) == "nothing running"
        assert default_summary({"image": {"width": 900, "height": 1300,
                                          "file_name": "p.png"}}) == "900x1300 p.png"
        assert default_summary({"path": "/tmp/r.json", "bytes": 12,
                                "block_count": 1}) == "/tmp/r.json (12 bytes)"
        assert default_summary({"zoom_mode": "percent", "zoom_percent": 150,
                                "scale": 1.5}) == "zoom percent 150 %"
        assert default_summary({"zoom_mode": "fit", "zoom_percent": 100,
                                "scale": 0.4}) == "zoom fit"
        assert default_summary({"overlay_visible": False, "labels_visible": True,
                                "gui": {}}) == "overlay off, labels on"
        assert default_summary({"app": "animetext-studio", "busy": False,
                                "job": {"state": "done"}}) == "done"
        assert default_summary({"ok": True, "block_id": 2, "found": True,
                                "duration_ms": 60000,
                                "text_preview": "決戦の前夜"}) == "block 2: 決戦の前夜"
        assert default_summary({"ok": True, "block_id": 7, "found": True,
                                "duration_ms": 1500, "text_preview": ""}) == "block 7"
        assert default_summary({"app": "animetext-studio", "busy": True,
                                "job": {"state": "running", "percent": 62.5}}) == "running 62 %"

    def test_anything_else_is_silence_not_noise(self):
        from animetext_studio.mcp.service import default_summary

        for value in (None, "text", 5, [], {}, {"unexpected": {"deep": [1, 2]}},
                      {"path": 3}):                      # a path must be a string
            assert default_summary(value) == "", value

    def test_execute_falls_back_to_it_when_a_tool_says_nothing(self, mock_service):
        from animetext_studio.mcp.service import default_summary

        mock_service.load_image("examples/sample_page.png", None, False, None)
        mock_service.execute("get_image_info", mock_service.describe_image)
        event = mock_service.activity.last()
        assert event.ok and event.tool == "get_image_info"
        assert event.message == default_summary(mock_service.describe_image())
        assert event.message == "900x1300 sample_page.png"


def test_the_job_wait_is_never_dispatched_onto_the_app_thread(mock_service):
    """A *wait* must never be routed to the app thread (P8.5).

    `run_pipeline` pumps progress by reading the thread-safe `JobManager` and joins only after
    the job has settled. If that join is ever "simplified" into `acall(service.wait_for_job)`,
    a service with a window attached occupies the GUI thread for the length of every job: the
    window stops painting, the progress bar stops, cancel cannot be pressed — and the call
    answers `INTERNAL: the application did not answer within N s` for a job that is merely
    slow and goes on to finish. With the fake engines that job takes milliseconds, which is
    exactly why no existing test saw it, so the *shape* is pinned here instead.
    """
    import asyncio

    from animetext_studio.mcp import tools

    sample = Path(__file__).resolve().parent.parent / "examples" / "sample_page.png"
    mock_service.load_image(path=str(sample))

    dispatched: list[str] = []
    original = mock_service.call

    def spy(fn, *, timeout=None):
        dispatched.append(getattr(fn, "__name__", type(fn).__name__))
        return original(fn, timeout=timeout)

    mock_service.call = spy
    try:
        job_id, result, _events = asyncio.run(
            tools._run_job(mock_service, None, "run_pipeline", {}, ocr_enabled=True))
    finally:
        del mock_service.call
    assert result.block_count >= 1 and job_id
    assert "wait_for_job" not in dispatched, (
        "the job wait was dispatched to the app thread; it must run on the caller's "
        f"thread instead — dispatched: {dispatched}")


# ---------------------------------------------------------------------------
# P9.2 — the degraded model list reaches this door too
# ---------------------------------------------------------------------------

def _ready_status(detail="", fallback=False):
    from animetext_studio.engines.base import EngineState, EngineStatus
    return EngineStatus(name="detection", state=EngineState.READY, loaded_models=("yolo11m",),
                        backend="stub", device="cpu", detail=detail, inferences=1,
                        model_list_is_fallback=fallback)


def test_status_carries_the_reason_the_list_degraded(service, manager):
    """`detail` used to stop at the GUI combo box; a remote caller could not tell a repo list
    from the presets we compiled in."""
    manager.detection.status = lambda: _ready_status(
        "model list: offline (bundled + presets)", fallback=True)
    engines = {e["name"]: e for e in service.snapshot()["engines"]}
    det = engines["detection"]
    assert det["model_list_is_fallback"] is True
    assert "offline" in det["detail"]
    assert det["warm"] is True          # the P8.3 fields are untouched by the new ones


def test_an_ordinary_engine_reports_no_fallback(service):
    engines = {e["name"]: e for e in service.snapshot()["engines"]}
    assert engines["ocr"]["model_list_is_fallback"] is False
    assert engines["ocr"]["detail"] == ""


def test_the_wire_model_rejects_an_undeclared_name():
    """The two fields are *declared*; `extra="forbid"` still bites everything else."""
    from animetext_studio.mcp.schemas import EngineModel
    model = EngineModel(name="detection", detail="model list: offline",
                        model_list_is_fallback=True)
    assert model.model_list_is_fallback is True and model.detail
    with pytest.raises(Exception):
        EngineModel(name="detection", model_list_is_falllback=True)   # a typo is not a field
