"""Shared fixtures: schema loading + a canonical sample result."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from animetext_studio.core.model_store import MODELS_DIR_ENV
from animetext_studio.core.types import (
    BBox, DetectionInfo, ImageInfo, OcrInfo, PipelineResult, TextBlock, TimingMs,
    color_for_index, sha256_bytes, DETECTION_REPO, OCR_REPO, TEXT_LABEL,
)


@pytest.fixture(autouse=True)
def _bundle_isolation(tmp_path, request, monkeypatch):
    """Unit tests never see the real on-premises bundle (decision D26).

    Vendoring weights into ``<package>/models`` must not change what an unrelated test
    observes — model listing, the ``source`` label and offline behaviour all read that
    directory. Point the store at an empty temp dir instead, so the suite is green both
    with and without a bundle. Real-model tests (``-m models``) opt back in.
    """
    if request.node.get_closest_marker("models"):
        yield
        return
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "empty-bundle"))
    yield

# Single source of truth for shipped schemas: the package itself (decision D6).
from animetext_studio.core.config import load_schema  # noqa: F401,E402  (re-exported)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "animetext_studio" / "schemas"


@pytest.fixture
def sample_result() -> PipelineResult:
    return PipelineResult(
        image=ImageInfo(
            source="file", file_name="page_01.png", width=1200, height=1700,
            sha256=sha256_bytes(b"fake-image-bytes"),
        ),
        detection=DetectionInfo(
            repo=DETECTION_REPO, model="yolo12m_animetext",
            iou_threshold=0.35, score_threshold=0.05, allow_dynamic=False,
        ),
        ocr=OcrInfo(
            repo=OCR_REPO, device="cpu", max_patches="auto", batch_size=16,
            max_new_tokens=128, crop_padding_px=2, normalize_nfkc=False,
        ),
        timing_ms=TimingMs(detect=412, ocr=1830, total=2260),
        blocks=(
            TextBlock(id=0, bbox=BBox(812, 96, 918, 402), label=TEXT_LABEL,
                      score=0.91, text="……なに!?", reading_order=0,
                      color=color_for_index(0)),
            TextBlock(id=1, bbox=BBox(100, 500, 220, 860), label=TEXT_LABEL,
                      score=0.77, text=None, reading_order=1,
                      color=color_for_index(1),
                      error={"code": "OCR_FAILED", "message": "decoder blew up"}),
        ),
        created_at="2026-11-19T12:00:00+08:00",
    )


# ---------------------------------------------------------------------------
# Part 7 — MCP fixtures (imported lazily: the suite must stay green without the SDK)
# ---------------------------------------------------------------------------

@pytest.fixture
def free_port():
    """A port that was free a moment ago (the usual trick; races are retried by the server)."""
    import socket

    def take() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    return take


@pytest.fixture
def mock_service():
    """A headless `StudioService` on the fake engines — the whole tool surface, no transport."""
    from types import SimpleNamespace

    from animetext_studio.core.config import AppConfig
    from animetext_studio.core.jobs import JobManager
    from animetext_studio.core.pipeline import Pipeline
    from animetext_studio.engines.base import FakeDetectionEngine, FakeOcrEngine
    from animetext_studio.mcp.service import StudioService

    config = AppConfig.default()
    engines = SimpleNamespace(detection=FakeDetectionEngine(), ocr=FakeOcrEngine())
    manager = JobManager(Pipeline(engines.detection, engines.ocr, config))
    return StudioService(manager, config, mock=True)


@pytest.fixture
def mcp_http(mock_service, free_port):
    """A live streamable-http server on a free loopback port: {service, server, info, url}."""
    from animetext_studio.core.config import McpConfig
    from animetext_studio.mcp.server import StudioServer

    mcp_config = McpConfig(enabled=True, transport="streamable-http",
                           host="127.0.0.1", port=free_port())
    server = StudioServer(mock_service, mcp_config)
    info = server.start()
    try:
        yield {"service": mock_service, "server": server, "info": info,
               "url": info["url"]}
    finally:
        server.stop()


@pytest.fixture
def slow_gate(mock_service):
    """Hold the detector of a running job until the test releases it.

    Yields (release, entered): `entered` is set when the job is inside `detect`, so a test
    can be sure it is racing a *running* job and not a job that already finished.
    """
    import threading

    release = threading.Event()
    entered = threading.Event()
    engine = mock_service.manager.pipeline.detection
    original = engine.detect

    def gated(image, model_name, **kwargs):
        entered.set()
        release.wait(20)
        return original(image, model_name, **kwargs)

    engine.detect = gated
    try:
        yield release, entered
    finally:
        release.set()
        engine.detect = original
