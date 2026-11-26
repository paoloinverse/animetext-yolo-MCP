"""Real-model integration tests (PLAN.md Part 2, opt-in: `pytest -m models`).

The engine-level half of the real-weights story; `test_p8_e2e_models.py` is the product-level
half — the same weights through the CLI, the MCP service door and the MCP wire.

The bundled tests need nothing but the on-premises weights. `TestRealDetection` is the part
that hits the network, and detection additionally needs access to the GATED repo
`deepghs/AnimeText_yolo`:

  1. accept the terms on https://huggingface.co/deepghs/AnimeText_yolo
  2. export HF_TOKEN=hf_...            (or pass --hf-token to the demo tools, or put the
                                        token in HF_token.txt at the project root)
  3. pytest -m models -k detection -s

Everything here is skipped automatically when no token is present, so the default
run (`pytest`) stays offline and fast. The OCR repo is public — its smoke test is
added in Part 3.

Weight volume: yolo12n_animetext is the smallest variant (~5 MB ONNX); tests use it
unless they need quality. Set STUDIO_TEST_MODEL to use another one.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

pytestmark = [pytest.mark.models, pytest.mark.timeout(1800)]

from animetext_studio.core.config import KNOWN_MODELS, MODEL_PRESETS  # noqa: E402
from animetext_studio.core.types import DETECTION_REPO, TEXT_LABEL  # noqa: E402
from animetext_studio.engines.detection import DetectionEngine  # noqa: E402

REPO_URL = f"https://huggingface.co/{DETECTION_REPO}"
SAMPLE_PAGE = Path(__file__).resolve().parent.parent / "examples" / "sample_page.png"
TEST_MODEL = os.environ.get("STUDIO_TEST_MODEL", "yolo12n_animetext")

def _token() -> str | None:
    """Env token or the local token file — the same rule the engine uses."""
    from animetext_studio.core.model_store import discover_hf_token

    return discover_hf_token()


requires_token = pytest.mark.skipif(
    not _token(),
    reason=("no HF token available (HF_TOKEN / HUGGING_FACE_HUB_TOKEN / HF_token.txt); the "
            f"detection repo is gated — accept the terms at {REPO_URL} and set one"),
)


def _bundled_or_skip(model_name: str):
    """Skip (not fail) when the on-premises bundle does not carry this variant."""
    from animetext_studio.core.model_store import ModelStore

    store = ModelStore()
    if not store.has_detection_variant(model_name):
        pytest.skip(f"{model_name} is not vendored on premises "
                    f"(run: python scripts/fetch_models.py --variants {model_name})")
    return store


@pytest.fixture(scope="module")
def sample_page() -> Image.Image:
    if not SAMPLE_PAGE.exists():
        pytest.skip(f"{SAMPLE_PAGE} missing (run: python scripts/make_sample_image.py)")
    return Image.open(SAMPLE_PAGE).convert("RGB")


@pytest.fixture(scope="module")
def engine() -> DetectionEngine:
    return DetectionEngine()


@requires_token
class TestRealDetection:
    def test_repo_model_list_matches_the_documented_presets(self, engine):
        models = engine.list_models()
        names = [m["name"] for m in models]
        missing = [n for n in KNOWN_MODELS if n not in names]
        assert not missing, f"repo is missing documented variants: {missing}"
        assert engine.status().detail == "", "model list should come from the repo, not the fallback"
        for model in models:
            if model["known"]:
                assert model["recommended_score_threshold"] == \
                    MODEL_PRESETS[model["name"]]["recommended_score_threshold"]

    def test_repo_metadata_matches_the_hardcoded_presets(self):
        """The tiny labels.json / threshold.json of every variant vs core constants.

        This is what keeps MODEL_PRESETS and TEXT_LABEL honest (both were authored from
        the model card; TEXT_LABEL was wrong there and this test caught it).
        """
        import json

        from huggingface_hub import hf_hub_download

        from animetext_studio.core.types import TEXT_LABEL

        token = _token()
        for name in KNOWN_MODELS:
            def get(fname):
                return json.loads(Path(hf_hub_download(
                    DETECTION_REPO, f"{name}/{fname}", repo_type="model", token=token
                )).read_text(encoding="utf-8"))

            labels = get("labels.json")
            assert labels == [TEXT_LABEL], f"{name}: labels.json={labels}"
            threshold = get("threshold.json")
            preset = MODEL_PRESETS[name]
            assert threshold["threshold"] == preset["recommended_score_threshold"], \
                f"{name}: threshold.json={threshold}"
            assert abs(threshold["f1_score"] - preset["f1"]) < 1e-9, \
                f"{name}: f1 {threshold['f1_score']} != preset {preset['f1']}"

    def test_download_load_and_detect(self, engine, sample_page):
        events: list[tuple] = []
        detections = engine.detect(sample_page, TEST_MODEL, iou_threshold=0.35,
                                   score_threshold=0.05,
                                   progress=lambda *a: events.append(a))
        assert engine.status().state.value == "ready"
        assert TEST_MODEL in engine.status().loaded_models
        # status must report the provider that will really run, not the "auto" request
        assert engine.status().device in {"cpu", "cuda", "tensorrt", "coreml",
                                          "openvino", "directml", "azure"}, engine.status()
        # First stage depends on the bundle: "download" only when the weights are not
        # on premises yet, otherwise straight to "load_model" (D26).
        assert [e[0] for e in events][0] in ("download", "load_model", "detect")
        assert events[-1][1] == 100.0
        for det in detections:
            assert det.label == TEXT_LABEL   # single-class repo -> 'text_block'
            assert det.bbox.is_valid()
            assert det.bbox.clamped(sample_page.width, sample_page.height) == det.bbox
            assert 0.0 <= det.score <= 1.0

    def test_synthetic_manga_page_yields_text_blocks(self, engine, sample_page):
        """The generated page has bubbles, vertical columns and a narration band."""
        detections = engine.detect(sample_page, TEST_MODEL, iou_threshold=0.35,
                                   score_threshold=0.05)
        assert len(detections) >= 3, (
            f"expected several text blocks on the sample page, got {len(detections)}; "
            f"labels={[d.label for d in detections]}")
        assert all(det.label == TEXT_LABEL for det in detections)

    def test_score_threshold_monotonicity(self, engine, sample_page):
        loose = engine.detect(sample_page, TEST_MODEL, score_threshold=0.05)
        strict = engine.detect(sample_page, TEST_MODEL, score_threshold=0.60)
        assert len(strict) <= len(loose)
        recommended = engine.detect(sample_page, TEST_MODEL, score_threshold=None)
        assert all(d.score >= 0.20 for d in recommended)  # repo threshold is F1-optimal

    def test_allow_dynamic_runs_and_is_aspect_preserving(self, engine, sample_page):
        detections = engine.detect(sample_page, TEST_MODEL, score_threshold=0.05,
                                   allow_dynamic=True)
        assert all(d.bbox.is_valid() for d in detections)

    def test_unknown_model_is_rejected_before_download(self, engine, sample_page):
        from animetext_studio.core.types import ErrorCode, StudioError
        with pytest.raises(StudioError) as ei:
            engine.detect(sample_page, "yolo12z_does_not_exist")
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND


@pytest.mark.models
class TestBundledDetection:
    """Real inference from the on-premises bundle — no token, no network (D26).

    These run with a plain ``pytest -m models`` even without HF_TOKEN, as long as the
    weights were vendored with ``scripts/fetch_models.py``. They are the proof that a
    copy of the project folder is a complete, offline installation.
    """

    def test_variant_is_vendored_with_its_metadata(self):
        store = _bundled_or_skip(TEST_MODEL)
        status = store.detection_variant_status(TEST_MODEL)
        assert status["bundled"] and status["complete"]
        assert status["files"]["threshold.json"], "F1-optimal threshold should ship with the bundle"

    def test_offline_detection_from_bundled_weights(self, sample_page):
        _bundled_or_skip(TEST_MODEL)
        eng = DetectionEngine(offline=True)          # no token, HF_HUB_OFFLINE=1
        detections = eng.detect(sample_page, TEST_MODEL, iou_threshold=0.35,
                                score_threshold=0.05)
        assert len(detections) >= 3, f"only {len(detections)} boxes from the bundled model"
        assert all(d.label == TEXT_LABEL and d.bbox.is_valid() for d in detections)
        assert eng.status().state.value == "ready"
        assert eng.status().device in {"cpu", "cuda", "tensorrt", "coreml",
                                       "openvino", "directml", "azure"}

    def test_offline_never_emits_a_download_stage(self, sample_page):
        _bundled_or_skip(TEST_MODEL)
        eng = DetectionEngine(offline=True)
        events: list[tuple] = []
        eng.detect(sample_page, TEST_MODEL, progress=lambda *a: events.append(a))
        stages = [e[0] for e in events]
        assert "download" not in stages, f"offline run reached the network: {stages}"
        assert stages[0] == "load_model"

    def test_recommended_threshold_comes_from_the_bundled_file(self):
        _bundled_or_skip(TEST_MODEL)
        eng = DetectionEngine(offline=True)
        eng.load(TEST_MODEL)
        assert eng.recommended_threshold(TEST_MODEL) == \
            MODEL_PRESETS[TEST_MODEL]["recommended_score_threshold"]

    def test_vendored_ocr_snapshot_matches_the_pinned_revision(self):
        """The bundle and `OCR_REVISION` must agree, or an offline install pulls a 2nd copy.

        Checked against the live repo file list too: the pin was found stale (an older
        snapshot without `yolo26m-text.pt` and with an older `processing_hayai.py`) and
        bumped to the bundled HEAD.
        """
        import json

        from animetext_studio.core.model_store import ModelStore
        from animetext_studio.core.types import OCR_REVISION

        store = ModelStore()
        if not store.has_ocr_snapshot():
            pytest.skip("OCR snapshot not vendored (python scripts/fetch_models.py --ocr)")
        origin = store.ocr_dir() / ".origin.json"
        if not origin.is_file():
            pytest.skip("vendored without .origin.json — re-run fetch_models.py --ocr")
        recorded = json.loads(origin.read_text(encoding="utf-8")).get("revision")
        assert recorded == OCR_REVISION, (
            f"bundle is at {recorded} but core.types.OCR_REVISION is {OCR_REVISION}")

    def test_unbundled_variant_is_refused_offline_with_a_fix_command(self):
        """Offline + a variant we did not vendor -> MODEL_NOT_FOUND, not a network try."""
        from animetext_studio.core.config import KNOWN_MODELS
        from animetext_studio.core.types import ErrorCode, StudioError

        store = _bundled_or_skip(TEST_MODEL)
        missing = [n for n in KNOWN_MODELS if not store.has_detection_variant(n)]
        if not missing:
            pytest.skip("every variant is vendored; nothing to refuse")
        eng = DetectionEngine(offline=True)
        with pytest.raises(StudioError) as ei:
            eng.load(missing[0])
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND
        assert any("fetch_models.py" in s for s in ei.value.details["steps"])
