"""Part 2 tests — engine scaffolding: status, helpers, and the deterministic fakes.

The fake detection engine is what the GUI (P5/P6), pipeline (P4) and MCP server (P7)
run against, so its determinism and threshold semantics are contractual.
"""
from __future__ import annotations

import threading

import pytest
from PIL import Image

from animetext_studio.core.config import KNOWN_MODELS, MODEL_PRESETS
from animetext_studio.core.types import (
    BBox, ErrorCode, StudioError, TEXT_LABEL,
)
from animetext_studio.engines.base import (
    EngineState, EngineStatus, FakeDetectionEngine, _iou, _nms, check_cancel,
    engines_badge_text, model_catalog, models_badge_text, report,
)

W, H = 800, 1200


@pytest.fixture
def image():
    return Image.new("RGB", (W, H), "white")


@pytest.fixture
def engine():
    return FakeDetectionEngine()


class TestHelpers:
    def test_report_clamps_and_skips_none(self):
        seen = []
        collect = lambda stage, percent, message: seen.append((stage, percent, message))
        report(None, "stage", 50, "ignored")  # must not raise
        report(collect, "stage", -10, "lo")
        report(collect, "stage", 400, "hi")
        assert [c[1] for c in seen] == [0.0, 100.0]

    def test_check_cancel(self):
        check_cancel(None)
        check_cancel(threading.Event())
        ev = threading.Event()
        ev.set()
        with pytest.raises(StudioError) as ei:
            check_cancel(ev)
        assert ei.value.code is ErrorCode.CANCELLED

    def test_iou_and_nms(self):
        a, b = BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)   # IoU = 1/3
        assert _iou(a, b) == pytest.approx(1 / 3)
        assert _iou(a, BBox(100, 100, 110, 110)) == 0.0
        assert _iou(BBox(0, 0, 0, 0), BBox(0, 0, 0, 0)) == 0.0

        from animetext_studio.core.types import Detection
        dets = [Detection(a, TEXT_LABEL, 0.5), Detection(b, TEXT_LABEL, 0.9)]
        assert [d.score for d in _nms(dets, 0.2)] == [0.9]          # suppressed
        assert [d.score for d in _nms(dets, 0.5)] == [0.9, 0.5]     # kept

    def test_model_catalog_uses_presets(self):
        cat = model_catalog(["yolo12m_animetext", "mystery_model"])
        assert cat[0]["known"] is True
        assert cat[0]["recommended_score_threshold"] == 0.299
        assert cat[0]["params_m"] == 20.1
        assert cat[1] == {"name": "mystery_model", "known": False, "params_m": None,
                          "flops_g": None, "f1": None, "recommended_score_threshold": None,
                          "source": "hf"}

    def test_model_catalog_reports_weight_source(self):
        """D26: every catalog record says whether the weights are on premises."""
        cat = model_catalog(["yolo12m_animetext", "yolo12n_animetext"],
                            sources={"yolo12m_animetext": "bundled"})
        assert {c["name"]: c["source"] for c in cat} == {
            "yolo12m_animetext": "bundled", "yolo12n_animetext": "hf"}


class TestEnginesBadge:
    """P8: `unload_engines` has to be *visible*, and the wording is decided here (Qt-free)."""

    @staticmethod
    def _status(name="detection", state=EngineState.UNLOADED, models=(), device="cpu"):
        return EngineStatus(name=name, state=state, loaded_models=tuple(models),
                            backend="stub", device=device)

    def test_both_unloaded_says_unloaded(self):
        assert engines_badge_text(self._status(), self._status("ocr")) == "engines: unloaded"

    def test_ready_engines_name_the_model(self):
        det = self._status(state=EngineState.READY, models=("yolo12m_animetext",))
        ocr = self._status("ocr", state=EngineState.READY)
        assert engines_badge_text(det, ocr) == "engines: det yolo12m · ocr ready"

    def test_only_one_engine_ready_is_still_shown(self):
        det = self._status(state=EngineState.READY, models=("yolo12n_animetext",))
        assert engines_badge_text(det, self._status("ocr")) == "engines: det yolo12n"

    def test_loading_and_error_are_distinguishable(self):
        det = self._status(state=EngineState.LOADING)
        ocr = self._status("ocr", state=EngineState.ERROR)
        assert engines_badge_text(det, ocr) == "engines: det loading · ocr error"

    def test_several_loaded_models_are_counted(self):
        det = self._status(state=EngineState.READY,
                           models=("yolo12n_animetext", "yolo12m_animetext"))
        assert engines_badge_text(det, self._status("ocr")) == "engines: det 2x"

    def test_a_real_run_then_unload_flips_the_text(self, image):
        engine = FakeDetectionEngine()
        engine.load("yolo12m_animetext")
        assert engines_badge_text(engine.status(),
                                  self._status("ocr")) == "engines: det yolo12m"
        engine.unload()
        assert engines_badge_text(engine.status(),
                                  self._status("ocr")) == "engines: unloaded"


class TestFakeDetectionEngine:
    def test_status_shape(self, engine):
        status = engine.status()
        assert status.name == "detection" and status.state is EngineState.UNLOADED
        assert status.backend == "fake" and status.device == "cpu"
        assert status.to_dict()["state"] == "unloaded"

    def test_list_models_is_the_known_catalog(self, engine):
        assert [m["name"] for m in engine.list_models()] == list(KNOWN_MODELS)

    def test_load_sets_ready_and_rejects_unknown(self, engine):
        events = []
        engine.load("yolo12s_animetext", progress=lambda *a: events.append(a))  # noqa: E731
        assert engine.status().state is EngineState.READY
        assert engine.status().loaded_models == ("yolo12s_animetext",)
        assert events[0][1] == 0.0 and events[-1][1] == 100.0
        with pytest.raises(StudioError) as ei:
            engine.load("nope")
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND
        assert set(ei.value.details["known_models"]) == set(KNOWN_MODELS)

    def test_detect_deterministic(self, engine, image):
        a = engine.detect(image, "yolo12m_animetext")
        b = engine.detect(Image.new("RGB", (W, H), "black"), "yolo12m_animetext")
        assert [d.to_dict() for d in a] == [d.to_dict() for d in b]
        assert all(d.label == TEXT_LABEL for d in a)
        assert all(d.bbox.clamped(W, H) == d.bbox for d in a)
        assert [d.score for d in a] == sorted([d.score for d in a], reverse=True)

    def test_detect_scales_with_image_size(self, engine):
        small = [d.bbox.as_tuple() for d in engine.detect(Image.new("RGB", (400, 600)), "yolo12m_animetext")]
        large = [d.bbox.as_tuple() for d in engine.detect(Image.new("RGB", (800, 1200)), "yolo12m_animetext")]
        assert [tuple(v * 2 for v in box) for box in small] == large

    def test_score_threshold_filters(self, engine, image):
        assert len(engine.detect(image, "yolo12m_animetext", score_threshold=0.05)) == 5
        assert [d.score for d in engine.detect(image, "yolo12m_animetext", score_threshold=0.87)] == [0.93, 0.88]
        assert engine.detect(image, "yolo12m_animetext", score_threshold=0.94) == []
        # None means "engine default" -> same as the 0.05 demo default for the fake
        assert len(engine.detect(image, "yolo12m_animetext", score_threshold=None)) == 5

    def test_iou_threshold_suppresses_the_overlapping_pair(self, engine, image):
        demo = engine.detect(image, "yolo12m_animetext", iou_threshold=0.35)
        loose = engine.detect(image, "yolo12m_animetext", iou_threshold=0.9)
        assert len(demo) == 5 and len(loose) == 6
        # the fake layout's first two boxes overlap with IoU ~= 0.67
        assert len(loose) - len(demo) == 1

    def test_detect_validates_inputs(self, engine, image):
        with pytest.raises(StudioError) as ei:
            engine.detect("not-an-image", "yolo12m_animetext")  # type: ignore[arg-type]
        assert ei.value.code is ErrorCode.INVALID_IMAGE
        with pytest.raises(StudioError) as ei:
            engine.detect(image, "yolo99_animetext")
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND
        ev = threading.Event()
        ev.set()
        with pytest.raises(StudioError) as ei:
            engine.detect(image, "yolo12m_animetext", cancel_event=ev)
        assert ei.value.code is ErrorCode.CANCELLED

    def test_unload_resets_state(self, engine, image):
        engine.detect(image, "yolo12m_animetext")
        assert engine.status().loaded_models
        engine.unload()
        assert engine.status().state is EngineState.UNLOADED
        assert engine.status().loaded_models == ()

    def test_threshold_resolution_contract(self, engine):
        """P4/P7 record a concrete score_threshold even when the caller passed None."""
        assert engine.resolved_score_threshold("yolo12m_animetext", 0.13) == 0.13
        assert engine.resolved_score_threshold("yolo12m_animetext", None) == 0.05  # fake constant
        assert engine.recommended_threshold("yolo12x_animetext") == 0.425
        assert engine.recommended_threshold("unknown_model") is None
        # base-class rule (used by the real engine when the repo has no threshold.json)
        assert engine.FALLBACK_SCORE_THRESHOLD == 0.25

    def test_all_presets_are_loadable(self, engine, image):
        for name in KNOWN_MODELS:
            assert engine.detect(image, name)
        assert set(engine.status().loaded_models) == set(KNOWN_MODELS)
        assert {m["name"] for m in engine.list_models()} == set(MODEL_PRESETS)


class TestModelsBadge:
    """P9.2: a fallback variant list must be visible, not merely recorded."""

    @staticmethod
    def _status(fallback=False, detail=""):
        return EngineStatus(name="detection", state=EngineState.READY, loaded_models=("yolo11m",),
                            backend="stub", device="cpu", detail=detail,
                            model_list_is_fallback=fallback)

    def test_a_live_list_shows_nothing(self):
        assert models_badge_text(self._status()) == ""

    def test_a_fallback_list_says_presets(self):
        assert models_badge_text(self._status(fallback=True)) == "models: presets"

    def test_the_flag_is_on_the_wire_too(self):
        """`to_dict` carries it, so `--model-status` and the MCP answer from the same field."""
        assert self._status(fallback=True).to_dict()["model_list_is_fallback"] is True
        assert self._status().to_dict()["model_list_is_fallback"] is False

    def test_an_engine_that_never_lists_answers_false(self):
        """OCR has no variant listing; the default must not read as a degraded one."""
        assert models_badge_text(EngineStatus(name="ocr", state=EngineState.READY,
                                              loaded_models=(), backend="stub", device="cpu")) == ""
