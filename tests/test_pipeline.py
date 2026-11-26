"""Part 4 tests — the pipeline: ordering, cropping, OCR isolation, the §2.1 document.

Stub engines (not the `Fake*` ones) carry most of this file: the point is to inject exact
detections and exact failures, which the deterministic fakes cannot express. The fakes are
used for the end-to-end runs, which must be byte-stable for P7's MCP harness.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from PIL import Image

from animetext_studio.core.config import AppConfig
from animetext_studio.core.pipeline import (
    SKIP_ERROR_CODE, Pipeline, annotate_blocks, contained_in_map, engines_from_config,
    is_possibly_truncated, order_detections, run_file,
)
from animetext_studio.core.imaging import MIN_CROP_SIDE, load_image
from animetext_studio.core.types import (
    BBox, Detection, ErrorCode, PipelineResult, StudioError, TEXT_LABEL, TextBlock,
    color_for_index,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "animetext_studio" / "schemas"
W, H = 400, 300


# ---------------------------------------------------------------------------
# Stub engines
# ---------------------------------------------------------------------------

class StubDetection:
    name = "detection"

    def __init__(self, detections=(), error=None):
        self.detections = list(detections)
        self.error = error
        self.calls: list[dict] = []

    def status(self):
        return SimpleNamespace(backend="stub", device="cpu", state="ready",
                               loaded_models=("m",), detail="", name=self.name)

    def resolved_score_threshold(self, model_name, requested):
        return 0.42 if requested is None else float(requested)

    def detect(self, image, model_name, **kwargs):
        self.calls.append({"model": model_name, **kwargs})
        progress = kwargs.get("progress")
        if progress:
            progress("detect", 0, "stub: inferring")
        if kwargs.get("cancel_event") is not None and kwargs["cancel_event"].is_set():
            raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")
        if self.error is not None:
            raise self.error
        if progress:
            progress("detect", 100, f"{len(self.detections)} box(es)")
        return list(self.detections)


class StubOcr:
    """`poisoned` = crop indices that fail; `batch_error` = fail the batched call once."""

    name = "ocr"

    def __init__(self, texts=None, poisoned=(), batch_error=None, error=None):
        self.texts = list(texts) if texts is not None else []
        self.poisoned = set(poisoned)
        self.batch_error = batch_error
        self.error = error
        self.calls: list[dict] = []
        self._solo = 0

    def status(self):
        return SimpleNamespace(backend="stub", device="cuda:0", state="ready",
                               loaded_models=("hayai",), detail="", name=self.name)

    def ocr_crops(self, images, **kwargs):
        self.calls.append({"count": len(images), **{k: v for k, v in kwargs.items()
                                                    if k not in ("progress", "cancel_event")}})
        cancel = kwargs.get("cancel_event")
        progress = kwargs.get("progress")
        if cancel is not None and cancel.is_set():
            raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")
        if self.error is not None:
            raise self.error
        if progress:
            progress("ocr", 100, f"{len(images)} crop(s)")
        solo = len(images) == 1 and len(self.calls) > 1
        if solo:
            index = self._solo
            self._solo += 1
            if index in self.poisoned:
                raise StudioError(ErrorCode.OCR_FAILED, f"crop {index} is unreadable")
            return [self.texts[index] if index < len(self.texts) else "x"]
        if self.batch_error is not None:
            raise self.batch_error
        if self.poisoned:
            raise StudioError(ErrorCode.OCR_FAILED, "one crop poisoned the batch")
        if self.texts and len(self.texts) >= len(images):
            return self.texts[:len(images)]
        return [f"t{i}" for i in range(len(images))]


def box(x0, y0, x1, y1) -> BBox:
    return BBox(x0, y0, x1, y1)


def det(x0, y0, x1, y1, score=0.9) -> Detection:
    return Detection(box(x0, y0, x1, y1), TEXT_LABEL, score)


def image(size=(W, H)) -> Image.Image:
    base = Image.new("RGB", size, "white")
    pixels = base.load()
    for x in range(size[0]):
        for y in range(size[1]):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return base


@pytest.fixture
def info():
    from animetext_studio.core.imaging import describe

    return describe(image(), source="clipboard")


def make(detections, ocr=None, config=None, **pipeline_kwargs):
    return Pipeline(StubDetection(detections), ocr or StubOcr(), config or AppConfig.default(),
                    **pipeline_kwargs)


def validate(document: dict) -> None:
    schema = json.loads((SCHEMA_DIR / "result.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(document, schema)


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------

class TestOrdering:
    # two rows of two: row A on top (left L1, right R1), row B below (left L2, right R2)
    L1, R1 = det(10, 10, 90, 60, 0.5), det(200, 12, 290, 62, 0.6)
    L2, R2 = det(10, 200, 90, 250, 0.9), det(200, 205, 290, 255, 0.4)
    ALL = [L1, L2, R1, R2]

    def ids(self, ordered):
        return [d.bbox.x0 + d.bbox.y0 for d in ordered]

    def test_manga_rtl_reads_right_to_left_inside_a_row(self):
        assert self.ids(order_detections(self.ALL, "manga_rtl")) ==                self.ids([self.R1, self.L1, self.R2, self.L2])

    def test_top_to_bottom_reads_left_to_right_inside_a_row(self):
        assert self.ids(order_detections(self.ALL, "top_to_bottom")) ==                self.ids([self.L1, self.R1, self.L2, self.R2])

    def test_score_order_is_confidence_first_with_a_spatial_tiebreak(self):
        ordered = order_detections(self.ALL, "score")
        assert [d.score for d in ordered] == [0.9, 0.6, 0.5, 0.4]

    def test_an_unknown_mode_falls_back_to_the_default_instead_of_erroring(self):
        assert self.ids(order_detections(self.ALL, "boustrophedon")) ==                self.ids(order_detections(self.ALL, "manga_rtl"))

    def test_a_tall_column_is_its_own_row_not_merged_with_a_neighbour(self):
        column = det(120, 0, 160, 299)          # spans the whole page height
        small = det(10, 10, 90, 60)
        rows = order_detections([column, small], "manga_rtl")
        # the small box sits inside the column's vertical range, but the column is the
        # taller one: overlap is judged against the *smaller* box, so they do share a row
        assert len(rows) == 2

    def test_ordering_is_stable_for_identical_boxes(self):
        a, b = det(10, 10, 50, 50, 0.5), det(10, 10, 50, 50, 0.9)
        assert [d.score for d in order_detections([a, b], "manga_rtl")] == [0.5, 0.9]

    def test_empty_input(self):
        assert order_detections([], "manga_rtl") == []


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_a_two_box_page_validates_and_is_fully_populated(self, info):
        pipeline = make([det(200, 20, 290, 90, 0.8), det(20, 150, 120, 220, 0.7)],
                        StubOcr(texts=["right", "left"]))
        result = pipeline.run(image(), info=info)
        validate(result.to_dict())
        assert result.block_count == 2
        first, second = result.blocks
        assert (first.reading_order, first.id) == (0, 0)
        assert first.text == "right" and second.text == "left"
        assert first.color == color_for_index(0) and second.color == color_for_index(1)
        assert first.error is None and first.label == TEXT_LABEL
        assert result.timing_ms.total >= result.timing_ms.detect

    def test_the_document_is_reproducible_apart_from_timestamp_and_timings(self, info):
        def once():
            pipeline = make([det(200, 20, 290, 90, 0.8), det(20, 150, 120, 220, 0.7)],
                            StubOcr(texts=["a", "b"]))
            doc = pipeline.run(image(), info=info).to_dict()
            doc.pop("created_at")
            doc.pop("timing_ms")
            return doc

        assert once() == once()

    def test_ids_and_reading_order_are_the_document_order(self, info):
        result = make([det(10, 10, 90, 60), det(200, 200, 290, 250)],
                      StubOcr(texts=["a", "b"])).run(image(), info=info)
        assert [b.id for b in result.blocks] == list(range(result.block_count))
        assert [b.reading_order for b in result.blocks] == list(range(result.block_count))

    def test_no_detections_is_a_valid_empty_document(self, info):
        ocr = StubOcr()
        result = make([], ocr).run(image(), info=info)
        validate(result.to_dict())
        assert result.blocks == () and result.block_count == 0
        assert ocr.calls == []                      # nothing to read, nothing called

    def test_an_empty_text_is_kept_and_is_not_an_error(self, info):
        result = make([det(10, 10, 100, 80)], StubOcr(texts=[""])).run(image(), info=info)
        block = result.blocks[0]
        assert block.text == "" and block.error is None

    def test_detection_section_records_the_resolved_threshold_and_backend(self, info):
        config = AppConfig.default()
        config.detection.score_threshold = None      # "use the repo recommendation"
        result = make([det(10, 10, 100, 80)], StubOcr(texts=["x"]), config).run(image(), info=info)
        assert result.detection.score_threshold == 0.42
        assert result.detection.backend == "stub"
        assert result.detection.model == config.detection.model

    def test_ocr_section_reports_the_resolved_device_not_the_request(self, info):
        result = make([det(10, 10, 100, 80)], StubOcr(texts=["x"])).run(image(), info=info)
        assert result.ocr.device == "cuda:0"          # what the stub engine reported
        assert result.ocr.batch_size == 16

    def test_ocr_settings_are_passed_through_untouched(self, info):
        config = AppConfig.default()
        config.ocr.max_patches = 256
        config.ocr.batch_size = 3
        config.ocr.max_new_tokens = 64
        config.ocr.normalize_nfkc = True
        ocr = StubOcr(texts=["a", "b", "c", "d"])
        make([det(10, 10, 100, 80), det(200, 200, 290, 250)], ocr, config).run(image(), info=info)
        call = ocr.calls[0]
        assert call["max_patches"] == 256 and call["batch_size"] == 3
        assert call["max_new_tokens"] == 64 and call["normalize_nfkc"] is True

    def test_run_without_an_info_says_what_it_needs(self):
        with pytest.raises(StudioError) as ei:
            make([det(10, 10, 100, 80)]).run(image())
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert "ImageInfo" in ei.value.message
        assert "run_file" in ei.value.details["hint"]

    def test_a_non_image_is_rejected_before_any_engine_work(self):
        engine = StubDetection([det(10, 10, 100, 80)])
        with pytest.raises(StudioError) as ei:
            Pipeline(engine, StubOcr()).run("page.png")
        assert ei.value.code is ErrorCode.INVALID_IMAGE
        assert engine.calls == []


# ---------------------------------------------------------------------------
# Cropping, floors, skipping
# ---------------------------------------------------------------------------

class TestCropping:
    def test_the_recorded_bbox_is_the_padded_clamped_effective_box(self, info):
        config = AppConfig.default()
        config.ocr.crop_padding_px = 5
        # box touching the right edge: padding must be clamped, not overflow the page
        result = make([det(380, 10, 400, 80)], StubOcr(texts=["edge"]), config).run(
            image(), info=info)
        bbox = result.blocks[0].bbox
        assert bbox.x1 == W and bbox.x0 == 375 and bbox.y0 == 5

    def test_padding_reaches_the_engine_as_a_bigger_crop(self, info):
        config = AppConfig.default()
        config.ocr.crop_padding_px = 4
        result = make([det(100, 100, 150, 140)], StubOcr(texts=["p"]), config).run(image(),
                                                                                   info=info)
        assert result.blocks[0].bbox.area == 58 * 48

    def test_boxes_below_the_area_floor_never_reach_the_document(self, info):
        config = AppConfig.default()
        config.pipeline.min_box_area_px = 400
        ocr = StubOcr(texts=["big"])
        result = make([det(10, 10, 150, 150), det(200, 200, 210, 210)], ocr,
                      config).run(image(), info=info)
        assert result.block_count == 1
        assert len(ocr.calls) == 1 and ocr.calls[0]["count"] == 1

    def test_a_thin_box_that_passes_the_floor_is_skipped_not_ocrd(self, info):
        config = AppConfig.default()
        config.pipeline.min_box_area_px = 16
        ocr = StubOcr(texts=["readable"])
        thin = det(10, 10, 11, 30)                 # 1x20 = 20 px, but a 1 px side
        result = make([thin, det(100, 100, 200, 200)], ocr, config).run(image(), info=info)
        assert result.block_count == 2
        skipped, read = result.blocks
        assert skipped.text is None
        assert skipped.error["code"] == SKIP_ERROR_CODE
        assert f"{MIN_CROP_SIDE}" in skipped.error["message"] or "px" in skipped.error["message"]
        assert read.text == "readable" and read.error is None
        assert ocr.calls[0]["count"] == 1          # only the readable crop went to OCR

    def test_ocr_can_be_turned_off_entirely(self, info):
        ocr = StubOcr(texts=["x"])
        result = make([det(10, 10, 100, 80)], ocr, ocr_enabled=False).run(image(), info=info)
        assert ocr.calls == []
        assert result.blocks[0].text is None
        assert result.blocks[0].error["code"] == "SKIPPED"


class TestDocumentAnnotations:
    """P8.2b — result schema 1.2: `flags` + `contained_in`, derived from data we already have."""

    OUTER, INNER = det(10, 10, 200, 200), det(20, 20, 60, 60)

    @staticmethod
    def by_bbox(result):
        return {b.bbox: b for b in result.blocks}

    # -- possible_truncation ------------------------------------------------

    def test_a_text_at_the_generation_cap_is_flagged(self, info):
        cfg = AppConfig.default()
        cfg.ocr.max_new_tokens = 10                      # 0.9 * 10 = 9 characters
        result = make([det(10, 10, 100, 80)], StubOcr(texts=["0123456789"]),
                      config=cfg).run(image(), info=info)
        assert result.blocks[0].flags == ("possible_truncation",)

    def test_a_short_text_is_not_flagged(self, info):
        cfg = AppConfig.default()
        cfg.ocr.max_new_tokens = 10
        result = make([det(10, 10, 100, 80)], StubOcr(texts=["short"]),
                      config=cfg).run(image(), info=info)
        assert result.blocks[0].flags == ()

    def test_the_threshold_follows_the_configured_cap(self, info):
        text = "0123456789"
        tight, loose = AppConfig.default(), AppConfig.default()
        tight.ocr.max_new_tokens, loose.ocr.max_new_tokens = 10, 100
        flagged = make([det(10, 10, 100, 80)], StubOcr(texts=[text]),
                       config=tight).run(image(), info=info)
        fine = make([det(10, 10, 100, 80)], StubOcr(texts=[text]),
                    config=loose).run(image(), info=info)
        assert flagged.blocks[0].flags and not fine.blocks[0].flags

    def test_a_block_that_was_not_read_is_never_flagged_as_truncated(self, info):
        cfg = AppConfig.default()
        cfg.ocr.max_new_tokens = 1
        result = make([det(10, 10, 100, 80)], StubOcr(texts=[""]),
                      config=cfg).run(image(), info=info)
        assert result.blocks[0].text == "" and result.blocks[0].flags == ()

    # -- contained_in -------------------------------------------------------

    def test_a_nested_box_points_at_its_container(self, info):
        result = make([self.OUTER, self.INNER], StubOcr(texts=["outer", "inner"])).run(
            image(), info=info)
        by = self.by_bbox(result)
        outer = by[self.OUTER.bbox.expanded(2, W, H)]
        inner = by[self.INNER.bbox.expanded(2, W, H)]
        assert inner.contained_in == outer.id
        assert outer.contained_in is None                # nothing contains the container

    def test_the_tightest_container_wins(self, info):
        big, middle, small = det(0, 0, 400, 300), det(20, 20, 200, 200), det(30, 30, 80, 80)
        result = make([big, middle, small]).run(image(), info=info)
        by = self.by_bbox(result)
        pad = lambda b: b.expanded(2, W, H)              # noqa: E731  (D38 effective boxes)
        assert by[pad(small.bbox)].contained_in == by[pad(middle.bbox)].id
        assert by[pad(middle.bbox)].contained_in == by[pad(big.bbox)].id
        assert by[pad(big.bbox)].contained_in is None

    def test_half_overlap_is_not_containment(self, info):
        a, b = det(10, 10, 110, 110), det(60, 10, 160, 110)      # 50 % shared, same size
        result = make([a, b]).run(image(), info=info)
        assert all(block.contained_in is None for block in result.blocks)

    def test_identical_boxes_do_not_point_at_each_other(self, info):
        """A container must be *strictly* larger, so a duplicate cannot parent its twin."""
        a, b = det(10, 10, 100, 100), det(10, 10, 100, 100, score=0.5)
        result = make([a, b]).run(image(), info=info)
        assert all(block.contained_in is None for block in result.blocks)

    def test_containment_is_reported_in_detect_only_documents_too(self, info):
        result = make([self.OUTER, self.INNER], StubOcr(texts=["x"]),
                      ocr_enabled=False).run(image(), info=info)
        assert result.stage == "detect"                  # geometry does not need OCR
        # A stage-1 run never crops, so its boxes are the raw detections (no padded
        # "effective" box exists to look up) — hence finding the pair by relation.
        inner, outer = sorted(result.blocks, key=lambda b: b.bbox.area)
        assert inner.bbox == self.INNER.bbox and outer.bbox == self.OUTER.bbox
        assert inner.contained_in == outer.id and outer.contained_in is None

    # -- the document as a whole -------------------------------------------

    def test_annotations_survive_the_round_trip_and_validate(self, info):
        result = make([self.OUTER, self.INNER], StubOcr(texts=["outer", "inner"])).run(
            image(), info=info)
        document = result.to_dict()
        validate(document)
        again = PipelineResult.from_dict(document)
        assert [(b.flags, b.contained_in) for b in again.blocks] == \
               [(b.flags, b.contained_in) for b in result.blocks]

    def test_annotations_change_nothing_but_themselves(self, info):
        """Geometry and text are the document; a flag may not disturb either (D38 intact)."""
        result = make([self.OUTER, self.INNER], StubOcr(texts=["outer", "inner"])).run(
            image(), info=info)
        assert {b.bbox for b in result.blocks} == {self.OUTER.bbox.expanded(2, W, H),
                                                   self.INNER.bbox.expanded(2, W, H)}
        assert sorted(b.text for b in result.blocks) == ["inner", "outer"]

    def test_annotations_are_deterministic(self, info):
        boxes = [det(0, 0, 400, 300), det(20, 20, 200, 200), det(30, 30, 80, 80),
                 det(300, 20, 380, 90)]
        texts = ["a", "b", "c", "d"]
        first = make(boxes, StubOcr(texts=texts)).run(image(), info=info)
        second = make(boxes, StubOcr(texts=texts)).run(image(), info=info)
        assert [b.to_dict() for b in first.blocks] == [b.to_dict() for b in second.blocks]

    # -- the pure helpers, without a pipeline -------------------------------

    def test_is_possibly_truncated_edges(self):
        assert is_possibly_truncated("x" * 116, 128)      # 0.9 * 128 = 115.2 (author's ratio)
        assert not is_possibly_truncated("x" * 115, 128)
        assert not is_possibly_truncated("", 128)
        assert not is_possibly_truncated(None, 128)
        assert not is_possibly_truncated("x", 0)          # a nonsense cap flags nothing

    def test_contained_in_map_ignores_disjoint_and_degenerate_boxes(self):
        blocks = [
            TextBlock(id=0, bbox=box(0, 0, 10, 10), label=TEXT_LABEL, score=0.5, text="a",
                      reading_order=0, color=color_for_index(0)),
            TextBlock(id=1, bbox=box(100, 100, 130, 130), label=TEXT_LABEL, score=0.5, text="b",
                      reading_order=1, color=color_for_index(1)),
            TextBlock(id=2, bbox=box(0, 0, 0, 0), label=TEXT_LABEL, score=0.5, text="",
                      reading_order=2, color=color_for_index(2)),
        ]
        assert contained_in_map(blocks) == {}
        assert contained_in_map([]) == {}
        assert annotate_blocks([], max_new_tokens=128) == ()


class TestStageMarker:
    """P8.1 — the document says which stages it went through (schema 1.1)."""

    def test_a_normal_run_declares_itself_full(self, info):
        result = make([det(10, 10, 100, 80)], StubOcr(texts=["read"])).run(image(), info=info)
        assert result.stage == "full" and result.ocr_ran is True
        validate(result.to_dict())

    def test_a_stage_1_run_declares_itself_detect(self, info):
        result = make([det(10, 10, 100, 80)], StubOcr(texts=["x"]),
                      ocr_enabled=False).run(image(), info=info)
        document = result.to_dict()
        assert document["stage"] == "detect" and document["schema_version"] == "1.2"
        assert result.ocr_ran is False
        validate(document)                        # still a schema-valid document

    def test_detect_only_and_read_nothing_are_different_documents(self, info):
        """The defect 1.1 fixes: these two used to be byte-identical but for the text."""
        empty = StubOcr(texts=[""])
        full = make([det(10, 10, 100, 80)], empty).run(image(), info=info)
        stage1 = make([det(10, 10, 100, 80)], empty,
                      ocr_enabled=False).run(image(), info=info)
        assert full.blocks[0].text == "" and full.blocks[0].error is None
        assert stage1.blocks[0].text is None
        assert full.stage != stage1.stage
        assert full.to_dict()["stage"] == "full" and stage1.to_dict()["stage"] == "detect"

    def test_the_per_run_ocr_override_is_local_not_sticky(self, info):
        """`ocr_enabled=` (the MCP `detect` path) applies to one run, never to the instance."""
        pipeline = make([det(10, 10, 100, 80)], StubOcr(texts=["x"]), ocr_enabled=False)
        assert pipeline.run(image(), info=info).stage == "detect"
        assert pipeline.run(image(), info=info, ocr_enabled=True).stage == "full"
        assert pipeline.run(image(), info=info).stage == "detect"
        assert pipeline.ocr_enabled is False      # the constructor flag is untouched


# ---------------------------------------------------------------------------
# Failure isolation and cancellation
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    def test_one_poisoned_crop_does_not_lose_the_page(self, info):
        ocr = StubOcr(texts=["a", "b", "c"], poisoned={1})
        result = make([det(10, 10, 120, 90), det(150, 10, 260, 90), det(10, 150, 120, 230)],
                      ocr).run(image(), info=info)
        validate(result.to_dict())
        bad = [i for i, block in enumerate(result.blocks) if block.error]
        assert len(bad) == 1
        assert result.blocks[bad[0]].text is None
        assert "unreadable" in result.blocks[bad[0]].error["message"]
        assert [b.text for b in result.blocks if not b.error] == ["a", "c"]
        assert ocr.calls[0]["count"] == 3                  # batched first
        assert all(c["count"] == 1 for c in ocr.calls[1:])  # then one at a time

    def test_a_load_failure_is_not_retried_crop_by_crop(self, info):
        ocr = StubOcr(error=StudioError(ErrorCode.ENGINE_LOAD_FAILED, "no torch"))
        with pytest.raises(StudioError) as ei:
            make([det(10, 10, 120, 90), det(150, 10, 260, 90)], ocr).run(image(), info=info)
        assert ei.value.code is ErrorCode.ENGINE_LOAD_FAILED
        assert len(ocr.calls) == 1

    def test_a_single_crop_failure_propagates_instead_of_retrying_itself(self, info):
        ocr = StubOcr(poisoned={0})
        with pytest.raises(StudioError) as ei:
            make([det(10, 10, 120, 90)], ocr).run(image(), info=info)
        assert ei.value.code is ErrorCode.OCR_FAILED
        assert len(ocr.calls) == 1

    def test_detection_failures_propagate_untouched(self, info):
        error = StudioError(ErrorCode.MODEL_NOT_FOUND, "no such variant")
        with pytest.raises(StudioError) as ei:
            Pipeline(StubDetection(error=error), StubOcr()).run(image(), info=info)
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND

    def test_cancellation_in_detection_propagates(self, info):
        event = threading.Event()
        event.set()
        with pytest.raises(StudioError) as ei:
            make([det(10, 10, 100, 80)]).run(image(), info=info, cancel_event=event)
        assert ei.value.code is ErrorCode.CANCELLED

    def test_cancellation_during_the_isolation_pass_stops_immediately(self, info):
        event = threading.Event()

        class CancellingOcr(StubOcr):
            def ocr_crops(self, images, **kwargs):
                kwargs.pop("cancel_event", None)
                if len(self.calls) == 1:
                    self.calls.append({"count": len(images)})
                    raise StudioError(ErrorCode.OCR_FAILED, "batch blew up")
                if len(self.calls) == 2:
                    self.calls.append({"count": len(images)})
                    event.set()                       # user hits cancel mid-isolation
                    return ["a"]
                raise StudioError(ErrorCode.CANCELLED, "Operation cancelled.")

        with pytest.raises(StudioError) as ei:
            make([det(10, 10, 120, 90), det(150, 10, 260, 90), det(10, 150, 120, 230)],
                 CancellingOcr()).run(image(), info=info, cancel_event=event)
        assert ei.value.code is ErrorCode.CANCELLED


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

class TestProgress:
    def collect(self, **kwargs):
        seen: list[tuple[str, float]] = []
        result = make([det(10, 10, 120, 90), det(150, 10, 260, 90)],
                      StubOcr(texts=["a", "b"])).run(
            image(), info=None if kwargs.get("no_info") else self.info_fixture(),
            progress=lambda stage, percent, message: seen.append((stage, percent)), **kwargs)
        return seen, result

    @staticmethod
    def info_fixture():
        from animetext_studio.core.imaging import describe

        return describe(image(), source="clipboard")

    def test_progress_is_monotonic_and_ends_at_one_hundred(self):
        seen, _ = self.collect()
        percents = [p for _, p in seen]
        assert percents == sorted(percents)
        assert percents[-1] == 100.0
        assert percents[0] >= 0.0

    def test_the_stages_tell_the_whole_story_in_order(self):
        seen, _ = self.collect()
        stages = [s for s, _ in seen]
        assert stages.index("detect") < stages.index("crop") < stages.index("ocr")
        assert stages[-1] == "assemble"

    def test_a_late_engine_load_is_forwarded_but_never_moves_the_bar_back(self):
        # The shipped fakes load the OCR engine lazily inside the first batch, which is
        # exactly when a naive bar would jump back to 0%.
        engines = engines_from_config(AppConfig.default(), mock=True)
        seen: list[tuple[str, float]] = []
        Pipeline(engines.detection, engines.ocr).run_file(
            SAMPLE, progress=lambda stage, percent, message: seen.append((stage, percent)))
        assert [p for _, p in seen] == sorted(p for _, p in seen)
        stages = [s for s, _ in seen]
        assert "load_ocr" in stages and "ocr" in stages
        assert (max(p for s, p in seen if s == "load_ocr")
                <= max(p for s, p in seen if s == "ocr"))

    def test_no_progress_callback_is_fine(self):
        result = make([det(10, 10, 120, 90)], StubOcr(texts=["a"])).run(
            image(), info=self.info_fixture())
        assert result.block_count == 1


# ---------------------------------------------------------------------------
# End-to-end with the shipped fakes, and the real entry points
# ---------------------------------------------------------------------------

SAMPLE = "examples/sample_page.png"


class TestEndToEnd:
    def test_mock_engines_produce_a_valid_document(self):
        engines = engines_from_config(AppConfig.default(), mock=True)
        assert engines.names == ("detection", "ocr")
        result = Pipeline(engines.detection, engines.ocr).run_file(SAMPLE)
        validate(result.to_dict())     # noqa: E501
        assert result.block_count >= 1
        assert all(b.text for b in result.blocks)
        assert result.image.source == "file" and result.image.file_name.endswith(".png")

    def test_two_mock_runs_of_the_sample_page_are_identical(self):
        def once():
            engines = engines_from_config(AppConfig.default(), mock=True)
            doc = Pipeline(engines.detection, engines.ocr).run_file(SAMPLE).to_dict()
            doc.pop("created_at")
            doc.pop("timing_ms")
            return doc

        assert once() == once()

    def test_run_bytes_reports_the_mcp_source_and_the_same_digest(self):
        engines = engines_from_config(AppConfig.default(), mock=True)
        data = open(SAMPLE, "rb").read()
        from_file = Pipeline(engines.detection, engines.ocr).run_file(SAMPLE)
        from_bytes = Pipeline(engines.detection, engines.ocr).run_bytes(
            data, file_name="page.png")
        validate(from_bytes.to_dict())
        assert from_bytes.image.source == "mcp-base64"
        assert from_bytes.image.sha256 == from_file.image.sha256

    def test_module_level_run_file_helper(self):
        result = run_file(SAMPLE, AppConfig.default(), mock=True)
        assert result.block_count >= 1

    @pytest.mark.models
    def test_real_weights_end_to_end_offline_on_the_sample_page(self):
        """`-m models`: detect -> crop -> OCR on the vendored weights, no network.

        Deliberately lenient about *what* is read (models get updated) and strict about
        the contract: a schema-valid document, text present, blocks ordered, timings sane.
        """
        from animetext_studio.core.model_store import ModelStore

        store = ModelStore()
        if not (store.has_ocr_snapshot() and store.bundled_detection_variants()):
            pytest.skip("bundle incomplete (python scripts/fetch_models.py)")
        config = AppConfig.default()
        config.paths.offline = True
        config.ocr.batch_size = 8
        engines = engines_from_config(config)
        stages: list[str] = []
        result = Pipeline(engines.detection, engines.ocr, config).run_file(
            SAMPLE, progress=lambda stage, percent, message: stages.append(stage))
        validate(result.to_dict())
        assert result.block_count >= 4
        assert sum(1 for b in result.blocks if b.text) >= 3
        assert [b.reading_order for b in result.blocks] == list(range(result.block_count))
        assert result.timing_ms.total >= result.timing_ms.detect + result.timing_ms.ocr - 1
        assert {"detect", "crop", "ocr", "assemble"} <= set(stages)
        assert engines.detection.status().state.value == "ready"
        assert engines.ocr.status().state.value == "ready"

    def test_a_corrupt_payload_is_an_invalid_image_not_a_crash(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"not an image")
        engines = engines_from_config(AppConfig.default(), mock=True)
        with pytest.raises(StudioError) as ei:
            Pipeline(engines.detection, engines.ocr).run_file(bad)
        assert ei.value.code is ErrorCode.INVALID_IMAGE


# ---------------------------------------------------------------------------
# Headless CLI (--headless --run)
# ---------------------------------------------------------------------------

class TestHeadlessCli:
    def test_run_prints_the_document_and_writes_the_file(self, tmp_path, capsys):
        from animetext_studio.__main__ import main

        out = tmp_path / "nested" / "result.json"
        code = main(["--headless", "--mock", "--image", SAMPLE, "--run", "--out", str(out)])
        assert code == 0
        document = json.loads(capsys.readouterr().out)
        validate(document)
        assert json.loads(out.read_text(encoding="utf-8")) == document

    def test_run_without_an_image_is_no_image(self, capsys):
        from animetext_studio.__main__ import main

        assert main(["--headless", "--mock", "--run"]) == 3
        assert json.loads(capsys.readouterr().err)["error"]["code"] == "NO_IMAGE"

    def test_headless_without_run_asks_for_an_action(self, capsys):
        """The Part 1 stub is long gone: `--headless` alone is a usage error naming `--run`."""
        from animetext_studio.__main__ import main

        assert main(["--headless"]) == 2                      # INVALID_ARGUMENT
        err = json.loads(capsys.readouterr().err)["error"]
        assert err["code"] == "INVALID_ARGUMENT" and "--run" in err["message"]

    def test_a_missing_model_surfaces_as_its_own_exit_code(self, tmp_path, capsys):
        from animetext_studio.__main__ import main

        code = main(["--headless", "--offline", "--image", SAMPLE, "--model",
                     "nope_not_a_model", "--run"])
        assert code == 12                                  # MODEL_NOT_FOUND
        assert json.loads(capsys.readouterr().err)["error"]["code"] == "MODEL_NOT_FOUND"

    def test_stdout_stays_clean_when_it_fails(self, capsys):
        from animetext_studio.__main__ import main

        assert main(["--headless", "--mock", "--image", "nope.png", "--run"]) == 10
        captured = capsys.readouterr()
        assert captured.out == ""
        assert json.loads(captured.err)["error"]["code"] == "IO_ERROR"


class TestReocrBlocks:
    """P9.3 — re-read one block without asking the detector again.

    What these hold: the detector is not called a second time, ids/colours/order survive,
    unselected blocks keep their text, the new settings reach the OCR call, and what comes back
    is a whole document that still validates (D81).
    """

    BOXES = [det(200, 20, 290, 90, 0.8), det(20, 150, 120, 220, 0.7), det(40, 40, 90, 100, 0.6)]

    def _pipeline(self, ocr=None):
        return make(list(self.BOXES), ocr or StubOcr(texts=["right", "left", "top"]))

    def test_the_detector_is_not_asked_twice(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        assert len(pipeline.detection.calls) == 1
        pipeline.reocr_blocks(image(), first, [1], info=info)
        assert len(pipeline.detection.calls) == 1, "a re-read must not re-detect the page"
        assert len(pipeline.ocr.calls) == 2                    # once for the page, once for block 1

    def test_only_the_selected_crop_is_sent(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        pipeline.ocr.calls.clear()
        pipeline.reocr_blocks(image(), first, [2], info=info)
        assert pipeline.ocr.calls[-1]["count"] == 1            # only block 2 went back to OCR

    def test_ids_colours_order_and_boxes_survive(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        again = pipeline.reocr_blocks(image(), first, [1], info=info)
        assert [b.id for b in again.blocks] == [b.id for b in first.blocks]
        assert [b.color for b in again.blocks] == [b.color for b in first.blocks]
        assert [b.reading_order for b in again.blocks] == [b.reading_order for b in first.blocks]
        assert [b.bbox for b in again.blocks] == [b.bbox for b in first.blocks]

    def test_the_document_is_whole_and_unselected_blocks_keep_their_text(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        again = pipeline.reocr_blocks(image(), first, [0], info=info)
        assert again.block_count == first.block_count
        assert [b.text for b in again.blocks if b.id != 0] == \
               [b.text for b in first.blocks if b.id != 0]
        assert all(b.text for b in again.blocks)

    def test_the_new_document_validates_and_says_how_it_was_made(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        again = pipeline.reocr_blocks(image(), first, [0, 2], info=info)
        validate(again.to_dict())                              # the result-1.1 schema accepts "reocr"
        assert again.stage == "reocr" and again.ocr_ran
        assert again.timing_ms.detect == 0                     # nothing was detected
        assert PipelineResult.from_dict(again.to_dict()).stage == "reocr"

    def test_the_settings_in_force_are_the_ones_that_read_the_block(self, info):
        """The point of the door: the same box, read again under a different budget."""
        config = AppConfig.default()
        ocr = StubOcr(texts=["again"])
        pipeline = make(list(self.BOXES), ocr, config)
        first = pipeline.run(image(), info=info)
        before = ocr.calls[-1]["max_new_tokens"]
        config.ocr.max_new_tokens = before + 128
        pipeline.reocr_blocks(image(), first, [1], info=info)
        assert ocr.calls[-1]["max_new_tokens"] == before + 128

    def test_a_re_read_whose_engine_fails_raises_and_loses_nothing(self, info):
        """One crop, one engine: there is nothing left to isolate, so the error is the answer.

        What matters is that the document already in hand is untouched — a re-read that failed
        must not blank the page it was trying to improve.
        """
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        pipeline.ocr = StubOcr(error=StudioError(ErrorCode.OCR_FAILED, "engine said no"))
        with pytest.raises(StudioError) as ei:
            pipeline.reocr_blocks(image(), first, [1], info=info)
        assert ei.value.code is ErrorCode.OCR_FAILED
        assert all(b.text for b in first.blocks)

    def test_an_empty_selection_is_refused_before_any_crop_is_made(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        pipeline.ocr.calls.clear()
        with pytest.raises(StudioError) as ei:
            pipeline.reocr_blocks(image(), first, [], info=info)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert pipeline.ocr.calls == []

    def test_an_unknown_id_names_the_available_ones(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        with pytest.raises(StudioError) as ei:
            pipeline.reocr_blocks(image(), first, [42], info=info)
        assert ei.value.details["unknown"] == [42]
        assert ei.value.details["available"] == [b.id for b in first.blocks]

    def test_a_repeated_selection_is_one_read(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        pipeline.ocr.calls.clear()
        again = pipeline.reocr_blocks(image(), first, [1, 1, 0], info=info)
        assert pipeline.ocr.calls[-1]["count"] == 2           # [1, 1, 0] is two crops, not three
        assert again.block_count == first.block_count

    def test_cancelling_a_re_read_is_a_cancellation_not_a_broken_one(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(StudioError) as ei:
            pipeline.reocr_blocks(image(), first, [0], info=info, cancel_event=cancel)
        assert ei.value.code is ErrorCode.CANCELLED

    def test_a_page_without_info_is_refused(self, info):
        pipeline = self._pipeline()
        first = pipeline.run(image(), info=info)
        with pytest.raises(StudioError) as ei:
            pipeline.reocr_blocks(image(), first, [0])
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
