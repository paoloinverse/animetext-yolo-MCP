"""Part 1 tests: domain types, geometry, errors, result contract."""
from __future__ import annotations

import dataclasses
import json
import re

import pytest

from animetext_studio.core.types import (
    BLOCK_FLAGS, BLOCK_PALETTE, CONTAINMENT_RATIO, TEXT_LABEL, TRUNCATION_RATIO, BBox,
    Detection, ErrorCode, EXIT_CODES, ImageInfo, JobProgress, JobState, PipelineResult,
    StudioError, TextBlock, color_for_index, sha256_bytes,
)


class TestBBox:
    def test_int_coercion(self):
        b = BBox(10.4, 20.6, 110.5, 220.1)
        assert b.as_tuple() == (10, 21, 110, 220)  # round-half-even on .5

    def test_geometry(self):
        b = BBox(10, 20, 110, 220)
        assert (b.width, b.height, b.area) == (100, 200, 20000)
        assert b.is_valid()
        assert not BBox(10, 20, 10, 220).is_valid()
        assert BBox(10, 20, 5, 220).area == 0

    def test_clamped(self):
        b = BBox(-5, -10, 1300, 1800).clamped(1200, 1700)
        assert b.as_tuple() == (0, 0, 1200, 1700)

    def test_expanded_clamps_to_image(self):
        b = BBox(2, 3, 100, 200).expanded(5, 1200, 1700)
        assert b.as_tuple() == (0, 0, 105, 205)
        b2 = BBox(1190, 1690, 1200, 1700).expanded(8, 1200, 1700)
        assert b2.as_tuple() == (1182, 1682, 1200, 1700)

    def test_roundtrips(self):
        b = BBox(1, 2, 3, 4)
        assert BBox.from_dict(b.to_dict()) == b
        assert BBox.from_tuple(b.as_tuple()) == b


class TestDetectionAndBlock:
    def test_detection_roundtrip(self):
        d = Detection(BBox(1, 2, 30, 40), "text_block", 0.87)
        assert Detection.from_dict(json.loads(json.dumps(d.to_dict()))) == d

    def test_block_roundtrip_with_error(self):
        b = TextBlock(id=3, bbox=BBox(0, 0, 10, 10), label="text_block", score=0.5,
                      text=None, reading_order=3, color="#E6194B",
                      error={"code": "OCR_FAILED", "message": "x"})
        assert TextBlock.from_dict(json.loads(json.dumps(b.to_dict()))) == b

    def test_block_roundtrip_with_cjk(self):
        b = TextBlock(id=0, bbox=BBox(0, 0, 10, 10), label="text_block", score=0.9,
                      text="……なに!? 中文 한국어", reading_order=0, color="#3CB44B")
        restored = TextBlock.from_dict(json.loads(json.dumps(b.to_dict(), ensure_ascii=False)))
        assert restored == b


class TestErrors:
    def test_roundtrip(self):
        e = StudioError(ErrorCode.BUSY, "job running", {"job_id": "j1"})
        assert StudioError.from_dict(e.to_dict()).to_dict() == e.to_dict()
        assert str(e) == "[BUSY] job running"

    def test_no_details_key_when_none(self):
        assert "details" not in StudioError(ErrorCode.NO_IMAGE, "no image").to_dict()

    def test_every_code_has_exit_code(self):
        assert set(EXIT_CODES) == set(ErrorCode)
        assert all(isinstance(v, int) and v > 0 for v in EXIT_CODES.values())

    def test_unknown_code_rejected(self):
        with pytest.raises(ValueError):
            StudioError("NOT_A_CODE", "x")


class TestResult:
    def test_roundtrip(self, sample_result: PipelineResult):
        data = json.loads(sample_result.to_json())
        assert data["block_count"] == 2
        assert PipelineResult.from_dict(data).to_dict() == sample_result.to_dict()

    def test_cjk_not_escaped(self, sample_result: PipelineResult):
        assert "なに" in sample_result.to_json()

    def test_created_at_autofilled(self, sample_result: PipelineResult):
        r = PipelineResult(image=sample_result.image, detection=sample_result.detection,
                           ocr=sample_result.ocr, timing_ms=sample_result.timing_ms)
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", r.created_at)
        assert r.block_count == 0


class TestResultStage:
    """P8.1 — result schema 1.1: a detect-only document must not look like an empty page."""

    def test_a_new_document_is_full_by_default(self, sample_result: PipelineResult):
        assert sample_result.stage == "full" and sample_result.ocr_ran is True
        assert sample_result.to_dict()["stage"] == "full"

    def test_stage_survives_the_round_trip(self, sample_result: PipelineResult):
        detect_only = dataclasses.replace(sample_result, stage="detect")
        data = json.loads(detect_only.to_json())
        assert data["stage"] == "detect" and data["schema_version"] == "1.2"
        back = PipelineResult.from_dict(data)
        assert back.stage == "detect" and back.ocr_ran is False
        assert back.to_dict() == detect_only.to_dict()

    def test_a_1_0_document_without_stage_reads_as_full(self, sample_result: PipelineResult):
        legacy = sample_result.to_dict()
        legacy["schema_version"] = "1.0"
        legacy.pop("stage")                       # 1.0 never had the key
        back = PipelineResult.from_dict(legacy)
        assert back.schema_version == "1.0" and back.stage == "full"
        assert back.ocr_ran is True

    @pytest.mark.parametrize("stage", ["ocr", "", None, 1])
    def test_an_unknown_stage_is_a_structured_error(self, sample_result, stage):
        data = {**sample_result.to_dict(), "stage": stage}
        with pytest.raises(StudioError) as ei:
            PipelineResult.from_dict(data)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        # "reocr" joined the list in P9.3; spelled out rather than read back from
        # RESULT_STAGES, so renaming a stage still fails here.
        assert ei.value.details["expected"] == ["detect", "full", "reocr"]

    def test_a_1_1_document_without_flags_reads_as_unflagged(self, sample_result: PipelineResult):
        """1.2 added `flags`/`contained_in`; a 1.1 document simply did not have them (P8.2b)."""
        legacy = sample_result.to_dict()
        legacy["schema_version"] = "1.1"
        for block in legacy["blocks"]:
            block.pop("flags", None)
            block.pop("contained_in", None)
        back = PipelineResult.from_dict(legacy)
        assert back.schema_version == "1.1"
        assert all(b.flags == () and b.contained_in is None for b in back.blocks)
        assert back.blocks == sample_result.blocks           # nothing was invented

    @pytest.mark.parametrize("version", ["2.0", "0.9", "1.3", ""])
    def test_an_unreadable_schema_version_says_so(self, sample_result, version):
        data = {**sample_result.to_dict(), "schema_version": version}
        with pytest.raises(StudioError) as ei:
            PipelineResult.from_dict(data)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert ei.value.details["supported"] == ["1.0", "1.1", "1.2"]


class TestBlockFlags:
    """P8.2b — result schema 1.2: `flags` + `contained_in` on the block itself."""

    def block(self, **overrides) -> TextBlock:
        data = dict(id=3, bbox=BBox(10, 10, 60, 40), label=TEXT_LABEL, score=0.5,
                    text="text", reading_order=3, color=color_for_index(3), error=None)
        data.update(overrides)
        return TextBlock(**data)

    def test_the_new_fields_default_to_nothing_observed(self):
        block = self.block()
        assert block.flags == () and block.contained_in is None
        # ... and the defaults are the *documented* nothing, not a missing key.
        assert block.to_dict()["flags"] == [] and block.to_dict()["contained_in"] is None

    def test_they_survive_the_round_trip(self):
        block = self.block(flags=("possible_truncation",), contained_in=0)
        again = TextBlock.from_dict(block.to_dict())
        assert again == block
        data = block.to_dict()
        assert data["flags"] == ["possible_truncation"] and data["contained_in"] == 0

    def test_a_list_coming_from_json_becomes_a_tuple(self):
        data = self.block().to_dict()
        data["flags"] = ["possible_truncation"]
        assert TextBlock.from_dict(data).flags == ("possible_truncation",)

    def test_the_known_flag_vocabulary_is_closed_and_documented(self):
        assert BLOCK_FLAGS == ("possible_truncation",)
        assert 0.0 < TRUNCATION_RATIO <= 1.0 and 0.0 < CONTAINMENT_RATIO <= 1.0

    def test_contained_ratio_is_the_fraction_of_self_covered(self):
        inner, outer = BBox(20, 20, 40, 40), BBox(0, 0, 100, 100)
        assert inner.contained_ratio(outer) == 1.0
        assert outer.contained_ratio(inner) == inner.area / outer.area     # asymmetric
        assert inner.contained_ratio(BBox(500, 500, 600, 600)) == 0.0
        assert BBox(0, 0, 0, 0).contained_ratio(outer) == 0.0              # no division by zero
        half = BBox(80, 20, 120, 60)                    # 40x40, half of it inside `outer`
        assert half.contained_ratio(outer) == 0.5


class TestMisc:
    def test_palette(self):
        assert len(BLOCK_PALETTE) == 20
        assert all(re.match(r"^#[0-9A-F]{6}$", c) for c in BLOCK_PALETTE)
        assert color_for_index(0) == color_for_index(20) == BLOCK_PALETTE[0]
        assert color_for_index(7) == BLOCK_PALETTE[7]

    def test_sha256_known_vector(self):
        assert sha256_bytes(b"abc") == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_job_progress(self):
        p = JobProgress("j1", JobState.RUNNING, "ocr", 150.0, "chunk 2/3")
        d = p.to_dict()
        assert d["percent"] == 100.0 and d["state"] == "running"
        assert JobProgress("j2", "idle", "", -3).to_dict()["percent"] == 0.0

    def test_image_info_roundtrip(self):
        i = ImageInfo("clipboard", None, 10, 20, sha256_bytes(b"x"))
        assert ImageInfo.from_dict(i.to_dict()) == i


class TestMalformedPayloads:
    """Part 1 inspection (Turn 4): a bad payload must never escape as KeyError.

    Payloads arrive from files, the clipboard and MCP clients; the app promises
    {code, message} for every failure (PLAN 2.3), so every from_dict maps malformed
    input to INVALID_ARGUMENT.
    """

    @pytest.mark.parametrize("loader,payload", [
        (BBox.from_dict, {}),
        (BBox.from_dict, {"x0": 0, "y0": 0, "x1": 5}),
        (BBox.from_dict, {"x0": "a", "y0": 0, "x1": 5, "y1": 5}),
        (BBox.from_dict, "not-a-dict"),
        (Detection.from_dict, {"bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1}}),
        (Detection.from_dict, {"bbox": {}, "label": "t", "score": 0.5}),
        (Detection.from_dict, {"bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                               "label": "t", "score": "high"}),
        (TextBlock.from_dict, {"id": 0}),
        (ImageInfo.from_dict, {"width": 10}),
        (ImageInfo.from_dict, {"source": "file", "file_name": None,
                               "width": 1, "height": 1}),
        (PipelineResult.from_dict, {"image": {}}),
        (PipelineResult.from_dict, []),
    ])
    def test_structured_error_not_keyerror(self, loader, payload):
        with pytest.raises(StudioError) as ei:
            loader(payload)
        assert ei.value.code is ErrorCode.INVALID_ARGUMENT
        assert ei.value.exit_code == 2
        assert "Malformed" in ei.value.message

    def test_available_keys_are_reported(self):
        with pytest.raises(StudioError) as ei:
            BBox.from_dict({"x0": 0, "y0": 0})
        assert ei.value.details["available_keys"] == ["x0", "y0"]

    def test_valid_payloads_still_round_trip(self, sample_result):
        assert PipelineResult.from_dict(json.loads(sample_result.to_json())) == sample_result
        assert TextBlock.from_dict(sample_result.blocks[0].to_dict()) == sample_result.blocks[0]

    def test_optional_backend_still_tolerated(self, sample_result):
        data = sample_result.to_dict()
        del data["detection"]["backend"]
        assert PipelineResult.from_dict(data).detection.backend == "onnxruntime"
