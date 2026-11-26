"""Part 1 tests: published JSON Schemas are valid and agree with native validation.

jsonschema is a dev dependency (decision D5); runtime validation is native.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from animetext_studio.core.config import (
    AppConfig, canonical_config_schema, partial_config_schema, validate_config_dict,
)
from conftest import load_schema

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(scope="module")
def config_validator():
    schema = load_schema("config.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture(scope="module")
def result_validator():
    schema = load_schema("result.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


class TestConfigSchema:
    def test_default_config_valid(self, config_validator):
        config_validator.validate(AppConfig.default().to_dict())

    def test_roundtrip_file_valid(self, config_validator, tmp_path):
        from animetext_studio.core.config import save_config_file
        path = tmp_path / "cfg.json"
        save_config_file(AppConfig.default(), path)
        config_validator.validate(json.loads(path.read_text(encoding="utf-8")))

    @pytest.mark.parametrize("section,key,bad", [
        ("detection", "iou_threshold", 1.5),
        ("detection", "allow_dynamic", "yes"),
        ("ocr", "device", "gpu"),
        ("ocr", "max_patches", 999),
        ("ocr", "batch_size", 0),
        ("pipeline", "reading_order", "sideways"),
        ("gui", "zoom_percent", 5),
        ("mcp", "port", 0),
        ("mcp", "transport", "grpc"),
    ])
    def test_native_and_schema_agree_invalid(self, config_validator, section, key, bad):
        data = AppConfig.default().to_dict()
        data[section][key] = bad
        errors, _ = validate_config_dict(data)
        assert errors, "native validator should reject"
        assert not config_validator.is_valid(data), "schema should reject"

    @pytest.mark.parametrize("section,key,good", [
        ("detection", "model", "yolo12n_animetext"),
        ("detection", "iou_threshold", 0.9),
        ("ocr", "device", "cuda:2"),
        ("ocr", "max_patches", 256),
        ("mcp", "transport", "stdio"),
        ("mcp", "auth_token", "secret"),
        ("gui", "zoom_mode", "percent"),
        ("paths", "hf_cache_dir", "/tmp/hf"),
    ])
    def test_native_and_schema_agree_valid(self, config_validator, section, key, good):
        data = AppConfig.default().to_dict()
        data[section][key] = good
        errors, _ = validate_config_dict(data)
        assert not errors
        assert config_validator.is_valid(data)

    def test_unknown_key_rejected_by_schema(self, config_validator):
        data = AppConfig.default().to_dict()
        data["surprise"] = 1
        assert not config_validator.is_valid(data)


# ---------------------------------------------------------------------------
# Native validator <-> published schema agreement (Part 1 inspection, Turn 4)
#
# A differential audit found three real divergences: integral floats for integer
# fields (native rejected, schema accepted), unknown keys and missing sections
# (native accepted per decision D10, schema rejected). Fixed by coercing integral
# floats (D22) and by publishing the accepted shape as config-partial-1.0 (D21).
# These tests are what keeps the two contracts from drifting again.
# ---------------------------------------------------------------------------

#: (dotted path, value) -> expected verdict of the *native* validator
VALID_CASES: list[tuple[str, Any]] = [
    ("ocr.batch_size", 16.0),            # integral float (JSON Schema allows it)
    ("ocr.max_patches", 256.0),
    ("mcp.port", 9000.0),
    ("gui.zoom_percent", 150.0),
    ("pipeline.min_box_area_px", 32.0),
    ("detection.iou_threshold", 1),
    ("detection.iou_threshold", 1.0),
    ("mcp.auth_token", ""),
    ("paths.export_dir", ""),
    ("detection.model", "brand_new_yolo"),   # unknown model is a warning, not an error
]

INVALID_CASES: list[tuple[str, Any]] = [
    ("ocr.batch_size", 16.5),            # non-integral float
    ("ocr.batch_size", True),            # bool is not an int here
    ("ocr.batch_size", 0),
    ("ocr.max_patches", 127),
    ("ocr.max_patches", True),
    ("ocr.max_new_tokens", 128.5),
    ("ocr.device", "gpu"),
    ("detection.iou_threshold", "0.5"),
    ("detection.iou_threshold", -0.0001),
    ("detection.allow_dynamic", 1),
    ("detection.model", ""),
    ("mcp.port", 0),
    ("mcp.port", "8765"),
    ("mcp.host", ""),
    ("mcp.transport", "grpc"),
    ("gui.zoom_percent", 9),
    ("gui.fill_opacity", 2),
    ("gui.overlay_visible", None),
    ("pipeline.reading_order", "rtl"),
    ("pipeline.min_box_area_px", -1),
    ("paths.hf_cache_dir", 5),
    ("hf.token_env", ""),
    ("schema_version", "2.0"),
]


def _set(data: dict, dotted: str, value: Any) -> None:
    """Set "section.key" or a top-level "key" in a config document."""
    if "." in dotted:
        section, key = dotted.split(".", 1)
        data.setdefault(section, {})[key] = value
    else:
        data[dotted] = value


#: Sections that are JSON objects (the scalar `schema_version` is not).
OBJECT_SECTIONS = {
    name: list(sub["properties"])
    for name, sub in canonical_config_schema()["properties"].items()
    if isinstance(sub, dict) and "properties" in sub
}


@pytest.fixture(scope="module")
def partial_validator():
    schema = partial_config_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


class TestNativeSchemaAgreement:
    """Native verdict must equal the published *accepted* shape, field by field."""

    @pytest.mark.parametrize("path,value", VALID_CASES)
    def test_valid_in_both(self, partial_validator, path, value):
        data = AppConfig.default().to_dict()
        _set(data, path, value)
        errors, _ = validate_config_dict(data)
        assert not errors, f"native rejected {path}={value!r}: {errors}"
        assert partial_validator.is_valid(data), f"partial schema rejected {path}={value!r}"

    @pytest.mark.parametrize("path,value", INVALID_CASES)
    def test_invalid_in_both(self, partial_validator, path, value):
        data = AppConfig.default().to_dict()
        _set(data, path, value)
        errors, _ = validate_config_dict(data)
        assert errors, f"native accepted {path}={value!r} but should not"
        assert not partial_validator.is_valid(data), f"partial schema accepted {path}={value!r}"

    def test_integral_floats_are_coerced_to_int(self):
        cfg = AppConfig.from_dict({"ocr": {"batch_size": 16.0, "max_patches": 256.0},
                                   "mcp": {"port": 9000.0}, "gui": {"zoom_percent": 150.0}})
        assert cfg.ocr.batch_size == 16 and isinstance(cfg.ocr.batch_size, int)
        assert cfg.ocr.max_patches == 256 and isinstance(cfg.ocr.max_patches, int)
        assert cfg.mcp.port == 9000 and cfg.gui.zoom_percent == 150

    def test_normalisation_does_not_mutate_input(self):
        data = AppConfig.default().to_dict()
        _set(data, "ocr.batch_size", 32.0)
        AppConfig.from_dict(data)
        assert data["ocr"]["batch_size"] == 32.0   # still a float in the caller's dict


class TestSchemaShapes:
    """Canonical = emitted, partial = accepted (decision D21)."""

    def test_canonical_requires_everything_and_forbids_unknown(self, config_validator):
        full = AppConfig.default().to_dict()
        config_validator.validate(full)
        missing = copy.deepcopy(full)
        del missing["gui"]
        assert not config_validator.is_valid(missing)
        extra = copy.deepcopy(full)
        extra["surprise"] = {}
        assert not config_validator.is_valid(extra)

    def test_partial_accepts_empty_and_partial_documents(self, config_validator):
        pv = jsonschema.Draft202012Validator(partial_config_schema())
        assert pv.is_valid({})
        assert pv.is_valid({"detection": {"model": "yolo12x_animetext"}})
        assert pv.is_valid({"no_such_section": {"x": 1}})
        # ... while the canonical schema rejects the same documents
        assert not config_validator.is_valid({})
        assert not config_validator.is_valid({"detection": {"model": "yolo12x_animetext"}})

    def test_partial_is_derived_not_duplicated(self):
        """Same field constraints as canonical; only required/additionalProperties differ."""
        canon, part = canonical_config_schema(), partial_config_schema()
        assert part["$id"] != canon["$id"]
        for name in OBJECT_SECTIONS:
            assert part["properties"][name]["properties"] == canon["properties"][name]["properties"]
            assert part["properties"][name].get("additionalProperties") is True

    def test_native_matches_partial_on_a_randomised_walk(self, partial_validator):
        """Fuzz-ish sweep: every mutation gets the same verdict from both validators."""
        import random

        pv = partial_validator
        rng = random.Random(20261119)
        pool: list[Any] = [0, 1, -1, 0.5, 16.0, "x", "", True, None, [], {}, 70000, "auto"]
        fields = OBJECT_SECTIONS
        mismatches = []
        for _ in range(400):
            data = AppConfig.default().to_dict()
            section = rng.choice(list(fields))
            _set(data, f"{section}.{rng.choice(fields[section])}", rng.choice(pool))
            native_ok = not validate_config_dict(data)[0]
            if native_ok != pv.is_valid(data):
                mismatches.append((section, data[section]))
        assert not mismatches, f"{len(mismatches)} divergences, e.g. {mismatches[:2]}"


class TestResultSchema:
    def test_sample_result_valid(self, result_validator, sample_result):
        result_validator.validate(sample_result.to_dict())

    def test_empty_blocks_valid(self, result_validator, sample_result):
        data = sample_result.to_dict()
        data["blocks"], data["block_count"] = [], 0
        result_validator.validate(data)

    def test_missing_block_count_invalid(self, result_validator, sample_result):
        data = sample_result.to_dict()
        del data["block_count"]
        assert not result_validator.is_valid(data)

    def test_bad_color_invalid(self, result_validator, sample_result):
        data = copy.deepcopy(sample_result.to_dict())
        data["blocks"][0]["color"] = "red"
        assert not result_validator.is_valid(data)

    def test_float_bbox_invalid(self, result_validator, sample_result):
        data = copy.deepcopy(sample_result.to_dict())
        data["blocks"][0]["bbox"]["x0"] = 1.5
        assert not result_validator.is_valid(data)

    def test_null_text_with_error_valid(self, result_validator, sample_result):
        data = sample_result.to_dict()
        assert data["blocks"][1]["text"] is None
        assert data["blocks"][1]["error"]["code"] == "OCR_FAILED"
        result_validator.validate(data)

    def test_bad_source_invalid(self, result_validator, sample_result):
        data = copy.deepcopy(sample_result.to_dict())
        data["image"]["source"] = "webcam"
        assert not result_validator.is_valid(data)


class TestResultSchemaVersions:
    """P8.1 `stage` (1.1) and P8.2b `flags`/`contained_in` (1.2) are published contract.

    Every version stays readable: the shipped schema validates what we write today, and the
    frozen copies validate the documents written back then — with no keys invented for them.
    """

    @staticmethod
    def _downgrade(data: dict, version: str) -> dict:
        """A faithful old document: drop the keys that version never had."""
        old = copy.deepcopy(data)
        old["schema_version"] = version
        if version < "1.1":
            old.pop("stage", None)
        if version < "1.2":
            for block in old.get("blocks", []):
                block.pop("flags", None)
                block.pop("contained_in", None)
        return old

    def test_the_shipped_schema_is_the_version_we_emit(self, result_validator):
        from animetext_studio.core.types import SCHEMA_VERSION

        assert result_validator.schema["$id"].endswith(f"result-{SCHEMA_VERSION}")
        assert result_validator.schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
        assert "stage" in result_validator.schema["required"]

    def test_stage_is_required(self, result_validator, sample_result):
        data = sample_result.to_dict()
        del data["stage"]
        assert not result_validator.is_valid(data)

    @pytest.mark.parametrize("stage", ["ocr", "FULL", "stage1", True])
    def test_stage_is_a_closed_enum(self, result_validator, sample_result, stage):
        data = copy.deepcopy(sample_result.to_dict())
        data["stage"] = stage
        assert not result_validator.is_valid(data)

    def test_detect_stage_documents_validate(self, result_validator, sample_result):
        data = sample_result.to_dict()
        data["stage"] = "detect"
        for block in data["blocks"]:
            block["text"] = None
            block["error"] = {"code": "SKIPPED", "message": "OCR disabled for this run"}
        result_validator.validate(data)

    @pytest.mark.parametrize("version", ["1.0", "1.1"])
    def test_the_frozen_schemas_still_read_documents_as_written(self, sample_result, version):
        """An old document validates against the schema of its own time, keys and all."""
        frozen = load_schema(f"result-{version}.schema.json")
        jsonschema.Draft202012Validator.check_schema(frozen)
        validator = jsonschema.Draft202012Validator(frozen)
        legacy = self._downgrade(sample_result.to_dict(), version)
        validator.validate(legacy)                       # old documents stay valid, forever
        assert not validator.is_valid(sample_result.to_dict())   # today's is a new version

    def test_1_2_added_flags_and_contained_in_to_every_block(self, result_validator):
        blocks = result_validator.schema["properties"]["blocks"]["items"]
        assert {"flags", "contained_in"} <= set(blocks["properties"])
        assert {"flags", "contained_in"} <= set(blocks["required"])

    def test_a_block_without_the_new_keys_is_not_a_1_2_document(self, result_validator,
                                                                 sample_result):
        data = sample_result.to_dict()                # version stays 1.2, keys go away
        for block in data["blocks"]:
            block.pop("flags")
            block.pop("contained_in")
        assert not result_validator.is_valid(data)

    def test_flags_are_a_closed_vocabulary(self, result_validator, sample_result):
        data = sample_result.to_dict()
        data["blocks"][0]["flags"] = ["looks_nice"]
        assert not result_validator.is_valid(data)

    def test_contained_in_is_an_id_or_null(self, result_validator, sample_result):
        data = sample_result.to_dict()
        data["blocks"][0]["contained_in"] = "3"          # a string id is not an id
        assert not result_validator.is_valid(data)
        data["blocks"][0]["contained_in"] = 3
        result_validator.validate(data)

    def test_nested_boxes_with_the_new_fields_validate(self, result_validator, sample_result):
        data = sample_result.to_dict()
        data["blocks"][0]["flags"] = ["possible_truncation"]
        data["blocks"][0]["contained_in"] = data["blocks"][1]["id"]
        result_validator.validate(data)
