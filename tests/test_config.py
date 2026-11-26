"""Part 1 tests: config defaults, validation, merge, import/export, precedence."""
from __future__ import annotations

import json

import pytest

from animetext_studio.core.cli import parse_args
from animetext_studio.core.config import (
    AppConfig, KNOWN_MODELS, MODEL_PRESETS, cli_overrides, load_config_file,
    resolve_config, save_config_file, validate_config_dict, validate_config_file,
)
from animetext_studio.core.types import ErrorCode, StudioError


class TestDefaults:
    def test_demo_defaults(self):
        cfg = AppConfig.default()
        assert cfg.detection.model == "yolo12m_animetext"
        assert cfg.detection.iou_threshold == 0.35
        assert cfg.detection.score_threshold == 0.05
        assert cfg.pipeline.reading_order == "manga_rtl"
        assert cfg.ocr.max_patches == "auto"
        assert cfg.gui.hover_opacity == 0.5
        assert (cfg.mcp.host, cfg.mcp.port, cfg.mcp.transport, cfg.mcp.enabled) == (
            "127.0.0.1", 8765, "streamable-http", True)

    def test_presets_match_model_card(self):
        assert KNOWN_MODELS == ("yolo12n_animetext", "yolo12s_animetext",
                                "yolo12m_animetext", "yolo12l_animetext", "yolo12x_animetext")
        expected = {"yolo12n_animetext": 0.251, "yolo12s_animetext": 0.272,
                    "yolo12m_animetext": 0.299, "yolo12l_animetext": 0.426,
                    "yolo12x_animetext": 0.425}
        for name, thr in expected.items():
            assert MODEL_PRESETS[name]["recommended_score_threshold"] == thr


class TestRoundtrip:
    def test_byte_stable(self):
        cfg = AppConfig.default()
        text = cfg.to_json()
        assert AppConfig.from_dict(json.loads(text)).to_json() == text

    def test_lenient_partial(self):
        from dataclasses import asdict
        cfg = AppConfig.from_dict({"detection": {"model": "yolo12n_animetext"}})
        assert cfg.detection.model == "yolo12n_animetext"
        assert cfg.detection.iou_threshold == 0.35  # default preserved
        assert asdict(cfg.ocr) == asdict(AppConfig.default().ocr)  # other sections default

    def test_empty_dict_is_default(self):
        assert AppConfig.from_dict({}).to_dict() == AppConfig.default().to_dict()


class TestValidation:
    @pytest.mark.parametrize("patch,path", [
        ({"detection": {"iou_threshold": 1.5}}, "detection.iou_threshold"),
        ({"detection": {"score_threshold": -0.1}}, "detection.score_threshold"),
        ({"detection": {"allow_dynamic": "yes"}}, "detection.allow_dynamic"),
        ({"detection": {"model": ""}}, "detection.model"),
        ({"ocr": {"device": "gpu"}}, "ocr.device"),
        ({"ocr": {"max_patches": 999}}, "ocr.max_patches"),
        ({"ocr": {"max_patches": True}}, "ocr.max_patches"),
        ({"ocr": {"batch_size": 0}}, "ocr.batch_size"),
        ({"ocr": {"max_new_tokens": 2000}}, "ocr.max_new_tokens"),
        ({"ocr": {"crop_padding_px": -1}}, "ocr.crop_padding_px"),
        ({"pipeline": {"reading_order": "sideways"}}, "pipeline.reading_order"),
        ({"gui": {"fill_opacity": -0.1}}, "gui.fill_opacity"),
        ({"gui": {"zoom_mode": "stretch"}}, "gui.zoom_mode"),
        ({"gui": {"zoom_percent": 5}}, "gui.zoom_percent"),
        ({"mcp": {"port": 0}}, "mcp.port"),
        ({"mcp": {"port": 70000}}, "mcp.port"),
        ({"mcp": {"transport": "grpc"}}, "mcp.transport"),
        ({"mcp": {"host": ""}}, "mcp.host"),
        ({"mcp": {"auth_token": 42}}, "mcp.auth_token"),
        ({"paths": {"hf_cache_dir": 3}}, "paths.hf_cache_dir"),
        ({"hf": {"token_env": ""}}, "hf.token_env"),
        ({"schema_version": "2.0"}, "schema_version"),
        ({"detection": []}, "detection"),
    ])
    def test_invalid_values_raise_with_path(self, patch, path):
        data = AppConfig.default().to_dict()
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                data[k].update(v)
            else:
                data[k] = v
        with pytest.raises(StudioError) as ei:
            AppConfig.from_dict(data)
        assert ei.value.code is ErrorCode.INVALID_CONFIG
        assert any(i["path"] == path for i in ei.value.details["issues"])

    def test_non_dict_rejected(self):
        with pytest.raises(StudioError):
            AppConfig.from_dict([1, 2, 3])

    def test_unknown_key_warning_not_error(self):
        errors, warnings = validate_config_dict({"detection": {"bogus": 1}})
        assert errors == []
        assert {"path": "detection.bogus", "message": "unknown key"} in warnings

    def test_unknown_key_strict_is_error(self):
        with pytest.raises(StudioError):
            AppConfig.from_dict({"bogus_top": 1}, strict=True)

    def test_unknown_model_is_warning(self):
        errors, warnings = validate_config_dict({"detection": {"model": "yolo99_future"}})
        assert errors == []
        assert any(w["path"] == "detection.model" for w in warnings)

    def test_no_duplicate_warnings(self):
        _e, warnings = validate_config_dict({"detection": {"bogus": 1}})
        assert len(warnings) == 1

    def test_valid_devices_and_patches(self):
        for device in ("auto", "cpu", "cuda", "cuda:0", "cuda:12"):
            AppConfig.from_dict({"ocr": {"device": device}})
        for mp in ("auto", 128, 192, 256, 384, 512):
            AppConfig.from_dict({"ocr": {"max_patches": mp}})


class TestMerge:
    def test_changed_keys_and_immutability(self):
        cfg = AppConfig.default()
        new, changed = cfg.merged({"detection": {"model": "yolo12x_animetext", "iou_threshold": 0.5}})
        assert changed == ["detection.iou_threshold", "detection.model"]
        assert new.detection.model == "yolo12x_animetext"
        assert cfg.detection.model == "yolo12m_animetext"  # untouched
        assert new.detection.score_threshold == 0.05  # sibling preserved

    def test_integral_float_patch_is_not_a_spurious_change(self):
        """P7's set_config reports changed keys; 16.0 must not look like a change to 16."""
        _new, changed = AppConfig.default().merged({"ocr": {"batch_size": 16.0}})
        assert changed == []

    def test_integral_float_patch_is_coerced_to_int(self):
        new, changed = AppConfig.default().merged(
            {"ocr": {"batch_size": 32.0, "max_patches": 192.0}, "mcp": {"port": 9001.0}})
        assert changed == ["mcp.port", "ocr.batch_size", "ocr.max_patches"]
        assert new.ocr.batch_size == 32 and isinstance(new.ocr.batch_size, int)
        assert new.ocr.max_patches == 192 and isinstance(new.ocr.max_patches, int)
        assert new.mcp.port == 9001 and isinstance(new.mcp.port, int)

    def test_noop_patch(self):
        cfg = AppConfig.default()
        new, changed = cfg.merged({"detection": {"model": cfg.detection.model}})
        assert changed == []
        assert new.to_dict() == cfg.to_dict()

    def test_null_override(self):
        cfg = AppConfig.default().merged({"mcp": {"auth_token": "secret"}})[0]
        new, changed = cfg.merged({"mcp": {"auth_token": None}})
        assert new.mcp.auth_token is None
        assert changed == ["mcp.auth_token"]

    def test_invalid_patch_leaves_original(self):
        cfg = AppConfig.default()
        with pytest.raises(StudioError):
            cfg.merged({"mcp": {"port": -1}})
        assert cfg.mcp.port == 8765


class TestFiles:
    def test_save_load_roundtrip(self, tmp_path):
        cfg = AppConfig.default().merged({"detection": {"model": "yolo12n_animetext"}})[0]
        path = tmp_path / "nested" / "cfg.json"
        save_config_file(cfg, path)
        assert load_config_file(path).to_dict() == cfg.to_dict()

    def test_missing_file_io_error(self, tmp_path):
        with pytest.raises(StudioError) as ei:
            load_config_file(tmp_path / "nope.json")
        assert ei.value.code is ErrorCode.IO_ERROR

    def test_malformed_json_invalid_config(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(StudioError) as ei:
            load_config_file(path)
        assert ei.value.code is ErrorCode.INVALID_CONFIG

    def test_validate_report(self, tmp_path):
        good = tmp_path / "good.json"
        save_config_file(AppConfig.default(), good)
        report = validate_config_file(good)
        assert report["valid"] and report["errors"] == [] and report["warnings"] == []

        partial = tmp_path / "partial.json"
        partial.write_text('{"detection": {"model": "yolo12n_animetext"}}', encoding="utf-8")
        report = validate_config_file(partial)
        assert report["valid"]
        assert any(w["message"].startswith("missing key") for w in report["warnings"])

        bad = tmp_path / "bad2.json"
        bad.write_text('{"mcp": {"port": 0}}', encoding="utf-8")
        report = validate_config_file(bad)
        assert not report["valid"]
        assert any(i["path"] == "mcp.port" for i in report["errors"])


class TestPrecedence:
    def test_cli_over_file_over_defaults(self, tmp_path):
        path = tmp_path / "cfg.json"
        save_config_file(AppConfig.default().merged({
            "detection": {"model": "yolo12n_animetext", "iou_threshold": 0.9},
        })[0], path)
        args = parse_args(["--config", str(path), "--model", "yolo12x_animetext"])
        cfg = resolve_config(args)
        assert cfg.detection.model == "yolo12x_animetext"  # CLI wins
        assert cfg.detection.iou_threshold == 0.9          # file wins over default
        assert cfg.detection.score_threshold == 0.05       # default

    def test_cli_only(self):
        cfg = resolve_config(parse_args(["--iou", "0.42", "--mcp-port", "9999"]))
        assert cfg.detection.iou_threshold == 0.42
        assert cfg.mcp.port == 9999

    def test_transport_none_disables(self):
        patch = cli_overrides(parse_args(["--mcp-transport", "none"]))
        assert patch["mcp"] == {"transport": "none", "enabled": False}

    def test_hf_token_never_persisted(self):
        patch = cli_overrides(parse_args(["--hf-token", "hf_secret"]))
        assert "hf_secret" not in json.dumps(patch)
        assert "hf" not in patch
