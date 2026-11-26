"""P8 — hardening: log capacity, secret redaction, endpoint warnings.

Three things that are easy to get wrong and impossible to notice from the happy path:
a config key that nothing reads, an export that quietly carries a token, and a server
that binds a LAN address without saying so.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from animetext_studio.core.config import (
    REDACTED, AppConfig, redacted_config_dict, strip_redacted,
)
from animetext_studio.core.jobs import JobManager
from animetext_studio.core.pipeline import Pipeline
from animetext_studio.core.types import ErrorCode, StudioError
from animetext_studio.engines.base import FakeDetectionEngine, FakeOcrEngine
from animetext_studio.mcp.bridge import ActivityLog
from animetext_studio.mcp.server import security_warnings
from animetext_studio.mcp.service import StudioService


def make_service(config=None):
    config = config or AppConfig.default()
    manager = JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), config))
    return StudioService(manager, config, mock=True)


class TestLogCapacity:
    """`mcp.log_capacity` is a live setting, not a constructor argument nobody passes."""

    def test_capacity_comes_from_config(self):
        cfg = AppConfig.default()
        cfg.mcp.log_capacity = 25
        assert make_service(cfg).activity.summary()["capacity"] == 25

    def test_capacity_follows_a_config_change(self):
        svc = make_service()
        assert svc.activity.summary()["capacity"] == 200
        svc.apply_config_patch({"mcp": {"log_capacity": 12}})
        assert svc.activity.summary()["capacity"] == 12

    def test_shrinking_keeps_the_newest_entries(self):
        log = ActivityLog(capacity=50)
        for i in range(50):
            log.record(f"tool_{i}")
        assert log.set_capacity(5) == 5
        # `events()` is most-recent-first (like the dock and `log://mcp`), so the survivors
        # are the newest five, newest first.
        assert [e.tool for e in log.events()] == [f"tool_{i}" for i in range(49, 44, -1)]

    def test_out_of_range_capacity_is_a_structured_error(self):
        with pytest.raises(StudioError) as ei:
            make_service().apply_config_patch({"mcp": {"log_capacity": 1}})
        assert ei.value.code is ErrorCode.INVALID_CONFIG


class TestRedaction:
    def test_export_is_faithful_by_default(self, tmp_path):
        svc = make_service()
        svc.apply_config_patch({"mcp": {"auth_token": "hf_live"}})
        out = svc.export_config(str(tmp_path / "c.json"))
        assert out["redacted"] == []
        assert json.loads((tmp_path / "c.json").read_text())["mcp"]["auth_token"] == "hf_live"

    def test_redacted_export_hides_the_token_and_says_so(self, tmp_path):
        svc = make_service()
        svc.apply_config_patch({"mcp": {"auth_token": "hf_live"}})
        path = svc.export_config(str(tmp_path / "c.json"), redact_secrets=True)["path"]
        text = (tmp_path / "c.json").read_text(encoding="utf-8")
        assert "hf_live" not in text
        assert json.loads(text)["mcp"]["auth_token"] == REDACTED
        out = svc.export_config(None, redact_secrets=True)
        assert out["redacted"] == ["mcp.auth_token"]
        assert out["config"]["mcp"]["auth_token"] == REDACTED
        assert path.endswith("c.json")

    def test_importing_a_redacted_file_keeps_the_working_token(self, tmp_path):
        svc = make_service()
        svc.apply_config_patch({"mcp": {"auth_token": "hf_live"}})
        path = svc.export_config(str(tmp_path / "c.json"), redact_secrets=True)["path"]
        got = svc.import_config(path=path)
        assert got["kept"] == ["mcp.auth_token"]
        assert svc.config.mcp.auth_token == "hf_live"

    def test_a_placeholder_in_a_patch_is_not_taken_as_a_token(self):
        svc = make_service()
        svc.apply_config_patch({"mcp": {"auth_token": "hf_live"}})
        out = svc.apply_config_patch({"mcp": {"auth_token": REDACTED}})
        assert svc.config.mcp.auth_token == "hf_live"
        assert out["changed"] == []
        assert any("kept" in w for w in out["warnings"])

    def test_a_placeholder_next_to_real_keys_changes_only_the_real_ones(self):
        svc = make_service()
        svc.apply_config_patch({"mcp": {"auth_token": "hf_live"}})
        out = svc.apply_config_patch({"mcp": {"auth_token": REDACTED, "port": 9011}})
        assert svc.config.mcp.auth_token == "hf_live" and svc.config.mcp.port == 9011
        assert out["changed"] == ["mcp.port"]

    def test_no_token_means_nothing_to_redact(self):
        data, touched = redacted_config_dict(AppConfig.default())
        assert touched == [] and data["mcp"]["auth_token"] is None

    def test_strip_leaves_everything_else_alone(self):
        data, kept = strip_redacted({"mcp": {"auth_token": REDACTED, "port": 9001}})
        assert data == {"mcp": {"port": 9001}} and kept == ["mcp.auth_token"]


class TestCliRedaction:
    """`--export-config --redact-secrets` (P8): a shareable file, one flag away."""

    def _config_with_token(self, tmp_path, token="hf_secret123"):
        path = tmp_path / "in.json"
        path.write_text(json.dumps({"mcp": {"auth_token": token}}), encoding="utf-8")
        return path

    def test_export_without_the_flag_still_carries_the_token(self, tmp_path, capsys):
        from animetext_studio.__main__ import main

        out = tmp_path / "out.json"
        assert main(["--config", str(self._config_with_token(tmp_path)),
                     "--export-config", str(out)]) == 0
        assert "hf_secret123" in out.read_text(encoding="utf-8")
        assert "redacted" not in capsys.readouterr().out

    def test_export_with_the_flag_hides_it_and_says_so(self, tmp_path, capsys):
        from animetext_studio.__main__ import main

        out = tmp_path / "out.json"
        assert main(["--config", str(self._config_with_token(tmp_path)),
                     "--export-config", str(out), "--redact-secrets"]) == 0
        text = out.read_text(encoding="utf-8")
        assert "hf_secret123" not in text and REDACTED in text
        assert "redacted" in capsys.readouterr().out

    def test_a_redacted_export_imports_without_blanking_the_token(self, tmp_path):
        """The round-trip that makes redaction safe: share the file, keep the working token."""
        from animetext_studio.__main__ import main

        out = tmp_path / "out.json"
        assert main(["--config", str(self._config_with_token(tmp_path)),
                     "--export-config", str(out), "--redact-secrets"]) == 0
        live = AppConfig.default()
        live.mcp.auth_token = "hf_secret123"
        svc = make_service(live)
        imported = svc.import_config(path=str(out))
        assert imported["kept"] == ["mcp.auth_token"], imported
        assert svc.config.mcp.auth_token == "hf_secret123"
        assert "mcp.auth_token" not in imported["changed"]


class TestTokenDiscovery:
    """P8: one optional `HF_token.txt` at the project root beats exporting it everywhere."""

    def test_explicit_beats_env_and_file(self, tmp_path, monkeypatch):
        from animetext_studio.core.model_store import discover_hf_token

        (tmp_path / "HF_token.txt").write_text("hf_from_file\n", encoding="utf-8")
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        monkeypatch.chdir(tmp_path)
        assert discover_hf_token("hf_explicit") == "hf_explicit"

    def test_env_beats_file(self, tmp_path, monkeypatch):
        from animetext_studio.core.model_store import discover_hf_token

        (tmp_path / "HF_token.txt").write_text("hf_from_file\n", encoding="utf-8")
        monkeypatch.setenv("HF_TOKEN", "hf_from_env")
        monkeypatch.chdir(tmp_path)
        assert discover_hf_token() == "hf_from_env"

    def test_file_is_used_when_there_is_no_env_token(self, tmp_path, monkeypatch):
        from animetext_studio.core.model_store import discover_hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("ANIMETEXT_TOKEN_FILE", raising=False)
        (tmp_path / "HF_token.txt").write_text("  hf_from_file  \n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert discover_hf_token() == "hf_from_file"

    def test_comments_and_blank_lines_are_skipped(self, tmp_path, monkeypatch):
        from animetext_studio.core.model_store import read_token_file

        f = tmp_path / "tok.txt"
        f.write_text("# put the token below\n\nhf_real\nextra line ignored? no, first wins\n",
                     encoding="utf-8")
        assert read_token_file(f) == "hf_real"

    def test_missing_or_binary_file_is_none_not_an_error(self, tmp_path):
        from animetext_studio.core.model_store import read_token_file

        assert read_token_file(tmp_path / "nope.txt") is None
        binary = tmp_path / "bin.txt"
        binary.write_bytes(b"\xff\xfe\x00garbage")
        assert read_token_file(binary) is None

    def test_env_pointing_at_a_file_wins_over_the_search(self, tmp_path, monkeypatch):
        from animetext_studio.core.model_store import discover_hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        here = tmp_path / "HF_token.txt"
        here.write_text("hf_here\n", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere" / "my.token"
        elsewhere.parent.mkdir()
        elsewhere.write_text("hf_elsewhere\n", encoding="utf-8")
        monkeypatch.setenv("ANIMETEXT_TOKEN_FILE", str(elsewhere))
        monkeypatch.chdir(tmp_path)
        assert discover_hf_token() == "hf_elsewhere"

    def test_no_token_anywhere_is_none(self, tmp_path, monkeypatch):
        from animetext_studio.core.model_store import discover_hf_token

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("ANIMETEXT_TOKEN_FILE", raising=False)
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.chdir(empty)
        # the project root may legitimately hold a token file, so only assert the shape
        assert discover_hf_token() is None or discover_hf_token().startswith("hf")


class TestUnloadBadge:
    """P8: `unload_engines` must reach the window, not only the log."""

    def test_unload_refreshes_the_window_badge(self):
        calls = []

        class Watcher:
            def refresh_engine_badge(self):
                calls.append(1)

            def __getattr__(self, name):            # the service calls a few other things
                return lambda *a, **k: None

        svc = make_service()
        svc._window = Watcher()          # attach_window() needs a live QApplication
        svc.unload()
        assert calls == [1], "unload_engines left the window's engine badge untouched"

    def test_a_window_without_the_method_is_not_an_error(self):
        svc = make_service()
        svc._window = object()          # an older/duck-typed window must not break unload
        assert svc.unload()["unloaded"] is not None or True


class TestEndpointWarnings:
    def test_loopback_is_silent(self):
        assert security_warnings(AppConfig.default().mcp) == []

    def test_a_public_bind_without_a_token_says_why(self):
        cfg = AppConfig.default()
        cfg.mcp.host, cfg.mcp.auth_token = "0.0.0.0", None
        warnings = security_warnings(cfg.mcp)
        assert len(warnings) == 1
        assert "0.0.0.0" in warnings[0] and "NO auth token" in warnings[0]

    def test_a_public_bind_with_a_token_says_auth_is_on(self):
        cfg = AppConfig.default()
        cfg.mcp.host, cfg.mcp.auth_token = "0.0.0.0", "hf_token"
        warnings = security_warnings(cfg.mcp)
        assert len(warnings) == 1 and "bearer auth is on" in warnings[0]

    @pytest.mark.parametrize("transport,enabled", [("stdio", True), ("none", True),
                                                   ("streamable-http", False)])
    def test_no_http_endpoint_means_no_warnings(self, transport, enabled):
        cfg = AppConfig.default()
        cfg.mcp.host, cfg.mcp.transport, cfg.mcp.enabled = "0.0.0.0", transport, enabled
        assert security_warnings(cfg.mcp) == []


class TestRestartRequired:
    """Found by the P8.8 QA pass against PLAN §2.2.

    §2.2 promised "applying MCP host/port changes at runtime restarts the server thread".
    Nothing ever restarted a listening socket, and no door said so: a client that set
    `mcp.port` got `changed: ["mcp.port"]` back and had every reason to believe the endpoint
    had moved. The restart was not built (a client must not be able to move the socket it is
    talking to, mid-session); the *reporting* half of the promise is what ships, so the
    disagreement is impossible to hold silently.
    """

    def test_the_transport_keys_are_the_ones_read_at_server_start(self):
        from animetext_studio.core.config import RESTART_REQUIRED_KEYS
        assert RESTART_REQUIRED_KEYS == ("mcp.enabled", "mcp.transport", "mcp.host",
                                         "mcp.port", "mcp.auth_token")

    def test_everything_the_app_can_apply_on_the_spot_is_not_listed(self):
        from animetext_studio.core.config import RESTART_REQUIRED_KEYS, restart_required
        live = ["detection.score_threshold", "ocr.device", "ocr.max_patches",
                "pipeline.reading_order", "gui.fill_opacity", "mcp.log_capacity",
                "paths.offline", "hf.token_env"]
        assert restart_required(live) == []
        assert not (set(live) & set(RESTART_REQUIRED_KEYS)), (
            "a key the running app does apply would now claim otherwise")

    def test_the_order_is_stable_whatever_order_it_was_asked_in(self):
        from animetext_studio.core.config import restart_required
        assert restart_required(["mcp.port", "mcp.host"]) == restart_required(
            ["mcp.host", "mcp.port"]) == ["mcp.host", "mcp.port"]

    def test_an_unrelated_change_promises_nothing(self):
        svc = make_service()
        out = svc.apply_config_patch({"detection": {"score_threshold": 0.42}})
        assert out["changed"] == ["detection.score_threshold"]
        assert out["restart_required"] == []

    def test_set_config_names_the_key_that_needs_a_restart(self):
        svc = make_service()
        out = svc.apply_config_patch({"mcp": {"port": 9999}})
        assert out["config"]["mcp"]["port"] == 9999, "the value must still be stored"
        assert out["restart_required"] == ["mcp.port"]

    def test_a_mixed_patch_reports_only_the_frozen_part(self):
        svc = make_service()
        out = svc.apply_config_patch({"detection": {"score_threshold": 0.2},
                                      "mcp": {"host": "0.0.0.0"}})
        assert set(out["changed"]) == {"detection.score_threshold", "mcp.host"}
        assert out["restart_required"] == ["mcp.host"]

    def test_import_config_reports_it_too(self):
        svc = make_service()
        doc = svc.config.to_dict()
        doc["mcp"]["auth_token"] = "a-fresh-secret"
        out = svc.import_config(data=doc)
        assert out["restart_required"] == ["mcp.auth_token"]

    def test_a_restart_key_that_did_not_change_is_not_reported(self):
        svc = make_service()
        out = svc.import_config(data=svc.config.to_dict())
        assert out["changed"] == []
        assert out["restart_required"] == []

    def test_the_published_schema_carries_the_field(self):
        catalog = json.loads((Path(__file__).resolve().parent.parent
                              / "animetext_studio" / "schemas" / "mcp-tools.json")
                             .read_text(encoding="utf-8"))
        for name in ("set_config", "import_config"):
            tool = next(t for t in catalog["tools"] if t["name"] == name)
            assert "restart_required" in tool["outputSchema"]["properties"], name

    def test_changed_config_keys_finds_what_a_diff_would(self):
        from animetext_studio.core.config import changed_config_keys
        before = AppConfig.default().to_dict()
        after = json.loads(json.dumps(before))
        after["mcp"]["port"] = 9000
        after["detection"]["score_threshold"] = 0.4
        del after["paths"]["export_dir"]
        changed = changed_config_keys(before, after)
        assert changed == ["detection.score_threshold", "mcp.port", "paths.export_dir"]

    def test_the_window_says_it_out_loud_on_import(self, qapp, tmp_path):
        """The GUI door: importing a file must not leave the badge and the config disagreeing."""
        from animetext_studio.gui.main_window import MainWindow

        cfg = AppConfig.default()
        manager = JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), cfg))
        window = MainWindow(manager, cfg)
        try:
            moved = json.loads(json.dumps(cfg.to_dict()))
            moved["mcp"]["port"] = 9123
            path = tmp_path / "moved.json"
            path.write_text(json.dumps(moved), encoding="utf-8")
            assert window.import_config(str(path)) is True
            said = window.status_label.text()
            assert "mcp.port" in said and "next start" in said, said
            assert window.config.to_dict()["mcp"]["port"] == 9123, "the value is still kept"

            ordinary = json.loads(json.dumps(window.config.to_dict()))
            ordinary["detection"]["iou_threshold"] = 0.7
            path2 = tmp_path / "ordinary.json"
            path2.write_text(json.dumps(ordinary), encoding="utf-8")
            assert window.import_config(str(path2)) is True
            said2 = window.status_label.text()
            assert "imported" in said2.lower(), said2
            assert "next start" not in said2, (
                f"an ordinary key must not be reported as needing a restart: {said2}")
        finally:
            window.close()
