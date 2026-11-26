"""Part 1 tests: CLI surface and entry-point exit codes."""
from __future__ import annotations

import json
import sys

import pytest

from animetext_studio.__main__ import main
from animetext_studio.core.config import AppConfig, load_config_file, save_config_file
from animetext_studio.core.types import EXIT_CODES, ErrorCode


class TestParser:
    def test_help(self, capsys):
        with pytest.raises(SystemExit) as ei:
            main(["--help"])
        assert ei.value.code == 0
        out = capsys.readouterr().out
        for flag in ("--mcp-host", "--mcp-port", "--mcp-transport", "--iou", "--score",
                     "--headless", "--mock", "--hf-token", "--dump-default-config"):
            assert flag in out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as ei:
            main(["--version"])
        assert ei.value.code == 0
        assert "animetext-studio 0.1.0" in capsys.readouterr().out

    @pytest.mark.parametrize("argv", [
        ["--iou", "1.5"], ["--score", "-0.2"], ["--mcp-port", "70000"],
        ["--mcp-port", "abc"], ["--device", "gpu"], ["--mcp-transport", "grpc"],
        ["--no-such-flag"],
    ])
    def test_bad_args_exit_2(self, argv):
        with pytest.raises(SystemExit) as ei:
            main(argv)
        assert ei.value.code == 2


class TestFoundationActions:
    def test_dump_default_config(self, capsys):
        assert main(["--dump-default-config"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data == AppConfig.default().to_dict()

    def test_validate_good(self, tmp_path, capsys):
        path = tmp_path / "cfg.json"
        save_config_file(AppConfig.default(), path)
        assert main(["--validate-config", str(path)]) == 0
        assert json.loads(capsys.readouterr().out)["valid"] is True

    def test_validate_bad_exit_2(self, tmp_path, capsys):
        path = tmp_path / "cfg.json"
        path.write_text('{"mcp": {"port": 0}}', encoding="utf-8")
        assert main(["--validate-config", str(path)]) == 2
        assert json.loads(capsys.readouterr().out)["valid"] is False

    def test_validate_missing_exit_10(self, tmp_path, capsys):
        assert main(["--validate-config", str(tmp_path / "nope.json")]) == 10
        err = json.loads(capsys.readouterr().err)
        assert err["error"]["code"] == "IO_ERROR"

    def test_export_resolved(self, tmp_path, capsys):
        out = tmp_path / "resolved.json"
        assert main(["--model", "yolo12x_animetext", "--iou", "0.425",
                     "--export-config", str(out)]) == 0
        cfg = load_config_file(out)
        assert cfg.detection.model == "yolo12x_animetext"
        assert cfg.detection.iou_threshold == 0.425

    def test_export_invalid_config_exit_2(self, tmp_path, capsys):
        bad = tmp_path / "bad.json"
        bad.write_text('{"ocr": {"device": "gpu"}}', encoding="utf-8")
        assert main(["--config", str(bad), "--export-config", str(tmp_path / "o.json")]) == 2
        err = json.loads(capsys.readouterr().err)
        assert err["error"]["code"] == "INVALID_CONFIG"


class TestStubs:
    def test_headless_run_is_now_the_pipeline_not_a_stub(self, capsys):
        """Part 4 replaced the NOT_IMPLEMENTED stub: `--run` executes the pipeline.

        Here the image is missing, so the taxonomy — not the exit code of a stub — is what
        comes back. The real behaviour is covered in tests/test_pipeline.py.
        """
        assert main(["--headless", "--image", "x.png", "--run"]) == 10      # IO_ERROR
        err = json.loads(capsys.readouterr().err)
        assert err["error"]["code"] == "IO_ERROR"

    def test_headless_without_an_action_is_an_argument_error(self, capsys):
        """Part 7 gave headless a second action, so this now names both instead of shrugging."""
        assert main(["--headless"]) == 2                      # INVALID_ARGUMENT
        err = json.loads(capsys.readouterr().err)
        assert err["error"]["code"] == "INVALID_ARGUMENT"
        assert "--run" in err["error"]["message"] and "--serve" in err["error"]["message"]

    def test_gui_smoke_builds_the_window_and_exits(self, monkeypatch, capsys):
        """`--gui-smoke` is the CI-safe way to prove the GUI comes up (Part 5)."""
        pytest.importorskip("PySide6")
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        assert main(["--mock", "--gui-smoke"]) == 0
        out = capsys.readouterr().out
        assert "GUI smoke OK" in out and "engines=mock" in out and "models=5" in out

    def test_gui_without_pyside6_is_a_structured_error(self, monkeypatch, capsys):
        """No Qt must mean a hint and an exit code, never a traceback (D8's discipline)."""
        # Both entries must be poisoned: with `PySide6.QtWidgets` still cached, a
        # `from PySide6.QtWidgets import QApplication` would succeed and start the loop.
        monkeypatch.setitem(sys.modules, "PySide6", None)
        monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        assert main(["--mock"]) == 8                            # ENGINE_LOAD_FAILED
        err = json.loads(capsys.readouterr().err)
        assert err["error"]["code"] == "ENGINE_LOAD_FAILED"
        details = err["error"]["details"]
        assert "PySide6" in details["hint"]
        assert any("--headless" in step for step in details["steps"])

    def test_verbose_prints_resolved_on_stderr(self, tmp_path, capsys):
        """`--verbose` must never pollute stdout: `--run` and `--export-config` share it."""
        target = tmp_path / "cfg.json"
        assert main(["--verbose", "--model", "yolo12n_animetext",
                     "--export-config", str(target)]) == 0
        captured = capsys.readouterr()
        assert "Resolved configuration" in captured.err
        assert "yolo12n_animetext" in captured.err
        assert "Resolved configuration" not in captured.out


class TestEmptyOptionValues:
    """An option that chooses an action, given `""`, is a mistake about *this* command — not a
    request to run the default one (D93).

    Every case here used to do something else entirely: `--warm-devices ""` and
    `--export-config ""` opened the window, `--out ""` wrote nothing, and all three exited 0.
    A value the user typed is never the same as an option they did not type.
    """

    @pytest.mark.parametrize("argv,option", [
        (["--warm-devices", ""], "--warm-devices"),
        (["--warm-devices", "   "], "--warm-devices"),
        (["--export-config", ""], "--export-config"),
        (["--validate-config", ""], "--validate-config"),
        (["--dump-mcp-tools", ""], "--dump-mcp-tools"),
        (["--headless", "--mock", "--run", "--image", "page.png", "--out", ""], "--out"),
    ])
    def test_an_empty_action_value_is_refused_not_ignored(self, argv, option, monkeypatch,
                                                          capsys) -> None:
        def no_window(*_a, **_k):
            raise AssertionError(f"{argv} fell through to the default action (the window)")

        monkeypatch.setattr("animetext_studio.gui.app.run_gui", no_window)
        assert main(argv) == EXIT_CODES[ErrorCode.INVALID_ARGUMENT]
        err = json.loads(capsys.readouterr().err)["error"]
        assert err["code"] == "INVALID_ARGUMENT", err
        assert err["details"]["option"] == option
        assert option in err["message"]

    def test_an_empty_config_path_is_still_unset_not_a_refusal(self, capsys) -> None:
        """The deliberate exception, pinned: `--config ""` means "no config file", which is the
        same answer a blank environment variable gives (D30). Only the options that decide an
        action or a destination are refused — not every option that takes a string."""
        assert main(["--config", "", "--model-status"]) == 0
        assert "models_dir" in json.loads(capsys.readouterr().out)


class TestDumpMcpTools:
    """`--dump-mcp-tools PATH` writes the catalog to PATH — and only there (D94)."""

    def test_a_path_gets_the_copy_and_the_shipped_catalog_is_left_alone(self, tmp_path, capsys):
        from animetext_studio.mcp.server import catalog_path

        shipped, target = catalog_path(), tmp_path / "copy.json"
        before = shipped.read_bytes()
        assert main(["--dump-mcp-tools", str(target)]) == 0
        assert target.read_bytes() == before, "the copy is the catalog the server publishes"
        assert shipped.read_bytes() == before, "a copy asked for elsewhere must not rewrite it"
        assert str(target) in capsys.readouterr().out

    def test_the_bare_flag_regenerates_the_shipped_catalog_and_prints_it(self, capsys):
        assert main(["--dump-mcp-tools"]) == 0
        assert json.loads(capsys.readouterr().out)["tools"], "'-' prints the catalog itself"


class TestWarmDevicesFlag:
    """P9.4 — `--warm-devices` (the per-device table) and `--preload-off` (the cold baseline)."""

    def test_the_sweep_prints_a_table_and_exits_zero(self, capsys):
        assert main(["--mock", "--warm-devices", "cpu,cuda"]) == 0
        out = capsys.readouterr().out
        assert out.splitlines()[0].split() == ["requested", "engine", "device", "state",
                                               "load_ms", "probe_ms", "total_ms"]
        assert "4 row(s) over 2 device(s)" in out

    def test_all_means_the_two_usual_devices(self, capsys):
        assert main(["--mock", "--warm-devices", "all"]) == 0
        assert "cuda" in capsys.readouterr().out

    def test_verbose_adds_the_machine_readable_report(self, capsys):
        assert main(["--mock", "--warm-devices", "cpu", "--verbose"]) == 0
        out = capsys.readouterr().out
        payload = json.loads(out[out.index("{"):])
        assert payload["devices"] == ["cpu"]
        assert payload["outcomes"][0]["requested_device"] == "cpu"

    def test_an_unknown_device_is_a_structured_error(self, capsys):
        code = main(["--mock", "--warm-devices", "tpu"])
        err = json.loads([l for l in capsys.readouterr().err.splitlines()
                          if l.startswith("{")][0])["error"]
        assert code != 0
        assert err["code"] == "INVALID_ARGUMENT"
        assert err["details"]["unknown"] == ["tpu"]

    def test_preload_and_preload_off_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as ei:
            main(["--mock", "--preload", "--preload-off"])
        assert ei.value.code == 2

    def test_preload_off_is_the_cold_baseline_even_read_as_a_flag(self):
        from animetext_studio.core.cli import parse_args
        assert parse_args(["--preload"]).preload is True
        assert parse_args(["--preload-off"]).preload is False
        assert parse_args([]).preload is False
