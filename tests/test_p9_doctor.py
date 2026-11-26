"""P9.5 — `--doctor`: the read-only environment report, and the two upstream issues.

The doctor exists because two `dghs-imgutils` behaviours produced symptoms that pointed
everywhere but at themselves (a `pip install` inside an import; `ast.Str` gone in Python
3.14). It is a diagnostic, which means it is held to a stricter rule than the rest of the
app: **it may describe the machine and nothing else** (decision D84). Half of this file is
therefore tripwires that fail if the doctor installs, downloads, writes or imports a
backend — the other half fabricates broken machines, because a report nobody can test
against a broken machine is a list of strings.

The last class pins `docs/UPSTREAM.md`: the two issues are drafts, and a draft that
misquotes the upstream source is worse than no draft.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from animetext_studio.__main__ import main
from animetext_studio.core.config import AppConfig
from animetext_studio.core.doctor import (
    FAIL, OK, SKIP, WARN, Context, DoctorReport, Finding, Probes, UPSTREAM_AST,
    UPSTREAM_AUTO_INSTALL, run_doctor,
)
from animetext_studio.core.model_store import MANIFEST_NAME
from animetext_studio.core.types import ErrorCode

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DOC = ROOT / "docs" / "UPSTREAM.md"
STATES = {OK, WARN, FAIL, SKIP}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def probes(**over) -> Probes:
    """A machine where nothing optional is installed, then patched per test.

    The default is deliberately bleak: no dists, no modules, no GPU, no token, no font.
    A test then hands back exactly the one thing it is talking about, so a check can
    never pass because the *rest* of the fabricated machine happens to look healthy.
    """
    base = dict(python_version=lambda: (3, 13, 0), dist_version=lambda name: None,
                module_present=lambda name: False, which=lambda name: None,
                disk_free_bytes=lambda path: 50 * 1024 ** 3, token_file=lambda: None,
                env=lambda key: None, font_face=lambda: (None, False))
    base.update(over)
    return Probes(**base)


def dists(**table) -> Probes:
    """A machine with exactly these distributions installed."""
    return probes(dist_version=lambda name: table.get(name))


def find(report: DoctorReport, name: str) -> Finding:
    for finding in report.findings:
        if finding.check == name:
            return finding
    raise AssertionError(f"no check named {name!r}; got "
                         f"{[f.check for f in report.findings]}")


def states(report: DoctorReport) -> dict[str, str]:
    return {f.check: f.severity for f in report.findings}


# ---------------------------------------------------------------------------
# The report is a document
# ---------------------------------------------------------------------------

class TestTheReportIsADocument:
    def test_every_finding_has_a_state_and_a_sentence(self):
        report = run_doctor(AppConfig.default(), probes=probes())
        assert report.findings, "the doctor reported nothing at all"
        for finding in report.findings:
            assert finding.check and finding.check == finding.check.strip()
            assert finding.severity in STATES, finding
            assert finding.detail.strip(), f"{finding.check} has no detail"

    def test_every_blocking_finding_says_what_to_do(self):
        """A FAIL without a fix is a shrug, and a shrug is not a doctor."""
        for argv, cfg in ((False, AppConfig.default()),):
            report = run_doctor(cfg, probes=probes())
            for finding in report.blockers:
                assert finding.hint.strip(), f"{finding.check} fails without a hint"

    def test_text_lists_every_check_and_a_verdict(self):
        report = run_doctor(AppConfig.default(), probes=probes())
        text = report.text()
        for finding in report.findings:
            assert finding.check in text
        assert "RESULT: not usable" in text
        assert "checks:" in text

    def test_json_and_text_describe_the_same_examination(self):
        report = run_doctor(AppConfig.default(), probes=probes(), mock=True)
        payload = json.loads(json.dumps(report.to_dict()))
        assert [c["check"] for c in payload["checks"]] == \
               [f.check for f in report.findings]
        assert payload["usable"] == report.ok
        assert payload["exit_code"] == report.exit_code
        assert payload["summary"]["ok"] + payload["summary"]["fail"] \
               + payload["summary"]["warn"] + payload["summary"]["skip"] \
               == len(report.findings)
        assert payload["mock"] is True and payload["version"]

    def test_the_report_is_json_serialisable_in_full(self):
        payload = run_doctor(AppConfig.default(), probes=probes()).to_dict()
        json.dumps(payload)                       # raises on any stray object

    def test_no_finding_excepts_on_a_probe_that_lies(self):
        """A probe that raises is a broken machine, not a broken doctor."""
        class Boom(Probes):
            pass

        broken = Probes(dist_version=lambda name: (_ for _ in ()).throw(RuntimeError("nope")))
        report = run_doctor(AppConfig.default(), probes=broken)
        assert report.findings                        # still answered


# ---------------------------------------------------------------------------
# D84: read-only, offline-safe
# ---------------------------------------------------------------------------

class TestTheDoctorNeverDoesTheThingItIsDiagnosing:
    def test_no_socket_is_opened(self, monkeypatch):
        def tripwire(*args, **kwargs):
            raise AssertionError("the doctor touched the network")

        monkeypatch.setattr(socket, "socket", tripwire)
        monkeypatch.setattr(socket, "create_connection", tripwire)
        monkeypatch.setattr(socket, "getaddrinfo", tripwire)
        report = run_doctor(AppConfig.default(), probes=probes())
        assert report.findings

    def test_no_process_is_spawned(self, monkeypatch):
        """No pip, no nvidia-smi, no `fetch_models.py` in a subprocess.

        The doctor reports the commands to run; it does not run them.
        """
        def tripwire(*args, **kwargs):
            raise AssertionError("the doctor spawned a process")

        monkeypatch.setattr(subprocess, "Popen", tripwire)
        monkeypatch.setattr(os, "system", tripwire)
        monkeypatch.setattr(os, "spawnl", tripwire)
        report = run_doctor(AppConfig.default(), probes=probes())
        assert report.findings

    def test_nothing_is_written_anywhere(self, tmp_path):
        bundle = tmp_path / "bundle"
        (bundle / "animetext-yolo" / "yolo12m_animetext").mkdir(parents=True)
        (bundle / "animetext-yolo" / "yolo12m_animetext" / "model.onnx").write_bytes(
            b"\0" * 4096)
        config = AppConfig.default()
        config.paths.models_dir = str(bundle)

        before = sorted((p.relative_to(tmp_path), p.stat().st_size, p.stat().st_mtime_ns)
                        for p in tmp_path.rglob("*") if p.is_file())
        report = run_doctor(config, probes=probes())
        after = sorted((p.relative_to(tmp_path), p.stat().st_size, p.stat().st_mtime_ns)
                       for p in tmp_path.rglob("*") if p.is_file())
        assert before == after
        assert find(report, "bundle.detection").severity == OK

    def test_a_read_only_bundle_directory_is_still_reportable(self, tmp_path):
        bundle = tmp_path / "ro"
        (bundle / "animetext-yolo" / "yolo12m_animetext").mkdir(parents=True)
        (bundle / "animetext-yolo" / "yolo12m_animetext" / "model.onnx").write_bytes(
            b"\0" * 4096)
        os.chmod(bundle, 0o500)
        try:
            config = AppConfig.default()
            config.paths.models_dir = str(bundle)
            report = run_doctor(config, probes=probes())
            assert find(report, "bundle.detection").severity == OK
        finally:
            os.chmod(bundle, 0o700)

    def test_a_broken_bundle_directory_never_raises(self, tmp_path):
        config = AppConfig.default()
        config.paths.models_dir = str(tmp_path / "does" / "not" / "exist")
        report = run_doctor(config, probes=probes())
        assert find(report, "bundle").severity == WARN           # online: fetchable
        assert find(report, "bundle").code is None

    def test_the_doctor_does_not_import_a_backend(self):
        """In a clean interpreter: importing imgutils is the fault being diagnosed.

        Run in a subprocess with the three backends poisoned, so the assertion is about
        the doctor and not about whatever the rest of the suite loaded.
        """
        script = (
            "import sys\n"
            "for name in ('imgutils', 'torch', 'onnxruntime', 'transformers'):\n"
            "    sys.modules[name] = None\n"
            "from animetext_studio.core.doctor import run_doctor\n"
            "from animetext_studio.core.config import AppConfig\n"
            "report = run_doctor(AppConfig.default())\n"
            "leaked = [n for n in ('imgutils', 'torch', 'onnxruntime')\n"
            "          if type(sys.modules.get(n)).__name__ != 'NoneType']\n"
            "print('LEAKED', leaked)\n"
            "print('CHECKS', len(report.findings))\n"
            "print('RESULT', 'usable' if report.ok else 'not usable')\n"
        )
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, cwd=str(ROOT), timeout=120)
        assert out.returncode == 0, out.stderr
        assert "LEAKED []" in out.stdout, out.stdout
        assert "CHECKS" in out.stdout and "RESULT" in out.stdout


# ---------------------------------------------------------------------------
# The Python floor and the 3.14 ceiling
# ---------------------------------------------------------------------------

class TestPython:
    @pytest.mark.parametrize("version,want", [
        ((3, 9, 18), FAIL), ((3, 10, 0), OK), ((3, 12, 7), OK), ((3, 13, 0), OK),
        ((3, 14, 0), FAIL), ((3, 15, 1), FAIL),
    ])
    def test_the_floor_and_the_imgutils_ceiling(self, version, want):
        report = run_doctor(AppConfig.default(),
                            probes=probes(python_version=lambda: version))
        assert find(report, "python").severity == want, find(report, "python").detail

    def test_3_14_names_the_upstream_cause_and_the_draft(self):
        finding = find(run_doctor(AppConfig.default(),
                                  probes=probes(python_version=lambda: (3, 14, 0))),
                       "python")
        assert "ast.Str" in finding.detail
        assert UPSTREAM_AST in finding.detail
        assert "docs/UPSTREAM.md" in finding.hint
        assert finding.code is ErrorCode.ENGINE_LOAD_FAILED

    def test_under_mock_a_modern_python_does_not_block(self):
        """The fakes need no backend, so 3.14 is a note, not a wall."""
        report = run_doctor(AppConfig.default(), mock=True,
                            probes=healthy(python_version=lambda: (3, 14, 0)))
        assert find(report, "python").severity == SKIP
        assert report.ok


# ---------------------------------------------------------------------------
# The two quirks, as checks
# ---------------------------------------------------------------------------

class TestTheImportTimeInstall:
    def test_missing_onnxruntime_beside_imgutils_is_the_promise_bender(self):
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"dghs-imgutils": "0.19.0"}))
        finding = find(report, "onnxruntime")
        assert finding.severity == FAIL
        assert "pip install at import time" in finding.detail
        assert UPSTREAM_AUTO_INSTALL in finding.detail
        assert "pip install onnxruntime" in finding.hint
        assert finding.code is ErrorCode.ENGINE_LOAD_FAILED

    def test_the_hint_points_at_the_draft_issue(self):
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"dghs-imgutils": "0.19.0"}))
        assert "UPSTREAM.md#1" in find(report, "onnxruntime").hint

    def test_onnxruntime_present_shuts_the_hazard_up(self):
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"dghs-imgutils": "0.19.0",
                                            "onnxruntime": "1.32.0"}))
        finding = find(report, "onnxruntime")
        assert finding.severity == OK
        assert "nothing to install" in finding.detail

    def test_the_gpu_build_counts_as_an_answer(self):
        report = run_doctor(AppConfig.default(),
                            probes=probes(dist_version=lambda n: {
                                "dghs-imgutils": "0.19.0"}.get(n),
                                module_present=lambda n: n == "onnxruntime-gpu"))
        assert find(report, "onnxruntime").severity == OK

    def test_without_imgutils_there_is_nothing_to_install(self):
        report = run_doctor(AppConfig.default(), probes=probes())
        assert find(report, "onnxruntime").severity == SKIP

    def test_mock_reports_the_hazard_but_does_not_block(self):
        table = {"dghs-imgutils": "0.19.0", "numpy": "1.26.4"}
        report = run_doctor(AppConfig.default(), mock=True,
                            probes=healthy(dist_version=lambda name: table.get(name),
                                           module_present=lambda name: False))
        assert find(report, "onnxruntime").severity == SKIP
        assert "pip install" in find(report, "onnxruntime").detail
        assert report.exit_code == 0


class TestTheNumpyPin:
    def test_the_pin_is_read_from_metadata_not_memory(self, monkeypatch):
        import animetext_studio.core.doctor as doctor

        monkeypatch.setattr(doctor.md, "requires",
                            lambda name: ["numpy (>=1.20)", "numpy (<2)",
                                          "pillow", "requests ; python_version >= \"3.8\""]
                            if name == "dghs-imgutils" else [])
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"dghs-imgutils": "0.19.0", "numpy": "2.5.5"}))
        finding = find(report, "numpy")
        assert finding.severity == WARN
        assert "numpy <2" in finding.detail
        assert "--no-deps" in finding.detail
        assert finding.data["pins"] == ["numpy >=1.20", "numpy <2"]

    def test_a_numpy_that_satisfies_the_pin_is_quiet(self, monkeypatch):
        import animetext_studio.core.doctor as doctor

        monkeypatch.setattr(doctor.md, "requires",
                            lambda name: ["numpy (<2)"] if name == "dghs-imgutils" else [])
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"dghs-imgutils": "0.19.0", "numpy": "1.26.4"}))
        assert find(report, "numpy").severity == OK

    def test_no_imgutils_no_opinion_about_the_pin(self, monkeypatch):
        import animetext_studio.core.doctor as doctor

        monkeypatch.setattr(doctor.md, "requires", lambda name: [])
        report = run_doctor(AppConfig.default(), probes=dists(**{"numpy": "2.5.5"}))
        assert find(report, "numpy").severity == OK


# ---------------------------------------------------------------------------
# torch, CUDA, and the two optional doors
# ---------------------------------------------------------------------------

class TestTorchAndCuda:
    def test_no_torch_means_no_ocr(self):
        report = run_doctor(AppConfig.default(), probes=probes())
        assert find(report, "torch").severity == FAIL
        assert find(report, "torch").code is ErrorCode.ENGINE_LOAD_FAILED
        assert '.[ocr]' in find(report, "torch").hint

    def test_a_cpu_wheel_asked_for_cuda_is_called_out(self):
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"torch": "2.11.0", "transformers": "5.5.4"}))
        # default ocr.device is "auto", so a CPU wheel is fine...
        assert find(report, "torch").severity == OK
        config = AppConfig.default()
        config.ocr.device = "cuda"
        report = run_doctor(config,
                            probes=dists(**{"torch": "2.11.0", "transformers": "5.5.4"}))
        assert find(report, "torch").severity == FAIL
        assert "cu128" in find(report, "torch").hint
        assert find(report, "cuda").severity == FAIL

    def test_a_cuda_build_is_reported_as_such(self):
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"torch": "2.11.0+cu130",
                                            "transformers": "5.5.4"}))
        assert find(report, "torch").severity == OK
        assert find(report, "torch").data["cuda_build"] == "cu130"

    def test_the_gpu_answer_is_handed_to_the_sweep(self):
        """The doctor describes; --warm-devices measures."""
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"torch": "2.11.0+cu130",
                                            "transformers": "5.5.4"}).which
                                  and probes(dist_version=lambda n: {
                                      "torch": "2.11.0+cu130",
                                      "transformers": "5.5.4"}.get(n),
                                      which=lambda n: "/usr/bin/nvidia-smi"))
        finding = find(report, "cuda")
        assert finding.severity == OK
        assert "--warm-devices" in finding.detail

    def test_a_cpu_only_machine_is_not_a_failure_by_default(self):
        report = run_doctor(AppConfig.default(), probes=probes())
        assert find(report, "cuda").severity == OK


class TestTheOptionalDoors:
    def test_no_pyside_is_a_warning_not_a_wall(self):
        report = run_doctor(AppConfig.default(), probes=probes())
        assert find(report, "PySide6").severity == WARN
        assert "--headless" in find(report, "PySide6").detail

    def test_pyside_under_its_published_name(self):
        report = run_doctor(AppConfig.default(),
                            probes=dists(**{"PySide6_Essentials": "6.12.0"}))
        assert find(report, "PySide6").severity == OK

    def test_an_mcp_older_than_the_server_layout_is_fatal(self):
        report = run_doctor(AppConfig.default(), probes=dists(**{"mcp": "1.9.4"}))
        finding = find(report, "mcp")
        assert finding.severity == FAIL
        assert "2.3" in finding.detail

    def test_an_mcp_bind_outside_loopback_is_warned_about(self):
        config = AppConfig.default()
        config.mcp.host = "0.0.0.0"
        report = run_doctor(config, probes=dists(**{"mcp": "2.3.0"}))
        warnings = [f.detail for f in report.findings if f.check == "mcp endpoint"]
        assert warnings and "NO auth token" in warnings[0]

    def test_loopback_is_silent(self):
        report = run_doctor(AppConfig.default(), probes=dists(**{"mcp": "2.3.0"}))
        assert not [f for f in report.findings if f.check == "mcp endpoint"]


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------

def fake_bundle(root: Path, *, variant: str = "yolo12m_animetext", ocr: bool = True):
    """A bundle that *counts*: ModelStore calls a weight file bundled at >= 1 MB."""
    from animetext_studio.core.model_store import OCR_REQUIRED_FILES

    vdir = root / "animetext-yolo" / variant
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "model.onnx").write_bytes(b"\0" * (1024 * 1024 + 16))
    if ocr:
        ocr_dir = root / "hayai-ocr-v2.7-supernova"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        for fname in OCR_REQUIRED_FILES:
            (ocr_dir / fname).write_bytes(b"\0" * 4096)
    return root


def healthy(**over) -> Probes:
    """A machine with every extra installed; a test overrides the one thing it doubts.

    `probes()` is bleak on purpose, which is right for "what does a missing X say" and
    wrong for "the bundle is the only problem here": a report about a bleak machine also
    fails on imgutils, torch and numpy, and the exit code then names the first of those
    instead of the thing under test.
    """
    table = {"dghs-imgutils": "0.19.0", "onnxruntime": "1.22.0", "numpy": "1.26.4",
             "torch": "2.7.0+cu128", "transformers": "4.57.0",
             "PySide6_Essentials": "6.8.0", "mcp": "2.3.0"}
    base = dict(python_version=lambda: (3, 13, 0),
                dist_version=lambda name: table.get(name),
                requires=lambda name: ["numpy (<2)"] if name == "dghs-imgutils" else [],
                module_present=lambda name: name == "onnxruntime",
                which=lambda name: None,
                disk_free_bytes=lambda path: 50 * 1024 ** 3,
                token_file=lambda: None, env=lambda key: None,
                font_face=lambda: ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                   True))
    base.update(over)
    return Probes(**base)


class TestTheBundle:
    def _config(self, tmp_path, **over):
        config = AppConfig.default()
        config.paths.models_dir = str(tmp_path / "bundle")
        for key, value in over.items():
            setattr(config.paths, key, value)
        return config

    def test_no_bundle_online_is_a_warning(self, tmp_path):
        report = run_doctor(self._config(tmp_path), probes=probes())
        assert find(report, "bundle").severity == WARN
        assert "fetch_models.py" in find(report, "bundle").hint

    def test_no_bundle_offline_is_model_not_found(self, tmp_path):
        report = run_doctor(self._config(tmp_path, offline=True), probes=healthy())
        assert find(report, "bundle").severity == FAIL
        assert find(report, "bundle").code is ErrorCode.MODEL_NOT_FOUND
        assert report.exit_code == 12

    def test_a_complete_bundle_is_green(self, tmp_path):
        fake_bundle(tmp_path / "bundle")
        report = run_doctor(self._config(tmp_path, offline=True), probes=healthy())
        assert find(report, "bundle.detection").severity == OK
        assert find(report, "bundle.ocr").severity == OK
        assert report.ok

    def test_a_variant_that_is_not_bundled_names_the_ones_that_are(self, tmp_path):
        fake_bundle(tmp_path / "bundle", variant="yolo12n_animetext")
        config = self._config(tmp_path, offline=True)
        config.detection.model = "yolo12x_animetext"
        finding = find(run_doctor(config, probes=healthy()), "bundle.detection")
        assert finding.severity == FAIL
        assert "yolo12n_animetext" in finding.detail
        assert "--variants yolo12x_animetext" in finding.hint

    def test_a_recorded_file_that_is_gone_is_reported(self, tmp_path):
        bundle = fake_bundle(tmp_path / "bundle")
        onnx = bundle / "animetext-yolo" / "yolo12m_animetext" / "model.onnx"
        recorded = onnx.stat().st_size
        (bundle / MANIFEST_NAME).write_text(json.dumps({"files": [
            {"relpath": "animetext-yolo/yolo12m_animetext/model.onnx",
             "path": str(onnx), "size": recorded, "sha256": "0" * 64}]}), encoding="utf-8")
        onnx.unlink()                                    # the half-copied bundle
        finding = find(run_doctor(self._config(tmp_path), probes=healthy()),
                       "bundle.manifest")
        assert finding.severity == WARN
        assert "gone" in finding.detail

    def test_a_recorded_size_that_does_not_match_is_fatal(self, tmp_path):
        """A truncated copy is the realistic corruption, and sizes catch it for free."""
        bundle = fake_bundle(tmp_path / "bundle")
        onnx = bundle / "animetext-yolo" / "yolo12m_animetext" / "model.onnx"
        (bundle / MANIFEST_NAME).write_text(json.dumps({"files": [
            {"relpath": "animetext-yolo/yolo12m_animetext/model.onnx",
             "path": str(onnx), "size": 999999, "sha256": "0" * 64}]}), encoding="utf-8")
        finding = find(run_doctor(self._config(tmp_path), probes=healthy()),
                       "bundle.manifest")
        assert finding.severity == FAIL
        assert finding.code is ErrorCode.ENGINE_LOAD_FAILED
        assert "--verify" in finding.hint

    def test_hashing_stays_out_of_the_doctor(self, tmp_path):
        """The manifest is cross-checked by size; sha256 belongs to fetch_models --verify."""
        bundle = fake_bundle(tmp_path / "bundle")
        onnx = bundle / "animetext-yolo" / "yolo12m_animetext" / "model.onnx"
        (bundle / MANIFEST_NAME).write_text(json.dumps({"files": [
            {"relpath": "animetext-yolo/yolo12m_animetext/model.onnx",
             "path": str(onnx), "size": onnx.stat().st_size,
             "sha256": "0" * 64}]}), encoding="utf-8")
        finding = find(run_doctor(self._config(tmp_path), probes=healthy()),
                       "bundle.manifest")
        assert finding.severity == OK
        assert "fetch_models.py --verify" in finding.detail

    def test_a_bundle_without_a_manifest_cannot_prove_itself(self, tmp_path):
        fake_bundle(tmp_path / "bundle")
        finding = find(run_doctor(self._config(tmp_path), probes=healthy()),
                       "bundle.manifest")
        assert finding.severity == WARN
        assert "truncated" in finding.detail

    def test_a_full_disk_is_warned_about(self, tmp_path):
        report = run_doctor(self._config(tmp_path),
                            probes=healthy(disk_free_bytes=lambda p: 100 * 1024 ** 2))
        assert find(report, "disk").severity == WARN
        assert "truncated" in find(report, "disk").hint


# ---------------------------------------------------------------------------
# The token: where it came from, never what it is
# ---------------------------------------------------------------------------

class TestTheToken:
    SECRET = "hf_" + "s" * 30                       # long enough to be believable

    def test_the_value_never_reaches_either_rendering(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", self.SECRET)
        report = run_doctor(AppConfig.default(), probes=probes(
            env=lambda key: self.SECRET if key == "HF_TOKEN" else None))
        assert self.SECRET not in report.text()
        assert self.SECRET not in json.dumps(report.to_dict())
        assert "env:HF_TOKEN" in find(report, "token").detail
        assert f"{len(self.SECRET)} characters" in find(report, "token").detail

    def test_an_explicit_flag_is_attributed_to_the_flag(self):
        report = run_doctor(AppConfig.default(), hf_token=self.SECRET, probes=probes())
        assert "--hf-token" in find(report, "token").detail
        assert self.SECRET not in report.text()

    def test_a_local_token_file_is_named(self, tmp_path):
        path = tmp_path / "HF_token.txt"
        path.write_text(self.SECRET + "\n", encoding="utf-8")
        report = run_doctor(AppConfig.default(),
                            probes=probes(token_file=lambda: path))
        assert "file:" in find(report, "token").detail
        assert self.SECRET not in report.text()

    def test_no_token_and_a_complete_bundle_is_not_a_problem(self, tmp_path):
        fake_bundle(tmp_path / "bundle")
        config = AppConfig.default()
        config.paths.models_dir = str(tmp_path / "bundle")
        report = run_doctor(config, probes=probes())
        finding = find(report, "token")
        assert finding.severity == OK
        assert "none needed" in finding.detail

    def test_no_token_and_a_variant_that_must_come_from_the_hub_warns(self, tmp_path):
        config = AppConfig.default()
        config.paths.models_dir = str(tmp_path / "empty")
        finding = find(run_doctor(config, probes=probes()), "token")
        assert finding.severity == WARN
        assert finding.code is ErrorCode.HF_AUTH_REQUIRED
        assert "check_hf_access.py" in finding.hint


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

class TestFonts:
    def test_no_face_is_a_warning_about_exports(self):
        report = run_doctor(AppConfig.default(), probes=probes())
        assert find(report, "fonts").severity == WARN
        assert "boxes-of-doom" in find(report, "fonts").hint

    def test_a_dejavu_fallback_is_still_a_warning(self):
        report = run_doctor(AppConfig.default(), probes=probes(
            font_face=lambda: ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", False)))
        assert find(report, "fonts").severity == WARN

    def test_a_cjk_face_is_the_answer(self):
        face = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        report = run_doctor(AppConfig.default(),
                            probes=probes(font_face=lambda: (face, True)))
        assert find(report, "fonts").severity == OK
        assert face in find(report, "fonts").detail


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

class TestTheCommand:
    def test_the_flag_exists_in_the_parser(self):
        from animetext_studio.core.cli import build_parser

        options = {opt for action in build_parser()._actions       # noqa: SLF001
                   for opt in action.option_strings}
        assert "--doctor" in options

    def test_it_prints_a_report_and_exits_zero_on_a_healthy_machine(self, capsys):
        assert main(["--mock", "--doctor"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("doctor — animetext-studio")
        assert "RESULT: usable" in out

    def test_verbose_adds_the_machine_readable_document(self, capsys):
        assert main(["--mock", "--doctor", "--verbose"]) == 0
        out = capsys.readouterr().out
        payload = json.loads(out[out.index("{"):])
        assert payload["checks"] and payload["usable"] is True
        assert payload["exit_code"] == 0

    def test_a_blocking_machine_exits_with_the_code_it_predicts(self, tmp_path, capsys):
        code = main(["--doctor", "--offline", "--models-dir", str(tmp_path / "nothing")])
        assert code == 12                                   # MODEL_NOT_FOUND
        assert "RESULT: not usable" in capsys.readouterr().out

    def test_it_is_an_action_not_a_window(self, monkeypatch, capsys):
        """`--doctor` must never fall through into the GUI, Qt or an engine."""
        monkeypatch.setitem(sys.modules, "PySide6", None)
        monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
        assert main(["--mock", "--doctor"]) == 0
        assert "RESULT" in capsys.readouterr().out

    def test_it_reports_the_bundle_it_was_pointed_at(self, tmp_path, capsys):
        fake_bundle(tmp_path / "bundle")
        assert main(["--doctor", "--models-dir", str(tmp_path / "bundle")]) == 0
        out = capsys.readouterr().out
        assert "bundle.detection" in out and "bundle.ocr" in out


# ---------------------------------------------------------------------------
# docs/UPSTREAM.md — the drafts
# ---------------------------------------------------------------------------

class TestTheUpstreamDrafts:
    def test_the_file_exists(self):
        assert UPSTREAM_DOC.is_file(), "docs/UPSTREAM.md is a P9.5 deliverable"

    def test_both_issues_are_present_and_say_what_to_ask_for(self):
        text = UPSTREAM_DOC.read_text(encoding="utf-8")
        assert "IMGUTILS_NO_AUTO_INSTALL" in text
        assert "ast.Constant" in text
        assert "deepghs/imgutils" in text

    def test_both_cite_the_file_and_lines_they_came_from(self):
        text = UPSTREAM_DOC.read_text(encoding="utf-8")
        assert "imgutils/utils/onnxruntime.py" in text
        assert "imgutils/generic/yolo.py" in text
        assert "_ensure_onnxruntime" in text and "_safe_eval_names_str" in text

    def test_it_says_plainly_that_neither_has_been_filed(self):
        text = UPSTREAM_DOC.read_text(encoding="utf-8").lower()
        assert "neither issue has been filed" in text
        assert "github" in text, "the reader must learn where they would go to file them"

    def test_it_says_what_we_do_in_the_meantime(self):
        text = UPSTREAM_DOC.read_text(encoding="utf-8")
        assert text.count("What we do in the meantime") >= 2

    def test_the_quoted_source_is_still_the_installed_source(self):
        """The drafts quote 0.19.0; if the installed copy changes, the drafts are stale.

        A failing test here is good news — upstream fixed something. Re-read the file,
        raise the ceiling in `core/doctor.py`, and retire the row.
        """
        import importlib.util

        spec = importlib.util.find_spec("imgutils")
        if spec is None or spec.origin is None:
            pytest.skip("dghs-imgutils is not installed here")
        pkg = Path(spec.origin).parent
        installer = (pkg / "utils" / "onnxruntime.py").read_text(encoding="utf-8")
        assert "pip_install" in installer, (
            "imgutils no longer installs at import — docs/UPSTREAM.md#1 and the doctor's "
            "onnxruntime finding are both stale")
        yolo = (pkg / "generic" / "yolo.py").read_text(encoding="utf-8")
        assert "ast.Str" in yolo, (
            "imgutils has migrated off ast.Str — docs/UPSTREAM.md#2 is stale and "
            "IMGUTILS_PYTHON_CEILING can move")
