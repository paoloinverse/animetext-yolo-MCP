"""On-premises model store (decision D26) — offline, pure stdlib.

Everything here runs with no network, no token and no [detect]/[ocr] extras: the
bundle is just files on disk, and the store must be usable (and reportable) in a bare
install. Weights are faked with padded files; only *presence and size* matter here —
real weight loading is covered by `-m models`.
"""
from __future__ import annotations

import json
import struct
import sys
import threading
import time
from pathlib import Path
import types

import pytest

from animetext_studio.core import model_store as ms
from animetext_studio.core.config import KNOWN_MODELS
from animetext_studio.core.types import ErrorCode, OCR_REPO, StudioError

MB = 1 << 20


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_variant(root, name, *, onnx=2 * MB, aux=()):
    """Create <root>/animetext-yolo/<name>/ with a plausible model.onnx (+ aux files)."""
    vdir = root / ms.DETECTION_DIRNAME / name
    vdir.mkdir(parents=True, exist_ok=True)
    if onnx:
        (vdir / "model.onnx").write_bytes(b"\0" * onnx)
    for fname in aux:
        (vdir / fname).write_text("{}", encoding="utf-8")
    return vdir


def make_ocr(root, *, skip=(), weights=2 * MB):
    """Create a complete (fake) OCR snapshot under <root>/hayai-ocr-v2.7-supernova."""
    ocr = root / ms.OCR_DIRNAME
    ocr.mkdir(parents=True, exist_ok=True)
    for fname in ms.OCR_REQUIRED_FILES:
        if fname in skip:
            continue
        path = ocr / fname
        path.write_bytes(b"\0" * weights) if fname.endswith(".safetensors") \
            else path.write_text("{}", encoding="utf-8")
    return ocr


def fake_safetensors(path, tensors=(("a", "F32", 8), ("b", "F32", 4)), pad=0):
    """Write a structurally valid .safetensors (header + tensor blob)."""
    header, offset, blob = {}, 0, bytearray()
    for name, dtype, size in tensors:
        header[name] = {"dtype": dtype, "shape": [size // 4], "data_offsets": [offset, offset + size]}
        blob += b"\0" * size
        offset += size
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * pad                                   # real writers pad to 8 bytes
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(blob))
    return path


# ---------------------------------------------------------------------------
# root resolution
# ---------------------------------------------------------------------------

class TestRootResolution:
    def test_package_default(self, monkeypatch):
        monkeypatch.delenv(ms.MODELS_DIR_ENV, raising=False)
        assert ms.default_models_dir() == ms.package_models_dir()
        assert ms.package_models_dir().name == "models"
        assert ms.package_models_dir().parent.name == "animetext_studio"

    def test_env_overrides_package_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ms.MODELS_DIR_ENV, str(tmp_path / "elsewhere"))
        assert ms.default_models_dir() == tmp_path / "elsewhere"

    def test_blank_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv(ms.MODELS_DIR_ENV, "   ")
        assert ms.default_models_dir() == ms.package_models_dir()

    def test_explicit_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ms.MODELS_DIR_ENV, str(tmp_path / "env"))
        assert ms.resolve_models_dir(str(tmp_path / "flag")) == tmp_path / "flag"

    def test_explicit_expands_user_and_relative(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ms.MODELS_DIR_ENV, str(tmp_path / "env"))   # must be ignored
        monkeypatch.chdir(tmp_path)
        assert ms.resolve_models_dir("bundle") == tmp_path / "bundle"
        assert ms.resolve_models_dir("~/bundle") == (
            ms.Path.home() / "bundle").resolve()

    def test_store_uses_the_same_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ms.MODELS_DIR_ENV, str(tmp_path / "env"))
        assert ms.ModelStore().root == tmp_path / "env"
        assert ms.ModelStore(tmp_path / "flag").root == tmp_path / "flag"
        assert ms.ModelStore().offline is False
        assert ms.ModelStore(offline=True).offline is True


# ---------------------------------------------------------------------------
# detection variants
# ---------------------------------------------------------------------------

class TestDetectionVariants:
    def test_missing_root_is_not_an_error(self, tmp_path):
        store = ms.ModelStore(tmp_path / "nope")
        assert store.exists is False
        assert store.bundled_detection_variants() == []
        assert store.has_detection_variant("yolo12m_animetext") is False

    def test_only_usable_variants_count(self, tmp_path):
        make_variant(tmp_path, "yolo12m_animetext")
        make_variant(tmp_path, "yolo12n_animetext", onnx=0)        # empty -> ignored
        (tmp_path / ms.DETECTION_DIRNAME / "not_a_dir.txt").write_text("x")
        store = ms.ModelStore(tmp_path)
        assert store.bundled_detection_variants() == ["yolo12m_animetext"]
        assert store.has_detection_variant("yolo12n_animetext") is False

    def test_truncated_onnx_is_rejected(self, tmp_path):
        """A half-copied download must read as 'not bundled', not as a usable model."""
        make_variant(tmp_path, "yolo12s_animetext", onnx=1024)     # < _MIN_WEIGHTS_BYTES
        assert ms.ModelStore(tmp_path).bundled_detection_variants() == []

    def test_known_variants_first_then_extras(self, tmp_path):
        for name in ("zz_custom", "yolo12x_animetext", "aa_custom", "yolo12n_animetext"):
            make_variant(tmp_path, name)
        assert ms.ModelStore(tmp_path).bundled_detection_variants() == [
            "yolo12n_animetext", "yolo12x_animetext", "aa_custom", "zz_custom"]

    def test_require_returns_bundled_path(self, tmp_path):
        vdir = make_variant(tmp_path, "yolo12m_animetext", aux=("threshold.json",))
        store = ms.ModelStore(tmp_path)
        assert store.require_detection_file("yolo12m_animetext") == vdir / "model.onnx"
        assert store.require_detection_file("yolo12m_animetext", "threshold.json") \
            == vdir / "threshold.json"

    def test_require_offline_points_at_fetch_models(self, tmp_path):
        store = ms.ModelStore(tmp_path, offline=True)
        with pytest.raises(StudioError) as ei:
            store.require_detection_file("yolo12l_animetext")
        err = ei.value
        assert err.code is ErrorCode.MODEL_NOT_FOUND and err.exit_code == 12
        assert err.details["offline"] is True
        assert err.details["bundled_variants"] == []
        assert any("fetch_models.py" in s for s in err.details["steps"])

    def test_require_online_explains_the_hub_path(self, tmp_path):
        with pytest.raises(StudioError) as ei:
            ms.ModelStore(tmp_path).require_detection_file("yolo12l_animetext")
        assert any("Hugging Face" in s for s in ei.value.details["steps"])

    def test_variant_status_reports_every_file(self, tmp_path):
        make_variant(tmp_path, "yolo12m_animetext", onnx=3 * MB, aux=("labels.json",))
        v = ms.ModelStore(tmp_path).detection_variant_status("yolo12m_animetext")
        assert v["bundled"] is True and v["complete"] is True
        assert v["bytes"] == 3 * MB
        assert v["files"]["model.onnx"] is True
        assert v["files"]["labels.json"] is True
        assert v["files"]["threshold.json"] is False
        missing = ms.ModelStore(tmp_path).detection_variant_status("yolo12x_animetext")
        assert missing["bundled"] is False and missing["bytes"] == 0


# ---------------------------------------------------------------------------
# OCR snapshot
# ---------------------------------------------------------------------------

class TestOcrSnapshot:
    def test_complete_snapshot(self, tmp_path):
        make_ocr(tmp_path)
        store = ms.ModelStore(tmp_path)
        assert store.missing_ocr_files() == []
        assert store.has_ocr_snapshot() is True
        assert store.require_ocr_snapshot() == tmp_path / ms.OCR_DIRNAME

    def test_missing_file_is_named(self, tmp_path):
        make_ocr(tmp_path, skip=("tokenizer.json", "modeling_hayai.py"))
        store = ms.ModelStore(tmp_path)
        assert store.has_ocr_snapshot() is False
        assert store.missing_ocr_files() == ["modeling_hayai.py", "tokenizer.json"]
        with pytest.raises(StudioError) as ei:
            store.require_ocr_snapshot()
        assert ei.value.code is ErrorCode.MODEL_NOT_FOUND
        assert ei.value.details["missing"] == ["modeling_hayai.py", "tokenizer.json"]
        assert any(str(tmp_path / ms.OCR_DIRNAME) in s for s in ei.value.details["steps"])

    def test_empty_file_counts_as_missing(self, tmp_path):
        ocr = make_ocr(tmp_path)
        (ocr / "config.json").write_text("")
        assert store_missing(tmp_path) == ["config.json"]

    def test_status_records_provenance(self, tmp_path):
        ocr = make_ocr(tmp_path)
        (ocr / ".origin.json").write_text(json.dumps(
            {"repo_id": OCR_REPO, "revision": "abc123"}), encoding="utf-8")
        info = ms.ModelStore(tmp_path).ocr_status()
        assert info["bundled"] is True and info["revision"] == "abc123"
        assert info["repo_id"] == OCR_REPO

    def test_status_falls_back_to_pinned_revision(self, tmp_path):
        make_ocr(tmp_path)
        info = ms.ModelStore(tmp_path).ocr_status()
        assert info["revision"] == ms.OCR_REVISION      # pinned in core.types

    def test_broken_origin_does_not_raise(self, tmp_path):
        ocr = make_ocr(tmp_path)
        (ocr / ".origin.json").write_text("{not json", encoding="utf-8")
        assert ms.ModelStore(tmp_path).ocr_status()["revision"] is None


def store_missing(root):
    return ms.ModelStore(root).missing_ocr_files()


# ---------------------------------------------------------------------------
# safetensors integrity check
# ---------------------------------------------------------------------------

class TestVerifySafetensors:
    def test_valid_file(self, tmp_path):
        path = fake_safetensors(tmp_path / "m.safetensors")
        rep = ms.verify_safetensors(path)
        assert rep["ok"] is True and rep["tensors"] == 2
        assert rep["dtype"] == "F32" and rep["delta_bytes"] == 0
        assert rep["declared_bytes"] == path.stat().st_size

    def test_padded_header_is_fine(self, tmp_path):
        path = fake_safetensors(tmp_path / "m.safetensors", pad=8)
        assert ms.verify_safetensors(path)["ok"] is True

    def test_truncation_is_caught(self, tmp_path):
        path = fake_safetensors(tmp_path / "m.safetensors")
        data = path.read_bytes()
        path.write_bytes(data[:-4])                      # tail of the tensor blob lost
        rep = ms.verify_safetensors(path)
        assert rep["ok"] is False and "truncated" in rep["error"]
        path.write_bytes(data[:64])                      # header itself cut short
        assert "truncated" not in ms.verify_safetensors(path)["error"]

    def test_lfs_pointer_is_caught(self, tmp_path):
        path = tmp_path / "m.safetensors"
        path.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
        rep = ms.verify_safetensors(path)
        assert rep["ok"] is False and rep["error"]

    def test_missing_file(self, tmp_path):
        rep = ms.verify_safetensors(tmp_path / "gone.safetensors")
        assert rep["ok"] is False and rep["error"] == "file not found"

    def test_ocr_status_flags_broken_weights(self, tmp_path):
        ocr = make_ocr(tmp_path)
        fake_safetensors(ocr / "model.safetensors", tensors=(("a", "F32", 64),))
        assert ms.ModelStore(tmp_path).ocr_status()["weights_ok"] is True
        ocr_data = (ocr / "model.safetensors").read_bytes()
        (ocr / "model.safetensors").write_bytes(ocr_data[:-32])
        info = ms.ModelStore(tmp_path).ocr_status()
        assert info["weights_ok"] is False and "truncated" in info["weights_error"]


# ---------------------------------------------------------------------------
# status report + manifest
# ---------------------------------------------------------------------------

class TestStatusReport:
    def test_shape_on_an_empty_dir(self, tmp_path):
        st = ms.ModelStore(tmp_path, offline=True).status()
        assert st["models_dir"] == str(tmp_path)
        assert st["exists"] is True and st["offline"] is True
        assert st["detection_bundled"] == [] and st["manifest"] is None
        assert set(st["detection_variants"]) == set(KNOWN_MODELS)   # all five listed
        assert st["total_bytes"] == 0
        assert st["ocr"]["bundled"] is False and st["ocr"]["missing"]

    def test_totals_and_bundled_list(self, tmp_path):
        make_variant(tmp_path, "yolo12n_animetext", onnx=2 * MB)
        make_variant(tmp_path, "yolo12m_animetext", onnx=4 * MB)
        make_ocr(tmp_path, weights=MB)
        st = ms.ModelStore(tmp_path).status()
        assert st["detection_bundled"] == ["yolo12n_animetext", "yolo12m_animetext"]
        assert st["ocr"]["bundled"] is True
        assert st["total_bytes"] >= 6 * MB

    def test_missing_root_still_reports(self, tmp_path):
        st = ms.ModelStore(tmp_path / "absent").status()
        assert st["exists"] is False and st["detection_variants"]

    def test_manifest_roundtrip(self, tmp_path):
        onnx = make_variant(tmp_path, "yolo12m_animetext") / "model.onnx"
        entry = {"path": str(onnx), "size": onnx.stat().st_size, "sha256": "ab" * 32}
        ms.write_manifest(tmp_path, [entry], extra={"note": "hi"})
        data = ms._read_manifest(tmp_path)
        # P8.7 (D76): an entry now also carries the path relative to the models root, so the
        # bundle's passport survives the copy. The absolute `path` stays — it is provenance.
        assert data["files"] == [{**entry,
                                  "relpath": "animetext-yolo/yolo12m_animetext/model.onnx"}]
        assert data["note"] == "hi"
        assert Path(data["models_root"]).is_absolute()
        assert "fetch_models.py" in data["tool"]
        assert (tmp_path / ms.MANIFEST_NAME).is_file()

    def test_partial_run_keeps_earlier_provenance(self, tmp_path):
        """`--ocr` alone must not erase the detection entries (and vice versa)."""
        det_file = make_variant(tmp_path, "yolo12n_animetext") / "model.onnx"
        ms.write_manifest(tmp_path, [{"path": str(det_file), "size": det_file.stat().st_size}])
        ocr_file = make_ocr(tmp_path) / "config.json"
        ms.write_manifest(tmp_path, [{"path": str(ocr_file), "size": 2}],
                          extra={"ocr": True})
        paths = [f["path"] for f in ms._read_manifest(tmp_path)["files"]]
        assert str(det_file) in paths and str(ocr_file) in paths

    def test_phantom_entries_are_never_recorded(self, tmp_path):
        """A failed download must not leave a 0-byte claim in the manifest."""
        real = make_variant(tmp_path, "yolo12n_animetext") / "model.onnx"
        ms.write_manifest(tmp_path, [{"path": str(real), "size": real.stat().st_size}])
        ghost = tmp_path / "animetext-yolo" / "yolo12x_animetext" / "labels.json"
        ms.write_manifest(tmp_path, [{"path": str(ghost), "size": 0, "sha256": None}])
        paths = [f["path"] for f in ms._read_manifest(tmp_path)["files"]]
        assert paths == [str(real)]

    def test_merge_can_be_disabled(self, tmp_path):
        a = make_variant(tmp_path, "yolo12n_animetext") / "model.onnx"
        ms.write_manifest(tmp_path, [{"path": str(a), "size": 1}])
        ms.write_manifest(tmp_path, [], merge=False)
        assert ms._read_manifest(tmp_path)["files"] == []

    def test_corrupt_or_absent_manifest_reads_as_none(self, tmp_path):
        assert ms._read_manifest(tmp_path) is None
        (tmp_path / ms.MANIFEST_NAME).write_text("[1,2]", encoding="utf-8")   # not an object
        assert ms._read_manifest(tmp_path) is None
        (tmp_path / ms.MANIFEST_NAME).write_text("{oops", encoding="utf-8")
        assert ms._read_manifest(tmp_path) is None

    def test_status_is_json_serialisable(self, tmp_path):
        make_variant(tmp_path, "yolo12m_animetext")
        json.dumps(ms.ModelStore(tmp_path).status())


# ---------------------------------------------------------------------------
# engine integration (fakes; the real thing is `-m models`)
# ---------------------------------------------------------------------------

class StubYOLO:
    """Minimal stand-in for imgutils' YOLOModel — deliberately *without* its private
    `_global_lock` / `_models` / `_model_types` fields, which is exactly the case the
    bundled subclass must survive. Keeps these tests working in a bare install."""

    names: list[str] = []
    names_error: Exception | None = None

    def __init__(self, repo_id, hf_token=None):
        self.repo_id, self.hf_token = repo_id, hf_token

    @property
    def model_names(self):
        if StubYOLO.names_error is not None:
            raise StubYOLO.names_error
        return list(StubYOLO.names)

    def clear(self):
        pass


@pytest.fixture
def stub_backend(monkeypatch):
    from animetext_studio.engines import detection as det
    StubYOLO.names, StubYOLO.names_error = [], None
    monkeypatch.setattr(det, "_import_yolo_model", lambda: StubYOLO)
    return det


class TestEngineUsesBundle:
    def test_list_models_marks_bundled_source(self, tmp_path, stub_backend):
        make_variant(tmp_path, "yolo12m_animetext")
        StubYOLO.names = ["yolo12m_animetext", "yolo12x_animetext"]
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path))
        cat = {m["name"]: m["source"] for m in eng.list_models()}
        assert cat["yolo12m_animetext"] == "bundled"
        assert cat["yolo12x_animetext"] == "hf"

    def test_dead_repo_falls_back_to_what_is_on_disk(self, tmp_path, stub_backend):
        """A gated/offline repo no longer hides the variants we actually have."""
        make_variant(tmp_path, "yolo12m_animetext")
        StubYOLO.names_error = RuntimeError("401 cannot access gated repo")
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path))
        names = eng.refresh_models()
        assert set(KNOWN_MODELS) <= set(names)         # presets keep the GUI usable
        assert names == [n for n in KNOWN_MODELS] + \
            [n for n in names if n not in KNOWN_MODELS]   # stable order, not bundled-first
        assert eng.model_list_is_fallback is True
        src = {m["name"]: m["source"] for m in eng.list_models()}
        assert src["yolo12m_animetext"] == "bundled"
        assert src["yolo12x_animetext"] == "hf"        # listed but not usable offline
        assert "error:RuntimeError" in eng.status().detail

    def test_listing_timeout_is_bounded_not_fatal(self, tmp_path, stub_backend, monkeypatch):
        """A wedged repo listing must not freeze the caller (decision D27)."""
        make_variant(tmp_path, "yolo12n_animetext")
        release = threading.Event()

        class WedgedYOLO(StubYOLO):
            @property
            def model_names(self):
                release.wait(30)                        # never answered during the test
                return ["never_arrives"]

        monkeypatch.setattr(stub_backend, "_import_yolo_model", lambda: WedgedYOLO)
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), listing_timeout=0.25)
        started = time.monotonic()
        try:
            names = eng.refresh_models()
        finally:
            release.set()
        assert time.monotonic() - started < 5.0          # bounded, not 30 s
        assert names[0] == "yolo12n_animetext"           # the bundle answered instead
        assert eng.model_list_is_fallback is True        # and it says so
        assert "timeout" in eng.status().detail

    def test_offline_never_builds_the_backend(self, tmp_path, stub_backend, monkeypatch):
        """Offline must not even construct the Hub-backed object (no fsspec, no socket)."""
        make_variant(tmp_path, "yolo12n_animetext")
        built: list[str] = []

        class NeverBuilt(StubYOLO):
            def __init__(self, repo_id, hf_token=None):
                built.append(repo_id)
                super().__init__(repo_id, hf_token)

        monkeypatch.setattr(stub_backend, "_import_yolo_model", lambda: NeverBuilt)
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)
        assert eng.refresh_models()[0] == "yolo12n_animetext"
        assert built == []
        assert eng.model_list_is_fallback is True
        assert "offline" in eng.status().detail

    def test_load_uses_bundled_weights_without_downloading(self, tmp_path, stub_backend,
                                                           monkeypatch):
        """D26 end to end (minus real onnxruntime): the bundled file is opened, nothing fetched.

        Only ``open_onnx_model`` is faked — resolution, the no-download decision and
        the session bookkeeping all run for real.
        """
        from animetext_studio.engines.base import EngineState

        make_variant(tmp_path, "yolo12m_animetext")
        opened: list[str] = []
        downloads: list[tuple] = []

        class FakeMeta:
            custom_metadata_map = {"imgsz": "[640, 640]", "names": "{0: 'text'}"}

        class FakeSession:
            def get_modelmeta(self):
                return FakeMeta()

            def get_providers(self):
                return ["CPUExecutionProvider"]

        def fake_open(path):
            opened.append(path)
            return FakeSession()

        fake_mod = types.ModuleType("imgutils.utils")
        fake_mod.open_onnx_model = fake_open
        monkeypatch.setitem(sys.modules, "imgutils.utils", fake_mod)
        monkeypatch.setattr(stub_backend, "_hf_download",
                            lambda *a, **k: downloads.append(a) or "/cache/x")

        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)
        eng.load("yolo12m_animetext")

        expected = tmp_path / ms.DETECTION_DIRNAME / "yolo12m_animetext" / "model.onnx"
        assert opened == [str(expected)]
        assert downloads == []                              # never touched the Hub
        assert eng.status().state is EngineState.READY
        assert eng.status().device == "cpu"                 # provider actually reported

    def test_local_file_resolution_needs_a_real_file(self, tmp_path, stub_backend):
        """Both attachment styles (store / bare detection dir) resolve the same files."""
        cls = stub_backend._bundled_yolo_class()
        obj = cls.__new__(cls)
        obj._init_bundled(ms.ModelStore(tmp_path))
        assert obj.local_file("yolo12m_animetext", "model.onnx") is None
        vdir = make_variant(tmp_path, "yolo12m_animetext")
        assert obj.local_file("yolo12m_animetext", "model.onnx") == vdir / "model.onnx"
        assert obj.local_names() == ["yolo12m_animetext"]
        obj._init_bundled(None)
        assert obj.local_file("yolo12m_animetext", "model.onnx") is None
        assert obj.local_names() == []
        obj._init_bundled(tmp_path / ms.DETECTION_DIRNAME)      # bare path still works
        assert obj.local_names() == ["yolo12m_animetext"]

    def test_one_definition_of_bundled_for_store_and_backend(self, tmp_path, stub_backend):
        """A 512-byte half-copied ONNX is not a model to the store, so the listing skips it."""
        make_variant(tmp_path, "yolo12m_animetext", onnx=512)
        cls = stub_backend._bundled_yolo_class()
        obj = cls.__new__(cls)
        obj._init_bundled(ms.ModelStore(tmp_path))
        assert obj.local_names() == []

    def test_threshold_and_type_read_from_the_bundle(self, tmp_path, stub_backend):
        vdir = make_variant(tmp_path, "yolo12m_animetext")
        (vdir / "threshold.json").write_text(json.dumps({"threshold": 0.42}),
                                             encoding="utf-8")
        (vdir / "model_type.json").write_text(json.dumps({"model_type": "yolo12"}),
                                              encoding="utf-8")
        cls = stub_backend._bundled_yolo_class()
        obj = cls.__new__(cls)
        obj._init_bundled(ms.ModelStore(tmp_path))
        assert obj._get_default_thresholds("yolo12m_animetext") == 0.42
        assert obj._get_model_type("yolo12m_animetext") == "yolo12"

    def test_garbage_aux_files_degrade_instead_of_raising(self, tmp_path, stub_backend):
        vdir = make_variant(tmp_path, "yolo12m_animetext")
        (vdir / "threshold.json").write_text("nope", encoding="utf-8")
        (vdir / "model_type.json").write_text("nope", encoding="utf-8")
        cls = stub_backend._bundled_yolo_class()
        obj = cls.__new__(cls)
        obj._init_bundled(ms.ModelStore(tmp_path))
        assert obj._get_default_thresholds("yolo12m_animetext") is None
        assert obj._get_model_type("yolo12m_animetext") == "yolo"

    def test_engine_reports_the_store_it_uses(self, tmp_path, stub_backend):
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)
        assert eng.model_store.root == tmp_path
        assert eng.model_store.offline is True


# ---------------------------------------------------------------------------
# P9.1 — a bundled file that will not open is diagnosed, not called "missing"
# ---------------------------------------------------------------------------

def vendored(tmp_path, name="yolo11m", *, corrupt=False):
    """A variant plus a manifest that recorded the *good* bytes; optionally ship bad ones."""
    vdir = tmp_path / ms.DETECTION_DIRNAME / name
    vdir.mkdir(parents=True)
    path = vdir / "model.onnx"
    path.write_bytes(b"real weights" * 100_000)           # 1.2 MB: above the "bundled" bar
    digest = ms.sha256_file(path)
    ms.write_manifest(tmp_path, [{"path": str(path), "sha256": digest,
                                 "size_bytes": path.stat().st_size}], merge=False)
    if corrupt:
        # 1.05 MB of *different* bytes: big enough to pass the size test that decides
        # "bundled", wrong enough that only a hash can tell — the real failure mode.
        path.write_bytes(b"truncated cop" * 87_500)
    return path, digest


class TestVerifyFile:
    def test_intact_file_verifies(self, tmp_path):
        path, digest = vendored(tmp_path)
        out = ms.verify_file(path, tmp_path)
        assert out["state"] == "match" and out["got"] == digest

    def test_truncated_file_is_a_mismatch_with_both_hashes(self, tmp_path):
        path, digest = vendored(tmp_path, corrupt=True)
        out = ms.verify_file(path, tmp_path)
        assert out["state"] == "mismatch"
        assert out["expected"] == digest and out["got"] != digest
        assert out["size_bytes"] == path.stat().st_size

    def test_a_file_the_bundle_never_recorded_gets_no_verdict(self, tmp_path):
        vendored(tmp_path)
        hand = tmp_path / ms.DETECTION_DIRNAME / "yolo9z" / "model.onnx"
        hand.parent.mkdir(parents=True)
        hand.write_bytes(b"copied by hand")
        assert ms.verify_file(hand, tmp_path)["state"] == "unrecorded"

    def test_missing_file_is_absent_not_unrecorded(self, tmp_path):
        vendored(tmp_path)
        assert ms.verify_file(tmp_path / ms.DETECTION_DIRNAME / "gone.onnx", tmp_path)["state"] == "absent"

    def test_hot_paths_stay_hash_free(self, tmp_path):
        """`detection_file`/`status` must not hash: the cost is on the failure path only."""
        path, _ = vendored(tmp_path)
        called = []
        real = ms.sha256_file
        monkey = lambda *a, **k: called.append(a) or real(*a, **k)
        ms.sha256_file = monkey
        try:
            store = ms.ModelStore(tmp_path)
            assert store.detection_file("yolo11m", "model.onnx") == path
            assert store.status()["total_bytes"] >= path.stat().st_size
        finally:
            ms.sha256_file = real
        assert called == []


class TestOpenFailureDiagnosis:
    def test_intact_weights_blame_the_runtime_not_the_file(self, tmp_path, stub_backend):
        path, _ = vendored(tmp_path)
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)
        with pytest.raises(StudioError) as exc:
            eng._open_failed("yolo11m", path, RuntimeError("Protobuf parsing failed."))
        err = exc.value
        assert err.code is ErrorCode.ENGINE_LOAD_FAILED
        assert err.details["weights_check"] == "match"
        assert "intact" in err.message and "ONNX runtime" in err.message
        assert any("onnxruntime" in s for s in err.details["steps"])

    def test_corrupt_weights_offline_name_the_file_and_the_fix(self, tmp_path, stub_backend):
        path, digest = vendored(tmp_path, corrupt=True)
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)
        with pytest.raises(StudioError) as exc:
            eng._open_failed("yolo11m", path, RuntimeError("Protobuf parsing failed."))
        err = exc.value
        assert err.details["weights_check"] == "mismatch"
        assert err.details["expected"] == digest
        assert err.details["path"] == str(path) and err.details["bundled"] is True
        assert "corrupt" in err.message
        assert any("fetch_models.py --verify" in s for s in err.details["steps"])
        assert any("--variants yolo11m" in s for s in err.details["steps"])

    def test_offline_never_reaches_for_the_network(self, tmp_path, stub_backend, monkeypatch):
        path, _ = vendored(tmp_path, corrupt=True)
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)

        def no_socket(*a, **k):
            raise AssertionError("offline mode opened a socket")
        monkeypatch.setattr(stub_backend, "_hf_download", no_socket)
        with pytest.raises(StudioError):
            eng._open_failed("yolo11m", path, RuntimeError("Protobuf parsing failed."))
        assert eng.recovered_via_hub == {}

    def test_corrupt_weights_online_retry_once_and_say_they_did(self, tmp_path, stub_backend,
                                                                monkeypatch):
        path, _ = vendored(tmp_path, corrupt=True)
        hub = tmp_path / "hubcache" / "model.onnx"
        hub.parent.mkdir(parents=True)
        hub.write_bytes(b"real weights" * 64)
        calls = []

        def fake_download(repo_id, filename, token=None, tqdm_class=None):
            calls.append(filename)
            return str(hub)
        monkeypatch.setattr(stub_backend, "_hf_download", fake_download)

        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=False)
        yolo = FakeYOLO()
        monkeypatch.setattr(eng, "_ensure_yolo", lambda: yolo)
        monkeypatch.setattr(eng, "_note_session_providers", lambda model: None)

        eng._open_failed("yolo11m", path, RuntimeError("Protobuf parsing failed."))   # no raise
        assert calls == ["yolo11m/model.onnx"]                       # once, not repeatedly
        assert yolo.override_files[("yolo11m", "model.onnx")] == hub  # the good copy wins now
        assert eng.recovered_via_hub["yolo11m"] == str(hub)
        assert "recovered from Hub" in eng.status().detail and "corrupt" in eng.status().detail

    def test_a_failed_retry_keeps_the_diagnosis_instead_of_hiding_it(self, tmp_path, stub_backend,
                                                                     monkeypatch):
        path, digest = vendored(tmp_path, corrupt=True)
        monkeypatch.setattr(stub_backend, "_hf_download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no route")))
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=False)
        with pytest.raises(StudioError) as exc:
            eng._open_failed("yolo11m", path, RuntimeError("Protobuf parsing failed."))
        assert exc.value.details["expected"] == digest
        assert "Hub was tried" in exc.value.message
        assert eng.recovered_via_hub == {}

    def test_a_non_load_error_is_not_rescued(self, tmp_path, stub_backend):
        """Only ENGINE_LOAD_FAILED on a bundled file gets the treatment; MODEL_NOT_FOUND passes."""
        path, _ = vendored(tmp_path, corrupt=True)
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)
        from animetext_studio.core.types import StudioError as SE
        original = SE(ErrorCode.MODEL_NOT_FOUND, "not there", {})
        with pytest.raises(StudioError) as exc:
            eng._open_failed("yolo11m", path, original)
        assert exc.value.code is ErrorCode.MODEL_NOT_FOUND


class FakeYOLO:
    """Just enough of the imgutils subclass for the retry: an opener and the override map."""

    def __init__(self):
        self.override_files = {}
        self.opened = []

    def _open_model(self, model_name):
        self.opened.append((model_name, self.override_files.get((model_name, "model.onnx"))))
        return object()


class TestStatusStaysPassive:
    """P9.2 put the flag on the status — without letting a status call go looking for one."""

    def test_status_never_asks_the_repo(self, tmp_path, stub_backend, monkeypatch):
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)

        def never(*args, **kwargs):
            raise AssertionError("status() spent time on a listing")
        monkeypatch.setattr(eng, "refresh_models", never)
        assert eng.status().model_list_is_fallback is False
        eng._names_offline = True                      # as a timed-out listing would leave it
        status = eng.status()
        assert status.model_list_is_fallback is True   # reported, without asking again
        assert status.to_dict()["model_list_is_fallback"] is True

    def test_the_property_form_still_asks(self, tmp_path, stub_backend, monkeypatch):
        """Two questions, two answers: "what do you know" (status) and "ask again" (property)."""
        eng = stub_backend.DetectionEngine(models_dir=str(tmp_path), offline=True)
        calls: list[int] = []
        monkeypatch.setattr(eng, "refresh_models", lambda *a, **k: calls.append(1) or [])
        assert eng.model_list_is_fallback is False
        assert calls == [1]
