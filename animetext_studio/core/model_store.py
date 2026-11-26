"""On-premises (bundled) model store — decision D26.

Why this exists: the weights must be *part of the app*, so that a copy of the project
folder is a complete, working, offline installation — no Hugging Face account, no
gated-repo token, no network. The Hub therefore becomes a **fetch fallback** instead
of a hard runtime dependency.

Layout (default root = ``<package>/models``, overridable by ``paths.models_dir``
config / ``--models-dir`` / ``ANIMETEXT_MODELS_DIR``)::

    models/
    ├── MANIFEST.json                    # provenance + sha256 (written by fetch_models.py)
    ├── animetext-yolo/                  # deepghs/AnimeText_yolo        (gated, GPL-3.0)
    │   └── <variant>/model.onnx + threshold.json + labels.json + model_type.json
    └── hayai-ocr-v2.7-supernova/        # JustANormalTinkerer/…        (public, Apache-2.0)
        └── full snapshot: config.json, modeling_hayai.py, model.safetensors, tokenizer…

Resolution order is always **bundled first, Hub second**: a file that exists on disk is
never re-downloaded, and with ``paths.offline = true`` nothing touches the network at
all (a missing bundle then raises MODEL_NOT_FOUND naming ``scripts/fetch_models.py``).

Pure stdlib, and nothing here imports torch/onnxruntime — status checks must work in a
bare install (and in tests) without the heavy extras.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .. import APP_NAME
from .config import KNOWN_MODELS
from .types import DETECTION_REPO, OCR_REPO, OCR_REVISION, ErrorCode, StudioError

# ---------------------------------------------------------------------------
# Names of things on disk
# ---------------------------------------------------------------------------

#: Environment override for the models root (wins over the package default, loses to
#: an explicit ``paths.models_dir`` / ``--models-dir``).
MODELS_DIR_ENV = "ANIMETEXT_MODELS_DIR"

DETECTION_DIRNAME = "animetext-yolo"
OCR_DIRNAME = "hayai-ocr-v2.7-supernova"
MANIFEST_NAME = "MANIFEST.json"

#: Files that make a bundled detection variant usable / merely nice to have.
DETECTION_REQUIRED_FILES: tuple[str, ...] = ("model.onnx",)
DETECTION_OPTIONAL_FILES: tuple[str, ...] = ("threshold.json", "labels.json", "model_type.json")

#: What `transformers.AutoModel/AutoProcessor.from_pretrained(<dir>)` needs. Kept
#: explicit so a half-copied snapshot is reported as broken instead of failing later
#: with an opaque remote-code ImportError deep inside transformers.
OCR_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "configuration_hayai.py",
    "modeling_hayai.py",
    "processing_hayai.py",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)
OCR_OPTIONAL_FILES: tuple[str, ...] = ("tokenizer.model", "README.md", "yolo26m-text.pt")

#: The OCR weights are big; report them in MB with enough digits to spot a truncated
#: copy (a partial download is the single most likely on-disk corruption).
_MIN_WEIGHTS_BYTES = 1_000_000


# ---------------------------------------------------------------------------
# HF token discovery (only ever needed for the *gated* detection repo)
# ---------------------------------------------------------------------------

#: Optional local token files, checked in this order under each search root. The app
#: never writes one and never puts one in an exported config; it exists so a developer
#: who already has a token does not have to export it into every shell, GUI launch or
#: subprocess. All of these names are in `.gitignore`, and nothing here ever logs a value.
TOKEN_FILE_NAMES: tuple[str, ...] = ("HF_token.txt", ".hf_token")

#: Environment override: path to a file holding one HF token (wins over the search).
TOKEN_FILE_ENV = "ANIMETEXT_TOKEN_FILE"


def token_search_roots() -> list[Path]:
    """Where a local token file may live: cwd, the project root, the models root."""
    roots = [Path.cwd(), package_models_dir().parent, default_models_dir()]
    out: list[Path] = []
    for root in roots:
        if root not in out:
            out.append(root)
    return out


def read_token_file(path: Union[str, Path]) -> Optional[str]:
    """The first non-empty, non-comment line of a token file, or None.

    Tolerant by design (a stray newline or a `# note` next to the token is normal) and
    silent on any I/O problem: not finding a token is the common case, not an error.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def discover_token_file(roots: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Path of the first existing local token file, or None."""
    env = os.environ.get(TOKEN_FILE_ENV)
    if env and env.strip():
        path = Path(env).expanduser()
        return path if path.is_file() else None
    for root in (roots if roots is not None else token_search_roots()):
        for name in TOKEN_FILE_NAMES:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def discover_hf_token(explicit: Optional[str] = None) -> Optional[str]:
    """``explicit`` > ``$HF_TOKEN`` > a local token file. Never raises, never logs.

    Used by the detection engine and by ``scripts/fetch_models.py`` so one file at the
    project root is enough for real-world (Hub) runs. The returned value must never be
    put in a log line, a status dict or an exported config — `mcp.bridge.redact_args`
    scrubs the `hf_token`/`token` keys, and `core.config` only ever exports `mcp.auth_token`.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ.get("HF_TOKEN")
    if env and env.strip():
        return env.strip()
    path = discover_token_file()
    return read_token_file(path) if path is not None else None


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------

def package_models_dir() -> Path:
    """The models directory shipped inside the package (``animetext_studio/models``)."""
    return Path(__file__).resolve().parent.parent / "models"


def default_models_dir() -> Path:
    """Package default, or ``$ANIMETEXT_MODELS_DIR`` when set (D26 precedence layer 2)."""
    env = os.environ.get(MODELS_DIR_ENV)
    if env and env.strip():
        return Path(env).expanduser().resolve()
    return package_models_dir()


def resolve_models_dir(explicit: Optional[Union[str, "os.PathLike[str]"]] = None) -> Path:
    """explicit (config/CLI) > ``$ANIMETEXT_MODELS_DIR`` > package default."""
    if explicit not in (None, ""):
        return Path(str(explicit)).expanduser().resolve()
    return default_models_dir()


# ---------------------------------------------------------------------------
# File-level integrity helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    """Streaming sha256 (used by the manifest, never by the hot status path)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_safetensors(path: Union[str, Path]) -> dict[str, Any]:
    """Structural check of a .safetensors file *without* loading it into RAM.

    Reads the 8-byte length + JSON header, then confirms that the union of the declared
    tensor byte ranges accounts for the file exactly. This catches the realistic
    failures of a hand-copied bundle: truncated download, wrong file, LFS pointer left
    unresolved. Does not verify the numbers themselves (use the manifest sha256 for that).
    """
    path = Path(path)
    report: dict[str, Any] = {"path": str(path), "ok": False, "error": None,
                              "tensors": 0, "dtype": None, "bytes": path.stat().st_size
                              if path.exists() else 0}
    if not path.exists():
        report["error"] = "file not found"
        return report
    try:
        with open(path, "rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                report["error"] = "file too small to be safetensors"
                return report
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len == 0 or header_len > path.stat().st_size:
                report["error"] = f"implausible header length {header_len}"
                return report
            header = json.loads(handle.read(header_len))
    except Exception as exc:                                  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    if not isinstance(header, dict) or not header:
        report["error"] = "header is not a non-empty JSON object"
        return report

    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1,
             "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2}
    top_end, dtypes = 0, {}
    for entry in entries.values():
        try:
            o0, o1 = entry["data_offsets"]
        except Exception:                                     # noqa: BLE001
            report["error"] = "tensor entry without data_offsets"
            return report
        top_end = max(top_end, int(o1))
        dtypes[entry.get("dtype")] = dtypes.get(entry.get("dtype"), 0) + 1
    expected = 8 + header_len + top_end
    report.update({
        "tensors": len(entries),
        "dtype": ",".join(sorted(dtypes)) or None,
        "declared_bytes": expected,
        "delta_bytes": path.stat().st_size - expected,
    })
    # A trailing delta is tolerated (some writers pad); a negative one means truncation.
    if expected > path.stat().st_size:
        report["error"] = (f"truncated: header declares {expected} bytes, file has "
                           f"{path.stat().st_size}")
    else:
        report["ok"] = True
    return report


def _file_ok(path: Path, minimum: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= minimum
    except OSError:
        return False


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

class ModelStore:
    """Read-only view of the on-premises model bundle.

    Cheap to construct (no I/O beyond ``is_dir`` checks when asked). The engines hold
    one instance each; the GUI/MCP report comes from :meth:`status`.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
        offline: bool = False,
    ) -> None:
        self._root = resolve_models_dir(models_dir)
        self._offline = bool(offline)

    # -- basics ------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def offline(self) -> bool:
        return self._offline

    @property
    def exists(self) -> bool:
        return self._root.is_dir()

    @property
    def detection_root(self) -> Path:
        return self._root / DETECTION_DIRNAME

    def detection_variant_dir(self, model_name: str) -> Path:
        return self.detection_root / str(model_name)

    def ocr_dir(self) -> Path:
        return self._root / OCR_DIRNAME

    # -- detection ---------------------------------------------------------

    def detection_file(self, model_name: str, filename: str) -> Optional[Path]:
        """Path to a bundled detection file, or None when absent/empty."""
        path = self.detection_variant_dir(model_name) / filename
        return path if _file_ok(path) else None

    def bundled_detection_variants(self) -> list[str]:
        """Every on-disk variant dir holding a usable ``model.onnx`` (sorted, KNOWN first)."""
        found: list[str] = []
        root = self.detection_root
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and _file_ok(child / "model.onnx", _MIN_WEIGHTS_BYTES):
                    found.append(child.name)
        ordered = [n for n in KNOWN_MODELS if n in found]
        ordered += [n for n in found if n not in KNOWN_MODELS]
        return ordered

    def has_detection_variant(self, model_name: str) -> bool:
        return self.detection_file(model_name, "model.onnx") is not None

    def require_detection_file(self, model_name: str, filename: str = "model.onnx") -> Path:
        """Bundled file when present; MODEL_NOT_FOUND (with the fix command) otherwise."""
        path = self.detection_file(model_name, filename)
        if path is not None:
            return path
        raise StudioError(
            ErrorCode.MODEL_NOT_FOUND,
            f"{model_name}/{filename} is not available on premises.",
            {
                "models_dir": str(self._root),
                "bundled_variants": self.bundled_detection_variants(),
                "offline": self._offline,
                "steps": (
                    [f"1. python scripts/fetch_models.py --variants {model_name}",
                     "2. re-run — the bundled copy is used automatically"]
                    if self._offline else
                    ["1. the file will be fetched from Hugging Face on load "
                     "(needs a token that accepted the repo terms)",
                     "2. to keep it permanently: python scripts/fetch_models.py --all"]
                ),
            },
        )

    # -- OCR ---------------------------------------------------------------

    def missing_ocr_files(self) -> list[str]:
        d = self.ocr_dir()
        return [f for f in OCR_REQUIRED_FILES if not _file_ok(d / f)]

    def has_ocr_snapshot(self) -> bool:
        return not self.missing_ocr_files()

    def require_ocr_snapshot(self) -> Path:
        """Local snapshot dir for ``from_pretrained``, or a MODEL_NOT_FOUND telling how to get it."""
        missing = self.missing_ocr_files()
        if not missing:
            return self.ocr_dir()
        raise StudioError(
            ErrorCode.MODEL_NOT_FOUND,
            f"The OCR model snapshot is incomplete in {self.ocr_dir()}.",
            {
                "missing": missing,
                "offline": self._offline,
                "steps": ([f"1. python scripts/fetch_models.py --ocr"
                           if self._offline else
                           f"1. will download {OCR_REPO} on load (public repo, no token needed)"]
                          + ["2. or copy the files manually into " + str(self.ocr_dir())]),
            },
        )

    # -- reporting ---------------------------------------------------------

    def detection_variant_status(self, model_name: str) -> dict[str, Any]:
        vdir = self.detection_variant_dir(model_name)
        files = {f: _file_ok(vdir / f) for f in DETECTION_REQUIRED_FILES + DETECTION_OPTIONAL_FILES}
        onnx = self.detection_file(model_name, "model.onnx")
        return {
            "name": model_name,
            "bundled": onnx is not None,
            "complete": all(files[f] for f in DETECTION_REQUIRED_FILES),
            "files": files,
            "path": str(vdir),
            "bytes": onnx.stat().st_size if onnx is not None else 0,
        }

    def ocr_status(self) -> dict[str, Any]:
        d = self.ocr_dir()
        missing = self.missing_ocr_files()
        weights = d / "model.safetensors"
        info: dict[str, Any] = {
            "repo_id": OCR_REPO,
            "bundled": not missing,
            "complete": not missing,
            "missing": missing,
            "path": str(d),
            "bytes": sum(f.stat().st_size for f in d.glob("*") if f.is_file()) if d.is_dir() else 0,
        }
        origin = d / ".origin.json"
        if origin.is_file():
            try:
                info["revision"] = json.loads(origin.read_text(encoding="utf-8")).get("revision")
            except Exception:                                   # noqa: BLE001
                info["revision"] = None
        else:
            info["revision"] = OCR_REVISION
        if weights.is_file():
            check = verify_safetensors(weights)
            info["weights_ok"] = check["ok"]
            info["tensors"] = check["tensors"]
            if not check["ok"]:
                info["weights_error"] = check["error"]
        return info

    def status(self) -> dict[str, Any]:
        """Machine-readable on-premises report (GUI status bar, MCP ``get_status``, doctor)."""
        variants = self.bundled_detection_variants()
        per_variant = {n: self.detection_variant_status(n) for n in KNOWN_MODELS}
        for extra in variants:
            per_variant.setdefault(extra, self.detection_variant_status(extra))
        ocr = self.ocr_status()
        total = sum(v["bytes"] for v in per_variant.values()) + ocr["bytes"]
        return {
            "models_dir": str(self._root),
            "exists": self.exists,
            "offline": self._offline,
            "detection_repo": DETECTION_REPO,
            "detection_bundled": variants,
            "detection_variants": per_variant,
            "ocr": ocr,
            "total_bytes": total,
            "manifest": _read_manifest(self._root),
        }


def manifest_entry_path(entry: dict[str, Any], root: Union[str, Path],
                        recorded_root: Optional[str] = None) -> Path:
    """Where a manifest entry lives *now*, given a bundle that may have moved.

    ``MANIFEST.json`` used to record only the absolute path of the machine that vendored it,
    which made the bundle's own passport expire the moment the folder moved: copy the project
    to another machine, install the wheel into a different prefix, or set
    ``$ANIMETEXT_MODELS_DIR``, and ``fetch_models.py --verify`` reported every recorded file
    as "RECORDED BUT ABSENT" and called a complete bundle "incomplete" — while the app itself
    loaded all of them happily (D26 promises the copy *is* the installation).

    Resolution order, newest format first:

    1. ``relpath`` — relative to ``root``. What a bundle carries once it has been re-written.
    2. a relative ``path`` — same thing, for a hand-written manifest.
    3. an absolute ``path`` that exists — the legacy format on the machine that wrote it.
    4. that same absolute path re-based onto ``root`` using the recorded ``models_root`` — the
       legacy format *after* a move. Only taken when the re-based file actually exists, so a
       genuinely missing file is still reported missing.
    """
    # Absolute, but without resolving symlinks: a bundle reached through a symlink must keep
    # resolving through it, and a *relative* root (a test, a `cd`-ed caller) must still be
    # comparable with the absolute paths a legacy manifest recorded.
    root = Path(os.path.abspath(str(root)))
    rel = entry.get("relpath")
    if isinstance(rel, str) and rel:
        return root / rel
    raw = entry.get("path")
    if not isinstance(raw, str) or not raw:
        return root
    candidate = Path(raw)
    if not candidate.is_absolute():
        return root / candidate
    if candidate.is_file():
        return candidate
    if recorded_root:
        try:
            rebase = root / candidate.relative_to(Path(os.path.abspath(str(recorded_root))))
        except ValueError:
            return candidate
        if rebase.is_file():
            return rebase
    return candidate


def resolve_manifest_files(manifest: Optional[dict[str, Any]],
                           root: Union[str, Path]) -> list[tuple[Path, dict[str, Any]]]:
    """``manifest["files"]`` resolved against ``root`` — `[(path, entry)]`, in file order."""
    if not isinstance(manifest, dict):
        return []
    recorded_root = manifest.get("models_root")
    return [(manifest_entry_path(item, root, recorded_root), item)
            for item in manifest.get("files", []) if isinstance(item, dict)]


def manifest_entry_for(path: Union[str, Path],
                       root: Union[str, Path, None] = None) -> Optional[dict[str, Any]]:
    """The ``MANIFEST.json`` entry describing ``path``, or None if the bundle does not know it.

    Matched on the *resolved* path, so a bundle that has been moved (D26) still recognises its
    own files: the entry's recorded absolute path is resolved by ``manifest_entry_path`` first,
    and only then compared.
    """
    target = Path(os.path.abspath(str(path)))
    root = target.parent if root is None else Path(root)
    for entry_path, entry in resolve_manifest_files(_read_manifest(root), root):
        if Path(os.path.abspath(str(entry_path))) == target:
            return entry
    return None


def verify_file(path: Union[str, Path],
                root: Union[str, Path, None] = None) -> dict[str, Any]:
    """Compare one file with the sha256 the manifest recorded for it.

    Returns ``{"state": …, "path": …, "got": …, "expected": …, "size_bytes": …}`` where
    ``state`` is:

    ``match``
        the bytes are the bytes the bundle says they are — a failure to *use* the file is
        therefore a runtime problem (ONNX build, provider), not a file problem;
    ``mismatch``
        the file is not what was vendored: truncated copy, interrupted download, wrong file.
        This is the case the message has to name, because "bundled" was true and the loader's
        own parse error says nothing about it (P9.1);
    ``unrecorded``
        the bundle's passport has no entry for this file (hand-copied weights, or a manifest
        older than the file) — no verdict is available, and the result says so;
    ``absent``
        there is no file to hash.

    Cold path only. Hashing 77-226 MB is real I/O, so this is called when an open has *already*
    failed (or by ``fetch_models.py --verify``), never from ``status()`` or ``detection_file()``.
    """
    file_path = Path(path)
    out: dict[str, Any] = {"state": "absent", "path": str(file_path), "got": None,
                           "expected": None, "size_bytes": None}
    if not file_path.is_file():
        return out
    entry = manifest_entry_for(file_path, root)
    if entry is None:
        out["state"] = "unrecorded"
        out["got"] = sha256_file(file_path)
        out["size_bytes"] = file_path.stat().st_size
        return out
    expected = entry.get("sha256")
    out["expected"] = expected
    out["size_bytes"] = file_path.stat().st_size
    if not expected:
        out["state"] = "unrecorded"
        return out
    got = sha256_file(file_path)
    out["got"] = got
    out["state"] = "match" if got == expected else "mismatch"
    return out


def _read_manifest(root: Path) -> Optional[dict[str, Any]]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:                                           # noqa: BLE001
        return None


def write_manifest(
    root: Union[str, Path],
    entries: Iterable[dict[str, Any]],
    extra: Optional[dict[str, Any]] = None,
    merge: bool = True,
) -> dict[str, Any]:
    """(Re)write ``MANIFEST.json``: what came from where, at which revision, sha256+size.

    ``merge`` keeps the entries of the previous run for files that are still on disk.
    Without it, a partial run (``--ocr`` alone, or a single ``--variants`` name) would
    erase the provenance of everything else in the bundle — and the whole point of the
    manifest is that *the folder* knows where its weights came from.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    by_path: dict[str, dict[str, Any]] = {}

    # Same reason as in `manifest_entry_path`: compare absolute-but-unresolved paths, so a
    # relative `root` still matches the absolute paths a legacy manifest recorded.
    abs_root = Path(os.path.abspath(str(root)))

    def key_for(entry: dict[str, Any]) -> str:
        # The relative path is the identity: it survives the move, the absolute one does not,
        # and a merge keyed on the absolute path would re-add a second entry for the same
        # file after the bundle has been relocated.
        resolved = manifest_entry_path(entry, abs_root)
        try:
            return resolved.relative_to(abs_root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def keep(entry: dict[str, Any]) -> Optional[dict[str, Any]]:
        resolved = manifest_entry_path(entry, abs_root)
        if not (resolved.is_file() and resolved.stat().st_size > 0):
            return None
        out = dict(entry)
        try:
            out["relpath"] = resolved.relative_to(abs_root).as_posix()
        except ValueError:
            out.pop("relpath", None)
        return out

    if merge:
        previous = _read_manifest(root) or {}
        for item in previous.get("files", []):
            if isinstance(item, dict):
                kept = keep(item)
                if kept is not None:
                    by_path[key_for(item)] = kept
    for item in entries:
        # Never record something that is not actually there: a failed download (gated
        # repo, interrupted) must not leave a phantom 0-byte entry claiming the bundle
        # has it — `--verify` and the GUI would then promise a model that cannot load.
        kept = keep(item)
        if kept is not None:
            by_path[key_for(item)] = kept
        else:
            by_path.pop(key_for(item), None)
    manifest: dict[str, Any] = {
        "tool": f"{APP_NAME} scripts/fetch_models.py",
        # Absolute: it is the anchor legacy entries are re-based from (D76), and a relative one
        # would make that arithmetic depend on the caller's cwd.
        "models_root": os.path.abspath(str(root)),
        "files": [by_path[k] for k in sorted(by_path)],
    }
    if extra:
        manifest.update(extra)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
