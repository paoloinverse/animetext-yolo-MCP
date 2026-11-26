"""The read-only environment report: what this machine can and cannot do.

``--doctor`` exists because two upstream behaviours in ``dghs-imgutils`` cost a full day
to diagnose on a real machine, and every symptom they produce points somewhere else:

* ``imgutils/utils/onnxruntime.py`` runs ``pip install`` **at import time** when
  ``onnxruntime`` is missing — so a machine that was promised never to reach the network
  does, quietly, in the middle of loading a model;
* ``imgutils/generic/yolo.py`` evaluates label metadata with ``ast.Str``/``ast.Num``,
  which Python 3.14 removed — so the same install works on 3.13 and dies on 3.14 part-way
  through loading a model, with an ``AttributeError`` that names no package at all.

Both are invisible to a status call and obvious in one table. The doctor answers the
question the README's Troubleshooting table asks, before anyone presses Generate.

Three rules, in order of how much trouble they prevent (decision D84):

1. **Never installs, never downloads, never writes.** No ``pip``, no Hub call, no temp
   file, no cache directory. The doctor is safe to run on a machine you have been asked
   not to change — which is most of the machines it gets run on.
2. **Never imports the backends.** Not ``imgutils`` (rule 1 is exactly what its import
   does), not ``torch``, not ``onnxruntime``. Presence and version come from
   ``importlib.metadata`` and ``importlib.util.find_spec``, which read the install
   without executing it. The door that measures the backends is ``--warm-devices``.
3. **Never prints a secret.** The token check reports *where* the token was found and
   how long it is; the value is never in the report, the JSON or any log line.

The checks are plain functions over a :class:`Probes` seam so the tests can fabricate a
broken machine (3.14, no onnxruntime, numpy 2 against a ``numpy<2`` pin, an empty bundle)
without owning one.
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .config import AppConfig, MODEL_PRESETS
from .model_store import (
    MANIFEST_NAME, ModelStore, discover_token_file, resolve_manifest_files,
)
from .types import DETECTION_REPO, EXIT_CODES, ErrorCode

__all__ = ["Finding", "DoctorReport", "Probes", "run_doctor", "DOCTOR_CHECKS"]

#: The Python floor of the app itself (`pyproject`: requires-python).
PYTHON_FLOOR = (3, 10)
#: The last Python ``dghs-imgutils`` still imports on: ``ast.Str``/``ast.Num`` were
#: removed in 3.14 and `imgutils/generic/yolo.py` still evaluates label metadata with
#: them, so detection — and only detection — cannot start there.
IMGUTILS_PYTHON_CEILING = (3, 14)
#: Where the two upstream behaviours live, quoted in the findings so a user can check
#: them without opening a debugger.
UPSTREAM_AUTO_INSTALL = "imgutils/utils/onnxruntime.py (_ensure_onnxruntime)"
UPSTREAM_AST = "imgutils/generic/yolo.py (_safe_eval_names_str)"
#: The drafted upstream issues, for the hint text.
UPSTREAM_DOC = "docs/UPSTREAM.md"

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_MARKS = {OK: "ok  ", WARN: "warn", FAIL: "FAIL", SKIP: "skip"}


# ---------------------------------------------------------------------------
# The probe seam
# ---------------------------------------------------------------------------

def _safe(fn: Callable[..., Any], *args: Any) -> Optional[Any]:
    try:
        return fn(*args)
    except Exception:                                        # not installed, or metadata is broken
        return None


def _spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):   # a half-installed package
        return None


def _disk_free(path: Path) -> Optional[int]:
    """Free bytes on the filesystem holding *path* (nearest existing ancestor)."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        return shutil.disk_usage(target).free
    except OSError:
        return None


#: Names that mean "this face has Japanese glyphs" — matched on the filename because
#: that is all we are allowed to look at without loading it.
_CJK_HINTS = ("cjk", "meiryo", "pingfang", "hiragino", "yu gothic", "notoserifcjk")


def _font_face() -> tuple[Optional[str], bool]:
    """First candidate face that exists, and whether it can plausibly draw Japanese.

    The canvas uses the Qt UI font; this face is what the **exported PNG** gets, which
    is the one that turns into boxes-of-doom on a machine with no CJK font.
    """
    from .annotation import FONT_CANDIDATES                 # Pillow only

    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            low = candidate.lower()
            return candidate, any(h in low for h in _CJK_HINTS)
    return None, False


@dataclass(frozen=True)
class Probes:
    """Everything the doctor is allowed to know about the machine.

    Each probe is read-only and side-effect free; a test passes a fabricated one and gets
    a broken machine on demand — 3.14, no onnxruntime, numpy 2 against a ``numpy<2`` pin,
    an empty bundle. Nothing here imports a backend: versions come from installed
    metadata, presence from :func:`importlib.util.find_spec`, hardware from a ``PATH``
    lookup.

    The defaults are plain lambdas, which are safe as dataclass defaults precisely
    because ``__init__`` copies them onto the instance — so ``probes.dist_version`` is
    the function, never a bound method of ``Probes``.
    """

    python_version: Callable[[], tuple[int, ...]] = lambda: tuple(sys.version_info[:3])
    dist_version: Callable[[str], Optional[str]] = lambda name: _safe(md.version, name)
    requires: Callable[[str], list[str]] = lambda name: list(_safe(md.requires, name) or [])
    module_present: Callable[[str], bool] = lambda name: _spec(name) is not None
    which: Callable[[str], Optional[str]] = shutil.which
    disk_free_bytes: Callable[[Path], Optional[int]] = lambda path: _disk_free(path)
    token_file: Callable[[], Optional[Path]] = discover_token_file
    env: Callable[[str], Optional[str]] = os.environ.get
    font_face: Callable[[], tuple[Optional[str], bool]] = _font_face


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One line of the report. `code` is the failure the app would produce next."""

    check: str
    severity: str
    detail: str
    hint: str = ""
    code: Optional[ErrorCode] = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity == FAIL

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"check": self.check, "state": self.severity,
                               "detail": self.detail}
        if self.hint:
            out["hint"] = self.hint
        if self.code is not None:
            out["code"] = self.code.value
            out["exit_code"] = EXIT_CODES[self.code]
        if self.data:
            out["data"] = dict(self.data)
        return out


@dataclass(frozen=True)
class DoctorReport:
    """The whole examination: findings, the machine they describe, and a verdict."""

    findings: tuple[Finding, ...]
    context: dict[str, Any] = field(default_factory=dict)
    took_ms: int = 0

    # -- verdict ------------------------------------------------------------

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def blockers(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == FAIL)

    @property
    def ok(self) -> bool:
        return not self.blockers

    @property
    def exit_code(self) -> int:
        """The exit code the next real command would give, not a new code.

        A doctor that always returns 0 cannot gate CI; inventing a 15th error code to say
        "the machine is short a dependency" would dilute the fourteen that mean something
        to a client. So the report borrows the code it predicts — ENGINE_LOAD_FAILED for
        a missing extra, MODEL_NOT_FOUND for a bundle that is not there offline.
        """
        if self.ok:
            return 0
        for finding in self.blockers:
            if finding.code is not None:
                return EXIT_CODES[finding.code]
        return EXIT_CODES[ErrorCode.ENGINE_LOAD_FAILED]

    # -- renderings ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "app": self.context.get("app"),
            "version": self.context.get("version"),
            "python": self.context.get("python"),
            "platform": self.context.get("platform"),
            "mock": self.context.get("mock", False),
            "offline": self.context.get("offline", False),
            "models_dir": self.context.get("models_dir"),
            "summary": {s: self.count(s) for s in (OK, WARN, FAIL, SKIP)},
            "usable": self.ok,
            "exit_code": self.exit_code,
            "took_ms": self.took_ms,
            "checks": [f.to_dict() for f in self.findings],
        }

    def text(self) -> str:
        head = (f"doctor — {self.context.get('app')} {self.context.get('version')} "
                f"on Python {self.context.get('python')} ({self.context.get('platform')})")
        mode = []
        if self.context.get("mock"):
            mode.append("mock engines")
        if self.context.get("offline"):
            mode.append("offline")
        if mode:
            head += " [" + ", ".join(mode) + "]"
        width = max((len(f.check) for f in self.findings), default=6)
        lines = [head, ""]
        for finding in self.findings:
            lines.append(f"  {_MARKS[finding.severity]}  {finding.check:<{width}}  "
                         f"{finding.detail}")
            if finding.hint:
                lines.append(f"        {' ' * width}  fix: {finding.hint}")
        parts = [f"{self.count(OK)} ok", f"{self.count(WARN)} warn",
                 f"{self.count(FAIL)} fail"]
        if self.count(SKIP):
            parts.append(f"{self.count(SKIP)} skip")
        lines += ["", f"  {len(self.findings)} checks: {', '.join(parts)} "
                      f"over {self.took_ms} ms"]
        lines.append("  RESULT: usable" if self.ok else
                     "  RESULT: not usable — fix the FAIL rows above")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The examination
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Context:
    """What a check may look at: the config, the mode, the probes, the store.

    Every probe call goes through a guard. A diagnostic that crashes on the one broken
    package it was asked about is worse than no diagnostic, so a probe that raises is
    answered as "unknown" and the checks say so in words.
    """

    config: AppConfig
    probes: Probes
    store: ModelStore
    mock: bool = False
    hf_token: Optional[str] = None

    # -- guarded probe access ------------------------------------------------

    def version(self, *dists: str) -> Optional[str]:
        """First distribution name that resolves.

        PySide6 is the reason for the *dists: the wheel that provides `import PySide6`
        is published as `PySide6_Essentials` (+ `_Addons`), so asking metadata for
        "PySide6" reports the window missing on a machine that has it — and a doctor that
        cries wolf about the thing it exists to check is worse than no doctor.
        """
        for dist in dists:
            found = _ask(lambda: self.probes.dist_version(dist))
            if found:
                return found
        return None

    def requires(self, dist: str) -> list[str]:
        """The `Requires-Dist` lines of an installed distribution ([] when unknown)."""
        found = _ask(lambda: list(self.probes.requires(dist)))
        return [str(line) for line in (found or [])]

    def has_module(self, name: str) -> bool:
        return bool(_ask(lambda: self.probes.module_present(name)))

    def executable(self, name: str) -> Optional[str]:
        return _ask(lambda: self.probes.which(name))

    def free_bytes(self, path: Path) -> Optional[int]:
        return _ask(lambda: self.probes.disk_free_bytes(path))

    def token_file(self) -> Optional[Path]:
        return _ask(lambda: self.probes.token_file())

    def env(self, key: str) -> Optional[str]:
        return _ask(lambda: self.probes.env(key))

    def font_face(self) -> tuple[Optional[str], bool]:
        found = _ask(lambda: self.probes.font_face())
        return found if isinstance(found, tuple) else (None, False)

    def python_version(self) -> tuple[int, ...]:
        version = _ask(lambda: tuple(self.probes.python_version()))
        return tuple(version[:3]) if version else tuple(sys.version_info[:3])

    # -- mode ---------------------------------------------------------------

    def needs_engines(self) -> bool:
        """False only under `--mock`, where the fakes stand in for both backends."""
        return not self.mock


def _ask(lookup: Callable[[], Any], default: Any = None) -> Any:
    """Run one probe; a probe that raises is an unknown, not a crash."""
    try:
        return lookup()
    except Exception:                                        # noqa: BLE001 - broken machine
        return default


def _vtext(parts: tuple[int, ...]) -> str:
    return ".".join(str(p) for p in parts)


def _parse_version(text: Optional[str]) -> tuple[int, ...]:
    out: list[int] = []
    for chunk in (text or "").replace("+", ".").split("."):
        if chunk.isdigit():
            out.append(int(chunk))
        else:
            break
    return tuple(out)


def check_python(ctx: Context) -> list[Finding]:
    version = ctx.python_version()
    text = _vtext(version)
    if version < PYTHON_FLOOR:
        return [Finding("python", FAIL, f"Python {text} is below the app floor "
                        f"{_vtext(PYTHON_FLOOR)}",
                        f"run the app on Python {_vtext(PYTHON_FLOOR)} or newer",
                        ErrorCode.ENGINE_LOAD_FAILED, {"version": text})]
    if version >= IMGUTILS_PYTHON_CEILING:
        # Detection only: the app, the GUI and the OCR stage are fine on 3.14.
        severity = FAIL if ctx.needs_engines() else SKIP
        return [Finding("python", severity,
                        f"Python {text}: dghs-imgutils cannot import on 3.14+ — it evaluates "
                        f"label metadata with ast.Str/ast.Num ({UPSTREAM_AST})",
                        f"run detection on Python <= 3.13; issue drafted in {UPSTREAM_DOC}",
                        None if severity == SKIP else ErrorCode.ENGINE_LOAD_FAILED,
                        {"version": text, "upstream": UPSTREAM_AST})]
    return [Finding("python", OK, f"Python {text} "
                    f"(floor {_vtext(PYTHON_FLOOR)}, detection ceiling < "
                    f"{_vtext(IMGUTILS_PYTHON_CEILING)})", data={"version": text})]


def _numpy_pins(ctx: Context) -> list[str]:
    """Every numpy requirement the *installed* imgutils declares.

    Read from its own metadata, never from a constant: when upstream moves the pin the
    doctor moves with it, and when a wheel declares two numpy lines (one per extra) the
    first one is not the whole answer.
    """
    pins: list[str] = []
    for line in ctx.requires("dghs-imgutils"):
        base = line.split(";")[0].strip()
        if base.lower().startswith("numpy"):
            # metadata spells it `numpy (<2)`; the report spells it `numpy <2`
            pins.append(base.replace("(", " ").replace(")", "").replace("  ", " ").strip())
    return pins


def check_dependencies(ctx: Context) -> list[Finding]:
    """The three dependency facts that produce the confusing symptoms.

    Order matters for the reading: imgutils' presence, then the ONNX runtime it installs
    for you, then the numpy pin it fights with.
    """
    out: list[Finding] = []
    py314 = ctx.python_version() >= IMGUTILS_PYTHON_CEILING

    # --- the detection backend ------------------------------------------------
    imgutils = ctx.version("dghs-imgutils")
    if imgutils is None:
        out.append(Finding("imgutils", FAIL if ctx.needs_engines() else SKIP,
                           "dghs-imgutils is not installed: stage 1 (YOLO) cannot run",
                           'pip install -e ".[detect]"  (then read the numpy note below)',
                           None if not ctx.needs_engines() else ErrorCode.ENGINE_LOAD_FAILED))
    else:
        declared = _numpy_pins(ctx)
        out.append(Finding("imgutils", OK, f"dghs-imgutils {imgutils} declared "
                        f"{', '.join(declared) or 'no numpy pin'}",
                           data={"version": imgutils, "numpy_pins": declared}))

    # --- the ONNX runtime, checked WITHOUT importing imgutils -----------------
    ort = ctx.version("onnxruntime")
    if ort is None:
        if ctx.has_module("onnxruntime-gpu") or ctx.has_module("onnxruntime_gpu"):
            out.append(Finding("onnxruntime", OK, "onnxruntime-gpu is installed "
                               "(the GPU build also provides `onnxruntime`)"))
        elif imgutils is not None and not py314:
            out.append(Finding(
                "onnxruntime", FAIL if ctx.needs_engines() else SKIP,
                f"onnxruntime is missing while dghs-imgutils {imgutils} is installed: "
                f"importing imgutils runs pip install at import time ({UPSTREAM_AUTO_INSTALL}) "
                "— a machine told to stay offline would reach the network here",
                f"pip install onnxruntime      # or onnxruntime-gpu for --device cuda; "
                f"issue drafted in {UPSTREAM_DOC}#1",
                None if not ctx.needs_engines() else ErrorCode.ENGINE_LOAD_FAILED,
                {"upstream": UPSTREAM_AUTO_INSTALL}))
        else:
            out.append(Finding("onnxruntime", SKIP,
                               "onnxruntime is not installed (and neither is imgutils)"))
    else:
        out.append(Finding("onnxruntime", OK,
                           f"onnxruntime {ort} is importable on its own, so imgutils has "
                           "nothing to install at import time",
                           data={"version": ort}))

    # --- numpy against the pin imgutils declares ------------------------------
    numpy = ctx.version("numpy")
    pins = _numpy_pins(ctx) if imgutils is not None else []
    ceilings = [_parse_version(p.partition("<")[2]) for p in pins
                if "<" in p and ">=" not in p.split("<")[0]]
    ceilings = [c for c in ceilings if c]
    if numpy is None:
        out.append(Finding("numpy", FAIL, "numpy is not installed", 'pip install "numpy>=1.24"',
                           ErrorCode.ENGINE_LOAD_FAILED))
    elif any(_parse_version(numpy) >= c for c in ceilings):
        violated = ", ".join(pins)
        out.append(Finding(
            "numpy", WARN,
            f"numpy {numpy} against imgutils' pin `{violated}`: it works because imgutils is "
            "installed with --no-deps, but a plain `pip install dghs-imgutils` would "
            "downgrade numpy underneath torch",
            "leave it as it is, or install numpy 1.x before installing imgutils "
            "(README §Install: the dghs-imgutils wrinkle)",
            data={"version": numpy, "pins": pins}))
    else:
        satisfied = ", ".join(pins)
        out.append(Finding("numpy", OK,
                           f"numpy {numpy} satisfies {satisfied}" if satisfied
                           else f"numpy {numpy}",
                           data={"version": numpy}))
    return out


def check_torch(ctx: Context) -> list[Finding]:
    """torch and CUDA, without importing either."""
    out: list[Finding] = []
    torch = ctx.version("torch")
    device = ctx.config.ocr.device
    wants_cuda = device.startswith("cuda")
    nvidia_smi = ctx.executable("nvidia-smi")

    if torch is None:
        out.append(Finding("torch", FAIL if ctx.needs_engines() else SKIP,
                           "torch is not installed: stage 2 (OCR) cannot run",
                           'pip install -e ".[ocr]"',
                           None if not ctx.needs_engines() else ErrorCode.ENGINE_LOAD_FAILED))
    else:
        build = torch.partition("+")[2]
        if build.lower().startswith("cu"):
            out.append(Finding("torch", OK, f"torch {torch} is a CUDA build",
                               data={"version": torch, "cuda_build": build}))
        elif wants_cuda:
            out.append(Finding(
                "torch", FAIL if nvidia_smi is None else WARN,
                f"torch {torch} is not a CUDA build but ocr.device is {device!r}: the OCR "
                "stage will answer ENGINE_LOAD_FAILED",
                "pip install torch --index-url https://download.pytorch.org/whl/cu128 "
                "(Blackwell needs cu128 or newer)",
                None if nvidia_smi is not None else ErrorCode.ENGINE_LOAD_FAILED,
                {"version": torch, "requested_device": device}))
        else:
            out.append(Finding("torch", OK, f"torch {torch} (CPU wheel)",
                               data={"version": torch}))

    transformers = ctx.version("transformers")
    if transformers is None:
        out.append(Finding("transformers", FAIL if ctx.needs_engines() else SKIP,
                           "transformers is not installed: the OCR snapshot cannot load",
                           'pip install -e ".[ocr]"',
                           None if not ctx.needs_engines() else ErrorCode.ENGINE_LOAD_FAILED))
    else:
        out.append(Finding("transformers", OK, f"transformers {transformers}",
                           data={"version": transformers}))

    if nvidia_smi is None:
        severity = FAIL if (wants_cuda and torch is not None) else (
            SKIP if not ctx.needs_engines() else OK)
        out.append(Finding("cuda", severity,
                           "no nvidia-smi on PATH: this looks like a CPU-only machine",
                           "--device cpu, or install the CUDA torch wheel for this GPU"
                           if wants_cuda else "",
                           None if severity != FAIL else ErrorCode.ENGINE_LOAD_FAILED,
                           {"requested_device": device}))
    else:
        out.append(Finding("cuda", OK,
                           f"nvidia-smi at {nvidia_smi}; whether torch and ONNX can use it is "
                           "measured by --warm-devices cpu,cuda",
                           data={"nvidia_smi": nvidia_smi}))
    return out


def check_doors(ctx: Context) -> list[Finding]:
    """The two optional doors: the window and the MCP endpoint."""
    out: list[Finding] = []
    pyside = ctx.version("PySide6", "PySide6_Essentials")
    if pyside is None and ctx.has_module("PySide6"):
        pyside = "(importable, no version metadata)"
    out.append(Finding("PySide6", OK if pyside else WARN,
                       f"PySide6 {pyside}" if pyside else
                       "PySide6 is not installed: the window cannot open, --headless still can",
                       "" if pyside else 'pip install -e ".[gui]"',
                       data={"version": pyside} if pyside else {}))

    mcp = ctx.version("mcp")
    if mcp is None:
        out.append(Finding("mcp", WARN if ctx.config.mcp.enabled else OK,
                           "the mcp SDK is not installed: --serve and the MCP tab are off",
                           'pip install -e ".[mcp]"'))
    else:
        floor = (2, 3)
        if _parse_version(mcp) < floor and ctx.config.mcp.enabled:
            out.append(Finding("mcp", FAIL,
                               f"mcp {mcp} is older than 2.3: the server imports "
                               "mcp.server.mcpserver, which 1.x spells differently",
                               "pip install -U 'mcp>=2.3'", ErrorCode.ENGINE_LOAD_FAILED,
                               {"version": mcp}))
        else:
            out.append(Finding("mcp", OK, f"mcp {mcp}", data={"version": mcp}))

    if ctx.config.mcp.enabled and ctx.config.mcp.transport == "streamable-http":
        try:                                        # the endpoint warning lives in one place
            from ..mcp.server import security_warnings

            for warning in security_warnings(ctx.config.mcp):
                out.append(Finding("mcp endpoint", WARN, warning,
                                   "set mcp.auth_token, or bind 127.0.0.1"))
        except Exception:                           # noqa: BLE001 - no SDK, no opinion
            pass
    return out


def check_bundle(ctx: Context) -> list[Finding]:
    """The on-premises weights: complete, truncated, or absent. No hashing (D90)."""
    out: list[Finding] = []
    store = ctx.store
    status = store.status()
    model = ctx.config.detection.model
    offline = store.offline

    if not status["exists"]:
        out.append(Finding("bundle", FAIL if offline else WARN,
                           f"no model bundle at {status['models_dir']}",
                           "python scripts/fetch_models.py --all --ocr",
                           ErrorCode.MODEL_NOT_FOUND if offline else None,
                           {"models_dir": status["models_dir"]}))
    else:
        variants = status["detection_bundled"]
        entry = status["detection_variants"].get(model, {})
        if not entry.get("bundled"):
            out.append(Finding(
                "bundle.detection", FAIL if offline else WARN,
                f"{model} is not in the bundle ({len(variants)} variant(s) are: "
                f"{', '.join(variants) or 'none'})",
                f"python scripts/fetch_models.py --variants {model}"
                + ("" if offline else f"   # or let it fetch from {DETECTION_REPO}"),
                ErrorCode.MODEL_NOT_FOUND if offline else ErrorCode.HF_AUTH_REQUIRED,
                {"model": model, "bundled": variants}))
        else:
            out.append(Finding("bundle.detection", OK,
                               f"{model} is on premises "
                               f"({entry['bytes'] / 1024 ** 2:.0f} MB, "
                               f"F1 {MODEL_PRESETS[model]['f1'] if model in MODEL_PRESETS else '?'})",
                               data={"model": model}))

        ocr = status["ocr"]
        if not ocr["bundled"]:
            out.append(Finding("bundle.ocr", FAIL if offline else WARN,
                               "the Hayai snapshot is incomplete"
                               + (f": missing {', '.join(ocr['missing'])}" if ocr["missing"] else ""),
                               "python scripts/fetch_models.py --ocr",
                               ErrorCode.MODEL_NOT_FOUND if offline else None))
        else:
            out.append(Finding("bundle.ocr", OK,
                               f"Hayai snapshot on premises ({ocr['bytes'] / 1024 ** 2:.0f} MB"
                               + (f", rev {ocr['revision'][:12]}" if ocr.get("revision") else "")
                               + ")"))

        # Manifest cross-check: recorded files that are gone, and recorded sizes that do
        # not match. Sizes only — hashing 1.3 GB belongs to `fetch_models.py --verify`,
        # not to a command that is supposed to answer in under a second (decision D90).
        manifest_path = Path(store.root) / MANIFEST_NAME
        if manifest_path.is_file():
            import json

            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                out.append(Finding("bundle.manifest", WARN,
                                   f"MANIFEST.json cannot be read: {type(exc).__name__}",
                                   "python scripts/fetch_models.py --verify"))
            else:
                absent, truncated = [], []
                for path, recorded in resolve_manifest_files(manifest, store.root):
                    if not path.is_file():
                        absent.append(path.name if path.parent == Path(store.root)
                                      else f"{path.parent.name}/{path.name}")
                    elif recorded.get("size") and path.stat().st_size != recorded["size"]:
                        truncated.append(f"{path.parent.name}/{path.name}")
                if truncated:
                    out.append(Finding("bundle.manifest", FAIL,
                                       f"{len(truncated)} recorded file(s) are a different size "
                                       f"than MANIFEST.json says: {', '.join(truncated[:5])}",
                                       "python scripts/fetch_models.py --verify",
                                       ErrorCode.ENGINE_LOAD_FAILED,
                                       {"truncated": truncated}))
                elif absent:
                    out.append(Finding("bundle.manifest", WARN,
                                       f"{len(absent)} file(s) MANIFEST.json recorded are gone: "
                                       f"{', '.join(absent[:5])}",
                                       "python scripts/fetch_models.py --verify",
                                       {"absent": absent}))
                else:
                    out.append(Finding("bundle.manifest", OK,
                                       f"MANIFEST.json: {len(manifest.get('files', []))} files, "
                                       "all present at their recorded sizes "
                                       "(hash-check them with fetch_models.py --verify)"))
        else:
            out.append(Finding("bundle.manifest", WARN,
                               "the bundle has no MANIFEST.json, so a truncated copy cannot be "
                               "told from the real thing",
                               "python scripts/fetch_models.py --all --ocr"))

    free = ctx.free_bytes(Path(store.root))
    if free is not None:
        low = free < 1024 ** 3
        out.append(Finding("disk", WARN if low else OK,
                           f"{free / 1024 ** 3:.1f} GB free where the bundle lives "
                           f"({store.root})",
                           "a full disk is the usual cause of a truncated weight file"
                           if low else "", data={"free_bytes": free}))
    return out


def check_token(ctx: Context) -> list[Finding]:
    """Where the token came from — never what it is."""
    env_name = ctx.config.hf.token_env
    source: Optional[str] = None
    length: Optional[int] = None
    if ctx.hf_token:
        source, length = "--hf-token", len(ctx.hf_token)
    elif ctx.env(env_name):
        source, length = f"env:{env_name}", len(ctx.env(env_name) or "")
    elif ctx.env("HUGGING_FACE_HUB_TOKEN"):
        source, length = "env:HUGGING_FACE_HUB_TOKEN", len(ctx.env("HUGGING_FACE_HUB_TOKEN") or "")
    else:
        found = ctx.token_file()
        if found is not None:
            from .model_store import read_token_file

            value = read_token_file(found)
            if value:
                source, length = f"file:{found}", len(value)
    if source is None:
        needs = not ctx.store.has_detection_variant(ctx.config.detection.model)
        if needs and not ctx.store.offline and ctx.needs_engines():
            return [Finding("token", WARN,
                            f"no token found, and {ctx.config.detection.model} is not on "
                            f"premises: {DETECTION_REPO} is gated and will answer 401",
                            f"accept the terms, then set {env_name} "
                            "(python scripts/check_hf_access.py proves it works)",
                            ErrorCode.HF_AUTH_REQUIRED)]
        return [Finding("token", OK,
                        "no token — and none needed: every weight this app loads is on "
                        "premises or public"
                        if not needs else "no token, and none needed while --offline",
                        data={"source": "none"})]
    return [Finding("token", OK, f"token from {source} ({length} characters; the value is "
                    "never printed here)", data={"source": source, "length": length})]


def check_fonts(ctx: Context) -> list[Finding]:
    face, cjk = ctx.font_face()
    if face is None:
        return [Finding("fonts", WARN, "no CJK-capable font found at the known paths",
                        "install one (fonts-noto-cjk): exported PNG labels fall back to "
                        "boxes-of-doom (□□□), the README's worst symptom")]
    if not cjk:
        return [Finding("fonts", WARN, f"the face Pillow will pick ({face}) has no Japanese "
                        "glyphs: exported labels become boxes-of-doom",
                        "install a Noto CJK font (fonts-noto-cjk)")]
    return [Finding("fonts", OK, f"CJK face available for exports: {face}",
                    data={"face": face})]


#: The examination, in the order it reads best on a terminal.
DOCTOR_CHECKS: tuple[Callable[[Context], list[Finding]], ...] = (
    check_python,
    check_dependencies,
    check_torch,
    check_doors,
    check_bundle,
    check_token,
    check_fonts,
)


def run_doctor(config: Optional[AppConfig] = None, *, mock: bool = False,
               hf_token: Optional[str] = None,
               probes: Optional[Probes] = None) -> DoctorReport:
    """Examine the machine. Read-only, offline-safe, and it never imports a backend."""
    from .. import APP_NAME, __version__

    config = config or AppConfig.default()
    started = time.perf_counter()
    ctx = Context(config=config, probes=probes or Probes(),
                  store=ModelStore(config.paths.models_dir, offline=config.paths.offline),
                  mock=mock, hf_token=hf_token)
    findings: list[Finding] = []
    for check in DOCTOR_CHECKS:
        findings.extend(check(ctx))
    took = int((time.perf_counter() - started) * 1000)
    return DoctorReport(
        findings=tuple(findings),
        context={"app": APP_NAME, "version": __version__,
                 "python": _vtext(ctx.python_version()),
                 "platform": platform.system().lower() or platform.platform(),
                 "mock": mock, "offline": config.paths.offline,
                 "models_dir": str(ctx.store.root)},
        took_ms=took,
    )
