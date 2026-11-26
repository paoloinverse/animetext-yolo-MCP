"""Configuration system: defaults, presets, validation, merge, import/export.

Contract: PLAN.md §2.2 (config schema 1.0). Validation is native Python for
precise, path-based error messages (decision D5); the published JSON Schema in
``animetext_studio/schemas/config.schema.json`` mirrors these rules and is
cross-checked in tests.

Semantics (decision D10):
- missing keys  -> defaults are used (warning when validating a file)
- unknown keys  -> warning (error in strict mode)
- invalid value -> always an error (StudioError INVALID_CONFIG)
"""
from __future__ import annotations

import codecs
import copy
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

from .types import (
    CONFIG_SCHEMA_VERSION, SUPPORTED_CONFIG_SCHEMA_VERSIONS, SCHEMA_VERSION,
    ErrorCode, StudioError,
)

# ---------------------------------------------------------------------------
# Model knowledge (from the deepghs/AnimeText_yolo model card)
# ---------------------------------------------------------------------------

#: name -> metadata; recommended_score_threshold is the F1-optimal threshold
#: published by the trainers (also shipped per-model as threshold.json).
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "yolo12n_animetext": {"params_m": 2.57, "flops_g": 6.48, "f1": 0.86,
                          "recommended_score_threshold": 0.251},
    "yolo12s_animetext": {"params_m": 9.25, "flops_g": 21.5, "f1": 0.88,
                          "recommended_score_threshold": 0.272},
    "yolo12m_animetext": {"params_m": 20.1, "flops_g": 67.7, "f1": 0.89,
                          "recommended_score_threshold": 0.299},
    "yolo12l_animetext": {"params_m": 26.4, "flops_g": 89.4, "f1": 0.89,
                          "recommended_score_threshold": 0.426},
    "yolo12x_animetext": {"params_m": 59.1, "flops_g": 200.0, "f1": 0.90,
                          "recommended_score_threshold": 0.425},
}
KNOWN_MODELS: tuple[str, ...] = tuple(MODEL_PRESETS)

READING_ORDERS = ("score", "top_to_bottom", "manga_rtl")
ZOOM_MODES = ("fit", "percent")
MCP_TRANSPORTS = ("streamable-http", "stdio", "none")

#: Bounds for `mcp.log_capacity` (the activity log / `log://mcp` ring buffer, P8).
LOG_CAPACITY_MIN, LOG_CAPACITY_MAX = 10, 10_000
LOG_CAPACITY_DEFAULT = 200
#: The patch budgets the OCR processor really supports. Upstream picks among
#: {128, 192, 256, 384, 512} (`processing_hayai._select_best_max_patches`) and its own
#: escalation targets 512; our engine used to stop at 384, which made the top tier
#: unreachable even though the vendored code implements it (found while writing
#: ADDENDUM.md, folded in as P8.2b). `max_scale=4.0` still decides whether a given crop
#: can actually use a bigger budget — asking for 512 on a small crop is legal and a no-op.
MAX_PATCHES_CHOICES: tuple[Union[str, int], ...] = ("auto", 128, 192, 256, 384, 512)

#: Integer-valued config fields. JSON Schema counts `16.0` as a valid integer, so the
#: native validator accepts integral floats too and `_normalise()` rewrites them to
#: real ints on load; without this the published schema and the app disagree (D22).
_INT_FIELDS: dict[str, tuple[str, ...]] = {
    "ocr": ("batch_size", "max_new_tokens", "crop_padding_px"),
    "pipeline": ("min_box_area_px",),
    "gui": ("zoom_percent",),
    "mcp": ("port", "log_capacity"),
}


def _is_integral(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()


def _normalise(data: dict[str, Any]) -> dict[str, Any]:
    """Integral floats -> ints for the fields in `_INT_FIELDS` (input untouched)."""
    if not isinstance(data, dict):      # non-dict input: let the validator report it
        return data
    out: dict[str, Any] = dict(data)
    for section, keys in _INT_FIELDS.items():
        sub = out.get(section)
        if not isinstance(sub, dict):
            continue
        sub = dict(sub)
        for key in (*keys, "max_patches"):
            if _is_integral(sub.get(key)):
                sub[key] = int(sub[key])
        out[section] = sub
    return out


DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:[0-9]+)?)$")


# ---------------------------------------------------------------------------
# Config dataclasses (defaults = official Space demo defaults, decision D3)
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    model: str = "yolo12m_animetext"
    iou_threshold: float = 0.35
    score_threshold: float = 0.05
    allow_dynamic: bool = False


@dataclass
class OcrConfig:
    device: str = "auto"
    max_patches: Union[str, int] = "auto"
    batch_size: int = 16
    max_new_tokens: int = 128
    crop_padding_px: int = 2
    normalize_nfkc: bool = False


@dataclass
class PipelineConfig:
    reading_order: str = "manga_rtl"
    min_box_area_px: int = 16


@dataclass
class GuiConfig:
    overlay_visible: bool = True
    fill_opacity: float = 0.18
    labels_visible: bool = True
    hover_opacity: float = 0.5
    zoom_mode: str = "fit"
    zoom_percent: int = 100


@dataclass
class McpConfig:
    enabled: bool = True
    transport: str = "streamable-http"
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token: Optional[str] = None
    #: How many recent tool calls the activity log / `log://mcp` keeps (P8). A long harness
    #: session wants more than the 200-entry ring buffer held; a memory-tight one wants fewer.
    log_capacity: int = 200


@dataclass
class PathsConfig:
    hf_cache_dir: Optional[str] = None
    last_image_dir: Optional[str] = None
    export_dir: Optional[str] = None
    #: On-premises model bundle (decision D26). None -> $ANIMETEXT_MODELS_DIR ->
    #: <package>/models. Weights found there are used instead of being downloaded.
    models_dir: Optional[str] = None
    #: True -> never touch the network: only bundled weights (or an already-populated
    #: HF cache) may be used, and anything missing fails with MODEL_NOT_FOUND.
    offline: bool = False


@dataclass
class HfConfig:
    token_env: str = "HF_TOKEN"


_SECTIONS: dict[str, type] = {
    "detection": DetectionConfig,
    "ocr": OcrConfig,
    "pipeline": PipelineConfig,
    "gui": GuiConfig,
    "mcp": McpConfig,
    "paths": PathsConfig,
    "hf": HfConfig,
}


@dataclass
class AppConfig:
    schema_version: str = CONFIG_SCHEMA_VERSION
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    hf: HfConfig = field(default_factory=HfConfig)

    # -- construction -------------------------------------------------------

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any], strict: bool = False) -> "AppConfig":
        data = _normalise(data)
        errors, _warnings = validate_config_dict(data, strict=strict)
        if errors:
            raise StudioError(
                ErrorCode.INVALID_CONFIG,
                f"Invalid configuration ({len(errors)} issue(s)).",
                details={"issues": errors},
            )
        sections: dict[str, Any] = {}
        for name, dc in _SECTIONS.items():
            sub = data.get(name) or {}
            sections[name] = dc(**{f.name: sub[f.name] for f in fields(dc) if f.name in sub})
        return cls(schema_version=data.get("schema_version", CONFIG_SCHEMA_VERSION), **sections)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema_version": self.schema_version}
        for name in _SECTIONS:
            section = getattr(self, name)
            out[name] = {f.name: getattr(section, f.name) for f in fields(section)}
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # -- partial update (MCP set_config semantics) ---------------------------

    def merged(self, patch: dict[str, Any], strict: bool = False) -> tuple["AppConfig", list[str]]:
        """Return (new_config, changed_key_paths). Self is never mutated.

        Invalid patches raise StudioError(INVALID_CONFIG); unknown keys are
        warnings unless strict=True.
        """
        base = self.to_dict()
        merged = _deep_merge(base, patch)
        new_cfg = AppConfig.from_dict(merged, strict=strict)
        before, after = _flatten(base), _flatten(new_cfg.to_dict())
        changed = sorted(
            k for k in set(before) | set(after) if before.get(k, _MISSING) != after.get(k, _MISSING)
        )
        return new_cfg, changed


_MISSING = object()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


# ---------------------------------------------------------------------------
# Validation (native; mirrors schemas/config.schema.json)
# ---------------------------------------------------------------------------

def _issue(path: str, message: str) -> dict[str, str]:
    return {"path": path, "message": message}


def _check_bool(errors: list, path: str, value: Any) -> None:
    if not isinstance(value, bool):
        errors.append(_issue(path, f"expected boolean, got {type(value).__name__}"))


def _check_number(errors: list, path: str, value: Any, lo: float, hi: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(_issue(path, f"expected number in [{lo}, {hi}], got {value!r}"))
    elif not (lo <= value <= hi):
        errors.append(_issue(path, f"value {value!r} out of range [{lo}, {hi}]"))


def _check_int(errors: list, path: str, value: Any, lo: int, hi: int) -> None:
    # integral floats (16.0) are accepted here and coerced to int by _normalise()
    if not _is_integral(value):
        errors.append(_issue(path, f"expected integer in [{lo}, {hi}], got {value!r}"))
    elif not (lo <= value <= hi):
        errors.append(_issue(path, f"value {value!r} out of range [{lo}, {hi}]"))


def _check_enum(errors: list, path: str, value: Any, choices: tuple) -> None:
    if value not in choices:
        errors.append(_issue(path, f"expected one of {list(choices)!r}, got {value!r}"))


def _check_opt_str(errors: list, path: str, value: Any) -> None:
    if value is not None and not isinstance(value, str):
        errors.append(_issue(path, f"expected string or null, got {value!r}"))


def validate_config_dict(
    data: Any, strict: bool = False, warn_missing: bool = False
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (errors, warnings) as lists of {path, message}.

    Errors make the config invalid; warnings cover unknown keys (errors when
    strict=True) and, with warn_missing=True, keys that will fall back to defaults.
    """
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    unknown_sink = errors if strict else warnings

    if not isinstance(data, dict):
        return [_issue("", f"config must be a JSON object, got {type(data).__name__}")], []
    data = _normalise(data)   # 16.0 -> 16 before both validation and construction

    known_top = {"schema_version", *_SECTIONS}
    for key in data:
        if key not in known_top:
            unknown_sink.append(_issue(str(key), "unknown key"))

    version = data.get("schema_version", CONFIG_SCHEMA_VERSION)
    if version not in SUPPORTED_CONFIG_SCHEMA_VERSIONS:
        errors.append(_issue(
            "schema_version",
            f"unsupported schema version {version!r} "
            f"(supported: {list(SUPPORTED_CONFIG_SCHEMA_VERSIONS)!r})",
        ))

    subs: dict[str, dict[str, Any]] = {}
    for section in _SECTIONS:
        value = data.get(section, {})
        if not isinstance(value, dict):
            errors.append(_issue(section, f"expected object, got {type(value).__name__}"))
            value = {}
        known = {f.name for f in fields(_SECTIONS[section])}
        for key in value:
            if key not in known:
                unknown_sink.append(_issue(f"{section}.{key}", "unknown key"))
        if warn_missing:
            for key in sorted(known - set(value)):
                warnings.append(_issue(f"{section}.{key}", "missing key; default will be used"))
        subs[section] = value

    def get(section: str, key: str, default: Any) -> Any:
        return subs[section].get(key, default)

    d = DetectionConfig()
    model = get("detection", "model", d.model)
    if not isinstance(model, str) or not model:
        errors.append(_issue("detection.model", "expected non-empty string"))
    elif model not in MODEL_PRESETS:
        warnings.append(_issue("detection.model", f"model {model!r} not in known presets {list(KNOWN_MODELS)}"))
    _check_number(errors, "detection.iou_threshold", get("detection", "iou_threshold", d.iou_threshold), 0.0, 1.0)
    _check_number(errors, "detection.score_threshold", get("detection", "score_threshold", d.score_threshold), 0.0, 1.0)
    _check_bool(errors, "detection.allow_dynamic", get("detection", "allow_dynamic", d.allow_dynamic))

    o = OcrConfig()
    device = get("ocr", "device", o.device)
    if not isinstance(device, str) or not DEVICE_RE.match(device):
        errors.append(_issue("ocr.device", f"expected 'auto', 'cpu', 'cuda' or 'cuda:N', got {device!r}"))
    max_patches = get("ocr", "max_patches", o.max_patches)
    if isinstance(max_patches, bool) or max_patches not in MAX_PATCHES_CHOICES:
        errors.append(_issue("ocr.max_patches", f"expected 'auto' or one of {list(MAX_PATCHES_CHOICES[1:])}, got {max_patches!r}"))
    _check_int(errors, "ocr.batch_size", get("ocr", "batch_size", o.batch_size), 1, 512)
    _check_int(errors, "ocr.max_new_tokens", get("ocr", "max_new_tokens", o.max_new_tokens), 1, 1024)
    _check_int(errors, "ocr.crop_padding_px", get("ocr", "crop_padding_px", o.crop_padding_px), 0, 64)
    _check_bool(errors, "ocr.normalize_nfkc", get("ocr", "normalize_nfkc", o.normalize_nfkc))

    p = PipelineConfig()
    _check_enum(errors, "pipeline.reading_order", get("pipeline", "reading_order", p.reading_order), READING_ORDERS)
    _check_int(errors, "pipeline.min_box_area_px", get("pipeline", "min_box_area_px", p.min_box_area_px), 0, 100_000)

    g = GuiConfig()
    _check_bool(errors, "gui.overlay_visible", get("gui", "overlay_visible", g.overlay_visible))
    _check_number(errors, "gui.fill_opacity", get("gui", "fill_opacity", g.fill_opacity), 0.0, 1.0)
    _check_bool(errors, "gui.labels_visible", get("gui", "labels_visible", g.labels_visible))
    _check_number(errors, "gui.hover_opacity", get("gui", "hover_opacity", g.hover_opacity), 0.0, 1.0)
    _check_enum(errors, "gui.zoom_mode", get("gui", "zoom_mode", g.zoom_mode), ZOOM_MODES)
    _check_int(errors, "gui.zoom_percent", get("gui", "zoom_percent", g.zoom_percent), 10, 800)

    m = McpConfig()
    _check_bool(errors, "mcp.enabled", get("mcp", "enabled", m.enabled))
    _check_enum(errors, "mcp.transport", get("mcp", "transport", m.transport), MCP_TRANSPORTS)
    host = get("mcp", "host", m.host)
    if not isinstance(host, str) or not host:
        errors.append(_issue("mcp.host", "expected non-empty string"))
    _check_int(errors, "mcp.port", get("mcp", "port", m.port), 1, 65535)
    _check_opt_str(errors, "mcp.auth_token", get("mcp", "auth_token", m.auth_token))
    _check_int(errors, "mcp.log_capacity", get("mcp", "log_capacity", m.log_capacity),
               LOG_CAPACITY_MIN, LOG_CAPACITY_MAX)

    pa = PathsConfig()
    for key in ("hf_cache_dir", "last_image_dir", "export_dir", "models_dir"):
        _check_opt_str(errors, f"paths.{key}", get("paths", key, getattr(pa, key)))
    _check_bool(errors, "paths.offline", get("paths", "offline", pa.offline))

    h = HfConfig()
    token_env = get("hf", "token_env", h.token_env)
    if not isinstance(token_env, str) or not token_env:
        errors.append(_issue("hf.token_env", "expected non-empty string"))

    return errors, warnings


# ---------------------------------------------------------------------------
# Published JSON Schemas — the contract MCP clients code against
#
# Two shapes, on purpose (decision D21):
#   canonical_config_schema() -> what the app EMITS (every key present, unknown keys
#                                forbidden): get_config / export_config outputs.
#   partial_config_schema()   -> what the app ACCEPTS (sections optional, unknown keys
#                                allowed as warnings): set_config / import_config input,
#                                mirroring the lenient native validator (decision D10).
# A single schema could not describe both, which is why the earlier draft disagreed
# with the native validator on unknown keys, missing sections and integral floats.
# ---------------------------------------------------------------------------

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def load_schema(name: str = "config.schema.json") -> dict[str, Any]:
    """Read a schema shipped inside the package (decision D6)."""
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def canonical_config_schema() -> dict[str, Any]:
    return load_schema("config.schema.json")


def partial_config_schema() -> dict[str, Any]:
    """Derive the accepted shape from the canonical one (so they cannot drift apart)."""
    schema = copy.deepcopy(canonical_config_schema())
    schema["$id"] = f"animetext-studio/config-partial-{CONFIG_SCHEMA_VERSION}"
    schema["title"] = "AnimeText OCR Studio configuration patch (accepted shape)"
    schema["description"] = (
        "Shape the application accepts on import and in MCP set_config/import_config: "
        "every section and key is optional (missing values fall back to defaults) and "
        "unknown keys are tolerated but reported as warnings. Field types and ranges "
        f"are identical to the canonical config-{CONFIG_SCHEMA_VERSION} schema, which "
        "describes what the "
        "application emits. Integral floats (16.0) are accepted for integer fields and "
        "normalised to ints."
    )
    schema.pop("required", None)
    schema["additionalProperties"] = True
    for sub in schema.get("properties", {}).values():
        if isinstance(sub, dict):
            sub.pop("required", None)
            sub["additionalProperties"] = True
    return schema


# ---------------------------------------------------------------------------
# Secrets in config documents (P8)
#
# The config document is deliberately faithful: `export_config` -> `import_config` has to
# round-trip a working bearer token, or the feature is broken for the people who use it.
# Redacting is therefore opt-in, per export. What is *not* acceptable is a redacted document
# that, when imported, turns the placeholder into a literal token (silent auth failure) - so
# the sentinel is recognised on the way back in and means "leave the live value alone".
# ---------------------------------------------------------------------------

#: Written in place of a secret by `redacted_config_dict`; recognised by `strip_redacted`.
REDACTED = "<redacted>"

#: Config paths holding a credential. Everything else is meant to be shared.
SECRET_CONFIG_PATHS: tuple[str, ...] = ("mcp.auth_token",)


def redacted_config_dict(config: "AppConfig") -> tuple[dict[str, Any], list[str]]:
    """`config.to_dict()` with every set secret replaced by `REDACTED`.

    Returns `(data, redacted_paths)`. A secret that is null/empty is left alone: there is
    nothing to hide, and writing the sentinel for "no token" would import back as a token.
    """
    data = config.to_dict()
    touched: list[str] = []
    for path in SECRET_CONFIG_PATHS:
        section, _, key = path.partition(".")
        sub = data.get(section)
        if isinstance(sub, dict) and sub.get(key) not in (None, ""):
            sub[key] = REDACTED
            touched.append(path)
    return data, touched


def restore_redacted(config: "AppConfig", current: "AppConfig") -> list[str]:
    """Replace `REDACTED` secrets in `config` with the live values from `current`.

    The import counterpart of :func:`redacted_config_dict`: importing yesterday's redacted
    file must not blank out a working token. Returns the paths that were kept.
    """
    kept: list[str] = []
    for dotted in SECRET_CONFIG_PATHS:
        section, key = dotted.split(".", 1)
        sub, live = getattr(config, section), getattr(current, section)
        if getattr(sub, key) == REDACTED:
            setattr(sub, key, getattr(live, key))
            kept.append(dotted)
    return kept


def strip_redacted(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop `REDACTED` placeholders from an incoming patch (input untouched).

    Returns `(patch_without_them, paths_kept)`. A kept path means "the document was redacted
    here, so the value already in the app stays" - which is what a user pasting a config from
    a bug report expects, and never what they expect from a literal `"<redacted>"` token.
    """
    kept: list[str] = []
    return _strip_redacted(data, "", kept), kept


def _strip_redacted(data: dict[str, Any], prefix: str, kept: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out[key] = _strip_redacted(value, path, kept)
        elif value == REDACTED and path in SECRET_CONFIG_PATHS:
            kept.append(path)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Live keys and restart keys (found by the P8.8 QA pass)
# ---------------------------------------------------------------------------

#: Keys a *running* process cannot re-apply to itself. `StudioServer` is built once from
#: `config.mcp` (`gui/app.py:start_mcp`) and the HTTP gateway captures the bind address, the
#: transport and the bearer token at that moment — nothing re-reads them afterwards.
#:
#: PLAN §2.2 originally promised "applying MCP host/port changes at runtime restarts the server
#: thread". Restarting a listening socket from inside a config patch was never built, and the
#: honest half of that promise is kept instead: the value is stored (so an export is faithful
#: and the next start uses it) and every door that changed it is *told* a restart is needed,
#: rather than being left to assume the port moved. Everything else in the document is live:
#: thresholds, device, budgets, overlay, reading order, `mcp.log_capacity` (8.2).
RESTART_REQUIRED_KEYS: tuple[str, ...] = (
    "mcp.enabled",
    "mcp.transport",
    "mcp.host",
    "mcp.port",
    "mcp.auth_token",
)


def restart_required(changed: Any) -> list[str]:
    """Which of the changed config paths need a process restart, in a stable order.

    Takes any iterable of dotted paths (the `changed` list the config tools already return).
    Empty means "the running app is now exactly what you asked for".
    """
    wanted = {str(path) for path in (changed or ())}
    return [key for key in RESTART_REQUIRED_KEYS if key in wanted]


def changed_config_keys(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Sorted dotted paths whose leaf value differs between two config documents.

    A door that swaps a whole config object (the GUI importing a file) has no `changed` list to
    hand around, and recomputing one at each call site is how the two halves start disagreeing.
    """
    flat_before, flat_after = _flatten(before), _flatten(after)
    missing = object()
    return sorted(path for path in set(flat_before) | set(flat_after)
                  if flat_before.get(path, missing) != flat_after.get(path, missing))


# ---------------------------------------------------------------------------
# File import / export
# ---------------------------------------------------------------------------

def _read_text_file(path: Path, *, what: str = "config file") -> str:
    """Read a text file with the two failures told apart (P8.4).

    `UnicodeDecodeError` is a `ValueError`, not an `OSError`, so the obvious `except OSError`
    lets a UTF-16 or binary file escape as a bare traceback — from `import_config`, which is a
    tool call, so the client would get INTERNAL for something as ordinary as "that is not a
    text file". Say which byte is the problem instead: it is the one detail that settles a
    "my config file is broken" report without opening the file at all.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise StudioError(ErrorCode.IO_ERROR, f"Cannot read {what}: {path}",
                          {"reason": str(exc)}) from exc
    if raw.startswith(codecs.BOM_UTF8):       # a Notepad-saved file: JSON forbids a BOM, we forgive
        raw = raw[len(codecs.BOM_UTF8):]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StudioError(
            ErrorCode.INVALID_CONFIG,
            f"{what.capitalize()} is not UTF-8 text: {path}",
            {"reason": str(exc), "byte_offset": exc.start,
             "first_bytes": raw[:min(len(raw), 8)].hex() or None,
             "hint": "save the file as UTF-8 (no BOM); JSON is UTF-8 by specification"},
        ) from exc


def load_config_file(path: Union[str, Path], strict: bool = False) -> AppConfig:
    path = Path(path)
    text = _read_text_file(path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StudioError(ErrorCode.INVALID_CONFIG, f"Config file is not valid JSON: {path}",
                          {"reason": str(exc)}) from exc
    return AppConfig.from_dict(data, strict=strict)


def save_config_file(config: AppConfig, path: Union[str, Path], *,
                     redact: bool = False) -> Path:
    """Write a config document. `redact` (P8) writes the shareable form (see `REDACTED`)."""
    path = Path(path)
    data, _touched = redacted_config_dict(config) if redact else (config.to_dict(), [])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    except OSError as exc:
        raise StudioError(ErrorCode.IO_ERROR, f"Cannot write config file: {path}", {"reason": str(exc)}) from exc
    return path


def validate_config_file(path: Union[str, Path]) -> dict[str, Any]:
    """Non-throwing report for --validate-config: {valid, errors, warnings}.

    Unreadable still throws (there is nothing to report on); not-JSON and not-UTF-8 are
    reported, because that is exactly what this action is for.
    """
    path = Path(path)
    try:
        text = _read_text_file(path)
    except StudioError as exc:
        if exc.code is ErrorCode.INVALID_CONFIG:      # undecodable: a report can carry that
            return {"valid": False, "errors": [_issue("", exc.message)], "warnings": []}
        raise
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"valid": False, "errors": [_issue("", f"not valid JSON: {exc}")], "warnings": []}
    errors, warnings = validate_config_dict(data, warn_missing=True)
    return {"valid": not errors, "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------------------
# CLI override layering (precedence: CLI > config file > defaults)
# ---------------------------------------------------------------------------

def cli_overrides(args: Any) -> dict[str, Any]:
    """Partial config dict from parsed CLI args (only explicitly-set options).

    Note: --hf-token is deliberately NOT persisted into config (decision D11).
    """
    patch: dict[str, Any] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            patch.setdefault(section, {})[key] = value

    put("detection", "model", getattr(args, "model", None))
    put("detection", "iou_threshold", getattr(args, "iou", None))
    put("detection", "score_threshold", getattr(args, "score", None))
    put("ocr", "device", getattr(args, "device", None))
    put("mcp", "host", getattr(args, "mcp_host", None))
    put("mcp", "port", getattr(args, "mcp_port", None))
    transport = getattr(args, "mcp_transport", None)
    if transport is not None:
        put("mcp", "transport", transport)
        patch.setdefault("mcp", {})["enabled"] = transport != "none"
    put("mcp", "auth_token", getattr(args, "mcp_token", None))
    put("paths", "hf_cache_dir", getattr(args, "hf_cache_dir", None))
    put("paths", "models_dir", getattr(args, "models_dir", None))
    if getattr(args, "offline", False):
        put("paths", "offline", True)
    return patch


def resolve_config(args: Any) -> AppConfig:
    """defaults -> optional --config file -> CLI overrides."""
    cfg = AppConfig.default()
    config_path = getattr(args, "config", None)
    if config_path:
        cfg = load_config_file(config_path)
    patch = cli_overrides(args)
    if patch:
        cfg, _changed = cfg.merged(patch)
    return cfg
