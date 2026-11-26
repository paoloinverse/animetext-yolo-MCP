"""Detection engine: deepghs/AnimeText_yolo via dghs-imgutils (ONNX Runtime).

Verified upstream API (research/imgutils_yolo.py):
    YOLOModel(repo_id, hf_token=None)
    .model_names -> list[str] | OFFLINE sentinel
    .predict(image, model_name, conf_threshold=None, iou_threshold=0.7,
             allow_dynamic=False) -> [((x0,y0,x1,y1), label, score), ...]
    .clear()

Design notes:
- Heavy imports (imgutils / huggingface_hub / tqdm) are lazy so the app runs
  without the [detect] extra (mock mode, GUI development).
- The repo is GATED: downloads need an HF token whose account accepted the
  terms at https://huggingface.co/deepghs/AnimeText_yolo -> HF_AUTH_REQUIRED
  with actionable steps. Token comes from --hf-token / HF_TOKEN env and is
  never persisted (decision D11).
- score_threshold=None delegates to the repo's per-model threshold.json
  (F1-optimal), exactly like imgutils.
- Backend is ONNX Runtime: CPU by default; installing onnxruntime-gpu makes
  CUDA providers available automatically. Provider selection is delegated to
  imgutils through the ONNX_MODE env var (see `configure_onnx_provider`).
- Thread-safe: one RLock guards load/detect/unload (imgutils additionally
  serializes per-session execution internally).

Verified upstream behaviour this wrapper relies on (imgutils 0.19.0, cross-checked
against research/imgutils_yolo.py and the installed package):
- `model_names` returns the private `_OFFLINE` sentinel (NOT a list) when offline,
  but *propagates* GatedRepoError/401 — hence the broad except in list_models().
- `_open_model` caches sessions per `(pid, thread_ident, model_name)`: warming a
  model on thread A does not reuse the session on thread B (the file is already
  on disk, so a re-open is cheap). Our `_loaded` set is therefore an *engine-level*
  logical flag, not a session handle (decision D14).
- `predict()` reads `model_type.json`/`threshold.json` lazily; pre-fetching them in
  `load()` surfaces auth problems early and makes later runs offline-friendly (D15).
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Optional

from PIL import Image

from ..core.config import KNOWN_MODELS
from ..core.model_store import ModelStore, discover_hf_token
from ..core.types import BBox, Detection, DETECTION_REPO, ErrorCode, StudioError
from dataclasses import replace

from .base import (
    BaseDetectionEngine, EngineState, EngineStatus, ProgressCallback,
    check_cancel, model_catalog, report,
)

INSTALL_HINT = ("pip install animetext-studio[detect]  "
                "(or: pip install dghs-imgutils onnxruntime huggingface_hub)")
GATED_REPO_URL = "https://huggingface.co/deepghs/AnimeText_yolo"

#: Accepted values for ONNX Runtime provider selection (engine/CLI level, not part
#: of config schema 1.0). "auto" leaves imgutils' own detection in charge:
#: CUDAExecutionProvider when the installed onnxruntime exposes it, else CPU.
PROVIDER_CHOICES: tuple[str, ...] = (
    "auto", "cpu", "cuda", "gpu", "tensorrt", "trt", "coreml", "openvino", "directml",
)


def configure_onnx_provider(provider: Optional[str] = None) -> None:
    """Set/clear ONNX_MODE, which imgutils reads when opening a session.

    imgutils: `open_onnx_model(ckpt, mode=None)` -> `get_onnx_provider(os.environ['ONNX_MODE'])`.
    'auto' clears the variable so that CUDA is picked up automatically when
    onnxruntime-gpu is installed; 'cpu' forces CPU even on a GPU machine.
    """
    if provider is not None and provider not in PROVIDER_CHOICES:
        raise StudioError(
            ErrorCode.INVALID_ARGUMENT,
            f"Unknown ONNX provider {provider!r}.",
            {"choices": list(PROVIDER_CHOICES)},
        )
    if provider in (None, "", "auto"):
        os.environ.pop("ONNX_MODE", None)
    else:
        os.environ["ONNX_MODE"] = provider


def configure_offline(offline: bool) -> None:
    """Turn Hugging Face network access off (never on) — decision D26.

    Same import-time trap as :func:`configure_hf_cache_dir`: huggingface_hub resolves
    ``HF_HUB_OFFLINE`` into a module constant at import, so both the env var and the
    constant are set. Only ever *enables* offline mode; a False argument leaves a
    user-supplied ``HF_HUB_OFFLINE=1`` untouched (we must not silently grant network
    access the operator withheld).
    """
    if not offline:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True


def configure_hf_cache_dir(cache_dir: Optional[str]) -> None:
    """Point huggingface_hub at `cache_dir` — even if it was already imported.

    huggingface_hub resolves HF_HUB_CACHE into a *module-level string constant*
    at import time (hf_hub_download does `cache_dir = constants.HF_HUB_CACHE`), so
    setting the env var alone is silently ignored in a long-running process such as
    the GUI/MCP server. We set the env var (late importers + subprocesses) and patch
    the constant when the module is already loaded (decision D13).
    """
    if not cache_dir:
        return
    path = os.path.abspath(os.path.expanduser(str(cache_dir)))
    os.environ["HF_HUB_CACHE"] = path
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_CACHE = path
    return path


# ---------------------------------------------------------------------------
# Lazy import seams (monkeypatched in unit tests)
# ---------------------------------------------------------------------------

def _import_yolo_model():
    """Import YOLOModel lazily; tolerate broken optional siblings of the subpackage.

    `imgutils.generic.__init__` eagerly imports every submodule (clip, yoloseg,
    timm/opencv-backed helpers). On a minimal `[detect]` install those can fail even
    though the YOLO path itself is perfectly usable, so we fall back to the direct
    module import (decision D16).
    """
    try:
        from imgutils.generic import YOLOModel
        return YOLOModel
    except ImportError as exc_pkg:
        try:
            from imgutils.generic.yolo import YOLOModel
            return YOLOModel
        except ImportError:
            raise StudioError(
                ErrorCode.ENGINE_LOAD_FAILED,
                "Detection backend not installed (dghs-imgutils).",
                {"hint": INSTALL_HINT, "reason": str(exc_pkg)},
            ) from exc_pkg


# ---------------------------------------------------------------------------
# On-premises (bundled) weights — decision D26
#
# imgutils always resolves `<variant>/model.onnx` through hf_hub_download, so the only
# clean way to feed it files from disk is a thin subclass that resolves locally first
# and delegates to the Hub otherwise. The class is built at call time because the base
# itself is lazily imported (the app must import without the [detect] extra).
# ---------------------------------------------------------------------------

_BUNDLED_CLS_CACHE: dict[tuple, type] = {}

#: Hard ceiling on the Hugging Face model listing (decision D27). imgutils resolves it
#: with an fsspec glob that has no timeout of its own, and a stalled or 401-looping
#: connection therefore blocks forever — unacceptable in a GUI or an MCP server, where
#: the listing sits on the first `detect()` call. Past this we answer from the bundle
#: plus the documented presets and say so in `status().detail`.
LISTING_TIMEOUT_SECONDS = 10.0

#: Fallback for bases that do not carry imgutils' private `_global_lock` (test
#: stand-ins, alternative backends). One process-wide RLock is enough there.
_FALLBACK_LOCK = threading.RLock()


def _listing_lock():
    """The base class' listing lock when it has one, else a process-wide lock."""
    return _FALLBACK_LOCK


def _bundled_yolo_class() -> type:
    """Build (once per base class) a YOLOModel subclass that prefers local files."""
    base = _import_yolo_model()
    key = (base.__module__, base.__name__, id(base))
    cached = _BUNDLED_CLS_CACHE.get(key)
    if cached is not None:
        return cached[0]

    import ast
    import json
    from pathlib import Path

    try:  # private helper; ast.literal_eval is a faithful fallback for `{0: 'x'}` metadata
        from imgutils.generic.yolo import _safe_eval_names_str as _eval_names
    except Exception:  # pragma: no cover - depends on imgutils internals
        def _eval_names(text):  # noqa: D103
            return ast.literal_eval(str(text))
    try:
        from imgutils.generic.yolo import _model_load_lock  # reuse upstream locking
    except Exception:  # pragma: no cover
        _guard = threading.RLock()

        class _NullLock:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def _model_load_lock(): return _guard   # type: ignore[misc]

    class BundledYOLOModel(base):     # type: ignore[misc, valid-type]
        """YOLOModel with an on-premises directory in front of the Hugging Face Hub.

        ``local_root`` holds `<variant>/model.onnx` (+ optional threshold.json /
        model_type.json / labels.json). Anything missing falls back to the upstream
        Hub path, so this is a strict superset of the parent's behaviour.
        """

        def _init_bundled(self, store, listing_timeout: float = LISTING_TIMEOUT_SECONDS) -> None:
            """Attach the on-premises bundle: a :class:`ModelStore`, a path, or None.

            Delegating the listing to ModelStore is the point: "is this variant usable
            on disk" is then answered identically by the engine, the catalog's
            ``source`` label and ``--model-status`` (a 1 kB half-copied ONNX must not
            count as bundled anywhere). A bare directory is accepted for callers that
            have no store (tests, ad-hoc scripts).
            """
            self.bundled = store if isinstance(store, ModelStore) else None
            self.bundled_root = None if self.bundled is not None else (
                Path(store) if store is not None else None)
            self.listing_timeout = float(listing_timeout)
            self.listing_state = "unset"     # ok | timeout | offline | error:<Type> | unset
            # (model, filename) -> path, for weights that replaced a bundled file *this session*
            # after the bundled one was found corrupt (P9.1). In-memory only: nothing on disk is
            # moved, renamed or deleted, because a corrupt file is the user's to decide about.
            self.override_files: dict[tuple[str, str], Path] = {}

        # -- resolution ----------------------------------------------------

        def _root_dir(self):
            if self.bundled is not None:
                return self.bundled.detection_root
            return getattr(self, "bundled_root", None)

        def local_file(self, model_name: str, filename: str):
            override = getattr(self, "override_files", {}).get((str(model_name), filename))
            if override is not None:
                return override
            root = self._root_dir()
            if root is None:
                return None
            path = root / str(model_name) / filename
            try:
                return path if (path.is_file() and path.stat().st_size > 0) else None
            except OSError:
                return None

        def local_names(self) -> list[str]:
            if self.bundled is not None:
                return self.bundled.bundled_detection_variants()
            root = self._root_dir()
            if root is None or not root.is_dir():
                return []
            out = []
            for child in sorted(root.iterdir()):
                try:
                    if child.is_dir() and (child / "model.onnx").stat().st_size > 0:
                        out.append(child.name)
                except OSError:
                    continue
            return out

        # -- overrides -----------------------------------------------------

        @property
        def model_names(self):
            """Bundled variants first, plus the repo's list — without ever wedging.

            Three guarantees, in order:

            1. **Offline never asks.** With ``paths.offline`` the answer is the bundle
               plus the documented presets, and no socket is opened. (imgutils' own
               listing ignores ``HF_HUB_OFFLINE`` — it builds an fsspec filesystem and
               globs it, which blocks on a live HTTPS call.)
            2. **Online is bounded.** The repo glob runs in a daemon thread with
               :data:`LISTING_TIMEOUT_SECONDS`; a stalled or 401-looping connection
               degrades to bundle + presets instead of freezing the GUI/MCP.
            3. **A failure never hides local weights.** Anything the repo cannot answer
               for, the bundle answers; with nothing bundled the parent's exception (or
               its OFFLINE sentinel) propagates unchanged, so the engine's own fallback
               and ``status().detail`` behave exactly as before.

            The parent's private fields are read defensively: the real ``imgutils``
            YOLOModel always has them, but a lightweight stand-in (unit tests, a future
            backend) must not turn a listing into an AttributeError.
            """
            cached = getattr(self, "_model_names", None)
            if cached is not None:
                return cached
            local = self.local_names()
            fallback = list(dict.fromkeys(local + list(KNOWN_MODELS)))

            if self.bundled is not None and self.bundled.offline:
                self.listing_state = "offline"
                self._model_names = fallback      # also seeds the parent's cache
                return fallback

            box: dict[str, Any] = {}

            def _ask_repo():
                try:
                    box["names"] = super(BundledYOLOModel, self).model_names  # type: ignore[misc]
                except BaseException as exc:      # noqa: BLE001 - reported to the caller
                    box["error"] = exc

            worker = threading.Thread(target=_ask_repo, name="hf-model-listing", daemon=True)
            worker.start()
            worker.join(self.listing_timeout)
            if worker.is_alive():
                # Deliberately not cached: a later call may reach the repo after all.
                self.listing_state = "timeout"
                return fallback
            if "error" in box:
                self.listing_state = f"error:{type(box['error']).__name__}"
                if local:
                    return fallback
                raise box["error"]
            remote = box["names"]
            self.listing_state = "ok"
            if not isinstance(remote, (list, tuple)):   # the OFFLINE sentinel
                return fallback if local else remote
            merged = list(local) + [n for n in remote if n not in local]
            self._model_names = merged
            return list(merged)

        def _open_model(self, model_name: str):
            path = self.local_file(model_name, "model.onnx")
            if path is None:
                return super()._open_model(model_name)          # noqa: super-with-arguments
            try:      # lazy: only opening bundled weights needs onnxruntime plumbing
                from imgutils.utils import open_onnx_model
            except ImportError as exc:
                raise StudioError(
                    ErrorCode.ENGINE_LOAD_FAILED,
                    "Cannot open the bundled ONNX model (dghs-imgutils missing).",
                    {"hint": INSTALL_HINT, "reason": str(exc)},
                ) from exc
            cache_key = os.getpid(), threading.get_ident(), model_name
            self.__dict__.setdefault("_models", {})
            with _model_load_lock():
                if cache_key not in self._models:
                    model = open_onnx_model(str(path))
                    metadata = model.get_modelmeta()
                    if "imgsz" in metadata.custom_metadata_map:
                        size = tuple(json.loads(metadata.custom_metadata_map["imgsz"]))
                        max_infer_size = size if len(size) == 2 else 640
                    else:
                        max_infer_size = 640
                    names_map = _eval_names(metadata.custom_metadata_map["names"])
                    labels = [names_map[i] for i in range(len(names_map))]
                    self._models[cache_key] = (model, max_infer_size, labels, threading.Lock())
            return self._models[cache_key]

        def _get_model_type(self, model_name: str):
            path = self.local_file(model_name, "model_type.json")
            if path is None:
                return super()._get_model_type(model_name)       # noqa: super-with-arguments
            cache = self.__dict__.setdefault("_model_types", {})
            with getattr(self, "_model_meta_lock", _FALLBACK_LOCK):
                if model_name not in cache:
                    try:
                        cache[model_name] = json.loads(path.read_text(encoding="utf-8"))["model_type"]
                    except Exception:
                        cache[model_name] = "yolo"
            return cache[model_name]

        def _get_default_thresholds(self, model_name: str):
            path = self.local_file(model_name, "threshold.json")
            if path is None:
                return super()._get_default_thresholds(model_name)  # noqa: super-with-arguments
            cache = self.__dict__.setdefault("_model_thresholds", {})
            with getattr(self, "_model_meta_lock", _FALLBACK_LOCK):
                if model_name not in cache:
                    try:
                        cache[model_name] = json.loads(path.read_text(encoding="utf-8"))["threshold"]
                    except Exception:
                        cache[model_name] = None
            return cache[model_name]

    # The base is stored alongside the subclass: it keeps the object alive, so the
    # `id()` in the key can never be recycled for a different class (which would hand
    # a later caller a subclass of a stale, garbage-collected base).
    _BUNDLED_CLS_CACHE[key] = (BundledYOLOModel, base)
    return BundledYOLOModel


def _hf_download(repo_id: str, filename: str, token: Optional[str] = None, tqdm_class=None) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise StudioError(
            ErrorCode.ENGINE_LOAD_FAILED,
            "huggingface_hub not installed.",
            {"hint": INSTALL_HINT, "reason": str(exc)},
        ) from exc
    return hf_hub_download(
        repo_id=repo_id, repo_type="model", filename=filename,
        token=token, tqdm_class=tqdm_class,
    )


# ---------------------------------------------------------------------------
# Error classification (by type name / HTTP status: no hard hub dependency)
# ---------------------------------------------------------------------------

_AUTH_NAMES = {"GatedRepoError"}
_NOT_FOUND_NAMES = {"EntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError"}
_NETWORK_NAMES = {"LocalEntryNotFoundError", "OfflineModeIsEnabled", "ConnectionError",
                  "ConnectTimeout", "ReadTimeout", "Timeout", "ProxyError", "SSLError",
                  "ChunkedEncodingError"}


def _classify(exc: Exception, context: str) -> StudioError:
    if isinstance(exc, StudioError):
        return exc
    name = type(exc).__name__
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if name in _AUTH_NAMES or status in (401, 403):
        return StudioError(
            ErrorCode.HF_AUTH_REQUIRED,
            f"Hugging Face denied access to {context}. The detection model repo is gated.",
            {
                "steps": [
                    f"1. Log in and accept the access terms at {GATED_REPO_URL}",
                    "2. Create a Read token at https://huggingface.co/settings/tokens",
                    "3. Provide it via the HF_TOKEN environment variable or --hf-token",
                ],
                "reason": str(exc),
            },
        )
    if name in _NOT_FOUND_NAMES or status == 404:
        return StudioError(ErrorCode.MODEL_NOT_FOUND, f"Not found on Hugging Face: {context}.",
                           {"reason": str(exc)})
    if name in _NETWORK_NAMES or isinstance(exc, (ConnectionError, TimeoutError)):
        return StudioError(ErrorCode.DOWNLOAD_FAILED, f"Download/network failure for {context}.",
                           {"reason": str(exc)})
    if isinstance(exc, ImportError):
        # imgutils itself lazily imports onnxruntime/gradio; always say how to fix it.
        return StudioError(ErrorCode.ENGINE_LOAD_FAILED,
                           f"Detection backend is incomplete at {context}.",
                           {"hint": INSTALL_HINT, "reason": f"{name}: {exc}"})
    return StudioError(ErrorCode.ENGINE_LOAD_FAILED, f"Detection backend failure at {context}.",
                       {"reason": f"{name}: {exc}"})


def _progress_tqdm(progress: ProgressCallback):
    """tqdm subclass reporting integer-percent download progress (throttled)."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None

    class _ProgressTqdm(tqdm):
        def update(self, n=1):
            super().update(n)
            total = getattr(self, "total", None)
            if total:
                percent = int(min(99.0, 100.0 * self.n / total))
                if percent != getattr(self, "_last_percent", -1):
                    self._last_percent = percent
                    report(progress, "download", percent, f"{self.n}/{total} bytes")

    return _ProgressTqdm


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DetectionEngine(BaseDetectionEngine):
    def __init__(
        self,
        repo_id: str = DETECTION_REPO,
        hf_token: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        provider: Optional[str] = None,
        models_dir: Optional[str] = None,
        offline: bool = False,
        listing_timeout: float = LISTING_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(backend="onnxruntime", device=provider or "auto")
        self._repo_id = repo_id
        self._hf_token = hf_token
        self._store = ModelStore(models_dir, offline=offline)
        self._listing_timeout = float(listing_timeout)
        self._yolo: Any = None
        self._names: Optional[list[str]] = None      # cached repo model list
        self._names_offline = False                  # True -> fallback list in use
        # model_name -> the Hub path that replaced a corrupt bundled file this session (P9.1).
        # In-memory only, and reported in `status().detail`; nothing on disk is touched.
        self.recovered_via_hub: dict[str, str] = {}
        configure_onnx_provider(provider)
        configure_hf_cache_dir(hf_cache_dir)
        configure_offline(offline)

    @property
    def model_store(self) -> ModelStore:
        """The on-premises bundle this engine resolves weights from (D26)."""
        return self._store

    #: ONNX Runtime provider name -> short device label used by the GUI/MCP status.
    PROVIDER_DEVICES: dict[str, str] = {
        "CPUExecutionProvider": "cpu",
        "CUDAExecutionProvider": "cuda",
        "TensorrtExecutionProvider": "tensorrt",
        "CoreMLExecutionProvider": "coreml",
        "OpenVINOExecutionProvider": "openvino",
        "DmlExecutionProvider": "directml",
        "AzureExecutionProvider": "azure",
    }

    # -- internals ----------------------------------------------------------

    def _token(self) -> Optional[str]:
        """Explicit token, then ``HF_TOKEN``, then a local token file (see D26/model_store).

        The value is only ever handed to the Hub client: it never reaches `status()`, a
        log line or an exported config.
        """
        return discover_hf_token(self._hf_token)

    def _note_session_providers(self, opened: Any) -> None:
        """Record the provider the session *actually* got.

        `status().device` starts as the configured request ("auto"/"cpu"/"cuda") but the
        GUI and MCP `get_status` must show what will really execute — on a machine whose
        onnxruntime has no CUDA EP (or a Blackwell build without sm_120 kernels), "auto"
        silently lands on CPU. imgutils always appends CPUExecutionProvider as a
        fallback, so the first entry is the one that runs. Never raises.
        """
        try:
            providers = list(opened[0].get_providers())
        except Exception:
            return
        if providers:
            self._device = self.PROVIDER_DEVICES.get(providers[0], providers[0])

    def _ensure_yolo(self):
        """Build the backend: the on-premises bundle in front of the Hub (D26).

        The store itself is handed to the backend (not just a path) so that listing,
        resolution and reporting share one definition of "bundled". With an empty or
        absent bundle every lookup misses and behaviour is exactly upstream's.
        """
        with self._lock:
            if self._yolo is None:
                cls = _bundled_yolo_class()
                self._yolo = cls(self._repo_id, hf_token=self._hf_token)
                self._yolo._init_bundled(self._store, listing_timeout=self._listing_timeout)
            return self._yolo

    def _unload_locked(self) -> None:
        had = self._yolo is not None
        if self._yolo is not None:
            try:
                self._yolo.clear()      # drops the cached InferenceSessions
            except Exception:
                pass
        self._yolo = None               # the wrapper held the store + the model lists
        self._names = None  # force a fresh repo glob after unload()
        self._names_offline = False
        if had:
            # Dropping the session is necessary but not sufficient: the freed arena memory stays
            # in glibc's pools unless it is asked for (P8.3, measured: 999 MB peak -> 808 MB on
            # unload alone -> 755 MB after gc + trim). Honest scope: this is the part that is
            # ours to return; what ONNX Runtime keeps in its process-wide state is not, and
            # `unload_engines` does not promise otherwise (tracking.md D62).
            try:
                import gc
                gc.collect()
                from ..core.memory import trim_heap
                trim_heap()
            except Exception:                                  # noqa: BLE001
                pass

    # -- public API ----------------------------------------------------------

    def refresh_models(self) -> list[str]:
        """Fetch (and cache) the repo's model names; fall back to KNOWN_MODELS.

        The fallback keeps the GUI usable when offline or unauthorized; auth
        problems still surface loudly at load()/detect() time. status().detail
        records why we fell back. Cached because `model_names` performs a network
        glob and `detect()` must not repeat it on every call.
        """
        with self._lock:
            if self._names is not None:
                return list(self._names)
            names: Optional[list[str]] = None
            if self._store.offline:
                # Nothing to ask: offline means "bundled weights + documented presets",
                # and the backend is not even constructed (no fsspec, no socket).
                names = list(dict.fromkeys(self._store.bundled_detection_variants()
                                           + list(KNOWN_MODELS)))
                self._names_offline = True
                self._detail = self._detail or "model list: offline (bundled + presets)"
            else:
                try:
                    raw = self._ensure_yolo().model_names
                    if isinstance(raw, (list, tuple)):  # OFFLINE sentinel is neither
                        names = [str(n) for n in raw]
                except Exception as exc:
                    self._detail = f"model list fallback ({type(exc).__name__}: {exc})"
                state = getattr(self._yolo, "listing_state", "ok")
                if state not in ("ok", "unset"):
                    self._names_offline = True   # the repo did not answer authoritatively
                    if not self._detail:
                        self._detail = (f"model list: {state} — answering from the bundle "
                                        f"and the built-in presets")
            if not names:  # None, empty, or imgutils OFFLINE sentinel
                names = list(KNOWN_MODELS)
                self._names_offline = True
                if not self._detail:
                    self._detail = "model list fallback (offline or sentinel)"
            ordered = [n for n in KNOWN_MODELS if n in names]
            ordered += sorted(set(names) - set(KNOWN_MODELS))
            self._names = ordered
            return list(ordered)

    def status(self) -> EngineStatus:
        """The base status, plus whether the variant list is a fallback (P9.2).

        Read from ``_names_offline`` — *not* from the ``model_list_is_fallback`` property below,
        which refreshes first. A status call must never be able to spend ten seconds on a socket:
        the GUI paints a badge from this, the MCP answers `get_status` with it, and a warm-up
        report reads it. What you get is the answer as it stands, which is the honest thing to
        print next to a badge anyway.
        """
        status = super().status()
        if status.model_list_is_fallback == bool(self._names_offline):
            return status
        return replace(status, model_list_is_fallback=bool(self._names_offline))

    @property
    def model_list_is_fallback(self) -> bool:
        """True when the model list came from the bundle/presets, not from the repo.

        This form *asks the repo first* (that is what makes "is it still down?" answerable). The
        passive answer — the one a badge, `get_status` or `--model-status` may read a hundred
        times without spending a second — is on `status().model_list_is_fallback` (P9.2).

        The GUI uses it to grey out "Hub-only" entries (``source == "hf"``).
        """
        self.refresh_models()
        return self._names_offline

    def list_models(self) -> list[dict[str, Any]]:
        names = self.refresh_models()
        bundled = set(self._store.bundled_detection_variants())
        return model_catalog(names, sources={n: ("bundled" if n in bundled else "hf")
                                             for n in names})

    def recommended_threshold(self, model_name: str) -> Optional[float]:
        """Repo `threshold.json` (F1-optimal) when the backend is live; preset otherwise.

        imgutils caches the value after the first read and `load()` already prefetched
        the file, so in practice this is a local read. Never raises: a dead backend or
        unknown model falls back to the documented preset (or None).
        """
        yolo = self._yolo
        getter = getattr(yolo, "_get_default_thresholds", None) if yolo is not None else None
        if callable(getter):
            try:
                value = getter(model_name)
                if value is not None:
                    return float(value)
            except Exception:
                pass
        return super().recommended_threshold(model_name)

    # -- a bundled file that will not open (P9.1) ---------------------------------------

    def _open_failed(self, model_name: str, bundled: Optional[Path], exc: BaseException,
                     progress: Optional[ProgressCallback] = None) -> None:
        """Diagnose a failed ONNX open; retry once via the Hub when *we* shipped the bad bytes.

        "Bundled" is decided by the file's size, so a copy that was interrupted — disk full, half
        a ``git clone``, a truncated USB transfer — reported ``"bundled": true`` and then failed
        inside onnxruntime with a parse error naming neither the bundle nor the fix. The bundle
        carries its own passport (``MANIFEST.json``, sha256 for every vendored file), so the
        honest answer is one hash away — taken here, on the failure path, and never on the way in:
        hashing 77-226 MB to answer a status question would tax every load to serve a rare bug.

        Returns normally only when a Hub copy opened successfully; otherwise raises. Nothing on
        disk is moved or deleted: a corrupt file stays exactly where the user put it.
        """
        base = _classify(exc, f"open session {model_name}")
        if bundled is None or base.code is not ErrorCode.ENGINE_LOAD_FAILED:
            raise base from exc
        from animetext_studio.core.model_store import verify_file
        try:
            check = verify_file(bundled, self._store.root)
        except Exception:                                    # a broken hasher must not hide the fault
            raise base from exc
        details = dict(base.details)
        details.update({
            "path": check["path"], "bundled": True, "size_bytes": check["size_bytes"],
            "expected": check["expected"], "got": check["got"],
            "weights_check": check["state"],
        })
        if check["state"] == "match":
            raise StudioError(
                ErrorCode.ENGINE_LOAD_FAILED,
                f"Cannot open {model_name}: the bundled weights are intact (their sha256 matches "
                "MANIFEST.json), so the file is not the problem — the ONNX runtime is.",
                {**details, "hint": INSTALL_HINT, "steps": [
                    "python -c \"import onnxruntime as o; "
                    "print(o.__version__, o.get_available_providers())\"",
                    "pip install --upgrade onnxruntime      # or onnxruntime-gpu for --device cuda",
                    "python scripts/fetch_models.py --verify",
                ]},
            ) from exc
        if check["state"] == "unrecorded":
            raise StudioError(
                ErrorCode.ENGINE_LOAD_FAILED,
                f"Cannot open the bundled weights for {model_name}, and MANIFEST.json has no "
                "entry for them, so their integrity cannot be confirmed either way.",
                {**details, "steps": [
                    "python scripts/fetch_models.py --verify",
                    f"python scripts/fetch_models.py --variants {model_name}   # re-vendor",
                ]},
            ) from exc
        # `mismatch`, or `absent` between the check and now: the bytes are not the vendored ones.
        recovered = None if self._store.offline else self._retry_via_hub(model_name, progress)
        if recovered is not None:
            self.recovered_via_hub[model_name] = recovered
            self._detail = (
                f"recovered from Hub: bundled {model_name}/model.onnx was corrupt "
                f"(sha256 {str(check['got'])[:12]}…, expected {str(check['expected'])[:12]}…)")
            report(progress, "load_model", 90,
                   f"{model_name}: bundled copy is corrupt — opened the Hub copy instead")
            return
        steps = [
            "python scripts/fetch_models.py --verify",
            f"python scripts/fetch_models.py --variants {model_name}   # re-vendor this variant",
        ]
        if self._store.offline:
            steps.append("offline: re-vendor the bundle, or unset offline mode to fetch this variant")
        raise StudioError(
            ErrorCode.ENGINE_LOAD_FAILED,
            f"Bundled weights for {model_name} are corrupt: the file is not the copy "
            "MANIFEST.json recorded, so it is not a usable variant."
            + ("" if self._store.offline else " (the Hub was tried and could not replace it.)"),
            {**details, "steps": steps},
        ) from exc

    def _retry_via_hub(self, model_name: str,
                       progress: Optional[ProgressCallback] = None) -> Optional[str]:
        """Fetch this variant from the repo and open *that* file. Returns its path, or None.

        The corrupt file is never deleted; the freshly downloaded path is remembered in the
        session's ``override_files`` so later loads on any thread skip the bad copy too.
        """
        from pathlib import Path                      # lazy, as everywhere in this module
        report(progress, "download", 0,
               f"{model_name}: bundled copy is corrupt, fetching from {self._repo_id}")
        try:
            path = _hf_download(self._repo_id, f"{model_name}/model.onnx", token=self._token())
        except Exception:
            return None
        yolo = self._ensure_yolo()
        opener = getattr(yolo, "_open_model", None)
        if not callable(opener):
            return None
        key = (str(model_name), "model.onnx")
        getattr(yolo, "override_files", {}).setdefault(key, Path(path))
        try:
            self._note_session_providers(opener(model_name))
        except Exception:
            getattr(yolo, "override_files", {}).pop(key, None)
            return None
        return str(path)

    def load(self, model_name: str, progress: Optional[ProgressCallback] = None) -> None:
        """Download (with progress) + open the ONNX session. Idempotent per model."""
        with self._lock:
            if model_name in self._loaded:
                report(progress, "load_model", 100, f"{model_name} already loaded")
                return
            self._state = EngineState.LOADING
            try:
                bundled = self._store.detection_file(model_name, "model.onnx")
                if bundled is not None:
                    # On-premises weights: no network, no token, no cache to warm (D26).
                    report(progress, "load_model", 10, f"{model_name}: on-premises weights")
                elif self._store.offline:
                    self._store.require_detection_file(model_name)   # raises MODEL_NOT_FOUND
                else:
                    report(progress, "download", 0, f"{model_name}: model.onnx")
                    tqdm_class = _progress_tqdm(progress) if progress is not None else None
                    try:
                        _hf_download(self._repo_id, f"{model_name}/model.onnx",
                                     token=self._token(), tqdm_class=tqdm_class)
                    except Exception as exc:
                        raise _classify(exc, f"{self._repo_id}/{model_name}/model.onnx")
                    for aux in ("model_type.json", "threshold.json"):
                        try:
                            _hf_download(self._repo_id, f"{model_name}/{aux}", token=self._token())
                        except Exception:
                            pass  # optional metadata; imgutils tolerates absence
                report(progress, "load_session", 90, f"{model_name}: opening ONNX session")
                yolo = self._ensure_yolo()
                opener = getattr(yolo, "_open_model", None)
                if callable(opener):
                    try:
                        self._note_session_providers(opener(model_name))
                    except Exception as exc:
                        # Raises, unless the bundled weights were the problem and the Hub could
                        # replace them (P9.1) — in which case the session is open and we carry on.
                        self._open_failed(model_name, bundled, exc, progress)
                self._loaded.add(model_name)
                self._state = EngineState.READY
                self._detail = ""
                report(progress, "load_model", 100, f"{model_name} ready")
            except Exception as exc:
                self._state = EngineState.ERROR
                raise _classify(exc, f"load {model_name}")

    def detect(
        self,
        image: Image.Image,
        model_name: str,
        *,
        iou_threshold: float = 0.35,
        score_threshold: Optional[float] = 0.05,
        allow_dynamic: bool = False,
        progress: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[Detection]:
        """Run YOLO detection; returns Detection list sorted by score (desc).

        score_threshold=None -> repo-recommended F1-optimal threshold.
        """
        if not isinstance(image, Image.Image):
            raise StudioError(ErrorCode.INVALID_IMAGE, "Expected a PIL.Image.Image.")
        for label, value in (("iou_threshold", iou_threshold), ("score_threshold", score_threshold)):
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise StudioError(ErrorCode.INVALID_ARGUMENT, f"{label} must be in [0, 1], got {value!r}")
        with self._lock:
            check_cancel(cancel_event)
            available = set(self.refresh_models())
            if model_name not in available:
                raise StudioError(ErrorCode.MODEL_NOT_FOUND, f"Unknown model {model_name!r}.",
                                  {"available": sorted(available)})
            if model_name not in self._loaded:
                self.load(model_name, progress=progress)
            check_cancel(cancel_event)
            report(progress, "detect", 0, f"{model_name}: inferring")
            w, h = image.size
            try:
                raw = self._ensure_yolo().predict(
                    image, model_name,
                    conf_threshold=score_threshold,
                    iou_threshold=iou_threshold,
                    allow_dynamic=allow_dynamic,
                )
            except Exception as exc:
                raise _classify(exc, f"predict({model_name})")
            detections: list[Detection] = []
            for box, label, score in raw:
                bbox = BBox.from_tuple(box).clamped(w, h)
                if bbox.is_valid():
                    detections.append(Detection(bbox, str(label), float(score)))
            detections.sort(key=lambda d: (-d.score, d.bbox.y0, d.bbox.x0))
            self._note_inference()
            report(progress, "detect", 100, f"{len(detections)} text block(s)")
            return detections
