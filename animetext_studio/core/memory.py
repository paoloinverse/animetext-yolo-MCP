"""Resident-memory readings (P8.3): the smallest thing that can prove a release.

`unload_engines` claims to free memory, and "it returned without an error" is not evidence.
What is evidence is the process's current RSS before and after — so this module exists, and
nothing else about it matters. `ru_maxrss` is deliberately *not* used: it is the high-water
mark and never goes down, which would make every drop test pass by accident on the second run.

Linux only for the /proc path; `psutil` is used when it happens to be installed; anywhere else
the reading is `None` and callers must skip the assertion rather than fail it.
"""
from __future__ import annotations

import os
from typing import Optional

__all__ = ["process_rss_mb", "rss_available", "format_rss_mb", "trim_heap", "live_sessions"]


def process_rss_mb() -> Optional[float]:
    """Current resident set size of this process in MiB, or None if unknowable here.

    Reads `/proc/self/statm` (one small file, no subprocess, no dependency): field 2 is RSS in
    pages. Falls back to `psutil` for platforms without /proc, and returns None when neither is
    available — a missing reading is reported as such, never as 0.
    """
    try:
        with open("/proc/self/statm", "rb") as handle:
            fields = handle.read().split()
        pages = int(fields[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 * 1024)
    except (OSError, ValueError, IndexError):                # noqa: PERF203
        pass
    try:
        import psutil                                         # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:                                         # noqa: BLE001
        return None


def rss_available() -> bool:
    """True when `process_rss_mb()` can be trusted on this machine (tests gate on it)."""
    return process_rss_mb() is not None


def format_rss_mb(value: Optional[float]) -> str:
    return "rss=?" if value is None else f"rss={value:.0f} MB"


def trim_heap() -> bool:
    """Ask the C allocator to hand freed heap pages back to the OS. True if it ran.

    Why this is here: unloading an engine drops every reference (verified — zero live ONNX
    sessions afterwards), and *still* most of the peak stays in RSS, because freed arena memory
    sits in glibc's pools rather than going back. On the sample page with `yolo12m_animetext`
    (CPU provider): 999 MB peak -> 808 MB after unload -> 755 MB after this call. It is not a
    fix for the floor ONNX Runtime keeps (see `tracking.md` D62), it is just the part that is
    actually ours to give back. No-op and False on non-glibc platforms.
    """
    import ctypes

    for name in ("libc.so.6", "libc.dylib"):
        try:
            libc = ctypes.CDLL(name)
        except OSError:
            continue
        trim = getattr(libc, "malloc_trim", None)      # musl has no malloc_trim
        if trim is None:
            return False
        try:
            trim(0)
            return True
        except Exception:                              # noqa: BLE001
            return False
    return False


def live_sessions() -> Optional[int]:
    """How many `onnxruntime.InferenceSession` objects are alive, or None if ORT is not loaded.

    The honest test for "did unload work": RSS is a blunt instrument (see `trim_heap`), but a
    live session object means the weights are definitely still held. Uses gc, so it sees
    references nobody else has to expose.
    """
    try:
        import onnxruntime as ort
    except Exception:                                  # noqa: BLE001
        return None
    try:
        import gc
        return sum(1 for obj in gc.get_objects() if isinstance(obj, ort.InferenceSession))
    except Exception:                                  # noqa: BLE001
        return None
