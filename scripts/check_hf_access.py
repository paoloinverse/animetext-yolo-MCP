#!/usr/bin/env python3
"""Hugging Face access preflight for AnimeText OCR Studio.

Checks — in order, each with an actionable message — that the *gated* detection repo
`deepghs/AnimeText_yolo` is actually reachable with the token you have, before you
wait for a 100 MB download to fail.

    python scripts/check_hf_access.py                 # env token (HF_TOKEN)
    python scripts/check_hf_access.py --token hf_xxx  # explicit token
    python scripts/check_hf_access.py --full          # also fetch the ONNX weights
    python scripts/check_hf_access.py --detect examples/sample_page.png   # end to end
    python scripts/check_hf_access.py --json          # machine-readable report

Exit codes follow the app's error taxonomy (core.types.EXIT_CODES):
0 ok · 6 HF_AUTH_REQUIRED · 7 DOWNLOAD_FAILED/network · 8 ENGINE_LOAD_FAILED · 12 MODEL_NOT_FOUND

The token is never written to disk and never echoed (only a masked fingerprint).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

REPO = "deepghs/AnimeText_yolo"
OCR_REPO = "JustANormalTinkerer/hayai-ocr-v2.7-supernova"
REPO_URL = f"https://huggingface.co/{REPO}"
TOKEN_PAGE = "https://huggingface.co/settings/tokens"
TERMS_STEPS = (
    f"1. Open {REPO_URL} while logged in and accept the access request "
    "(the repo is 'gated: auto', so this is normally instant).",
    f"2. Create a **Read** token at {TOKEN_PAGE} (fine-grained tokens must list "
    f"'{REPO}' with Read access).",
    "3. export HF_TOKEN=hf_...   (Windows PowerShell: $env:HF_TOKEN='hf_...')",
    "   or pass --hf-token / --token on the command line.",
    "4. Re-run this script. Note: `huggingface-cli login` also works, but HF_TOKEN "
    "takes precedence in this app.",
)


def mask(token: Optional[str]) -> str:
    if not token:
        return "<not set>"
    if len(token) <= 8:
        return f"{'*' * len(token)} (len={len(token)})"
    return f"{token[:6]}\u2026{token[-4:]} (len={len(token)})"


def resolve_token(cli_token: Optional[str]) -> tuple[Optional[str], str]:
    """--token wins, then HF_TOKEN / HUGGING_FACE_HUB_TOKEN (the app's own order).

    A `huggingface-cli login` credential is used implicitly by huggingface_hub when
    nothing else is set; that case shows up as "anonymous" here but may still work.
    """
    if cli_token:
        return cli_token, "--token"
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var], f"env:{var}"
    return None, "none"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", help="HF token (else HF_TOKEN / HUGGING_FACE_HUB_TOKEN env)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--model", default="yolo12n_animetext",
                    help="variant used for the probe files (default: smallest, yolo12n_animetext)")
    ap.add_argument("--full", action="store_true", help="also download model.onnx (tens of MB)")
    ap.add_argument("--detect", metavar="IMAGE", help="run one real detection on IMAGE and print the box count")
    ap.add_argument("--hf-cache-dir", metavar="PATH", help="use a specific HF cache directory")
    ap.add_argument("--json", action="store_true", help="print a JSON report instead of prose")
    args = ap.parse_args(argv)

    report: dict[str, Any] = {"repo": args.repo, "model": args.model, "checks": []}
    exit_code = 0

    def check(name: str, ok: bool, detail: str = "", hint: str = "", fatal: int = 0) -> bool:
        """Record one check; the first fatal failure decides the exit code."""
        report["checks"].append({"name": name, "ok": ok, "detail": detail, "hint": hint})
        if not ok and fatal:
            report.setdefault("fatal", fatal)
        return ok

    token, source = resolve_token(args.token)
    report["token"] = {"source": source, "fingerprint": mask(token)}

    # 1. dependencies -------------------------------------------------------
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                            HfHubHTTPError, RepositoryNotFoundError)
        check("dependencies", True, "huggingface_hub importable")
    except ImportError as exc:
        check("dependencies", False, str(exc),
              "pip install 'huggingface_hub>=0.26'", fatal=8)
        print(json.dumps(report, indent=2) if args.json else _prose(report))
        return report.get("fatal", 8)

    if args.hf_cache_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from animetext_studio.engines.detection import configure_hf_cache_dir
        configure_hf_cache_dir(args.hf_cache_dir)
        report["hf_cache_dir"] = os.environ.get("HF_HUB_CACHE")

    # 2. who am I -----------------------------------------------------------
    identity = "anonymous"
    try:
        me = HfApi(token=token).whoami()
        identity = me.get("name") or me.get("fullname") or "unknown"
        check("token_valid", True, f"authenticated as {identity} ({mask(token)} via {source})")
    except Exception as exc:
        name = type(exc).__name__
        if token is None:
            check("token_valid", False, f"no token provided ({name})",
                  "\n     " + "\n     ".join(TERMS_STEPS), fatal=6)
        else:
            check("token_valid", False, f"{name}: {exc}",
                  "The token is invalid, expired, or network access to huggingface.co is blocked.",
                  fatal=6)

    # 3. repo visibility ----------------------------------------------------
    try:
        info = HfApi(token=token).model_info(args.repo, files_metadata=False)
        check("repo_visible", True,
              f"repo found; gated={getattr(info, 'gated', '?')}, "
              f"access_requested={getattr(info, 'access_request_status', None)}")
    except RepositoryNotFoundError:
        check("repo_visible", False, "repository not found (404 without access)",
              "For gated repos a 404 usually means 'no access'.\n     "
              + "\n     ".join(TERMS_STEPS), fatal=6)
    except Exception as exc:
        check("repo_visible", False, f"{type(exc).__name__}: {exc}",
              "Check connectivity to huggingface.co (proxy/VPN?).", fatal=7)

    # 4. the tiny probe file (proves download rights, not just listing) ----
    probe_ok = False
    try:
        path = hf_hub_download(repo_id=args.repo, repo_type="model",
                               filename=f"{args.model}/threshold.json", token=token)
        probe_ok = True
        content = Path(path).read_text(encoding="utf-8").strip()
        check("probe_download", True, f"{args.model}/threshold.json -> {content}")
    except GatedRepoError as exc:
        check("probe_download", False, f"GatedRepoError: {str(exc).splitlines()[0]}",
              "Terms not accepted by this token's account, or the token lacks Read "
              "access to this repo.\n     " + "\n     ".join(TERMS_STEPS), fatal=6)
    except EntryNotFoundError as exc:
        check("probe_download", False, f"EntryNotFoundError: {exc}",
              f"Variant {args.model!r} not in the repo? Try --model yolo12n_animetext.",
              fatal=12)
    except Exception as exc:
        check("probe_download", False, f"{type(exc).__name__}: {exc}",
              "Network/storage problem.", fatal=7)

    # 5. optional: real weights + one real inference ------------------------
    onnx = None
    if args.full or args.detect:
        try:
            onnx = hf_hub_download(repo_id=args.repo, repo_type="model",
                                   filename=f"{args.model}/model.onnx", token=token)
            size_mb = Path(onnx).stat().st_size / 1e6
            check("weights_download", True, f"{args.model}/model.onnx ({size_mb:.1f} MB) at {onnx}")
        except Exception as exc:
            check("weights_download", False, f"{type(exc).__name__}: {exc}",
                  "Could not fetch the ONNX weights.", fatal=7)
            onnx = None
    if args.detect and onnx is None:
        check("real_detection", False, "skipped: the ONNX weights are unavailable",
              "Resolve the download/auth failure above first.")
    elif args.detect and onnx:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from PIL import Image
            from animetext_studio.engines.detection import DetectionEngine
            image = Image.open(args.detect).convert("RGB")
            engine = DetectionEngine(hf_token=token)
            dets = engine.detect(image, args.model, iou_threshold=0.35, score_threshold=0.05)
            report["detection"] = {
                "image": str(args.detect), "size": list(image.size), "count": len(dets),
                "top": [{"bbox": d.bbox.to_dict(), "score": round(d.score, 3)}
                        for d in dets[:5]],
            }
            check("real_detection", True, f"{len(dets)} text block(s) detected")
        except Exception as exc:
            check("real_detection", False, f"{type(exc).__name__}: {exc}",
                  "pip install 'animetext-studio[detect]' (dghs-imgutils + onnxruntime).",
                  fatal=8)

    # 6. OCR repo (public, no token needed) ---------------------------------
    try:
        ocr_info = HfApi(token=token).model_info(OCR_REPO)
        check("ocr_repo_visible", True,
              f"{OCR_REPO} is public (gated={getattr(ocr_info, 'gated', False)})")
    except Exception as exc:
        check("ocr_repo_visible", False, f"{type(exc).__name__}: {exc}",
              "The OCR repo is public; a failure here means no network access.", fatal=7)

    report["ok"] = any(c["name"] == "probe_download" and c["ok"] for c in report["checks"])
    exit_code = report.get("fatal", 0)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else _prose(report))
    return exit_code


def _prose(report: dict[str, Any]) -> str:
    lines = [f"Hugging Face access report — {report['repo']}",
             f"token: {report['token']['fingerprint']} (source: {report['token']['source']})", ""]
    for c in report["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
        if not c["ok"] and c["hint"]:
            lines.append(f"         -> {c['hint']}")
    if report.get("detection"):
        d = report["detection"]
        lines.append(f"\nReal detection on {d['image']} ({d['size'][0]}x{d['size'][1]}): "
                     f"{d['count']} block(s)")
        for b in d["top"]:
            lines.append(f"    {b['bbox']}  score={b['score']}")
    lines.append("")
    lines.append("RESULT: ready" if report["ok"] else
                 "RESULT: not ready — fix the FAIL items above (see steps).")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
