#!/usr/bin/env python3
"""Vendor the model weights *into the app* so a folder copy is a complete install.

This is the write side of the on-premises model store (decision D26, see
``animetext_studio/core/model_store.py``). Run it once on a machine with network +
HF access; afterwards the app runs with ``--offline`` forever, and backing up the
project folder backs up the models.

    python scripts/fetch_models.py                    # n + s + m  (all < 100 MB each)
    python scripts/fetch_models.py --all --ocr        # every variant + the OCR snapshot
    python scripts/fetch_models.py --verify           # check what is on disk (no network)
    python scripts/fetch_models.py --dry-run --all    # just show the plan and sizes

Files that are already present with the expected size are never re-downloaded, so it
is safe to interrupt and re-run, and it will not touch a bundle you copied by hand.
The gated detection repo needs a token that accepted its terms (``HF_TOKEN`` /
``--hf-token``); the OCR repo is public.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run from a checkout

from animetext_studio.core.config import KNOWN_MODELS, MODEL_PRESETS   # noqa: E402
from animetext_studio.core.model_store import (                        # noqa: E402
    DETECTION_DIRNAME, MANIFEST_NAME, OCR_DIRNAME, ModelStore, discover_hf_token,
    manifest_entry_path, resolve_manifest_files, resolve_models_dir, sha256_file,
    verify_safetensors,
)
from animetext_studio.core.types import DETECTION_REPO, OCR_REPO       # noqa: E402

#: Variants fetched by default: the app default (m) plus the two smaller ones, all of
#: which stay under the ~100 MB per-file limit of constrained environments. `l`/`x`
#: are 105.8 / 236.7 MB and only come with --all.
DEFAULT_VARIANTS = ("yolo12n_animetext", "yolo12s_animetext", "yolo12m_animetext")
DETECTION_FILES = ("model.onnx", "threshold.json", "labels.json", "model_type.json")
#: Files a variant folder may carry that the app never loads — the PyTorch originals
#: behind the ONNX export (re-exporting and fine-tuning are out of scope here). They are
#: **recorded when present** so a folder backup is fully accounted for by MANIFEST.json
#: and `--verify` notices a truncated copy, but they are never downloaded: the runtime
#: only ever reads the ONNX.
DETECTION_EXTRA_FILES: tuple[str, ...] = ("model.pt",)
#: Files of the OCR snapshot we do not need (repo bookkeeping), besides the big one we
#: already keep for completeness.
OCR_SKIP = (".gitattributes",)


def _fmt(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("kB", 1024)):
        if n >= div:
            return f"{n / div:8.1f} {unit}"
    return f"{n:8d} B"


def _token(args: argparse.Namespace) -> Optional[str]:
    """``--hf-token`` > ``$HF_TOKEN`` > a local ``HF_token.txt`` at the project root.

    The file lookup is the app-wide rule (`model_store.discover_hf_token`); the value is
    only ever passed to the Hub client and is never printed.
    """
    return discover_hf_token(args.hf_token)


def _remote_files(repo_id: str, token: Optional[str]) -> dict[str, dict[str, Any]]:
    """filename -> {size, sha256?} from the Hub (one API call, works for both repos)."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).repo_info(repo_id, files_metadata=True, revision="main")
    out: dict[str, dict[str, Any]] = {}
    for sib in info.siblings or []:
        out[sib.rfilename] = {
            "size": getattr(sib, "size", None),
            # LFS files carry the git oid (sha256); plain files carry a weak sha.
            "sha256": getattr(sib, "lfs", None) and getattr(sib.lfs, "sha256", None),
        }
    return out, info.sha


def _already_there(path: Path, meta: dict[str, Any], manifest: dict[str, Any],
                   root: Optional[Path] = None) -> bool:
    """Present and matching (sha256 if we recorded one, else the remote size)."""
    if not path.is_file():
        return False
    recorded = None
    for entry in manifest.get("files", []):
        if not isinstance(entry, dict):
            continue
        where = manifest_entry_path(entry, root or path.parent, manifest.get("models_root"))
        if where == path:
            recorded = entry
            break
    if recorded and recorded.get("sha256"):
        return recorded["sha256"] == sha256_file(path)
    size = path.stat().st_size
    expected = meta.get("size")
    return expected is not None and size == expected


def _entry(path: Path, repo_id: str, revision: str, with_sha: bool,
           root: Optional[Path] = None) -> dict[str, Any]:
    rel: dict[str, Any] = {}
    if root is not None:
        try:
            rel["relpath"] = path.relative_to(Path(root)).as_posix()
        except ValueError:
            pass
    return {
        **rel,
        "path": str(path),
        "repo_id": repo_id,
        "revision": revision,
        "size": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if (with_sha and path.is_file()) else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="fetch_models.py",
        description="Vendor AnimeText YOLO + Hayai OCR weights into the app (on-premises bundle).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("After fetching:  animetext-studio --model-status   and   "
                "animetext-studio --offline --model yolo12m_animetext ..."),
    )
    ap.add_argument("--models-dir", metavar="PATH", default=None,
                    help="bundle root (default: $ANIMETEXT_MODELS_DIR or <package>/models)")
    ap.add_argument("--variants", nargs="*", metavar="NAME",
                    help=f"detection variants to vendor (default: {' '.join(DEFAULT_VARIANTS)})")
    ap.add_argument("--all", action="store_true", dest="all_variants",
                    help=f"all five variants: {' '.join(KNOWN_MODELS)}")
    ap.add_argument("--no-detection", action="store_true", help="skip the YOLO repo")
    ap.add_argument("--ocr", action="store_true", help="also vendor the OCR snapshot (~667 MB)")
    ap.add_argument("--all-both", action="store_true", help="shorthand for --all --ocr")
    ap.add_argument("--verify", action="store_true", help="check the bundle on disk; no network")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if the file is present")
    ap.add_argument("--with-sha256", action="store_true",
                    help="record sha256 of every vendored file in MANIFEST.json")
    ap.add_argument("--hf-token", metavar="TOKEN", help="token for the gated detection repo")
    args = ap.parse_args(argv)

    if args.all_both:
        args.all_variants = args.ocr = True
    root = resolve_models_dir(args.models_dir)
    store = ModelStore(root)
    manifest_existing = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8")) \
        if (root / MANIFEST_NAME).is_file() else {}

    # ---------------- verify mode (offline) --------------------------------
    if args.verify:
        status = store.status()
        print(f"On-premises bundle: {status['models_dir']}")
        print(f"  exists: {status['exists']}")
        for name in list(KNOWN_MODELS) + [n for n in status["detection_bundled"]
                                          if n not in KNOWN_MODELS]:
            v = status["detection_variants"][name]
            mark = "OK  " if v["bundled"] else "--  "
            aux = "".join(f" +{f.split('.')[0]}" for f, ok in v["files"].items()
                          if ok and f != "model.onnx")
            print(f"  {mark}{name:20}{_fmt(v['bytes'])}{aux}")
        ocr = status["ocr"]
        mark = "OK  " if ocr["bundled"] else "--  "
        print(f"  {mark}{OCR_DIRNAME:20}{_fmt(ocr['bytes'])}  (OCR snapshot"
              f"{', ' + ocr['revision'][:12] if ocr.get('revision') else ''})")
        if ocr["missing"]:
            print(f"      missing: {', '.join(ocr['missing'])}")
        weights = store.ocr_dir() / "model.safetensors"
        if weights.is_file():
            check = verify_safetensors(weights)
            print(f"      safetensors: {'structurally sound' if check['ok'] else 'BROKEN — ' + str(check['error'])}"
                  f" ({check['tensors']} tensors, {check.get('dtype')})")
        # manifest cross-check: every recorded file must exist and hash-match. Resolved
        # through the bundle root, because the recorded absolute path is where the file was
        # *when it was fetched*: a bundle that has been copied elsewhere — which D26 says is a
        # complete installation — used to be reported as "RECORDED BUT ABSENT (33)" and
        # `RESULT: incomplete` while every one of those files loaded perfectly well.
        bad, missing = [], []
        for path, f in resolve_manifest_files(manifest_existing, root):
            if not path.is_file():
                missing.append(str(path))
            elif f.get("sha256") and sha256_file(path) != f["sha256"]:
                bad.append(str(path))
        if bad:
            print(f"  HASH MISMATCH ({len(bad)}): {', '.join(bad)}")
        if missing:
            print(f"  RECORDED BUT ABSENT ({len(missing)}): {', '.join(missing)}")
        elif manifest_existing and not bad:
            print(f"  manifest: {len(manifest_existing.get('files', []))} files, hashes verified")
        if not manifest_existing:
            print("  manifest: none (run without --verify to create it)")
        ok = bool(status["detection_bundled"]) and ocr["bundled"] and not bad and not missing
        print("RESULT:", "usable offline" if ok else "incomplete")
        return 0 if ok else 1

    # ---------------- plan --------------------------------------------------
    variants = list(KNOWN_MODELS if args.all_variants else
                    (args.variants or DEFAULT_VARIANTS))
    unknown = [v for v in variants if v not in KNOWN_MODELS]
    if unknown:
        print(f"note: not in the documented presets: {', '.join(unknown)}", file=sys.stderr)
    plan: list[tuple[str, str, Path]] = []          # (repo, remote filename, local path)
    if not args.no_detection:
        for variant in variants:
            for fname in DETECTION_FILES:
                plan.append((DETECTION_REPO, f"{variant}/{fname}",
                             store.detection_variant_dir(variant) / fname))
    if args.ocr:
        plan.append(("__ocr__", "__snapshot__", store.ocr_dir()))

    print(f"Models root : {root}")
    print(f"Token       : {'yes' if _token(args) else 'none (public repos only)'}")
    for repo, fname, dest in plan:
        print(f"  {repo if repo != '__ocr__' else OCR_REPO:34} {fname:38} -> {dest}")
    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return 0

    # ---------------- download ---------------------------------------------
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 2

    entries: list[dict[str, Any]] = []
    revisions: dict[str, str] = {}
    failures: list[str] = []
    cache: dict[str, tuple[dict[str, dict[str, Any]], str]] = {}

    def remote(repo: str):
        if repo not in cache:
            cache[repo] = _remote_files(repo, _token(args))
        return cache[repo]

    # OCR snapshot: resolve the file list from the repo itself.
    if args.ocr:
        files, rev = remote(OCR_REPO)
        revisions[OCR_REPO] = rev
        print(f"\n{OCR_REPO} @ {rev[:12]}")
        for fname, meta in sorted(files.items()):
            if fname in OCR_SKIP:
                continue
            dest = store.ocr_dir() / fname
            if not args.force and _already_there(dest, meta, manifest_existing, root):
                print(f"  keep   {fname:28}{_fmt(meta['size'])} (already on premises)")
            else:
                print(f"  fetch  {fname:28}{_fmt(meta['size'])}")
                try:
                    hf_hub_download(repo_id=OCR_REPO, filename=fname,
                                    local_dir=str(store.ocr_dir()), token=_token(args))
                except Exception as exc:                                  # noqa: BLE001
                    failures.append(f"{OCR_REPO}/{fname}: {type(exc).__name__}: {exc}")
                    print(f"    FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
            entries.append(_entry(dest, OCR_REPO, rev, args.with_sha256, root))
        origin = store.ocr_dir() / ".origin.json"
        origin.write_text(json.dumps({
            "repo_id": OCR_REPO, "revision": rev, "source": "huggingface.co",
            "license": "apache-2.0",
            "note": "Vendored for offline/on-premises use (decision D26). "
                    "Re-create with: python scripts/fetch_models.py --ocr",
        }, indent=2) + "\n", encoding="utf-8")

    if not args.no_detection:
        files, rev = remote(DETECTION_REPO)
        revisions[DETECTION_REPO] = rev
        print(f"\n{DETECTION_REPO} @ {rev[:12]}  (gated: needs a token that accepted the terms)")
        for repo, fname, dest in plan:
            if repo == "__ocr__":
                continue
            meta = files.get(fname, {})
            if not args.force and _already_there(dest, meta, manifest_existing, root):
                print(f"  keep   {fname:38}{_fmt(meta.get('size'))} (already on premises)")
            elif meta.get("size") is None and fname.rsplit("/", 1)[-1] not in ("model.onnx",):
                continue          # optional metadata this variant does not ship
            else:
                print(f"  fetch  {fname:38}{_fmt(meta.get('size'))}")
                try:
                    hf_hub_download(repo_id=DETECTION_REPO, filename=fname,
                                    local_dir=str(store.detection_root), token=_token(args))
                except Exception as exc:                                  # noqa: BLE001
                    failures.append(f"{DETECTION_REPO}/{fname}: {type(exc).__name__}: {exc}")
                    print(f"    FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue          # never record a file we did not get, optional or not
            entries.append(_entry(dest, DETECTION_REPO, rev, args.with_sha256, root))

        # Extras that happen to be on disk (e.g. a hand-copied model.pt): record their
        # provenance too, so `--verify` hashes them and a folder backup is self-describing.
        planned = {dest for _repo, _fname, dest in plan}
        for variant in variants:
            for fname in DETECTION_EXTRA_FILES:
                extra = store.detection_variant_dir(variant) / fname
                if extra.is_file() and extra not in planned:
                    print(f"  record {str(extra.relative_to(store.detection_root)):38}"
                          f"{_fmt(extra.stat().st_size)} (not used at runtime)")
                    entries.append(_entry(extra, DETECTION_REPO, rev, args.with_sha256, root))

    # ---------------- manifest + summary -----------------------------------
    # The report describes the *bundle on disk*, not just this run: a partial fetch
    # (--ocr, or one variant) must not rewrite history for the rest of it.
    from animetext_studio.core.model_store import write_manifest
    all_revisions = {**(manifest_existing.get("revisions") or {}), **revisions}
    write_manifest(root, entries, extra={
        "revisions": all_revisions,
        "detection_variants": store.bundled_detection_variants(),
        "ocr": store.has_ocr_snapshot(),
        "usage": "animetext-studio --model-status | --offline",
    })
    # hf_hub_download(local_dir=...) leaves a .cache/huggingface tree (metadata, lock
    # files, and `.incomplete` blobs from interrupted runs) inside the target directory,
    # and importing the OCR repo's remote code drops __pycache__ next to it. Neither is
    # part of the model, and the bundle is meant to be copied/backed up as-is.
    junk = [p for p in root.rglob("*")
            if p.is_dir() and p.name in (".cache", "__pycache__")]
    if junk:
        import shutil
        for p in junk:
            shutil.rmtree(p, ignore_errors=True)
        print(f"\ncleaned {len(junk)} cache dir(s) from the bundle")

    total = sum(e["size"] for e in entries)
    print(f"\nBundle: {root}  ({len(entries)} file(s) this run, {total / 1024 ** 2:.1f} MB"
          f"; {store.status()['total_bytes'] / 1024 ** 2:.1f} MB on premises)")
    print(f"Manifest: {root / MANIFEST_NAME}")
    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 7
    print("Next: python scripts/fetch_models.py --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
