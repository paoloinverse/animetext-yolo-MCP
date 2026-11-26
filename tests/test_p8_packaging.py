"""P8.7 — the README and the packaging metadata, pinned.

Documentation rots the way code rots, so it gets the same treatment: every statement in
`README.md` that also exists somewhere machine-readable — the argument parser, the config
schema, the MCP catalog, `EXIT_CODES`, `pyproject.toml`, the presets table — is compared with
that source here. A reference table nobody regenerates is a reference table that is already
wrong, and this project has been bitten by exactly that (Turn 16 found `docs/MCP.md` stale).

The second half is decision D26 as an executable claim — *a copy of the folder is a complete
offline installation*. That claim is about two things only: the `package-data` globs have to
capture every file the runtime reads, and the paths in `MANIFEST.json` have to survive the
copy. Reviewing the packaging for 8.7 found one failure of each kind.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from animetext_studio import __version__
from animetext_studio.core.cli import build_parser
from animetext_studio.core.config import MODEL_PRESETS, AppConfig
from animetext_studio.core.model_store import (
    MANIFEST_NAME, manifest_entry_path, resolve_manifest_files, write_manifest,
)
from animetext_studio.core.types import EXIT_CODES, ErrorCode

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE = ROOT / "animetext_studio"
CATALOG = PACKAGE / "schemas" / "mcp-tools.json"

#: D95: the bundle is fetched, not published — so a clone has none of it, and every check that
#: reads a bundle file must say which of the two copies it is talking about. The declaration in
#: `pyproject.toml` holds in both; only a built folder can be walked file by file.
BUNDLE_ABSENT = ("no bundle in this copy — a clone builds it with "
                 "`python scripts/fetch_models.py` (D95)")


def bundle_present() -> bool:
    return (PACKAGE / "models" / MANIFEST_NAME).is_file()


@pytest.fixture(scope="module")
def readme() -> str:
    assert README.is_file(), "README.md is a P8.7 deliverable; it must exist"
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _leaf_keys(node: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out += _leaf_keys(value, f"{path}.")
        else:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# The README against the CLI
# ---------------------------------------------------------------------------

class TestReadmeDocumentsTheCli:
    def test_every_long_flag_appears(self, readme: str) -> None:
        """A flag that works but is not written down is a flag nobody uses (D57 learned this
        from the opposite direction: the flag was written down and did not work)."""
        missing = [option for action in build_parser()._actions          # noqa: SLF001
                   for option in action.option_strings
                   if option.startswith("--") and option not in readme]
        assert missing == [], f"undocumented CLI flags: {missing}"

    def test_the_exit_code_table_matches_the_taxonomy(self, readme: str) -> None:
        # Matched by shape (`| CODE | number |`), not by scanning every table row: the
        # troubleshooting table names codes in prose too, and a doc check that only passes
        # because of where the words happen to sit fails on the next rewording.
        rows = dict(re.findall(r"^\|\s*`([A-Z_]+)`\s*\|\s*`?(\d+)`?\s*\|", readme, re.MULTILINE))
        for code, number in EXIT_CODES.items():
            assert code.value in rows, f"{code.value} has no row in the exit-code table"
            assert rows[code.value] == str(number), (
                f"{code.value}: README says {rows[code.value]}, EXIT_CODES says {number}")

    def test_every_error_code_is_answered_somewhere(self, readme: str) -> None:
        """Either the README names the code, or `docs/ERROR-MATRIX.md` does — the pair has to
        cover the taxonomy, not one document alone."""
        matrix = (ROOT / "docs" / "ERROR-MATRIX.md").read_text(encoding="utf-8")
        unexplained = [code.value for code in ErrorCode
                       if code.value not in readme and code.value not in matrix]
        assert unexplained == []


# ---------------------------------------------------------------------------
# The README against the rest of the product
# ---------------------------------------------------------------------------

class TestReadmeDocumentsTheProduct:
    def test_every_mcp_tool_is_named(self, readme: str) -> None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        missing = [tool["name"] for tool in catalog["tools"] if f"`{tool['name']}`" not in readme]
        assert missing == [], f"MCP tools the README never mentions: {missing}"

    def test_every_config_key_is_documented(self, readme: str) -> None:
        missing = [path for path in _leaf_keys(AppConfig.default().to_dict())
                   if f"`{path}`" not in readme]
        assert missing == [], f"config keys missing from the README reference: {missing}"

    def test_the_model_table_matches_the_presets(self, readme: str) -> None:
        for name, preset in MODEL_PRESETS.items():
            assert f"`{name}`" in readme, f"{name} is not in the README"
            assert str(preset["f1"]) in readme, f"{name}'s F1 ({preset['f1']}) is stale"
            assert str(preset["recommended_score_threshold"]) in readme

    def test_relative_links_point_at_files_that_exist(self, readme: str) -> None:
        for target in re.findall(r"\]\((?!https?:)([^)#]+)", readme):
            assert (ROOT / target).exists(), f"README links to a missing file: {target}"

    def test_the_version_is_stated(self, readme: str) -> None:
        assert __version__ in readme


# ---------------------------------------------------------------------------
# package-data: the wheel has to contain what the runtime reads (D26)
# ---------------------------------------------------------------------------

def _shipped_data_files(patterns: list[str]) -> set[str]:
    """What setuptools would ship, using its own glob semantics (`glob.glob`, so `*` does not
    match a leading dot — which is exactly how `.origin.json` went missing)."""
    shipped: set[str] = set()
    for pattern in patterns:
        for hit in glob.glob(str(PACKAGE / pattern), recursive=True):
            if os.path.isfile(hit):
                shipped.add(str(Path(hit).relative_to(PACKAGE)))
    return shipped


class TestPackageData:
    def test_every_data_file_on_disk_ships(self, pyproject: dict) -> None:
        patterns = pyproject["tool"]["setuptools"]["package-data"]["animetext_studio"]
        shipped = _shipped_data_files(patterns)
        on_disk = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix not in (".py", ".pyc")}
        assert sorted(on_disk - shipped) == [], (
            "data files that a built wheel would silently drop")

    def test_the_ocr_provenance_file_ships(self, pyproject: dict) -> None:
        """`models/*/*.json` does not match `.origin.json` — stdlib glob skips dotfiles — and
        that file is where the OCR snapshot's revision is recorded for an installed wheel."""
        patterns = pyproject["tool"]["setuptools"]["package-data"]["animetext_studio"]
        assert "models/*/.origin.json" in patterns, (
            "the dot-file is invisible to every other pattern here; without this line a wheel "
            "loses the Hayai revision and `--model-status` silently falls back")
        if not bundle_present():
            pytest.skip(BUNDLE_ABSENT)     # the declaration holds; only a bundle can expand it
        shipped = _shipped_data_files(patterns)
        assert "models/hayai-ocr-v2.7-supernova/.origin.json" in shipped

    def test_the_schemas_and_the_manifest_ship(self, pyproject: dict) -> None:
        patterns = pyproject["tool"]["setuptools"]["package-data"]["animetext_studio"]
        # The schemas are in the repository; the manifest is written by the fetch that makes
        # the bundle, so in a clone only the declaration can be checked (D95).
        assert f"models/{MANIFEST_NAME}" in patterns, MANIFEST_NAME
        shipped = _shipped_data_files(patterns)
        for name in ("schemas/result.schema.json", "schemas/config.schema.json",
                     "schemas/mcp-tools.json"):
            assert name in shipped, name
        if not bundle_present():
            pytest.skip(BUNDLE_ABSENT)
        assert f"models/{MANIFEST_NAME}" in shipped, MANIFEST_NAME


# ---------------------------------------------------------------------------
# MANIFEST.json: the passport has to survive the copy it describes (D26)
# ---------------------------------------------------------------------------

class TestManifestPortability:
    """The bundle used to record absolute paths only, so `fetch_models.py --verify` called a
    relocated — but complete — bundle "incomplete" on every machine except the one that
    vendored it. The fix is `relpath` plus re-basing a legacy absolute path onto the recorded
    `models_root`; both halves are pinned here, on a bundle small enough to move."""

    @staticmethod
    def _bundle(root: Path) -> Path:
        (root / "animetext-yolo" / "yolo12n_animetext").mkdir(parents=True)
        (root / "animetext-yolo" / "yolo12n_animetext" / "model.onnx").write_bytes(b"onnx")
        (root / "animetext-yolo" / "yolo12n_animetext" / "threshold.json").write_text("{}")
        return root

    def test_new_entries_record_a_relpath(self, tmp_path: Path) -> None:
        root = self._bundle(tmp_path / "bundle")
        onnx = root / "animetext-yolo" / "yolo12n_animetext" / "model.onnx"
        manifest = write_manifest(root, [{"path": str(onnx), "size": 4, "sha256": None}])
        assert manifest["files"][0]["relpath"] == "animetext-yolo/yolo12n_animetext/model.onnx"
        # the absolute path is still recorded — it is the provenance of where it came from
        assert Path(manifest["files"][0]["path"]).is_absolute()

    def test_a_moved_bundle_still_resolves(self, tmp_path: Path) -> None:
        root = self._bundle(tmp_path / "bundle")
        onnx = root / "animetext-yolo" / "yolo12n_animetext" / "model.onnx"
        write_manifest(root, [{"path": str(onnx), "size": 4, "sha256": None}])
        target = tmp_path / "installed" / "models"
        shutil.move(str(root), str(target))
        manifest = json.loads((target / MANIFEST_NAME).read_text())
        # `models_root` still names the machine that wrote it; relpath is what saves us
        resolved = resolve_manifest_files(manifest, target)
        assert [p.relative_to(target) for p, _ in resolved] == [
            Path("animetext-yolo/yolo12n_animetext/model.onnx")]
        assert resolved[0][0].is_file()

    def test_a_legacy_absolute_manifest_survives_a_move(self, tmp_path: Path) -> None:
        """The format shipped before this fix: absolute `path`, no `relpath`. Reading one after
        the move must re-base it, not report the file missing."""
        root = self._bundle(tmp_path / "vendored")
        onnx = root / "animetext-yolo" / "yolo12n_animetext" / "model.onnx"
        legacy = {"models_root": str(root), "files": [{"path": str(onnx), "size": 4}]}
        target = tmp_path / "installed" / "models"
        shutil.move(str(root), str(target))
        (target / MANIFEST_NAME).write_text(json.dumps(legacy))

        resolved = resolve_manifest_files(legacy, target)
        assert len(resolved) == 1
        assert resolved[0][0] == target / "animetext-yolo" / "yolo12n_animetext" / "model.onnx"
        assert resolved[0][0].is_file(), "the re-based path must be the real file"

    def test_a_genuinely_missing_file_is_still_missing(self, tmp_path: Path) -> None:
        root = self._bundle(tmp_path / "bundle")
        legacy = {"models_root": str(root),
                  "files": [{"path": str(root / "nope.onnx"), "size": 4}]}
        resolved = resolve_manifest_files(legacy, tmp_path / "somewhere-else")
        assert resolved and not resolved[0][0].is_file()

    def test_merging_after_a_move_does_not_duplicate(self, tmp_path: Path) -> None:
        root = self._bundle(tmp_path / "bundle")
        onnx = root / "animetext-yolo" / "yolo12n_animetext" / "model.onnx"
        write_manifest(root, [{"path": str(onnx), "size": 4, "sha256": None}])
        target = tmp_path / "installed" / "models"
        shutil.move(str(root), str(target))
        merged = write_manifest(target, [], merge=True)
        rels = [entry["relpath"] for entry in merged["files"]]
        assert rels == ["animetext-yolo/yolo12n_animetext/model.onnx"], (
            "a merge keyed on the old absolute path re-adds the same file twice")


# ---------------------------------------------------------------------------
# pyproject metadata
# ---------------------------------------------------------------------------

class TestPyprojectMetadata:
    def test_version_matches_the_package(self, pyproject: dict) -> None:
        assert pyproject["project"]["version"] == __version__

    def test_the_readme_field_points_at_the_readme(self, pyproject: dict) -> None:
        assert pyproject["project"]["readme"] == "README.md"
        assert (ROOT / "README.md").is_file()

    def test_all_extra_is_the_union_and_cannot_drift(self, pyproject: dict) -> None:
        extras = pyproject["project"]["optional-dependencies"]
        assert extras["all"] == ["animetext-studio[gui,detect,ocr,mcp]"]

    def test_the_mcp_floor_is_the_major_version_the_code_imports(self, pyproject: dict) -> None:
        """`mcp/server.py` imports `mcp.server.mcpserver`, which is the 2.x module path; a
        `>=1.9` floor let pip install a version that fails at runtime with ENGINE_LOAD_FAILED."""
        requirement = next(r for r in pyproject["project"]["optional-dependencies"]["mcp"]
                           if r.startswith("mcp"))
        floor = re.match(r"mcp>=(\d+)\.(\d+)", requirement)
        assert floor and (int(floor.group(1)), int(floor.group(2))) >= (2, 3), requirement

    def test_the_metadata_names_the_repository_and_agrees_with_the_readme(self,
                                                                        pyproject: dict,
                                                                        readme: str) -> None:
        """D95 put the project in a repository; a wheel that cannot say where it came from is
        the provenance question again, three months later, asked by someone with no way to ask."""
        # Parsed rather than pattern-matched: the offline-suite rule forbids a URL literal in a
        # test file, and the subject of this test *is* a URL — so the URL stays in
        # `pyproject.toml`, where it belongs, and this reads it.
        from urllib.parse import urlsplit
        urls = pyproject["project"].get("urls", {})
        for key in ("Homepage", "Repository", "Issues"):
            assert key in urls, f"[project.urls] has no {key}"
        home, repo = urlsplit(urls["Homepage"]), urlsplit(urls["Repository"])
        assert (home.scheme, home.hostname) == ("https", "github.com"), \
            "the project should say where it lives, over https, on the host it lives on"
        assert repo.path.removesuffix(".git") == home.path, "Homepage and Repository disagree"
        slug = home.path.strip("/")
        assert slug.count("/") == 1, f"not an owner/name: {slug!r}"
        assert "animetext-yolo-MCP" in slug, "the repository URL is not this project's"
        assert slug in readme, "README never names the repository the metadata names"

    def test_the_console_script_target_exists(self, pyproject: dict) -> None:
        import importlib
        target = pyproject["project"]["scripts"]["animetext-studio"]
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute))


# ---------------------------------------------------------------------------
# Licence — what ships, under what terms.
#
# The distributor chose Apache-2.0 for the application code on 2026-11-24. Until then the
# metadata carried no claim at all, which was the right default *until* it was answered (the
# wrong answer here is worse than none: it either overpromises rights over GPL weights or
# silently imposes obligations nobody intended). Now that there is an answer, it is pinned:
# the terms have to be the real Apache-2.0 text, the packaging has to declare it, the wheel
# has to carry it, and the weights have to keep their own licence beside them.
# ---------------------------------------------------------------------------

LICENSE_FILE = ROOT / "LICENSE"
NOTICE_FILE = ROOT / "NOTICE"
GPL_FOR_WEIGHTS = PACKAGE / "models" / "animetext-yolo" / "LICENSE.GPL-3.0.txt"

#: sha256 of everything in `LICENSE` *before* the "APPENDIX: How to apply the Apache License"
#: heading — the terms themselves. Pinning the body rather than the file lets the copyright
#: line in the appendix be personalised without weakening the one claim that matters: the
#: licence text is canonical, not an edited approximation of it.
APACHE_BODY_SHA256 = "809fa1ed21450f59827d1e9aec720bbc4b687434fa22283c6cb5dd82a47ab9c0"
APACHE_APPENDIX = "   APPENDIX: How to apply the Apache License to your work."
APACHE_SECTIONS = (
    "1. Definitions.",
    "2. Grant of Copyright License",
    "3. Grant of Patent License",
    "4. Redistribution",
    "5. Submission of Contributions",
    "6. Trademarks",
    "7. Disclaimer of Warranty",
    "8. Limitation of Liability",
    "9. Accepting Warranty or Additional Liability",
)

#: sha256 of the whole `LICENSE.GPL-3.0.txt`; the GPL text has nothing to personalise, so the
#: file may be hashed as-is (it is the gnu.org text, byte for byte).
GPL_TEXT_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"

#: Runtime dependencies named in NOTICE, each with a licence *family* that must appear in the
#: licence that the distribution reports about itself. Written from memory, NOTICE would drift
#: the first time a dependency relicensed; this way it drifts into a failing test instead.
#: (It already earned its keep: `dghs-imgutils` is MIT, not the Apache-2.0 memory said it was.)
NOTICE_DEPS = {
    "PySide6-Essentials": "LGPL",
    "shiboken6": "LGPL",
    "dghs-imgutils": "MIT",
    "hfutils": "Apache",
    "onnxruntime": "MIT",
    "torch": "BSD",
    "transformers": "Apache",
    "huggingface_hub": "Apache",
    "accelerate": "Apache",
    "pillow": "MIT",
    "numpy": "BSD",
    "mcp": "MIT",
    "uvicorn": "BSD",
}


def _notice_claims() -> dict[str, str]:
    """`name .... licence` lines from NOTICE, one entry per (comma-separated) name."""
    claims: dict[str, str] = {}
    for line in NOTICE_FILE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s{2}(.+?)\s*\.{2,}\s*(.+)$", line)
        if match:
            for name in match.group(1).split(","):
                claims[name.strip().strip("`").lower()] = match.group(2).strip()
    return claims


class TestLicence:
    def test_the_code_is_apache_2_0(self, pyproject: dict) -> None:
        assert pyproject["project"]["license"] == "Apache-2.0"
        assert "LICENSE" in pyproject["project"]["license-files"]
        assert "NOTICE" in pyproject["project"]["license-files"]
        assert LICENSE_FILE.is_file() and NOTICE_FILE.is_file()

    def test_the_apache_text_is_the_canonical_one(self) -> None:
        text = LICENSE_FILE.read_text(encoding="utf-8")
        body = text.split(APACHE_APPENDIX)[0]
        assert APACHE_APPENDIX in text, "the appendix is what makes the copyright notice apply"
        assert hashlib.sha256(body.encode("utf-8")).hexdigest() == APACHE_BODY_SHA256, (
            "the licence body differs from the Apache-2.0 text — an edited licence is a "
            "different licence, and not the one pyproject declares")
        for heading in APACHE_SECTIONS:
            assert heading in text, f"section missing from LICENSE: {heading}"
        assert "END OF TERMS AND CONDITIONS" in text

    def test_the_copyright_line_is_filled_in(self) -> None:
        text = LICENSE_FILE.read_text(encoding="utf-8")
        assert "[yyyy]" not in text and "[name of copyright owner]" not in text, (
            "the appendix placeholders were never replaced — that is a licence with no owner")
        assert re.search(r"^   Copyright \d{4} \S.*", text, re.M), (
            "expected a filled-in 'Copyright <year> <holder>' line in the appendix")

    def test_the_classifier_list_makes_no_second_licence_claim(self, pyproject: dict) -> None:
        """PEP 639: `license` is an SPDX expression and supersedes the `License ::` trove
        classifiers, which setuptools >= 77 deprecates. Two claims, one truth."""
        assert not any(c.startswith("License ::")
                       for c in pyproject["project"].get("classifiers", []))

    def test_a_built_wheel_declares_and_carries_the_licence(self, tmp_path, monkeypatch) -> None:
        """Not just a string in a file: the wheel an installer reads must carry the licence text.

        Runs setuptools' own `dist_info` step against a temporary metadata directory. Only
        metadata is produced — no data files are copied, so the 651 MB safetensors is never
        touched and the sandbox's 100 MiB RLIMIT (which makes a real wheel build impossible
        here, see the 8.7 notes) does not apply.
        """
        build_meta = pytest.importorskip("setuptools.build_meta")
        monkeypatch.chdir(ROOT)
        produced = build_meta.prepare_metadata_for_build_wheel(str(tmp_path))
        metadata = (tmp_path / produced / "METADATA").read_text(encoding="utf-8")
        assert "License-Expression: Apache-2.0" in metadata, metadata[:400]
        licenses_dir = tmp_path / produced / "licenses"
        shipped = {p.name for p in licenses_dir.iterdir()} if licenses_dir.is_dir() else set()
        assert {"LICENSE", "NOTICE"} <= shipped, (
            f"the wheel ships {sorted(shipped)}; a declared licence whose text is absent is "
            "the failure mode PEP 639's `license-files` exists to prevent")

    def test_the_gpl_weights_ship_their_own_text(self, pyproject: dict) -> None:
        """Bundling GPL-3.0 weights means shipping the GPL text with them — `LICENSE` at the
        root says nothing about them and cannot substitute for it.

        Which is an obligation of whoever *ships* the weights. A clone ships none of them
        (D95), so there is nothing for the text to be beside — but the declaration is checked
        either way, because the wheel built from a fetched bundle is exactly the artifact that
        must not drop it.
        """
        patterns = pyproject["tool"]["setuptools"]["package-data"]["animetext_studio"]
        assert "models/*/*.txt" in patterns, "the GPL text beside the weights would be dropped"
        if not bundle_present():
            pytest.skip(BUNDLE_ABSENT)
        assert GPL_FOR_WEIGHTS.is_file(), (
            "the bundled detection weights are GPL-3.0 and must ship their licence beside them")
        raw = GPL_FOR_WEIGHTS.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == GPL_TEXT_SHA256, "not the canonical GPL-3.0 text"
        assert "models/animetext-yolo/LICENSE.GPL-3.0.txt" in _shipped_data_files(patterns), (
            "the GPL text exists on disk but a wheel would drop it")

    def test_the_notice_names_every_bundled_component(self) -> None:
        notice = NOTICE_FILE.read_text(encoding="utf-8")
        for component in ("deepghs/AnimeText_yolo", "JustANormalTinkerer/hayai-ocr-v2.7-supernova"):
            assert component in notice, f"{component} ships in the tree but NOTICE never names it"
        assert "GPL-3.0" in notice and "Apache-2.0" in notice
        assert "LICENSE.GPL-3.0.txt" in notice, "NOTICE should say where the GPL text lives"
        assert "sample_scans" in notice, "the user's scans are the one material that must not " \
                                         "look licensed for redistribution"

    @pytest.mark.parametrize("dist,family", sorted(NOTICE_DEPS.items()))
    def test_the_notice_agrees_with_what_the_package_reports(self, dist: str, family: str) -> None:
        import importlib.metadata as md
        try:
            meta = md.metadata(dist)
        except md.PackageNotFoundError:
            pytest.skip(f"{dist} is not installed in this environment")
        reported = " | ".join(filter(None, [
            meta.get("License-Expression") or "", meta.get("License") or "",
            *(meta.get_all("Classifier") or [])]))
        assert family.lower() in reported.lower(), (
            f"{dist} reports {reported[:120]!r}, NOTICE says {family}")
        claim = _notice_claims().get(dist.lower())
        assert claim is not None, f"NOTICE has no 'name .... licence' line for {dist}"
        assert family.lower() in claim.lower(), (
            f"NOTICE line for {dist} is {claim!r}, its metadata says {family}")

    def test_the_readme_states_both_halves(self, readme: str) -> None:
        section = readme.split("## Licensing", 1)
        assert len(section) == 2, "README lost its Licensing section"
        body = section[1]
        assert "Apache-2.0" in body, "the code's own licence must be in the README"
        assert "GPL-3.0" in body and "deepghs/AnimeText_yolo" in body
        assert "hayai-ocr-v2.7-supernova" in body
        assert "(LICENSE)" in body and "(NOTICE)" in body, "both files must be linked, not named"
        assert "fetch_models.py" in body, "the slim-wheel route for GPL-averse distributions"

    def test_nothing_else_claims_a_conflicting_licence(self) -> None:
        """One licence per file. A stray `-*- license: GPL -*-` header or a second LICENSE file
        at the root is how a codebase ends up with an unanswerable provenance question."""
        strays = [p.relative_to(ROOT) for p in ROOT.iterdir()
                  if p.is_file() and re.match(r"(LICENSE|COPYING|LICEN[CS]E)", p.name, re.I)
                  and p.name != "LICENSE"]
        assert strays == [], f"second licence file at the root: {strays}"
        headers = []
        for path in list(PACKAGE.rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
            if "models" in path.parts:
                continue          # the vendored Hayai sources keep their own headers
            head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:12]).lower()
            if re.search(r"licen[cs]e:\s*(gpl|agpl|lgpl|mit|bsd)", head):
                headers.append(str(path.relative_to(ROOT)))
        assert headers == [], f"per-file licence headers contradicting the root licence: {headers}"


# ---------------------------------------------------------------------------
# What the repository publishes (decision D95, amending D26)
# ---------------------------------------------------------------------------

#: GitHub's hard per-file limit. A file above this is *rejected by the push*, not warned about —
#: which is the difference between a rule and a preference.
PUBLISH_LIMIT = 100 * 1024 * 1024

#: `/home/you/...` is the placeholder the recorded MCP sessions use; anything else under
#: `/home/` or `/Users/` is somebody's real machine, published by accident.
MACHINE_PATH = re.compile(rb"/Users/[A-Za-z0-9._-]+|/home/(?!you\b)[A-Za-z0-9._-]+")


def tracked_files() -> list[str]:
    """The published set — `git ls-files`. Empty-tree and non-repo copies return []."""
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=str(ROOT), capture_output=True)
    if proc.returncode != 0:
        return []
    return sorted(f for f in proc.stdout.decode("utf-8", "replace").split("\0") if f)


@pytest.mark.skipif(not (ROOT / ".git").is_dir(), reason="a folder copy, not a repository")
class TestWhatTheRepositoryPublishes:
    """D26 said the bundle is part of the deliverable; D95 said the *repository* is not a deliverable.

    The three reasons for that are not interchangeable, so they are checked separately: the
    AnimeText weights are their author's to distribute (a licence question), the Hayai snapshot
    is 604 MB on its own (a hosting question, and GitHub answers it with a 403 rather than a
    conversation), and `sample_scans/` is the user's own material (a permission question). A
    single rule that conflates them gets argued away the first time one reason stops applying.
    """

    def test_the_gitignore_keeps_the_bundle_and_the_scans_out(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for rule in ("animetext_studio/models/", "sample_scans/",
                     "HF_token.txt", "GH_token.txt", ".npm/"):
            assert re.search(rf"(?m)^{re.escape(rule)}$", gitignore), \
                f"`.gitignore` no longer carries the rule {rule!r}"
        negated = [line for line in gitignore.splitlines()
                   if line.startswith("!") and "models" in line]
        assert negated == [], f"a negation re-admits the bundle: {negated}"

    def test_no_weight_and_no_scan_is_tracked(self) -> None:
        files = tracked_files()
        assert files, "git lists nothing — the repository has no commit yet"
        offenders = [f for f in files
                     if f.startswith(("animetext_studio/models/", "sample_scans/"))
                     or f.endswith((".onnx", ".safetensors", ".pt"))]
        assert offenders == [], f"published material that is not ours to publish: {offenders}"

    def test_nothing_tracked_passes_the_publishing_limit(self) -> None:
        over = [(f, (ROOT / f).stat().st_size) for f in tracked_files()
                if (ROOT / f).is_file() and (ROOT / f).stat().st_size > PUBLISH_LIMIT]
        assert over == [], [f"{(ROOT / f).stat().st_size // 1048576} MB  {f}"
                            for f, _ in over]

    def test_no_tracked_text_carries_a_machine_path(self) -> None:
        """A recorded session transcript records the machine it was recorded on — until it is
        published, when it starts recording the author's home directory instead."""
        home = os.path.expanduser("~").encode()
        hits = []
        for f in tracked_files():
            path = ROOT / f
            if not path.is_file() or path.stat().st_size > 2_000_000:
                continue
            body = path.read_bytes()
            if MACHINE_PATH.search(body) or home in body:
                hits.append(f)
        assert hits == [], f"tracked files naming a real machine: {hits}"

    def test_a_clone_is_told_how_to_get_the_weights(self) -> None:
        """The rule is only half of it: a clone has to be able to find the other half."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "A clone is not a copy" in readme, "the README never says a clone has no bundle"
        assert "D95" in readme and "D26" in readme, "the amendment must name what it amends"
        assert "scripts/fetch_models.py" in readme
        assert (ROOT / "scripts" / "fetch_models.py").is_file()


