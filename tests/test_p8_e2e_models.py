"""P8.5 — the real-model end-to-end suite (opt-in: `python -m pytest -m models`).

Every other file in this suite tests a *part* against real weights (an engine, a crop, a
detection). This file tests the **product**: one page, the vendored `yolo12m_animetext` ONNX
and the pinned Hayai snapshot, taken through every door the app has, with the answer checked
against the published schemas rather than against a fixture written by the same hand that
wrote the code.

What is pinned here, and why only here:

* **the document** — a real run validates against `result.schema.json`. Until now that schema
  had only ever been exercised against hand-written documents.
* **the invariants** — ids, reading order, palette colours, boxes inside the page, the D39
  rule (null text carries an error), `contained_in` re-derived from the boxes it cites,
  `flags` re-derived from `max_new_tokens`.
* **that it read the page** — a canary against the sample page's own ground truth. Deliberately
  tolerant: models get re-released, so this asks *was the page read*, not *was it read exactly
  the way it was today*.
* **determinism** — two real runs differ only in `created_at` and `timing_ms`. Mocks have
  simulated that promise since P4; this is the first time it is tested on the real thing.
* **every door** — the headless CLI in a real subprocess, the MCP service façade, the MCP wire
  over HTTP with the official client, the warm-up job, a cancellation inside a real OCR pass,
  and the real manga scans in `sample_scans/` (geometry only: those files stay opaque).

No token, no network, no display: the weights are on premises (D26), `paths.offline` is on,
and one test proves it by refusing every socket the run tries to open.

CPU-tolerant by construction: one pair of engines is loaded once for the module, the canonical
run is shared by every document assertion, and no test asserts a wall-clock performance
promise. Expect ~2 minutes on a CPU box.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.models, pytest.mark.timeout(1800)]

jsonschema = pytest.importorskip("jsonschema",
                                 reason="jsonschema is a dev extra (pip install -e .[dev])")

from animetext_studio.core.config import AppConfig, McpConfig, load_schema   # noqa: E402
from animetext_studio.core.jobs import JobManager                            # noqa: E402
from animetext_studio.core.model_store import ModelStore                     # noqa: E402
from animetext_studio.core.pipeline import Pipeline, engines_from_config     # noqa: E402
from animetext_studio.core.types import (                                    # noqa: E402
    BLOCK_FLAGS, CONTAINMENT_RATIO, DETECTION_REPO, OCR_REPO, OCR_REVISION,
    SCHEMA_VERSION, TEXT_LABEL, TRUNCATION_RATIO, BBox, JobState,
    color_for_index,
)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "sample_page.png"
SCANS = sorted((ROOT / "sample_scans").glob("*.jpg"))
MODEL = "yolo12m_animetext"

#: What `scripts/make_sample_image.py` draws on `examples/sample_page.png`. Used as a canary
#: (was the page read?) and guarded against drift by
#: `TestTheDocument::test_the_canary_still_matches_the_page_generator`.
GROUND_TRUTH = [
    "この攻撃で決まりだ！", "まだだよ、私は", "絶対に負けない", "何!? その力は",
    "明日、学校で会おう", "真相はいつも闇の中",
    "決戦の前夜", "風が止んだ", "第４話", "その日、空は赤かった",
]


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


# ---------------------------------------------------------------------------
# Fixtures: one bundle, one pair of engines, one canonical run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle() -> ModelStore:
    """The on-premises bundle, or a skip that says how to get it (never a failure)."""
    store = ModelStore(offline=True)
    if not store.has_detection_variant(MODEL):
        pytest.skip(f"{MODEL} is not vendored — "
                    f"python scripts/fetch_models.py --variants {MODEL}")
    if not store.has_ocr_snapshot():
        pytest.skip("the Hayai snapshot is not vendored — python scripts/fetch_models.py")
    return store


@pytest.fixture(scope="module")
def real_config() -> AppConfig:
    """The stock config with the network door shut — what `--offline` means."""
    config = AppConfig.default()
    config.paths.offline = True
    config.detection.model = MODEL
    return config


@pytest.fixture(scope="module")
def engines(real_config):
    """One pair of real engines for the whole module: loading is seconds, not free."""
    pair = engines_from_config(real_config)
    yield pair
    pair.unload_all()


@pytest.fixture(scope="module")
def canonical(engines, real_config):
    """The one full run every document assertion reads (progress recorded, nothing else).

    Shared on purpose: the page is deterministic, and re-running it per assertion would turn
    an opt-in suite into a five-minute one.
    """
    if not SAMPLE.exists():
        pytest.skip(f"{SAMPLE} missing")
    stages: list[str] = []
    pipeline = Pipeline(engines.detection, engines.ocr, real_config)
    started = time.monotonic()
    result = pipeline.run_file(str(SAMPLE),
                               progress=lambda stage, _p, _m: stages.append(stage))
    return SimpleNamespace(result=result, document=result.to_dict(),
                           stages=list(dict.fromkeys(stages)),
                           seconds=time.monotonic() - started)


@pytest.fixture(scope="module")
def validate():
    """`validate(document)` → the document, or an AssertionError listing every violation."""
    schema = load_schema("result.schema.json")
    validator = jsonschema.Draft202012Validator(schema)

    def _validate(document: dict, where: str = "") -> dict:
        errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
        if errors:
            detail = "\n".join(
                f"    {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
                for e in errors[:12])
            raise AssertionError(f"not valid against result.schema.json"
                                 f"{(' ' + where) if where else ''}:\n{detail}")
        return document

    return _validate


@pytest.fixture(scope="module")
def service(engines, real_config):
    """The MCP façade over the same real engines, headless (`--serve --mock` is the other one)."""
    from animetext_studio.mcp.service import StudioService

    manager = JobManager(Pipeline(engines.detection, engines.ocr, real_config))
    svc = StudioService(manager, real_config, mock=False)
    yield svc
    svc.unload()


def _bbox(data: dict) -> BBox:
    return BBox(int(data["x0"]), int(data["y0"]), int(data["x1"]), int(data["y1"]))


def _area(box: BBox) -> int:
    return max(0, box.x1 - box.x0) * max(0, box.y1 - box.y0)


def check_block_invariants(document: dict) -> None:
    """Re-derive every claim a real document makes about its own blocks.

    Not a schema check (the schema cannot see between blocks): ids, order, colours, geometry,
    the D39 null-text rule, and both 1.2 flags, computed here from the boxes and the settings
    the document itself reports.
    """
    blocks = document["blocks"]
    width = document["image"]["width"]
    height = document["image"]["height"]
    max_new_tokens = int(document["ocr"]["max_new_tokens"])
    by_id = {int(b["id"]): b for b in blocks}
    assert len(by_id) == len(blocks), "block ids are not unique"

    assert [b["id"] for b in blocks] == list(range(len(blocks))), "ids must be 0..n-1 in order"
    assert sorted(b["reading_order"] for b in blocks) == list(range(len(blocks))), \
        "reading_order must be a permutation of 0..n-1"
    assert document["block_count"] == len(blocks)

    for block in blocks:
        box = _bbox(block["bbox"])
        where = f"block {block['id']}"
        assert 0 <= box.x0 < box.x1 <= width, f"{where}: x outside the page ({box})"
        assert 0 <= box.y0 < box.y1 <= height, f"{where}: y outside the page ({box})"
        assert 0.0 <= float(block["score"]) <= 1.0, f"{where}: score out of range"
        assert block["label"] == TEXT_LABEL, f"{where}: unexpected label"
        assert block["color"] == color_for_index(int(block["id"])), \
            f"{where}: colour is not the palette colour for its index"
        # D39: a null text is never a silent one.
        assert block["text"] is not None or block["error"], \
            f"{where}: text is null with no error to explain it"
        assert set(block["flags"]) <= set(BLOCK_FLAGS), f"{where}: unknown flag"
        truncated = "possible_truncation" in block["flags"]
        expect_truncated = bool(block["text"]) and \
            len(block["text"]) >= max_new_tokens * TRUNCATION_RATIO
        assert truncated == expect_truncated, \
            f"{where}: truncation flag does not match len(text) vs max_new_tokens"
        parent = block["contained_in"]
        if parent is not None:
            assert parent in by_id and parent != block["id"], f"{where}: contained_in is not a block"
            outer = _bbox(by_id[parent]["bbox"])
            assert _area(outer) > _area(box), f"{where}: container is not strictly larger"
            ratio = box.contained_ratio(outer)
            assert ratio >= CONTAINMENT_RATIO, \
                f"{where}: contained_in cites a box covering only {ratio:.2f} of it"


# ---------------------------------------------------------------------------
# 1. The weights that ran
# ---------------------------------------------------------------------------

class TestTheWeightsThatRun:
    def test_the_bundle_carries_both_pinned_repositories(self, bundle):
        status = bundle.status()
        assert status["offline"] is True
        assert status["detection_repo"] == DETECTION_REPO
        assert MODEL in status["detection_variants"], "the test variant is not listed"
        assert status["ocr"]["bundled"] is True
        assert status["ocr"]["revision"] == OCR_REVISION, \
            "the bundle the store reports is not the revision the code pins"

    def test_the_ocr_snapshot_is_the_revision_the_code_pins(self, bundle):
        origin = json.loads((bundle.ocr_dir() / ".origin.json").read_text(encoding="utf-8"))
        assert origin["revision"] == OCR_REVISION, (
            "the vendored Hayai snapshot and core/types.py:OCR_REVISION disagree — the "
            "document would claim a model that is not on disk")

    def test_a_real_run_opens_no_socket_and_needs_no_token(self, real_config):
        """"Copy the folder" is only an offline install if nothing reaches for the network.

        Fresh engines on purpose: the load is the half that would phone Home, so it has to
        happen inside the guard, not before it.
        """
        pair = engines_from_config(real_config)
        attempts: list = []
        real_connect = socket.socket.connect

        def spy(self, address, *args, **kwargs):
            attempts.append(address)
            return real_connect(self, address, *args, **kwargs)

        socket.socket.connect = spy
        try:
            result = Pipeline(pair.detection, pair.ocr, real_config).run_file(str(SAMPLE))
        finally:
            socket.socket.connect = real_connect
            pair.unload_all()

        assert result.block_count >= 1
        assert attempts == [], f"the run reached for the network: {attempts}"
        assert os.environ.get("HF_HUB_OFFLINE") == "1"


# ---------------------------------------------------------------------------
# 2. The document
# ---------------------------------------------------------------------------

class TestTheDocument:
    def test_one_page_becomes_a_schema_valid_document(self, canonical, validate):
        validate(canonical.document, "(full run of the sample page)")
        assert canonical.document["schema_version"] == SCHEMA_VERSION
        assert canonical.document["stage"] == "full"
        assert canonical.document["block_count"] >= 4, "the sample page has 12 boxes on purpose"

    def test_the_document_records_the_run_that_produced_it(self, canonical, real_config):
        document = canonical.document
        digest = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
        assert document["image"] == {
            "source": "file", "file_name": SAMPLE.name,
            "width": document["image"]["width"], "height": document["image"]["height"],
            "sha256": digest,
        }
        detection = document["detection"]
        assert detection["repo"] == DETECTION_REPO
        assert detection["model"] == MODEL
        assert detection["backend"] == "onnxruntime"
        assert detection["iou_threshold"] == real_config.detection.iou_threshold
        assert detection["score_threshold"] == real_config.detection.score_threshold
        ocr = document["ocr"]
        assert ocr["repo"] == OCR_REPO
        assert ocr["device"] == "cpu" or ocr["device"].startswith("cuda"), ocr["device"]
        assert ocr["max_patches"] == real_config.ocr.max_patches
        assert ocr["batch_size"] == real_config.ocr.batch_size
        assert ocr["max_new_tokens"] == real_config.ocr.max_new_tokens
        assert ocr["crop_padding_px"] == real_config.ocr.crop_padding_px
        assert ocr["normalize_nfkc"] == real_config.ocr.normalize_nfkc

    def test_every_block_is_a_complete_fact(self, canonical):
        check_block_invariants(canonical.document)

    def test_the_page_was_actually_read(self, canonical):
        """A canary, not a quality gate: it asks whether the page was *read*."""
        texts = [b["text"] for b in canonical.document["blocks"] if b["text"]]
        assert len(texts) >= 3, f"only {len(texts)} blocks came back with text"
        kana = [t for t in texts if any("\u3040" <= c <= "\u30ff" for c in t)]
        assert len(kana) >= 2, f"no Japanese in the output: {texts}"
        joined = _nfkc(" ".join(texts))
        hits = [line for line in GROUND_TRUTH if _nfkc(line) in joined]
        assert len(hits) >= 3, (
            "the page was not recognised — expected several of the sample page's own lines "
            f"in the output.\n  hits: {hits}\n  texts: {texts}")

    def test_the_canary_still_matches_the_page_generator(self):
        source = (ROOT / "scripts" / "make_sample_image.py").read_text(encoding="utf-8")
        missing = [line for line in GROUND_TRUTH if line not in source]
        assert not missing, (
            f"scripts/make_sample_image.py no longer draws {missing}; the OCR canary in this "
            "file is testing a page that does not exist")

    def test_two_runs_differ_only_by_the_clock(self, engines, real_config):
        """The deterministic-client promise (PLAN §4), on real weights for the first time."""
        pipeline = Pipeline(engines.detection, engines.ocr, real_config)
        first = pipeline.run_file(str(SAMPLE)).to_dict()
        second = pipeline.run_file(str(SAMPLE)).to_dict()
        for document in (first, second):
            document.pop("created_at")
            document.pop("timing_ms")
        assert first == second, "two runs of the same page and settings produced two documents"

    def test_the_document_round_trips_through_its_own_json(self, canonical):
        from animetext_studio.core.types import PipelineResult

        text = canonical.result.to_json()
        back = PipelineResult.from_dict(json.loads(text))
        assert back.to_dict() == canonical.document
        assert back.stage == "full" and back.ocr_ran is True

    def test_a_stage_one_run_reads_nothing_and_still_validates(self, engines, real_config,
                                                               validate, canonical):
        result = Pipeline(engines.detection, engines.ocr, real_config).run_file(
            str(SAMPLE), ocr_enabled=False)
        document = validate(result.to_dict(), "(stage-one run)")
        assert document["stage"] == "detect"
        assert document["block_count"] == canonical.document["block_count"], \
            "the detector found a different number of boxes for the same page"
        for block in document["blocks"]:
            assert block["text"] is None
            assert block["error"] and block["error"]["code"] == "SKIPPED"
        check_block_invariants(document)

    def test_the_timings_are_real_and_the_stages_are_the_run_stages(self, canonical):
        timing = canonical.document["timing_ms"]
        assert timing["detect"] > 0 and timing["ocr"] > 0
        assert timing["total"] >= max(timing["detect"], timing["ocr"])
        assert "detect" in canonical.stages and "ocr" in canonical.stages
        assert "download" not in canonical.stages, "an offline run reported a download stage"
        # A hang guard, not a performance promise: 12 crops on one page must not take 5 minutes.
        assert canonical.seconds < 300, f"the run took {canonical.seconds:.0f}s"


# ---------------------------------------------------------------------------
# 3. The 1.2 flags, on data nobody wrote for the test
# ---------------------------------------------------------------------------

class TestTheFlagsOnRealData:
    def _run(self, engines, real_config, **ocr_overrides):
        config = copy.deepcopy(real_config)
        for key, value in ocr_overrides.items():
            setattr(config.ocr, key, value)
        return Pipeline(engines.detection, engines.ocr, config).run_file(str(SAMPLE))

    def test_the_truncation_flag_fires_when_the_cap_bites(self, engines, real_config, validate,
                                                          canonical):
        """`possible_truncation` has to be derivable from the document, and it has to fire.

        With the default 128-token cap nothing on this page is flagged, which proves nothing;
        with an 8-token cap the same boxes must say so. The flag is only worth having because
        the second half of that sentence is checkable from outside the app.
        """
        assert not any(b["flags"] for b in canonical.document["blocks"]), \
            "something is already truncated at the default cap"
        document = self._run(engines, real_config, max_new_tokens=8).to_dict()
        validate(document, "(max_new_tokens=8)")
        flagged = [b for b in document["blocks"] if "possible_truncation" in b["flags"]]
        assert flagged, (
            "an 8-token cap cut this page and nothing was flagged: "
            f"{[b['text'] for b in document['blocks']]}")
        check_block_invariants(document)

    def test_the_top_patch_budget_is_reachable(self, engines, real_config, validate, canonical):
        """512 was added in 8.2b to close a gap in the GUI; nobody had run it against Hayai.

        Measured here rather than assumed: the answer is the same text as `auto` for a page
        this size, because `auto` already fits it. The point is that the tier is accepted by
        the engine and produces a valid document, not that it changes the reading.
        """
        document = self._run(engines, real_config, max_patches=512).to_dict()
        validate(document, "(max_patches=512)")
        assert document["ocr"]["max_patches"] == 512
        assert document["block_count"] == canonical.document["block_count"]
        assert [b["text"] for b in document["blocks"] if b["text"]], "512 read nothing"
        check_block_invariants(document)


# ---------------------------------------------------------------------------
# 4. The doors
# ---------------------------------------------------------------------------

class TestTheCliDoor:
    """`--headless --offline --run` in a real subprocess: the door a script uses.

    In-process `main()` tests already exist; what only a subprocess can prove is that the
    package is importable as installed, that stdout carries *nothing* but the document (a
    tqdm bar on stdout would corrupt `--run > out.json`), and that a cold process agrees
    with a warm one about the same page.
    """

    def test_a_headless_run_prints_the_document_and_nothing_else(self, tmp_path, canonical):
        out = tmp_path / "cli.json"
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                   PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
        started = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", "animetext_studio", "--headless", "--offline",
             "--image", str(SAMPLE), "--run", "--out", str(out)],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, f"exit {proc.returncode}:\n{proc.stderr[-3000:]}"
        assert "Traceback" not in proc.stderr, proc.stderr[-3000:]

        document = json.loads(proc.stdout)          # stdout is exactly one JSON document
        assert document["stage"] == "full"
        assert out.exists() and json.loads(out.read_text(encoding="utf-8")) == document

        # The same page through a cold process is the same document, clock aside.
        mine = canonical.document
        assert {k: v for k, v in document.items() if k not in ("created_at", "timing_ms")} \
            == {k: v for k, v in mine.items() if k not in ("created_at", "timing_ms")}
        assert proc.stderr.strip(), "a run that says nothing on stderr is a run you cannot watch"
        assert time.monotonic() - started < 600

    def test_the_model_status_door_reports_the_bundled_variant(self):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
                   PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]))
        proc = subprocess.run([sys.executable, "-m", "animetext_studio", "--offline",
                               "--model-status"],
                              cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert MODEL in proc.stdout
        assert "download" not in proc.stdout.lower(), proc.stdout[-1500:]


class TestTheMcpServiceDoor:
    """The façade an embedded caller (or `--serve`) uses, on the real engines."""

    def test_load_run_answer_save_and_annotate_a_real_page(self, service, tmp_path, validate,
                                                           canonical):
        loaded = service.load_image(path=str(SAMPLE))
        assert loaded["loaded"] is True
        assert loaded["image"]["file_name"] == SAMPLE.name

        job_id = service.start_job()
        settled = service.wait_for_job()
        assert settled["state"] == "done", settled
        assert settled["job_id"] == job_id

        payload = service.result_payload()
        document = validate(payload["result"], "(run_pipeline payload)")
        assert document["stage"] == "full"
        assert document["block_count"] == canonical.document["block_count"], \
            "the MCP door read the page differently from the pipeline door"
        check_block_invariants(document)

        saved = service.save_result(str(tmp_path / "mcp.json"))
        assert saved["block_count"] == document["block_count"]
        assert json.loads((tmp_path / "mcp.json").read_text(encoding="utf-8")) == document

        png, meta = service.annotated()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert (meta["width"], meta["height"]) == (document["image"]["width"],
                                                   document["image"]["height"])

    def test_the_status_and_resources_describe_the_real_engines(self, service):
        snapshot = service.snapshot()
        by_name = {e["name"]: e for e in snapshot["engines"]}
        assert by_name["detection"]["loaded"] is True
        assert by_name["ocr"]["loaded"] is True
        assert by_name["detection"]["backend"] == "onnxruntime"
        for engine in by_name.values():
            assert engine["device"] in ("cpu",) or engine["device"].startswith("cuda")
            assert engine["inferences"] >= 1 and engine["warm"] is True, \
                "a page has run through this engine and it does not say so"
        assert snapshot["has_result"] is True and snapshot["block_count"] >= 4
        assert snapshot["mock"] is False and snapshot["offline"] is True

        assert json.loads(service.result_document())["stage"] == "full"
        models = json.loads(service.models_document())
        assert models["ocr_repo"] == OCR_REPO and models["detection_repo"] == DETECTION_REPO
        assert models["offline"] is True
        assert json.loads(service.config_document())["detection"]["model"] == MODEL

    def test_a_real_warm_up_is_a_job_and_not_a_page(self, service):
        """`--preload` over the wire, with the weights that make it cost anything."""
        service.unload()
        cold = {e["name"]: e for e in service.snapshot()["engines"]}
        assert cold["ocr"]["loaded"] is False and cold["ocr"]["warm"] is False

        page = service.result          # the page the previous test put up
        handle = service.warm_engines()
        assert handle["kind"] == "warmup" and handle["busy"] is True
        settled = service.wait_for_job()
        assert settled["state"] == "done", settled
        assert settled["kind"] == "warmup"

        report = service.manager.warmup_report
        assert report is not None and report.ok, report.to_dict() if report else None
        assert {o.engine for o in report.outcomes} == {"detection", "ocr"}
        for outcome in report.outcomes:
            assert outcome.state == "ready" and outcome.probe_ms > 0, outcome.to_dict()

        warm = {e["name"]: e for e in service.snapshot()["engines"]}
        assert warm["detection"]["warm"] is True and warm["ocr"]["warm"] is True
        assert service.result is page, "a warm-up replaced the page the user was looking at"

    def test_cancelling_inside_a_real_ocr_pass_leaves_the_app_usable(self, engines,
                                                                     real_config):
        """Cancellation has to interrupt the real torch loop, not just a fake one.

        Held by a gate rather than raced: the wrapper stops the call reaching Hayai until the
        cancellation is already set, so the assertion is that the engine's own chunk loop
        notices the event on its first `check_cancel` — not that this test won a race. The
        follow-up run is the half that matters: an engine that cancels but cannot be used
        again is worse than one that cannot be cancelled.
        """
        real_ocr = engines.ocr
        original = real_ocr.ocr_crops
        entered, release = threading.Event(), threading.Event()

        def gated(images, **kwargs):
            entered.set()
            release.wait(300)
            return original(images, **kwargs)      # the real engine, after the cancel is in

        config = copy.deepcopy(real_config)
        config.ocr.batch_size = 1
        manager = JobManager(Pipeline(engines.detection, real_ocr, config))
        real_ocr.ocr_crops = gated
        try:
            started = manager.start_file(str(SAMPLE))
            assert entered.wait(300), "the run never reached the OCR stage"
            manager.cancel()
            release.set()
            settled = manager.wait(300)
        finally:
            real_ocr.__dict__.pop("ocr_crops", None)
            release.set()
        assert started.job_id == settled.job_id
        assert settled.state is JobState.CANCELLED, (
            f"the real engine ignored a cancellation set before its first chunk: "
            f"{settled.state}")
        assert manager.result is None, "a cancelled job published a result"

        again = JobManager(Pipeline(engines.detection, engines.ocr, real_config))
        assert again.run_file(str(SAMPLE), timeout=600).block_count >= 4


async def _wire(url: str, image: str) -> dict:
    """One real MCP session against a real server with real engines."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with contextlib.AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(streamable_http_client(url))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        loaded = await session.call_tool("load_image", {"path": image})
        assert not loaded.is_error, loaded.structured_content

        # `run_pipeline` takes no path on purpose: it runs the page that is loaded, which is
        # the difference between a tool and a command line.
        ran = await session.call_tool("run_pipeline", {"include_annotated_image": True})
        assert not ran.is_error, ran.structured_content
        got = await session.call_tool("get_result", {})
        assert not got.is_error, got.structured_content

        latest = await session.read_resource("result://latest")
        annotated = _blob_of(await session.read_resource("image://annotated"))
        log = await session.read_resource("log://mcp")
        return {"load": loaded.structured_content, "run": ran.structured_content,
                "get": got.structured_content,
                "latest": latest.contents[0].text,
                "annotated": annotated,
                "log": log.contents[0].text}


def _blob_of(result) -> bytes:
    """The bytes of a resource read, blob or text.

    The SDK sends `image://*` as base64 in `blob` and the JSON resources as `text`; a test
    that hard-codes one of the two breaks when the server starts choosing the other.
    """
    item = result.contents[0]
    raw = getattr(item, "blob", None)
    return base64.b64decode(raw) if raw else (item.text or "").encode("utf-8")


class TestTheMcpWireDoor:
    """The whole stack: client → streamable-HTTP → service → job → Hayai → back again.

    This is the door an agent actually uses, so it is the one worth proving end to end:
    the answer validates against the *published* tool schema (the catalog a client is told)
    and the document inside it validates against `result.schema.json`.
    """

    def test_a_session_runs_a_real_page_and_validates_twice(self, service, free_port, validate):
        catalog = load_schema("mcp-tools.json")
        tool = {t["name"]: t for t in catalog["tools"]}["run_pipeline"]

        config = copy.deepcopy(service.config)
        config.mcp = McpConfig(enabled=True, transport="streamable-http",
                               host="127.0.0.1", port=free_port())
        service.apply_config_patch(config.to_dict())

        from animetext_studio.mcp.server import StudioServer

        server = StudioServer(service, config.mcp)
        info = server.start()
        assert info["running"], info
        try:
            import asyncio
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                outcome = asyncio.run(_wire(info["url"], str(SAMPLE)))
        finally:
            server.stop()

        run = outcome["run"]
        jsonschema.Draft202012Validator(tool["outputSchema"]).validate(run)
        document = validate(run["result"], "(run_pipeline over the wire)")
        assert document["stage"] == "full" and document["block_count"] >= 4
        assert run["block_count"] == document["block_count"]
        assert run["progress_events"] > 0, "a real run streamed no progress to the client"
        assert run["annotated_image"]["bytes"] > 0

        # The text survived the wire: JSON escapes nothing that should be lost.
        assert any("\u3040" <= c <= "\u30ff"
                   for b in document["blocks"] for c in (b["text"] or ""))

        assert json.loads(outcome["latest"]) == document
        assert outcome["get"]["result"] == document, "get_result answered with another page"
        assert outcome["annotated"][:8] == b"\x89PNG\r\n\x1a\n", "image://annotated is not a PNG"
        logged = json.loads(outcome["log"])
        tools = {e["tool"] for e in logged["entries"]}
        assert {"load_image", "run_pipeline", "get_result"} <= tools
        assert logged["errors"] == 0, logged["entries"][:3]


# ---------------------------------------------------------------------------
# 5. Real pages
# ---------------------------------------------------------------------------

class TestRealPages:
    """`sample_scans/` — opaque inputs. Geometry and validity only, never content.

    These are the only pages here nobody generated: real JPEGs, real sizes, real noise. The
    standing rule is that their content is not inspected, so every assertion below is about
    shape — a box that is inside the page, an id that is sequential, a document that
    validates — and none of it can be broken by a model that reads better tomorrow.
    """

    @pytest.mark.parametrize("scan", SCANS[:3], ids=lambda p: p.name)
    def test_a_real_page_is_detected_into_a_valid_document(self, engines, real_config, scan,
                                                           validate):
        result = Pipeline(engines.detection, engines.ocr, real_config).run_file(
            str(scan), ocr_enabled=False)
        document = validate(result.to_dict(), f"({scan.name}, stage one)")
        assert document["stage"] == "detect"
        assert document["image"]["width"] > 0 and document["image"]["height"] > 0
        assert document["block_count"] >= 1, f"{scan.name}: no text box at all?"
        check_block_invariants(document)

    def test_one_real_page_survives_a_full_run(self, engines, real_config, validate):
        if not SCANS:
            pytest.skip("no sample_scans/ on disk")
        scan = max(SCANS[:4], key=lambda p: p.stat().st_size)
        result = Pipeline(engines.detection, engines.ocr, real_config).run_file(str(scan))
        document = validate(result.to_dict(), f"({scan.name}, full)")
        assert document["stage"] == "full"
        assert document["block_count"] >= 1
        assert result.timing_ms.ocr > 0
        check_block_invariants(document)
        # Content is not inspected; *having* content is.
        assert any(b["text"] for b in document["blocks"]), \
            f"{scan.name}: a full run read nothing anywhere on the page"
