"""P8.8 — final QA: every original requirement, one row, one piece of evidence.

PLAN §3 Part 8 closes with "Final QA checklist against every original requirement (§7)". A
checklist written as prose is worth the afternoon it was written, so this one is a table with
rules, and the rules are tests:

* every row cites **evidence** — a test node id, a command, a check defined in this file, or
  (rarely, and named) a human review of a screenshot;
* a cited test must exist in the suite, and a cited command must be real with real flags —
  so an evidence column cannot quietly rot into folklore;
* no row may be left un-met. What is not done lives in `FOLLOW_UPS`, argued as *not a
  requirement*, so "open" is always a decision and never an accident.

`docs/QA.md` is generated from this table (`python3 scripts/gen_qa_report.py`, `--check` fails
on drift) — the same no-rot rule `docs/ERROR-MATRIX.md` follows.

The second half of the file is the checks themselves: the cross-cutting ones that no single
part's test file owns — that the three doors produce the same document, that nothing outside
the window needs Qt, that PLAN's CLI/tool/contract surfaces are the shipped ones, that the
generated docs and the example config are current, and that no secret travels in the tree.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from animetext_studio import __version__
from animetext_studio.core.cli import build_parser
from animetext_studio.core.config import AppConfig
from animetext_studio.core.types import EXIT_CODES, ErrorCode

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "PLAN.md"
README = ROOT / "README.md"
CATALOG = ROOT / "animetext_studio" / "schemas" / "mcp-tools.json"
DOCS = ROOT / "docs"

#: Recorded at sign-off (2026-11-24). `test_the_suite_has_not_shrunk` treats these as floors:
#: the suite may grow, it may not quietly lose a requirement's evidence.
SIGNOFF = {
    "date": "2026-11-26",
    # Measured on the sign-off run, not remembered from the last one: 1253 passed / 41
    # deselected offline in ~41 s, and `-m models` 40 passed / 1 skipped offline in ~50 s.
    # 1258 from D95, which added the publication boundary: five tests that hold what the
    # repository may publish, and one secret scan that now knows GitHub's token shapes too.
    # 1259 once the repository had a URL of its own to be checked against; 1260 with the
    # manual/worksheet rule, which keeps build-log residue out of the README.
    "offline_tests": 1260,
    "model_tests": 40,
    "deselected": 41,
    # 20 through P8 (PLAN's 18 + `unload_engines` + `warm_engines`); 21 from P9.3, which added
    # `reocr_blocks`. The count is asserted end to end — catalog, README, docs/MCP.md — so a new
    # tool moves all four together or the suite goes red. That is the intended friction.
    "tools": 21,
    "resources": 6,
    "error_codes": 14,
    # 84 through P8.4; 91 from P9.6, which put the doors P9 opened *in* the matrix — the four
    # re-OCR refusals at the service door, the corrupt bundled file at the offline door, and the
    # two empty-option refusals (D93) at the CLI — each with a case the sweeps run.
    "matrix_faults": 91,
}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Requirement:
    id: str
    text: str
    source: str                     # where the requirement comes from
    evidence: tuple[str, ...]       # tests/… | qa: … | cmd: … | manual: …
    status: str = "met"
    note: str = ""                  # deviation / measurement, shown in the report


MET_STATUSES = {"met", "met · deviation", "met · measured"}

GROUPS: tuple[tuple[str, str], ...] = (
    ("GOAL", "The product the goal paragraph asked for"),
    ("ARCH", "Architecture and threading (§1)"),
    ("DOC", "The result document (§2.1)"),
    ("CFG", "The config document (§2.2)"),
    ("ERR", "The error taxonomy (§2.3)"),
    ("P1", "Part 1 — foundation"),
    ("P2", "Part 2 — detection engine"),
    ("P3", "Part 3 — OCR engine + imaging"),
    ("P4", "Part 4 — pipeline, jobs, headless"),
    ("P5", "Part 5 — GUI shell"),
    ("P6", "Part 6 — visualisation + JSON pane"),
    ("P7", "Part 7 — MCP server + GUI bridge"),
    ("MCP", "Deterministic-client guarantees (§4)"),
    ("CLI", "CLI surface (§5)"),
    ("ENV", "Dependencies and environment (§6)"),
    ("RISK", "Risks and mitigations (§7)"),
    ("P8", "What Part 8 added"),
    ("P9", "What Part 9 added (the five approved enhancements, D80–D94)"),
    ("PUB", "The publication boundary (D95, amending D26)"),
)

R: list[Requirement] = []


def req(group: str, rid: str, text: str, source: str, *evidence: str,
        status: str = "met", note: str = "") -> None:
    R.append(Requirement(f"{group}-{rid}", text, source, evidence, status, note))


# -- the goal paragraph ------------------------------------------------------
req("GOAL", "1", "Python desktop app with a PySide6 window", "PLAN §Goal",
    "tests/test_gui.py::TestWindow::test_the_central_widget_is_scrollable_both_ways",
    "cmd: python3 -m animetext_studio --gui-smoke --mock")
req("GOAL", "2", "Detects anime/manga text blocks with deepghs/AnimeText_yolo YOLO12",
    "PLAN §Goal",
    "tests/test_models.py::TestRealDetection::test_synthetic_manga_page_yields_text_blocks",
    "tests/test_detection.py::TestListModels::test_repo_list_with_presets")
req("GOAL", "3", "OCRs every detected crop with hayai-ocr-v2.7-supernova", "PLAN §Goal",
    "tests/test_p8_e2e_models.py::TestTheDocument::test_one_page_becomes_a_schema_valid_document",
    "tests/test_ocr.py::TestOcrCrops::test_progress_is_per_chunk_and_monotonic")
req("GOAL", "4", "Produces a copy-able JSON mapping bounding boxes to text", "PLAN §Goal",
    "tests/test_gui.py::TestJsonPane::test_copy_all_puts_the_exact_document_on_the_clipboard",
    "qa:document-contracts")
req("GOAL", "5", "Interactive annotated-image view", "PLAN §Goal",
    "tests/test_gui.py::TestCanvas::test_the_hover_panel_stays_inside_the_viewport",
    "tests/test_gui.py::TestCanvas::test_save_annotated_writes_the_page_at_full_resolution")
req("GOAL", "6", "Fully remote-controllable over MCP, bind host/port/token configurable",
    "PLAN §Goal",
    "tests/test_mcp_http.py::test_a_full_session_round_trips_and_validates",
    "tests/test_mcp_stdio.py::test_the_stdio_handshake_publishes_the_same_catalog_as_http",
    "qa:cli-surface")
req("GOAL", "7", "Every MCP interaction is visually reflected in the GUI", "PLAN §Goal",
    "tests/test_mcp_gui.py::TestBridge::test_every_call_leaves_exactly_one_row_in_the_activity_dock",
    "tests/test_mcp_gui.py::TestBridge::test_the_badge_announces_the_server",
    note="implemented once in the bridge, not per tool — so a new tool cannot forget it")

# -- architecture (§1) -------------------------------------------------------
req("ARCH", "1", "All Qt widget access happens on the main thread", "PLAN §1 rule 1",
    "tests/test_mcp_bridge.py::test_qt_dispatcher_runs_the_callable_on_the_main_thread",
    "tests/test_mcp_bridge.py::test_qt_dispatcher_stays_out_of_its_own_thread")
req("ARCH", "2", "Inference runs on a worker thread owned by JobManager", "PLAN §1 rule 2",
    "tests/test_jobs.py::TestCancellation::test_cancel_stops_the_worker_and_lands_on_cancelled",
    note="one dedicated worker for the process lifetime, which is also what makes ONNX "
         "session reuse work (D14)")
req("ARCH", "3", "Single-flight: one job at a time, others get BUSY + the job id",
    "PLAN §1 rule 2",
    "tests/test_mcp_http.py::test_a_second_run_is_busy_with_the_running_job_id",
    "tests/test_mcp_service.py::test_a_second_job_is_busy_and_names_the_running_one")
req("ARCH", "4", "Jobs are cancellable", "PLAN §1 rule 2",
    "tests/test_jobs.py::TestCancellation::test_cancel_stops_the_worker_and_lands_on_cancelled",
    "tests/test_p8_errors.py::test_a_cancel_storm_leaves_one_settled_job")
req("ARCH", "5", "MCP server on its own thread with its own asyncio loop", "PLAN §1 rule 3",
    "tests/test_mcp_server.py::test_http_server_starts_reports_and_stops",
    "tests/test_mcp_service.py::test_the_job_wait_is_never_dispatched_onto_the_app_thread")
req("ARCH", "6", "Tool handlers never touch Qt: they marshal through the bridge",
    "PLAN §1 rule 3", "qa:no-qt-outside-the-window",
    "tests/test_mcp_bridge.py::test_qt_dispatcher_runs_the_callable_on_the_main_thread")
req("ARCH", "7", "Headless parity: --headless --run runs the same Pipeline without Qt",
    "PLAN §1", "tests/test_cli.py::TestStubs::test_headless_run_is_now_the_pipeline_not_a_stub",
    "qa:three-doors")

# -- the result document (§2.1) ---------------------------------------------
req("DOC", "1", "Every §2.1 key is present, and the document validates against the schema",
    "PLAN §2.1", "qa:document-contracts",
    "tests/test_mcp_service.py::test_run_produces_a_schema_valid_document")
req("DOC", "2", "bbox is integer xyxy in original-image space", "PLAN §2.1",
    "tests/test_types.py::TestBBox::test_to_dict_is_xyxy_integers")
req("DOC", "3", "id is the stable index after ordering; reading_order matches it", "PLAN §2.1",
    "tests/test_pipeline.py::TestDocument::test_ids_and_reading_order_are_the_document_order")
req("DOC", "4", "colour is a deterministic palette assignment", "PLAN §2.1",
    "tests/test_types.py::TestMisc::test_palette",
    "tests/test_pipeline.py::TestOrdering::test_ordering_is_stable_for_identical_boxes")
req("DOC", "5", "A per-block OCR failure does not abort the job: text null + error code",
    "PLAN §2.1", "tests/test_jobs.py::TestErrors::test_an_isolated_block_failure_is_still_a_successful_job",
    "tests/test_pipeline.py::TestFailureIsolation::test_one_poisoned_crop_does_not_lose_the_page")
req("DOC", "6", "created_at and timing_ms are the only non-deterministic fields", "PLAN §2.1",
    "qa:determinism", "tests/test_pipeline.py::TestOrdering::test_ordering_is_stable_for_identical_boxes")
req("DOC", "7", "image.source is one of file / clipboard / mcp-base64 / mcp-path", "PLAN §2.1",
    "qa:three-doors", "tests/test_mcp_service.py::test_load_image_from_path_records_mcp_provenance")
req("DOC", "8", "Schema is versioned, published, and old documents still read", "PLAN §2.1 + P8.1",
    "tests/test_schemas.py::TestResultSchemaVersions", "qa:document-contracts",
    status="met · deviation",
    note="shipped as 1.2, not 1.0: 1.1 added `stage`, 1.2 added `flags`/`contained_in`. "
         "1.0 and 1.1 are read (frozen copies ship in the wheel)")

# -- the config document (§2.2) ---------------------------------------------
req("CFG", "1", "Precedence: CLI flags > imported config file > built-in defaults", "PLAN §2.2",
    "tests/test_config.py::TestPrecedence", "qa:cli-surface")
req("CFG", "2", "reading_order: score | top_to_bottom | manga_rtl (default, column-aware)",
    "PLAN §2.2", "tests/test_pipeline.py::TestOrdering::test_manga_rtl_reads_right_to_left_inside_a_row")
req("CFG", "3", "Import: missing keys default, unknown keys warn, invalid values error",
    "PLAN §2.2", "tests/test_config.py::TestRoundtrip::test_lenient_partial",
    "tests/test_mcp_service.py::test_set_config_rejects_an_invalid_value_with_the_path")
req("CFG", "4", "Export writes the live state", "PLAN §2.2",
    "tests/test_mcp_service.py::test_config_export_and_import_round_trip")
req("CFG", "5", "The whole app state is exportable/importable, MCP settings included",
    "PLAN §2.2", "tests/test_mcp_service.py::test_config_export_and_import_round_trip",
    "tests/test_p8_hardening.py::TestRedaction::test_export_is_faithful_by_default",
    note="the one secret (`mcp.auth_token`) is there on purpose, with `--redact-secrets` / "
         "`redact_secrets` for the shareable copy (8.2)")
req("CFG", "6", "A changed MCP transport key is reported as needing a restart", "PLAN §2.2",
    "tests/test_p8_hardening.py::TestRestartRequired", "qa:restart-keys-documented",
    status="met · deviation",
    note="found by this QA pass. §2.2 promised an in-place server restart; nothing ever "
         "re-bound the socket and no door said so. The restart is deliberately not built "
         "(a client must not be able to move the socket it is talking to); the reporting is: "
         "`restart_required` on both config tools, a status-bar line on GUI import")
req("CFG", "7", "Everything else applies live, without a restart", "PLAN §2.2",
    "tests/test_p8_hardening.py::TestLogCapacity::test_capacity_follows_a_config_change",
    "tests/test_mcp_service.py::test_overlay_and_zoom_write_through_to_the_config")

# -- the error taxonomy (§2.3) ----------------------------------------------
req("ERR", "1", "One shared taxonomy of 14 codes for GUI, CLI and MCP", "PLAN §2.3",
    "qa:error-taxonomy", "tests/test_mcp_server.py::test_catalog_records_the_error_taxonomy_and_versions")
req("ERR", "2", "Every error is {code, message, details?}", "PLAN §2.3",
    "tests/test_types.py::TestMalformedPayloads::test_structured_error_not_keyerror")
req("ERR", "3", "MCP errors come back as isError:true with that JSON payload", "PLAN §2.3",
    "tests/test_mcp_http.py::test_errors_are_iserror_with_the_shared_taxonomy")
req("ERR", "4", "Every code maps to a fixed process exit code", "PLAN §2.3",
    "qa:error-taxonomy", "tests/test_cli.py::TestFoundationActions::test_validate_missing_exit_10")
req("ERR", "5", "No fault escapes as a bare exception, on any of the eight doors", "P8.4",
    "tests/test_p8_errors.py::test_no_matrix_fault_escapes_as_a_bare_exception",
    "cmd: python3 scripts/gen_error_matrix.py --check")

# -- Part 1 -----------------------------------------------------------------
req("P1", "1", "--help shows the full CLI surface", "PLAN Part 1 accept",
    "tests/test_cli.py::TestParser::test_help", "qa:cli-surface")
req("P1", "2", "Config round-trip is byte-stable", "PLAN Part 1 accept",
    "tests/test_config.py::TestRoundtrip::test_byte_stable")
req("P1", "3", "Validation reports precise, path-based errors", "PLAN Part 1",
    "tests/test_config.py::TestValidation::test_invalid_values_raise_with_path")
req("P1", "4", "Schemas ship inside the package, so wheels carry them (D6)", "PLAN §1 layout",
    "tests/test_mcp_server.py::test_catalog_is_the_file_that_ships_in_the_wheel",
    "tests/test_p8_packaging.py::TestPackageData::test_the_schemas_and_the_manifest_ship")

# -- Part 2 -----------------------------------------------------------------
req("P2", "1", "Engine protocol (load/status/progress/clear) + deterministic fake", "PLAN Part 2",
    "tests/test_engines_base.py::TestFakeDetectionEngine::test_list_models_is_the_known_catalog",
    "tests/test_engines_base.py::TestFakeDetectionEngine::test_load_sets_ready_and_rejects_unknown")
req("P2", "2", "list_models(): Hub listing with an offline/blocked fallback", "PLAN Part 2",
    "tests/test_detection.py::TestListModels::test_offline_sentinel_falls_back",
    "tests/test_detection.py::TestListModels::test_auth_failure_falls_back_to_known_models")
req("P2", "3", "Lazy per-model load with download progress", "PLAN Part 2",
    "tests/test_detection.py::TestLoad::test_download_reports_progress_percent")
req("P2", "4", "score=None resolves to the repo-recommended threshold", "PLAN Part 2",
    "tests/test_detection.py::TestThresholds::test_documented_preset_when_backend_is_cold",
    "tests/test_models.py::TestBundledDetection::test_recommended_threshold_comes_from_the_bundled_file")
req("P2", "5", "Gated-repo errors become HF_AUTH_REQUIRED with instructions", "PLAN Part 2",
    "tests/test_detection.py::TestLoad::test_auth_error_carries_actionable_steps")
req("P2", "6", "ONNX provider selection (CPU default, CUDA when the runtime exposes it)",
    "PLAN Part 2", "tests/test_detection.py::TestConfigureHelpers::test_provider_cpu_sets_onnx_mode",
    "tests/test_detection.py::TestDeviceReporting::test_device_resolves_to_the_real_provider_after_load",
    status="met · deviation",
    note="exposed as --provider / ONNX_MODE, not as a config key, so config schema 1.0 stayed "
         "frozen (D17)")
req("P2", "7", "unload_all() frees the sessions", "PLAN Part 2",
    "tests/test_detection.py::TestUnload::test_unload_clears_sessions_and_forces_refetch",
    "tests/test_p8_lifecycle.py::test_unload_releases_the_session_even_though_rss_has_a_floor")

# -- Part 3 -----------------------------------------------------------------
req("P3", "1", "Image IO is EXIF-aware and always RGB", "PLAN Part 3",
    "tests/test_imaging.py::TestLoad::test_exif_orientation_is_applied",
    "tests/test_imaging.py::TestLoad::test_palette_and_gray_and_rgba_become_rgb")
req("P3", "2", "Crops clamp padding to the image and guard the minimum size", "PLAN Part 3",
    "tests/test_imaging.py::TestCrop::test_padding_is_clamped_to_the_image")
req("P3", "3", "Deterministic 20-colour palette", "PLAN Part 3", "tests/test_types.py::TestMisc::test_palette")
req("P3", "4", "Hayai loads lazily, pinned revision, trust_remote_code", "PLAN Part 3",
    "tests/test_p8_e2e_models.py::TestTheWeightsThatRun::test_the_bundle_carries_both_pinned_repositories",
    "tests/test_models.py::TestBundledDetection::test_vendored_ocr_snapshot_matches_the_pinned_revision")
req("P3", "5", "Device auto: CUDA gets fp16/cuda-graphs, CPU gets fp32 eager", "PLAN Part 3",
    "tests/test_ocr.py::TestLoad::test_load_reports_progress_and_ends_ready",
    "tests/test_p8_e2e_models.py::TestTheFlagsOnRealData::test_the_top_patch_budget_is_reachable")
req("P3", "6", "Batched OCR with chunking, progress and cancellation between chunks",
    "PLAN Part 3", "tests/test_ocr.py::TestOcrCrops::test_progress_is_per_chunk_and_monotonic",
    "tests/test_ocr.py::TestFakeOcrEngine::test_chunking_and_progress")
req("P3", "7", "Optional NFKC normalisation", "PLAN Part 3",
    "tests/test_imaging.py::TestNormalize::test_nfkc_fullwidth_and_ligature",
    "tests/test_imaging.py::TestNormalize::test_nfkc_can_be_disabled")
req("P3", "8", "unload() releases the model", "PLAN Part 3",
    "tests/test_engines_base.py::TestEnginesBadge::test_a_real_run_then_unload_flips_the_text")

# -- Part 4 -----------------------------------------------------------------
req("P4", "1", "detect -> order -> crop -> OCR -> assemble, GUI-independent", "PLAN Part 4",
    "tests/test_pipeline.py::TestEndToEnd::test_a_corrupt_payload_is_an_invalid_image_not_a_crash",
    "qa:three-doors")
req("P4", "2", "JobManager: states, stage+percent progress, result cache = latest result",
    "PLAN Part 4", "tests/test_jobs.py::TestListeners::test_progress_events_carry_the_stage_names_the_gui_renders")
req("P4", "3", "--headless --run prints the JSON, --out writes it, exit codes are the taxonomy",
    "PLAN Part 4", "tests/test_cli.py::TestStubs::test_headless_run_is_now_the_pipeline_not_a_stub",
    "cmd: python3 -m animetext_studio --headless --mock --image examples/sample_page.png --run")
req("P4", "4", "Golden result JSON with fake engines, schema-validated", "PLAN Part 4 accept",
    "tests/test_pipeline.py::TestDocument", "qa:determinism")
req("P4", "5", "--mock works headless with no downloads", "PLAN Part 4",
    "cmd: python3 -m animetext_studio --headless --mock --image examples/sample_page.png --run",
    "qa:three-doors")

# -- Part 5 -----------------------------------------------------------------
req("P5", "1", "Control panel + preview + output, central widget scrollable both ways",
    "PLAN Part 5", "tests/test_gui.py::TestWindow::test_the_central_widget_is_scrollable_both_ways")
req("P5", "2", "Model combo with params/F1/recommended threshold per variant", "PLAN Part 5",
    "tests/test_gui.py::TestConfigWiring::test_offline_marks_hub_only_variants_unusable",
    "tests/test_engines_base.py::TestHelpers::test_model_catalog_uses_presets")
req("P5", "3", "IoU/score sliders with spinboxes + a Recommended button", "PLAN Part 5",
    "tests/test_gui.py::TestConfigWiring::test_the_recommended_threshold_button_uses_the_preset")
req("P5", "4", "OCR group, reading order, Generate and Cancel", "PLAN Part 5",
    "tests/test_gui.py::TestRun::test_generate_with_mock_engines_produces_a_result",
    "tests/test_gui.py::TestRun::test_cancel_from_the_panel_stops_the_job")
req("P5", "5", "Config Export/Import buttons in the window", "PLAN Part 5",
    "tests/test_gui.py::TestViewConfigRoundTrip::test_export_then_import_restores_the_view_and_the_canvas")
req("P5", "6", "MCP status indicator", "PLAN Part 5",
    "tests/test_gui.py::TestWindow::test_the_mcp_indicator_starts_off_and_can_be_flipped")
req("P5", "7", "Image input: browse, drag-and-drop, Ctrl+V, preview with name and size",
    "PLAN Part 5", "tests/test_gui.py::TestImageInput::test_browse_path_sets_preview_and_caption",
    "tests/test_gui.py::TestImageInput::test_drag_and_drop_of_an_image_file_loads_it",
    "tests/test_gui.py::TestImageInput::test_paste_of_a_file_url_loads_that_file")
req("P5", "8", "QThread adapter over JobManager; buttons follow the job", "PLAN Part 5",
    "tests/test_gui.py::TestRun::test_generate_without_an_image_reports_no_image",
    "tests/test_gui.py::TestRun::test_a_failing_engine_shows_the_error_code_not_a_crash")

# -- Part 6 -----------------------------------------------------------------
req("P6", "1", "Translucent fill + solid border + label chip, over a copy of the page",
    "PLAN Part 6", "tests/test_gui.py::TestCanvas::test_the_fill_is_translucent_not_a_sticker",
    "tests/test_annotation.py::TestRender::test_the_fill_is_translucent_and_the_border_is_not")
req("P6", "2", "Hover highlights the box and shows the full text in a floating panel",
    "PLAN Part 6", "tests/test_gui.py::TestCanvas::test_hovering_shows_the_full_text_and_leaving_hides_it",
    "tests/test_annotation.py::TestText::test_the_hover_panel_carries_the_whole_text")
req("P6", "3", "Click selects a block and syncs the table and the JSON pane", "PLAN Part 6",
    "tests/test_gui.py::TestViewConfigRoundTrip::test_a_result_draws_through_the_configured_view",
    "tests/test_gui.py::TestJsonPane::test_the_copy_block_button_uses_the_current_selection")
req("P6", "4", "Overlay/label toggles, fill-opacity slider, zoom fit/percent/Ctrl+wheel",
    "PLAN Part 6", "tests/test_gui.py::TestCanvas::test_unticking_show_labels_changes_the_picture",
    "tests/test_gui.py::TestCanvas::test_the_fill_slider_reaches_the_canvas",
    "tests/test_gui.py::TestCanvas::test_ctrl_wheel_walks_the_zoom_steps")
req("P6", "5", "Save annotated PNG bakes boxes and labels at full resolution", "PLAN Part 6",
    "tests/test_gui.py::TestCanvas::test_save_annotated_writes_the_page_at_full_resolution",
    "tests/test_gui.py::TestCanvasExtras::test_the_export_carries_every_block")
req("P6", "6", "JSON pane: readable UTF-8, read-only, Copy All, Save As, per-block copy",
    "PLAN Part 6", "tests/test_gui.py::TestJsonPane::test_the_pane_holds_the_document_byte_for_byte",
    "tests/test_gui.py::TestJsonPane::test_the_pane_is_read_only",
    "tests/test_gui.py::TestJsonPane::test_save_as_writes_readable_utf8",
    "tests/test_gui.py::TestJsonPane::test_copy_block_text_and_json")
req("P6", "7", "Chips do not sit on each other, and nothing is dropped silently", "P8.6",
    "tests/test_p8_annotation.py::TestPlaceChips",
    "tests/test_p8_annotation.py::TestTheWindowShowsTheObservations")

# -- Part 7 -----------------------------------------------------------------
req("P7", "1", "Official mcp SDK (FastMCP), streamable-http + stdio + none", "PLAN Part 7",
    "tests/test_mcp_server.py::test_build_mcp_server_registers_tools_and_resources",
    "tests/test_mcp_server.py::test_transport_none_never_binds_and_reports_itself")
req("P7", "2", "Server thread + own loop; optional bearer token; loopback by default",
    "PLAN Part 7", "qa:defaults-and-auth",
    "tests/test_mcp_server.py::test_gateway_health_is_open_and_the_mcp_endpoint_is_not")
req("P7", "3", "Bridge: queued invocation, futures with timeouts, busy policy", "PLAN Part 7",
    "tests/test_mcp_bridge.py::test_qt_dispatcher_runs_the_callable_on_the_main_thread")
req("P7", "4", "Activity hook: one row per call, with args, outcome and duration", "PLAN Part 7",
    "tests/test_mcp_bridge.py::test_activity_log_numbers_and_keeps_order",
    "tests/test_mcp_gui.py::TestBridge::test_every_call_leaves_exactly_one_row_in_the_activity_dock")
req("P7", "5", "MCP-driven jobs move the GUI progress bar and send progress notifications",
    "PLAN Part 7", "tests/test_mcp_http.py::test_progress_notifications_arrive_during_a_run")
req("P7", "6", "Every §4 tool exists, with input and output schemas", "PLAN Part 7",
    "qa:tool-list", "tests/test_mcp_server.py::test_catalog_covers_the_spec_and_documents_every_tool")
req("P7", "7", "Machine-readable catalog at schemas/mcp-tools.json", "PLAN Part 7",
    "tests/test_mcp_server.py::test_write_catalog_writes_where_it_is_told_and_nowhere_else",
    "qa:mcp-doc-is-the-catalog")
req("P7", "8", "SDK demo client + raw curl transcript for non-SDK programs", "PLAN Part 7",
    "cmd: python3 -m animetext_studio.tools.mcp_client_demo --url http://127.0.0.1:8765/mcp",
    "cmd: ./scripts/mcp_curl_demo.sh 8765",
    note="re-run live for the P9.6 sign-off: 21 tools answered, every payload matched its "
         "published output schema, and the demo's re-OCR step came back with the same boxes and "
         "stage `reocr` (84 ms for one block against the page's 949 ms, real weights on CPU), "
         "exit 0")

# -- §4 deterministic-client guarantees -------------------------------------
req("MCP", "1", "Stable ids and colours for the same page and parameters", "PLAN §4",
    "qa:determinism", "tests/test_pipeline.py::TestOrdering::test_ordering_is_stable_for_identical_boxes")
req("MCP", "2", "BUSY carries the running job id; clients may poll or retry", "PLAN §4",
    "tests/test_mcp_http.py::test_a_second_run_is_busy_with_the_running_job_id")
req("MCP", "3", "Optional bearer auth; default bind is loopback", "PLAN §4", "qa:defaults-and-auth")
req("MCP", "4", "A non-loopback bind warns, louder without a token", "PLAN §4 + P8.2",
    "tests/test_p8_hardening.py::TestEndpointWarnings")
req("MCP", "5", "docs/MCP.md carries a curl-level session transcript", "PLAN §4",
    "qa:mcp-doc-is-the-catalog", "cmd: python3 scripts/mcp_http_transcript.py --port 8765")
req("MCP", "6", "Six resources answer", "PLAN §4", "tests/test_mcp_http.py::test_resources_answer",
    "qa:tool-list")
req("MCP", "7", "Malformed tool arguments are INVALID_ARGUMENT, never a broken socket", "PLAN §4",
    "tests/test_p8_errors.py::test_the_server_is_fine_after_a_storm_of_garbage")

# -- §5 CLI -----------------------------------------------------------------
req("CLI", "1", "Every flag PLAN §5 names exists and is parsed", "PLAN §5", "qa:cli-surface")
req("CLI", "2", "Precedence CLI > config file > defaults, including the MCP flags", "PLAN §5",
    "tests/test_config.py::TestPrecedence")
req("CLI", "3", "Every ErrorCode has its own exit code, and the README states the number",
    "PLAN §5 + P8.7", "qa:error-taxonomy",
    "tests/test_p8_packaging.py::TestReadmeDocumentsTheCli::test_the_exit_code_table_matches_the_taxonomy")
req("CLI", "4", "--dump-default-config / --validate-config / --export-config work standalone",
    "PLAN §5 (D9)", "tests/test_cli.py::TestFoundationActions::test_validate_bad_exit_2",
    "cmd: python3 -m animetext_studio --dump-default-config")

# -- §6 dependencies --------------------------------------------------------
req("ENV", "1", "Extras split so a headless detector install needs neither torch nor Qt",
    "PLAN §6 + D7", "qa:extras-and-floor",
    "tests/test_p8_packaging.py::TestPyprojectMetadata::test_all_extra_is_the_union_and_cannot_drift")
req("ENV", "2", "Python >= 3.10", "PLAN §6", "qa:extras-and-floor")
req("ENV", "3", "CUDA wheels for the target hardware documented (Blackwell needs cu128)",
    "PLAN §6", "qa:readme-commands",
    note="documented in README Install; the sandbox here is CPU-only, so the CUDA path is "
         "documented and reviewed, not executed")
req("ENV", "4", "A CJK-capable font is found, or the labels fall back without crashing",
    "PLAN §6", "qa:cjk-font", "tests/test_annotation.py::TestText::test_the_chip_is_id_score_and_the_first_words")
req("ENV", "5", "Real-model tests are opt-in and the default suite never touches the network",
    "PLAN §6", "qa:default-suite-is-offline",
    "tests/test_p8_e2e_models.py::TestTheWeightsThatRun::test_a_real_run_opens_no_socket_and_needs_no_token")

# -- §7 risks ---------------------------------------------------------------
req("RISK", "1", "Gated repo: HF_AUTH_REQUIRED with exact steps, GUI banner, offline fallback",
    "PLAN §7 risk 1", "tests/test_detection.py::TestLoad::test_auth_error_carries_actionable_steps",
    "tests/test_model_store.py::TestEngineUsesBundle::test_require_offline_points_at_fetch_models")
req("RISK", "2", "trust_remote_code pinned to a revision, with a real smoke test",
    "PLAN §7 risk 2",
    "tests/test_p8_e2e_models.py::TestTheWeightsThatRun::test_the_bundle_carries_both_pinned_repositories",
    "cmd: python3 scripts/fetch_models.py --verify")
req("RISK", "3", "No Qt/asyncio deadlock: bridge pattern, no Qt imports outside the window",
    "PLAN §7 risk 3", "qa:no-qt-outside-the-window",
    "tests/test_mcp_bridge.py::test_qt_dispatcher_stays_out_of_its_own_thread")
req("RISK", "4", "Tall pages: allow_dynamic is exposed and its trade-off documented",
    "PLAN §7 risk 4", "qa:readme-commands",
    "tests/test_config.py::TestValidation::test_valid_devices_and_patches")
req("RISK", "5", "CPU OCR stays usable: batching, chunked progress, patch budgets",
    "PLAN §7 risk 5", "tests/test_ocr.py::TestOcrCrops::test_progress_is_per_chunk_and_monotonic",
    "tests/test_p8_e2e_models.py::TestTheFlagsOnRealData")
req("RISK", "6", "Memory: lazy loading, unload, engine status in get_status", "PLAN §7 risk 6",
    "tests/test_p8_lifecycle.py::TestMemoryRelease",
    note="measured (D62): the objects go, RSS keeps a ~200 MB ONNX Runtime floor — the test "
         "pins that instead of a nicer fiction")
req("RISK", "7", "Determinism for harnesses: stable order/colours, volatile fields documented",
    "PLAN §7 risk 7", "qa:determinism")
req("RISK", "8", "Licensing stated honestly: Apache-2.0 code, GPL-3.0 weights, slim wheel option",
    "PLAN §7 risk 8", "tests/test_p8_packaging.py::TestLicence",
    status="met · measured",
    note="the distributor chose Apache-2.0 on 2026-11-24; LICENSE + NOTICE ship in the wheel, "
         "the GPL text ships beside the weights")

# -- what Part 8 added ------------------------------------------------------
req("P8", "1", "stage: detect|full distinguishes a stage-1 document (result 1.1)", "P8.1",
    "tests/test_pipeline.py::TestStageMarker", "tests/test_schemas.py::TestResultSchemaVersions")
req("P8", "2", "Log capacity is a live config key; exports can redact secrets", "P8.2",
    "tests/test_p8_hardening.py::TestLogCapacity", "tests/test_p8_hardening.py::TestRedaction")
req("P8", "3", "Cheap correctness flags: possible_truncation and contained_in (result 1.2)",
    "P8.2b", "tests/test_pipeline.py::TestDocumentAnnotations",
    "tests/test_schemas.py::TestResultSchemaVersions::test_1_2_added_flags_and_contained_in_to_every_block")
req("P8", "4", "Warm means loaded plus one inference, and it is a cancellable job", "P8.3",
    "tests/test_p8_lifecycle.py::TestWarmEngines",
    "tests/test_p8_e2e_models.py::TestTheMcpServiceDoor::test_a_real_warm_up_is_a_job_and_not_a_page")
req("P8", "5", "Engine warmth is reported, not hoped for", "P8.3",
    "tests/test_p8_lifecycle.py::TestWarmEngines::test_it_loads_both_engines_and_runs_one_inference_through_each")
req("P8", "6", "The error matrix: named faults × 8 doors, generated into docs", "P8.4",
    "tests/test_p8_errors.py", "cmd: python3 scripts/gen_error_matrix.py --check",
    note="84 rows at the P8.8 sign-off; 91 since P9.6 folded in the re-OCR door, the corrupt-"
         "bundle diagnosis and the empty-option refusals. The count lives in the sign-off line "
         "below, not in a row that would then go stale.")
req("P8", "7", "Real weights through every door, offline, in one opt-in suite", "P8.5",
    "tests/test_p8_e2e_models.py", "cmd: python3 -m pytest -q -m models")
req("P8", "8", "One de-collision rule shared by the canvas and the export", "P8.6",
    "tests/test_p8_annotation.py::TestPlaceChips")
req("P8", "9", "README documents the shipped surface, and a test says so", "P8.7",
    "tests/test_p8_packaging.py", "qa:readme-commands")
req("P8", "10", "The on-premises bundle is the install (D26): verify it, offline", "P8 / D26",
    "cmd: python3 scripts/fetch_models.py --verify", "tests/test_model_store.py")

# -- Part 9: the five approved enhancements ----------------------------------
# Each of these was a line in `enhancements_proposition.md`, approved as a set. The rows are the
# argument that the shipped thing is the thing that was proposed — including the two defects the
# sign-off run found on the way (D93, D94), which are rows here rather than footnotes.
req("P9", "1", "Weights that are present and wrong are diagnosed as that, not as absent",
    "P9.1 / backlog 7",
    "tests/test_model_store.py::TestOpenFailureDiagnosis::test_corrupt_weights_offline_name_the_file_and_the_fix",
    "tests/test_model_store.py::TestOpenFailureDiagnosis::test_intact_weights_blame_the_runtime_not_the_file",
    "tests/test_model_store.py::TestOpenFailureDiagnosis::test_a_non_load_error_is_not_rescued",
    "tests/test_p8_errors.py::test_each_matrix_row_carries_the_code_the_matrix_promises")
req("P9", "2", "The hash is a failure-path tool: the hot paths never pay for it", "P9.1",
    "tests/test_model_store.py::TestVerifyFile::test_hot_paths_stay_hash_free",
    "tests/test_model_store.py::TestVerifyFile::test_truncated_file_is_a_mismatch_with_both_hashes")
req("P9", "3", "Online, corrupt weights are re-vendored once — and the retry is recorded, "
                "offline it is refused and says that too", "P9.1",
    "tests/test_model_store.py::TestOpenFailureDiagnosis::test_corrupt_weights_online_retry_once_and_say_they_did",
    "tests/test_model_store.py::TestOpenFailureDiagnosis::test_a_failed_retry_keeps_the_diagnosis_instead_of_hiding_it",
    "tests/test_model_store.py::TestOpenFailureDiagnosis::test_offline_never_reaches_for_the_network")
req("P9", "4", "One field says where the model list came from; every door reads that field (D90)",
    "P9.2 / backlog 8",
    "tests/test_engines_base.py::TestModelsBadge::test_the_flag_is_on_the_wire_too",
    "tests/test_mcp_service.py::test_status_carries_the_reason_the_list_degraded",
    "tests/test_mcp_service.py::test_an_ordinary_engine_reports_no_fallback",
    "cmd: animetext-studio --model-status")
req("P9", "5", "The window says it too: the fifth badge, and Refresh models", "P9.2",
    "tests/test_gui.py::TestModelsBadge::test_the_badge_appears_with_the_reason_behind_it",
    "tests/test_gui.py::TestModelsBadge::test_answering_empty_hides_it_rather_than_blanking_it",
    "tests/test_engines_base.py::TestModelsBadge::test_a_live_list_shows_nothing")
req("P9", "6", "Re-OCR one block without asking the detector again — the 21st tool", "P9.3",
    "tests/test_pipeline.py::TestReocrBlocks::test_the_detector_is_not_asked_twice",
    "tests/test_pipeline.py::TestReocrBlocks::test_only_the_selected_crop_is_sent",
    "tests/test_mcp_server.py::test_catalog_covers_the_spec_and_documents_every_tool")
req("P9", "7", "A re-read answers with a whole document, not a diff (D81)", "P9.3",
    "tests/test_pipeline.py::TestReocrBlocks::test_the_document_is_whole_and_unselected_blocks_keep_their_text",
    "tests/test_pipeline.py::TestReocrBlocks::test_the_new_document_validates_and_says_how_it_was_made",
    "tests/test_pipeline.py::TestReocrBlocks::test_ids_colours_order_and_boxes_survive")
req("P9", "8", "The re-OCR door refuses in the order a caller can use", "P9.3",
    "tests/test_p8_errors.py::TestReocrDoor::test_an_unknown_id_names_the_ones_that_exist",
    "tests/test_p8_errors.py::TestReocrDoor::test_with_no_page_loaded_the_answer_is_no_image",
    "tests/test_pipeline.py::TestReocrBlocks::test_cancelling_a_re_read_is_a_cancellation_not_a_broken_one")
req("P9", "9", "The block table can ask for it: right-click → Re-read this block", "P9.3",
    "tests/test_gui.py::TestReocrFromTheTable::test_re_reading_a_block_replaces_the_document_with_a_reocr_one",
    "tests/test_gui.py::TestReocrFromTheTable::test_the_menu_is_wired_to_the_table",
    "tests/test_gui.py::TestReocrFromTheTable::test_no_selection_means_no_job")
req("P9", "10", "Per-device cold latency, measured on the machine you are on (D84)", "P9.4",
    "tests/test_p8_lifecycle.py::TestDeviceSweep::test_the_device_you_asked_for_survives_next_to_the_one_you_got",
    "tests/test_p8_lifecycle.py::TestDeviceSweep::test_a_failing_engine_is_a_row_not_a_crash",
    "tests/test_cli.py::TestWarmDevicesFlag::test_the_sweep_prints_a_table_and_exits_zero",
    "cmd: animetext-studio --warm-devices cpu")
req("P9", "11", "`--preload-off` names the cold baseline a benchmark measured against", "P9.4",
    "tests/test_cli.py::TestWarmDevicesFlag::test_preload_off_is_the_cold_baseline_even_read_as_a_flag",
    "tests/test_cli.py::TestWarmDevicesFlag::test_preload_and_preload_off_are_mutually_exclusive",
    "tests/test_p8_lifecycle.py::TestDeviceSweep::test_each_pair_is_released_so_the_next_row_starts_cold")
req("P9", "12", "`--doctor`: read-only, never installs, and it exits with the code a run would "
                "have hit (D88, D89)", "P9.5",
    "tests/test_p9_doctor.py::TestTheDoctorNeverDoesTheThingItIsDiagnosing::test_a_read_only_bundle_directory_is_still_reportable",
    "tests/test_p9_doctor.py::TestTheCommand::test_a_blocking_machine_exits_with_the_code_it_predicts",
    "cmd: animetext-studio --doctor")
req("P9", "13", "The two upstream traps are written up as files, and pinned to a test that goes "
                "red when upstream fixes them (D92)", "P9.5",
    "tests/test_p9_doctor.py::TestTheUpstreamDrafts",
    "tests/test_p9_doctor.py::TestPython::test_3_14_names_the_upstream_cause_and_the_draft")
req("P9", "14", "An option that was given an empty value is refused, not ignored (D93)",
    "P9.6 finding",
    "tests/test_cli.py::TestEmptyOptionValues",
    "tests/test_p8_errors.py::test_the_cli_answers_with_the_matrix_code")
req("P9", "15", "The catalog generator writes where it is told, and nowhere else (D94)",
    "P9.6 finding",
    "tests/test_mcp_server.py::test_write_catalog_writes_where_it_is_told_and_nowhere_else",
    "tests/test_cli.py::TestDumpMcpTools",
    "tests/test_mcp_server.py::test_catalog_is_the_file_that_ships_in_the_wheel")
req("P9", "16", "The trail moved with the code: catalog, matrix, checklist, README, floors", "P9.6",
    "cmd: python3 scripts/gen_error_matrix.py --check",
    "cmd: python3 scripts/gen_qa_report.py --check",
    "tests/test_p8_packaging.py", "tests/test_mcp_server.py::test_catalog_is_the_file_that_ships_in_the_wheel")

# The publication boundary. Not in PLAN §3 — it arrived with the repository (2026-11-26), and
# D26 had to be amended (D95) before it could be stated honestly.
req("PUB", "1", "The repository publishes only what the project may publish: no weights, no "
                "scans, no machine paths, nothing over GitHub's 100 MB limit (D95)", "D95",
    "tests/test_p8_packaging.py::TestWhatTheRepositoryPublishes",
    "tests/test_p8_qa.py::TestQaChecks::test_no_secret_travels_in_the_tree",
    "cmd: python3 scripts/fetch_models.py --verify")

REQUIREMENTS: tuple[Requirement, ...] = tuple(R)

#: Not requirements, and not defects: what is deliberately still open, with the reason. The
#: report prints these so "open" is always an argument, never an omission.
FOLLOW_UPS: tuple[tuple[str, str], ...] = (
    ("File the two `dghs-imgutils` issues (import-time pip install; `ast.Str` on 3.14)",
     "drafted and pinned in docs/UPSTREAM.md (D92). Filing needs an account on someone else's "
     "repository, and only a classic token can write to one — a fine-grained token, which is "
     "what this project was offered, cannot (GitHub's own limitation). Pre-flight on "
     "2026-11-26 also found that Issue 2 is already fixed upstream in "
     "unmerged PR #178, so the artefact there is a release request, not a bug report. "
     "`TestTheUpstreamDrafts` goes red if the quoted source moves underneath them"),
    ("`scripts/bench_models.py` — the benchmark harness ADDENDUM item E asked for",
     "`--warm-devices` is the measurement primitive it needed (D84); the harness is the part "
     "that runs a corpus through variants and devices and keeps the numbers. Nothing in the "
     "plan requires it, so it stays a proposal, not a row"),
    ("Block colours that differ between two runs of the same page",
     "identity, not correctness: the chips and the table key on `id`, which is stable (D32). "
     "A re-read (`reocr_blocks`) keeps the colours on purpose, which is as far as this goes"),
    ("Language policy, sub-crop splitting, nested-box de-duplication",
     "decided out of scope (D58, D60): the caller owns policy, the app reports — do not "
     "re-propose these as bugs"),
    ("Warm-up reports, the device sweep and `--doctor` stay out of the MCP surface",
     "decided, not deferred (D83, D88): a client that wants to know how warm the machine is can "
     "read `engines[]`; a client that wants to benchmark someone else's GPU should not be "
     "paying for it over a tool call"),
)


# ---------------------------------------------------------------------------
# The rules the table lives by
# ---------------------------------------------------------------------------

def _test_index() -> dict[str, set[str]]:
    """`{test_file.py: {TestThing::test_name, test_name, ...}}`, read from the AST.

    Cheaper and steadier than spawning pytest: an evidence row may name a file, a file + class,
    or a file + class + test (a parametrized family names its base function, which is what the
    AST holds).
    """
    index: dict[str, set[str]] = {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        names: set[str] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names.add(node.name)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(f"{node.name}::{sub.name}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
        index[path.name] = names
    return index


def _app_flags() -> set[str]:
    return {opt for action in build_parser()._actions for opt in action.option_strings}


def _script_flags(script: Path) -> set[str]:
    return set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', script.read_text(encoding="utf-8")))


def _split(cmd: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"\s+", cmd.replace("\\\n", " ").strip()) if part)


APP_COMMANDS = ("python3 -m animetext_studio", "python -m animetext_studio", "animetext-studio")


class TestTheTableItself:
    """A checklist you cannot check is a list of wishes."""

    def test_ids_are_unique_and_grouped(self):
        ids = [r.id for r in REQUIREMENTS]
        assert len(ids) == len(set(ids)), "duplicate requirement ids"
        known = {key for key, _ in GROUPS}
        assert sorted({r.id.split("-")[0] for r in REQUIREMENTS}) == sorted(known & {
            r.id.split("-")[0] for r in REQUIREMENTS}), "a row lives in a group nobody prints"

    def test_every_row_has_a_source_and_evidence(self):
        broken = [r.id for r in REQUIREMENTS if not r.source or not r.evidence]
        assert broken == [], f"rows with nothing behind them: {broken}"

    def test_nothing_is_left_unmet(self):
        unmet = [(r.id, r.status) for r in REQUIREMENTS if r.status not in MET_STATUSES]
        assert unmet == [], f"statuses nobody has argued for: {unmet}"

    def test_evidence_points_at_tests_that_exist(self):
        index = _test_index()
        missing = []
        for row in REQUIREMENTS:
            for cite in row.evidence:
                if not cite.startswith("tests/"):
                    continue
                parts = cite.split("::")
                name = Path(parts[0]).name
                if name not in index:
                    missing.append(f"{row.id}: {cite} (no such test file)")
                    continue
                if len(parts) > 1 and parts[1] not in index[name]:
                    missing.append(f"{row.id}: {cite} (not collected from {name})")
        assert missing == [], "evidence that does not exist is worse than no evidence:\n  " + \
            "\n  ".join(missing)

    def test_evidence_points_at_commands_that_are_real(self):
        flags = _app_flags()
        problems = []
        for row in REQUIREMENTS:
            for cite in row.evidence:
                if not cite.startswith("cmd:"):
                    continue
                cmd = cite[4:].strip()
                parts = _split(cmd)
                joined = " ".join(parts)
                if joined.startswith("pytest") or "-m pytest" in joined:
                    continue
                script = next((p for p in parts if p.endswith(".py") or p.endswith(".sh")), "")
                if script and not (ROOT / script.lstrip("./")).is_file():
                    problems.append(f"{row.id}: {cmd} (no such script: {script})")
                    continue
                is_app_cli = ("-m animetext_studio" in joined and ".tools." not in joined) \
                    or joined.startswith("animetext-studio ")
                if is_app_cli:
                    for opt in (p for p in parts if p.startswith("--")):
                        base = opt.split("=")[0]
                        if base not in flags:
                            problems.append(f"{row.id}: {cmd} (unknown flag {base})")
                elif script.endswith(".py") and "animetext_studio" not in script:
                    available = _script_flags(ROOT / script.lstrip("./"))
                    for opt in (p for p in parts if p.startswith("--")):
                        if opt.split("=")[0] not in available:
                            problems.append(
                                f"{row.id}: {cmd} ({script} has no {opt.split('=')[0]})")
        assert problems == [], "\n  ".join(problems)

    def test_human_review_is_named_and_rare(self):
        manual = [(r.id, cite) for r in REQUIREMENTS
                  for cite in r.evidence if cite.startswith("manual:")]
        assert len(manual) <= 3, f"too much of the checklist rests on eyes: {manual}"
        unnamed = [rid for rid, cite in manual if "turn" not in cite.lower()]
        assert unnamed == [], f"an eyes-only row must name the review: {unnamed}"

    def test_the_open_items_are_argument_not_silence(self):
        assert FOLLOW_UPS, "an empty open list is how an omission becomes invisible"
        assert all(reason.strip() for _, reason in FOLLOW_UPS)


# ---------------------------------------------------------------------------
# The cross-cutting checks the table cites as `qa:`
# ---------------------------------------------------------------------------

def _strip_volatile(doc: dict) -> dict:
    out = {k: v for k, v in doc.items() if k not in ("created_at", "timing_ms")}
    out["image"] = {k: v for k, v in (doc.get("image") or {}).items() if k != "source"}
    return out


def _service_door_document() -> tuple[dict, dict]:
    """The document the MCP tools produce, without a server or a display."""
    from types import SimpleNamespace

    from animetext_studio.core.jobs import JobManager
    from animetext_studio.core.pipeline import Pipeline
    from animetext_studio.engines.base import FakeDetectionEngine, FakeOcrEngine
    from animetext_studio.mcp.service import StudioService

    config = AppConfig.default()
    manager = JobManager(Pipeline(FakeDetectionEngine(), FakeOcrEngine(), config))
    service = StudioService(manager, config, mock=True)
    service.load_image(path=str(ROOT / "examples" / "sample_page.png"))
    service.start_job(ocr_enabled=True)
    service.wait_for_job()
    return service.result_payload()["result"], service.snapshot()


def _cli_door_document() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "animetext_studio", "--headless", "--mock",
         "--image", "examples/sample_page.png", "--run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
    assert proc.returncode == 0, proc.stderr[-800:]
    return json.loads(proc.stdout)


class TestQaChecks:
    def test_plan_cli_surface_is_the_shipped_surface(self):
        """qa:cli-surface — every flag PLAN §5 names exists, with the parser behind it."""
        block = PLAN.read_text(encoding="utf-8").split("## 5. CLI specification", 1)[1]
        block = block.split("```", 2)[1]
        planned = sorted(set(re.findall(r"--[a-z][a-z0-9-]*", block)))
        shipped = _app_flags()
        assert planned, "PLAN §5 has no flags to check"
        missing = [flag for flag in planned if flag not in shipped]
        assert missing == [], f"PLAN §5 promises flags nobody parses: {missing}"

    def test_plan_tools_and_resources_all_ship(self):
        """qa:tool-list — §4's 18 tools, plus the two Part 8 added, plus the 6 resources."""
        section = PLAN.read_text(encoding="utf-8").split("### Tools", 1)[1]
        section = section.split("### Resources", 1)[0]
        planned = re.findall(r"^\| `([a-z_]+)` \|", section, re.M)
        assert len(planned) == 18, f"expected §4's 18 tools, found {len(planned)}"
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        shipped = {tool["name"] for tool in catalog["tools"]}
        assert set(planned) <= shipped, f"missing tools: {sorted(set(planned) - shipped)}"
        assert len(shipped) == SIGNOFF["tools"], (
            f"{len(shipped)} tools published; update SIGNOFF and the README with a reason")
        uris = {resource["uri"] for resource in catalog["resources"]}
        assert len(uris) == SIGNOFF["resources"]
        assert {"config://current", "result://latest", "image://current",
                "image://annotated", "models://list", "log://mcp"} == uris
        for tool in catalog["tools"]:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", tool["name"]), (
                f"§4 convention is snake_case, got {tool['name']}")
            assert tool["inputSchema"].get("type") == "object", tool["name"]
            assert tool["outputSchema"], f"{tool['name']} publishes no output schema"

    def test_the_documented_contracts_are_the_shipped_contracts(self):
        """qa:document-contracts — §2.1, §2.2 and §2.3, key by key, against the real thing."""
        doc = _cli_door_document()
        for key in ("schema_version", "created_at", "image", "detection", "ocr",
                    "timing_ms", "blocks", "block_count"):
            assert key in doc, f"result document lost §2.1 key {key}"
        for key in ("source", "file_name", "width", "height", "sha256"):
            assert key in doc["image"], f"image lost {key}"
        for key in ("repo", "model", "iou_threshold", "score_threshold", "allow_dynamic",
                    "backend"):
            assert key in doc["detection"], f"detection lost {key}"
        for key in ("repo", "device", "max_patches", "batch_size", "max_new_tokens",
                    "crop_padding_px", "normalize_nfkc"):
            assert key in doc["ocr"], f"ocr lost {key}"
        block = doc["blocks"][0]
        for key in ("id", "bbox", "label", "score", "text", "reading_order", "color", "error"):
            assert key in block, f"block lost {key}"
        assert set(block["bbox"]) == {"x0", "y0", "x1", "y1"}
        assert all(isinstance(v, int) for v in block["bbox"].values()), "bbox must be integer pixels"

        # §2.2: every key the plan's config example names is still a real config key.
        snippet = PLAN.read_text(encoding="utf-8").split("### 2.2 Config JSON", 1)[1]
        snippet = snippet.split("```", 2)[1]
        planned = set(re.findall(r'"([a-z_]+)":', snippet))
        from animetext_studio.core.config import _flatten
        live = set(_flatten(AppConfig.default().to_dict()))
        leaf_names = {path.split(".")[-1] for path in live}
        sections = {path.split(".")[0] for path in live}
        lost = {k for k in planned if k not in leaf_names | sections}
        assert lost <= {"json", "path"}, f"config document lost {lost}"

    def test_nothing_outside_the_window_needs_qt(self):
        """qa:no-qt-outside-the-window — PLAN §7 risk 3, proven by import, not by grep."""
        probe = (
            "import importlib, pkgutil, sys\n"
            "sys.modules['PySide6'] = None\n"          # any Qt import now raises ImportError
            "import animetext_studio\n"
            "bad = []\n"
            "for mod in pkgutil.walk_packages(animetext_studio.__path__, 'animetext_studio.'):\n"
            "    if '.gui' in mod.name: continue\n"
            "    try: importlib.import_module(mod.name)\n"
            "    except ImportError as exc: bad.append(mod.name + ': ' + str(exc))\n"
            "print('BLOCKED=' + '|'.join(bad))\n"
        )
        proc = subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=300)
        line = [l for l in proc.stdout.splitlines() if l.startswith("BLOCKED=")]
        assert line, f"the probe did not run: {proc.stderr[-400:]}"
        blocked = line[0][len("BLOCKED="):]
        assert blocked == "", f"modules that cannot be imported without Qt:\n  {blocked}"

    def test_the_three_doors_produce_the_same_document(self):
        """qa:three-doors — the GUI's pipeline, the CLI and the MCP service are one pipeline."""
        cli_doc = _cli_door_document()
        service_doc, snapshot = _service_door_document()
        assert _strip_volatile(cli_doc) == _strip_volatile(service_doc), (
            "the CLI door and the service door disagree about the same page and config")
        assert cli_doc["image"]["source"] == "file"
        assert service_doc["image"]["source"] == "mcp-path", (
            "the provenance marker is the one field allowed to differ — and it must say which "
            "door the page came in through")
        assert cli_doc["block_count"] == service_doc["block_count"] > 0
        assert [b["text"] for b in cli_doc["blocks"]] == [b["text"] for b in service_doc["blocks"]]
        assert snapshot["app"] == "animetext-studio"

    def test_two_runs_differ_only_in_the_volatile_fields(self):
        """qa:determinism — the §2.1 promise, on the same page in the same process."""
        first, second = _cli_door_document(), _cli_door_document()
        volatile = {k for k in set(first) | set(second) if first.get(k) != second.get(k)}
        assert volatile <= {"created_at", "timing_ms"}, (
            f"non-deterministic fields nobody documented: {sorted(volatile)}")
        assert first["blocks"] == second["blocks"], "boxes, colours, ids or text moved"

    def test_the_error_taxonomy_is_one_taxonomy(self):
        """qa:error-taxonomy — §2.3's list, the codes, the exit numbers, the published catalog."""
        section = PLAN.read_text(encoding="utf-8").split("### 2.3", 1)[1]
        # §2.3 keeps its codes inside one backticked run; take that, not the heading
        # (whose "shared by GUI, CLI, MCP" is prose, not a code).
        chunk = "NO_IMAGE" + section.split("NO_IMAGE", 1)[1].split("`", 1)[0]
        planned = set(re.findall(r"[A-Z][A-Z_]{3,}", chunk))
        assert len(planned) >= 12, f"could not read §2.3's code list: {sorted(planned)}"
        codes = {code.value for code in ErrorCode}
        assert len(codes) == SIGNOFF["error_codes"] == len(EXIT_CODES)
        assert codes == {code.value for code in EXIT_CODES}, "a code without an exit number"
        # §2.3 promises a *fixed* number per code, not a unique one: INVALID_CONFIG and
        # INVALID_ARGUMENT share 2, which is also argparse's own usage-error number, so a shell
        # sees one "you asked wrong" outcome either way. Any other sharing would be a bug.
        shared = {number for number in EXIT_CODES.values()
                  if sum(1 for value in EXIT_CODES.values() if value == number) > 1}
        assert shared == {2}, f"exit numbers shared by accident: {sorted(shared)}"
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        assert set(catalog["errorCodes"]) == codes
        readme = README.read_text(encoding="utf-8")
        assert all(f"`{code.value}`" in readme for code in ErrorCode), \
            "the README must state every code (test_p8_packaging pins the numbers)"
        assert planned <= codes | {"SKIPPED"}

    def test_the_defaults_are_loopback_and_auth_is_off_by_default(self):
        """qa:defaults-and-auth — §4's safety defaults, read from the shipped document."""
        mcp = AppConfig.default().mcp
        assert mcp.host in ("127.0.0.1", "localhost"), mcp.host
        assert mcp.port == 8765
        assert mcp.auth_token is None, "a default token is not a secret"
        assert mcp.transport == "streamable-http"
        from animetext_studio.mcp.server import security_warnings
        assert security_warnings(mcp) == [], "the default bind must not warn"

    def test_docs_mcp_md_matches_the_catalog(self):
        """qa:mcp-doc-is-the-catalog — the prose door and the machine door say the same 20."""
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        text = (DOCS / "MCP.md").read_text(encoding="utf-8")
        names = [tool["name"] for tool in catalog["tools"]]
        missing = [name for name in names if f"`{name}`" not in text]
        assert missing == [], f"tools docs/MCP.md never explains: {missing}"
        headings = set(re.findall(r"^#{3,4} `([a-z_]+)`", text, re.M))
        assert headings <= set(names), f"docs/MCP.md documents tools that do not exist: " \
                                       f"{sorted(headings - set(names))}"
        for resource in catalog["resources"]:
            assert resource["uri"] in text, resource["uri"]
        assert __version__ in text or "0.1.0" in text, "the doc never says which version it describes"

    def test_readme_commands_are_runnable(self):
        """qa:readme-commands — a README command that errors is a broken front door."""
        flags = _app_flags()
        text = README.read_text(encoding="utf-8")
        problems = []
        for line in text.splitlines():
            stripped = line.strip().lstrip("$ ").strip()
            if not any(stripped.startswith(app) for app in APP_COMMANDS):
                continue
            parts = _split(stripped)
            for opt in (p for p in parts if p.startswith("--")):
                if opt.split("=")[0] not in flags:
                    problems.append(f"{stripped[:70]}  ->  unknown flag {opt.split('=')[0]}")
        assert problems == [], "README commands that would fail:\n  " + "\n  ".join(problems)

    def test_extras_and_the_python_floor(self):
        """qa:extras-and-floor — §6's install story, as the metadata states it."""
        import tomllib
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = data["project"]["optional-dependencies"]
        assert {"gui", "detect", "ocr", "mcp", "dev", "all"} <= set(extras)
        assert data["project"]["requires-python"] == ">=3.10"
        assert data["project"]["license"] == "Apache-2.0"

    def test_a_cjk_font_is_found_or_the_fallback_is_used(self):
        """qa:cjk-font — labels and hover panels must render Japanese, or say so quietly."""
        from animetext_studio.core.annotation import FONT_CANDIDATES, load_font
        assert FONT_CANDIDATES, "no font candidates at all"
        font = load_font(14)
        assert font is not None, "load_font must always hand back something drawable"

    def test_the_default_suite_never_names_the_network(self):
        """qa:default-suite-is-offline — the rule is only real if a file cannot drift into it."""
        offenders = []
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            # git-lfs.github.com appears inside a fake git-lfs pointer file (a data literal the
            # store parses, never a request); everything else remote must sit behind the marker.
            allowed = r"https?://(127\.0\.0\.1|localhost|example\.(com|org)|" \
                      r"www\.(apache|gnu|huggingface)\.org|git-lfs\.github\.com)"
            urls = {u for u in re.findall(r"https?://[^\s\"'`)]+", text)
                    if not re.match(allowed, u)}
            if urls and "pytest.mark.models" not in text:
                offenders.append(f"{path.name}: {sorted(urls)[:3]}")
        assert offenders == [], (
            "default tests that reach out must be marked models:\n  " + "\n  ".join(offenders))

    def test_restart_keys_are_documented(self):
        """qa:restart-keys-documented — the fix is only honest if the doors say it too."""
        from animetext_studio.core.config import RESTART_REQUIRED_KEYS
        readme = README.read_text(encoding="utf-8")
        mcp_doc = (DOCS / "MCP.md").read_text(encoding="utf-8")
        assert "restart_required" in readme and "restart_required" in mcp_doc
        for key in RESTART_REQUIRED_KEYS:
            assert key in mcp_doc, f"docs/MCP.md does not name {key}"
        assert "restart" in readme.split("## Configuration reference", 1)[1]

    #: Every token shape this project is ever handed, by the services that issue them.
    #: A rule that only knows one provider's prefix is a rule that fails quietly the first
    #: time the other one is pasted in — which is the exact mistake this test exists to catch.
    TOKEN_PATTERNS = {
        "Hugging Face": rb"hf_[A-Za-z0-9]{20,}",
        "GitHub classic PAT": rb"gh[pousr]_[A-Za-z0-9]{30,}",
        "GitHub fine-grained PAT": rb"github_pat_[A-Za-z0-9_]{30,}",
    }

    def test_no_secret_travels_in_the_tree(self):
        """A token that lands in a doc, a log or an example is the one mistake you cannot undo."""
        skip = {".venv", ".cache", ".unsloth-packages", "unsloth-tmp", ".npm", ".mcp-inspector",
                "sample_scans", "uploads", "__pycache__", ".probe", "node_modules", ".git"}
        token_files = {"HF_token.txt", "GH_token.txt"}
        hits = []
        for path in ROOT.rglob("*"):
            if any(part in skip for part in path.parts) or not path.is_file():
                continue
            if path.name in token_files:
                continue
            if path.stat().st_size > 2_000_000:
                continue
            body = path.read_bytes()
            for who, pattern in self.TOKEN_PATTERNS.items():
                if re.search(pattern, body):
                    hits.append(f"{path.relative_to(ROOT)} ({who})")
        assert hits == [], f"token lookalikes in the workspace: {hits}"
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for name in sorted(token_files):
            assert name in gitignore, f"{name} must never be committable"

    def test_the_generated_documents_are_current(self):
        """Both generated docs are rebuilt from their source table and compared, in CI terms."""
        for script in ("scripts/gen_error_matrix.py", "scripts/gen_qa_report.py"):
            proc = subprocess.run([sys.executable, script, "--check"], cwd=str(ROOT),
                                  capture_output=True, text=True, timeout=300,
                                  env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
            assert proc.returncode == 0, f"{script} --check:\n{proc.stdout}{proc.stderr}"

    def test_the_suite_has_not_shrunk_since_sign_off(self, request):
        """Floors, not ceilings: evidence for a requirement must not quietly disappear."""
        count = len(request.session.items)
        targets = list(getattr(request.config.option, "file_or_dir", None) or [])
        marked = [item for item in request.session.items
                  if item.get_closest_marker("models") is not None]
        if marked:
            pytest.skip("this is the `-m models` run; the floor is recorded for the default run")
        if targets and targets != ["tests"]:
            pytest.skip(f"a partial run ({targets}) cannot be measured against the floor")
        assert count >= SIGNOFF["offline_tests"], (
            f"{count} tests collected, sign-off recorded {SIGNOFF['offline_tests']}: a suite "
            "that shrinks has lost someone's evidence")



