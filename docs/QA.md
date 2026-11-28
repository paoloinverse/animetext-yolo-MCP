# Final QA — every original requirement, and what proves it

This is PLAN §3 Part 8's last deliverable: *"Final QA checklist against every original
requirement"*. It is generated from `tests/test_p8_qa.py`, so it cannot drift from the suite:

```bash
python3 scripts/gen_qa_report.py          # rewrite this file from the table
python3 scripts/gen_qa_report.py --check  # CI: fail if the two disagree
```

**How to read a row.** Every requirement cites evidence: a test node id, a command, or a check
defined in `tests/test_p8_qa.py` itself. Four rules keep the list honest, and all four are tests:
a row may not be empty (`test_every_requirement_has_evidence`), a cited test must exist in the
suite (`test_evidence_points_at_tests_that_exist`, resolved against the test files' ASTs, so a
rename fails the suite instead of turning a row into fiction), a cited command must exist with
those flags against the real parser, and no row may be left un-met. The counts at the top are
floors, not targets: a suite that shrinks has lost someone's evidence.

Anything not done sits under [Still open, on purpose](#still-open-on-purpose) with the argument
for why it is not a requirement — "open" is a decision here, never an omission.

**Status vocabulary.** ✅ means met. *met · deviation* means the shipped behaviour satisfies the
intent but differs from the letter of the plan, and the note says how and why. *met · measured*
means the claim is a number someone actually took.

**What QA found on the way** (all fixed, each with a test):

1. **`restart_required`** — PLAN §2.2 promised that changing `mcp.host`/`mcp.port` at runtime
   restarts the server thread. Nothing ever re-bound the socket, and no door said so: a client
   that set the port got `changed: ["mcp.port"]` back and had every reason to believe the
   endpoint had moved. The restart stays unbuilt on purpose — a client must not be able to move
   the socket it is talking to, mid-session — but the disagreement is now impossible to hold
   silently: both config tools answer with `restart_required`, the GUI says it in the status bar
   on import, and the README and `docs/MCP.md` list the five keys.
   `tests/test_p8_hardening.py::TestRestartRequired`
2. **`docs/MCP.md` said the config document was v1.1.** It has been 1.2 since 8.2b. Fixed, and
   `test_docs_mcp_md_matches_the_catalog` now pins the doc against the published catalog.
3. **Licensing was an open question.** The code is now Apache-2.0 (`LICENSE`, `NOTICE`,
   `pyproject`), with the GPL-3.0 text shipped beside the weights it covers and the slim-wheel
   route documented for distributions that cannot carry them.
   `tests/test_p8_packaging.py::TestLicence`

## The checklist

132 requirements, each with the evidence that proves it. A row cites a test node id, a command, or a check in `tests/test_p8_qa.py`; `TestTheTableItself::test_evidence_points_at_tests_that_exist` fails the suite if one of those stops existing.

Suite at sign-off (2026-11-26): **1260 offline tests** (+40 `-m models`), 21 MCP tools, 6 resources, 14 error codes, 91 matrix faults. `test_the_suite_has_not_shrunk_since_sign_off` treats those as floors.

| Group | Requirements |
|---|---|
| [GOAL](#goal) — The product the goal paragraph asked for | 7 |
| [ARCH](#arch) — Architecture and threading (§1) | 7 |
| [DOC](#doc) — The result document (§2.1) | 8 |
| [CFG](#cfg) — The config document (§2.2) | 7 |
| [ERR](#err) — The error taxonomy (§2.3) | 5 |
| [P1](#p1) — Part 1 — foundation | 4 |
| [P2](#p2) — Part 2 — detection engine | 7 |
| [P3](#p3) — Part 3 — OCR engine + imaging | 8 |
| [P4](#p4) — Part 4 — pipeline, jobs, headless | 5 |
| [P5](#p5) — Part 5 — GUI shell | 8 |
| [P6](#p6) — Part 6 — visualisation + JSON pane | 7 |
| [P7](#p7) — Part 7 — MCP server + GUI bridge | 8 |
| [MCP](#mcp) — Deterministic-client guarantees (§4) | 7 |
| [CLI](#cli) — CLI surface (§5) | 4 |
| [ENV](#env) — Dependencies and environment (§6) | 5 |
| [RISK](#risk) — Risks and mitigations (§7) | 8 |
| [P8](#p8) — What Part 8 added | 10 |
| [P9](#p9) — What Part 9 added (the five approved enhancements, D80–D94) | 16 |
| [PUB](#pub) — The publication boundary (D95, amending D26) | 1 |
| **Total** | **132** |

### GOAL — The product the goal paragraph asked for

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `GOAL-1` | Python desktop app with a PySide6 window | PLAN §Goal | ✅ | `test_gui.py` `TestWindow::test_the_central_widget_is_scrollable_both_ways`<br>`python3 -m animetext_studio --gui-smoke --mock` |
| `GOAL-2` | Detects anime/manga text blocks with deepghs/AnimeText_yolo YOLO12 | PLAN §Goal | ✅ | `test_models.py` `TestRealDetection::test_synthetic_manga_page_yields_text_blocks`<br>`test_detection.py` `TestListModels::test_repo_list_with_presets` |
| `GOAL-3` | OCRs every detected crop with hayai-ocr-v2.7-supernova | PLAN §Goal | ✅ | `test_p8_e2e_models.py` `TestTheDocument::test_one_page_becomes_a_schema_valid_document`<br>`test_ocr.py` `TestOcrCrops::test_progress_is_per_chunk_and_monotonic` |
| `GOAL-4` | Produces a copy-able JSON mapping bounding boxes to text | PLAN §Goal | ✅ | `test_gui.py` `TestJsonPane::test_copy_all_puts_the_exact_document_on_the_clipboard`<br>check `document-contracts` in `tests/test_p8_qa.py` |
| `GOAL-5` | Interactive annotated-image view | PLAN §Goal | ✅ | `test_gui.py` `TestCanvas::test_the_hover_panel_stays_inside_the_viewport`<br>`test_gui.py` `TestCanvas::test_save_annotated_writes_the_page_at_full_resolution` |
| `GOAL-6` | Fully remote-controllable over MCP, bind host/port/token configurable | PLAN §Goal | ✅ | `test_mcp_http.py` `test_a_full_session_round_trips_and_validates`<br>`test_mcp_stdio.py` `test_the_stdio_handshake_publishes_the_same_catalog_as_http`<br>check `cli-surface` in `tests/test_p8_qa.py` |
| `GOAL-7` | Every MCP interaction is visually reflected in the GUI | PLAN §Goal | ✅ | `test_mcp_gui.py` `TestBridge::test_every_call_leaves_exactly_one_row_in_the_activity_dock`<br>`test_mcp_gui.py` `TestBridge::test_the_badge_announces_the_server`<br>*implemented once in the bridge, not per tool — so a new tool cannot forget it* |

### ARCH — Architecture and threading (§1)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `ARCH-1` | All Qt widget access happens on the main thread | PLAN §1 rule 1 | ✅ | `test_mcp_bridge.py` `test_qt_dispatcher_runs_the_callable_on_the_main_thread`<br>`test_mcp_bridge.py` `test_qt_dispatcher_stays_out_of_its_own_thread` |
| `ARCH-2` | Inference runs on a worker thread owned by JobManager | PLAN §1 rule 2 | ✅ | `test_jobs.py` `TestCancellation::test_cancel_stops_the_worker_and_lands_on_cancelled`<br>*one dedicated worker for the process lifetime, which is also what makes ONNX session reuse work (D14)* |
| `ARCH-3` | Single-flight: one job at a time, others get BUSY + the job id | PLAN §1 rule 2 | ✅ | `test_mcp_http.py` `test_a_second_run_is_busy_with_the_running_job_id`<br>`test_mcp_service.py` `test_a_second_job_is_busy_and_names_the_running_one` |
| `ARCH-4` | Jobs are cancellable | PLAN §1 rule 2 | ✅ | `test_jobs.py` `TestCancellation::test_cancel_stops_the_worker_and_lands_on_cancelled`<br>`test_p8_errors.py` `test_a_cancel_storm_leaves_one_settled_job` |
| `ARCH-5` | MCP server on its own thread with its own asyncio loop | PLAN §1 rule 3 | ✅ | `test_mcp_server.py` `test_http_server_starts_reports_and_stops`<br>`test_mcp_service.py` `test_the_job_wait_is_never_dispatched_onto_the_app_thread` |
| `ARCH-6` | Tool handlers never touch Qt: they marshal through the bridge | PLAN §1 rule 3 | ✅ | check `no-qt-outside-the-window` in `tests/test_p8_qa.py`<br>`test_mcp_bridge.py` `test_qt_dispatcher_runs_the_callable_on_the_main_thread` |
| `ARCH-7` | Headless parity: --headless --run runs the same Pipeline without Qt | PLAN §1 | ✅ | `test_cli.py` `TestStubs::test_headless_run_is_now_the_pipeline_not_a_stub`<br>check `three-doors` in `tests/test_p8_qa.py` |

### DOC — The result document (§2.1)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `DOC-1` | Every §2.1 key is present, and the document validates against the schema | PLAN §2.1 | ✅ | check `document-contracts` in `tests/test_p8_qa.py`<br>`test_mcp_service.py` `test_run_produces_a_schema_valid_document` |
| `DOC-2` | bbox is integer xyxy in original-image space | PLAN §2.1 | ✅ | `test_types.py` `TestBBox::test_to_dict_is_xyxy_integers` |
| `DOC-3` | id is the stable index after ordering; reading_order matches it | PLAN §2.1 | ✅ | `test_pipeline.py` `TestDocument::test_ids_and_reading_order_are_the_document_order` |
| `DOC-4` | colour is a deterministic palette assignment | PLAN §2.1 | ✅ | `test_types.py` `TestMisc::test_palette`<br>`test_pipeline.py` `TestOrdering::test_ordering_is_stable_for_identical_boxes` |
| `DOC-5` | A per-block OCR failure does not abort the job: text null + error code | PLAN §2.1 | ✅ | `test_jobs.py` `TestErrors::test_an_isolated_block_failure_is_still_a_successful_job`<br>`test_pipeline.py` `TestFailureIsolation::test_one_poisoned_crop_does_not_lose_the_page` |
| `DOC-6` | created_at and timing_ms are the only non-deterministic fields | PLAN §2.1 | ✅ | check `determinism` in `tests/test_p8_qa.py`<br>`test_pipeline.py` `TestOrdering::test_ordering_is_stable_for_identical_boxes` |
| `DOC-7` | image.source is one of file / clipboard / mcp-base64 / mcp-path | PLAN §2.1 | ✅ | check `three-doors` in `tests/test_p8_qa.py`<br>`test_mcp_service.py` `test_load_image_from_path_records_mcp_provenance` |
| `DOC-8` | Schema is versioned, published, and old documents still read | PLAN §2.1 + P8.1 | met · deviation | `test_schemas.py` `TestResultSchemaVersions`<br>check `document-contracts` in `tests/test_p8_qa.py`<br>*shipped as 1.2, not 1.0: 1.1 added `stage`, 1.2 added `flags`/`contained_in`. 1.0 and 1.1 are read (frozen copies ship in the wheel)* |

### CFG — The config document (§2.2)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `CFG-1` | Precedence: CLI flags > imported config file > built-in defaults | PLAN §2.2 | ✅ | `test_config.py` `TestPrecedence`<br>check `cli-surface` in `tests/test_p8_qa.py` |
| `CFG-2` | reading_order: score | top_to_bottom | manga_rtl (default, column-aware) | PLAN §2.2 | ✅ | `test_pipeline.py` `TestOrdering::test_manga_rtl_reads_right_to_left_inside_a_row` |
| `CFG-3` | Import: missing keys default, unknown keys warn, invalid values error | PLAN §2.2 | ✅ | `test_config.py` `TestRoundtrip::test_lenient_partial`<br>`test_mcp_service.py` `test_set_config_rejects_an_invalid_value_with_the_path` |
| `CFG-4` | Export writes the live state | PLAN §2.2 | ✅ | `test_mcp_service.py` `test_config_export_and_import_round_trip` |
| `CFG-5` | The whole app state is exportable/importable, MCP settings included | PLAN §2.2 | ✅ | `test_mcp_service.py` `test_config_export_and_import_round_trip`<br>`test_p8_hardening.py` `TestRedaction::test_export_is_faithful_by_default`<br>*the one secret (`mcp.auth_token`) is there on purpose, with `--redact-secrets` / `redact_secrets` for the shareable copy (8.2)* |
| `CFG-6` | A changed MCP transport key is reported as needing a restart | PLAN §2.2 | met · deviation | `test_p8_hardening.py` `TestRestartRequired`<br>check `restart-keys-documented` in `tests/test_p8_qa.py`<br>*found by this QA pass. §2.2 promised an in-place server restart; nothing ever re-bound the socket and no door said so. The restart is deliberately not built (a client must not be able to move the socket it is talking to); the reporting is: `restart_required` on both config tools, a status-bar line on GUI import* |
| `CFG-7` | Everything else applies live, without a restart | PLAN §2.2 | ✅ | `test_p8_hardening.py` `TestLogCapacity::test_capacity_follows_a_config_change`<br>`test_mcp_service.py` `test_overlay_and_zoom_write_through_to_the_config` |

### ERR — The error taxonomy (§2.3)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `ERR-1` | One shared taxonomy of 14 codes for GUI, CLI and MCP | PLAN §2.3 | ✅ | check `error-taxonomy` in `tests/test_p8_qa.py`<br>`test_mcp_server.py` `test_catalog_records_the_error_taxonomy_and_versions` |
| `ERR-2` | Every error is {code, message, details?} | PLAN §2.3 | ✅ | `test_types.py` `TestMalformedPayloads::test_structured_error_not_keyerror` |
| `ERR-3` | MCP errors come back as isError:true with that JSON payload | PLAN §2.3 | ✅ | `test_mcp_http.py` `test_errors_are_iserror_with_the_shared_taxonomy` |
| `ERR-4` | Every code maps to a fixed process exit code | PLAN §2.3 | ✅ | check `error-taxonomy` in `tests/test_p8_qa.py`<br>`test_cli.py` `TestFoundationActions::test_validate_missing_exit_10` |
| `ERR-5` | No fault escapes as a bare exception, on any of the eight doors | P8.4 | ✅ | `test_p8_errors.py` `test_no_matrix_fault_escapes_as_a_bare_exception`<br>`python3 scripts/gen_error_matrix.py --check` |

### P1 — Part 1 — foundation

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P1-1` | --help shows the full CLI surface | PLAN Part 1 accept | ✅ | `test_cli.py` `TestParser::test_help`<br>check `cli-surface` in `tests/test_p8_qa.py` |
| `P1-2` | Config round-trip is byte-stable | PLAN Part 1 accept | ✅ | `test_config.py` `TestRoundtrip::test_byte_stable` |
| `P1-3` | Validation reports precise, path-based errors | PLAN Part 1 | ✅ | `test_config.py` `TestValidation::test_invalid_values_raise_with_path` |
| `P1-4` | Schemas ship inside the package, so wheels carry them (D6) | PLAN §1 layout | ✅ | `test_mcp_server.py` `test_catalog_is_the_file_that_ships_in_the_wheel`<br>`test_p8_packaging.py` `TestPackageData::test_the_schemas_and_the_manifest_ship` |

### P2 — Part 2 — detection engine

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P2-1` | Engine protocol (load/status/progress/clear) + deterministic fake | PLAN Part 2 | ✅ | `test_engines_base.py` `TestFakeDetectionEngine::test_list_models_is_the_known_catalog`<br>`test_engines_base.py` `TestFakeDetectionEngine::test_load_sets_ready_and_rejects_unknown` |
| `P2-2` | list_models(): Hub listing with an offline/blocked fallback | PLAN Part 2 | ✅ | `test_detection.py` `TestListModels::test_offline_sentinel_falls_back`<br>`test_detection.py` `TestListModels::test_auth_failure_falls_back_to_known_models` |
| `P2-3` | Lazy per-model load with download progress | PLAN Part 2 | ✅ | `test_detection.py` `TestLoad::test_download_reports_progress_percent` |
| `P2-4` | score=None resolves to the repo-recommended threshold | PLAN Part 2 | ✅ | `test_detection.py` `TestThresholds::test_documented_preset_when_backend_is_cold`<br>`test_models.py` `TestBundledDetection::test_recommended_threshold_comes_from_the_bundled_file` |
| `P2-5` | Gated-repo errors become HF_AUTH_REQUIRED with instructions | PLAN Part 2 | ✅ | `test_detection.py` `TestLoad::test_auth_error_carries_actionable_steps` |
| `P2-6` | ONNX provider selection (CPU default, CUDA when the runtime exposes it) | PLAN Part 2 | met · deviation | `test_detection.py` `TestConfigureHelpers::test_provider_cpu_sets_onnx_mode`<br>`test_detection.py` `TestDeviceReporting::test_device_resolves_to_the_real_provider_after_load`<br>*exposed as --provider / ONNX_MODE, not as a config key, so config schema 1.0 stayed frozen (D17)* |
| `P2-7` | unload_all() frees the sessions | PLAN Part 2 | ✅ | `test_detection.py` `TestUnload::test_unload_clears_sessions_and_forces_refetch`<br>`test_p8_lifecycle.py` `test_unload_releases_the_session_even_though_rss_has_a_floor` |

### P3 — Part 3 — OCR engine + imaging

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P3-1` | Image IO is EXIF-aware and always RGB | PLAN Part 3 | ✅ | `test_imaging.py` `TestLoad::test_exif_orientation_is_applied`<br>`test_imaging.py` `TestLoad::test_palette_and_gray_and_rgba_become_rgb` |
| `P3-2` | Crops clamp padding to the image and guard the minimum size | PLAN Part 3 | ✅ | `test_imaging.py` `TestCrop::test_padding_is_clamped_to_the_image` |
| `P3-3` | Deterministic 20-colour palette | PLAN Part 3 | ✅ | `test_types.py` `TestMisc::test_palette` |
| `P3-4` | Hayai loads lazily, pinned revision, trust_remote_code | PLAN Part 3 | ✅ | `test_p8_e2e_models.py` `TestTheWeightsThatRun::test_the_bundle_carries_both_pinned_repositories`<br>`test_models.py` `TestBundledDetection::test_vendored_ocr_snapshot_matches_the_pinned_revision` |
| `P3-5` | Device auto: CUDA gets fp16/cuda-graphs, CPU gets fp32 eager | PLAN Part 3 | ✅ | `test_ocr.py` `TestLoad::test_load_reports_progress_and_ends_ready`<br>`test_p8_e2e_models.py` `TestTheFlagsOnRealData::test_the_top_patch_budget_is_reachable` |
| `P3-6` | Batched OCR with chunking, progress and cancellation between chunks | PLAN Part 3 | ✅ | `test_ocr.py` `TestOcrCrops::test_progress_is_per_chunk_and_monotonic`<br>`test_ocr.py` `TestFakeOcrEngine::test_chunking_and_progress` |
| `P3-7` | Optional NFKC normalisation | PLAN Part 3 | ✅ | `test_imaging.py` `TestNormalize::test_nfkc_fullwidth_and_ligature`<br>`test_imaging.py` `TestNormalize::test_nfkc_can_be_disabled` |
| `P3-8` | unload() releases the model | PLAN Part 3 | ✅ | `test_engines_base.py` `TestEnginesBadge::test_a_real_run_then_unload_flips_the_text` |

### P4 — Part 4 — pipeline, jobs, headless

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P4-1` | detect -> order -> crop -> OCR -> assemble, GUI-independent | PLAN Part 4 | ✅ | `test_pipeline.py` `TestEndToEnd::test_a_corrupt_payload_is_an_invalid_image_not_a_crash`<br>check `three-doors` in `tests/test_p8_qa.py` |
| `P4-2` | JobManager: states, stage+percent progress, result cache = latest result | PLAN Part 4 | ✅ | `test_jobs.py` `TestListeners::test_progress_events_carry_the_stage_names_the_gui_renders` |
| `P4-3` | --headless --run prints the JSON, --out writes it, exit codes are the taxonomy | PLAN Part 4 | ✅ | `test_cli.py` `TestStubs::test_headless_run_is_now_the_pipeline_not_a_stub`<br>`python3 -m animetext_studio --headless --mock --image examples/sample_page.png --run` |
| `P4-4` | Golden result JSON with fake engines, schema-validated | PLAN Part 4 accept | ✅ | `test_pipeline.py` `TestDocument`<br>check `determinism` in `tests/test_p8_qa.py` |
| `P4-5` | --mock works headless with no downloads | PLAN Part 4 | ✅ | `python3 -m animetext_studio --headless --mock --image examples/sample_page.png --run`<br>check `three-doors` in `tests/test_p8_qa.py` |

### P5 — Part 5 — GUI shell

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P5-1` | Control panel + preview + output, central widget scrollable both ways | PLAN Part 5 | ✅ | `test_gui.py` `TestWindow::test_the_central_widget_is_scrollable_both_ways` |
| `P5-2` | Model combo with params/F1/recommended threshold per variant | PLAN Part 5 | ✅ | `test_gui.py` `TestConfigWiring::test_offline_marks_hub_only_variants_unusable`<br>`test_engines_base.py` `TestHelpers::test_model_catalog_uses_presets` |
| `P5-3` | IoU/score sliders with spinboxes + a Recommended button | PLAN Part 5 | ✅ | `test_gui.py` `TestConfigWiring::test_the_recommended_threshold_button_uses_the_preset` |
| `P5-4` | OCR group, reading order, Generate and Cancel | PLAN Part 5 | ✅ | `test_gui.py` `TestRun::test_generate_with_mock_engines_produces_a_result`<br>`test_gui.py` `TestRun::test_cancel_from_the_panel_stops_the_job` |
| `P5-5` | Config Export/Import buttons in the window | PLAN Part 5 | ✅ | `test_gui.py` `TestViewConfigRoundTrip::test_export_then_import_restores_the_view_and_the_canvas` |
| `P5-6` | MCP status indicator | PLAN Part 5 | ✅ | `test_gui.py` `TestWindow::test_the_mcp_indicator_starts_off_and_can_be_flipped` |
| `P5-7` | Image input: browse, drag-and-drop, Ctrl+V, preview with name and size | PLAN Part 5 | ✅ | `test_gui.py` `TestImageInput::test_browse_path_sets_preview_and_caption`<br>`test_gui.py` `TestImageInput::test_drag_and_drop_of_an_image_file_loads_it`<br>`test_gui.py` `TestImageInput::test_paste_of_a_file_url_loads_that_file` |
| `P5-8` | QThread adapter over JobManager; buttons follow the job | PLAN Part 5 | ✅ | `test_gui.py` `TestRun::test_generate_without_an_image_reports_no_image`<br>`test_gui.py` `TestRun::test_a_failing_engine_shows_the_error_code_not_a_crash` |

### P6 — Part 6 — visualisation + JSON pane

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P6-1` | Translucent fill + solid border + label chip, over a copy of the page | PLAN Part 6 | ✅ | `test_gui.py` `TestCanvas::test_the_fill_is_translucent_not_a_sticker`<br>`test_annotation.py` `TestRender::test_the_fill_is_translucent_and_the_border_is_not` |
| `P6-2` | Hover highlights the box and shows the full text in a floating panel | PLAN Part 6 | ✅ | `test_gui.py` `TestCanvas::test_hovering_shows_the_full_text_and_leaving_hides_it`<br>`test_annotation.py` `TestText::test_the_hover_panel_carries_the_whole_text` |
| `P6-3` | Click selects a block and syncs the table and the JSON pane | PLAN Part 6 | ✅ | `test_gui.py` `TestViewConfigRoundTrip::test_a_result_draws_through_the_configured_view`<br>`test_gui.py` `TestJsonPane::test_the_copy_block_button_uses_the_current_selection` |
| `P6-4` | Overlay/label toggles, fill-opacity slider, zoom fit/percent/Ctrl+wheel | PLAN Part 6 | ✅ | `test_gui.py` `TestCanvas::test_unticking_show_labels_changes_the_picture`<br>`test_gui.py` `TestCanvas::test_the_fill_slider_reaches_the_canvas`<br>`test_gui.py` `TestCanvas::test_ctrl_wheel_walks_the_zoom_steps` |
| `P6-5` | Save annotated PNG bakes boxes and labels at full resolution | PLAN Part 6 | ✅ | `test_gui.py` `TestCanvas::test_save_annotated_writes_the_page_at_full_resolution`<br>`test_gui.py` `TestCanvasExtras::test_the_export_carries_every_block` |
| `P6-6` | JSON pane: readable UTF-8, read-only, Copy All, Save As, per-block copy | PLAN Part 6 | ✅ | `test_gui.py` `TestJsonPane::test_the_pane_holds_the_document_byte_for_byte`<br>`test_gui.py` `TestJsonPane::test_the_pane_is_read_only`<br>`test_gui.py` `TestJsonPane::test_save_as_writes_readable_utf8`<br>`test_gui.py` `TestJsonPane::test_copy_block_text_and_json` |
| `P6-7` | Chips do not sit on each other, and nothing is dropped silently | P8.6 | ✅ | `test_p8_annotation.py` `TestPlaceChips`<br>`test_p8_annotation.py` `TestTheWindowShowsTheObservations` |

### P7 — Part 7 — MCP server + GUI bridge

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P7-1` | Official mcp SDK (FastMCP), streamable-http + stdio + none | PLAN Part 7 | ✅ | `test_mcp_server.py` `test_build_mcp_server_registers_tools_and_resources`<br>`test_mcp_server.py` `test_transport_none_never_binds_and_reports_itself` |
| `P7-2` | Server thread + own loop; optional bearer token; loopback by default | PLAN Part 7 | ✅ | check `defaults-and-auth` in `tests/test_p8_qa.py`<br>`test_mcp_server.py` `test_gateway_health_is_open_and_the_mcp_endpoint_is_not` |
| `P7-3` | Bridge: queued invocation, futures with timeouts, busy policy | PLAN Part 7 | ✅ | `test_mcp_bridge.py` `test_qt_dispatcher_runs_the_callable_on_the_main_thread` |
| `P7-4` | Activity hook: one row per call, with args, outcome and duration | PLAN Part 7 | ✅ | `test_mcp_bridge.py` `test_activity_log_numbers_and_keeps_order`<br>`test_mcp_gui.py` `TestBridge::test_every_call_leaves_exactly_one_row_in_the_activity_dock` |
| `P7-5` | MCP-driven jobs move the GUI progress bar and send progress notifications | PLAN Part 7 | ✅ | `test_mcp_http.py` `test_progress_notifications_arrive_during_a_run` |
| `P7-6` | Every §4 tool exists, with input and output schemas | PLAN Part 7 | ✅ | check `tool-list` in `tests/test_p8_qa.py`<br>`test_mcp_server.py` `test_catalog_covers_the_spec_and_documents_every_tool` |
| `P7-7` | Machine-readable catalog at schemas/mcp-tools.json | PLAN Part 7 | ✅ | `test_mcp_server.py` `test_write_catalog_writes_where_it_is_told_and_nowhere_else`<br>check `mcp-doc-is-the-catalog` in `tests/test_p8_qa.py` |
| `P7-8` | SDK demo client + raw curl transcript for non-SDK programs | PLAN Part 7 | ✅ | `python3 -m animetext_studio.tools.mcp_client_demo --url http://127.0.0.1:8765/mcp`<br>`./scripts/mcp_curl_demo.sh 8765`<br>*re-run live for the P9.6 sign-off: 21 tools answered, every payload matched its published output schema, and the demo's re-OCR step came back with the same boxes and stage `reocr` (84 ms for one block against the page's 949 ms, real weights on CPU), exit 0* |

### MCP — Deterministic-client guarantees (§4)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `MCP-1` | Stable ids and colours for the same page and parameters | PLAN §4 | ✅ | check `determinism` in `tests/test_p8_qa.py`<br>`test_pipeline.py` `TestOrdering::test_ordering_is_stable_for_identical_boxes` |
| `MCP-2` | BUSY carries the running job id; clients may poll or retry | PLAN §4 | ✅ | `test_mcp_http.py` `test_a_second_run_is_busy_with_the_running_job_id` |
| `MCP-3` | Optional bearer auth; default bind is loopback | PLAN §4 | ✅ | check `defaults-and-auth` in `tests/test_p8_qa.py` |
| `MCP-4` | A non-loopback bind warns, louder without a token | PLAN §4 + P8.2 | ✅ | `test_p8_hardening.py` `TestEndpointWarnings` |
| `MCP-5` | docs/MCP.md carries a curl-level session transcript | PLAN §4 | ✅ | check `mcp-doc-is-the-catalog` in `tests/test_p8_qa.py`<br>`python3 scripts/mcp_http_transcript.py --port 8765` |
| `MCP-6` | Six resources answer | PLAN §4 | ✅ | `test_mcp_http.py` `test_resources_answer`<br>check `tool-list` in `tests/test_p8_qa.py` |
| `MCP-7` | Malformed tool arguments are INVALID_ARGUMENT, never a broken socket | PLAN §4 | ✅ | `test_p8_errors.py` `test_the_server_is_fine_after_a_storm_of_garbage` |

### CLI — CLI surface (§5)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `CLI-1` | Every flag PLAN §5 names exists and is parsed | PLAN §5 | ✅ | check `cli-surface` in `tests/test_p8_qa.py` |
| `CLI-2` | Precedence CLI > config file > defaults, including the MCP flags | PLAN §5 | ✅ | `test_config.py` `TestPrecedence` |
| `CLI-3` | Every ErrorCode has its own exit code, and the README states the number | PLAN §5 + P8.7 | ✅ | check `error-taxonomy` in `tests/test_p8_qa.py`<br>`test_p8_packaging.py` `TestReadmeDocumentsTheCli::test_the_exit_code_table_matches_the_taxonomy` |
| `CLI-4` | --dump-default-config / --validate-config / --export-config work standalone | PLAN §5 (D9) | ✅ | `test_cli.py` `TestFoundationActions::test_validate_bad_exit_2`<br>`python3 -m animetext_studio --dump-default-config` |

### ENV — Dependencies and environment (§6)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `ENV-1` | Extras split so a headless detector install needs neither torch nor Qt | PLAN §6 + D7 | ✅ | check `extras-and-floor` in `tests/test_p8_qa.py`<br>`test_p8_packaging.py` `TestPyprojectMetadata::test_all_extra_is_the_union_and_cannot_drift` |
| `ENV-2` | Python >= 3.10 | PLAN §6 | ✅ | check `extras-and-floor` in `tests/test_p8_qa.py` |
| `ENV-3` | CUDA wheels for the target hardware documented (Blackwell needs cu128) | PLAN §6 | ✅ | check `readme-commands` in `tests/test_p8_qa.py`<br>*documented in README Install; the sandbox here is CPU-only, so the CUDA path is documented and reviewed, not executed* |
| `ENV-4` | A CJK-capable font is found, or the labels fall back without crashing | PLAN §6 | ✅ | check `cjk-font` in `tests/test_p8_qa.py`<br>`test_annotation.py` `TestText::test_the_chip_is_id_score_and_the_first_words` |
| `ENV-5` | Real-model tests are opt-in and the default suite never touches the network | PLAN §6 | ✅ | check `default-suite-is-offline` in `tests/test_p8_qa.py`<br>`test_p8_e2e_models.py` `TestTheWeightsThatRun::test_a_real_run_opens_no_socket_and_needs_no_token` |

### RISK — Risks and mitigations (§7)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `RISK-1` | Gated repo: HF_AUTH_REQUIRED with exact steps, GUI banner, offline fallback | PLAN §7 risk 1 | ✅ | `test_detection.py` `TestLoad::test_auth_error_carries_actionable_steps`<br>`test_model_store.py` `TestEngineUsesBundle::test_require_offline_points_at_fetch_models` |
| `RISK-2` | trust_remote_code pinned to a revision, with a real smoke test | PLAN §7 risk 2 | ✅ | `test_p8_e2e_models.py` `TestTheWeightsThatRun::test_the_bundle_carries_both_pinned_repositories`<br>`python3 scripts/fetch_models.py --verify` |
| `RISK-3` | No Qt/asyncio deadlock: bridge pattern, no Qt imports outside the window | PLAN §7 risk 3 | ✅ | check `no-qt-outside-the-window` in `tests/test_p8_qa.py`<br>`test_mcp_bridge.py` `test_qt_dispatcher_stays_out_of_its_own_thread` |
| `RISK-4` | Tall pages: allow_dynamic is exposed and its trade-off documented | PLAN §7 risk 4 | ✅ | check `readme-commands` in `tests/test_p8_qa.py`<br>`test_config.py` `TestValidation::test_valid_devices_and_patches` |
| `RISK-5` | CPU OCR stays usable: batching, chunked progress, patch budgets | PLAN §7 risk 5 | ✅ | `test_ocr.py` `TestOcrCrops::test_progress_is_per_chunk_and_monotonic`<br>`test_p8_e2e_models.py` `TestTheFlagsOnRealData` |
| `RISK-6` | Memory: lazy loading, unload, engine status in get_status | PLAN §7 risk 6 | ✅ | `test_p8_lifecycle.py` `TestMemoryRelease`<br>*measured (D62): the objects go, RSS keeps a ~200 MB ONNX Runtime floor — the test pins that instead of a nicer fiction* |
| `RISK-7` | Determinism for harnesses: stable order/colours, volatile fields documented | PLAN §7 risk 7 | ✅ | check `determinism` in `tests/test_p8_qa.py` |
| `RISK-8` | Licensing stated honestly: Apache-2.0 code, GPL-3.0 weights, slim wheel option | PLAN §7 risk 8 | met · measured | `test_p8_packaging.py` `TestLicence`<br>*the distributor chose Apache-2.0 on 2026-11-24; LICENSE + NOTICE ship in the wheel, the GPL text ships beside the weights* |

### P8 — What Part 8 added

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P8-1` | stage: detect|full distinguishes a stage-1 document (result 1.1) | P8.1 | ✅ | `test_pipeline.py` `TestStageMarker`<br>`test_schemas.py` `TestResultSchemaVersions` |
| `P8-2` | Log capacity is a live config key; exports can redact secrets | P8.2 | ✅ | `test_p8_hardening.py` `TestLogCapacity`<br>`test_p8_hardening.py` `TestRedaction` |
| `P8-3` | Cheap correctness flags: possible_truncation and contained_in (result 1.2) | P8.2b | ✅ | `test_pipeline.py` `TestDocumentAnnotations`<br>`test_schemas.py` `TestResultSchemaVersions::test_1_2_added_flags_and_contained_in_to_every_block` |
| `P8-4` | Warm means loaded plus one inference, and it is a cancellable job | P8.3 | ✅ | `test_p8_lifecycle.py` `TestWarmEngines`<br>`test_p8_e2e_models.py` `TestTheMcpServiceDoor::test_a_real_warm_up_is_a_job_and_not_a_page` |
| `P8-5` | Engine warmth is reported, not hoped for | P8.3 | ✅ | `test_p8_lifecycle.py` `TestWarmEngines::test_it_loads_both_engines_and_runs_one_inference_through_each` |
| `P8-6` | The error matrix: named faults × 8 doors, generated into docs | P8.4 | ✅ | `test_p8_errors.py` <br>`python3 scripts/gen_error_matrix.py --check`<br>*84 rows at the P8.8 sign-off; 91 since P9.6 folded in the re-OCR door, the corrupt-bundle diagnosis and the empty-option refusals. The count lives in the sign-off line below, not in a row that would then go stale.* |
| `P8-7` | Real weights through every door, offline, in one opt-in suite | P8.5 | ✅ | `test_p8_e2e_models.py` <br>`python3 -m pytest -q -m models` |
| `P8-8` | One de-collision rule shared by the canvas and the export | P8.6 | ✅ | `test_p8_annotation.py` `TestPlaceChips` |
| `P8-9` | README documents the shipped surface, and a test says so | P8.7 | ✅ | `test_p8_packaging.py` <br>check `readme-commands` in `tests/test_p8_qa.py` |
| `P8-10` | The on-premises bundle is the install (D26): verify it, offline | P8 / D26 | ✅ | `python3 scripts/fetch_models.py --verify`<br>`test_model_store.py`  |

### P9 — What Part 9 added (the five approved enhancements, D80–D94)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `P9-1` | Weights that are present and wrong are diagnosed as that, not as absent | P9.1 / backlog 7 | ✅ | `test_model_store.py` `TestOpenFailureDiagnosis::test_corrupt_weights_offline_name_the_file_and_the_fix`<br>`test_model_store.py` `TestOpenFailureDiagnosis::test_intact_weights_blame_the_runtime_not_the_file`<br>`test_model_store.py` `TestOpenFailureDiagnosis::test_a_non_load_error_is_not_rescued`<br>`test_p8_errors.py` `test_each_matrix_row_carries_the_code_the_matrix_promises` |
| `P9-2` | The hash is a failure-path tool: the hot paths never pay for it | P9.1 | ✅ | `test_model_store.py` `TestVerifyFile::test_hot_paths_stay_hash_free`<br>`test_model_store.py` `TestVerifyFile::test_truncated_file_is_a_mismatch_with_both_hashes` |
| `P9-3` | Online, corrupt weights are re-vendored once — and the retry is recorded, offline it is refused and says that too | P9.1 | ✅ | `test_model_store.py` `TestOpenFailureDiagnosis::test_corrupt_weights_online_retry_once_and_say_they_did`<br>`test_model_store.py` `TestOpenFailureDiagnosis::test_a_failed_retry_keeps_the_diagnosis_instead_of_hiding_it`<br>`test_model_store.py` `TestOpenFailureDiagnosis::test_offline_never_reaches_for_the_network` |
| `P9-4` | One field says where the model list came from; every door reads that field (D90) | P9.2 / backlog 8 | ✅ | `test_engines_base.py` `TestModelsBadge::test_the_flag_is_on_the_wire_too`<br>`test_mcp_service.py` `test_status_carries_the_reason_the_list_degraded`<br>`test_mcp_service.py` `test_an_ordinary_engine_reports_no_fallback`<br>`animetext-studio --model-status` |
| `P9-5` | The window says it too: the fifth badge, and Refresh models | P9.2 | ✅ | `test_gui.py` `TestModelsBadge::test_the_badge_appears_with_the_reason_behind_it`<br>`test_gui.py` `TestModelsBadge::test_answering_empty_hides_it_rather_than_blanking_it`<br>`test_engines_base.py` `TestModelsBadge::test_a_live_list_shows_nothing` |
| `P9-6` | Re-OCR one block without asking the detector again — the 21st tool | P9.3 | ✅ | `test_pipeline.py` `TestReocrBlocks::test_the_detector_is_not_asked_twice`<br>`test_pipeline.py` `TestReocrBlocks::test_only_the_selected_crop_is_sent`<br>`test_mcp_server.py` `test_catalog_covers_the_spec_and_documents_every_tool` |
| `P9-7` | A re-read answers with a whole document, not a diff (D81) | P9.3 | ✅ | `test_pipeline.py` `TestReocrBlocks::test_the_document_is_whole_and_unselected_blocks_keep_their_text`<br>`test_pipeline.py` `TestReocrBlocks::test_the_new_document_validates_and_says_how_it_was_made`<br>`test_pipeline.py` `TestReocrBlocks::test_ids_colours_order_and_boxes_survive` |
| `P9-8` | The re-OCR door refuses in the order a caller can use | P9.3 | ✅ | `test_p8_errors.py` `TestReocrDoor::test_an_unknown_id_names_the_ones_that_exist`<br>`test_p8_errors.py` `TestReocrDoor::test_with_no_page_loaded_the_answer_is_no_image`<br>`test_pipeline.py` `TestReocrBlocks::test_cancelling_a_re_read_is_a_cancellation_not_a_broken_one` |
| `P9-9` | The block table can ask for it: right-click → Re-read this block | P9.3 | ✅ | `test_gui.py` `TestReocrFromTheTable::test_re_reading_a_block_replaces_the_document_with_a_reocr_one`<br>`test_gui.py` `TestReocrFromTheTable::test_the_menu_is_wired_to_the_table`<br>`test_gui.py` `TestReocrFromTheTable::test_no_selection_means_no_job` |
| `P9-10` | Per-device cold latency, measured on the machine you are on (D84) | P9.4 | ✅ | `test_p8_lifecycle.py` `TestDeviceSweep::test_the_device_you_asked_for_survives_next_to_the_one_you_got`<br>`test_p8_lifecycle.py` `TestDeviceSweep::test_a_failing_engine_is_a_row_not_a_crash`<br>`test_cli.py` `TestWarmDevicesFlag::test_the_sweep_prints_a_table_and_exits_zero`<br>`animetext-studio --warm-devices cpu` |
| `P9-11` | `--preload-off` names the cold baseline a benchmark measured against | P9.4 | ✅ | `test_cli.py` `TestWarmDevicesFlag::test_preload_off_is_the_cold_baseline_even_read_as_a_flag`<br>`test_cli.py` `TestWarmDevicesFlag::test_preload_and_preload_off_are_mutually_exclusive`<br>`test_p8_lifecycle.py` `TestDeviceSweep::test_each_pair_is_released_so_the_next_row_starts_cold` |
| `P9-12` | `--doctor`: read-only, never installs, and it exits with the code a run would have hit (D88, D89) | P9.5 | ✅ | `test_p9_doctor.py` `TestTheDoctorNeverDoesTheThingItIsDiagnosing::test_a_read_only_bundle_directory_is_still_reportable`<br>`test_p9_doctor.py` `TestTheCommand::test_a_blocking_machine_exits_with_the_code_it_predicts`<br>`animetext-studio --doctor` |
| `P9-13` | The two upstream traps are written up as files, and pinned to a test that goes red when upstream fixes them (D92) | P9.5 | ✅ | `test_p9_doctor.py` `TestTheUpstreamDrafts`<br>`test_p9_doctor.py` `TestPython::test_3_14_names_the_upstream_cause_and_the_draft` |
| `P9-14` | An option that was given an empty value is refused, not ignored (D93) | P9.6 finding | ✅ | `test_cli.py` `TestEmptyOptionValues`<br>`test_p8_errors.py` `test_the_cli_answers_with_the_matrix_code` |
| `P9-15` | The catalog generator writes where it is told, and nowhere else (D94) | P9.6 finding | ✅ | `test_mcp_server.py` `test_write_catalog_writes_where_it_is_told_and_nowhere_else`<br>`test_cli.py` `TestDumpMcpTools`<br>`test_mcp_server.py` `test_catalog_is_the_file_that_ships_in_the_wheel` |
| `P9-16` | The trail moved with the code: catalog, matrix, checklist, README, floors | P9.6 | ✅ | `python3 scripts/gen_error_matrix.py --check`<br>`python3 scripts/gen_qa_report.py --check`<br>`test_p8_packaging.py` <br>`test_mcp_server.py` `test_catalog_is_the_file_that_ships_in_the_wheel` |

### PUB — The publication boundary (D95, amending D26)

| ID | Requirement | Source | Status | Evidence |
|---|---|---|---|---|
| `PUB-1` | The repository publishes only what the project may publish: no weights, no scans, no machine paths, nothing over GitHub's 100 MB limit (D95) | D95 | ✅ | `test_p8_packaging.py` `TestWhatTheRepositoryPublishes`<br>`test_p8_qa.py` `TestQaChecks::test_no_secret_travels_in_the_tree`<br>`python3 scripts/fetch_models.py --verify` |

## Still open, on purpose

Not requirements and not defects. Each one is a decision that was argued, and the argument is the thing to re-read before reopening it.

| Item | Why it stays open |
|---|---|
| File the two `dghs-imgutils` issues (import-time pip install; `ast.Str` on 3.14) | drafted and pinned in docs/UPSTREAM.md (D92). Filing needs an account on someone else's repository, and only a classic token can write to one — a fine-grained token, which is what this project was offered, cannot (GitHub's own limitation). Pre-flight on 2026-11-26 also found that Issue 2 is already fixed upstream in unmerged PR #178, so the artefact there is a release request, not a bug report. `TestTheUpstreamDrafts` goes red if the quoted source moves underneath them |
| `Pipeline.reocr_blocks` reads a slightly different crop than the run it is re-reading (found by the bench self-test, 2026-11-27) | `run()` stores the padding-adjusted bbox and then reocr pads it a second time, so the crop fed to the model is `2 * crop_padding_px` larger than the one the text came from — measured at 4 px on the default padding, and enough to change the text of 3 blocks in a 14-block page with no setting changed. The docstring promises the opposite. The fix is one line (`padding=0` in the reocr crop) plus a test that a re-read at unchanged settings reproduces the text exactly; it is held because the operator is mid-review of the current behaviour and a behaviour change should not arrive inside an evaluation. `scripts/bench_models.py` sidesteps it by running both sides through `Pipeline.run()` |
| Block colours that differ between two runs of the same page | identity, not correctness: the chips and the table key on `id`, which is stable (D32). A re-read (`reocr_blocks`) keeps the colours on purpose, which is as far as this goes |
| Language policy, sub-crop splitting, nested-box de-duplication | decided out of scope (D58, D60): the caller owns policy, the app reports — do not re-propose these as bugs |
| Warm-up reports, the device sweep and `--doctor` stay out of the MCP surface | decided, not deferred (D83, D88): a client that wants to know how warm the machine is can read `engines[]`; a client that wants to benchmark someone else's GPU should not be paying for it over a tool call |
---

## How this was verified

Every command below was run for the sign-off recorded in the table; the suite is the one that
ships, not a QA-only harness.

| Command | What it proves |
|---|---|
| `QT_QPA_PLATFORM=offscreen python3 -m pytest -q` | the whole offline suite, including this checklist's rules |
| `python3 -m pytest -q -m models` | the vendored weights through every door, offline and tokenless |
| `python3 scripts/fetch_models.py --verify` | the bundle is complete and hash-clean (D26) |
| `python3 scripts/gen_error_matrix.py --check` | the error documentation matches the fault table |
| `python3 scripts/gen_qa_report.py --check` | this document matches the requirement table |
| `QT_QPA_PLATFORM=offscreen python3 -m animetext_studio --gui-smoke --mock` | the window builds and lays out |
| `python3 -m animetext_studio --headless --mock --image examples/sample_page.png --run` | the CLI door, end to end |
| `python3 -m animetext_studio.tools.mcp_client_demo --url http://127.0.0.1:8765/mcp` | the MCP door, end to end, against the published schemas — 21 tools published, 21 in the catalog, and the re-OCR step re-read one block in 84 ms against the page's 949 ms (real weights, CPU) |
| `python3 -m animetext_studio --doctor` | the environment door: 15 checks, 14 ok / 1 warn (the known `numpy <2` pin) / 0 fail, exit 0 |
| `python3 -m animetext_studio --warm-devices cpu --mock` | the device-sweep door: a table on stdout, a row per engine, exit 0 |

The three doors are compared against each other rather than described:
`test_the_three_doors_produce_the_same_document` runs the CLI door as a subprocess and the
service door in-process and asserts the two documents are the same object apart from the
documented volatile fields and the `image.source` provenance marker.
