# Error matrix

Every way this application can say "no", in one place.

The point is not the list; it is that the list is **complete**. A fault that reaches a caller as a
bare Python exception, or as a sentence where a code belongs, is a bug — even when nothing
else went wrong. `tests/test_p8_errors.py` implements the table below as 91 executable cases plus
the invariants at the end, and `test_every_row_of_the_matrix_is_documented` fails if the table and
the tests ever disagree.

The one door deliberately **not** in the table is `--doctor`: it never refuses, because it is not
running the thing it examines. It reports what would fail and exits with the code the next real
command would have returned (D89), so "this machine cannot run the app" is a read-only answer a
caller can branch on — with no page loaded, no model touched and nothing installed.

An option that takes a value and is given `""` is in it, though. Typing `--warm-devices ""` is a
mistake about *this* command, not a request to run the default one, and for a while it was
answered by opening a window (D93). A value the user typed is never the same as an option they
did not type.

---

## The shape of a refusal

Three doors, one taxonomy (`animetext_studio/core/types.py`, `ErrorCode`):

| Door | Failure arrives as |
|---|---|
| MCP tool call | `isError: true`, with `{code, message, details}` in `structuredContent` **and** the same object as text in `content[0]` |
| MCP resource read | JSON-RPC error: `-32602` when the caller can fix it (`NO_IMAGE`, `INVALID_ARGUMENT`, `INVALID_CONFIG`, `INVALID_IMAGE`, `BUSY`), `-32603` otherwise; `error.data` is the same `{code, message, details}` object |
| CLI | one JSON line `{"error": {code, message, details}}` on **stderr** (stdout is reserved for documents) and a process exit code |

Two rules hold everywhere, and both are load-bearing:

1. **`INTERNAL` is reserved.** It means "the engine or the app broke its own contract". Five
   rows produce it: four for an engine returning something that is not what the pipeline
   contract promises, and one for a job worker raising something that is not a `StudioError`.
   A user mistake, a missing file, a bad config value or a refused job never come back as
   `INTERNAL`.
2. **A refusal changes nothing.** The live config, the loaded page, the published result and the
   job machine all survive every row below; `test_the_app_still_works_after_the_entire_matrix`
   throws the whole table at one process and then asks it to run a page.

---

## The taxonomy

`details` is always optional and always meant to be acted on: it carries the key path of a bad
config value, the known model names, the steps to fix a gated repo, the byte offset of a
non-UTF-8 file.

| Code | Exit | Means | What the caller does |
|---|---|---|---|
| `INVALID_ARGUMENT` | 2 | the request itself is wrong: a missing argument, a bad type, two sources where one is expected, an out-of-range number | fix the call; the tool's `inputSchema` is in `list_tools()` and `schemas/mcp-tools.json` |
| `INVALID_CONFIG` | 2 | a config document or patch is invalid; `details.issues` lists `{path, message}` | apply the listed keys only — nothing was changed |
| `INVALID_IMAGE` | 11 | the bytes are not a readable image, or are too large to decode | re-encode or downscale; a truncated download looks like this |
| `NO_IMAGE` | 3 | no page is loaded (or no result is published yet) | `load_image` / run the pipeline first — the `details.hint` says which |
| `BUSY` | 4 | single-flight: another job owns the machine | poll `get_status`, or `cancel_job`; `details.job_id` names the incumbent |
| `CANCELLED` | 5 | you cancelled it | nothing; the previous page and result are untouched |
| `MODEL_NOT_FOUND` | 12 | the weights are not on premises and cannot be fetched | `scripts/fetch_models.py --variants …`, or clear `--offline` |
| `HF_AUTH_REQUIRED` | 6 | Hugging Face refused: the detection repo is gated | accept the terms, then set `HF_TOKEN` (`details.steps`) |
| `DOWNLOAD_FAILED` | 7 | the network answered badly (or not at all) | retry, or run `--offline` against a populated bundle |
| `ENGINE_LOAD_FAILED` | 8 | an engine could not be loaded (missing extra, broken weights, no provider) | follow `details.hint` — it names the pip extra or the file |
| `OCR_FAILED` | 9 | OCR failed on the page, or on one crop (`blocks[].error`) | lower `max_patches` / `max_new_tokens`, or read the per-block error |
| `IO_ERROR` | 10 | a file could not be read or written | check the path and permissions; `details.reason` is the OS message |
| `INTERNAL` | 1 | the app or an engine broke its own contract | report it; `details.reason` names the exception |
| `NOT_IMPLEMENTED` | 20 | declared for forward compatibility; **no current code path produces it** | — |

---

## The matrix

91 faults, grouped by the door they come through. Each row is a test in
`tests/test_p8_errors.py`; the code shown is what that test asserts, today.

### imaging

| Fault | Code |
|---|---|
| `missing file` | `IO_ERROR` |
| `path is a directory` | `IO_ERROR` |
| `path is empty` | `INVALID_ARGUMENT` |
| `path is not a path` | `INVALID_ARGUMENT` |
| `file is unreadable` | `IO_ERROR` |
| `file is empty` | `INVALID_IMAGE` |
| `file is not an image` | `INVALID_IMAGE` |
| `png truncated` | `INVALID_IMAGE` |
| `payload is empty bytes` | `INVALID_IMAGE` |
| `payload is a str, not bytes` | `INVALID_ARGUMENT` |
| `base64 is empty` | `INVALID_ARGUMENT` |
| `base64 is not base64` | `INVALID_ARGUMENT` |
| `base64 decodes to garbage` | `INVALID_IMAGE` |
| `data: URL without a payload` | `INVALID_ARGUMENT` |
| `decoder says decompression bomb` | `INVALID_IMAGE` |
| `over MAX_IMAGE_PIXELS` | `INVALID_IMAGE` |
| `crop outside the image` | `INVALID_ARGUMENT` |

### pipeline

| Fault | Code |
|---|---|
| `run without an ImageInfo` | `INVALID_ARGUMENT` |
| `run with a non-image` | `INVALID_IMAGE` |
| `engine returns None` | `INTERNAL` |
| `engine returns dicts` | `INTERNAL` |
| `ocr returns the wrong count` | `INTERNAL` |
| `ocr returns a non-string` | `INTERNAL` |
| `engine raises MODEL_NOT_FOUND` | `MODEL_NOT_FOUND` |
| `engine raises HF_AUTH_REQUIRED` | `HF_AUTH_REQUIRED` |
| `engine raises DOWNLOAD_FAILED` | `DOWNLOAD_FAILED` |
| `ocr fails on the only crop` | `OCR_FAILED` |

### jobs

| Fault | Code |
|---|---|
| `second job while running` | `BUSY` |
| `reset while running` | `BUSY` |
| `cancel a running job` | `CANCELLED` |
| `cancel mid-warmup` | `CANCELLED` |
| `worker raises a bare exception` | `INTERNAL` |
| `clear_result while running` | `BUSY` |

### config

| Fault | Code |
|---|---|
| `document is a list` | `INVALID_CONFIG` |
| `document is a string` | `INVALID_CONFIG` |
| `section is a list` | `INVALID_CONFIG` |
| `value has the wrong type` | `INVALID_CONFIG` |
| `value out of range` | `INVALID_CONFIG` |
| `unknown schema_version` | `INVALID_CONFIG` |
| `file is not JSON` | `INVALID_CONFIG` |
| `file is not UTF-8` | `INVALID_CONFIG` |
| `file does not exist` | `IO_ERROR` |
| `path is a directory` | `IO_ERROR` |
| `both json and path given` | `INVALID_ARGUMENT` |
| `neither json nor path given` | `INVALID_ARGUMENT` |

### service

| Fault | Code |
|---|---|
| `no image loaded` | `NO_IMAGE` |
| `no result yet` | `NO_IMAGE` |
| `two image sources` | `INVALID_ARGUMENT` |
| `empty image source` | `INVALID_ARGUMENT` |
| `from_clipboard without a GUI` | `INVALID_ARGUMENT` |
| `block_ids is not a list` | `INVALID_ARGUMENT` |
| `block_ids has a non-integer` | `INVALID_ARGUMENT` |
| `no block id matches` | `INVALID_ARGUMENT` |
| `highlight an unknown block` | `INVALID_ARGUMENT` |
| `save_result to an empty path` | `INVALID_ARGUMENT` |
| `save_result into an unwritable path` | `IO_ERROR` |
| `annotated scale out of range` | `INVALID_ARGUMENT` |
| `annotated scale would render too much` | `INVALID_ARGUMENT` |
| `overlay with nothing to set` | `INVALID_ARGUMENT` |
| `zoom mode unknown` | `INVALID_ARGUMENT` |
| `zoom percent out of range` | `INVALID_ARGUMENT` |
| `base64 over the size cap` | `INVALID_ARGUMENT` |
| `re-OCR with an empty id list` | `INVALID_ARGUMENT` |
| `re-OCR an id that is not in the document` | `INVALID_ARGUMENT` |
| `re-OCR with no page loaded` | `NO_IMAGE` |
| `re-OCR a page that has no document yet` | `INVALID_ARGUMENT` |

### mcp

| Fault | Code |
|---|---|
| `unknown tool` | `INVALID_ARGUMENT` |
| `missing required argument` | `INVALID_ARGUMENT` |
| `argument of the wrong type` | `INVALID_ARGUMENT` |
| `unknown argument name` | `INVALID_ARGUMENT` |
| `argument out of range` | `INVALID_ARGUMENT` |
| `config value rejected` | `INVALID_CONFIG` |
| `no image loaded` | `NO_IMAGE` |
| `job cancelled by another client` | `CANCELLED` |
| `second run while busy` | `BUSY` |
| `unknown model` | `MODEL_NOT_FOUND` |
| `resource read with no state` | `NO_IMAGE` |
| `unknown resource` | `INVALID_ARGUMENT` |

### offline

| Fault | Code |
|---|---|
| `offline and the bundle is missing` | `MODEL_NOT_FOUND` |
| `gated repo, no token` | `HF_AUTH_REQUIRED` |
| `repo not found` | `MODEL_NOT_FOUND` |
| `network unreachable` | `DOWNLOAD_FAILED` |
| `backend not installed` | `ENGINE_LOAD_FAILED` |
| `bundled weights are corrupt` | `ENGINE_LOAD_FAILED` |

### cli

| Fault | Code |
|---|---|
| `--run without --image` | `NO_IMAGE` |
| `--run on a corrupt image` | `INVALID_IMAGE` |
| `--config pointing at garbage` | `INVALID_CONFIG` |
| `--out into an unwritable path` | `IO_ERROR` |
| `--headless with no action` | `INVALID_ARGUMENT` |
| `--warm-devices with no devices` | `INVALID_ARGUMENT` |
| `--export-config with an empty path` | `INVALID_ARGUMENT` |
---

## What each tool can answer with

A tool only ever returns codes from this list, and every one of them is in the table above. Read
it as "what a harness has to branch on".

| Tool | Codes | Note |
|---|---|---|
| `get_status` | — | always answers; it is how you find out why something else failed |
| `list_models` | `ENGINE_LOAD_FAILED` | the model report itself is offline-safe |
| `get_config` | — | |
| `set_config` | `INVALID_CONFIG`, `INVALID_ARGUMENT` | `details.issues[].path` names the key; nothing is applied on failure |
| `export_config` | `IO_ERROR` | a `path` that cannot be written is the only way |
| `import_config` | `INVALID_CONFIG`, `INVALID_ARGUMENT`, `IO_ERROR` | exactly one of `json` / `path`; a `<redacted>` secret keeps the live value |
| `load_image` | `INVALID_ARGUMENT`, `INVALID_IMAGE`, `IO_ERROR` | one source only; base64 may be MIME-wrapped or carry newlines, and is capped at 64 MB decoded |
| `get_image_info` | `NO_IMAGE` | |
| `detect` | `NO_IMAGE`, `BUSY`, `MODEL_NOT_FOUND`, `ENGINE_LOAD_FAILED`, `CANCELLED`, `INVALID_CONFIG` | overrides are validated as config before the job starts |
| `run_pipeline` | as `detect`, plus `OCR_FAILED` | one activity entry either way; `details` of a job error is the job's own error |
| `cancel_job` | — | `cancelled: false` with nothing to cancel is a success, not an error |
| `get_result` | `NO_IMAGE`, `INVALID_ARGUMENT` | `block_ids` must be a list of integers; none of them existing is `INVALID_ARGUMENT`, empty is not |
| `save_result` | `NO_IMAGE`, `INVALID_ARGUMENT`, `IO_ERROR` | |
| `get_annotated_image` | `NO_IMAGE`, `INVALID_ARGUMENT` | `scale` outside 0.05–8.0 is refused, and the rendered area is capped so one call cannot exhaust RAM |
| `set_overlay_visible` | `INVALID_ARGUMENT` | at least one of the three arguments must be given |
| `set_zoom` | `INVALID_ARGUMENT` | mode must be `fit` or `percent`; percent 10–800 |
| `highlight_block` | `NO_IMAGE`, `INVALID_ARGUMENT` | an id that is not in the result answers `found: false` — a question, not an error |
| `reset` | `BUSY` | cancel first; reset never cancels for you |
| `unload_engines` | — | answers even when nothing was loaded; memory claims are in `unload_engines`' own description, not in a code |
| `warm_engines` | `BUSY`, `ENGINE_LOAD_FAILED` | returns a job handle; a failing load shows up in `get_status` and in `warmup_report`, not here |
| `reocr_blocks` | `INVALID_ARGUMENT`, `INVALID_CONFIG`, `NO_IMAGE`, `BUSY`, `CANCELLED` | an id that is not in the document answers `INVALID_ARGUMENT` with `unknown` and `available`; a well-formed ask with no page loaded is `NO_IMAGE`, and a page with no result yet is `INVALID_ARGUMENT` saying so |

## Resources

`config://current`, `models://list` and `log://mcp` always answer. `result://latest`,
`image://current` and `image://annotated` fail with `NO_IMAGE` — and since P8.4 that code reaches
the client, in `error.data`, instead of stopping in the server log as a generic
`Error reading resource …`.

## Invariants (the part worth keeping)

1. **No bare exceptions.** Every matrix case asserts the raised object *is* a `StudioError`, not a
   `TypeError` from `pathlib`, a `ValueError` from `int()`, or a `UnicodeDecodeError` from a
   UTF-16 config file. Those four were real bugs, found by writing this table.
2. **Every refusal is one activity entry**, with the same code the caller got — including the
   calls the MCP SDK rejects before any handler runs (bad argument, unknown tool). Before P8.4 a
   client could hammer the server with garbage and the window would show nothing.
3. **Nothing is half-applied.** A refused config patch or import leaves the live config
   byte-identical; a cancelled or failed job leaves the previous page and result in place.
4. **`INTERNAL` stays rare.** `test_internal_is_reserved_for_the_engine_breaking_its_contract`
   pins the exact set of rows allowed to produce it, so a new one has to be *added to a test*
   before it can ship.
5. **The app is usable afterwards**, always — the last test in the file runs the whole table and
   then processes a page.

## Regenerating the tables

The matrix tables above are generated from `MATRIX` in `tests/test_p8_errors.py`:

```bash
QT_QPA_PLATFORM=offscreen python3 scripts/gen_error_matrix.py
```

Add a row to `MATRIX`, add the case, run the generator, and the documentation cannot drift —
`test_every_row_of_the_matrix_is_documented` checks it row by row either way.
