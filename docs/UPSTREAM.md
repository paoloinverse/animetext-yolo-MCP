# Upstream: two behaviours in `dghs-imgutils` that bend promises this app makes

Drafted 2026-11-25 against **dghs-imgutils 0.19.0** — the version this project installs, and
the latest on PyPI on that date. **Neither issue has been filed**: this repository has no
GitHub credentials in it, so the drafts below are written to be pasted into
<https://github.com/deepghs/imgutils/issues/new> by whoever holds an account. Every line
reference was read out of the installed wheel, not out of a changelog.

They are here because both behaviours are invisible to a status call, produce symptoms that
point at *our* code, and cost a day each to diagnose. `--doctor`
([`core/doctor.py`](animetext_studio/core/doctor.py)) now names both before anything is
loaded; this file is the half that belongs upstream.

---

## Issue 1 — importing `imgutils` can `pip install` packages, with no opt-out

> **Title:** `_ensure_onnxruntime()` installs packages at import time; please add an opt-out
>
> **Project:** deepghs/imgutils · **Version:** 0.19.0 · **Python:** any

### What happens

`imgutils/utils/onnxruntime.py` decides at **import time** whether `onnxruntime` is
available, and installs it if it is not:

```python
# imgutils/utils/onnxruntime.py
 10  from hbutils.system import pip_install
 …
 17  def _ensure_onnxruntime():
 18      try:
 19          import onnxruntime
 20      except (ImportError, ModuleNotFoundError):
 21          logging.warning('Onnx runtime not installed, preparing to install ...')
 22          if shutil.which('nvidia-smi'):
 23              logging.info('Installing onnxruntime-gpu ...')
 24              pip_install(['onnxruntime-gpu'], silent=True)
 25          else:
 26              logging.info('Installing onnxruntime (cpu) ...')
 27              pip_install(['onnxruntime'], silent=True)
 …
 30  _ensure_onnxruntime()
```

`imgutils.utils.onnxruntime` is reached from `imgutils.generic`, so any of

```python
from imgutils.generic import YOLOModel          # or .yolo, .yoloseg, …
```

may run `pip install` as a side effect of the import statement.

### Why it matters

* **A library import is not allowed to change the environment.** It writes into the
  interpreter that only meant to read from it: new packages appear, and nothing in
  `pip freeze` at the start of the run predicted them.
* **It breaks the offline promise.** An air-gapped or egress-restricted machine — the
  normal deployment for on-prem document processing — does not get "no network": it gets a
  pip run that hangs until its own timeout, in the middle of loading a model, with a
  traceback that names `hbutils`, not the network.
* **`silent=True` hides it.** The `logging.warning` on line 21 is the only announcement, and
  a host application that has not configured logging sees nothing at all. The first sign is
  usually a wall of `Collecting onnxruntime` on stderr.
* **PEP 668 makes it fail differently on the same code.** On a Debian/Ubuntu system Python
  (externally-managed), pip refuses outright — so the behaviour is "silently installs" on
  one machine and "explains nothing" on the next.
* **The GPU choice is a guess.** `shutil.which('nvidia-smi')` says a driver is present, not
  that the wheel it picks will load against the installed torch/CUDA — which is how you end
  up with `onnxruntime-gpu` on a machine that needed the CPU build.

### What we would like

Any of these, in decreasing order of preference:

1. An opt-out honoured before anything is installed, e.g. `IMGUTILS_NO_AUTO_INSTALL=1`
   (or `IMGUTILS_AUTO_INSTALL=0`), which raises instead:
   `ImportError: onnxruntime is required by imgutils; install it with 'pip install
   onnxruntime' or 'pip install onnxruntime-gpu'.`
2. Or simply raise that `ImportError` unconditionally and let the caller install. One
   implicit `pip install` inside a library import is a dependency decision the library
   should not be making.
3. Or at minimum: keep the behaviour but announce it loudly and non-silently — name the
   exact command, and say how to turn it off.

### What we do in the meantime

* `animetext-studio --doctor` checks for `onnxruntime` **without importing imgutils** and
  says, in one line, that the next `import imgutils` would reach the network — plus the
  `pip install` to run first.
* The README's Troubleshooting table names the symptom (a wall of `Collecting onnxruntime`
  on first import) and the fix (install a provider up front).

---

## Issue 2 — `ast.Str` / `ast.Num` were removed in Python 3.14; detection cannot load

> **Title:** `_safe_eval_names_str` uses `ast.Str`/`ast.Num`, removed in Python 3.14
>
> **Project:** deepghs/imgutils · **Version:** 0.19.0 · **Python:** 3.14+

### What happens

Label metadata is parsed with the `ast` module's deprecated literal aliases:

```python
# imgutils/generic/yolo.py
451  def _safe_eval_names_str(names_str):
 …
466      node = ast.parse(names_str, mode='eval')
468          if isinstance(key, (ast.Str, ast.Num)):
469              key = ast.literal_eval(key)
 …
474          if isinstance(value, (ast.Str, ast.Num)):
475              value = ast.literal_eval(value)
```

`ast.Str` and `ast.Num` were deprecated in Python 3.8 and **removed in 3.14** (the
deprecation warning on 3.13 says so in as many words: *"ast.Str is deprecated and will be
removed in Python 3.14; use ast.Constant instead"*). On 3.14 the attribute lookup raises, so
the first model whose metadata carries a `names` string fails:

* `imgutils/generic/yolo.py:612` — `names_map = _safe_eval_names_str(model_metadata.custom_metadata_map['names'])`
* `imgutils/generic/yoloseg.py:341` — the same call in the segmentation model.

The failure lands inside model loading, as an `AttributeError` about the `ast` module. The
application cannot tell the user anything more useful than "the detection backend broke on
this Python", because the traceback is entirely inside imgutils.

### Why it matters

It is a one-line migration that decides whether the whole 3.14 generation of interpreters
can use this package. Until it happens, downstreams have to pin `python_requires` to
`<3.14` — which is what we now document — and that gets harder to walk back the longer it
stands.

### The patch

`ast.Constant` has covered both string and number literals since 3.8, so the change is
mechanical and back-compatible to the declared floor:

```diff
-        if isinstance(key, (ast.Str, ast.Num)):
+        if isinstance(key, ast.Constant):
             key = ast.literal_eval(key)
         else:
             raise RuntimeError(f"Invalid key type: {key!r}, this should be a bug, "
                                f"please open an issue to dghs-imgutils.")  # pragma: no cover

-        if isinstance(value, (ast.Str, ast.Num)):
+        if isinstance(value, ast.Constant):
             value = ast.literal_eval(value)
```

(`ast.literal_eval(node)` accepts an `ast.Constant` node directly. If the intent is "string
or number only", `isinstance(key, ast.Constant) and isinstance(key.value, (str, int, float))`
is the exact equivalent, and it also stops accepting `None`/`True` literals, which the old
pair did not reject.)

### Upstream already has this fix — unmerged

Checked against the live repository on 2026-11-26: **PR #178**, *Fix (some) Python 3.14
incompatibility* (opened 2026-06-07, still open, not merged), rewrites exactly these two
`isinstance` lines to `ast.Constant` with a `.value` type check — and relaxes `numpy<2` to
`numpy<3` in `requirements.txt` while it is there, which is the other behaviour this project
documents. PyPI's latest is still 0.19.0, so the published wheel still carries the bug.

That changes what is worth filing: not a bug report, which would duplicate a PR, but a short
issue (or a comment on #178) saying the fix exists, no release has taken it, and a downstream
pinned itself to Python ≤ 3.13 in the meantime. The draft below stays as the record of what
the failure is and why `ast.Constant` is the right replacement.

### What we do in the meantime

* `--doctor` reports `Python 3.14: dghs-imgutils cannot import on 3.14+` as a blocking
  finding for the detection stage — with this file named in the hint — and the app's
  documented floor stays **3.10 – 3.13**.
* The README's Troubleshooting table carries the symptom and the workaround (run the
  detection stage on ≤ 3.13).

---

## When either is fixed

1. Re-read the two line references above against the new version; both drift.
2. In `core/doctor.py`, the facts live in `IMGUTILS_PYTHON_CEILING`, `UPSTREAM_AUTO_INSTALL`
   and `UPSTREAM_AST`; raise the ceiling or drop the finding once upstream says so.
3. `tests/test_p9_doctor.py::TestTheUpstreamDrafts::test_the_quoted_source_is_still_the_installed_source`
   re-reads the installed package and fails when these lines change — that failure is the
   reminder, not a bug. The same class pins the shape of this file: both issues present, both
   citing the file and lines they came from, both saying plainly that neither has been filed.
4. Update the README rows (Install: *The `dghs-imgutils` wrinkle*, and the two
   Troubleshooting entries) and this file's status line.
