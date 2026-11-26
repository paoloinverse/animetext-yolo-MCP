#!/usr/bin/env python3
"""Regenerate the matrix tables in `docs/ERROR-MATRIX.md` from the test module.

`MATRIX` in `tests/test_p8_errors.py` is the source of truth: it is what the tests assert. The
documentation is generated from it so the two cannot drift, and
`test_every_row_of_the_matrix_is_documented` checks the rows individually anyway — this script
only saves the typing.

Usage:  QT_QPA_PLATFORM=offscreen python3 scripts/gen_error_matrix.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

DOC = ROOT / "docs" / "ERROR-MATRIX.md"
START = "## The matrix"


def render() -> str:
    import test_p8_errors as matrix_module           # noqa: PLC0415 (path set above)

    rows = [(entry, fault, code.value)
            for entry, fault, code in matrix_module.MATRIX]
    out = [START, "",
           f"{len(rows)} faults, grouped by the door they come through. Each row is a test in",
           "`tests/test_p8_errors.py`; the code shown is what that test asserts, today."]
    last = None
    for entry, fault, code in rows:
        if entry != last:
            out += ["", f"### {entry}", "", "| Fault | Code |", "|---|---|"]
            last = entry
        out.append(f"| `{fault}` | `{code}` |")
    return "\n".join(out) + "\n"


def splice(text: str, block: str) -> str:
    """Replace the generated section, leaving the prose before and after it untouched."""
    start = text.index(START)
    end = text.index("\n---", start)
    return text[:start] + block + text[end + 1:]


def main() -> int:
    check = "--check" in sys.argv[1:]
    current = DOC.read_text(encoding="utf-8")
    updated = splice(current, render())
    if updated == current:
        print(f"{DOC.name}: matrix tables are up to date")
        return 0
    if check:
        print(f"{DOC.name}: matrix tables are STALE — run scripts/gen_error_matrix.py")
        return 1
    DOC.write_text(updated, encoding="utf-8")
    print(f"{DOC.name}: matrix tables rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
