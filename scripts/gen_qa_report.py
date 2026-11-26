#!/usr/bin/env python3
"""Regenerate the checklist tables in `docs/QA.md` from `tests/test_p8_qa.py`.

`REQUIREMENTS` in the test module is the source of truth: it is what the tests assert, and
`TestTheTableItself` refuses to pass a row whose evidence does not exist. The document is
generated from it for the same reason `docs/ERROR-MATRIX.md` is — a checklist maintained by
hand is a checklist that quietly stops describing the product.

Usage:  python3 scripts/gen_qa_report.py [--check]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

DOC = ROOT / "docs" / "QA.md"
START = "## The checklist"
END = "\n---"


def _evidence_cell(cites: tuple[str, ...]) -> str:
    """Test ids are long; show the file once and the tail of the path."""
    parts = []
    for cite in cites:
        if cite.startswith("tests/"):
            file_, _, rest = cite[len("tests/"):].partition("::")
            parts.append(f"`{file_}` " + (f"`{rest}`" if rest else ""))
        elif cite.startswith("qa:"):
            parts.append(f"check `{cite[3:].strip()}` in `tests/test_p8_qa.py`")
        elif cite.startswith("cmd:"):
            parts.append(f"`{cite[4:].strip()}`")
        else:
            parts.append(cite)
    return "<br>".join(parts)


def render() -> str:
    import test_p8_qa as qa                          # noqa: PLC0415 (path set above)

    rows = qa.REQUIREMENTS
    groups = dict(qa.GROUPS)
    signoff = qa.SIGNOFF
    out = [START, "",
           f"{len(rows)} requirements, each with the evidence that proves it. A row cites a test "
           f"node id, a command, or a check in `tests/test_p8_qa.py`; "
           f"`TestTheTableItself::test_evidence_points_at_tests_that_exist` fails the suite if "
           f"one of those stops existing.", "",
           f"Suite at sign-off ({signoff['date']}): **{signoff['offline_tests']} offline tests** "
           f"(+{signoff['model_tests']} `-m models`), {signoff['tools']} MCP tools, "
           f"{signoff['resources']} resources, {signoff['error_codes']} error codes, "
           f"{signoff['matrix_faults']} matrix faults. "
           f"`test_the_suite_has_not_shrunk_since_sign_off` treats those as floors.", ""]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.id.split("-")[0]] = counts.get(row.id.split("-")[0], 0) + 1
    out += ["| Group | Requirements |", "|---|---|"]
    for key, title in qa.GROUPS:
        if counts.get(key):
            out.append(f"| [{key}](#{key.lower()}) — {title} | {counts[key]} |")
    out.append(f"| **Total** | **{len(rows)}** |")

    for key, title in qa.GROUPS:
        mine = [row for row in rows if row.id.split("-")[0] == key]
        if not mine:
            continue
        out += ["", f"### {key} — {title}", "",
                "| ID | Requirement | Source | Status | Evidence |", "|---|---|---|---|---|"]
        for row in mine:
            status = row.status if row.status != "met" else "✅"
            note = f"<br>*{row.note}*" if row.note else ""
            out.append(f"| `{row.id}` | {row.text} | {row.source} | {status} | "
                       f"{_evidence_cell(row.evidence)}{note} |")

    out += ["", "## Still open, on purpose", "",
            "Not requirements and not defects. Each one is a decision that was argued, and the "
            "argument is the thing to re-read before reopening it.", "",
            "| Item | Why it stays open |", "|---|---|"]
    for item, reason in qa.FOLLOW_UPS:
        out.append(f"| {item} | {reason} |")
    return "\n".join(out) + "\n"


def splice(text: str, block: str) -> str:
    start = text.index(START)
    end = text.index(END, start)
    return text[:start] + block + text[end + 1:]


def main() -> int:
    check = "--check" in sys.argv[1:]
    if not DOC.is_file():
        if check:
            print(f"{DOC.name}: missing — run scripts/gen_qa_report.py")
            return 1
        DOC.write_text(render(), encoding="utf-8")
        print(f"{DOC.name}: created")
        return 0
    current = DOC.read_text(encoding="utf-8")
    updated = splice(current, render())
    if updated == current:
        print(f"{DOC.name}: checklist tables are up to date")
        return 0
    if check:
        print(f"{DOC.name}: checklist tables are STALE — run scripts/gen_qa_report.py")
        return 1
    DOC.write_text(updated, encoding="utf-8")
    print(f"{DOC.name}: checklist tables rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
