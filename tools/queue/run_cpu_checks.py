#!/usr/bin/env python
"""A10: one-command reproduction of the pure-CPU queue checks.

Runs, in order, and reports the real exit code of each step:

1. queue state structural validation   (tools/queue/queue_state.py --validate)
2. queue/repo consistency gate         (tools/queue/check_consistency.py)
3. engineering index freshness         (tools/queue/a01_engineering_index.py --check)
4. the offline test suite              (python -m pytest tests/queue -q)

Nothing here launches Isaac Sim / Kit: the only external programs are git, gh
and this interpreter. See docs/queue/README_zh.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STEPS = (
    ("queue state validate", [sys.executable, "tools/queue/queue_state.py", "--validate"]),
    ("consistency gate", [sys.executable, "tools/queue/check_consistency.py"]),
    ("engineering index", [sys.executable, "tools/queue/a01_engineering_index.py", "--check"]),
    ("offline tests", [sys.executable, "-m", "pytest", "tests/queue", "-q"]),
)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    results = []
    for label, command in STEPS:
        print("== %s: %s" % (label, " ".join(command)))
        proc = subprocess.run(command, cwd=str(REPO))
        results.append((label, proc.returncode))
        print("-- %s exit=%d" % (label, proc.returncode))
    failed = [label for label, code in results if code != 0]
    print("")
    print("summary: %d/%d steps passed" % (len(results) - len(failed), len(results)))
    for label, code in results:
        print("  %-22s exit=%d" % (label, code))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
