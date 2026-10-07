#!/usr/bin/env python3
"""Run every project test suite and print an evidence table (coordinator tool).

Usage: ./env_isaaclab/bin/python tools/run_all_tests.py [--pattern tests/**/test_*.py]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover(pattern: str) -> list[Path]:
    files = sorted(ROOT.glob(pattern))
    return [f for f in files if f.name.startswith('test_')]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='tests/**/test_*.py')
    args = parser.parse_args()
    files = discover(args.pattern)
    if not files:
        print('no test files found for pattern', args.pattern)
        return 1
    failures = 0
    total_tests = 0
    print('%-52s %s' % ('suite', 'result'))
    print('-' * 72)
    for path in files:
        rel = path.relative_to(ROOT)
        proc = subprocess.run([sys.executable, str(path)], cwd=ROOT,
                              capture_output=True, text=True)
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        summary = tail[-1] if tail else '(no output)'
        ran = next((l for l in reversed(tail) if l.strip().startswith('Ran ')), '')
        count = 0
        if ran:
            try:
                count = int(ran.split()[1])
            except (IndexError, ValueError):
                count = 0
        else:
            # pytest-style suites report "<N> passed in <t>s" instead of unittest's "Ran N tests".
            # Without this branch a pytest file would show rc=0 with 0 tests, i.e. a silent zero-count pass.
            # pytest decorates the summary with '=' padding ("===== 16 passed in 2.9s ====="), so the count is the
            # integer token *before* the word "passed", not the first token of the line.
            passed = next((l for l in reversed(tail) if ' passed' in l), '')
            m = re.search(r'(\d+)\s+passed', passed) if passed else None
            count = int(m.group(1)) if m else 0
        total_tests += count
        ok = proc.returncode == 0
        failures += 0 if ok else 1
        print('%-52s %s   (rc=%d, %d tests)' % (rel, 'OK' if ok else 'FAIL', proc.returncode, count))
    print('-' * 72)
    print('suites: %d  failures: %d  tests: %d' % (len(files), failures, total_tests))
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
