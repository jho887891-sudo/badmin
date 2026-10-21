#!/usr/bin/env python
"""A09: assertion provenance registry + loader + generator.

Every gate assertion in a result.json must be traceable to

  * the source file and line that computes it,
  * the SPEC clause that freezes its threshold,
  * the unit it is expressed in,
  * the data file it reads,
  * the acceptance decision it encodes (and the decision actually recorded).

The registry is a JSON document (see tools/queue/provenance/assertion_registry.json).
This module loads it, verifies that every cited source line really exists and still
contains the anchored text, and generates a per-run provenance table from a real
result.json.  An unregistered gate is a hard failure (non-zero exit): a gate whose
origin nobody recorded is exactly the "looks like a PASS, means nothing" failure the
master spec forbids.

Sources live on other branches of this repository (the P2/P3 gate scripts and the
PHASE_P2_*/PHASE_P3_* documents are not on every ref), so each registry entry may
carry a "ref".  Resolution order for a cited path: the worktree, then
"git show <ref>:<path>".  Line numbers were verified by reading the real files.

Usage
-----
    python tools/queue/assertion_provenance.py --self-check
    python tools/queue/assertion_provenance.py --result-json R.json --out prov.json
    python tools/queue/assertion_provenance.py --result-json R.json --strict-units

Exit codes: 0 = every gate registered and every source verified,
            2 = unregistered gate / missing field / source line mismatch, 1 = usage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REGISTRY_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "tools" / "queue" / "provenance" / "assertion_registry.json"

REQUIRED_ENTRY_FIELDS = ("unit", "data_file", "acceptance", "acceptance_decision")

#: what a recorded gate value MEANS.  "pass_when_true" is the only polarity used in
#: P2/P3 today; adding a new one is a deliberate act, not an accident.
ACCEPTANCE_POLARITIES = {"pass_when_true", "pass_when_false"}


# -------------------------------------------------------------------- reading
class SourceResolver:
    """Read a cited file from the worktree, falling back to a git ref."""

    def __init__(self, repo_root=None, runner=None):
        self.repo_root = Path(repo_root or REPO_ROOT)
        self._cache: dict = {}
        self._runner = runner or self._git_show

    def _git_show(self, ref: str, path: str):
        try:
            completed = subprocess.run(
                ["git", "show", "%s:%s" % (ref, path)],
                cwd=str(self.repo_root), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout

    def read(self, path: str, ref=None):
        key = (path, ref)
        if key in self._cache:
            return self._cache[key]
        text = None
        candidate = self.repo_root / path
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = None
        if text is None and ref:
            text = self._runner(ref, path)
        self._cache[key] = text
        return text

    def lines(self, path: str, ref=None):
        text = self.read(path, ref=ref)
        if text is None:
            return None
        return text.splitlines()


# ------------------------------------------------------------------- registry
class ProvenanceRegistry:
    """Loader for the assertion provenance registry."""

    def __init__(self, payload: dict, path=None):
        self.payload = payload or {}
        self.path = path
        self.entries = self.payload.get("entries") or {}

    @classmethod
    def load(cls, path=None) -> "ProvenanceRegistry":
        path = Path(path or DEFAULT_REGISTRY)
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle), path=path)

    def gates(self) -> list:
        return sorted(self.entries)

    def entry(self, gate: str):
        return self.entries.get(gate)

    # -- structure -----------------------------------------------------------
    def verify_entries(self) -> list:
        problems: list = []
        for gate, entry in sorted(self.entries.items()):
            if not isinstance(entry, dict):
                problems.append({"code": "ENTRY_NOT_OBJECT", "gate": gate,
                                 "reason": "%r is not an object" % gate})
                continue
            for field in REQUIRED_ENTRY_FIELDS:
                if not entry.get(field):
                    problems.append({
                        "code": "ENTRY_INCOMPLETE", "gate": gate,
                        "reason": "gate %r has no %r: a gate without unit, data file, "
                                  "acceptance or decision is not traceable" % (gate, field),
                    })
            for key in ("sources", "spec_clauses"):
                rows = entry.get(key)
                if not isinstance(rows, list) or not rows:
                    problems.append({
                        "code": "ENTRY_INCOMPLETE", "gate": gate,
                        "reason": "gate %r has no %r" % (gate, key),
                    })
                    continue
                for row in rows:
                    if not isinstance(row, dict) or not row.get("path") \
                            or not isinstance(row.get("line"), int):
                        problems.append({
                            "code": "ENTRY_INCOMPLETE", "gate": gate,
                            "reason": "gate %r has a %s row without path+line: %r"
                                      % (gate, key, row),
                        })
        return problems

    # -- source lines --------------------------------------------------------
    def verify_sources(self, resolver=None) -> list:
        resolver = resolver or SourceResolver()
        problems: list = []
        for gate, entry in sorted(self.entries.items()):
            if not isinstance(entry, dict):
                continue
            for key in ("sources", "spec_clauses"):
                for row in entry.get(key) or ():
                    if not isinstance(row, dict):
                        continue
                    path, line = row.get("path"), row.get("line")
                    if not path or not isinstance(line, int):
                        continue
                    lines = resolver.lines(path, ref=row.get("ref"))
                    if lines is None:
                        problems.append({
                            "code": "SOURCE_UNRESOLVED", "gate": gate,
                            "reason": "%s: cannot read %s (ref %r)"
                                      % (key, path, row.get("ref")),
                        })
                        continue
                    if line < 1 or line > len(lines):
                        problems.append({
                            "code": "SOURCE_LINE_OUT_OF_RANGE", "gate": gate,
                            "reason": "%s:%d does not exist (%s has %d lines)"
                                      % (path, line, path, len(lines)),
                        })
                        continue
                    anchor = row.get("line_contains")
                    if anchor and anchor not in lines[line - 1]:
                        problems.append({
                            "code": "SOURCE_ANCHOR_MISMATCH", "gate": gate,
                            "reason": "%s:%d no longer contains %r (found %r): the cited "
                                      "provenance drifted"
                                      % (path, line, anchor, lines[line - 1].strip()[:120]),
                        })
        return problems

    def verify(self, resolver=None) -> list:
        return self.verify_entries() + self.verify_sources(resolver=resolver)


# ------------------------------------------------------------------ generation
def scan_result_gates(result_payload: dict) -> dict:
    """Every gate the result.json asserts, as gate -> recorded decision."""
    gates = {}
    if not isinstance(result_payload, dict):
        return gates
    for key in ("gates", "gate_values"):
        rows = result_payload.get(key)
        if isinstance(rows, dict):
            for name, value in rows.items():
                gates.setdefault(str(name), value)
    nested = result_payload.get("diagnostics")
    if isinstance(nested, dict):
        for key in ("static_gates_as_diagnostics", "gates"):
            rows = nested.get(key)
            if isinstance(rows, dict):
                for name, value in rows.items():
                    gates.setdefault("diagnostics.%s" % name, value)
    return gates


def generate(result_payload, registry: ProvenanceRegistry, source_label=None) -> tuple:
    """Build the per-run provenance table; unregistered gates are failures."""
    observed = scan_result_gates(result_payload)
    problems: list = []
    rows: list = []
    for gate, value in sorted(observed.items()):
        # diagnostics.* keep the real gate name after the prefix so a demoted static
        # gate still traces back to the same assertion
        lookup = gate[len("diagnostics."):] if gate.startswith("diagnostics.") else gate
        entry = registry.entry(lookup)
        if entry is None:
            problems.append({
                "code": "UNREGISTERED_GATE",
                "gate": gate,
                "reason": "gate %r (registry key %r) appears in %s but has no provenance "
                          "registry entry"
                          % (gate, lookup, source_label or "the result"),
            })
            continue
        recorded = value
        polarity = entry.get("acceptance_decision")
        if polarity not in ACCEPTANCE_POLARITIES:
            problems.append({
                "code": "DECISION_POLARITY_UNKNOWN",
                "gate": gate,
                "reason": "gate %r records acceptance_decision %r; only %s are "
                          "defined, so nobody can say what a true value means"
                          % (gate, polarity, ", ".join(sorted(ACCEPTANCE_POLARITIES))),
            })
        if not isinstance(recorded, bool):
            # a missing / NaN / empty / "unknown" gate is not a PASS (master spec
            # section 4: missing, NaN and empty gates must never be accepted)
            problems.append({
                "code": "NONBOOLEAN_GATE",
                "gate": gate,
                "reason": "gate %r recorded %r (%s); a non-boolean gate is UNKNOWN, "
                          "never a PASS"
                          % (gate, recorded, type(recorded).__name__),
            })
        rows.append({
            "gate": gate,
            "recorded_decision": recorded,
            "acceptance_decision": polarity,
            "unit": entry.get("unit"),
            "data_file": entry.get("data_file"),
            "acceptance": entry.get("acceptance"),
            "sources": entry.get("sources") or [],
            "spec_clauses": entry.get("spec_clauses") or [],
            "note": entry.get("note"),
        })
    payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "task": "A09",
        "registry": registry.path.as_posix() if registry.path else None,
        "source": source_label,
        "gate_result": (result_payload or {}).get("gate_result")
        if isinstance(result_payload, dict) else None,
        "run_id": (result_payload or {}).get("run_id")
        if isinstance(result_payload, dict) else None,
        "gates": rows,
        "problems": problems,
        "totals": {"observed": len(observed), "registered": len(rows),
                   "problems": len(problems)},
    }
    return payload, problems


# ----------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="A09 assertion provenance")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--self-check", action="store_true",
                        help="verify the registry itself (fields + cited source lines)")
    parser.add_argument("--list", action="store_true", help="print the registered gates")
    args = parser.parse_args(argv)

    registry = ProvenanceRegistry.load(args.registry)

    if args.list:
        for gate in registry.gates():
            entry = registry.entry(gate)
            print("%-32s unit=%-10s decision=%s"
                  % (gate, entry.get("unit"), entry.get("acceptance_decision")))
        return 0

    if args.self_check:
        problems = registry.verify()
        if problems:
            print("FAIL (A09_REGISTRY): %d problem(s) in %s"
                  % (len(problems), args.registry), file=sys.stderr)
            for problem in problems:
                print("  - %s [%s]: %s" % (problem["code"], problem.get("gate"),
                                           problem["reason"]), file=sys.stderr)
            return 2
        print("PASS (A09_REGISTRY): %d gates, every cited source line verified"
              % len(registry.gates()))
        return 0

    if not args.result_json:
        print("--result-json or --self-check is required", file=sys.stderr)
        return 1
    path = Path(args.result_json)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            result_payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print("FAIL (RESULT_UNREADABLE): %s: %s" % (path.as_posix(), exc), file=sys.stderr)
        return 1

    payload, problems = generate(result_payload, registry,
                                 source_label=path.as_posix())
    if args.out:
        Path(args.out).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + chr(10),
            encoding="utf-8")
    if problems:
        print("FAIL (A09_PROVENANCE): %d problem(s) for %s"
              % (len(problems), path.as_posix()), file=sys.stderr)
        for problem in problems:
            print("  - %s [%s]: %s" % (problem["code"], problem.get("gate"),
                                       problem["reason"]), file=sys.stderr)
        return 2
    print("PASS (A09_PROVENANCE): %s gates %d/%d registered"
          % (path.as_posix(), payload["totals"]["registered"], payload["totals"]["observed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
