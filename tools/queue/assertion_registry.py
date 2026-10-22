#!/usr/bin/env python
"""R10-C3: assertion-provenance registry (pure CPU).

Every claim the batch publishes as ASSERTED must be independently
re-derivable: source file + line + quote, SPEC clause, unit, run id or trace
path, and a threshold registered before the measurement.  A claim that is not
proven stays UNKNOWN, and an UNKNOWN claim may carry no number at all (no
"about 6%", no "2693 N", no placeholder value).

Exit codes: 0 = registry satisfies the contract, 1 = violations, 3 = the
checker could not run (missing/unreadable registry - fail closed).

This tool never measures anything itself: it only proves that an ASSERTED
claim is anchored, and that an UNKNOWN claim is not quietly numeric.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"
REGISTRY_PATH = "management/morph_one_rs/R10_C_ASSERTION_REGISTRY.json"

STATUSES = ("ASSERTED", "UNKNOWN", "NOT_RUN")
REQUIRED_FIELDS = (
    "id", "task", "claim", "status", "source", "spec_clause", "unit",
    "evidence", "threshold",
)

# Identifiers that legitimately contain digits even inside an UNKNOWN claim.
IDENTIFIER_PATTERNS = (
    r"\b[A-H][0-9]{1,2}\b",            # batch clause ids: C1, A2, H4
    r"\bP[0-9]-[0-9]{2}\b",            # queue task ids: P0-07, P5-07
    r"\bR10\b",
    r"\b[0-9a-f]{7,40}\b",             # commit shas
    r"\b[0-9]+\.[0-9]+\.[0-9]+\b",     # versions: 3.11.9
    r"\bIssue\s*#[0-9]+\b",
    r"\bPR\s*#[0-9]+\b",
)
MEASUREMENT_UNITS = (
    "m", "mm", "cm", "m/s", "rad", "deg", "s", "ms", "N", "N*m", "Hz", "px",
    "kg", "count", "ratio", "percent", "%", "boolean", "verdict",
)

CODES = {
    "ENTRY_NOT_OBJECT": "entry is not a JSON object",
    "FIELD_MISSING": "a required provenance field is absent",
    "STATUS_INVALID": "status is not ASSERTED/UNKNOWN/NOT_RUN",
    "ID_DUPLICATE": "two entries share the same assertion id",
    "ASSERTED_SOURCE_INVALID": "source file/line/quote is missing or does not resolve",
    "ASSERTED_QUOTE_MISMATCH": "the quoted text is not on the declared source line",
    "SPEC_CLAUSE_INVALID": "spec_clause is not a batch clause id (A-H + number)",
    "ASSERTED_UNIT_MISSING": "an ASSERTED claim must state its unit",
    "ASSERTED_EVIDENCE_INVALID": "exactly one run_id or trace_path is required and must resolve",
    "ASSERTED_THRESHOLD_INVALID": "threshold must be registered and carry value+unit",
    "THRESHOLD_NOT_PREREGISTERED": "threshold was not pre-registered before measurement",
    "UNKNOWN_CARRIES_NUMBER": "an UNKNOWN claim must carry no number",
    "UNKNOWN_CARRIES_VALUE": "an UNKNOWN claim must not carry unit/threshold/evidence values",
}


class Violation:
    __slots__ = ("code", "entry_id", "detail")

    def __init__(self, code: str, entry_id: str, detail: str):
        if code not in CODES:
            raise KeyError("unknown violation code: %s" % code)
        self.code = code
        self.entry_id = entry_id
        self.detail = detail

    def __str__(self) -> str:
        return "%s [%s] %s: %s" % (self.code, self.entry_id, CODES[self.code], self.detail)


def strip_identifiers(text: str) -> str:
    stripped = text
    for pattern in IDENTIFIER_PATTERNS:
        stripped = re.sub(pattern, " ", stripped)
    return stripped


def has_number(text: str) -> bool:
    return re.search(r"[0-9]", strip_identifiers(text)) is not None


def _tree_paths(repo: Path, commit: str, cache: dict):
    if commit not in cache:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", check=True,
            ).stdout
            cache[commit] = set(out.splitlines())
        except (OSError, subprocess.CalledProcessError):
            cache[commit] = set()
    return cache[commit]


def check_source(repo: Path, entry: dict, violations: list, cache: dict) -> None:
    entry_id = entry.get("id", "<no-id>")
    source = entry.get("source")
    if not isinstance(source, dict):
        violations.append(Violation("ASSERTED_SOURCE_INVALID", entry_id, "source is not an object"))
        return
    file_ref = source.get("file")
    line = source.get("line")
    quote = source.get("quote")
    if not isinstance(file_ref, str) or not file_ref.strip():
        violations.append(Violation("ASSERTED_SOURCE_INVALID", entry_id, "source.file missing"))
        return
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        violations.append(Violation("ASSERTED_SOURCE_INVALID", entry_id,
                                    "source.line must be a positive integer, got %r" % (line,)))
        return
    if not isinstance(quote, str) or not quote.strip():
        violations.append(Violation("ASSERTED_SOURCE_INVALID", entry_id,
                                    "source.quote missing: a line number alone is not provenance"))
        return
    path = repo / file_ref
    if path.is_file():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line > len(lines):
            violations.append(Violation("ASSERTED_SOURCE_INVALID", entry_id,
                                        "%s has only %d lines, line %d requested"
                                        % (file_ref, len(lines), line)))
            return
        if quote.strip() not in lines[line - 1]:
            violations.append(Violation(
                "ASSERTED_QUOTE_MISMATCH", entry_id,
                "line %d of %s is %r" % (line, file_ref, lines[line - 1].strip()[:120])))
        return
    commit = source.get("commit")
    if isinstance(commit, str) and commit.strip():
        if file_ref not in _tree_paths(repo, commit, cache):
            violations.append(Violation("ASSERTED_SOURCE_INVALID", entry_id,
                                        "%s is not in commit %s" % (file_ref, commit)))
        return
    violations.append(Violation("ASSERTED_SOURCE_INVALID", entry_id,
                                "%s does not exist in the worktree and no source.commit is given"
                                % file_ref))


def check_asserted_evidence(repo: Path, entry: dict, violations: list, cache: dict) -> None:
    entry_id = entry.get("id", "<no-id>")
    evidence = entry.get("evidence")
    if not isinstance(evidence, dict):
        violations.append(Violation("ASSERTED_EVIDENCE_INVALID", entry_id,
                                    "evidence is not an object"))
        return
    run_id = evidence.get("run_id")
    trace = evidence.get("trace_path")
    has_run = isinstance(run_id, str) and bool(run_id.strip())
    has_trace = isinstance(trace, str) and bool(trace.strip())
    if has_run == has_trace:
        violations.append(Violation(
            "ASSERTED_EVIDENCE_INVALID", entry_id,
            "exactly one of run_id / trace_path is required (run_id=%r, trace_path=%r)"
            % (run_id, trace)))
        return
    if has_trace:
        if not (repo / trace).exists():
            violations.append(Violation("ASSERTED_EVIDENCE_INVALID", entry_id,
                                        "trace_path %s does not exist" % trace))
        return
    if (repo / "outputs").is_dir():
        for candidate in (repo / "outputs").rglob(run_id):
            if candidate.is_dir():
                return
    commit = evidence.get("commit")
    if isinstance(commit, str) and commit.strip():
        prefix = "/%s/" % run_id
        if any(prefix in ("/" + name + "/") for name in _tree_paths(repo, commit, cache)):
            return
        violations.append(Violation("ASSERTED_EVIDENCE_INVALID", entry_id,
                                    "run_id %s is not in commit %s" % (run_id, commit)))
        return
    violations.append(Violation("ASSERTED_EVIDENCE_INVALID", entry_id,
                                "run_id %s is not present in the worktree and no evidence.commit"
                                " is given" % run_id))


def check_asserted(entry: dict, repo: Path, violations: list, cache: dict) -> None:
    entry_id = entry.get("id", "<no-id>")
    clause = entry.get("spec_clause")
    if not isinstance(clause, str) or not re.fullmatch(r"[A-H][0-9]{1,2}", clause.strip()):
        violations.append(Violation("SPEC_CLAUSE_INVALID", entry_id,
                                    "spec_clause %r is not a batch clause id" % (clause,)))
    unit = entry.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        violations.append(Violation("ASSERTED_UNIT_MISSING", entry_id,
                                    "unit %r is not a non-empty string" % (unit,)))
    check_source(repo, entry, violations, cache)
    check_asserted_evidence(repo, entry, violations, cache)
    threshold = entry.get("threshold")
    if not isinstance(threshold, dict):
        violations.append(Violation("ASSERTED_THRESHOLD_INVALID", entry_id,
                                    "threshold is not an object"))
        return
    value = threshold.get("value")
    if value is None or (isinstance(value, str) and not value.strip()):
        violations.append(Violation("ASSERTED_THRESHOLD_INVALID", entry_id,
                                    "threshold.value missing"))
    if not isinstance(threshold.get("unit"), str) or not threshold["unit"].strip():
        violations.append(Violation("ASSERTED_THRESHOLD_INVALID", entry_id,
                                    "threshold.unit missing"))
    if threshold.get("pre_registered") is not True:
        violations.append(Violation("THRESHOLD_NOT_PREREGISTERED", entry_id,
                                    "pre_registered is %r, not true"
                                    % (threshold.get("pre_registered"),)))
    registered = threshold.get("registered_commit")
    if not isinstance(registered, str) or not re.fullmatch(r"[0-9a-f]{7,40}", registered.strip()):
        violations.append(Violation("THRESHOLD_NOT_PREREGISTERED", entry_id,
                                    "registered_commit %r is not a commit sha" % (registered,)))


def check_unknown(entry: dict, violations: list) -> None:
    entry_id = entry.get("id", "<no-id>")
    for field in ("unit", "threshold", "evidence"):
        if entry.get(field) not in (None, {}, ""):
            violations.append(Violation("UNKNOWN_CARRIES_VALUE", entry_id,
                                        "%s must be null for an UNKNOWN claim, got %r"
                                        % (field, entry.get(field))))
    claim = entry.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        violations.append(Violation("FIELD_MISSING", entry_id, "claim is not a non-empty string"))
        return
    if has_number(claim):
        residual = strip_identifiers(claim)
        violations.append(Violation("UNKNOWN_CARRIES_NUMBER", entry_id,
                                    "claim keeps a number outside identifiers: %r"
                                    % residual.strip()[:120]))


def check_registry(repo: Path, registry_path: Path) -> list:
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot read registry %s: %s" % (registry_path, exc))
    if not isinstance(payload, dict):
        raise RuntimeError("registry is not a JSON object")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("registry carries no entries")
    violations = []
    seen = set()
    cache = {}
    for entry in entries:
        if not isinstance(entry, dict):
            violations.append(Violation("ENTRY_NOT_OBJECT", "<no-id>", repr(entry)[:80]))
            continue
        entry_id = entry.get("id", "<no-id>")
        if not isinstance(entry_id, str) or not entry_id.strip():
            violations.append(Violation("FIELD_MISSING", "<no-id>", "id missing"))
            continue
        if entry_id in seen:
            violations.append(Violation("ID_DUPLICATE", entry_id, "duplicate id"))
        seen.add(entry_id)
        for field in REQUIRED_FIELDS:
            if field not in entry and not (field in ("unit", "threshold", "evidence")
                                           and entry.get("status") in ("UNKNOWN", "NOT_RUN")):
                violations.append(Violation("FIELD_MISSING", entry_id,
                                            "required field %r missing" % field))
        status = entry.get("status")
        if status not in STATUSES:
            violations.append(Violation("STATUS_INVALID", entry_id, "status %r" % (status,)))
            continue
        if status == "ASSERTED":
            check_asserted(entry, repo, violations, cache)
        else:
            check_unknown(entry, violations)
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="R10-C3 assertion provenance registry checker (pure CPU).")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--registry", default=REGISTRY_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[2]
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = repo / registry_path
    try:
        violations = check_registry(repo, registry_path)
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps({"ok": not violations,
                          "violations": [{"code": v.code, "id": v.entry_id, "detail": v.detail}
                                         for v in violations]}, ensure_ascii=False, indent=2))
    else:
        for violation in violations:
            print("VIOLATION %s" % violation)
        if violations:
            print("%d violation(s)" % len(violations))
        else:
            print("assertion provenance satisfied")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
