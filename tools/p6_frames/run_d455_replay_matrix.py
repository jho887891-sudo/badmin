# -*- coding: utf-8 -*-
"""Run the P6-03 replay validator over every fixture and record REAL exit codes.

The expectation table below is the contract for this slice. The tool exits 0 only
if every fixture produces exactly the expected verdict and a superset of the
expected violation codes, so the committed evidence file cannot drift from the
code without this tool failing.

Everything here is OFFLINE-SYNTHETIC: no D455 was attached, no GPU, no Isaac.

Usage:
    python tools/p6_frames/run_d455_replay_matrix.py            # exit 0 expected
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulation.sensors import d455_replay as rp  # noqa: E402

FIXTURES = REPO_ROOT / "tests/simulation/fixtures/p6_d455_replay"
OUT_DIR = (
    REPO_ROOT
    / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/evidence"
)

# fixture -> (expected exit code, required violation codes)
EXPECTED: dict[str, tuple[int, tuple[str, ...]]] = {
    "good": (0, ()),
    "external_witness": (0, ()),
    "missing_frames": (
        1,
        ("MISSING_FRAME_FILE", "FRAME_COUNT_MISMATCH", "DIRECTORY_COUNT_MISMATCH"),
    ),
    "non_monotonic": (1, ("NON_MONOTONIC_TIMESTAMP", "TIMESTAMP_GAP_EXCEEDED")),
    "gap": (1, ("TIMESTAMP_GAP_EXCEEDED",)),
    "mm_depth": (1, ("DEPTH_UNIT_NOT_METRES",)),
    "mm_value_in_metres_field": (1, ("METADATA_INVALID",)),
    "unknown_intrinsics": (1, ("INTRINSICS_UNKNOWN",)),
    "bad_sha": (1, ("METADATA_INVALID",)),
    "deg_fov": (1, ("METADATA_INVALID",)),
    "no_fps": (1, ("FPS_UNKNOWN", "INTRINSICS_UNKNOWN")),
    "alignment_undeclared": (1, ("ALIGNMENT_UNKNOWN", "INTRINSICS_UNKNOWN")),
    "no_metadata_record": (1, ("INTRINSICS_UNKNOWN",)),
}


def run(out_dir: Path = OUT_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for name, (want_exit, want_codes) in EXPECTED.items():
        fixture = FIXTURES / (name if name != "no_metadata_record" else "good")
        meta = FIXTURES / "good/metadata.json" if name == "no_metadata_record" else (
            fixture / "metadata.json"
        )
        report = rp.validate_replay(
            fixture,
            metadata_path=meta if meta.is_file() and name != "no_metadata_record"
            else None,
        )
        payload = report.to_dict()
        report_path = out_dir / f"d455_replay_{name}.json"
        report.write(report_path)
        got_codes = [v.code for v in report.violations]
        ok_exit = report.exit_code == want_exit
        ok_codes = set(want_codes) <= set(got_codes)
        rows.append(
            {
                "fixture": name,
                "data_class": report.data_class,
                "report": str(report_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "expected_exit_code": want_exit,
                "actual_exit_code": report.exit_code,
                "expected_codes": list(want_codes),
                "actual_codes": got_codes,
                "verdict": report.verdict,
                "matches_expectation": bool(ok_exit and ok_codes),
            }
        )
        if not (ok_exit and ok_codes):
            failures.append(name)

    summary = {
        "schema": "p6_d455_replay_matrix/1",
        "data_class": "OFFLINE-SYNTHETIC",
        "calibration_claimed": False,
        "no_device_attached": True,
        "generated_by": "tools/p6_frames/run_d455_replay_matrix.py",
        "thresholds": {
            "max_gap_factor": rp.DEFAULT_MAX_GAP_FACTOR,
            "max_gap_factor_limit": rp.MAX_GAP_FACTOR_LIMIT,
            "gap_rule": "dt > max_gap_factor / fps",
            "monotonic_rule": "strictly increasing",
        },
        "fixtures": rows,
        "summary": {
            "total": len(rows),
            "passing_replays": sum(1 for r in rows if r["actual_exit_code"] == 0),
            "failing_replays": sum(1 for r in rows if r["actual_exit_code"] != 0),
            "mismatched_expectations": failures,
            "all_match": not failures,
        },
    }
    target = out_dir / "d455_replay_matrix.json"
    target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {target}")
    for row in rows:
        print(
            f"{row['fixture']:26s} exit={row['actual_exit_code']} "
            f"expected={row['expected_exit_code']} codes={','.join(row['actual_codes']) or '-'} "
            f"{'OK' if row['matches_expectation'] else 'MISMATCH'}"
        )
    if failures:
        print(f"FAIL: {len(failures)} fixture(s) diverged: {failures}")
        return 1
    print(f"PASS: {len(rows)} fixtures matched their expectations")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    return run(Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
