"""Read-only audit of R11 M1 position sigma from committed tracks.jsonl.

The old 4x4 state covariance [x,y,vx,vy] mixes px and px/s units. Taking the
largest eigenvalue of all 4 dimensions is not a valid position standard
deviation. This tool recomputes xy sigma from the top-left 2x2 block only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def xy_sigma_from_state_covariance(matrix):
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("expected 4x4 state covariance")
    a = float(matrix[0][0])
    b = (float(matrix[0][1]) + float(matrix[1][0])) / 2.0
    c = float(matrix[1][1])
    if not all(math.isfinite(v) for v in (a, b, c)):
        raise ValueError("nonfinite xy covariance")
    lam = (a + c + math.sqrt((a - c) * (a - c) + 4 * b * b)) / 2.0
    if min(a, c, lam) < -1e-8 or a * c - b * b < -1e-7:
        raise ValueError("non PSD xy covariance")
    return math.sqrt(max(lam, 0.0))


def audit(path):
    by_horizon, ts = {}, []
    count = 0
    with Path(path).open(encoding="utf8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            count += 1
            ts.append(float(record["timestamp_s"]))
            for item in record.get("horizon", []):
                horizon = str(item["horizon_s"])
                full_matrix = item.get("state_covariance_mixed_units",
                                       item.get("covariance_px2"))
                sigma = xy_sigma_from_state_covariance(full_matrix)
                old = item.get("sigma_major_px")
                row = {"position_sigma_px": sigma}
                if old is not None:
                    row["reported_sigma_px"] = float(old)
                by_horizon.setdefault(horizon, []).append(row)
    if not count:
        raise ValueError("no input frames")
    horizons = {}
    for horizon, entries in by_horizon.items():
        position_values = [row["position_sigma_px"] for row in entries]
        reported_values = [row["reported_sigma_px"] for row in entries
                           if "reported_sigma_px" in row]
        horizons[horizon] = {
            "n": len(position_values),
            "position_sigma_median_px": statistics.median(position_values),
            "old_reported_sigma_median_px":
                statistics.median(reported_values) if reported_values else None,
        }
    gaps = [next_ts - prev_ts for prev_ts, next_ts in zip(ts, ts[1:])]
    return {"source_path": str(path), "frame_count": count,
            "first_timestamps_s": ts[:4],
            "max_frame_gap_s": max(gaps) if gaps else None,
            "horizons": horizons,
            "claim": "OFFLINE_AUDIT_OF_COMMITTED_TRACKS_NOT_NEW_MODEL_RUN"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.tracks)
    data = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite old audit output")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data, encoding="utf8")
    else:
        print(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
