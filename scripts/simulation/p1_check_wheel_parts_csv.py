"""Cross-check: does this round reproduce the historical wheel_parts.csv rows?

wheel_parts.csv was produced from the same STEP file on other branches (commit
53d71ff, branch feat/morph-one-wheel-kinematics).  Its four wheel-like rows are
the origin of the "48.653 x 180.782 x 157.783 mm" number that this round has to
explain, so before explaining it we must show it can be regenerated exactly.

The historical file is read with "git show <rev>:<path>" instead of a checkout,
so the check works from any branch of this repository.

Usage:

    python scripts/simulation/p1_check_wheel_parts_csv.py
        --rev 53d71ff
        --csv outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.csv
        --json outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.json
        --repo-root .

Exit code 0 means every compared column matched.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys

HISTORICAL_PATH = (
    "outputs/simulation/rebot_b601_rs/morph_one/wheels_b0853ef/wheel_parts.csv"
)
DEFAULT_EIDS = (435700, 759491, 773650, 871375)


def historical_rows(repo_root, rev):
    completed = subprocess.run(
        ["git", "show", "%s:%s" % (rev, HISTORICAL_PATH)],
        capture_output=True,
        cwd=repo_root,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "git show failed with exit code %d: %s"
            % (completed.returncode, completed.stderr.decode("utf-8", "replace").strip())
        )
    reader = csv.DictReader(io.StringIO(completed.stdout.decode("utf-8")))
    rows = {}
    for row in reader:
        token = (row.get("step_eid") or "").strip()
        if not token:
            continue
        rows[int(token)] = row
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="reproduce wheel_parts.csv from this round")
    parser.add_argument("--rev", default="53d71ff", help="commit that contains wheel_parts.csv")
    parser.add_argument("--csv", required=True, help="step_remeasure.csv from this round")
    parser.add_argument("--json", required=True, help="step_remeasure.json from this round")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--eids", default=",".join(str(e) for e in DEFAULT_EIDS))
    args = parser.parse_args(argv)

    eids = [int(token) for token in args.eids.split(",") if token.strip()]
    old = historical_rows(args.repo_root, args.rev)
    with open(args.csv, "r", encoding="utf-8") as handle:
        new = {int(row["step_eid"]): row for row in csv.DictReader(handle)}
    with open(args.json, "r", encoding="utf-8") as handle:
        measured = {int(solid["step_eid"]): solid for solid in json.load(handle)["solids"]}

    print("historical source: %s:%s" % (args.rev, HISTORICAL_PATH))
    ok = True
    for eid in eids:
        if eid not in old:
            print("%d  MISSING in the historical csv" % eid)
            ok = False
            continue
        row_old, row_new, detail = old[eid], new[eid], measured[eid]
        old_size = [
            round(float(row_old["size_x_mm"]), 3),
            round(float(row_old["size_y_mm"]), 3),
            round(float(row_old["size_z_mm"]), 3),
        ]
        new_size = [
            round(float(row_new["vertex_bbox_x_mm"]), 3),
            round(float(row_new["vertex_bbox_y_mm"]), 3),
            round(float(row_new["vertex_bbox_z_mm"]), 3),
        ]
        circles = int(row_new["circle_count"])
        cylinders = int(row_new["cylindrical_surface_count"])
        features = int(row_old["circular_features"])
        same_box = old_size == new_size
        same_features = features == circles + cylinders
        same_radius = round(float(row_old["max_circle_radius_mm"]), 3) == round(
            float(row_new["max_circle_radius_mm"]), 3
        )
        ok = ok and same_box and same_features and same_radius
        print("%d  size old=%s new=%s equal=%s" % (eid, old_size, new_size, same_box))
        print(
            "      circular_features old=%d  CIRCLE=%d + CYLINDRICAL_SURFACE=%d = %d  equal=%s"
            % (features, circles, cylinders, circles + cylinders, same_features)
        )
        print(
            "      max_circle_radius old=%s new=%s equal=%s"
            % (row_old["max_circle_radius_mm"], row_new["max_circle_radius_mm"], same_radius)
        )
        print(
            "      this round adds: vertex_points=%d cartesian_points=%d envelope_bbox=%s"
            % (
                detail["vertex_points"],
                detail["cartesian_points_in_closure"],
                [
                    detail["surface_envelope_bbox"]["size"][0],
                    detail["surface_envelope_bbox"]["size"][1],
                    detail["surface_envelope_bbox"]["size"][2],
                ],
            )
        )
    print("")
    print("CROSSCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
