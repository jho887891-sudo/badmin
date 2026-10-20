"""Read-only regression of the 224 archived P4 matches under corrected containment.

This intentionally does NOT claim that the unmatched 58 become matched.
Those need full STEP/real USD re-extraction on the remote Isaac host.
"""
import argparse
import json
from pathlib import Path

from scripts.simulation.map_step_solids_to_usd_meshes import sandwich


ARCHIVE = Path("outputs/simulation/rebot_b601_rs/morph_one/dev_2270cb3/p4/step_to_usd_mapping.json")


def audit(path=ARCHIVE):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    passed, failed = [], []
    for row in source["pairs"]:
        a, b, c = (row["step_vertex_bbox_mm"], row["step_closure_bbox_mm"], row["usd_bbox_mm"])
        outcome = sandwich(a["min"], a["max"], b["min"], b["max"], c["min"], c["max"])
        (passed if outcome["ok"] else failed).append(row["step_eid"])
    return {
        "source": str(path),
        "reported_pairs": len(source["pairs"]),
        "corrected_sandwich_passed": len(passed),
        "corrected_sandwich_failed": len(failed),
        "failed_step_eids": failed,
        "original_unmatched_solids": len(source["unmatched_solids"]),
        "repaired_unmatched_count": None,
        "caveat": "Archived matched-pair bboxes only; cannot recompute whole matching or rejected candidates.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=str(ARCHIVE))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    report = audit(args.mapping)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    if report["corrected_sandwich_failed"]:
        raise SystemExit("Archived pair(s) fail corrected geometry check")


if __name__ == "__main__":
    main()
