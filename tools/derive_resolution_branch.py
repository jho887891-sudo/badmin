#!/usr/bin/env python3
"""Apply the pre-registered branch rule to a resolution-response probe run.

The rule itself lives in tools/analyze_resolution_response.py (summarise + compute_branch) and was frozen before any
number was seen; this driver only reshapes a probe JSON into the row format that module expects, so the branch is
chosen by the frozen code rather than by reading the tables.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import analyze_resolution_response as arr  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_from_probe(probe: dict) -> list:
    """One row per (resolution, GT) in the shape summarise() consumes."""
    out = []
    for R, block in probe["resolutions"].items():
        for g in block["gt"]:
            pairs = [e["pairs"] for e in block["per_image_iou_conf"][g["image"]]
                     if e["gt_index"] == g["gt_index"]][0]
            out.append({"imgsz": int(R), "bucket": g["bucket"],
                        "matched_op": bool(g["best_iou_weak"] >= 0.5 and g["best_conf_at_best_iou"] >= 0.25),
                        "best_iou_weak": float(g["best_iou_weak"]),
                        "net_px": 1.6 * float(g["eq640"]) * int(R) / 1024.0,
                        "n_candidates": len(pairs)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args([] if argv is None else argv)
    probe_path = Path(a.json)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    rows = rows_from_probe(probe)
    res = sorted({r["imgsz"] for r in rows})
    summaries = [arr.summarise(rows, R) for R in res]
    decision = arr.compute_branch(summaries)
    payload = {"tag": a.tag, "probe_json": str(probe_path), "probe_sha256": sha256(probe_path),
               "weights_sha256": probe["weights_sha256"], "env": probe["env"],
               "resolutions": res, "n_gt": len(rows) // len(res),
               "rule_source": "tools/analyze_resolution_response.py:compute_branch (frozen before the run)",
               "summaries": summaries, "decision": decision}
    out = Path(a.out) if a.out else REPO / "outputs/shuttle_capability/metrics/resolution_response_v1_decision.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("tag=%s  n_gt=%d  probe=%s" % (a.tag, payload["n_gt"], probe_path))
    for s in summaries:
        b = s["by_bucket"]
        print("  R=%4d  <4 weak>=.1: %2d/%d | 4-6: %2d/%d | 6-8: %2d/%d | strict_sub_stride(=%d GT) weak>=.1: %s"
              % (s["imgsz"], b.get("<4", {}).get("weak_ge_0.1", 0), b.get("<4", {}).get("gt", 0),
                 b.get("4-6", {}).get("weak_ge_0.1", 0), b.get("4-6", {}).get("gt", 0),
                 b.get("6-8", {}).get("weak_ge_0.1", 0), b.get("6-8", {}).get("gt", 0),
                 s["strict_sub_stride"]["gt"], s["strict_sub_stride"]["weak_ge_0.1"]))
    print("DECISION: %s" % decision["branch"])
    print("  %s" % decision["why"])
    print("  sub_stride_weak_ge_0.1=%s  mid_bucket_weak_ge_0.1=%s"
          % (decision["sub_stride_weak_ge_0.1"], decision["mid_bucket_weak_ge_0.1"]))
    print("WROTE %s (%d B)" % (out, out.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
