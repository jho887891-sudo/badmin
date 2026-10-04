#!/usr/bin/env python3
"""Build the Stage A vs Stage B(e10) hard-negative comparison table from the per-set eval JSONs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# set -> (in_stage_b_training, provenance)
SET_META = {
    "hard_negative": (True, "hard_negatives/raw + hard_negatives2/raw (96 imgs, split=train)"),
    "normal_neg_coco_val": (False, "coco_bg split=val (1000 imgs, unseen)"),
    "normal_neg_bg_val": (False, "bg_negative split=val (6 imgs, unseen)"),
    "frozen_real_backgrounds": (False, "real_images/backgrounds (30 imgs, frozen, never trained)"),
}
CKPTS = ["stageA_best", "stageB_best_e10"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--metrics-dir", default="outputs/shuttle_capability/metrics")
    ap.add_argument("--out", default="outputs/shuttle_capability/metrics/hard_negative_eval_compare.csv")
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    mdir = repo / args.metrics_dir
    cols = ["set", "in_stage_b_training", "provenance", "images", "threshold"]
    for c in CKPTS:
        cols += ["%s_total_FP" % c, "%s_FP_per_image" % c, "%s_image_FP_rate" % c, "%s_max_conf" % c]
    cols += ["delta_total_FP_B_minus_A", "delta_FP_per_image_B_minus_A", "delta_image_FP_rate_B_minus_A", "delta_max_conf_B_minus_A"]
    rows = []
    for path in sorted(mdir.glob("hard_negative_eval_*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        name = obj["set"]
        in_train, prov = SET_META.get(name, (None, "UNKNOWN"))
        for thr in [str(t) for t in obj["thresholds"]]:
            row = {"set": name, "in_stage_b_training": in_train, "provenance": prov, "images": obj["images"], "threshold": thr}
            got = {}
            for c in CKPTS:
                r = obj["results"].get(c, {}).get(thr)
                if r is None:
                    continue
                got[c] = r
                row["%s_total_FP" % c] = r["total_FP"]
                row["%s_FP_per_image" % c] = round(r["FP_per_image"], 6)
                row["%s_image_FP_rate" % c] = round(r["image_FP_rate"], 6)
                row["%s_max_conf" % c] = None if r["max_confidence_FP"] is None else round(r["max_confidence_FP"], 6)
            a, b = got.get("stageA_best"), got.get("stageB_best_e10")
            if a and b:
                row["delta_total_FP_B_minus_A"] = b["total_FP"] - a["total_FP"]
                row["delta_FP_per_image_B_minus_A"] = round(b["FP_per_image"] - a["FP_per_image"], 6)
                row["delta_image_FP_rate_B_minus_A"] = round(b["image_FP_rate"] - a["image_FP_rate"], 6)
                mc = (None if b["max_confidence_FP"] is None else b["max_confidence_FP"])
                ma = (None if a["max_confidence_FP"] is None else a["max_confidence_FP"])
                row["delta_max_conf_B_minus_A"] = None if (mc is None or ma is None) else round(mc - ma, 6)
            rows.append(row)
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out, "rows", len(rows))
    for r in rows:
        print(r["set"], r["threshold"], "A_FP/img=%s" % r.get("stageA_best_FP_per_image"), "B_FP/img=%s" % r.get("stageB_best_e10_FP_per_image"), "delta=%s" % r.get("delta_FP_per_image_B_minus_A"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())