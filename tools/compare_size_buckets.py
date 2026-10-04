#!/usr/bin/env python3
"""N-way size-bucket comparison from an existing size_bucket_metrics_<split>.json (no inference).

The evaluator itself only writes a first-vs-second checkpoint table; this tool generalises it to every
checkpoint present in the metrics JSON, with deltas against the first checkpoint (the reference).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BUCKET_ORDER = ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--metrics", default="outputs/shuttle_capability/metrics/size_bucket_metrics_val.json")
    ap.add_argument("--out", default="outputs/shuttle_capability/metrics/size_bucket_three_way_compare_val.csv")
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    obj = json.loads((repo / args.metrics).read_text(encoding="utf-8"))
    names = list(obj.keys())
    ref = names[0]
    cols = ["bucket", "GT"]
    for n in names:
        cols += ["%s_TP" % n, "%s_FN" % n, "%s_Recall" % n, "%s_AP50" % n, "%s_AP50-95" % n]
    for n in names[1:]:
        cols += ["delta_recall_%s_minus_%s" % (n, ref), "delta_AP50_%s_minus_%s" % (n, ref), "delta_AP50-95_%s_minus_%s" % (n, ref)]
    rows = []
    for b in BUCKET_ORDER:
        gts = [obj[n]["buckets"].get(b) for n in names]
        if any(g is None for g in gts):
            continue
        row = {"bucket": b, "GT": gts[0]["gt"]}
        for n, g in zip(names, gts):
            row["%s_TP" % n] = g["tp"]
            row["%s_FN" % n] = g["fn"]
            row["%s_Recall" % n] = round(g["recall"], 6)
            row["%s_AP50" % n] = round(g["ap50"], 6)
            row["%s_AP50-95" % n] = round(g["ap5095"], 6)
        for n in names[1:]:
            g, r = obj[n]["buckets"][b], obj[ref]["buckets"][b]
            row["delta_recall_%s_minus_%s" % (n, ref)] = round(g["recall"] - r["recall"], 6)
            row["delta_AP50_%s_minus_%s" % (n, ref)] = round(g["ap50"] - r["ap50"], 6)
            row["delta_AP50-95_%s_minus_%s" % (n, ref)] = round(g["ap5095"] - r["ap5095"], 6)
        rows.append(row)
    # summary rows: OVERALL / COMBINED_6-16 / COMBINED_4-16 / SECONDARY_CENTER
    def summary(label, getter):
        vals = [getter(obj[n]) for n in names]
        row = {"bucket": label, "GT": vals[0]["gt"]}
        for n, v in zip(names, vals):
            row["%s_TP" % n] = v.get("tp")
            row["%s_FN" % n] = v.get("fn")
            row["%s_Recall" % n] = round(v["recall"], 6)
            row["%s_AP50" % n] = None if v.get("mAP50") is None else round(v["mAP50"], 6)
            row["%s_AP50-95" % n] = None if v.get("mAP50-95") is None else round(v["mAP50-95"], 6)
        for n in names[1:]:
            row["delta_recall_%s_minus_%s" % (n, ref)] = round(obj[n] and vals[names.index(n)]["recall"] - vals[0]["recall"], 6)
        rows.append(row)
    summary("OVERALL", lambda v: dict(v["overall"]))
    summary("COMBINED_6-16", lambda v: dict(v["combined_recall_6_16"]))
    summary("COMBINED_4-16", lambda v: dict(v["combined_recall_4_16"]))
    summary("SECONDARY_CENTER_25px", lambda v: dict(v["secondary_center_distance"]["overall"]))
    out = repo / args.out
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out, "checkpoints:", names, "rows:", len(rows))
    for r in rows:
        print(r["bucket"], "GT=" + str(r["GT"]), " | ".join("%s R=%.4f AP50=%.4f AP50-95=%.4f" % (n, r["%s_Recall" % n], r["%s_AP50" % n] or 0, r["%s_AP50-95" % n] or 0) for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())