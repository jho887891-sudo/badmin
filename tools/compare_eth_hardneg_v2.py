#!/usr/bin/env python3
"""V1 vs V2 causal comparison and the frozen decision rule (Task 5 of the V2 plan).

The primary endpoint is the pooled real no-target FP/image with its raw count and 95% CI (ruling 3); the
secondary endpoint is val|eth_unseen accuracy. Every number here comes from canonical evaluator outputs, so the
comparison cannot drift from the SSOT metric code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_eth_only_v1_errors as A  # noqa: E402  (pooled FP/image + exact CI)

USEFUL, HARMFUL, LOW_VALUE, INCONCLUSIVE = "USEFUL", "HARMFUL_TRADEOFF", "LOW_VALUE", "INCONCLUSIVE"
FP_REDUCTION_TARGET = 0.20
RECALL_TOLERANCE = -0.02
RECALL_HARM = -0.05
LOW_VALUE_BAND = 0.10
ACCURACY_STABLE = 0.02
COMPARISON_COLUMNS = ("set", "model", "Precision", "Recall", "F1", "AP50", "mAP50-95", "Recall_<8",
                      "FP_per_image", "delta_vs_v1")


def compute_v2_decision(v1: dict, v2: dict) -> dict:
    """Apply the frozen thresholds. Order: harmful trade-off, useful, low value, otherwise inconclusive."""
    fp1, fp2 = v1.get("fp_per_image"), v2.get("fp_per_image")
    r1, r2 = v1.get("recall", 0.0), v2.get("recall", 0.0)
    m1, m2 = v1.get("map5095", 0.0), v2.get("map5095", 0.0)
    s1, s2 = v1.get("recall_lt8"), v2.get("recall_lt8")
    reduction = ((fp1 - fp2) / fp1) if (fp1 and fp2 is not None) else None
    d_recall = (r2 - r1) if (r1 is not None and r2 is not None) else None
    d_map = (m2 - m1) if (m1 is not None and m2 is not None) else None
    d_lt8 = (s2 - s1) if (s1 is not None and s2 is not None) else None
    decision, why = INCONCLUSIVE, "outside every frozen band"
    if d_recall is not None and d_recall <= RECALL_HARM:
        decision, why = HARMFUL, "recall drop %.4f <= %.2f even though FP changed by %s" % (
            d_recall, RECALL_HARM, ("%.1f%%" % (100 * reduction)) if reduction is not None else "n/a")
    elif reduction is not None and reduction >= FP_REDUCTION_TARGET and (d_recall or 0) > RECALL_TOLERANCE:
        decision, why = USEFUL, "FP/image down %.1f%% (>= 20%%) with recall change %+.4f > -0.02" % (
            100 * reduction, d_recall if d_recall is not None else 0.0)
    elif (reduction is not None and abs(reduction) < LOW_VALUE_BAND
          and abs(d_recall or 0.0) < ACCURACY_STABLE and abs(d_map or 0.0) < ACCURACY_STABLE):
        decision, why = LOW_VALUE, "FP/image changed %.1f%% (< 10%%) with accuracy effectively unchanged" % (
            100 * reduction)
    return {
        "decision": decision,
        "why": why,
        "fp_per_image_v1": fp1, "fp_per_image_v2": fp2,
        "fp_reduction_relative": reduction,
        "recall_v1": r1, "recall_v2": r2, "delta_recall": d_recall,
        "map5095_v1": m1, "map5095_v2": m2, "delta_map5095": d_map,
        "recall_lt8_v1": s1, "recall_lt8_v2": s2, "delta_recall_lt8": d_lt8,
        "thresholds": {"fp_reduction_target": FP_REDUCTION_TARGET, "recall_tolerance": RECALL_TOLERANCE,
                       "recall_harm": RECALL_HARM, "low_value_band": LOW_VALUE_BAND,
                       "accuracy_stable": ACCURACY_STABLE},
    }


def pooled_fp(negatives_csv, threshold=0.25) -> dict:
    """Pooled real no-target FP/image for one checkpoint, from the canonical frozen-set negatives table."""
    rows = list(csv.DictReader(open(negatives_csv, encoding="utf-8")))
    return A.summarize_false_positives(rows, threshold)


def _index(rows, key_cols=("model", "set")) -> dict:
    return {tuple(str(r[c]) for c in key_cols): r for r in rows}


def build_comparison(v1_rows, v2_rows, v1_pooled, v2_pooled, v1_model="eth_only_v1_best",
                     v2_model="eth_real_hardneg_v2_best") -> list:
    """One row per (set, model) with the V1-referenced delta, plus the pooled real no-target row."""
    out = []
    v1i, v2i = _index(v1_rows), _index(v2_rows)
    sets = []
    for key in v2i:
        if key[0] == v2_model and key[1] not in sets:
            sets.append(key[1])
    for key in v1i:
        if key[0] == v1_model and key[1] not in sets:
            sets.append(key[1])
    for set_name in sets:
        a = v1i.get((v1_model, set_name), {})
        b = v2i.get((v2_model, set_name), {})
        for model, r in ((v1_model, a), (v2_model, b)):
            if not r:
                continue
            row = {"set": set_name, "model": model,
                   "Precision": r.get("Precision"), "Recall": r.get("Recall"), "F1": r.get("F1"),
                   "AP50": r.get("AP50"), "mAP50-95": r.get("mAP50-95"),
                   "Recall_<8": r.get("Recall_<8"), "FP_per_image": r.get("FP_per_image"),
                   "delta_vs_v1": ""}
            if model == v2_model and a:
                for col, key in (("Recall", "delta_recall"), ("mAP50-95", "delta_map5095"),
                                 ("Recall_<8", "delta_recall_lt8"), ("FP_per_image", "delta_fp_per_image")):
                    if col == "FP_per_image":
                        try:
                            d = float(b["FP_per_image"]) - float(a["FP_per_image"])
                        except (KeyError, TypeError, ValueError):
                            continue
                    else:
                        try:
                            d = float(b[col]) - float(a[col])
                        except (KeyError, TypeError, ValueError):
                            continue
                    row["delta_vs_v1"] = ((row["delta_vs_v1"] + "; ") if row["delta_vs_v1"] else "") + \
                        "%s %+.5f" % (key, d)
            out.append(row)
    # pooled real no-target endpoint (FP/image only; accuracy columns stay empty by construction)
    for model, pool in ((v1_model, v1_pooled), (v2_model, v2_pooled)):
        if not pool or not pool.get("images"):
            continue
        row = {c: "" for c in COMPARISON_COLUMNS}
        row.update({"set": "real_no_target_pooled(%d images)" % pool["images"], "model": model,
                    "FP_per_image": pool.get("fp_per_image")})
        if model == v2_model and v1_pooled and v1_pooled.get("fp_per_image"):
            rel = (v1_pooled["fp_per_image"] - pool["fp_per_image"]) / v1_pooled["fp_per_image"]
            row["delta_vs_v1"] = "delta_fp_per_image_relative %+.5f" % (-rel)
        out.append(row)
    return out


def write_comparison(rows, path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COMPARISON_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COMPARISON_COLUMNS})
    return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare ETH hard-negative V2 against V1 and decide")
    ap.add_argument("--v1-table", default="outputs/shuttle_capability/metrics/eth_vs_ours_eth_only_v1.csv")
    ap.add_argument("--v2-table", default="outputs/shuttle_capability/metrics/eth_vs_ours_eth_hardneg_v2.csv")
    ap.add_argument("--v1-negatives", default="outputs/shuttle_capability/metrics/eth_only_v1_frozen_test_negatives.csv")
    ap.add_argument("--v2-negatives", default="outputs/shuttle_capability/metrics/eth_hardneg_v2_frozen_test_negatives.csv")
    ap.add_argument("--v1-model", default="eth_only_v1_best")
    ap.add_argument("--v2-model", default="eth_real_hardneg_v2_best")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out-csv", default="outputs/shuttle_capability/metrics/eth_real_hardneg_v2_vs_v1.csv")
    ap.add_argument("--out-json", default="outputs/shuttle_capability/metrics/eth_real_hardneg_v2_decision.json")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    v1_pooled = pooled_fp(repo / a.v1_negatives, a.threshold) if (repo / a.v1_negatives).is_file() else {}
    v2_pooled = pooled_fp(repo / a.v2_negatives, a.threshold) if (repo / a.v2_negatives).is_file() else {}
    v1_rows = list(csv.DictReader(open(repo / a.v1_table, encoding="utf-8"))) if (repo / a.v1_table).is_file() else []
    v2_rows = list(csv.DictReader(open(repo / a.v2_table, encoding="utf-8"))) if (repo / a.v2_table).is_file() else []
    rows = build_comparison(v1_rows, v2_rows, v1_pooled, v2_pooled, a.v1_model, a.v2_model)
    write_comparison(rows, repo / a.out_csv)

    def pick(rows_, set_name, model, col):
        for r in rows_:
            if r["set"] == set_name and r["model"] == model and r.get(col) not in (None, ""):
                return float(r[col])
        return None
    unseen = "val|eth_unseen"
    decision = compute_v2_decision(
        {"fp_per_image": v1_pooled.get("fp_per_image"), "recall": pick(v1_rows, unseen, a.v1_model, "Recall"),
         "map5095": pick(v1_rows, unseen, a.v1_model, "mAP50-95"),
         "recall_lt8": pick(v1_rows, unseen, a.v1_model, "Recall_<8")},
        {"fp_per_image": v2_pooled.get("fp_per_image"), "recall": pick(v2_rows, unseen, a.v2_model, "Recall"),
         "map5095": pick(v2_rows, unseen, a.v2_model, "mAP50-95"),
         "recall_lt8": pick(v2_rows, unseen, a.v2_model, "Recall_<8")})
    decision["primary_endpoint"] = "pooled_real_no_target_fp_per_image"
    decision["secondary_endpoint"] = unseen
    decision["v1_pooled"] = v1_pooled
    decision["v2_pooled"] = v2_pooled
    Path(repo / a.out_json).parent.mkdir(parents=True, exist_ok=True)
    (repo / a.out_json).write_text(json.dumps(decision, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[cmp] %s: %s" % (decision["decision"], decision["why"]))
    print("[cmp] wrote %s and %s" % (a.out_csv, a.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
