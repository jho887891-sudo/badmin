#!/usr/bin/env python3
"""Tiny-object recovery V1: V1 / V2 / Tiny comparison and the pre-registered decision matrix (design section 19).

Everything is derived from canonical evaluator outputs; the no-target endpoint reuses the exact Poisson/Wilson
intervals from tools/analyze_eth_only_v1_errors.py, and every change is reported as raw counts plus absolute and
relative deltas (design section 23 forbids percentages alone).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_eth_only_v1_errors as A  # noqa: E402  (pooled FP/image + exact intervals)

USEFUL, STRONG, NO_GAIN, FP_REGRESSION, SIZE_TRADEOFF, HARMFUL = (
    "USEFUL", "STRONG_SUCCESS", "NO_TINY_GAIN", "FP_REGRESSION", "SIZE_TRADEOFF", "HARMFUL")
MAIN_COLUMNS = ("metric", "v1", "v2_hardneg", "tiny_recovery", "delta_vs_v2", "delta_kind")


def compute_tiny_decision(v2: dict, tiny: dict) -> dict:
    """Apply the frozen gates in a fixed precedence so a single run cannot select its own label."""
    r2, r3 = v2.get("recall"), tiny.get("recall")
    m2, m3 = v2.get("map5095"), tiny.get("map5095")
    tp2, tp3 = v2.get("tp_lt8"), tiny.get("tp_lt8")
    fp3 = tiny.get("fp_no_target")
    e2, e3 = v2.get("recall_8_16"), tiny.get("recall_8_16")
    d_recall = (r3 - r2) if (r2 is not None and r3 is not None) else None
    d_map = (m3 - m2) if (m2 is not None and m3 is not None) else None
    d_8_16 = (e3 - e2) if (e2 is not None and e3 is not None) else None

    decision, why = NO_GAIN, "no rule matched"
    if (d_recall is not None and d_recall <= -0.02) or (d_map is not None and d_map <= -0.01):
        decision, why = HARMFUL, "recall change %s or mAP change %s breached the harm guard" % (d_recall, d_map)
    elif fp3 is not None and int(fp3) >= 4:
        decision, why = FP_REGRESSION, "%d false positives on the frozen real no-target pool (>= 4)" % int(fp3)
    elif d_8_16 is not None and d_8_16 < -0.02:
        decision, why = SIZE_TRADEOFF, "8-16 px recall fell by %.4f (>= 0.02) even though tiny may have improved" % d_8_16
    elif (tp3 is not None and int(tp3) >= 8 and (d_recall is None or d_recall > -0.02)
          and (fp3 is not None and int(fp3) <= 3) and (d_map is None or d_map > -0.01)):
        if int(tp3) >= 12:
            decision, why = STRONG, "TP_<8 reached %d (>= 12) with every guard satisfied" % int(tp3)
        else:
            decision, why = USEFUL, ("TP_<8 recovered to %d (>= 8, V1 level) with recall guard %s, FP %s and mAP "
                                     "guard %s satisfied" % (int(tp3), d_recall, fp3, d_map))
    else:
        decision, why = NO_GAIN, ("TP_<8 is %s (target >= 8) and no other rule triggered; tiny exposure did not "
                                  "buy tiny recall" % tp3)
    return {
        "decision": decision,
        "why": why,
        "tp_lt8_v2": tp2, "tp_lt8_tiny": tp3, "delta_tp_lt8": (None if (tp2 is None or tp3 is None) else int(tp3) - int(tp2)),
        "recall_v2": r2, "recall_tiny": r3, "delta_recall": d_recall,
        "map5095_v2": m2, "map5095_tiny": m3, "delta_map5095": d_map,
        "fp_no_target_v2": v2.get("fp_no_target"), "fp_no_target_tiny": fp3,
        "recall_lt8_v2": v2.get("recall_lt8"), "recall_lt8_tiny": tiny.get("recall_lt8"),
        "recall_8_16_v2": e2, "recall_8_16_tiny": e3, "delta_recall_8_16": d_8_16,
        "thresholds": {"tp_lt8_min": 8, "tp_lt8_strong": 12, "recall_min": 0.1840, "fp_no_target_max": 3,
                       "map5095_delta_min": -0.01, "recall_8_16_delta_min": -0.02},
        "precedence": [HARMFUL, FP_REGRESSION, SIZE_TRADEOFF, STRONG, USEFUL, NO_GAIN],
    }


def build_main_table(v1, v2, tiny) -> list:
    """Metric / v1 / v2 / tiny / delta rows for the design section 22 main table."""
    f = lambda d, k: (d or {}).get(k)
    rows = [
        ("Precision", f(v1, "precision"), f(v2, "precision"), f(tiny, "precision"), "absolute"),
        ("Recall", f(v1, "recall"), f(v2, "recall"), f(tiny, "recall"), "absolute"),
        ("mAP50-95", f(v1, "map5095"), f(v2, "map5095"), f(tiny, "map5095"), "absolute"),
        ("TP<8", f(v1, "tp_lt8"), f(v2, "tp_lt8"), f(tiny, "tp_lt8"), "absolute"),
        ("Recall<8", f(v1, "recall_lt8"), f(v2, "recall_lt8"), f(tiny, "recall_lt8"), "absolute"),
        ("Recall 8-16", f(v1, "recall_8_16"), f(v2, "recall_8_16"), f(tiny, "recall_8_16"), "absolute"),
        ("no-target FP", f(v1, "fp_no_target"), f(v2, "fp_no_target"), f(tiny, "fp_no_target"), "absolute"),
        ("no-target FP/image", f(v1, "fp_per_image"), f(v2, "fp_per_image"), f(tiny, "fp_per_image"), "absolute"),
        ("no-target FP/image relative", None, None, None, "relative"),
    ]
    out = []
    for metric, a, b, c, kind in rows:
        delta = None
        if kind == "absolute" and b is not None and c is not None:
            delta = float(c) - float(b)
        out.append({"metric": metric, "v1": a, "v2_hardneg": b, "tiny_recovery": c,
                    "delta_vs_v2": delta, "delta_kind": kind})
    return out


def metrics_from_row(row: dict) -> dict:
    """Pull the decision inputs out of one canonical comparison CSV row."""
    def num(key):
        v = (row or {}).get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return {"precision": num("Precision"), "recall": num("Recall"), "map5095": num("mAP50-95"),
            "tp_lt8": num("TP_<8"), "recall_lt8": num("Recall_<8"), "recall_8_16": num("Recall_8_16")}


def _rows(path):
    p = Path(path)
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.is_file() else []


def _one(rows, model, set_name="val|eth_unseen"):
    for r in rows:
        if r.get("model") == model and r.get("set") == set_name:
            return r
    return {}


def write_csv(path, rows, columns=None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or list(rows[0].keys())
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare tiny-object recovery against V2 and V1, then decide")
    ap.add_argument("--v1-table", default="outputs/shuttle_capability/metrics/eth_vs_ours_eth_only_v1.csv")
    ap.add_argument("--v2-table", default="outputs/shuttle_capability/metrics/eth_vs_ours_eth_hardneg_v2.csv")
    ap.add_argument("--tiny-table", default="outputs/shuttle_capability/metrics/eth_vs_ours_tiny_recovery_v1.csv")
    ap.add_argument("--v1-negatives", default="outputs/shuttle_capability/metrics/eth_only_v1_frozen_test_negatives.csv")
    ap.add_argument("--v2-negatives", default="outputs/shuttle_capability/metrics/eth_hardneg_v2_frozen_test_negatives.csv")
    ap.add_argument("--tiny-negatives", default="outputs/shuttle_capability/metrics/tiny_recovery_v1_frozen_test_negatives.csv")
    ap.add_argument("--v1-model", default="eth_only_v1_best")
    ap.add_argument("--v2-model", default="eth_real_hardneg_v2_best")
    ap.add_argument("--tiny-model", default="eth_tiny_recovery_v1_best")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--cohorts", default="v1,v2,tiny",
                    help="which cohorts to emit; use v1,v2 before the tiny run exists")
    ap.add_argument("--out-dir", default="outputs/shuttle_capability/metrics")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    out = repo / a.out_dir

    def pooled(path):
        p = repo / path
        return A.summarize_false_positives(_rows(p), a.threshold) if p.is_file() else {}

    v1_pool, v2_pool, tiny_pool = pooled(a.v1_negatives), pooled(a.v2_negatives), pooled(a.tiny_negatives)
    v1_rows, v2_rows, tiny_rows = _rows(repo / a.v1_table), _rows(repo / a.v2_table), _rows(repo / a.tiny_table)

    def bundle(rows, model, pool):
        m = metrics_from_row(_one(rows, model))
        m["fp_no_target"] = pool.get("fp_total")
        m["fp_per_image"] = pool.get("fp_per_image")
        m["fp_per_image_ci95"] = pool.get("fp_per_image_ci95")
        m["fp_total_ci95"] = pool.get("fp_total_ci95")
        m["pooled_images"] = pool.get("images")
        m["pooled_per_set"] = pool.get("per_set")
        return m

    v1, v2, tiny = (bundle(v1_rows, a.v1_model, v1_pool), bundle(v2_rows, a.v2_model, v2_pool),
                    bundle(tiny_rows, a.tiny_model, tiny_pool))

    main_rows = build_main_table(v1, v2, tiny)
    write_csv(out / "tiny_recovery_v1_vs_v2.csv", main_rows, MAIN_COLUMNS)

    wanted = [c.strip() for c in str(a.cohorts).split(",") if c.strip()]
    cohort_models = {"v1": (a.v1_model, v1_rows), "v2": (a.v2_model, v2_rows),
                     "tiny": (a.tiny_model, tiny_rows)}
    size_rows = []
    for _cohort in wanted:
        model, rows = cohort_models.get(_cohort, (None, []))
        if model is None:
            continue
        for r in rows:
            if r.get("set") != "val|eth_unseen":
                continue
            bucket_cols = [c for c in r if c.startswith("Recall_") or c.startswith("TP_") or c.startswith("GT_")]
            for c in sorted(bucket_cols):
                size_rows.append({"model": model, "metric": c, "value": r[c]})
    write_csv(out / "tiny_recovery_v1_size_buckets.csv", size_rows,
              ["model", "metric", "value"])

    fp_rows = []
    for name, model, pool in (("v1", a.v1_model, v1_pool), ("v2", a.v2_model, v2_pool),
                              ("tiny", a.tiny_model, tiny_pool)):
        if name not in wanted:
            continue
        fp_rows.append({"cohort": name, "model": model, "images": pool.get("images"),
                        "fp_total": pool.get("fp_total"), "fp_per_image": pool.get("fp_per_image"),
                        "fp_total_ci95_low": (pool.get("fp_total_ci95") or [None, None])[0],
                        "fp_total_ci95_high": (pool.get("fp_total_ci95") or [None, None])[1],
                        "image_fp_rate": pool.get("image_fp_rate"),
                        "image_fp_rate_ci95_low": (pool.get("image_fp_rate_ci95") or [None, None])[0],
                        "image_fp_rate_ci95_high": (pool.get("image_fp_rate_ci95") or [None, None])[1]})
    for name, pool in (("v1", v1_pool), ("v2", v2_pool), ("tiny", tiny_pool)):
        if name not in wanted:
            continue
        for set_name, s in (pool.get("per_set") or {}).items():
            fp_rows.append({"cohort": name + ":" + set_name, "model": "", "images": s.get("images"),
                            "fp_total": s.get("fp_total"), "fp_per_image": s.get("fp_per_image"),
                            "fp_total_ci95_low": "", "fp_total_ci95_high": "", "image_fp_rate": s.get("image_fp_rate"),
                            "image_fp_rate_ci95_low": "", "image_fp_rate_ci95_high": ""})
    write_csv(out / "tiny_recovery_v1_no_target_fp.csv", fp_rows)

    decision = compute_tiny_decision(v2, tiny)
    if "tiny" not in wanted:
        decision["status"] = "BASELINE_ONLY"
        decision["note"] = ("tiny cohort not evaluated yet: the decision fields compare V2 against empty tiny inputs "
                            "and must not be read as a result")
    decision["cohorts_requested"] = wanted
    decision["primary_endpoint"] = "val|eth_unseen TP_<8"
    decision["cohorts"] = {"v1": v1, "v2": v2, "tiny": tiny}
    (out / "tiny_recovery_v1_decision.json").write_text(json.dumps(decision, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
    print("[tiny-cmp] %s: %s" % (decision["decision"], decision["why"]))
    print("[tiny-cmp] TP_<8 v2=%s tiny=%s | recall %s -> %s | no-target FP %s -> %s"
          % (v2.get("tp_lt8"), tiny.get("tp_lt8"), v2.get("recall"), tiny.get("recall"),
             v2.get("fp_no_target"), tiny.get("fp_no_target")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
