#!/usr/bin/env python3
"""Read-only pre-training V1 false-positive / false-negative diagnostic (Task 3 of the V2 plan).

Nothing here can change V2 membership or any hyperparameter: it only reads frozen evaluation outputs and the
canonical evaluator's own GT decode and matcher (tools/eval_yolo26_v1.py) to explain the V1 error structure.

Ruling 3 endpoint: the pooled 496-image real no-target FP/image is reported with the RAW FP count and a 95%
confidence interval (exact Poisson for the rate, Wilson for the image-level rate), keeping per-set breakdowns.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_yolo26_v1 as ev  # noqa: E402  (canonical GT decode + matcher)

REAL_NO_TARGET_SETS = ("real_images/backgrounds", "real_images/raw", "real_video/frames",
                       "real_match_frames/images", "real_train/raw")
ETH_UNSEEN_CLASSES = ("B_same_location_not_trained", "C_outside_eth_dataset",
                      "C_eth_benchmark_other_split")


def poisson_cdf(lam: float, k: int) -> float:
    return sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(k + 1)) if lam > 0 else 1.0


def _solve_lambda(target: float, k: int) -> float:
    """lambda with Poisson cdf(k; lambda) == target (the cdf is decreasing in lambda)."""
    lo, hi = 1e-9, max(10.0, (k + 1) * 10.0)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if poisson_cdf(mid, k) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def poisson_ci95(k: int):
    """Exact Poisson interval for a count, by bisection on the CDF: deterministic and scipy-free.

    Lower limit solves cdf(k-1; lambda) = 0.975, upper limit solves cdf(k; lambda) = 0.025
    (equivalent to the Garwood/chi-square interval)."""
    if k <= 0:
        return [0.0, 3.689]          # Pois(3.689) P(X=0) = 0.025
    return [round(_solve_lambda(0.975, k - 1), 6), round(_solve_lambda(0.025, k), 6)]


def wilson_ci95(k: int, n: int):
    """Wilson score interval for an image-level rate k/n."""
    if n <= 0:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def summarize_false_positives(rows, threshold=0.25, sets=REAL_NO_TARGET_SETS) -> dict:
    """Pool the raw FP counts of the requested sets at one operating confidence."""
    want = [s.strip() for s in sets] if sets else None
    per_set, tot_img, tot_fp, tot_img_with = {}, 0, 0, 0
    for r in rows:
        if str(r.get("threshold")) not in (str(threshold), "%.2f" % float(threshold)):
            continue
        name = str(r.get("set"))
        if want is not None and name not in want:
            continue
        n = int(float(r.get("images") or 0))
        fp = int(float(r.get("total_FP") or 0))
        iwf = int(float(r.get("images_with_FP") or 0))
        per_set[name] = {"images": n, "fp_total": fp, "fp_per_image": (fp / n if n else None),
                         "images_with_fp": iwf, "image_fp_rate": (iwf / n if n else None),
                         "max_confidence_fp": float(r["max_confidence_FP"]) if r.get("max_confidence_FP") else None}
        tot_img += n
        tot_fp += fp
        tot_img_with += iwf
    rate = (tot_fp / tot_img) if tot_img else None
    ci = poisson_ci95(tot_fp) if tot_img else [None, None]
    return {
        "threshold": float(threshold),
        "sets": sorted(per_set),
        "images": tot_img,
        "fp_total": tot_fp,
        "fp_per_image": rate,
        "fp_per_image_ci95": [round(ci[0] / tot_img, 6) if tot_img else None,
                              round(ci[1] / tot_img, 6) if tot_img else None],
        "fp_total_ci95": ci,
        "images_with_fp": tot_img_with,
        "image_fp_rate": (tot_img_with / tot_img) if tot_img else None,
        "image_fp_rate_ci95": wilson_ci95(tot_img_with, tot_img),
        "per_set": per_set,
    }


def summarize_false_negatives(per_gt) -> dict:
    """FN structure over per-GT match rows (matched=False means missed at the chosen IoU)."""
    by_bucket, gt_by_bucket, confs = {}, {}, []
    fn_total = gt_total = 0
    lt8_gt = lt8_fn = 0
    for r in per_gt:
        b = str(r.get("bucket") or "?")
        gt_by_bucket[b] = gt_by_bucket.get(b, 0) + 1
        gt_total += 1
        if not bool(r.get("matched")):
            by_bucket[b] = by_bucket.get(b, 0) + 1
            fn_total += 1
            if r.get("best_iou") is not None:
                confs.append(float(r["best_iou"]))
        if b in ("<4", "4-6", "6-8"):
            lt8_gt += 1
            if not bool(r.get("matched")):
                lt8_fn += 1
    return {
        "gt_total": gt_total,
        "fn_total": fn_total,
        "fn_rate": (fn_total / gt_total) if gt_total else None,
        "fn_by_bucket": by_bucket,
        "gt_by_bucket": gt_by_bucket,
        "fn_rate_by_bucket": {b: (by_bucket.get(b, 0) / gt_by_bucket[b]) for b in gt_by_bucket if gt_by_bucket[b]},
        "small_lt8_gt": lt8_gt,
        "small_lt8_fn": lt8_fn,
        "small_lt8_fn_rate": (lt8_fn / lt8_gt) if lt8_gt else None,
        "median_best_iou_of_fn": (sorted(confs)[len(confs) // 2] if confs else None),
    }


def compare_eth_tp_v1_fn(eth_rows, v1_rows) -> dict:
    """GT that the ETH official model hits but V1 misses: the recoverable recall gap."""
    key = lambda r: (str(r.get("image")), int(r.get("gt_index", 0)))
    eth = {key(r): bool(r.get("matched")) for r in eth_rows}
    v1 = {key(r): bool(r.get("matched")) for r in v1_rows}
    both = sorted(set(eth) & set(v1))
    eth_tp_v1_fn = [k for k in both if eth[k] and not v1[k]]
    v1_tp_eth_fn = [k for k in both if v1[k] and not eth[k]]
    return {
        "common_gt": len(both),
        "eth_tp_v1_fn": len(eth_tp_v1_fn),
        "v1_tp_eth_fn": len(v1_tp_eth_fn),
        "eth_tp_v1_fn_examples": ["%s#%d" % k for k in eth_tp_v1_fn[:20]],
        "eth_tp_total": sum(1 for k in both if eth[k]),
        "v1_tp_total": sum(1 for k in both if v1[k]),
    }


def load_rows(path) -> list:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def build_val_match_table(dump_json, dataset_manifest, leakage_inventory=None, classes=None,
                          split="val", iou_thr=0.5, conf_thr=0.25) -> list:
    """Per-GT match rows for one checkpoint dump, reusing the canonical GT decode and greedy matcher.

    conf_thr must stay at the operating confidence (0.25): matching against the dump's AP floor (0.001) would
    over-count hits and disagree with the evaluator's published operating-point Recall."""
    dump = json.loads(Path(dump_json).read_text(encoding="utf-8"))
    dets = dump.get("dets") or {}
    allowed = None
    if leakage_inventory and classes:
        keep = set()
        for r in load_rows(leakage_inventory):
            if str(r.get("leakage_class")) in set(classes):
                keep.add(str(r.get("image")).replace(chr(92), "/").lower())
        allowed = keep
    rows = []
    for r in load_rows(dataset_manifest):
        if split and str(r.get("split")) != split:
            continue
        img = str(r.get("image") or "")
        label = str(r.get("label") or "")
        if not img:
            continue
        if allowed is not None and img.replace(chr(92), "/").lower() not in allowed:
            continue
        gts = ev.load_gt_boxes(img, label) if label else []
        det = dets.get(img) or dets.get(str(Path(img).resolve())) or []
        pairs = [(float(d["conf"]), [float(x) for x in d["box"]]) for d in det
                 if float(d["conf"]) >= float(conf_thr)]
        res = ev.match_greedy([g["box"] for g in gts], pairs, iou_thr, metric="iou")
        best = {}
        for di, gj in res["matches"]:
            best[gj] = pairs[di][0]
        for gj, g in enumerate(gts):
            rows.append({"image": img, "gt_index": gj, "bucket": g["bucket"], "eq640": g["eq640"],
                         "matched": bool(res["gt_hit"][gj]), "iou_thr": iou_thr, "conf_thr": conf_thr,
                         "conf": best.get(gj)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V1 FP/FN diagnostic over frozen evaluation outputs (read-only)")
    ap.add_argument("--v1-dump", default="_scratch_localization_audit/loc_eth_only_v1_best_val.json")
    ap.add_argument("--eth-dump", default="_scratch_localization_audit/loc_eth_official_val.json")
    ap.add_argument("--dataset-manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--leakage-inventory", default="outputs/shuttle_capability/metrics/eth_data_leakage_inventory.csv")
    ap.add_argument("--no-target-negatives",
                    default="outputs/shuttle_capability/metrics/eth_only_v1_frozen_test_negatives.csv")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out-json", default="outputs/shuttle_capability/metrics/eth_only_v1_error_diagnostic.json")
    ap.add_argument("--out-csv", default="outputs/shuttle_capability/metrics/eth_only_v1_error_diagnostic.csv")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    out = {"scope": "read-only pre-training V1 diagnostic; cannot change V2 membership or hyperparameters"}

    neg_path = repo / a.no_target_negatives
    if neg_path.is_file():
        fp = summarize_false_positives(load_rows(neg_path), a.threshold)
        out["false_positives_real_no_target"] = fp
        out["false_positives_synthetic_diagnostic"] = summarize_false_positives(
            load_rows(neg_path), a.threshold,
            sets=("synthetic_on_real_bg/images", "synthetic_3d/images"))
    else:
        out["false_positives_real_no_target"] = {"error": "negatives file missing: %s" % neg_path}

    dump = repo / a.v1_dump
    if dump.is_file():
        v1_rows = build_val_match_table(dump, repo / a.dataset_manifest, repo / a.leakage_inventory,
                                        ETH_UNSEEN_CLASSES, split="val", iou_thr=a.iou_thr,
                                        conf_thr=a.threshold)
        out["val_eth_unseen_v1"] = summarize_false_negatives(v1_rows)
        eth_dump = repo / a.eth_dump
        if eth_dump.is_file():
            eth_rows = build_val_match_table(eth_dump, repo / a.dataset_manifest, repo / a.leakage_inventory,
                                             ETH_UNSEEN_CLASSES, split="val", iou_thr=a.iou_thr,
                                             conf_thr=a.threshold)
            out["val_eth_unseen_v1_vs_eth"] = compare_eth_tp_v1_fn(eth_rows, v1_rows)
    else:
        out["val_eth_unseen_v1"] = {"error": "V1 prediction dump missing: %s" % dump}

    out["headline"] = ("V1 real no-target FP total %s over %s images (FP/image %s, 95%% CI %s); "
                       "val|eth_unseen V1 misses %s of %s GT"
                       % (out.get("false_positives_real_no_target", {}).get("fp_total"),
                          out.get("false_positives_real_no_target", {}).get("images"),
                          out.get("false_positives_real_no_target", {}).get("fp_per_image"),
                          out.get("false_positives_real_no_target", {}).get("fp_per_image_ci95"),
                          out.get("val_eth_unseen_v1", {}).get("fn_total"),
                          out.get("val_eth_unseen_v1", {}).get("gt_total")))
    Path(repo / a.out_json).parent.mkdir(parents=True, exist_ok=True)
    (repo / a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    flat = []

    def emit(scope, metric, value, set_name=""):
        flat.append({"scope": scope, "set": set_name, "metric": metric, "value": value})

    fp = out.get("false_positives_real_no_target") or {}
    for k in ("images", "fp_total", "fp_per_image", "images_with_fp", "image_fp_rate"):
        emit("real_no_target_pooled", k, fp.get(k))
    emit("real_no_target_pooled", "fp_per_image_ci95_low", (fp.get("fp_per_image_ci95") or [None, None])[0])
    emit("real_no_target_pooled", "fp_per_image_ci95_high", (fp.get("fp_per_image_ci95") or [None, None])[1])
    for name, s in (fp.get("per_set") or {}).items():
        for k in ("images", "fp_total", "fp_per_image", "image_fp_rate"):
            emit("real_no_target_per_set", k, s.get(k), name)
    fn = out.get("val_eth_unseen_v1") or {}
    for k in ("gt_total", "fn_total", "fn_rate", "small_lt8_gt", "small_lt8_fn", "small_lt8_fn_rate"):
        emit("val_eth_unseen", k, fn.get(k))
    for b, v in (fn.get("fn_rate_by_bucket") or {}).items():
        emit("val_eth_unseen_bucket", "fn_rate", v, b)
    cmp_ = out.get("val_eth_unseen_v1_vs_eth") or {}
    for k in ("common_gt", "eth_tp_v1_fn", "v1_tp_eth_fn", "eth_tp_total", "v1_tp_total"):
        emit("val_eth_unseen_eth_vs_v1", k, cmp_.get(k))
    with (repo / a.out_csv).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "set", "metric", "value"])
        w.writeheader()
        for r in flat:
            w.writerow(r)
    print("[diag] " + out["headline"])
    print("[diag] wrote %s and %s" % (a.out_json, a.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
