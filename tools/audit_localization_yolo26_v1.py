#!/usr/bin/env python3
"""Localization audit for the YOLO26 V1 shuttle detector (DIAGNOSTIC ONLY).

Answers one question with numbers: when mAP50-95 rises between two checkpoints, is that because the
detector finds MORE objects (recall) or because it LOCATES the objects it already finds more
precisely (bbox quality)?

Hard scope rules: reads existing checkpoints and existing evaluation data only; never trains; never
touches model/config/dataset/labels/sampler/augmentation; unlabeled sets are NEVER used for bbox
localization; IoU 0.30-0.50 near misses are reported separately and never counted as TP.

Outputs (default outputs/shuttle_capability/metrics):
    localization_ap_by_iou.csv            AP50..AP95 per checkpoint x set
    localization_iou_distribution.csv     IoU stats + histogram over matched TPs
    localization_error_summary.csv        center / width / height error stats
    localization_size_buckets.csv         per-bucket localization metrics (equiv_size_640)
    localization_checkpoint_compare.csv   A vs B bucket deltas
    localization_near_miss.csv            A/B/C/D classification per GT
and example images under outputs/shuttle_capability/localization_examples/.

Run:  python tools/audit_localization_yolo26_v1.py --repo . --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_yolo26_v1 as ev  # noqa: E402

AUDIT_IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
BUCKET_AP_THRESHOLDS = [0.50, 0.75, 0.90, 0.95]
TP_IOU_MIN = 0.50
NEAR_MISS_BANDS = [("A", 0.50, 2.0), ("B", 0.30, 0.50), ("C", 0.10, 0.30), ("D", -1.0, 0.10)]
IOU_HIST_BINS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.0001)]
IOU_HIST_LABELS = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", ">=0.95"]
KEY_BUCKETS = ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]
EXAMPLES_PER_KIND = 30
DEFAULT_SETS = ["val", "controlled_capability/images", "challenge_test/images"]


# ---------------------------------------------------------------------------------------
# pure geometry / error functions
# ---------------------------------------------------------------------------------------
def box_wh(box):
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def box_center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_error(gt_box, pred_box):
    """Signed dx, dy and euclidean center distance, in ORIGINAL image pixels."""
    gx, gy = box_center(gt_box)
    px, py = box_center(pred_box)
    dx, dy = px - gx, py - gy
    return dx, dy, math.hypot(dx, dy)


def width_relative_error(gt_box, pred_box):
    gw = box_wh(gt_box)[0]
    pw = box_wh(pred_box)[0]
    return abs(pw - gw) / gw if gw > 0 else None


def height_relative_error(gt_box, pred_box):
    gh = box_wh(gt_box)[1]
    ph = box_wh(pred_box)[1]
    return abs(ph - gh) / gh if gh > 0 else None


def area_ratio(gt_box, pred_box):
    gw, gh = box_wh(gt_box)
    pw, ph = box_wh(pred_box)
    if gw * gh <= 0:
        return None
    return (pw * ph) / (gw * gh)


def signed_size_ratios(gt_box, pred_box):
    gw, gh = box_wh(gt_box)
    pw, ph = box_wh(pred_box)
    wr = (pw / gw) if gw > 0 else None
    hr = (ph / gh) if gh > 0 else None
    lwr = math.log(wr) if (wr and wr > 0) else None
    lhr = math.log(hr) if (hr and hr > 0) else None
    return wr, hr, lwr, lhr


def normalized_center_error(gt_box, pred_box, dist=None):
    """center error / sqrt(gt_w*gt_h): scale free, so 2 px on a 100 px box != 2 px on a 6 px shuttle."""
    gw, gh = box_wh(gt_box)
    if gw * gh <= 0:
        return None
    d = center_error(gt_box, pred_box)[2] if dist is None else dist
    return d / math.sqrt(gw * gh)


def scale_error(px, img_w, img_h, target):
    """Re-express a length measured in original pixels inside a target-sized input frame."""
    if px is None:
        return None
    m = max(img_w, img_h)
    return px * (float(target) / m) if m > 0 else None


# ---------------------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------------------
def quantiles(values, qs=(0.10, 0.25, 0.75, 0.90, 0.95)):
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    return {("P%d" % round(q * 100)): ev.percentile(vals, q) for q in qs}


def basic_stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def iou_histogram(ious):
    hist = {lab: 0 for lab in IOU_HIST_LABELS}
    for v in ious:
        for (lo, hi), lab in zip(IOU_HIST_BINS, IOU_HIST_LABELS):
            if lo <= v < hi:
                hist[lab] += 1
                break
    return hist


def near_miss_class(best_iou):
    for name, lo, hi in NEAR_MISS_BANDS:
        if lo <= best_iou < hi:
            return name
    return "D"


# ---------------------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------------------
def match_by_iou(gt_boxes, dets, min_iou=0.0):
    """One-to-one greedy matching by DESCENDING IoU (near-miss diagnosis only).

    dets: list of (conf, box) -> [(gt_index, det_index, iou), ...].
    """
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for di, (_c, d) in enumerate(dets):
            v = ev.iou_xyxy(g, d)
            if v >= min_iou:
                pairs.append((v, gi, di))
    pairs.sort(key=lambda t: -t[0])
    gused, dused, out = set(), set(), []
    for v, gi, di in pairs:
        if gi in gused or di in dused:
            continue
        gused.add(gi)
        dused.add(di)
        out.append((gi, di, v))
    return out


# ---------------------------------------------------------------------------------------
# per-image evaluation
# ---------------------------------------------------------------------------------------
def det_pairs(sample, conf_op=0.0):
    return [(d["conf"], d["box"]) for d in sample["dets"] if d["conf"] >= conf_op]


def error_record(gt_box, pred_box, img_w, img_h):
    """Every localization error of one matched GT/prediction pair."""
    dx, dy, dist = center_error(gt_box, pred_box)
    gw, gh = box_wh(gt_box)
    pw, ph = box_wh(pred_box)
    wr, hr, lwr, lhr = signed_size_ratios(gt_box, pred_box)
    return {
        "dx": dx, "dy": dy, "abs_dx": abs(dx), "abs_dy": abs(dy),
        "dist": dist, "dist640": scale_error(dist, img_w, img_h, 640),
        "norm": normalized_center_error(gt_box, pred_box, dist),
        "w_rel": width_relative_error(gt_box, pred_box),
        "h_rel": height_relative_error(gt_box, pred_box),
        "area_ratio": area_ratio(gt_box, pred_box),
        "w_ratio": wr, "h_ratio": hr, "log_w": lwr, "log_h": lhr,
        "gt_w": gw, "gt_h": gh,
        "eq640": ev.equiv_size_640(gw, gh, img_w, img_h),
    }


def stats_with_q(values, qs=(0.75, 0.90, 0.95)):
    out = basic_stats(values)
    vals = [v for v in values if v is not None]
    for q in qs:
        out["P%d" % round(q * 100)] = ev.percentile(vals, q) if vals else None
    return out


def summarize_errors(recs):
    if not recs:
        return {"n": 0}
    def col(k):
        return [r.get(k) for r in recs]
    return {
        "n": len(recs),
        "center_dist_px": stats_with_q(col("dist")),
        "center_dist_px_640": stats_with_q(col("dist640")),
        "normalized_center_error": stats_with_q(col("norm")),
        "center_dx_signed": stats_with_q(col("dx")),
        "center_dy_signed": stats_with_q(col("dy")),
        "center_abs_dx": stats_with_q(col("abs_dx")),
        "center_abs_dy": stats_with_q(col("abs_dy")),
        "width_rel_err": stats_with_q(col("w_rel")),
        "height_rel_err": stats_with_q(col("h_rel")),
        "area_ratio": stats_with_q(col("area_ratio")),
        "width_ratio_signed": stats_with_q(col("w_ratio")),
        "height_ratio_signed": stats_with_q(col("h_ratio")),
        "log_width_ratio": stats_with_q(col("log_w")),
        "log_height_ratio": stats_with_q(col("log_h")),
        "gt_size_px_640": stats_with_q(col("eq640")),
    }


def evaluate_localization(samples, conf_op=0.25, iou_thresholds=None, bucket_aps=None):
    """Localization metrics for one checkpoint on one labeled set.

    samples: [{"key", "image", "img_w", "img_h", "gt": [{"box","eq640","bucket"}], "dets": [{"box","conf"}]}]
    AP convention is the repository one: per image AP over the confidence-sorted PR curve, then a
    GT-count weighted mean across images (same as the val / frozen_test reports).
    """
    iou_thresholds = list(iou_thresholds or AUDIT_IOU_THRESHOLDS)
    bucket_aps = list(bucket_aps or BUCKET_AP_THRESHOLDS)
    ap_acc = {t: [] for t in iou_thresholds}
    bucket_ap_acc = {lab: {t: [] for t in bucket_aps} for lab in ev.BUCKET_LABELS}
    tp_ious, tp_recs = [], []
    nm_recs = {name: [] for name, _lo, _hi in NEAR_MISS_BANDS}
    buckets = {}
    for lab in ev.BUCKET_LABELS:
        buckets[lab] = {"bucket": lab, "gt": 0, "tp50": 0, "recall": None, "iou": [], "recs": []}
    n_img = n_gt = n_det_op = n_det_all = n_tp = 0
    nm_counts = {name: 0 for name, _lo, _hi in NEAR_MISS_BANDS}
    for s in samples:
        gts = s.get("gt") or []
        gboxes = [g["box"] for g in gts]
        dets_all = det_pairs(s, 0.0)
        dets_op = det_pairs(s, conf_op)
        W = float(s.get("img_w") or 1.0)
        H = float(s.get("img_h") or 1.0)
        n_img += 1
        n_gt += len(gts)
        n_det_all += len(dets_all)
        n_det_op += len(dets_op)
        if gts:
            for t in iou_thresholds:
                ap_acc[t].append((ev.ap_for_gt_set(gboxes, dets_all, t), len(gboxes)))
            by_bucket = {}
            for gi, g in enumerate(gts):
                by_bucket.setdefault(g["bucket"], []).append(gi)
                buckets[g["bucket"]]["gt"] += 1
            for lab, idxs in by_bucket.items():
                sub = [gboxes[i] for i in idxs]
                for t in bucket_aps:
                    bucket_ap_acc[lab][t].append((ev.ap_for_gt_set(sub, dets_all, t), len(sub)))
        # --- TPs at IoU >= 0.5, operating confidence ---
        if gts:
            m05 = ev.match_greedy(gboxes, dets_op, TP_IOU_MIN, metric="iou")
            for di, gj in m05["matches"]:
                v = ev.iou_xyxy(dets_op[di][1], gboxes[gj])
                rec = error_record(gboxes[gj], dets_op[di][1], W, H)
                rec["iou"] = v
                rec["bucket"] = gts[gj]["bucket"]
                rec["key"] = s.get("key")
                rec["box"] = gboxes[gj]
                rec["pred_box"] = dets_op[di][1]
                rec["conf"] = dets_op[di][0]
                tp_ious.append(v)
                tp_recs.append(rec)
                buckets[gts[gj]["bucket"]]["tp50"] += 1
                buckets[gts[gj]["bucket"]]["iou"].append(v)
                buckets[gts[gj]["bucket"]]["recs"].append(rec)
                n_tp += 1
            # --- near-match classification of EVERY GT (one-to-one, best IoU) ---
            m00 = ev.match_greedy(gboxes, dets_op, 0.0, metric="iou")
            paired = {gj: di for di, gj in m00["matches"]}
            for gj in range(len(gts)):
                di = paired.get(gj)
                v = ev.iou_xyxy(dets_op[di][1], gboxes[gj]) if di is not None else 0.0
                cls = near_miss_class(v)
                nm_counts[cls] += 1
                entry = {"key": s.get("key"), "iou": v, "bucket": gts[gj]["bucket"]}
                if di is not None:
                    entry.update(error_record(gboxes[gj], dets_op[di][1], W, H))
                entry["box"] = gboxes[gj]
                entry["pred_box"] = dets_op[di][1] if di is not None else None
                entry["conf"] = dets_op[di][0] if di is not None else None
                nm_recs[cls].append(entry)
    ap_by_iou = {}
    for t in iou_thresholds:
        acc = [a for a in ap_acc[t] if a[0] is not None]
        n = sum(a[1] for a in acc)
        ap_by_iou[t] = (sum((a[0] or 0.0) * a[1] for a in acc) / n) if n else None
    aps = [v for v in ap_by_iou.values() if v is not None]
    for lab in ev.BUCKET_LABELS:
        b = buckets[lab]
        b["recall"] = (b["tp50"] / b["gt"]) if b["gt"] else None
        for t in bucket_aps:
            acc = [a for a in bucket_ap_acc[lab][t] if a[0] is not None]
            n = sum(a[1] for a in acc)
            b["AP%d" % round(t * 100)] = (sum((a[0] or 0.0) * a[1] for a in acc) / n) if n else None
        b["iou_mean"] = statistics.fmean(b["iou"]) if b["iou"] else None
        b["iou_median"] = statistics.median(b["iou"]) if b["iou"] else None
        errs = summarize_errors(b["recs"])
        b["center_dist_px_640_median"] = (errs.get("center_dist_px_640") or {}).get("median")
        b["normalized_center_error_median"] = (errs.get("normalized_center_error") or {}).get("median")
        b["width_rel_err_median"] = (errs.get("width_rel_err") or {}).get("median")
        b["height_rel_err_median"] = (errs.get("height_rel_err") or {}).get("median")
    overall_errs = summarize_errors(tp_recs)
    return {
        "images": n_img, "gt": n_gt, "detections_all": n_det_all, "detections_op": n_det_op,
        "tp50": n_tp, "fp50": max(0, n_det_op - n_tp), "fn50": max(0, n_gt - n_tp),
        "recall50": (n_tp / n_gt) if n_gt else None,
        "precision_op": (n_tp / n_det_op) if n_det_op else None,
        "ap_by_iou": ap_by_iou,
        "mAP50-95": (statistics.fmean(aps) if aps else None),
        "iou_dist": {"n": len(tp_ious), "stats": basic_stats(tp_ious),
                     "quantiles": quantiles(tp_ious, (0.10, 0.25, 0.50, 0.75, 0.90)),
                     "histogram": iou_histogram(tp_ious)},
        "errors": overall_errs,
        "buckets": buckets,
        "near_miss": {"counts": nm_counts,
                      "bands": {k: len(v) for k, v in nm_recs.items()},
                      "records": nm_recs},
    }


# ---------------------------------------------------------------------------------------
# csv writers
# ---------------------------------------------------------------------------------------
def _num(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return ""
        return round(x, nd)
    return x


def write_ap_by_iou(path, results):
    ths = AUDIT_IOU_THRESHOLDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "set", "images", "GT", "detections_op", "TP@0.5", "FP@0.5", "FN@0.5",
                    "Precision@op", "Recall@0.5"]
                   + ["AP%d" % round(t * 100) for t in ths] + ["mAP50-95", "AP_drop_50_to_95"])
        for (name, set_key), r in results.items():
            aps = [r["ap_by_iou"][t] for t in ths]
            drop = (aps[0] - aps[-1]) if (aps[0] is not None and aps[-1] is not None) else None
            w.writerow([name, set_key, r["images"], r["gt"], r["detections_op"], r["tp50"], r["fp50"],
                        r["fn50"], _num(r["precision_op"]), _num(r["recall50"])]
                       + [_num(a) for a in aps] + [_num(r["mAP50-95"]), _num(drop)])


def write_iou_distribution(path, results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "set", "n_tp", "mean", "median", "std", "P10", "P25", "P50", "P75", "P90"]
                   + [("bin_" + lab) for lab in IOU_HIST_LABELS])
        for (name, set_key), r in results.items():
            st = r["iou_dist"]["stats"]
            q = r["iou_dist"]["quantiles"]
            w.writerow([name, set_key, r["iou_dist"]["n"], _num(st["mean"]), _num(st["median"]),
                        _num(st["std"]), _num(q.get("P10")), _num(q.get("P25")), _num(q.get("P50")),
                        _num(q.get("P75")), _num(q.get("P90"))]
                       + [r["iou_dist"]["histogram"][lab] for lab in IOU_HIST_LABELS])


def write_error_summary(path, results):
    cols = [
        ("center_dist_px", "px"), ("center_dist_px_640", "px640"), ("normalized_center_error", "norm"),
        ("center_abs_dx", "px"), ("center_abs_dy", "px"), ("width_rel_err", ""), ("height_rel_err", ""),
        ("area_ratio", ""), ("width_ratio_signed", ""), ("height_ratio_signed", ""),
        ("log_width_ratio", ""), ("log_height_ratio", ""),
    ]
    header = ["checkpoint", "set", "n_tp"]
    for key, _u in cols:
        for stat in ["mean", "median", "std", "P75", "P90", "P95"]:
            header.append("%s_%s" % (key, stat))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (name, set_key), r in results.items():
            e = r["errors"]
            row = [name, set_key, e.get("n", 0)]
            for key, _u in cols:
                d = e.get(key) or {}
                for stat in ["mean", "median", "std", "P75", "P90", "P95"]:
                    row.append(_num(d.get(stat)))
            w.writerow(row)


def write_size_buckets(path, results):
    header = ["checkpoint", "set", "bucket", "GT", "TP@0.5", "Recall@0.5", "iou_mean", "iou_median",
              "center_err_px640_median", "norm_center_err_median", "width_rel_err_median",
              "height_rel_err_median"] + ["AP%d" % round(t * 100) for t in BUCKET_AP_THRESHOLDS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (name, set_key), r in results.items():
            for lab in ev.BUCKET_LABELS:
                b = r["buckets"][lab]
                w.writerow([name, set_key, lab, b["gt"], b["tp50"], _num(b["recall"]), _num(b["iou_mean"]),
                            _num(b["iou_median"]), _num(b["center_dist_px_640_median"]),
                            _num(b["normalized_center_error_median"]), _num(b["width_rel_err_median"]),
                            _num(b["height_rel_err_median"])]
                           + [_num(b.get("AP%d" % round(t * 100))) for t in BUCKET_AP_THRESHOLDS])


def write_checkpoint_compare(path, results, name_a="a_best", name_b="b_best"):
    sets = sorted({k[1] for k in results})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "bucket", "GT_a", "GT_b", "Recall_a", "Recall_b", "d_Recall", "iou_med_a",
                    "iou_med_b", "d_iou_med", "center_px640_a", "center_px640_b", "d_center_px640",
                    "width_rel_a", "width_rel_b", "d_width_rel", "height_rel_a", "height_rel_b",
                    "d_height_rel", "norm_center_a", "norm_center_b", "d_norm_center"]
                   + ["AP%d_a" % round(t * 100) for t in BUCKET_AP_THRESHOLDS]
                   + ["AP%d_b" % round(t * 100) for t in BUCKET_AP_THRESHOLDS]
                   + ["d_AP%d" % round(t * 100) for t in BUCKET_AP_THRESHOLDS])
        for set_key in sets:
            ra, rb = results.get((name_a, set_key)), results.get((name_b, set_key))
            if ra is None or rb is None:
                continue
            rows = [("ALL", ra, rb)] + [(lab, ra["buckets"][lab], rb["buckets"][lab]) for lab in ev.BUCKET_LABELS]
            for scope, a, b in rows:
                def d(x, y):
                    return (x - y) if (isinstance(x, float) and isinstance(y, float)) else None
                w.writerow([set_key, scope, a.get("gt"), b.get("gt"), _num(a.get("recall")), _num(b.get("recall")),
                            _num(d(b.get("recall"), a.get("recall"))), _num(a.get("iou_median")),
                            _num(b.get("iou_median")), _num(d(b.get("iou_median"), a.get("iou_median"))),
                            _num(a.get("center_dist_px_640_median")), _num(b.get("center_dist_px_640_median")),
                            _num(d(b.get("center_dist_px_640_median"), a.get("center_dist_px_640_median"))),
                            _num(a.get("width_rel_err_median")), _num(b.get("width_rel_err_median")),
                            _num(d(b.get("width_rel_err_median"), a.get("width_rel_err_median"))),
                            _num(a.get("height_rel_err_median")), _num(b.get("height_rel_err_median")),
                            _num(d(b.get("height_rel_err_median"), a.get("height_rel_err_median"))),
                            _num(a.get("normalized_center_error_median")), _num(b.get("normalized_center_error_median")),
                            _num(d(b.get("normalized_center_error_median"), a.get("normalized_center_error_median")))]
                           + [_num(a.get("AP%d" % round(t * 100))) for t in BUCKET_AP_THRESHOLDS]
                           + [_num(b.get("AP%d" % round(t * 100))) for t in BUCKET_AP_THRESHOLDS]
                           + [_num(d(b.get("AP%d" % round(t * 100)), a.get("AP%d" % round(t * 100))))
                              for t in BUCKET_AP_THRESHOLDS])


def write_near_miss(path, results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "set", "scope", "class", "GT", "count", "share", "mean_iou",
                    "mean_center_px640", "median_center_px640", "median_norm_center",
                    "mean_width_rel", "mean_height_rel"])
        for (name, set_key), r in results.items():
            recs = r["near_miss"]["records"]
            total = r["gt"] or 0
            for cls, _lo, _hi in NEAR_MISS_BANDS:
                w.writerow([name, set_key, "ALL", cls, total, len(recs[cls]),
                            _num(len(recs[cls]) / total if total else None),
                            _num(statistics.fmean([x["iou"] for x in recs[cls]]) if recs[cls] else None),
                            _num(statistics.fmean([x["dist640"] for x in recs[cls] if x.get("dist640") is not None])
                                 if any(x.get("dist640") is not None for x in recs[cls]) else None),
                            _num(statistics.median([x["dist640"] for x in recs[cls] if x.get("dist640") is not None])
                                 if any(x.get("dist640") is not None for x in recs[cls]) else None),
                            _num(statistics.median([x["norm"] for x in recs[cls] if x.get("norm") is not None])
                                 if any(x.get("norm") is not None for x in recs[cls]) else None),
                            _num(statistics.fmean([x["w_rel"] for x in recs[cls] if x.get("w_rel") is not None])
                                 if any(x.get("w_rel") is not None for x in recs[cls]) else None),
                            _num(statistics.fmean([x["h_rel"] for x in recs[cls] if x.get("h_rel") is not None])
                                 if any(x.get("h_rel") is not None for x in recs[cls]) else None)])
            for lab in ev.BUCKET_LABELS:
                sub = {c: [x for x in recs[c] if x["bucket"] == lab] for c, _l, _h in NEAR_MISS_BANDS}
                ngt = r["buckets"][lab]["gt"] or 0
                for cls, _lo, _hi in NEAR_MISS_BANDS:
                    w.writerow([name, set_key, lab, cls, ngt, len(sub[cls]),
                                _num(len(sub[cls]) / ngt if ngt else None),
                                _num(statistics.fmean([x["iou"] for x in sub[cls]]) if sub[cls] else None),
                                "", "", "", "", ""])


# ---------------------------------------------------------------------------------------
# sample building (val + labeled frozen sets only; unlabeled sets are never used here)
# ---------------------------------------------------------------------------------------
def build_val_samples(repo, manifest, remaps, max_images=0):
    from PIL import Image
    rows = ev.load_manifest(repo / manifest, "val", remaps)
    if max_images:
        rows = rows[:max_images]
    samples, skipped = [], 0
    for r in rows:
        img = ev.remap(r["image"], remaps)
        lab = ev.remap(r["label"], remaps) if r.get("label") else ""
        gts = ev.load_gt_boxes(img, lab)
        if gts is None:
            skipped += 1
            continue
        try:
            with Image.open(img) as im:
                W, H = im.size
        except Exception:
            skipped += 1
            continue
        samples.append({"key": img, "image": img, "label": lab, "img_w": W, "img_h": H, "gt": gts})
    return samples, skipped


def build_frozen_samples(repo, set_key, max_images=0):
    from PIL import Image
    set_name = set_key.split("/")[0]
    groups = ev.discover_frozen_groups(repo / ev.FROZEN_ROOT_REL, only=[set_name])
    want = [g for g in groups if g.key == set_key]
    if not want:
        raise SystemExit("no such frozen set: %s (found: %s)" % (set_key, [g.key for g in groups]))
    g = want[0]
    if not g.labeled:
        raise SystemExit("set %s is UNLABELED: refusing to compute bbox localization on it" % set_key)
    lb = g.label_by_stem
    images = g.images[:max_images] if max_images else g.images
    samples, skipped = [], 0
    for p in images:
        lab = lb.get(p.stem)
        gts = ev.load_gt_boxes(p, lab)
        if gts is None:
            skipped += 1
            continue
        try:
            with Image.open(p) as im:
                W, H = im.size
        except Exception:
            skipped += 1
            continue
        samples.append({"key": str(p), "image": str(p), "label": str(lab) if lab else "",
                        "img_w": W, "img_h": H, "gt": gts})
    return samples, skipped


def holdout_root(repo):
    return repo / ev.FROZEN_ROOT_REL


# ---------------------------------------------------------------------------------------
# prediction with an on-disk cache (cache is documentation, never used for training)
# ---------------------------------------------------------------------------------------
def attach_detections(repo, ckpt, set_key, samples, args, cache_dir):
    tag = ("loc_%s_%s" % (ckpt["name"], set_key)).replace("/", "_")
    cache = cache_dir / (tag + ".json")
    sha = ckpt.get("sha256") or ev.file_sha256(ckpt["path"])
    if cache.exists() and not args.no_cache:
        try:
            blob = json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            blob = None
        if blob and blob.get("imgsz") == args.imgsz and blob.get("conf") == args.conf_floor \
                and blob.get("sha256") == sha and blob.get("n_images") == len(samples):
            print("  cache hit: %s" % cache.name, flush=True)
            dets = blob["dets"]
            for s in samples:
                s["dets"] = dets.get(os.path.abspath(s["image"]), [])
            return
        print("  cache stale, re-predicting: %s" % cache.name, flush=True)
    keys = [s["image"] for s in samples]
    print("  predicting %d images with %s (imgsz=%d conf=%.4f) ..." % (len(keys), ckpt["name"], args.imgsz,
                                                                        args.conf_floor), flush=True)
    by_path = ev.predict_group(ckpt["path"], keys, cache_dir, tag, args.imgsz, args.conf_floor, args.device)
    dets = {k: v for k, v in by_path.items()}
    for s in samples:
        s["dets"] = dets.get(os.path.abspath(s["image"]), [])
    cache.write_text(json.dumps({"imgsz": args.imgsz, "conf": args.conf_floor, "sha256": sha,
                                 "n_images": len(samples), "checkpoint": ckpt["name"], "set": set_key,
                                 "dets": dets}), encoding="utf-8")
    print("  cached -> %s" % cache.name, flush=True)


# ---------------------------------------------------------------------------------------
# example visualisation (GT = green, prediction = red, zoom inset around the GT)
# ---------------------------------------------------------------------------------------
def draw_example(image_path, gt_boxes, dets, out_path, target_gt, target_pred, caption, conf_op, zoom_size=256):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(14, W // 55))
    except Exception:
        font = ImageFont.load_default()
    for (c, b) in dets:
        if c < conf_op:
            continue
        d.rectangle(list(b), outline=(255, 80, 80), width=1)
    for b in gt_boxes:
        d.rectangle(list(b), outline=(0, 220, 0), width=2)
    if target_gt is not None:
        d.rectangle(list(target_gt), outline=(0, 255, 255), width=3)
    if target_pred is not None:
        d.rectangle(list(target_pred), outline=(255, 210, 0), width=3)
    try:
        tb = d.textbbox((8, 8), caption[:190], font=font)
        d.rectangle([tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3], fill=(0, 0, 0))
    except Exception:
        pass
    d.text((8, 8), caption[:190], fill=(255, 255, 90), font=font)
    if target_gt is not None:
        cx, cy = (target_gt[0] + target_gt[2]) / 2.0, (target_gt[1] + target_gt[3]) / 2.0
        side = max(48.0, max(target_gt[2] - target_gt[0], target_gt[3] - target_gt[1]) * 4.0)
        x1, y1 = cx - side / 2.0, cy - side / 2.0
        x1, y1 = max(0.0, min(x1, W - side)), max(0.0, min(y1, H - side))
        x2, y2 = min(W, x1 + side), min(H, y1 + side)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            crop = im.crop((int(x1), int(y1), int(x2), int(y2))).resize((zoom_size, zoom_size))
            k = zoom_size / (x2 - x1)
            cd = ImageDraw.Draw(crop)
            def zbox(b):
                return [ (b[0] - x1) * k, (b[1] - y1) * k, (b[2] - x1) * k, (b[3] - y1) * k ]
            for (c, b) in dets:
                if c < conf_op:
                    continue
                cd.rectangle(zbox(b), outline=(255, 80, 80), width=1)
            for b in gt_boxes:
                cd.rectangle(zbox(b), outline=(0, 220, 0), width=1)
            if target_gt is not None:
                cd.rectangle(zbox(target_gt), outline=(0, 255, 255), width=2)
            if target_pred is not None:
                cd.rectangle(zbox(target_pred), outline=(255, 210, 0), width=2)
            px = max(0, W - zoom_size - 2)
            im.paste(crop, (px, 2))
            ImageDraw.Draw(im).rectangle([px - 1, 1, min(W - 1, px + zoom_size), min(H - 1, zoom_size + 2)],
                                         outline=(255, 255, 255), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(out_path), quality=88)


def collect_example_candidates(result):
    """Ranked TP / near-miss records per example kind; every record keeps its image key and both boxes."""
    tps = []
    for lab in ev.BUCKET_LABELS:
        tps.extend(result["buckets"][lab]["recs"])
    kinds = {}
    kinds["low_iou"] = sorted(tps, key=lambda r: (r["iou"], -r["dist640"] if r["dist640"] is not None else 0.0))
    kinds["center_error"] = sorted([r for r in tps if r["dist640"] is not None], key=lambda r: -r["dist640"])
    kinds["width_error"] = sorted([r for r in tps if r["w_rel"] is not None], key=lambda r: -r["w_rel"])
    kinds["height_error"] = sorted([r for r in tps if r["h_rel"] is not None], key=lambda r: -r["h_rel"])
    nb = result["near_miss"]["records"]["B"]
    kinds["near_miss"] = sorted([r for r in nb if r.get("pred_box") is not None], key=lambda r: -r["iou"])
    return kinds


def export_examples(result, samples_by_key, out_root, ckpt_name, set_key, kinds, per_kind, conf_op):
    rows = []
    set_slug = set_key.replace("/", "_")
    for kind, recs in kinds.items():
        used, rank = set(), 0
        for r in recs:
            if rank >= per_kind:
                break
            key = r["key"]
            if key in used or key not in samples_by_key:
                continue
            used.add(key)
            rank += 1
            s = samples_by_key[key]
            gts = [g["box"] for g in s["gt"]]
            dets = [(d["conf"], d["box"]) for d in s["dets"]]
            bits = ["%s | %s | %s #%d" % (ckpt_name, set_key, kind, rank)]
            if r.get("iou") is not None:
                bits.append("IoU=%.3f" % r["iou"])
            if r.get("dist640") is not None:
                bits.append("ctr640=%.2fpx norm=%.2f" % (r["dist640"], r["norm"] or 0.0))
            if r.get("w_rel") is not None:
                bits.append("w_rel=%.2f" % r["w_rel"])
            if r.get("h_rel") is not None:
                bits.append("h_rel=%.2f" % r["h_rel"])
            if r.get("eq640") is not None:
                bits.append("gt640=%.1f" % r["eq640"])
            if r.get("conf") is not None:
                bits.append("conf=%.3f" % r["conf"])
            caption = " ".join(bits)
            fname = "%02d_%s.jpg" % (rank, os.path.splitext(os.path.basename(key))[0])
            out_path = out_root / kind / set_slug / fname
            draw_example(key, gts, dets, out_path, r.get("box"), r.get("pred_box"), caption, conf_op)
            rows.append([ckpt_name, set_key, kind, rank, os.path.basename(key), key,
                         _num(r.get("iou")), _num(r.get("dist640")), _num(r.get("norm")),
                         _num(r.get("w_rel")), _num(r.get("h_rel")), _num(r.get("eq640")),
                         _num(r.get("conf")), str(out_path.relative_to(out_root.parent))])
    return rows


# ---------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------
def md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return chr(10).join(out)


def fmt(x, nd=4):
    return "n/a" if x is None else ("%%.%df" % nd) % x


def decision_rule(ra, rb, tol_flat=0.01, tol_up=0.01):
    """CASE A = recall only, CASE B = true localization gain, CASE C = both."""
    if not ra or not rb:
        return "n/a", {}
    d50 = (rb["ap_by_iou"][0.50] or 0.0) - (ra["ap_by_iou"][0.50] or 0.0)
    d75 = (rb["ap_by_iou"][0.75] or 0.0) - (ra["ap_by_iou"][0.75] or 0.0)
    d90 = (rb["ap_by_iou"][0.90] or 0.0) - (ra["ap_by_iou"][0.90] or 0.0)
    d95 = (rb["ap_by_iou"][0.95] or 0.0) - (ra["ap_by_iou"][0.95] or 0.0)
    im_a = ra["iou_dist"]["stats"]["median"]
    im_b = rb["iou_dist"]["stats"]["median"]
    diou = (im_b - im_a) if (im_a is not None and im_b is not None) else None
    ca = (ra["errors"].get("center_dist_px_640") or {}).get("median")
    cb = (rb["errors"].get("center_dist_px_640") or {}).get("median")
    dcen = (cb - ca) if (ca is not None and cb is not None) else None
    na = (ra["errors"].get("normalized_center_error") or {}).get("median")
    nb = (rb["errors"].get("normalized_center_error") or {}).get("median")
    dnorm = (nb - na) if (na is not None and nb is not None) else None
    eps = 1e-9  # float noise must not flip a case that sits exactly on the tolerance
    hi = (d75 > tol_up + eps) and (d90 > tol_up + eps) and (diou is not None and diou > tol_flat + eps)
    lo = (d75 <= tol_flat + eps) and (d90 <= tol_flat + eps) and (diou is None or diou <= tol_flat + eps)
    if hi and d50 > tol_up + eps:
        verdict = "CASE C (both recall and localization improve)"
    elif hi:
        verdict = "CASE B (real bbox localization gain)"
    elif d50 > tol_up and lo:
        verdict = "CASE A (recall-driven only)"
    else:
        verdict = "INDETERMINATE (mixed / no clear AP shift)"
    return verdict, {"dAP50": d50, "dAP75": d75, "dAP90": d90, "dAP95": d95,
                     "d_iou_median": diou, "d_center_px640_median": dcen, "d_norm_center_median": dnorm}


def write_report(path, results, ckpts, args, sets, examples_rows):
    L = []
    add = L.append
    add("# LOCALIZATION AUDIT — YOLO26 V1 羽毛球超小目标检测")
    add("")
    add("> 本文档由 `tools/audit_localization_yolo26_v1.py` 自动生成（表格与判定规则）+ 人工结论（第 0 节，需手动维护）。")
    add("> 只读审计：不训练、不改模型/配置/数据/标签/采样器/增强；只读现有 checkpoint 与标注数据；无标注集不参与 bbox 定位。")
    add("")
    add("## 0. 结论（人工分析）")
    add("")
    add("（若本目录中是工具刚重跑的结果，本节为空；人工结论见 `LOCALIZATION_AUDIT_CONCLUSION.md` 或本文件的后续人工追加。）")
    add("")
    add("## 1. 口径与数据集")
    add("")
    add("- AP：按置信度排序的标准 PR 曲线（all-point），单图 AP 再按 GT 数做加权平均（与仓库既有 val / fixed evaluation set 报告同口径）。")
    add("- `conf_op = %.2f`（TP/FP/FN/精度/召回的工作点）；AP 曲线用 `conf_floor = %.4f` 的全部检测。" % (args.conf_op, args.conf_floor))
    add("- TP 定义：IoU >= 0.50 的一对一匹配；近失匹配 0.30-0.50 记为 class B，**从不计入 TP**。")
    add("- `equiv_size_640 = sqrt(w_px*h_px) * 640/max(W,H)`（半开区间分桶），不按 1024 输入像素分桶。")
    add("- `normalized_center_error = center_error / sqrt(gt_w*gt_h)`；`center_err_px640` 是把原始像素误差折算到 640 短边输入后的等效像素。")
    add("- 术语：这些固定评估集是 `fixed evaluation set` / `development holdout`；若后续 Stage C 依据本报告调参，最终判断必须另用全新的未见 holdout。")
    add("")
    add("评估集（本文件只列出有标注、可用于 bbox 定位的集合）：")
    add("")
    add(md_table(["set", "images", "GT"], [[k, r["images"], r["gt"]] for (n, k), r in sorted(results.items())
                                              if n == sorted({x[0] for x in results})[0]]))
    add("")
    add("## 2. 检查点指纹")
    add("")
    add(md_table(["name", "epoch", "role", "sha256(前16)", "bytes", "path"],
                 [[c["name"], c.get("epoch"), c.get("role"), (c.get("sha256") or "")[:16], c.get("bytes"), c.get("path")]
                  for c in ckpts]))
    add("")
    add("## 3. AP50..AP95（标准 PR 曲线）")
    add("")
    hdr = ["checkpoint", "set", "GT", "TP@0.5", "FP@0.5", "R@0.5", "P@op"] + \
          ["AP%d" % round(t * 100) for t in AUDIT_IOU_THRESHOLDS] + ["mAP50-95"]
    rows = []
    for (name, set_key), r in sorted(results.items()):
        rows.append([name, set_key, r["gt"], r["tp50"], r["fp50"], fmt(r["recall50"]), fmt(r["precision_op"])]
                    + [fmt(r["ap_by_iou"][t]) for t in AUDIT_IOU_THRESHOLDS] + [fmt(r["mAP50-95"])])
    add(md_table(hdr, rows))
    add("")
    add("## 4. IoU 分布（IoU >= 0.5 的 TP）")
    add("")
    hdr = ["checkpoint", "set", "n_tp", "mean", "median", "std", "P10", "P25", "P50", "P75", "P90"] + \
          ["bin " + lab for lab in IOU_HIST_LABELS]
    rows = []
    for (name, set_key), r in sorted(results.items()):
        st, q = r["iou_dist"]["stats"], r["iou_dist"]["quantiles"]
        rows.append([name, set_key, r["iou_dist"]["n"], fmt(st["mean"]), fmt(st["median"]), fmt(st["std"]),
                     fmt(q.get("P10")), fmt(q.get("P25")), fmt(q.get("P50")), fmt(q.get("P75")), fmt(q.get("P90"))]
                    + [r["iou_dist"]["histogram"][lab] for lab in IOU_HIST_LABELS])
    add(md_table(hdr, rows))
    add("")
    add("## 5. 定位误差（IoU >= 0.5 的 TP）")
    add("")
    hdr = ["checkpoint", "set", "n_tp", "center_px mean", "center_px median", "center_px P90",
           "center_px640 mean", "center_px640 median", "center_px640 P90", "norm center median",
           "w_rel median", "h_rel median", "area_ratio median", "w_ratio mean", "h_ratio mean"]
    rows = []
    for (name, set_key), r in sorted(results.items()):
        e = r["errors"]
        rows.append([name, set_key, e.get("n", 0),
                     fmt((e.get("center_dist_px") or {}).get("mean"), 2),
                     fmt((e.get("center_dist_px") or {}).get("median"), 2),
                     fmt((e.get("center_dist_px") or {}).get("P90"), 2),
                     fmt((e.get("center_dist_px_640") or {}).get("mean"), 2),
                     fmt((e.get("center_dist_px_640") or {}).get("median"), 2),
                     fmt((e.get("center_dist_px_640") or {}).get("P90"), 2),
                     fmt((e.get("normalized_center_error") or {}).get("median"), 3),
                     fmt((e.get("width_rel_err") or {}).get("median"), 3),
                     fmt((e.get("height_rel_err") or {}).get("median"), 3),
                     fmt((e.get("area_ratio") or {}).get("median"), 3),
                     fmt((e.get("width_ratio_signed") or {}).get("mean"), 3),
                     fmt((e.get("height_ratio_signed") or {}).get("mean"), 3)])
    add(md_table(hdr, rows))
    add("")
    add("### 5.1 中心误差与尺寸误差分位数（px / px640 / 归一化 / 相对误差）")
    add("")
    hdr = ["checkpoint", "set", "metric", "mean", "median", "std", "P75", "P90", "P95"]
    rows = []
    for (name, set_key), r in sorted(results.items()):
        e = r["errors"]
        for key in ["center_dist_px", "center_dist_px_640", "normalized_center_error", "center_abs_dx",
                    "center_abs_dy", "width_rel_err", "height_rel_err", "area_ratio",
                    "width_ratio_signed", "height_ratio_signed", "log_width_ratio", "log_height_ratio"]:
            d = e.get(key) or {}
            rows.append([name, set_key, key] + [fmt(d.get(s)) for s in ["mean", "median", "std", "P75", "P90", "P95"]])
    add(md_table(hdr, rows))
    add("")
    add("## 6. 尺寸分桶（equiv_size_640，TP@0.5 与逐桶 AP）")
    add("")
    hdr = ["checkpoint", "set", "bucket", "GT", "TP@0.5", "Recall", "IoU mean", "IoU median",
           "center_px640 median", "norm center median", "w_rel median", "h_rel median"] + \
          ["AP%d" % round(t * 100) for t in BUCKET_AP_THRESHOLDS]
    rows = []
    for (name, set_key), r in sorted(results.items()):
        for lab in ev.BUCKET_LABELS:
            b = r["buckets"][lab]
            rows.append([name, set_key, lab, b["gt"], b["tp50"], fmt(b["recall"]), fmt(b["iou_mean"]),
                         fmt(b["iou_median"]), fmt(b["center_dist_px_640_median"], 2),
                         fmt(b["normalized_center_error_median"], 3), fmt(b["width_rel_err_median"], 3),
                         fmt(b["height_rel_err_median"], 3)]
                        + [fmt(b.get("AP%d" % round(t * 100))) for t in BUCKET_AP_THRESHOLDS])
    add(md_table(hdr, rows))
    add("")
    add("## 7. A vs B 分桶增量（B 减 A；正数表示 B 更大）")
    add("")
    na, nb = args.compare_a, args.compare_b
    hdr = ["set", "bucket", "GT_a", "GT_b", "R_a", "R_b", "dR", "IoU_med_a", "IoU_med_b", "dIoU_med",
           "ctr640_a", "ctr640_b", "dctr640", "w_rel_a", "w_rel_b", "dw_rel", "h_rel_a", "h_rel_b", "dh_rel"]
    rows = []
    for set_key in sorted({k[1] for k in results}):
        ra, rb = results.get((na, set_key)), results.get((nb, set_key))
        if not ra or not rb:
            continue
        def iou_med(r):
            return r["iou_dist"]["stats"]["median"]
        def ctr(r):
            return (r["errors"].get("center_dist_px_640") or {}).get("median")
        def wr(r):
            return (r["errors"].get("width_rel_err") or {}).get("median")
        def hr(r):
            return (r["errors"].get("height_rel_err") or {}).get("median")
        rows.append([set_key, "ALL", ra["gt"], rb["gt"], fmt(ra["recall50"]), fmt(rb["recall50"]),
                     fmt((rb["recall50"] or 0) - (ra["recall50"] or 0)), fmt(iou_med(ra)), fmt(iou_med(rb)),
                     fmt((iou_med(rb) or 0) - (iou_med(ra) or 0)), fmt(ctr(ra), 2), fmt(ctr(rb), 2),
                     fmt((ctr(rb) or 0) - (ctr(ra) or 0), 2), fmt(wr(ra), 3), fmt(wr(rb), 3),
                     fmt((wr(rb) or 0) - (wr(ra) or 0), 3), fmt(hr(ra), 3), fmt(hr(rb), 3),
                     fmt((hr(rb) or 0) - (hr(ra) or 0), 3)])
        for lab in ev.BUCKET_LABELS:
            a, b = ra["buckets"][lab], rb["buckets"][lab]
            rows.append([set_key, lab, a["gt"], b["gt"], fmt(a["recall"]), fmt(b["recall"]),
                         fmt((b["recall"] or 0) - (a["recall"] or 0)) if (a["recall"] is not None and b["recall"] is not None) else "n/a",
                         fmt(a["iou_median"]), fmt(b["iou_median"]),
                         fmt((b["iou_median"] or 0) - (a["iou_median"] or 0)) if (a["iou_median"] is not None and b["iou_median"] is not None) else "n/a",
                         fmt(a["center_dist_px_640_median"], 2), fmt(b["center_dist_px_640_median"], 2),
                         fmt((b["center_dist_px_640_median"] or 0) - (a["center_dist_px_640_median"] or 0), 2) if (a["center_dist_px_640_median"] is not None and b["center_dist_px_640_median"] is not None) else "n/a",
                         fmt(a["width_rel_err_median"], 3), fmt(b["width_rel_err_median"], 3),
                         fmt((b["width_rel_err_median"] or 0) - (a["width_rel_err_median"] or 0), 3) if (a["width_rel_err_median"] is not None and b["width_rel_err_median"] is not None) else "n/a",
                         fmt(a["height_rel_err_median"], 3), fmt(b["height_rel_err_median"], 3),
                         fmt((b["height_rel_err_median"] or 0) - (a["height_rel_err_median"] or 0), 3) if (a["height_rel_err_median"] is not None and b["height_rel_err_median"] is not None) else "n/a"])
    add(md_table(hdr, rows))
    add("")
    add("## 8. Near-miss 分类（每个 GT 的最佳一对一 IoU）")
    add("")
    add("class A = IoU >= 0.50（TP）；class B = 0.30-0.50（近失，**不计 TP**）；class C = 0.10-0.30；class D < 0.10 或无匹配。")
    add("")
    hdr = ["checkpoint", "set", "class", "count", "share", "mean IoU"]
    rows = []
    for (name, set_key), r in sorted(results.items()):
        total = r["gt"] or 0
        for cls, _lo, _hi in NEAR_MISS_BANDS:
            recs = r["near_miss"]["records"][cls]
            rows.append([name, set_key, cls, len(recs), fmt((len(recs) / total) if total else None),
                         fmt(statistics.fmean([x["iou"] for x in recs]) if recs else None, 3)])
    add(md_table(hdr, rows))
    add("")
    add("## 9. 判定规则（自动，容差 0.01）")
    add("")
    add("CASE A = AP50 上升但 AP75/AP90 与 IoU 中位数基本不动（纯粹多找到了目标）；CASE B = AP75/AP90 与 IoU 中位数一起上升、中心/尺寸误差下降（真正的定位改善）；CASE C = 两者同时改善。")
    add("")
    hdr = ["set", "verdict", "dAP50", "dAP75", "dAP90", "dAP95", "d_IoU_median", "d_center_px640_med", "d_norm_center_med"]
    rows = []
    for set_key in sorted({k[1] for k in results}):
        ra, rb = results.get((na, set_key)), results.get((nb, set_key))
        v, d = decision_rule(ra, rb)
        rows.append([set_key, v] + [fmt(d.get(k)) for k in ["dAP50", "dAP75", "dAP90", "dAP95",
                                                            "d_iou_median", "d_center_px640_median",
                                                            "d_norm_center_median"]])
    add(md_table(hdr, rows))
    add("")
    add("## 10. 复现命令与产物")
    add("")
    add("```")
    add(args.command_line)
    add("```")
    add("")
    add("- 检查点与标注数据均为只读输入；本审计不产生任何训练/权重/标签变更。")
    add("- 输出 CSV：localization_ap_by_iou.csv / localization_iou_distribution.csv / localization_error_summary.csv /")
    add("  localization_size_buckets.csv / localization_checkpoint_compare.csv / localization_near_miss.csv；")
    add("  另有 localization_audit.json（含每个 GT 的匹配记录）与 localization_examples/（示例图 + index.csv）。")
    add("- 预测缓存位于 `_scratch_localization_audit/`（临时，可删；删除后用同一命令可完整重生成）。")
    if examples_rows:
        add("- 示例图共 %d 张（low_iou / center_error / width_error / height_error / near_miss，各 %d 张/集合）。"
            % (len(examples_rows), args.examples_per_kind))
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(L) + chr(10), encoding="utf-8")


# ---------------------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Localization audit for the YOLO26 V1 shuttle detector (read-only).")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--sets", default=",".join(DEFAULT_SETS),
                    help="comma list; val and/or labeled frozen sets (unlabeled sets are refused)")
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH, repeatable")
    ap.add_argument("--ckpt-registry", default="configs/shuttle_detection/checkpoints_v1.yaml")
    ap.add_argument("--ckpt-name", action="append", default=[],
                    help="registry names (default: a_best b_best b_last)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf-op", type=float, default=0.25, help="operating confidence for TP/FP/FN")
    ap.add_argument("--conf-floor", type=float, default=0.001, help="confidence floor stored for AP curves")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--out-dir", default="outputs/shuttle_capability/metrics")
    ap.add_argument("--cache-dir", default="_scratch_localization_audit")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--examples-dir", default="outputs/shuttle_capability/localization_examples")
    ap.add_argument("--examples-ckpt", action="append", default=[], help="checkpoints to draw examples for")
    ap.add_argument("--examples-per-kind", type=int, default=EXAMPLES_PER_KIND)
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--report", default="outputs/shuttle_capability/reports/LOCALIZATION_AUDIT.md")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--compare-a", default="a_best")
    ap.add_argument("--compare-b", default="b_best")
    args = ap.parse_args()
    args.command_line = "python tools/audit_localization_yolo26_v1.py " + " ".join(sys.argv[1:])

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = repo / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    remaps = []
    for m in args.map:
        if "=" in m:
            a, b = m.split("=", 1)
            remaps.append((a, b))
    names = list(args.ckpt_name) or ([] if args.ckpt else ["a_best", "b_best", "b_last"])
    ckpts = ev.resolve_checkpoints(args.ckpt, args.ckpt_registry, names)
    for c in ckpts:
        if c.get("path") and os.path.exists(c["path"]):
            c["sha256"] = ev.file_sha256(c["path"])
            c["bytes"] = os.path.getsize(c["path"])
    sets = [s for s in args.sets.split(",") if s]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo / ".yolo_cfg"))

    print("checkpoints: %s" % ", ".join("%s(e%s)" % (c["name"], c.get("epoch")) for c in ckpts), flush=True)
    print("sets: %s" % ", ".join(sets), flush=True)
    results, samples_by = {}, {}
    for set_key in sets:
        if set_key == "val":
            samples, skipped = build_val_samples(repo, args.manifest, remaps, args.max_images)
        else:
            samples, skipped = build_frozen_samples(repo, set_key, args.max_images)
        n_gt = sum(len(s["gt"]) for s in samples)
        print("=== set %s: %d images, %d GT, %d skipped ===" % (set_key, len(samples), n_gt, skipped), flush=True)
        if not samples:
            continue
        for ckpt in ckpts:
            path = ckpt.get("path")
            if not path or not os.path.exists(path):
                print("SKIP %s: checkpoint missing (%s)" % (ckpt["name"], path), flush=True)
                continue
            attach_detections(repo, ckpt, set_key, samples, args, cache_dir)
            r = evaluate_localization(samples, conf_op=args.conf_op)
            results[(ckpt["name"], set_key)] = r
            samples_by[(ckpt["name"], set_key)] = {s["key"]: s for s in samples}
            print("  %-22s %-28s GT=%-5d TP@0.5=%-5d R=%.4f P=%.4f AP50=%.4f AP75=%.4f AP90=%.4f AP95=%.4f "
                  "mAP50-95=%.4f IoU_med=%.4f ctr640_med=%.3f" % (
                      ckpt["name"], set_key, r["gt"], r["tp50"], r["recall50"] or 0.0, r["precision_op"] or 0.0,
                      r["ap_by_iou"][0.50] or 0.0, r["ap_by_iou"][0.75] or 0.0, r["ap_by_iou"][0.90] or 0.0,
                      r["ap_by_iou"][0.95] or 0.0, r["mAP50-95"] or 0.0,
                      r["iou_dist"]["stats"]["median"] or 0.0,
                      (r["errors"].get("center_dist_px_640") or {}).get("median") or 0.0), flush=True)
    if not results:
        raise SystemExit("no results produced (check --sets / --ckpt / data availability)")

    m = out_dir
    write_ap_by_iou(m / "localization_ap_by_iou.csv", results)
    write_iou_distribution(m / "localization_iou_distribution.csv", results)
    write_error_summary(m / "localization_error_summary.csv", results)
    write_size_buckets(m / "localization_size_buckets.csv", results)
    write_checkpoint_compare(m / "localization_checkpoint_compare.csv", results, args.compare_a, args.compare_b)
    write_near_miss(m / "localization_near_miss.csv", results)
    (m / "localization_audit.json").write_text(json.dumps({
        "config": {"imgsz": args.imgsz, "conf_op": args.conf_op, "conf_floor": args.conf_floor,
                   "iou_thresholds": AUDIT_IOU_THRESHOLDS, "bucket_ap_thresholds": BUCKET_AP_THRESHOLDS,
                   "tp_iou_min": TP_IOU_MIN, "command": args.command_line},
        "checkpoints": [{k: v for k, v in c.items()} for c in ckpts],
        "results": {"%s|%s" % k: v for k, v in results.items()}}, indent=1, default=str), encoding="utf-8")
    print("wrote 6 csv + json -> %s" % m, flush=True)

    examples_rows = []
    if not args.no_examples:
        ex_root = repo / args.examples_dir
        if args.examples_ckpt:
            ex_ckpts = list(args.examples_ckpt)
        elif any(k[0] == args.compare_b for k in results):
            ex_ckpts = [args.compare_b]
        else:
            ex_ckpts = sorted({k[0] for k in results})[:1]
            print("examples: %s not evaluated, falling back to %s" % (args.compare_b, ex_ckpts[0]), flush=True)
        for ck_name in ex_ckpts:
            if not any(k[0] == ck_name for k in results):
                print("examples: no results for %s, skipped" % ck_name, flush=True)
                continue
            for set_key in sets:
                res = results.get((ck_name, set_key))
                if res is None:
                    continue
                kinds = collect_example_candidates(res)
                examples_rows.extend(export_examples(res, samples_by[(ck_name, set_key)], ex_root, ck_name,
                                                     set_key, kinds, args.examples_per_kind, args.conf_op))
        idx = ex_root / "index.csv"
        if examples_rows:
            with open(idx, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["checkpoint", "set", "kind", "rank", "image", "image_path", "iou", "center_err_px640",
                            "norm_center_err", "width_rel_err", "height_rel_err", "gt_size_px640", "conf", "file"])
                w.writerows(examples_rows)
            print("examples: %d images -> %s" % (len(examples_rows), ex_root), flush=True)

    if not args.no_report:
        write_report(repo / args.report, results, ckpts, args, sets, examples_rows)
        print("report -> %s" % (repo / args.report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())