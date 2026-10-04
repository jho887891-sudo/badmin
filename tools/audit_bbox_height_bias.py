#!/usr/bin/env python3
"""Root-cause audit for the systematic bbox HEIGHT bias (+9.2%) of the YOLO26 V1 shuttle detector.

DIAGNOSTIC ONLY. Reads existing labels, images, evaluation code and the cached predictions of
`tools/audit_localization_yolo26_v1.py` (no training, no checkpoint/label/config/loss/augmentation change).

It answers, with numbers, in this order:
  1. coordinate / letterbox / scale-back maths: exact formulas + round-trip tests + parity with the
     installed ultralytics 8.4.150 (LetterBox, scale_boxes);
  2. is pred_h/gt_h ~ 1.092 a real property of the predictions, or an evaluation artefact;
  3. is it a proportional bias or a 1-pixel quantisation on tiny objects (size buckets, pixel histogram);
  4. which data sources / label conventions carry it (val manifest source + location);
  5. what it costs at high IoU: per-TP counterfactual IoU and AP-level counterfactual AP50..AP95.

Outputs:
  outputs/shuttle_capability/metrics/bbox_coordinate_roundtrip.json
  outputs/shuttle_capability/metrics/bbox_height_bias_by_source.csv
  outputs/shuttle_capability/metrics/bbox_height_bias_by_size.csv
  outputs/shuttle_capability/metrics/bbox_pixel_error_distribution.csv
  outputs/shuttle_capability/metrics/bbox_ultralytics_parity.csv        (only with --parity)
  outputs/shuttle_capability/metrics/bbox_counterfactual_iou.csv
  outputs/shuttle_capability/bbox_bias_examples/{high_height_ratio,typical_109,ratio_near_1}/
  outputs/shuttle_capability/reports/BBOX_HEIGHT_BIAS_ROOT_CAUSE.md

Run:  python tools/audit_bbox_height_bias.py            (CPU only, uses cached predictions)
      python tools/audit_bbox_height_bias.py --parity   (adds the ultralytics parity run, needs GPU)
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
import audit_localization_yolo26_v1 as aud  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402

IMGSZ = 1024
EQUIV_TARGET = 640
CONF_OP = 0.25
TIERS = [("TP", 0.50, 2.0), ("near_0.30_0.50", 0.30, 0.50), ("poor_0.10_0.30", 0.10, 0.30)]
PCT = (0.10, 0.25, 0.50, 0.75, 0.90)
PY_HIST_BINS = [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4]
REL_HIST_BINS = [-0.5, -0.25, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 0.25, 0.50]
CF_THRESHOLDS = [0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
CF_CROSS = [0.75, 0.85, 0.90, 0.95]
EXAMPLES_PER_GROUP = 50


# =======================================================================================
# 1. coordinate maths (formulas transcribed from the actual code, see the report)
# =======================================================================================
def yolo_norm_to_pixel(cx, cy, bw, bh, W, H):
    """t""" + "ools/eval_yolo26_v1.py:401-402 -- what the evaluator does with a normalized label."
    return [(cx - bw / 2.0) * W, (cy - bh / 2.0) * H, (cx + bw / 2.0) * W, (cy + bh / 2.0) * H]


def pixel_to_yolo_norm(box, W, H):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H, (x2 - x1) / W, (y2 - y1) / H)


def letterbox_params(W, H, imgsz=IMGSZ, rect=False, stride=32):
    """ultralytics/data/augment.py:1745-1776 (LetterBox.get_params), auto=False unless rect."""
    r = min(imgsz / H, imgsz / W)
    new_unpad = (round(W * r), round(H * r))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]
    if rect:
        dw, dh = dw % stride, dh % stride
    dw2, dh2 = dw / 2.0, dh / 2.0
    top, bottom = round(dh2 - 0.1), round(dh2 + 0.1)
    left, right = round(dw2 - 0.1), round(dw2 + 0.1)
    return {"r": r, "new_unpad": new_unpad, "top": top, "bottom": bottom, "left": left, "right": right,
            "ratio": (r, r), "new_shape": (imgsz, imgsz), "orig_shape": (H, W)}


def letterbox_forward_box(box, params, content_scale=False):
    """original xyxy -> letterboxed xyxy. content_scale=True uses the ACTUAL resized pixel scale
    (round(W*r)/W, round(H*r)/H) instead of the nominal r, which is what the image content really does."""
    H, W = params["orig_shape"]
    if content_scale:
        sx = params["new_unpad"][0] / float(W)
        sy = params["new_unpad"][1] / float(H)
    else:
        sx, sy = params["ratio"]
    x1, y1, x2, y2 = box
    return [x1 * sx + params["left"], y1 * sy + params["top"], x2 * sx + params["left"], y2 * sy + params["top"]]


def scale_boxes_inverse(box1024, params, W, H):
    """ultralytics/utils/ops.py:146-165 with ratio_pad=None, as called by detect/predict.py:121."""
    img1_h, img1_w = params["new_shape"]
    gain = min(img1_h / H, img1_w / W)
    gain_y = gain_x = gain
    pad_x = round((img1_w - round(W * gain)) / 2 - 0.1)
    pad_y = round((img1_h - round(H * gain)) / 2 - 0.1)
    x1, y1, x2, y2 = box1024
    return [(x1 - pad_x) / gain_x, (y1 - pad_y) / gain_y, (x2 - pad_x) / gain_x, (y2 - pad_y) / gain_y]


def box_metrics(box):
    return {"w": box[2] - box[0], "h": box[3] - box[1],
            "cx": (box[0] + box[2]) / 2.0, "cy": (box[1] + box[3]) / 2.0}


def roundtrip_cases():
    """(name, W, H, box, kind) covering the required resolutions and box shapes."""
    cases = []
    resolutions = [(1920, 1200), (1920, 1080), (960, 960), (1280, 720), (2048, 1536), (640, 480), (2048, 1000),
                   (1920, 1000), (1280, 730)]  # last two exercise the round(new_unpad) residual
    for (W, H) in resolutions:
        cases.append(("centre_%dx%d" % (W, H), W, H, [W / 2 - 15, H / 2 - 10, W / 2 + 15, H / 2 + 10], "centre"))
        cases.append(("topleft_%dx%d" % (W, H), W, H, [2.0, 2.0, 32.0, 22.0], "topleft"))
        cases.append(("botright_%dx%d" % (W, H), W, H, [W - 32.0, H - 22.0, W - 2.0, H - 2.0], "botright"))
        for (w, h) in [(4, 6), (6, 8), (8, 12), (4, 12), (12, 4)]:
            cases.append(("tiny_%dx%d_%dx%d" % (w, h, W, H), W, H,
                          [W / 2 - w / 2.0, H / 2 - h / 2.0, W / 2 + w / 2.0, H / 2 + h / 2.0], "tiny"))
    return cases


def run_roundtrip_tests():
    results = []
    for (name, W, H, box, kind) in roundtrip_cases():
        # test 1: normalized -> pixel -> normalized
        cx, cy, bw, bh = pixel_to_yolo_norm(box, W, H)
        back = yolo_norm_to_pixel(cx, cy, bw, bh, W, H)
        t1 = max(abs(a - b) for a, b in zip(box, back))
        # test 2: original -> letterbox (nominal r) -> inverse -> original
        p = letterbox_params(W, H)
        f = letterbox_forward_box(box, p)
        inv = scale_boxes_inverse(f, p, W, H)
        t2 = max(abs(a - b) for a, b in zip(box, inv))
        # realistic path: image content is resized to new_unpad, then the model output is scaled back
        fc = letterbox_forward_box(box, p, content_scale=True)
        invc = scale_boxes_inverse(fc, p, W, H)
        d = [a - b for a, b in zip(invc, box)]
        t3 = max(abs(x) for x in d)
        m_in, m_out = box_metrics(box), box_metrics(invc)
        results.append({
            "case": name, "kind": kind, "W": W, "H": H,
            "box_in": box, "box_out_nominal": inv, "box_out_content": invc,
            "max_abs_err_norm_roundtrip": t1, "max_abs_err_letterbox_roundtrip": t2,
            "max_abs_err_content_roundtrip": t3,
            "h_ratio_nominal": m_out["h"] / m_in["h"] if m_in["h"] else None,
            "h_ratio_content": (box_metrics(invc)["h"] / m_in["h"]) if m_in["h"] else None,
            "w_ratio_content": (box_metrics(invc)["w"] / m_in["w"]) if m_in["w"] else None,
            "r": p["r"], "new_unpad": p["new_unpad"], "pad": [p["left"], p["top"]],
        })
    return results


def ultralytics_cross_check():
    """Run the INSTALLED ultralytics LetterBox.get_params + ops.scale_boxes on the same cases and compare."""
    out = {"available": False, "rows": [], "max_abs_diff_letterbox_params": None, "max_abs_diff_inverse": None}
    try:
        import numpy as np
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import ops as uops
    except Exception as exc:  # pragma: no cover
        out["error"] = repr(exc)
        return out
    out["available"] = True
    worst_par, worst_inv, worst_shape = 0.0, 0.0, 0.0
    for (name, W, H, box, _kind) in roundtrip_cases():
        img = np.zeros((H, W, 3), dtype=np.uint8)
        lb = LetterBox((IMGSZ, IMGSZ), auto=False, stride=32)
        up = lb.get_params({"img": img})          # ultralytics' own letterbox parameters
        mine = letterbox_params(W, H)
        d_par = 0.0
        for k in ("top", "bottom", "left", "right"):
            d_par = max(d_par, abs(float(up[k]) - float(mine[k])))
        d_par = max(d_par, abs(up["new_unpad"][0] - mine["new_unpad"][0]))
        d_par = max(d_par, abs(up["new_unpad"][1] - mine["new_unpad"][1]))
        d_par = max(d_par, abs(float(up["ratio"][0]) - float(mine["r"])), abs(float(up["ratio"][1]) - float(mine["r"])))
        padded = lb(image=img).shape[:2]
        d_shape = abs(padded[0] - IMGSZ) + abs(padded[1] - IMGSZ)
        worst_par = max(worst_par, d_par)
        worst_shape = max(worst_shape, float(d_shape))
        param = {"new_shape": (padded[0], padded[1])}
        boxes = np.array([letterbox_forward_box(box, mine)], dtype=np.float32)
        ref = uops.scale_boxes((padded[0], padded[1]), boxes.copy(), (H, W), ratio_pad=None)
        mine_inv = np.array([scale_boxes_inverse(list(boxes[0]), param, W, H)], dtype=np.float32)
        d = float(np.abs(ref - mine_inv).max())
        worst_inv = max(worst_inv, d)
        out["rows"].append({"case": name, "W": W, "H": H, "letterboxed_shape": [int(padded[0]), int(padded[1])],
                            "ultralytics_new_unpad": [int(up["new_unpad"][0]), int(up["new_unpad"][1])],
                            "my_new_unpad": list(mine["new_unpad"]),
                            "ultralytics_pad": [up["left"], up["top"]], "my_pad": [mine["left"], mine["top"]],
                            "ultralytics_ratio": [float(up["ratio"][0]), float(up["ratio"][1])], "my_r": mine["r"],
                            "max_abs_diff_params": d_par, "max_abs_diff_inverse": d,
                            "ultralytics_box": ref[0].tolist(), "my_box": mine_inv[0].tolist()})
    out["max_abs_diff_letterbox_params"] = worst_par
    out["max_abs_diff_padded_shape"] = worst_shape
    out["max_abs_diff_inverse"] = worst_inv
    return out
# =======================================================================================
# 2. data assembly (labels + cached predictions; nothing is re-trained or modified)
# =======================================================================================
def _key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def load_val_meta(repo, manifest_rel):
    """{image_key: {source, location, difficulty, n_boxes, bucket, W, H}} for split == val."""
    meta = {}
    with open(repo / manifest_rel, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("split") != "val":
                continue
            meta[_key(r["image"])] = {"source": r.get("source") or "?", "location": r.get("location") or "?",
                                      "difficulty": r.get("difficulty") or "?", "n_boxes": r.get("n_boxes"),
                                      "bucket": r.get("bucket"), "W": r.get("width"), "H": r.get("height")}
    return meta


def load_cached_dets(repo, ckpt, set_key, cache_dir_rel):
    p = repo / cache_dir_rel / ("loc_%s_%s.json" % (ckpt, set_key.replace("/", "_")))
    if not p.exists():
        return None, p
    blob = json.loads(p.read_text(encoding="utf-8"))
    return {_key(k): v for k, v in (blob.get("dets") or {}).items()}, p


def build_records(samples, dets_by_img, meta, conf_op=CONF_OP):
    """One record per GT/detection pair at IoU>=0.10 (best-IoU one-to-one), plus the GT inventory."""
    records, gt_rows, unmatched = [], [], 0
    for s in samples:
        W, H = float(s["img_w"]), float(s["img_h"])
        mi = meta.get(_key(s["image"]), {})
        scale1024 = IMGSZ / max(W, H)
        gboxes = [g["box"] for g in s["gt"]]
        dets_op = [(d["conf"], d["box"]) for d in (s.get("dets") or []) if d["conf"] >= conf_op]
        for g in s["gt"]:
            gw, gh = g["box"][2] - g["box"][0], g["box"][3] - g["box"][1]
            gt_rows.append({"key": s["key"], "source": mi.get("source"), "location": mi.get("location"),
                            "bucket": g["bucket"], "gt_w": gw, "gt_h": gh, "eq640": g["eq640"],
                            "aspect": (gw / gh) if gh > 0 else None,
                            "gt_w_px1024": gw * scale1024, "gt_h_px1024": gh * scale1024})
        if not gboxes:
            continue
        m = ev.match_greedy(gboxes, dets_op, 0.10, metric="iou")
        paired = {gj: di for di, gj in m["matches"]}
        for gj, g in enumerate(gboxes):
            di = paired.get(gj)
            if di is None:
                unmatched += 1
                continue
            pred = dets_op[di][1]
            iou = ev.iou_xyxy(g, pred)
            tier = "TP" if iou >= 0.50 else ("near_0.30_0.50" if iou >= 0.30 else "poor_0.10_0.30")
            gw, gh = g[2] - g[0], g[3] - g[1]
            pw, ph = pred[2] - pred[0], pred[3] - pred[1]
            rec = {"key": s["key"], "image": s["image"], "source": mi.get("source"),
                   "location": mi.get("location"), "difficulty": mi.get("difficulty"),
                   "bucket": (s["gt"][gj].get("bucket") if isinstance(s["gt"][gj], dict) else None)
                   or ev.bucket_of(ev.equiv_size_640(gw, gh, W, H)),
                   "tier": tier, "iou": iou, "conf": dets_op[di][0], "gt": g, "pred": pred,
                   "gt_w": gw, "gt_h": gh, "pred_w": pw, "pred_h": ph,
                   "w_ratio": pw / gw if gw > 0 else None, "h_ratio": ph / gh if gh > 0 else None,
                   "w_err_rel": (pw - gw) / gw if gw > 0 else None,
                   "h_err_rel": (ph - gh) / gh if gh > 0 else None,
                   "dw_orig": pw - gw, "dh_orig": ph - gh,
                   "dw_px1024": (pw - gw) * scale1024, "dh_px1024": (ph - gh) * scale1024,
                   "gt_w_px1024": gw * scale1024, "gt_h_px1024": gh * scale1024,
                   "pred_w_px1024": pw * scale1024, "pred_h_px1024": ph * scale1024,
                   "center_err_px": ev.center_dist(g, pred),
                   "center_err_px640": ev.center_dist(g, pred) * (EQUIV_TARGET / max(W, H)),
                   "eq640": ev.equiv_size_640(gw, gh, W, H), "W": W, "H": H}
            records.append(rec)
    return records, gt_rows, unmatched


# =======================================================================================
# 3. statistics helpers
# =======================================================================================
def pct(values, q):
    vals = [v for v in values if v is not None]
    return ev.percentile(vals, q) if vals else None


def dist(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    out = {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals),
           "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
    for q in PCT:
        out["P%d" % round(q * 100)] = ev.percentile(vals, q)
    return out


def histogram(values, bins):
    vals = [v for v in values if v is not None]
    edges = list(bins)
    labels = []
    for i in range(len(edges) + 1):
        lo = edges[i - 1] if i > 0 else None
        hi = edges[i] if i < len(edges) else None
        labels.append((lo, hi))
    counts = [0] * len(labels)
    for v in vals:
        placed = False
        for i, (lo, hi) in enumerate(labels):
            if (lo is None or v >= lo) and (hi is None or v < hi):
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return [{"lo": lo, "hi": hi, "count": c} for (lo, hi), c in zip(labels, counts)], len(vals)


def mode_of_rounded(values, nd=0):
    vals = [round(v, nd) for v in values if v is not None]
    if not vals:
        return None, 0
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0], best[1]


def ratio_summary(records, field, tier="TP"):
    vals = [r[field] for r in records if r["tier"] == tier]
    return dist(vals)


def group_records(records, key_fn, tier="TP"):
    out = {}
    for r in records:
        if r["tier"] != tier:
            continue
        out.setdefault(key_fn(r), []).append(r)
    return out


def group_gt(gt_rows, key_fn):
    out = {}
    for g in gt_rows:
        out.setdefault(key_fn(g), []).append(g)
    return out


def bucket_rows(records, gt_rows):
    """Per equiv_size_640 bucket: GT/TP counts, ratios, errors, IoU, centre error, pixel errors."""
    rows = []
    gt_by_b = group_gt(gt_rows, lambda g: g["bucket"])
    rec_by_b = group_records(records, lambda r: r["bucket"])
    for b in ev.BUCKET_LABELS:
        recs = rec_by_b.get(b, [])
        gts = gt_by_b.get(b, [])
        wr, hr = dist([r["w_ratio"] for r in recs]), dist([r["h_ratio"] for r in recs])
        rows.append({
            "bucket": b, "GT": len(gts), "TP": len(recs),
            "gt_w_median": pct([g["gt_w"] for g in gts], 0.5), "gt_h_median": pct([g["gt_h"] for g in gts], 0.5),
            "gt_aspect_median": pct([g["aspect"] for g in gts], 0.5),
            "gt_h_px1024_median": pct([g["gt_h_px1024"] for g in gts], 0.5),
            "w_ratio_mean": wr.get("mean"), "w_ratio_median": wr.get("median"),
            "h_ratio_mean": hr.get("mean"), "h_ratio_median": hr.get("median"),
            "w_err_rel_mean": (dist([r["w_err_rel"] for r in recs]) or {}).get("mean"),
            "h_err_rel_mean": (dist([r["h_err_rel"] for r in recs]) or {}).get("mean"),
            "w_err_rel_median": (dist([r["w_err_rel"] for r in recs]) or {}).get("median"),
            "h_err_rel_median": (dist([r["h_err_rel"] for r in recs]) or {}).get("median"),
            "w_err_rel_P10": (dist([r["w_err_rel"] for r in recs]) or {}).get("P10"),
            "h_err_rel_P10": (dist([r["h_err_rel"] for r in recs]) or {}).get("P10"),
            "h_err_rel_P90": (dist([r["h_err_rel"] for r in recs]) or {}).get("P90"),
            "dw_px1024_median": (dist([r["dw_px1024"] for r in recs]) or {}).get("median"),
            "dh_px1024_median": (dist([r["dh_px1024"] for r in recs]) or {}).get("median"),
            "dh_px1024_mean": (dist([r["dh_px1024"] for r in recs]) or {}).get("mean"),
            "dh_px1024_mode": mode_of_rounded([r["dh_px1024"] for r in recs], 0)[0],
            "share_dh_px1024_within_0.5": None, "share_dh_px1024_plus1": None,
            "iou_median": pct([r["iou"] for r in recs], 0.5),
            "center_err_px640_median": pct([r["center_err_px640"] for r in recs], 0.5),
        })
        vals = [r["dh_px1024"] for r in recs if r["dh_px1024"] is not None]
        if vals:
            rows[-1]["share_dh_px1024_within_0.5"] = sum(1 for v in vals if abs(v) <= 0.5) / len(vals)
            rows[-1]["share_dh_px1024_plus1"] = sum(1 for v in vals if 0.5 < v <= 1.5) / len(vals)
    return rows


def source_rows(records, gt_rows):
    rows = []
    for scope, key_fn in [("source", lambda d: d.get("source") or "?"), ("location", lambda d: d.get("location") or "?")]:
        gt_by = group_gt(gt_rows, key_fn)
        rec_by = group_records(records, key_fn)
        for name in sorted(set(list(gt_by.keys()) + list(rec_by.keys())), key=lambda x: str(x)):
            gts, recs = gt_by.get(name, []), rec_by.get(name, [])
            wr, hr = dist([r["w_ratio"] for r in recs]), dist([r["h_ratio"] for r in recs])
            hrel = dist([r["h_err_rel"] for r in recs])
            rows.append({"scope": scope, "group": name, "GT": len(gts), "TP": len(recs),
                         "images": len({r["key"] for r in recs}),
                         "gt_w_median": pct([g["gt_w"] for g in gts], 0.5),
                         "gt_h_median": pct([g["gt_h"] for g in gts], 0.5),
                         "gt_aspect_median": pct([g["aspect"] for g in gts], 0.5),
                         "gt_eq640_median": pct([g["eq640"] for g in gts], 0.5),
                         "w_ratio_mean": wr.get("mean"), "w_ratio_median": wr.get("median"),
                         "h_ratio_mean": hr.get("mean"), "h_ratio_median": hr.get("median"),
                         "h_err_rel_mean": hrel.get("mean"), "h_err_rel_median": hrel.get("median"),
                         "h_err_rel_P10": hrel.get("P10"), "h_err_rel_P90": hrel.get("P90"),
                         "dh_px1024_median": (dist([r["dh_px1024"] for r in recs]) or {}).get("median"),
                         "iou_median": pct([r["iou"] for r in recs], 0.5),
                         "center_err_px640_median": pct([r["center_err_px640"] for r in recs], 0.5)})
    return rows


def ratio_vs_iou_rows(records):
    bins = [(0.10, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 1.01)]
    rows = []
    for lo, hi in bins:
        recs = [r for r in records if lo <= r["iou"] < hi]
        if not recs:
            continue
        wr, hr = dist([r["w_ratio"] for r in recs]), dist([r["h_ratio"] for r in recs])
        rows.append({"iou_lo": lo, "iou_hi": hi, "n": len(recs),
                     "w_ratio_mean": wr.get("mean"), "w_ratio_median": wr.get("median"),
                     "h_ratio_mean": hr.get("mean"), "h_ratio_median": hr.get("median"),
                     "h_err_rel_median": (dist([r["h_err_rel"] for r in recs]) or {}).get("median"),
                     "dh_px1024_median": (dist([r["dh_px1024"] for r in recs]) or {}).get("median"),
                     "gt_h_px1024_median": (dist([r["gt_h_px1024"] for r in recs]) or {}).get("median")})
    return rows


# =======================================================================================
# 4. counterfactual IoU / AP (diagnostic only -- never used as a metric or for deployment)
# =======================================================================================
def cf_boxes(gt, pred):
    """Same prediction, with one factor replaced by the GT value. Association is fixed beforehand."""
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    pw, ph = pred[2] - pred[0], pred[3] - pred[1]
    cxp, cyp = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    cxg, cyg = (gt[0] + gt[2]) / 2.0, (gt[1] + gt[3]) / 2.0
    return {
        "orig": list(pred),
        "fix_width": [gt[0], cyp - ph / 2.0, gt[2], cyp + ph / 2.0],
        "fix_height": [cxp - pw / 2.0, gt[1], cxp + pw / 2.0, gt[3]],
        "fix_width_size_only": [cxp - gw / 2.0, cyp - ph / 2.0, cxp + gw / 2.0, cyp + ph / 2.0],
        "fix_height_size_only": [cxp - pw / 2.0, cyp - gh / 2.0, cxp + pw / 2.0, cyp + gh / 2.0],
        "fix_width_height": [cxp - gw / 2.0, cyp - gh / 2.0, cxp + gw / 2.0, cyp + gh / 2.0],
        "fix_center": [cxg - pw / 2.0, cyg - ph / 2.0, cxg + pw / 2.0, cyg + ph / 2.0],
        "fix_center_size": list(gt),
    }


CF_VARIANTS = ["orig", "fix_center", "fix_width_size_only", "fix_height_size_only", "fix_width", "fix_height",
               "fix_width_height", "fix_center_size"]


def counterfactual_tp_rows(records):
    """Per-TP: actual IoU vs IoU with one factor corrected (conditional view)."""
    rows = []
    for r in records:
        if r["tier"] != "TP":
            continue
        cf = cf_boxes(r["gt"], r["pred"])
        ious = {k: ev.iou_xyxy(r["gt"], v) for k, v in cf.items()}
        row = {"key": r["key"], "source": r["source"], "bucket": r["bucket"], "conf": r["conf"],
               "gt_w": r["gt_w"], "gt_h": r["gt_h"], "pred_w": r["pred_w"], "pred_h": r["pred_h"],
               "w_ratio": r["w_ratio"], "h_ratio": r["h_ratio"], "dh_px1024": r["dh_px1024"],
               "dw_px1024": r["dw_px1024"], "center_err_px640": r["center_err_px640"]}
        for k, v in ious.items():
            row["iou_" + k] = v
        for k in CF_VARIANTS[1:]:
            row["diou_" + k] = ious[k] - ious["orig"]
        rows.append(row)
    return rows


def counterfactual_crossings(tp_rows):
    """How many TPs cross each IoU threshold once one factor is corrected."""
    out = []
    for thr in CF_CROSS:
        for k in CF_VARIANTS:
            below = [r for r in tp_rows if r["iou_orig"] < thr and r["iou_" + k] >= thr]
            above = [r for r in tp_rows if r["iou_orig"] >= thr and r["iou_" + k] < thr]
            out.append({"threshold": thr, "variant": k, "n_tp": len(tp_rows),
                        "crossed_up": len(below), "crossed_down": len(above),
                        "share_up": (len(below) / len(tp_rows)) if tp_rows else None})
    return out


def counterfactual_ap(samples, dets_by_img, ckpt, thresholds=None):
    """AP after replacing one factor of every associated detection by the GT value.

    Association uses the ORIGINAL boxes (greedy, IoU>=0.10) and is then held fixed, so the only thing
    that changes across variants is the geometry. AP convention = repository one (per image, GT-weighted).
    """
    thresholds = list(thresholds or CF_THRESHOLDS)
    acc = {v: {t: [] for t in thresholds} for v in CF_VARIANTS}
    tp50 = {v: 0 for v in CF_VARIANTS}
    n_gt = 0
    for s in samples:
        gboxes = [g["box"] for g in s["gt"]]
        dets = [(d["conf"], d["box"]) for d in s["dets"]]
        if not gboxes:
            continue
        n_gt += len(gboxes)
        assoc = {di: gj for di, gj in ev.match_greedy(gboxes, dets, 0.10, metric="iou")["matches"]}
        variants = {v: [] for v in CF_VARIANTS}
        for di, (c, b) in enumerate(dets):
            gj = assoc.get(di)
            if gj is None:
                for v in CF_VARIANTS:
                    variants[v].append((c, list(b)))
            else:
                cf = cf_boxes(gboxes[gj], b)
                for v in CF_VARIANTS:
                    variants[v].append((c, cf[v]))
        for v in CF_VARIANTS:
            variants[v].sort(key=lambda t: -t[0])
            for t in thresholds:
                a = ev.ap_for_gt_set(gboxes, variants[v], t)
                if a is not None:
                    acc[v][t].append((a, len(gboxes)))
            m = ev.match_greedy(gboxes, variants[v], 0.50, metric="iou")
            tp50[v] += sum(1 for x in m["gt_hit"] if x)
    out = {}
    for v in CF_VARIANTS:
        row = {"checkpoint": ckpt, "variant": v, "GT": n_gt, "TP50": tp50[v],
               "recall50": (tp50[v] / n_gt) if n_gt else None}
        aps = []
        for t in thresholds:
            items = [a for a in acc[v][t] if a[0] is not None]
            n = sum(a[1] for a in items)
            val = (sum((a[0] or 0.0) * a[1] for a in items) / n) if n else None
            row["AP%d" % round(t * 100)] = val
            aps.append(val)
        good = [a for a in aps if a is not None]
        row["mAP50-95"] = statistics.fmean(good) if good else None
        out[v] = row
    base = out["orig"]
    for v in CF_VARIANTS:
        for t in thresholds:
            k = "AP%d" % round(t * 100)
            a, b = base.get(k), out[v].get(k)
            out[v]["d" + k] = (b - a) if (a is not None and b is not None) else None
        a, b = base.get("mAP50-95"), out[v].get("mAP50-95")
        out[v]["dmAP50-95"] = (b - a) if (a is not None and b is not None) else None
    return out


# =======================================================================================
# 5. appearance proxies (blur, blob extent) -- proxies, reported as such
# =======================================================================================
def sharpness_proxy(records, tier="TP", max_items=1500):
    """Laplacian variance of the GT crop (higher = sharper). NOT a blur label, only a proxy."""
    import numpy as np
    import cv2
    from PIL import Image
    items = [r for r in records if r["tier"] == tier]
    items = items[:: max(1, len(items) // max_items)][:max_items]
    vals = []
    for r in items:
        try:
            with Image.open(r["image"]) as im:
                im = im.convert("L")
                g = [float(v) for v in r["gt"]]
                pad = max(6.0, max(g[2] - g[0], g[3] - g[1]))
                x1, y1 = int(max(0, g[0] - pad)), int(max(0, g[1] - pad))
                x2, y2 = int(min(im.size[0], g[2] + pad)), int(min(im.size[1], g[3] + pad))
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                crop = np.array(im.crop((x1, y1, x2, y2)))
        except Exception:
            continue
        vals.append({"sharpness": float(cv2.Laplacian(crop, cv2.CV_64F).var()), "h_ratio": r["h_ratio"],
                     "w_ratio": r["w_ratio"], "eu": r["eq640"], "key": r["key"], "source": r["source"]})
    if not vals:
        return {"available": False}
    thr = statistics.median([v["sharpness"] for v in vals])
    low = [v for v in vals if v["sharpness"] <= thr]
    high = [v for v in vals if v["sharpness"] > thr]
    return {"available": True, "n": len(vals), "median_sharpness": thr,
            "low_sharp": {"n": len(low), "h_ratio_median": pct([v["h_ratio"] for v in low], 0.5),
                          "h_ratio_mean": (dist([v["h_ratio"] for v in low]) or {}).get("mean"),
                          "w_ratio_median": pct([v["w_ratio"] for v in low], 0.5)},
            "high_sharp": {"n": len(high), "h_ratio_median": pct([v["h_ratio"] for v in high], 0.5),
                           "h_ratio_mean": (dist([v["h_ratio"] for v in high]) or {}).get("mean"),
                           "w_ratio_median": pct([v["w_ratio"] for v in high], 0.5)},
            "pearson_sharpness_vs_h_ratio": _pearson([v["sharpness"] for v in vals],
                                                     [v["h_ratio"] for v in vals]),
            "note": "proxy only: Laplacian variance of a crop around the GT box; there is no blur label"}


def _pearson(xs, ys):
    import numpy as np
    a, b = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def appearance_extent_proxy(records, tier="TP", max_items=400):
    """Ratio between the GT box and the locally salient blob around it (proxy for label tightness)."""
    import numpy as np
    import cv2
    from PIL import Image
    items = [r for r in records if r["tier"] == tier]
    if len(items) > max_items:
        items = items[:: max(1, len(items) // max_items)][:max_items]
    out = []
    for r in items:
        try:
            with Image.open(r["image"]) as im:
                im = im.convert("L")
                W, H = im.size
                g = [float(v) for v in r["gt"]]
                gw, gh = g[2] - g[0], g[3] - g[1]
                pad = max(12.0, 2.5 * max(gw, gh))
                x1, y1 = int(max(0, g[0] - pad)), int(max(0, g[1] - pad))
                x2, y2 = int(min(W, g[2] + pad)), int(min(H, g[3] + pad))
                crop = np.array(im.crop((x1, y1, x2, y2)))
        except Exception:
            continue
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            continue
        bg = cv2.medianBlur(crop, 2 * (max(crop.shape) // 12) + 1)
        diff = cv2.absdiff(crop, bg).astype(np.float32)
        thr = max(10.0, 2.5 * float(diff.std()))
        mask = (diff > thr).astype(np.uint8) * 255
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            continue
        gcx, gcy = (g[0] + g[2]) / 2.0 - x1, (g[1] + g[3]) / 2.0 - y1
        best, best_d = None, None
        for i in range(1, n):
            x, y, w, h, area = stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]
            if area < 6:
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            contains = x <= gcx <= x + w and y <= gcy <= y + h
            d = math.hypot(cx - gcx, cy - gcy) - (1e6 if contains else 0.0)
            if best_d is None or d < best_d:
                best, best_d = (w, h, area), d
        if best is None:
            continue
        out.append({"key": r["key"], "source": r["source"], "bucket": r["bucket"], "h_ratio": r["h_ratio"],
                    "w_ratio": r["w_ratio"], "gt_w": gw, "gt_h": gh,
                    "blob_w": float(best[0]), "blob_h": float(best[1]),
                    "blob_w_over_gt_w": best[0] / gw if gw > 0 else None,
                    "blob_h_over_gt_h": best[1] / gh if gh > 0 else None})
    if not out:
        return {"available": False}
    return {"available": True, "n": len(out), "sampled": len(items),
            "blob_h_over_gt_h_median": pct([o["blob_h_over_gt_h"] for o in out], 0.5),
            "blob_w_over_gt_w_median": pct([o["blob_w_over_gt_w"] for o in out], 0.5),
            "blob_h_over_gt_h_mean": (dist([o["blob_h_over_gt_h"] for o in out]) or {}).get("mean"),
            "blob_w_over_gt_w_mean": (dist([o["blob_w_over_gt_w"] for o in out]) or {}).get("mean"),
            "rows": out,
            "note": "proxy only: connected component closest to the GT centre after local-background"
                    " subtraction; cluttered real photos can merge the object with the background"}


def gt_convention_train_vs_val(repo, manifest_rel, remaps, train_step=8):
    """Are val GT boxes shorter than train GT boxes of the same width? (label convention test)"""
    import numpy as np
    rows = []
    data = {}
    with open(repo / manifest_rel, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            split = r.get("split")
            if split not in ("train", "val"):
                continue
            if split == "train" and (i % train_step):
                continue
            data.setdefault(split, []).append(r)
    for split in ("train", "val"):
        acc = {}
        for r in data.get(split, []):
            lab = ev.remap(r["label"], remaps) if r.get("label") else ""
            if not lab or not os.path.exists(lab):
                continue
            try:
                W, H = float(r["width"]), float(r["height"])
            except (TypeError, ValueError):
                continue
            try:
                with open(lab, encoding="utf-8", errors="replace") as g:
                    lines = [ln.split() for ln in g.read().splitlines() if ln.strip()]
            except Exception:
                continue
            for parts in lines:
                if len(parts) < 5:
                    continue
                try:
                    bw, bh = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                w_px, h_px = bw * W, bh * H
                eq = ev.equiv_size_640(w_px, h_px, W, H)
                b = ev.bucket_of(eq)
                acc.setdefault(b, []).append((w_px, h_px))
        for b in ev.BUCKET_LABELS:
            vals = acc.get(b, [])
            if not vals:
                continue
            ws = [v[0] for v in vals]
            hs = [v[1] for v in vals]
            rows.append({"split": split, "bucket": b, "n_boxes": len(vals),
                         "gt_w_median": pct(ws, 0.5), "gt_h_median": pct(hs, 0.5),
                         "gt_aspect_median": pct([v[0] / v[1] for v in vals if v[1] > 0], 0.5),
                         "gt_aspect_mean": (dist([v[0] / v[1] for v in vals if v[1] > 0]) or {}).get("mean"),
                         "gt_h_over_w_median": pct([v[1] / v[0] for v in vals if v[0] > 0], 0.5)})
    return rows


def gt_convention_by_source_width(repo, manifest_rel, remaps, train_step=8):
    """At the SAME width (in the 1024 input frame), are val boxes shorter than train boxes?

    Images with different resolutions are made comparable by expressing every box in the 1024 letterbox
    frame (the frame the network actually regresses in). Aspect ratio is scale free.
    """
    bins = [(0, 6), (6, 10), (10, 14), (14, 20), (20, 28), (28, 40), (40, 60), (60, 1e9)]
    rows = []
    data = {}
    with open(repo / manifest_rel, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            split = r.get("split")
            if split not in ("train", "val"):
                continue
            if split == "train" and (i % train_step):
                continue
            data.setdefault(split, []).append(r)
    for split in ("train", "val"):
        acc = {}
        for r in data.get(split, []):
            lab = ev.remap(r["label"], remaps) if r.get("label") else ""
            if not lab or not os.path.exists(lab):
                continue
            try:
                W, H = float(r["width"]), float(r["height"])
            except (TypeError, ValueError):
                continue
            scale = IMGSZ / max(W, H)
            src = r.get("source") or "?"
            try:
                with open(lab, encoding="utf-8", errors="replace") as g:
                    lines = [ln.split() for ln in g.read().splitlines() if ln.strip()]
            except Exception:
                continue
            for parts in lines:
                if len(parts) < 5:
                    continue
                try:
                    bw, bh = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                w1024, h1024 = bw * W * scale, bh * H * scale
                for lo, hi in bins:
                    if lo <= w1024 < hi:
                        acc.setdefault((src, lo, hi), []).append((w1024, h1024))
                        break
        for (src, lo, hi), vals in acc.items():
            ws = [v[0] for v in vals]
            hs = [v[1] for v in vals]
            rows.append({"split": split, "source": src,
                         "w1024_bin": ("[%g,%g)" % (lo, hi)) if hi < 1e8 else ("[%g,inf)" % lo),
                         "n_boxes": len(vals), "median_w1024": pct(ws, 0.5), "median_h1024": pct(hs, 0.5),
                         "median_aspect": pct([v[0] / v[1] for v in vals if v[1] > 0], 0.5),
                         "median_h_over_w": pct([v[1] / v[0] for v in vals if v[0] > 0], 0.5)})
    rows.sort(key=lambda r: (r["split"], str(r["source"]), r["w1024_bin"]))
    return rows


def aspect_vs_h_ratio(records, tier="TP"):
    """Does the height offset concentrate in the flat (wide, short) GT boxes?"""
    abins = [(0.0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.7), (1.7, 1e9)]
    rows = []
    for lo, hi in abins:
        recs = []
        for r in records:
            if r["tier"] != tier or not r["gt_h"]:
                continue
            a = r["gt_w"] / r["gt_h"]
            if lo <= a < hi:
                recs.append(r)
        if not recs:
            continue
        rows.append({"aspect_bin": ("[%g,%g)" % (lo, hi)) if hi < 1e8 else ("[%g,inf)" % lo),
                     "n": len(recs),
                     "gt_w_px1024_median": pct([r["gt_w_px1024"] for r in recs], 0.5),
                     "gt_h_px1024_median": pct([r["gt_h_px1024"] for r in recs], 0.5),
                     "w_ratio_mean": (dist([r["w_ratio"] for r in recs]) or {}).get("mean"),
                     "h_ratio_mean": (dist([r["h_ratio"] for r in recs]) or {}).get("mean"),
                     "h_ratio_median": pct([r["h_ratio"] for r in recs], 0.5),
                     "dh_px1024_median": pct([r["dh_px1024"] for r in recs], 0.5),
                     "dw_px1024_median": pct([r["dw_px1024"] for r in recs], 0.5),
                     "iou_median": pct([r["iou"] for r in recs], 0.5)})
    return rows


# =======================================================================================
# 6. writers
# =======================================================================================
def _num(x, nd=6):
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return ""
        return round(x, nd)
    return x


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_num(r.get(h)) for h in header])
    return len(rows)


def write_roundtrip_json(path, formulas, roundtrip, ultralytics_cc, parity, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    tol = 1e-6
    worst1 = max(r["max_abs_err_norm_roundtrip"] for r in roundtrip)
    worst2 = max(r["max_abs_err_letterbox_roundtrip"] for r in roundtrip)
    worst3 = max(r["max_abs_err_content_roundtrip"] for r in roundtrip)
    worst_h = max(abs(r["h_ratio_content"] - 1.0) for r in roundtrip)
    worst_w = max(abs(r["w_ratio_content"] - 1.0) for r in roundtrip)
    payload = {
        "formulas": formulas,
        "tests": {
            "T1_norm_pixel_norm_roundtrip": {"worst_abs_err": worst1, "tolerance": tol,
                                             "pass": bool(worst1 <= tol)},
            "T2_letterbox_inverse_roundtrip": {"worst_abs_err": worst2, "tolerance": tol,
                                               "pass": bool(worst2 <= tol)},
            "T3_content_roundtrip_actual_pixels": {"worst_abs_err": worst3,
                                                   "worst_h_ratio_bias": worst_h,
                                                   "worst_w_ratio_bias": worst_w,
                                                   "note": "image is resized to round(W*r)/round(H*r);"
                                                           " the inverse divides by the nominal r"},
            "T4_no_axis_specific_gain": {"gain_x_equals_gain_y": True,
                                         "source": "utils/ops.py:147-148 (gain_y = gain_x = gain)"},
        },
        "roundtrip_rows": roundtrip,
        "ultralytics_cross_check": ultralytics_cc,
        "parity": parity,
    }
    payload.update(extra or {})
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return payload


def write_pixel_error(path, records, tier="TP"):
    header = ["kind", "checkpoint", "metric", "unit", "scope", "bin_lo", "bin_hi", "count", "share",
              "mode", "mean", "median", "std", "P10", "P25", "P50", "P75", "P90"]
    rows = []
    metrics = [("dh", "dh_px1024"), ("dw", "dw_px1024"), ("dh_orig", "dh_orig"), ("dw_orig", "dw_orig"),
               ("h_rel", "h_err_rel"), ("w_rel", "w_err_rel")]
    for ck, recs_all in records.items():
        recs = [r for r in recs_all if r["tier"] == tier]
        for name, field in metrics:
            vals = [r[field] for r in recs]
            bins = PY_HIST_BINS if name in ("dh", "dw") else (REL_HIST_BINS if name in ("h_rel", "w_rel") else PY_HIST_BINS)
            units = {"dh": "px1024", "dw": "px1024", "dh_orig": "px_orig", "dw_orig": "px_orig",
                     "h_rel": "rel", "w_rel": "rel"}
            hs, n = histogram(vals, bins)
            for h in hs:
                rows.append({"kind": "histogram", "checkpoint": ck, "metric": name, "unit": units[name],
                             "scope": "ALL", "bin_lo": h["lo"], "bin_hi": h["hi"], "count": h["count"],
                             "share": (h["count"] / n) if n else None})
            d = dist(vals)
            mode, mode_n = mode_of_rounded(vals, 0 if name in ("dh", "dw", "dh_orig", "dw_orig") else 2)
            rows.append({"kind": "summary", "checkpoint": ck, "metric": name, "unit": units[name], "scope": "ALL",
                         "count": d.get("n"), "mode": mode, "mean": d.get("mean"), "median": d.get("median"),
                         "std": d.get("std"), "P10": d.get("P10"), "P25": d.get("P25"), "P50": d.get("P50"),
                         "P75": d.get("P75"), "P90": d.get("P90")})
            if name == "dh":
                rows.append({"kind": "summary", "checkpoint": ck, "metric": "dh_mode_count", "unit": "px1024",
                             "scope": "ALL", "count": mode_n, "mode": mode})
                for lo, hi, lab in [(0.5, 1.5, "share_dh_px1024_(0.5,1.5]"), (1.5, 2.5, "share_dh_px1024_(1.5,2.5]"),
                                    (-0.5, 0.5, "share_dh_px1024_[-0.5,0.5]")]:
                    sel = [v for v in vals if v is not None and lo < v <= hi]
                    rows.append({"kind": "share", "checkpoint": ck, "metric": lab, "unit": "px1024", "scope": "ALL",
                                 "count": len(sel), "share": (len(sel) / n) if n else None})
    return write_csv(path, header, rows)


# =======================================================================================
# 7. examples
# =======================================================================================
def example_groups(records, per_group=EXAMPLES_PER_GROUP):
    tps = [r for r in records if r["tier"] == "TP"]
    by_high = sorted(tps, key=lambda r: -(r["h_ratio"] or 0))[:per_group]
    by_109 = sorted(tps, key=lambda r: abs((r["h_ratio"] or 1.0) - 1.09))[:per_group]
    by_1 = sorted(tps, key=lambda r: abs((r["h_ratio"] or 1.0) - 1.0))[:per_group]
    return [("high_height_ratio", by_high, "top height_ratio"),
            ("typical_109", by_109, "closest to height_ratio 1.09"),
            ("ratio_near_1", by_1, "closest to height_ratio 1.00")]


def export_examples(ckpt, records, samples_by_key, out_root, image_dets, per_group=EXAMPLES_PER_GROUP):
    rows = []
    for name, recs, desc in example_groups(records, per_group):
        for rank, r in enumerate(recs, start=1):
            s = samples_by_key.get(r["key"])
            if s is None:
                continue
            gts = [g["box"] for g in s["gt"]]
            dets = image_dets.get(_key(r["image"]), [])
            dets_t = [(d["conf"], d["box"]) for d in dets]
            cap = ("%s | %s | %s | gt %.0fx%.0f pred %.0fx%.0f | w_r %.3f h_r %.3f | IoU %.3f conf %.2f"
                   % (ckpt, r["source"], r["bucket"], r["gt_w"], r["gt_h"], r["pred_w"], r["pred_h"],
                      r["w_ratio"] or 0.0, r["h_ratio"] or 0.0, r["iou"], r["conf"]))
            fname = "%02d_%s_h%.3f.jpg" % (rank, os.path.splitext(os.path.basename(r["image"]))[0], r["h_ratio"] or 0.0)
            out_path = out_root / name / fname
            aud.draw_example(r["image"], gts, dets_t, out_path, r["gt"], r["pred"], cap, CONF_OP)
            rows.append({"group": name, "rank": rank, "checkpoint": ckpt, "image": os.path.basename(r["image"]),
                         "image_path": r["image"], "source": r["source"], "bucket": r["bucket"],
                         "gt_w": r["gt_w"], "gt_h": r["gt_h"], "pred_w": r["pred_w"], "pred_h": r["pred_h"],
                         "w_ratio": r["w_ratio"], "h_ratio": r["h_ratio"], "iou": r["iou"], "conf": r["conf"],
                         "dh_px1024": r["dh_px1024"], "center_err_px640": r["center_err_px640"],
                         "file": str(out_path.relative_to(out_root.parent))})
    return rows


# =======================================================================================
# 8. ultralytics parity (optional, needs GPU)
# =======================================================================================
def run_parity(repo, ckpt_path, samples, cached_dets, n_images, device, out_dir, conf_robust=CONF_OP):
    """Native ultralytics predict (list source, rect=False) vs the cached pipeline (txt-file source).

    Coordinates are compared only for boxes that both runs agree on, matched one-to-one by IoU>=0.5 at a
    ROBUST confidence (default 0.25). Boxes at the 0.001 floor are numerically fragile (a different
    preprocessing path can flip them across the floor), so counts there are reported separately.
    """
    from ultralytics import YOLO
    items = [s for s in samples if s["gt"]][:: max(1, len(samples) // (n_images * 4))][:n_images]
    keys = [s["image"] for s in items]
    model = YOLO(str(ckpt_path))
    res = model.predict(source=keys, imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300, device=device,
                        rect=False, verbose=False, save=False)
    native_by = {}
    for r in res:
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            for b, c in zip(xyxy, confs):
                boxes.append((float(c), [float(x) for x in b]))
        native_by[_key(r.path)] = boxes
    rows, worst, worst_mean = [], 0.0, 0.0
    pairs_matched = unmatched_native = unmatched_cached = 0
    n_native_all = n_cached_all = n_native_rob = n_cached_rob = 0
    imgs_with_unmatched = 0
    for s in items:
        native = sorted(native_by.get(_key(s["image"]), []), key=lambda t: -t[0])
        cached = sorted([(d["conf"], d["box"]) for d in cached_dets.get(_key(s["image"]), [])],
                        key=lambda t: -t[0])
        n_native_all += len(native)
        n_cached_all += len(cached)
        a = [t for t in native if t[0] >= conf_robust]
        b = [t for t in cached if t[0] >= conf_robust]
        n_native_rob += len(a)
        n_cached_rob += len(b)
        if not a and not b:
            continue
        m = ev.match_greedy([t[1] for t in a], b, 0.50, metric="iou")
        used_native, used_cached = set(), set()
        for ci, nj in m["matches"]:
            used_native.add(nj)
            used_cached.add(ci)
            ba, bb = a[nj][1], b[ci][1]
            d = [ba[k] - bb[k] for k in range(4)]
            worst = max(worst, max(abs(x) for x in d))
            worst_mean = max(worst_mean, statistics.fmean(abs(x) for x in d))
            pairs_matched += 1
            rows.append({"image": os.path.basename(s["image"]), "index": pairs_matched,
                         "conf_native": a[nj][0], "conf_cached": b[ci][0],
                         "dx1": d[0], "dy1": d[1], "dx2": d[2], "dy2": d[3],
                         "dw": (ba[2] - ba[0]) - (bb[2] - bb[0]), "dh": (ba[3] - ba[1]) - (bb[3] - bb[1]),
                         "x1_native": ba[0], "y1_native": ba[1], "x2_native": ba[2], "y2_native": ba[3],
                         "x1_cached": bb[0], "y1_cached": bb[1], "x2_cached": bb[2], "y2_cached": bb[3],
                         "center_err": ev.center_dist(ba, bb), "iou": ev.iou_xyxy(ba, bb)})
        un = len(a) - len(used_native)
        uc = len(b) - len(used_cached)
        unmatched_native += un
        unmatched_cached += uc
        if un or uc:
            imgs_with_unmatched += 1
            rows.append({"image": os.path.basename(s["image"]), "index": -1,
                         "note": "unmatched at conf>=%.2f: native=%d cached=%d" % (conf_robust, un, uc),
                         "conf_native": max([t[0] for t in a], default=None),
                         "conf_cached": max([t[0] for t in b], default=None)})
    path = out_dir / "bbox_ultralytics_parity.csv"
    header = ["image", "index", "note", "conf_native", "conf_cached", "dx1", "dy1", "dx2", "dy2", "dw", "dh",
              "x1_native", "y1_native", "x2_native", "y2_native", "x1_cached", "y1_cached", "x2_cached",
              "y2_cached", "center_err", "iou"]
    write_csv(path, header, rows)
    summary = {"images": len(items), "conf_robust": conf_robust,
               "boxes_compared": pairs_matched, "unmatched_native": unmatched_native,
               "unmatched_cached": unmatched_cached, "images_with_unmatched": imgs_with_unmatched,
               "max_abs_coord_diff": worst, "max_mean_abs_coord_diff": worst_mean,
               "boxes_at_0.001_native": n_native_all, "boxes_at_0.001_cached": n_cached_all,
               "boxes_at_conf_robust_native": n_native_rob, "boxes_at_conf_robust_cached": n_cached_rob,
               "identical": bool(worst == 0.0), "mismatch_rows": unmatched_native + unmatched_cached}
    return summary, rows
# =======================================================================================
# 9. report
# =======================================================================================
def md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return chr(10).join(out)


def _fnum(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def f4(x, nd=4):
    x = _fnum(x)
    return "n/a" if x is None else ("%%.%df" % nd) % x


def _formula_lines():
    return [
        "GT（本仓库评估器 tools/eval_yolo26_v1.py:401-402，无 letterbox、无 padding）:",
        "    w_px = bw*W ; h_px = bh*H             # W,H = PIL Image.open(img).size",
        "    box  = [(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H]",
        "",
        "letterbox（ultralytics 8.4.150 data/augment.py, LetterBox）:",
        "    r = min(imgsz/H, imgsz/W)              # 1745  同一个 r 用于 x 与 y",
        "    new_unpad = (round(W*r), round(H*r))   # 1751",
        "    dw,dh = imgsz-round(W*r), imgsz-round(H*r) ; dw/=2 ; dh/=2      # 1752,1761-1762",
        "    top=round(dh-0.1) bottom=round(dh+0.1) left=round(dw-0.1) right=round(dw+0.1)  # 1764-1765",
        "    img = cv2.resize(img, new_unpad) ; copyMakeBorder(top,bottom,left,right,114)   # 1793,1801",
        "    boxes: denormalize(W,H) -> *ratio(r,r) -> +(left,top)                          # 1884-1886",
        "    rect(auto=True) 仅在同形状批次且 args.rect 时启用；predict 默认关闭 -> 恒 1024x1024  # predictor.py:212-219",
        "",
        "scale-back（models/yolo/detect/predict.py:121 -> utils/ops.py:146-165, ratio_pad=None）:",
        "    gain = min(img1_h/H, img1_w/W) ; gain_y = gain_x = gain        # 147-148",
        "    pad_x = round((img1_w-round(W*gain))/2-0.1) ; pad_y 同理     # 149-150",
        "    x1-=pad_x y1-=pad_y x2-=pad_x y2-=pad_y ; /=gain ; clip_boxes # 155-165",
        "",
        "本仓库评估器不自己解码预测：直接用 ultralytics 返回的 res.boxes.xyxy（已回代到原图）。",
        "equiv_size_640 / center_err_px640 只用于分桶与展示，不参与 pred/GT 比值（比值恒在原始像素系）。",
    ]


def write_report(path, args, ckpts, rt, ucc, parity, pooled, by_size, by_source, cf_ap, cf_cross, cf_tp_summary,
                 sharp, app, conv, conv2, aspect_rows, pixel_rows, examples_rows):
    L = []
    add = L.append
    add("# BBOX HEIGHT BIAS ROOT-CAUSE AUDIT - YOLO26 V1 shuttle ultra-small-target detector")
    add("")
    add("> 自动部分（1-11 节）由 tools/audit_bbox_height_bias.py 生成；0 节结论与 12 节根因等级表为人工撰写（工具重跑会清空）。")
    add("> 只读审计：不训练、不改 checkpoint/标签/配置/模型结构/loss/augmentation；只用现有标签、图像与已缓存预测。")
    add("")
    add("## 0. 结论（人工）")
    add("")
    add("（待填。）")
    add("")
    add("## 1. 坐标路径与公式（逐行取自实际代码）")
    add("")
    add("```")
    for line in _formula_lines():
        add(line)
    add("```")
    add("")
    add("先验排除的两类原因：")
    add("")
    add("- gain_x == gain_y（各向同性）：纯缩放/letterbox 误差会让 w 与 h 产生相同比例偏差，无法产生 w_ratio≈1.00 而 h_ratio≈1.09 的方向性模式。")
    add("- padding 只平移不改尺寸：pad 以相同方式进入 x1/y1/x2/y2，只影响中心误差。")
    add("- 唯一轴向不对称来自 new_unpad 取整：内容按 round(W*r)/W 与 round(H*r)/H 缩放、回代除以名义 r，残差 ≤0.5/(min(W,H)*r)，量级 ≤0.1%，方向偏小（见 2 节 T3）。")
    add("")
    add("## 2. round-trip 与数学一致性（bbox_coordinate_roundtrip.json）")
    add("")
    rows = [
        ["T1 normalized->pixel->normalized", "PASS" if rt["T1"]["pass"] else "FAIL", "%.3e" % rt["T1"]["worst_abs_err"], "容差 1e-6"],
        ["T2 original->letterbox(名义 r)->inverse->original", "PASS" if rt["T2"]["pass"] else "FAIL", "%.3e" % rt["T2"]["worst_abs_err"], "容差 1e-6"],
        ["T3 实际像素路径 round(new_unpad)", "info", "%.3e" % rt["T3"]["worst_abs_err"], "h_ratio 偏差 %.2e / w_ratio 偏差 %.2e（负=偏小）" % (rt["T3"]["worst_h_ratio_bias"], rt["T3"]["worst_w_ratio_bias"])],
        ["T4 gain_x == gain_y", "PASS", "-", "ops.py:147-148"],
    ]
    add(md_table(["测试", "结果", "最坏误差", "说明"], rows))
    add("")
    res_list = sorted({("%dx%d" % (r["W"], r["H"])) for r in rt["rows"]})
    add("覆盖分辨率 %s，框类型：中央 / 左上 / 右下 / 4x6 / 6x8 / 8x12 / 4x12 / 12x4，共 %d 例。" % (", ".join(res_list), len(rt["rows"])))
    add("")
    if ucc.get("available"):
        add("与安装版 ultralytics 数值交叉验证：我的 scale_boxes 复现 vs ultralytics.utils.ops.scale_boxes，最坏逐坐标差 %s；letterbox 输出形状差 %s。" % (f4(ucc.get("max_abs_diff_inverse"), 9), f4(ucc.get("max_abs_diff_padded_shape"), 3)))
    else:
        add("ultralytics 交叉验证不可用：%s" % ucc.get("error"))
    add("")
    add("## 3. Ultralytics 原生 vs 自定义 evaluator（逐框）")
    add("")
    if parity is None:
        add("未运行（加 --parity 用 GPU 跑 16 张 val 图）。自定义 evaluator 不自己解码预测，直接用 ultralytics 的 res.boxes.xyxy（同源），GT 侧只做 1.1 节解码且已经 T1 验证。")
    else:
        add(md_table(["项目", "值"], [["图片数", parity["images"]], ["逐框比较数", parity["boxes_compared"]], ["多余框行数", parity["mismatch_rows"]], ["最大 |Δ坐标| 像素", f4(parity["max_abs_coord_diff"], 12)], ["完全一致", parity["identical"]]]))
    add("")
    add("## 4. 事实复核：pred/GT 尺寸比（val，TP，conf>=0.25）")
    add("")
    rows = []
    for ck in ckpts:
        p = pooled[ck]
        rows.append([ck, p["TP"], f4(p["w_ratio_mean"]), f4(p["w_ratio_median"]), f4(p["h_ratio_mean"]), f4(p["h_ratio_median"]), f4(p["h_err_rel_median"]), f4(p["h_err_rel_P10"]), f4(p["h_err_rel_P90"]), str(p["dh_mode"]), f4(p["dh_median"], 3)])
    add(md_table(["checkpoint", "TP", "w_ratio mean", "w_ratio med", "h_ratio mean", "h_ratio med", "h_err med", "h_err P10", "h_err P90", "dh_px1024 mode", "dh_px1024 med"], rows))
    add("")
    add("### 4.1 ratio 与 IoU 的关系（检验条件化造成的选择效应）")
    add("")
    add(md_table(["IoU 区间", "n", "w_ratio mean", "w_ratio med", "h_ratio mean", "h_ratio med", "h_err med", "dh_px1024 med", "gt_h_px1024 med"],
                 [["[%.2f,%.2f)" % (r["iou_lo"], r["iou_hi"]), r["n"], f4(r["w_ratio_mean"]), f4(r["w_ratio_median"]), f4(r["h_ratio_mean"]), f4(r["h_ratio_median"]), f4(r["h_err_rel_median"]), f4(r["dh_px1024_median"], 3), f4(r["gt_h_px1024_median"], 2)] for r in rt["ratio_vs_iou"]]))
    add("")
    add("## 5. 按尺寸桶（equiv_size_640）")
    add("")
    add(md_table(["ckpt", "bucket", "GT", "TP", "w_ratio mean", "h_ratio mean", "h_ratio med", "h_err med", "dh_px1024 mean", "dh_px1024 med", "dh mode", "share |dh|<=0.5px", "share dh in (0.5,1.5]", "IoU med"],
                 [[r["checkpoint"], r["bucket"], r["GT"], r["TP"], f4(r["w_ratio_mean"]), f4(r["h_ratio_mean"]), f4(r["h_ratio_median"]), f4(r["h_err_rel_median"]), f4(r["dh_px1024_mean"], 3), f4(r["dh_px1024_median"], 3), str(r["dh_px1024_mode"]), f4(r["share_dh_px1024_within_0.5"], 3), f4(r["share_dh_px1024_plus1"], 3), f4(r["iou_median"])] for r in by_size]))
    add("")
    add("## 6. 按数据来源（source）")
    add("")
    add(md_table(["scope", "group", "ckpt", "GT", "TP", "gt_w med", "gt_h med", "gt aspect med", "gt eq640 med", "w_ratio mean", "h_ratio mean", "h_ratio med", "h_err med", "h_err P10", "h_err P90", "dh_px1024 med", "IoU med"],
                 [[r["scope"], r["group"], r["checkpoint"], r["GT"], r["TP"], f4(r["gt_w_median"], 2), f4(r["gt_h_median"], 2), f4(r["gt_aspect_median"], 3), f4(r["gt_eq640_median"], 2), f4(r["w_ratio_mean"]), f4(r["h_ratio_mean"]), f4(r["h_ratio_median"]), f4(r["h_err_rel_median"]), f4(r["h_err_rel_P10"]), f4(r["h_err_rel_P90"]), f4(r["dh_px1024_median"], 3), f4(r["iou_median"])] for r in by_source if r["scope"] == "source"]))
    add("")
    add("## 7. 像素空间：+1 px 量化还是连续比例偏差（bbox_pixel_error_distribution.csv）")
    add("")
    rows = []
    for r in pixel_rows:
        if r.get("kind") != "summary":
            continue
        rows.append([r.get("checkpoint"), r.get("metric"), r.get("unit"), str(r.get("mode")), f4(r.get("median"), 3), f4(r.get("mean"), 3), f4(r.get("P10"), 3), f4(r.get("P90"), 3)])
    add(md_table(["ckpt", "metric", "unit", "mode", "median", "mean", "P10", "P90"], rows))
    add("")
    add("share 行（同一 CSV 内 kind=share）：|dh|<=0.5px、dh in (0.5,1.5]、dh in (1.5,2.5] 的占比。")
    add("")
    add("## 8. 外观代理（模糊 / 局部显著范围）：仅为代理，不是标注")
    add("")
    if sharp.get("available"):
        add(md_table(["分组", "n", "h_ratio med", "h_ratio mean", "w_ratio med"], [
            ["低清晰度半 (Laplacian var <= %.1f 中位)" % sharp["median_sharpness"], sharp["low_sharp"]["n"], f4(sharp["low_sharp"]["h_ratio_median"]), f4(sharp["low_sharp"]["h_ratio_mean"]), f4(sharp["low_sharp"]["w_ratio_median"])],
            ["高清晰度半", sharp["high_sharp"]["n"], f4(sharp["high_sharp"]["h_ratio_median"]), f4(sharp["high_sharp"]["h_ratio_mean"]), f4(sharp["high_sharp"]["w_ratio_median"])],
        ]))
        add("")
        add("Pearson(清晰度, h_ratio) = %s（n=%d）。" % (f4(sharp["pearson_sharpness_vs_h_ratio"], 4), sharp["n"]))
    else:
        add("模糊代理不可用。")
    add("")
    if app.get("available"):
        add("局部显著 blob / GT（代理标签是否比可见目标紧）：blob_h/gt_h 中位 %s、均值 %s；blob_w/gt_w 中位 %s、均值 %s（n=%d，抽样 %d）。" % (f4(app["blob_h_over_gt_h_median"], 3), f4(app["blob_h_over_gt_h_mean"], 3), f4(app["blob_w_over_gt_w_median"], 3), f4(app["blob_w_over_gt_w_mean"], 3), app["n"], app["sampled"]))
        add("")
        add("说明：%s" % app["note"])
    else:
        add("外观范围代理不可用。")
    add("")
    add("## 9. Counterfactual：分别纠正 center / width / height")
    add("")
    add("### 9.1 TP 层面（逐 TP 重算 IoU）")
    add("")
    add(md_table(["variant", "n_tp", "IoU mean", "IoU med", "dIoU mean", "dIoU med", "跨过 0.75", "跨过 0.85", "跨过 0.90", "跨过 0.95"],
                 [[v["variant"], v["n_tp"], f4(v["iou_mean"]), f4(v["iou_median"]), f4(v["d_iou_mean"]), f4(v["d_iou_median"]), v["up_75"], v["up_85"], v["up_90"], v["up_95"]] for v in cf_tp_summary]))
    add("")
    add("### 9.2 AP 层面（重算整个检测集合的 AP，association 固定为原框匹配）")
    add("")
    add(md_table(["ckpt", "variant", "TP@0.5", "AP50", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95", "mAP50-95", "dAP75", "dAP85", "dAP90", "dAP95", "dmAP50-95"],
                 [[r["checkpoint"], r["variant"], r["TP50"], f4(r.get("AP50")), f4(r.get("AP70")), f4(r.get("AP75")), f4(r.get("AP80")), f4(r.get("AP85")), f4(r.get("AP90")), f4(r.get("AP95")), f4(r.get("mAP50-95")), f4(r.get("dAP75")), f4(r.get("dAP85")), f4(r.get("dAP90")), f4(r.get("dAP95")), f4(r.get("dmAP50-95"))] for r in cf_ap]))
    add("")
    add("### 9.3 跨阈值计数（TP 集合内）")
    add("")
    sel = [r for r in cf_cross if r["variant"] != "orig"]
    add(md_table(["threshold", "variant", "n_tp", "crossed_up", "crossed_down", "share_up"],
                 [[r["threshold"], r["variant"], r["n_tp"], r["crossed_up"], r["crossed_down"], f4(r["share_up"], 4)] for r in sel]))
    add("")
    add("## 10. 标注风格：train vs val 的 GT 尺寸分布（同一 manifest）")
    add("")
    add(md_table(["split", "bucket", "boxes", "gt_w med", "gt_h med", "gt aspect med", "gt h/w med"],
                 [[r["split"], r["bucket"], r["n_boxes"], f4(r["gt_w_median"], 2), f4(r["gt_h_median"], 2), f4(r["gt_aspect_median"], 3), f4(r["gt_h_over_w_median"], 3)] for r in conv]))
    add("")
    add("### 10.1 同一宽度桶下 train 与 val 的 GT 高度（1024 输入系，分辨率归一）")
    add("")
    add(md_table(["split", "source", "w1024 区间", "boxes", "w1024 med", "h1024 med", "aspect med", "h/w med"],
                 [[r["split"], r["source"], r["w1024_bin"], r["n_boxes"], f4(r["median_w1024"], 2),
                   f4(r["median_h1024"], 2), f4(r["median_aspect"], 3), f4(r["median_h_over_w"], 3)]
                  for r in conv2]))
    add("")
    add("### 10.2 GT 宽高比 vs 高度误差（TP）")
    add("")
    add(md_table(["GT aspect 区间", "n", "gt_w1024 med", "gt_h1024 med", "w_ratio mean", "h_ratio mean",
                  "h_ratio med", "dh_px1024 med", "dw_px1024 med", "IoU med"],
                 [[r["aspect_bin"], r["n"], f4(r["gt_w_px1024_median"], 2), f4(r["gt_h_px1024_median"], 2),
                   f4(r["w_ratio_mean"]), f4(r["h_ratio_mean"]), f4(r["h_ratio_median"]),
                   f4(r["dh_px1024_median"], 3), f4(r["dw_px1024_median"], 3), f4(r["iou_median"])]
                  for r in aspect_rows]))
    add("")
    add("## 11. 复现与产物")
    add("")
    add("```")
    add(args.command_line)
    add("```")
    add("")
    add("- 输入：val 标签与图像尺寸（manifest %s）+ 已缓存预测 %s/loc_<ckpt>_val.json。" % (args.manifest, args.cache_dir))
    add("- 输出：bbox_coordinate_roundtrip.json、bbox_height_bias_by_source.csv、bbox_height_bias_by_size.csv、bbox_pixel_error_distribution.csv、bbox_counterfactual_iou.csv、bbox_counterfactual_tp_detail.csv、bbox_gt_convention_train_vs_val.csv、bbox_ultralytics_parity.csv(--parity)、示例图 %d 张。" % len(examples_rows))
    add("- 本轮未修改模型/标签/配置/loss/augmentation，未覆盖旧报告与旧 metrics。")
    add("")
    add("## 12. 根因等级表与 Q1-Q5（人工）")
    add("")
    add("（待填。）")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(L) + chr(10), encoding="utf-8")


# =======================================================================================
# 10. cli
# =======================================================================================
def ckpt_summary(recs):
    tps = [r for r in recs if r["tier"] == "TP"]
    wr, hr = dist([r["w_ratio"] for r in tps]), dist([r["h_ratio"] for r in tps])
    hrel = dist([r["h_err_rel"] for r in tps])
    dh = [r["dh_px1024"] for r in tps]
    return {"TP": len(tps), "w_ratio_mean": wr.get("mean"), "w_ratio_median": wr.get("median"),
            "h_ratio_mean": hr.get("mean"), "h_ratio_median": hr.get("median"),
            "h_err_rel_median": hrel.get("median"), "h_err_rel_P10": hrel.get("P10"),
            "h_err_rel_P90": hrel.get("P90"), "dh_mode": mode_of_rounded(dh, 0)[0],
            "dh_median": (dist(dh) or {}).get("median")}


def cf_tp_summary_rows(tp_rows):
    out = []
    for v in CF_VARIANTS:
        key = "iou_" + v
        vals = [r[key] for r in tp_rows]
        base = [r["iou_orig"] for r in tp_rows]
        dv, db = dist(vals) or {}, dist(base) or {}
        row = {"variant": v, "n_tp": len(tp_rows), "iou_mean": dv.get("mean"), "iou_median": dv.get("median"),
               "d_iou_mean": (dv.get("mean") or 0) - (db.get("mean") or 0),
               "d_iou_median": (dv.get("median") or 0) - (db.get("median") or 0)}
        for thr in CF_CROSS:
            row["up_%d" % round(thr * 100)] = sum(1 for r in tp_rows if r[key] >= thr and r["iou_orig"] < thr)
        out.append(row)
    return out


def cf_csv_rows(cf_ap, cf_cross, cf_tp_summary):
    rows = []
    for r in cf_ap:
        d = {"section": "ap"}
        d.update(r)
        rows.append(d)
    for r in cf_cross:
        d = {"section": "crossing"}
        d.update(r)
        rows.append(d)
    for r in cf_tp_summary:
        d = {"section": "tp_summary"}
        d.update(r)
        rows.append(d)
    return rows


CF_HEADER = ["section", "checkpoint", "variant", "threshold", "GT", "TP50", "recall50", "AP50", "AP70", "AP75",
             "AP80", "AP85", "AP90", "AP95", "mAP50-95", "dAP50", "dAP70", "dAP75", "dAP80", "dAP85", "dAP90",
             "dAP95", "dmAP50-95", "n_tp", "crossed_up", "crossed_down", "share_up", "iou_mean", "iou_median",
             "d_iou_mean", "d_iou_median", "up_75", "up_85", "up_90", "up_95"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Root-cause audit of the systematic bbox height bias (read-only).")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--ckpt-name", action="append", default=[], help="default: a_best b_best b_last")
    ap.add_argument("--ckpt-registry", default="configs/shuttle_detection/checkpoints_v1.yaml")
    ap.add_argument("--cache-dir", default="_scratch_localization_audit")
    ap.add_argument("--conf-op", type=float, default=CONF_OP)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--out-dir", default="outputs/shuttle_capability/metrics")
    ap.add_argument("--examples-dir", default="outputs/shuttle_capability/bbox_bias_examples")
    ap.add_argument("--examples-ckpt", default="a_best")
    ap.add_argument("--examples-per-group", type=int, default=EXAMPLES_PER_GROUP)
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--proxy-ckpt", default="a_best")
    ap.add_argument("--appearance-sample", type=int, default=400)
    ap.add_argument("--train-step", type=int, default=8)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--parity-images", type=int, default=16)
    ap.add_argument("--parity-device", default="0")
    ap.add_argument("--report", default="outputs/shuttle_capability/reports/BBOX_HEIGHT_BIAS_ROOT_CAUSE.md")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    args.command_line = "python tools/audit_bbox_height_bias.py " + " ".join(sys.argv[1:])

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    remaps = []
    for m in args.map:
        if "=" in m:
            a, b = m.split("=", 1)
            remaps.append((a, b))
    names = list(args.ckpt_name) or ["a_best", "b_best", "b_last"]
    ckpts = ev.resolve_checkpoints([], args.ckpt_registry, names)

    print("[1] coordinate round-trip + ultralytics cross-check", flush=True)
    roundtrip = run_roundtrip_tests()
    ucc = ultralytics_cross_check()
    rt = {"rows": roundtrip,
          "T1": {"worst_abs_err": max(r["max_abs_err_norm_roundtrip"] for r in roundtrip)},
          "T2": {"worst_abs_err": max(r["max_abs_err_letterbox_roundtrip"] for r in roundtrip)},
          "T3": {"worst_abs_err": max(r["max_abs_err_content_roundtrip"] for r in roundtrip),
                 "worst_h_ratio_bias": max(abs(r["h_ratio_content"] - 1.0) for r in roundtrip),
                 "worst_w_ratio_bias": max(abs(r["w_ratio_content"] - 1.0) for r in roundtrip)}}
    rt["T1"]["pass"] = bool(rt["T1"]["worst_abs_err"] <= 1e-6)
    rt["T2"]["pass"] = bool(rt["T2"]["worst_abs_err"] <= 1e-6)
    print("    T1 worst=%.3e pass=%s | T2 worst=%.3e pass=%s | T3 worst=%.3e (h_ratio bias %.2e, w_ratio bias %.2e)"
          % (rt["T1"]["worst_abs_err"], rt["T1"]["pass"], rt["T2"]["worst_abs_err"], rt["T2"]["pass"],
             rt["T3"]["worst_abs_err"], rt["T3"]["worst_h_ratio_bias"], rt["T3"]["worst_w_ratio_bias"]), flush=True)
    if ucc.get("available"):
        print("    ultralytics scale_boxes cross-check: worst |delta| = %.3e ; padded-shape delta = %s"
              % (ucc["max_abs_diff_inverse"], ucc["max_abs_diff_padded_shape"]), flush=True)

    print("[2] val samples + manifest metadata", flush=True)
    samples, skipped = aud.build_val_samples(repo, args.manifest, remaps, args.max_images)
    meta = load_val_meta(repo, args.manifest)
    n_gt = sum(len(s["gt"]) for s in samples)
    print("    images=%d GT=%d skipped=%d meta_rows=%d" % (len(samples), n_gt, skipped, len(meta)), flush=True)
    samples_by_key = {s["key"]: s for s in samples}

    print("[3] records from cached predictions", flush=True)
    records, gt_rows, per_ckpt_dets, pooled = {}, None, {}, {}
    for ck in ckpts:
        dets, cpath = load_cached_dets(repo, ck["name"], "val", args.cache_dir)
        if dets is None:
            print("    SKIP %s: no cache %s" % (ck["name"], cpath), flush=True)
            continue
        for s in samples:
            s["dets"] = dets.get(_key(s["image"]), [])
        recs, gts, unmatched = build_records(samples, dets, meta, args.conf_op)
        records[ck["name"]] = recs
        per_ckpt_dets[ck["name"]] = dets
        if gt_rows is None:
            gt_rows = gts
        pooled[ck["name"]] = ckpt_summary(recs)
        p = pooled[ck["name"]]
        print("    %-8s TP=%d w_ratio mean/med %.4f/%.4f h_ratio mean/med %.4f/%.4f dh_px1024 mode=%s median=%.3f"
              % (ck["name"], p["TP"], p["w_ratio_mean"] or 0, p["w_ratio_median"] or 0, p["h_ratio_mean"] or 0,
                 p["h_ratio_median"] or 0, p["dh_mode"], p["dh_median"] or 0), flush=True)
    if not records:
        raise SystemExit("no cached predictions found; run tools/audit_localization_yolo26_v1.py first")
    ck_names = list(records.keys())

    print("[4] aggregations", flush=True)
    by_size, by_source = [], []
    for ck in ck_names:
        for r in bucket_rows(records[ck], gt_rows):
            r["checkpoint"] = ck
            by_size.append(r)
        for r in source_rows(records[ck], gt_rows):
            r["checkpoint"] = ck
            by_source.append(r)
    size_header = ["checkpoint", "bucket", "GT", "TP", "gt_w_median", "gt_h_median", "gt_aspect_median",
                   "gt_h_px1024_median", "w_ratio_mean", "w_ratio_median", "h_ratio_mean", "h_ratio_median",
                   "w_err_rel_mean", "h_err_rel_mean", "w_err_rel_median", "h_err_rel_median", "w_err_rel_P10",
                   "h_err_rel_P10", "h_err_rel_P90", "dw_px1024_median", "dh_px1024_median", "dh_px1024_mean",
                   "dh_px1024_mode", "share_dh_px1024_within_0.5", "share_dh_px1024_plus1", "iou_median",
                   "center_err_px640_median"]
    src_header = ["checkpoint", "scope", "group", "GT", "TP", "images", "gt_w_median", "gt_h_median",
                  "gt_aspect_median", "gt_eq640_median", "w_ratio_mean", "w_ratio_median", "h_ratio_mean",
                  "h_ratio_median", "h_err_rel_mean", "h_err_rel_median", "h_err_rel_P10", "h_err_rel_P90",
                  "dh_px1024_median", "iou_median", "center_err_px640_median"]
    n1 = write_csv(out_dir / "bbox_height_bias_by_size.csv", size_header, by_size)
    n2 = write_csv(out_dir / "bbox_height_bias_by_source.csv", src_header, by_source)
    n3 = write_pixel_error(out_dir / "bbox_pixel_error_distribution.csv", records)
    ratio_vs_iou = ratio_vs_iou_rows(records[ck_names[0]])
    rv_header = ["iou_lo", "iou_hi", "n", "w_ratio_mean", "w_ratio_median", "h_ratio_mean", "h_ratio_median",
                 "h_err_rel_median", "dh_px1024_median", "gt_h_px1024_median"]
    write_csv(out_dir / "bbox_ratio_vs_iou.csv", rv_header, ratio_vs_iou)
    print("    by_size=%d by_source=%d pixel=%d ratio_vs_iou=%d" % (n1, n2, n3, len(ratio_vs_iou)), flush=True)

    print("[5] counterfactual", flush=True)
    cf_ap_all, cf_cross_all, cf_sum_all, tp_detail_all = [], [], [], []
    for ck in ck_names:
        for s in samples:
            s["dets"] = per_ckpt_dets[ck].get(_key(s["image"]), [])
        tp_rows = counterfactual_tp_rows(records[ck])
        for r in tp_rows:
            r["checkpoint"] = ck
            tp_detail_all.append(r)
        summ = cf_tp_summary_rows(tp_rows)
        for r in summ:
            r["checkpoint"] = ck
        cf_sum_all.extend(summ)
        cross = counterfactual_crossings(tp_rows)
        for r in cross:
            r["checkpoint"] = ck
        cf_cross_all.extend(cross)
        apd = counterfactual_ap(samples, per_ckpt_dets[ck], ck)
        for v in CF_VARIANTS:
            cf_ap_all.append(apd[v])
        o, h, c = apd["orig"], apd["fix_height"], apd["fix_center"]
        print("    %-8s AP75/85/90/95 orig %.4f/%.4f/%.4f/%.4f | fix_height %.4f/%.4f/%.4f/%.4f | fix_center %.4f/%.4f/%.4f/%.4f"
              % (ck, o.get("AP75") or 0, o.get("AP85") or 0, o.get("AP90") or 0, o.get("AP95") or 0,
                 h.get("AP75") or 0, h.get("AP85") or 0, h.get("AP90") or 0, h.get("AP95") or 0,
                 c.get("AP75") or 0, c.get("AP85") or 0, c.get("AP90") or 0, c.get("AP95") or 0), flush=True)
    write_csv(out_dir / "bbox_counterfactual_iou.csv", CF_HEADER, cf_csv_rows(cf_ap_all, cf_cross_all, cf_sum_all))
    tp_header = ["checkpoint", "key", "source", "bucket", "conf", "gt_w", "gt_h", "pred_w", "pred_h", "w_ratio",
                 "h_ratio", "dh_px1024", "dw_px1024", "center_err_px640"]
    tp_header += ["iou_" + v for v in CF_VARIANTS] + ["diou_" + v for v in CF_VARIANTS[1:]]
    write_csv(out_dir / "bbox_counterfactual_tp_detail.csv", tp_header, tp_detail_all)

    print("[6] appearance proxies (cv2)", flush=True)
    sharp = sharpness_proxy(records[args.proxy_ckpt]) if args.proxy_ckpt in records else {"available": False}
    app = (appearance_extent_proxy(records[args.proxy_ckpt], max_items=args.appearance_sample)
           if args.proxy_ckpt in records else {"available": False})
    if sharp.get("available"):
        print("    sharpness n=%d low_h=%.4f high_h=%.4f pearson=%.4f"
              % (sharp["n"], sharp["low_sharp"]["h_ratio_median"] or 0,
                 sharp["high_sharp"]["h_ratio_median"] or 0, sharp["pearson_sharpness_vs_h_ratio"] or 0), flush=True)
    if app.get("available"):
        print("    blob/GT h %.3f w %.3f (n=%d)"
              % (app["blob_h_over_gt_h_median"], app["blob_w_over_gt_w_median"], app["n"]), flush=True)

    print("[7] GT convention train vs val", flush=True)
    conv = gt_convention_train_vs_val(repo, args.manifest, remaps, args.train_step)
    conv2 = gt_convention_by_source_width(repo, args.manifest, remaps, args.train_step)
    write_csv(out_dir / "bbox_gt_convention_train_vs_val_by_source_width.csv",
              ["split", "source", "w1024_bin", "n_boxes", "median_w1024", "median_h1024", "median_aspect",
               "median_h_over_w"], conv2)
    aspect_rows = aspect_vs_h_ratio(records[args.proxy_ckpt]) if args.proxy_ckpt in records else []
    write_csv(out_dir / "bbox_aspect_vs_h_ratio.csv",
              ["aspect_bin", "n", "gt_w_px1024_median", "gt_h_px1024_median", "w_ratio_mean", "h_ratio_mean",
               "h_ratio_median", "dh_px1024_median", "dw_px1024_median", "iou_median"], aspect_rows)
    print("    conv rows=%d by_source_width=%d aspect_rows=%d" % (len(conv), len(conv2), len(aspect_rows)),
          flush=True)
    conv_header = ["split", "bucket", "n_boxes", "gt_w_median", "gt_h_median", "gt_aspect_median",
                   "gt_aspect_mean", "gt_h_over_w_median"]
    write_csv(out_dir / "bbox_gt_convention_train_vs_val.csv", conv_header, conv)
    print("    rows=%d" % len(conv), flush=True)

    print("[8] parity (optional)", flush=True)
    parity = None
    if args.parity:
        ck0 = next((c for c in ckpts if c["name"] in records), None)
        parity, _prows = run_parity(repo, ck0["path"], samples, per_ckpt_dets[ck0["name"]], args.parity_images,
                                    args.parity_device, out_dir)
        print("    parity images=%d boxes=%d mismatches=%d max|dcoord|=%.3e identical=%s"
              % (parity["images"], parity["boxes_compared"], parity["mismatch_rows"],
                 parity["max_abs_coord_diff"], parity["identical"]), flush=True)

    print("[9] examples", flush=True)
    examples_rows = []
    if not args.no_examples and args.examples_ckpt in records:
        ex_root = repo / args.examples_dir
        examples_rows = export_examples(args.examples_ckpt, records[args.examples_ckpt], samples_by_key, ex_root,
                                        per_ckpt_dets[args.examples_ckpt], args.examples_per_group)
        print("    examples=%d -> %s" % (len(examples_rows), ex_root), flush=True)

    print("[10] write json + report", flush=True)
    write_roundtrip_json(out_dir / "bbox_coordinate_roundtrip.json", _formula_lines(), roundtrip, ucc, parity,
                         {"config": {"conf_op": args.conf_op, "imgsz": IMGSZ, "manifest": args.manifest,
                                     "command": args.command_line, "cache_dir": args.cache_dir},
                          "checkpoints": [c.get("name") for c in ckpts],
                          "pooled_tp_summary": pooled,
                          "ratio_vs_iou": ratio_vs_iou,
                          "by_size": by_size, "by_source": by_source,
                          "counterfactual_ap": cf_ap_all, "counterfactual_crossings": cf_cross_all,
                          "counterfactual_tp_summary": cf_sum_all,
                          "sharpness_proxy": dict(sharp),
                          "appearance_proxy": {k: v for k, v in app.items() if k != "rows"},
                          "gt_convention_train_vs_val": conv})
    rt["ratio_vs_iou"] = ratio_vs_iou
    if not args.no_report:
        with open(out_dir / "bbox_pixel_error_distribution.csv", encoding="utf-8") as f:
            pixel_rows = list(csv.DictReader(f))
        write_report(repo / args.report, args, ck_names, rt, ucc, parity, pooled, by_size, by_source,
                     cf_ap_all, cf_cross_all, cf_sum_all, sharp, app, conv, conv2, aspect_rows,
                     pixel_rows, examples_rows)
        print("    report -> %s" % (repo / args.report), flush=True)
    print("done. outputs in %s" % out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
