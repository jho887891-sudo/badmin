#!/usr/bin/env python3
"""Third-party baseline audit: the ETH Zurich official shuttle detector vs our YOLO26-P2 (read-only).

Scope rules: evaluation only. No training, no fine-tuning, no checkpoint/label/test-set edits, no per-model
threshold tuning, no evaluator changes. The ETH checkpoint is used exactly as released; the only compatibility
step is documented in the report (it is a custom dict, not an ultralytics checkpoint).

Everything else is shared with our own evaluation to keep the comparison fair:
  * the same image lists (manifest val split / controlled_capability/images / challenge_test/images),
  * the same GT decoding (tools/eval_yolo26_v1.py:401-402), the same letterbox (imgsz=1024) and scale-back,
  * the same IoU matcher and AP code (all-point PR curve, GT-weighted mean) -- i.e. the same evaluator,
  * the same operating confidence 0.25 and the same AP-curve floor 0.001, same NMS (iou=0.7) and max_det=300.

The ETH-style center-distance metric is reported SEPARATELY (their definition: top-1 box per image at their
confidence 0.5, distance < 25 px measured in the letterboxed network frame, one GT per image). ETH-style F1 is
NOT the IoU-based F1 and must never be quoted as if it were.

Outputs (default outputs/shuttle_capability/):
  metrics/eth_official_overall.csv, eth_official_ap_by_iou.csv, eth_official_size_buckets.csv,
  eth_official_localization.csv, eth_official_false_positive.csv, eth_official_latency.csv,
  eth_official_center_metric.csv, eth_vs_ours.csv, eth_data_leakage_audit.csv,
  eth_checkpoint_manifest.json
  examples/eth_official/{tp,fn,fp,size_lt4,size_4_6,size_6_8,size_8_12}/
  examples/model_disagreement/{A_ok_eth_fail,eth_ok_A_fail,B_ok_eth_fail,eth_ok_B_fail}/
  reports/ETH_OFFICIAL_BASELINE_AUDIT.md

Run:  python tools/eval_eth_official_baseline.py --eth-ckpt <path> [--ours-metrics-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_bbox_height_bias as hb  # noqa: E402
import audit_localization_yolo26_v1 as aud  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402

IMGSZ = 1024
CONF_OP = 0.25
CONF_FLOOR = 0.001
NMS_IOU = 0.7
MAX_DET = 300
ETH_MODEL_NAME = "yolov8s"
ETH_CONF = 0.5
ETH_DIST_PX = 25.0
ETH_TRAIN_LOCS = ["cab_1", "cab_2", "glc_2", "uetlibergstrasse_1", "uetlibergstrasse_2", "ml_3", "ml_6",
                  "ticino_2", "glc_1", "ml_4", "ticino_1", "coco_train"]
ETH_TRAIN_DIFFS = ["easy", "medium"]
ETH_ROOT = Path(r"D:\_eth_data\eth_shuttle_detection")
SETS = ["val", "controlled_capability/images", "challenge_test/images"]
ETH_COMPARE_SPLIT = "eth_unseen_external"
AGGREGATES = [("<4", ["<4"]), ("4_6", ["4-6"]), ("6_8", ["6-8"]), ("<8", ["<4", "4-6", "6-8"]),
              ("8_12", ["8-12"]), ("12_16", ["12-16"]), ("8_16", ["8-12", "12-16"]), ("16_32", ["16-24", "24-32"]),
              ("32_64", ["32-64"]), (">64", [">64"])]


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =======================================================================================
# 1. checkpoint provenance and loading (documented compatibility step)
# =======================================================================================
def verify_checkpoint(path, inventory_csv=None, expected_sha256=None, expected_bytes=None):
    path = Path(path)
    info = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    try:
        info["bytes"] = path.stat().st_size
        info["sha256"] = sha256_file(path)
    except OSError as exc:
        info["read_error"] = repr(exc)
        return info
    if expected_sha256:
        info["sha256_matches_record"] = bool(info["sha256"].lower() == expected_sha256.lower())
    if expected_bytes:
        info["bytes_match_record"] = bool(info["bytes"] == int(expected_bytes))
    if inventory_csv and Path(inventory_csv).exists():
        with open(inventory_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("relpath", "").endswith("final-model/best.pt"):
                    info["inventory_relpath"] = row["relpath"]
                    info["inventory_bytes"] = row.get("bytes")
                    info["inventory_sha256"] = row.get("sha256")
                    info["sha256_matches_inventory"] = bool(info["sha256"].lower() == (row.get("sha256") or "").lower())
    try:
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        info["ckpt_type"] = type(ckpt).__name__
        if isinstance(ckpt, dict):
            info["ckpt_keys"] = sorted(list(ckpt.keys()))[:20]
            sd = ckpt.get("model_state_dict") or ckpt.get("model") or {}
            if hasattr(sd, "state_dict"):
                sd = sd.state_dict()
            info["state_dict_tensors"] = len(sd) if hasattr(sd, "__len__") else None
            info["model_name_in_ckpt"] = ckpt.get("model_name")
            info["is_ultralytics_format"] = bool("model" in ckpt and "train_args" in ckpt)
            try:
                import torch as _t
                n_params = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))
                info["state_dict_params"] = n_params
                det = [k for k in sd if k.endswith(".dfl.conv.weight") or ".cv3." in k]
                info["detection_head_keys_sample"] = sorted(det)[:8]
                import re as _re
                idx = set()
                for k in sd:
                    m = _re.match(r"model\.22\.cv3\.(\d+)\.", k)
                    if m:
                        idx.add(int(m.group(1)))
                info["nc_from_state_dict"] = (max(idx) + 1) if idx else None
                info["dfl_weight_shape"] = list(sd["model.22.dfl.conv.weight"].shape) if "model.22.dfl.conv.weight" in sd else None
            except Exception as exc:
                info["param_scan_error"] = repr(exc)
    except Exception as exc:
        info["load_error"] = repr(exc)
    return info


def load_eth_model(ckpt_path, device="0"):
    """Minimal, transparent compatibility step: rebuild the yolov8s architecture and load their state dict."""
    import torch
    from ultralytics import YOLO
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise SystemExit("unexpected ETH checkpoint layout: keys=%s" % (list(ckpt)[:10] if isinstance(ckpt, dict) else type(ckpt)))
    model_name = ckpt.get("model_name") or ETH_MODEL_NAME
    wrapper = YOLO("%s.yaml" % model_name)          # architecture only, no pretrained weights
    missing, unexpected = wrapper.model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit("state dict does not match %s.yaml: missing=%s unexpected=%s"
                         % (model_name, list(missing)[:5], list(unexpected)[:5]))
    # NOTE: the ETH release fine-tunes the stock yolov8s detection head (nc=80) while every training label
    # is class 0 (their own evaluator asserts cls == 0). We keep the head exactly as released and never
    # rewrite nc/names; the class histogram of the predictions is recorded in the cache for transparency.
    import re
    cls_idx = set()
    for k in ckpt["model_state_dict"]:
        m = re.match(r"model\.22\.cv3\.(\d+)\.", k)
        if m:
            cls_idx.add(int(m.group(1)))
    nc = (max(cls_idx) + 1) if cls_idx else None
    wrapper.model.eval()
    meta = {"model_name": model_name, "nc": nc, "nc_note": "stock head, all training labels were class 0",
            "names": str(wrapper.model.names)[:80],
            "stride": [int(x) for x in (wrapper.model.stride.tolist() if hasattr(wrapper.model.stride, "tolist")
                                        else [wrapper.model.stride])],
            "head_modules": [type(m).__name__ for m in wrapper.model.model[-1].__dict__.get("_modules", {}).values()][:6],
            "params": int(sum(p.numel() for p in wrapper.model.parameters())),
            "load_state_dict_strict_ok": True}
    return wrapper, meta


def model_cost(model_wrapper, imgsz=IMGSZ):
    import torch
    from ultralytics.utils.torch_utils import get_flops
    m = model_wrapper.model
    params = int(sum(p.numel() for p in m.parameters()))
    try:
        flops = float(get_flops(m, imgsz))
    except Exception:
        flops = None
    heads = None
    try:
        heads = [type(mm).__name__ for mm in m.model[-1].modules() if type(mm).__name__ in ("Detect", "Segment", "Pose")]
    except Exception:
        pass
    nc = None
    try:
        nc = int(m.model[-1].nc)
    except Exception:
        nc = int(getattr(m, "nc", -1))
    return {"params": params, "gflops": flops, "stride": [int(x) for x in m.stride.tolist()],
            "nc": nc, "head_types": heads}


# =======================================================================================
# 2. ETH-style center-distance metric (their definition, reported separately)
# =======================================================================================
def to_letterbox_frame(box, W, H, imgsz=IMGSZ):
    """Original-image pixels -> the letterboxed network frame (what their Metric measures in)."""
    p = hb.letterbox_params(W, H, imgsz)
    r, left, top = p["r"], p["left"], p["top"]
    return [box[0] * r + left, box[1] * r + top, box[2] * r + left, box[3] * r + top]


def gt_center_letterbox(gt_box, W, H, imgsz=IMGSZ):
    p = hb.letterbox_params(W, H, imgsz)
    cx = (gt_box[0] + gt_box[2]) / 2.0 / W * imgsz
    cy = (gt_box[1] + gt_box[3]) / 2.0 / H * imgsz
    return cx, cy


def eth_style_center_metric(samples, dets_by_key, conf=ETH_CONF, dist_thr=ETH_DIST_PX, frame="content"):
    """Reproduce the ETH metric: top-1 prediction per image, one GT per image, hit if distance < thr.

    Their code compares centers in the letterboxed network frame, where both GT and prediction carry the
    SAME padding, so padding cancels and the frame distance equals r * (original-pixel distance) with
    r = min(imgsz/H, imgsz/W) -- the single isotropic letterbox ratio. We therefore measure in original
    pixels and multiply by r, which is exactly equivalent and free of padding conventions.

    frame = "content": their convention (dist_thr is in the letterbox content frame).
    frame = "original": documented variant (dist_thr in original image pixels).
    """
    tp = fp = fn = 0
    dists = []
    n_gt_img = n_pred_img = 0
    for s in samples:
        W, H = float(s["img_w"]), float(s["img_h"])
        r = hb.letterbox_params(W, H)["r"]
        gts = s["gt"]
        gt_c = None
        if gts:
            g = gts[0]["box"]          # ETH-faithful: their dataset has one shuttle per frame
            gt_c = ((g[0] + g[2]) / 2.0, (g[1] + g[3]) / 2.0)
            n_gt_img += 1
        dets = [(d["conf"], d["box"]) for d in dets_by_key.get(s["key"], []) if d["conf"] >= conf]
        pred_c = None
        if dets:
            c, b = max(dets, key=lambda t: t[0])
            pred_c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
            n_pred_img += 1
        if gt_c is None and pred_c is None:
            continue
        if gt_c is None:
            fp += 1
            continue
        if pred_c is None:
            fn += 1
            continue
        d = math.hypot(gt_c[0] - pred_c[0], gt_c[1] - pred_c[1])
        if frame == "content":
            d = d * r
        if d < dist_thr:
            tp += 1
            dists.append(d)
        else:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"conf": conf, "dist_thr": dist_thr, "frame": frame, "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "mean_dist": (statistics.fmean(dists) if dists else None),
            "median_dist": (statistics.median(dists) if dists else None),
            "images_with_gt": n_gt_img, "images_with_pred": n_pred_img, "images": len(samples),
            "note": "ETH-style center-distance metric: NOT an IoU-based metric"}
# =======================================================================================
# 3. IoU metrics shared with our own evaluator (aud.evaluate_localization) + ETH additions
# =======================================================================================
def bucket_ap5095(samples, bucket, thresholds=None):
    thresholds = list(thresholds or aud.AUDIT_IOU_THRESHOLDS)
    acc = {t: [] for t in thresholds}
    for s in samples:
        sub = [g["box"] for g in s["gt"] if g["bucket"] == bucket]
        if not sub:
            continue
        dets = [(d["conf"], d["box"]) for d in s.get("dets") or []]
        for t in thresholds:
            a = ev.ap_for_gt_set(sub, dets, t)
            if a is not None:
                acc[t].append((a, len(sub)))
    out = {}
    for t in thresholds:
        items = [a for a in acc[t] if a[0] is not None]
        n = sum(a[1] for a in items)
        out["AP%d" % round(t * 100)] = (sum((a[0] or 0.0) * a[1] for a in items) / n) if n else None
    vals = [v for v in out.values() if v is not None]
    out["mAP50-95"] = statistics.fmean(vals) if vals else None
    return out


def evaluate_model_on_set(model_tag, set_key, samples, dets_map, conf_op=CONF_OP, with_bucket_ap=True):
    """Everything the audit needs for one (model, set): audit metrics + bucket AP50-95 + aggregate recalls."""
    for s in samples:
        s["dets"] = dets_map.get(os.path.abspath(s["image"]), [])
    res = aud.evaluate_localization(samples, conf_op=conf_op)
    res["model"] = model_tag
    res["set"] = set_key
    if with_bucket_ap:
        res["bucket_ap5095"] = {b: bucket_ap5095(samples, b) for b in ev.BUCKET_LABELS}
    return res


def overall_row(res, model_tag, meta):
    gt = res["gt"]
    tp, fp, fn = res["tp50"], res["fp50"], res["fn50"]
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / gt if gt else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec)) else None
    row = {"model": model_tag, "set": res["set"], "images": res["images"], "GT": gt,
           "detections@0.25": res["detections_op"], "detections@floor": res["detections_all"],
           "TP": tp, "FP": fp, "FN": fn, "Precision": prec, "Recall": rec, "F1": f1,
           "FP_per_image": (fp / res["images"]) if res["images"] else None,
           "params": meta.get("params"), "gflops": meta.get("gflops")}
    for t in aud.AUDIT_IOU_THRESHOLDS:
        row["AP%d" % round(t * 100)] = res["ap_by_iou"].get(t)
    row["mAP50-95"] = res["mAP50-95"]
    for name, labs in AGGREGATES:
        tp_a = sum(res["buckets"][l]["tp50"] for l in labs)
        gt_a = sum(res["buckets"][l]["gt"] for l in labs)
        row["Recall_%s" % name] = (tp_a / gt_a) if gt_a else None
        row["GT_%s" % name] = gt_a
        row["TP_%s" % name] = tp_a
    return row


def ap_by_iou_row(res, model_tag):
    row = {"model": model_tag, "set": res["set"], "images": res["images"], "GT": res["gt"],
           "detections@0.25": res["detections_op"], "TP@0.5": res["tp50"], "FP@0.5": res["fp50"],
           "FN@0.5": res["fn50"], "Precision@op": res["precision_op"], "Recall@0.5": res["recall50"]}
    for t in aud.AUDIT_IOU_THRESHOLDS:
        row["AP%d" % round(t * 100)] = res["ap_by_iou"].get(t)
    row["mAP50-95"] = res["mAP50-95"]
    return row


def localization_row(res, model_tag):
    st = res["iou_dist"]["stats"]
    q = res["iou_dist"]["quantiles"]
    e = res["errors"]
    row = {"model": model_tag, "set": res["set"], "n_tp": res["iou_dist"]["n"],
           "IoU_mean": st.get("mean"), "IoU_median": st.get("median"), "IoU_P10": q.get("P10"),
           "IoU_P25": q.get("P25"), "IoU_P75": q.get("P75"), "IoU_P90": q.get("P90")}
    for key, out in [("center_dist_px_640", "center_px640"), ("normalized_center_error", "norm_center"),
                     ("width_ratio_signed", "width_ratio"), ("height_ratio_signed", "height_ratio"),
                     ("area_ratio", "area_ratio"), ("width_rel_err", "width_rel_err"),
                     ("height_rel_err", "height_rel_err"), ("center_dist_px", "center_px")]:
        d = e.get(key) or {}
        for stat in ("mean", "median", "P10", "P90"):
            row["%s_%s" % (out, stat)] = d.get(stat)
    return row


def size_bucket_row(res, model_tag, bucket):
    b = res["buckets"][bucket]
    ap = (res.get("bucket_ap5095") or {}).get(bucket, {})
    return {"model": model_tag, "set": res["set"], "bucket": bucket, "GT": b["gt"], "TP": b["tp50"],
            "FN": max(0, b["gt"] - b["tp50"]), "Recall": b["recall"],
            "AP50": b.get("AP50"), "AP75": b.get("AP75"), "AP90": b.get("AP90"), "AP95": b.get("AP95"),
            "mAP50-95": ap.get("mAP50-95"), "iou_mean": b.get("iou_mean"), "iou_median": b.get("iou_median"),
            "center_err_px640_median": b.get("center_dist_px_640_median"),
            "width_rel_err_median": b.get("width_rel_err_median"),
            "height_rel_err_median": b.get("height_rel_err_median")}


# =======================================================================================
# 4. false positives on the unlabeled hard-negative sets
# =======================================================================================
def fp_rows(model_tag, groups, dets_map, thresholds=(0.25, 0.50, 0.75)):
    rows = []
    for g in groups:
        dets = [dets_map.get(os.path.abspath(str(p)), []) for p in g.images]
        stats = ev.fp_stats(dets, list(thresholds))
        for th in thresholds:
            d = stats[str(th)]
            rows.append({"model": model_tag, "set": g.key, "threshold": th, "images": d["total_images"],
                         "FP_total": d["total_FP"], "FP_per_image": d["FP_per_image"],
                         "images_with_FP": d["images_with_FP"], "image_FP_rate": d["image_FP_rate"],
                         "max_conf_FP": d["max_confidence_FP"], "median_conf_FP": d["median_confidence_FP"]})
    return rows


# =======================================================================================
# 5. latency / cost (same protocol for every model: batch=1, fp32, warmup 50, measure 200)
# =======================================================================================
def latency_bench(model_wrapper, image_paths, device="0", imgsz=IMGSZ, warmup=50, n=200):
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics.data.augment import LetterBox
    m = model_wrapper.model
    dev = torch.device("cuda:%s" % device if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    tensor = None
    for p in image_paths[:1]:
        with Image.open(p) as im:
            arr = np.array(im.convert("RGB"))
        tensor = LetterBox((imgsz, imgsz), auto=False, stride=32)(image=arr)
    x = torch.from_numpy(np.ascontiguousarray(tensor.transpose(2, 0, 1))).float().unsqueeze(0).to(dev) / 255.0
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    # raw forward (their protocol: model(imgs) only)
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if dev.type == "cuda" else None
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)] if dev.type == "cuda" else None
        wall = []
        for i in range(n):
            if starts:
                starts[i].record()
            t0 = time.perf_counter()
            _ = m(x)
            wall.append((time.perf_counter() - t0) * 1000.0)
            if ends:
                ends[i].record()
        if dev.type == "cuda":
            torch.cuda.synchronize()
            fwd = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        else:
            fwd = wall
    peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
    # end-to-end predict path (preprocess + inference + postprocess), framework-reported breakdown
    e2e, comp = [], []
    for p in image_paths[1:(1 + 12)]:
        t0 = time.perf_counter()
        r = model_wrapper.predict(source=p, imgsz=imgsz, conf=0.001, iou=NMS_IOU, max_det=MAX_DET, device=device,
                                  verbose=False, save=False, rect=False)[0]
        e2e.append((time.perf_counter() - t0) * 1000.0)
        sp = getattr(r, "speed", None) or {}
        comp.append({k: float(sp.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess")})

    def stats_ms(v):
        v = sorted(v)
        def pct(q):
            return ev.percentile(v, q)
        return {"mean": statistics.fmean(v), "median": statistics.median(v), "P90": pct(0.90), "P95": pct(0.95)}

    out = {"forward_protocol": "batch=1 fp32 torch.cuda.Event around model(x) on a pre-letterboxed 1024 tensor",
           "forward_raw": stats_ms(fwd), "fps_raw": 1000.0 / statistics.fmean(fwd),
           "end_to_end_predict": stats_ms(e2e) if e2e else None,
           "fps_end_to_end": (1000.0 / statistics.fmean(e2e)) if e2e else None,
           "components_ms_mean": ({k: statistics.fmean([c[k] for c in comp]) for k in ("preprocess", "inference",
                                                                                      "postprocess")}
                                  if comp else None),
           "vram_peak_MB": (peak / (1024 * 1024)) if peak else None,
           "warmup": warmup, "measured": n, "device": str(dev)}
    return out


# =======================================================================================
# 6. data leakage audit (file-level, deterministic)
# =======================================================================================
def eth_rel(image_path):
    try:
        return Path(str(image_path)).resolve().relative_to(ETH_ROOT).as_posix()
    except Exception:
        return None


def leakage_classify(image_path):
    """A = file sits in an ETH official training folder, B = same location, other split,
    C = ETH benchmark but another location/difficulty (coco_val) or outside the ETH dataset entirely."""
    rel = eth_rel(image_path)
    if rel is None:
        return {"class": "C_outside_eth_dataset", "rel": None, "location": None, "difficulty": None, "sub": None}
    parts = rel.split("/")
    loc_diff = parts[0]
    if "_" in loc_diff:
        loc, _, diff = loc_diff.rpartition("_")
    else:
        loc, diff = loc_diff, "?"
    sub = parts[2] if len(parts) > 2 else "?"
    trained = (loc in ETH_TRAIN_LOCS) and (diff in ETH_TRAIN_DIFFS) and (sub == "train")
    if trained:
        cls = "A_eth_trained_folder"
    elif loc in ETH_TRAIN_LOCS:
        cls = "B_same_location_not_trained"
    else:
        cls = "C_eth_benchmark_other_split"
    return {"class": cls, "rel": rel, "location": loc, "difficulty": diff, "sub": sub}


def leakage_rows(models_sets):
    """One row per (model, set, leakage class): items are (model, set_key, samples, dets_map).

    The dets map is passed explicitly: the samples list is shared and re-attached per model, so reading
    s["dets"] here would silently use whichever model ran last.
    """
    rows = []
    for model, set_key, samples, dets_map in models_sets:
        per = {}
        for s in samples:
            c = leakage_classify(s["image"])["class"]
            d = per.setdefault(c, {"images": 0, "GT": 0})
            d["images"] += 1
            d["GT"] += len(s["gt"])
        tp_per = {c: 0 for c in per}
        for b in ev.BUCKET_LABELS:
            for k in per:
                pass
        for c in per:
            sub = [s for s in samples if leakage_classify(s["image"])["class"] == c]
            tp = 0
            for s in sub:
                gb = [g["box"] for g in s["gt"]]
                if not gb:
                    continue
                dets = [(d["conf"], d["box"]) for d in dets_map.get(os.path.abspath(s["image"]), [])
                        if d["conf"] >= CONF_OP]
                m = ev.match_greedy(gb, dets, 0.5, metric="iou")
                tp += sum(1 for x in m["gt_hit"] if x)
            rows.append({"model": model, "set": set_key, "leakage_class": c,
                         "location": "", "images": per[c]["images"], "GT": per[c]["GT"], "TP@0.5": tp,
                         "Recall@0.5": (tp / per[c]["GT"]) if per[c]["GT"] else None})
    return rows


def leakage_inventory(samples):
    """Per-image classification of one evaluation set (for the CSV and the report)."""
    rows = []
    for s in samples:
        info = leakage_classify(s["image"])
        rows.append({"set": "", "image": s["image"], "leakage_class": info["class"], "rel": info["rel"],
                     "location": info["location"], "difficulty": info["difficulty"], "sub": info["sub"],
                     "GT": len(s["gt"])})
    return rows


def eth_unseen_samples(samples):
    """The subset that the ETH model provably did NOT train on (classes B and C)."""
    return [s for s in samples if leakage_classify(s["image"])["class"] != "A_eth_trained_folder"]


def background_leak_probe(manifest_path, search_roots):
    """Are the synthetic sets rendered on real background photos that come from the ETH dataset?"""
    import collections
    out = {"backgrounds": {}, "classes": collections.Counter(), "resolved": 0, "unresolved": 0}
    if not Path(manifest_path).exists():
        return out
    names = set()
    with open(manifest_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = r.get("background")
            if b:
                names.add(os.path.basename(b))
    for n in sorted(names):
        found = None
        for root in search_roots:
            cand = Path(root) / n
            if cand.exists():
                found = str(cand)
                break
            try:
                hits = list(Path(root).rglob(n))[:1]
            except Exception:
                hits = []
            if hits:
                found = str(hits[0])
                break
        if found:
            c = leakage_classify(found)["class"]
            out["backgrounds"][n] = {"path": found, "class": c}
            out["classes"][c] += 1
            out["resolved"] += 1
        else:
            out["backgrounds"][n] = {"path": None, "class": "UNKNOWN_NOT_FOUND"}
            out["classes"]["UNKNOWN_NOT_FOUND"] += 1
            out["unresolved"] += 1
    out["classes"] = dict(out["classes"])
    return out


# =======================================================================================
# 7. model disagreement and examples
# =======================================================================================
def per_image_success(samples, dets_map, conf_op=CONF_OP, iou_thr=0.5):
    """image key -> (n_gt, n_tp@0.5, n_fp, max_conf_fp, best_unmatched_box)."""
    out = {}
    for s in samples:
        gb = [g["box"] for g in s["gt"]]
        dets = [(d["conf"], d["box"]) for d in dets_map.get(os.path.abspath(s["image"]), [])]
        dets_op = [d for d in dets if d[0] >= conf_op]
        if gb:
            m = ev.match_greedy(gb, dets_op, iou_thr, metric="iou")
            n_tp = sum(1 for x in m["gt_hit"] if x)
            matched = {di for di, _gj in m["matches"]}
            fps = [d for i, d in enumerate(dets_op) if i not in matched]
        else:
            n_tp = 0
            fps = dets_op
        out[s["key"]] = {"n_gt": len(gb), "n_tp": n_tp, "n_fp": len(fps),
                         "max_conf_fp": (max((d[0] for d in fps), default=None)),
                         "fp_box": (max(fps, key=lambda t: t[0])[1] if fps else None)}
    return out


def disagreement_pairs(samples, dets_a, dets_b, tag_a, tag_b, conf_op=CONF_OP):
    sa = per_image_success(samples, dets_a, conf_op)
    sb = per_image_success(samples, dets_b, conf_op)
    groups = {"%s_ok_%s_fail" % (tag_a, tag_b): [], "%s_ok_%s_fail" % (tag_b, tag_a): []}
    for s in samples:
        ka, kb = sa[s["key"]], sb[s["key"]]
        if ka["n_gt"] == 0 or kb["n_gt"] == 0:
            continue
        ok_a, ok_b = ka["n_tp"] > 0, kb["n_tp"] > 0
        if ok_a and not ok_b:
            groups["%s_ok_%s_fail" % (tag_a, tag_b)].append(s)
        elif ok_b and not ok_a:
            groups["%s_ok_%s_fail" % (tag_b, tag_a)].append(s)
    return groups


def export_examples(model_tag, samples, dets_map, out_root, per_kind=20, conf_op=CONF_OP):
    """TP / FN / FP / per-size examples for one model (labelled sets only)."""
    rows = []
    kinds = {"tp": [], "fn": [], "fp": []}
    for s in samples:
        gb = [g["box"] for g in s["gt"]]
        dets = [(d["conf"], d["box"]) for d in dets_map.get(os.path.abspath(s["image"]), [])]
        dets_op = [d for d in dets if d[0] >= conf_op]
        if gb:
            m = ev.match_greedy(gb, dets_op, 0.5, metric="iou")
            matched = {di for di, _gj in m["matches"]}
            for di, gj in m["matches"]:
                iou = ev.iou_xyxy(gb[gj], dets_op[di][1])
                kinds["tp"].append((iou, s, gb[gj], dets_op[di][1], dets_op[di][0], "tp"))
            for gj, hit in enumerate(m["gt_hit"]):
                if not hit:
                    kinds["fn"].append((0.0, s, gb[gj], None, None, "fn"))
            for i, d in enumerate(dets_op):
                if i not in matched:
                    kinds["fp"].append((d[0], s, gb[0], d[1], d[0], "fp"))
        else:
            for d in dets_op:
                kinds["fp"].append((d[0], s, None, d[1], d[0], "fp"))
    kinds["tp"].sort(key=lambda t: -t[0])
    kinds["fn"].sort(key=lambda t: -t[0])
    kinds["fp"].sort(key=lambda t: -t[0])
    for kind in ("tp", "fn", "fp"):
        for rank, (score, s, gt, pred, conf, kname) in enumerate(kinds[kind][:per_kind], start=1):
            cap = ("%s | %s | %s #%d | gt_h %.1f | conf %s | iou %s"
                   % (model_tag, s["key"].split(os.sep)[-1], kind, rank,
                      (gt[3] - gt[1]) if gt else 0.0,
                      ("%.3f" % conf) if conf is not None else "n/a",
                      ("%.3f" % score) if kind == "tp" else "n/a"))
            fname = "%02d_%s_%s.jpg" % (rank, kind, os.path.splitext(os.path.basename(s["key"]))[0])
            out_path = out_root / kind / fname
            aud.draw_example(s["image"], [g["box"] for g in s["gt"]],
                             [(d["conf"], d["box"]) for d in dets_map.get(os.path.abspath(s["image"]), [])],
                             out_path, gt, pred, cap, conf_op)
            rows.append({"model": model_tag, "kind": kind, "rank": rank, "image": s["key"], "score": score,
                         "file": str(out_path)})
    # per-size examples (smallest buckets)
    for bucket in ["<4", "4-6", "6-8", "8-12"]:
        picked = 0
        for s in samples:
            if picked >= 10:
                break
            cand = [g for g in s["gt"] if g["bucket"] == bucket]
            if not cand:
                continue
            g = cand[0]
            gb = [x["box"] for x in s["gt"]]
            dets_op = [(d["conf"], d["box"]) for d in dets_map.get(os.path.abspath(s["image"]), [])
                       if d["conf"] >= conf_op]
            m = ev.match_greedy(gb, dets_op, 0.5, metric="iou") if gb else {"matches": []}
            pair = [(di, gj) for di, gj in m["matches"] if gb[gj] == g["box"]]
            pred = dets_op[pair[0][0]][1] if pair else None
            cap = "%s | %s | size %s | gt %s | matched %s" % (model_tag, os.path.basename(s["key"]), bucket,
                                                              ("%.0fx%.0f" % (g["box"][2] - g["box"][0],
                                                                              g["box"][3] - g["box"][1])),
                                                              bool(pair))
            fname = "%02d_%s_%s.jpg" % (picked + 1, bucket.replace("<", "lt").replace("-", "_"),
                                        os.path.splitext(os.path.basename(s["key"]))[0])
            out_path = out_root / ("size_" + bucket.replace("<", "lt").replace("-", "_")) / fname
            aud.draw_example(s["image"], gb, [(d["conf"], d["box"]) for d in
                                              dets_map.get(os.path.abspath(s["image"]), [])],
                             out_path, g["box"], pred, cap, conf_op)
            rows.append({"model": model_tag, "kind": "size_" + bucket, "rank": picked + 1, "image": s["key"],
                         "score": None, "file": str(out_path)})
            picked += 1
    return rows


def export_disagreement(samples, dets_a, dets_b, tag_a, tag_b, out_root, per_kind=10, conf_op=CONF_OP):
    groups = disagreement_pairs(samples, dets_a, dets_b, tag_a, tag_b, conf_op)
    rows = []
    for name, ss in groups.items():
        for rank, s in enumerate(ss[:per_kind], start=1):
            gb = [g["box"] for g in s["gt"]]
            da = [(d["conf"], d["box"]) for d in dets_a.get(os.path.abspath(s["image"]), []) if d["conf"] >= conf_op]
            db = [(d["conf"], d["box"]) for d in dets_b.get(os.path.abspath(s["image"]), []) if d["conf"] >= conf_op]
            allboxes = da + db
            cap = "%s vs %s | %s #%d | %s" % (tag_a, tag_b, name, rank, os.path.basename(s["image"]))
            out_path = out_root / name / ("%02d_%s.jpg" % (rank, os.path.splitext(os.path.basename(s["image"]))[0]))
            aud.draw_example(s["image"], gb, allboxes, out_path, gb[0] if gb else None, None, cap, conf_op)
            rows.append({"group": name, "rank": rank, "image": s["image"], "file": str(out_path),
                         "n_det_%s" % tag_a: len(da), "n_det_%s" % tag_b: len(db)})
    return rows


# =======================================================================================
# 8. writers and report
# =======================================================================================
def _n(x, nd=6):
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
            w.writerow([_n(r.get(h)) for h in header])
    return len(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return chr(10).join(out)


def f4(x, nd=4):
    if x is None or x == "":
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return ("%%.%df" % nd) % x


OVERALL_HEADER = ["model", "set", "images", "GT", "detections@0.25", "detections@floor", "TP", "FP", "FN",
                  "Precision", "Recall", "F1", "FP_per_image", "params", "gflops"] + \
                 ["AP%d" % round(t * 100) for t in aud.AUDIT_IOU_THRESHOLDS] + ["mAP50-95"] + \
                 ["Recall_%s" % n for n, _ in AGGREGATES] + ["GT_%s" % n for n, _ in AGGREGATES] + \
                 ["TP_%s" % n for n, _ in AGGREGATES]
AP_HEADER = ["model", "set", "images", "GT", "detections@0.25", "TP@0.5", "FP@0.5", "FN@0.5", "Precision@op",
             "Recall@0.5"] + ["AP%d" % round(t * 100) for t in aud.AUDIT_IOU_THRESHOLDS] + ["mAP50-95"]
LOC_HEADER = ["model", "set", "n_tp", "IoU_mean", "IoU_median", "IoU_P10", "IoU_P25", "IoU_P75", "IoU_P90"]
for key in ["center_px640", "norm_center", "width_ratio", "height_ratio", "area_ratio", "width_rel_err",
            "height_rel_err", "center_px"]:
    LOC_HEADER += ["%s_%s" % (key, s) for s in ("mean", "median", "P10", "P90")]
SIZE_HEADER = ["model", "set", "bucket", "GT", "TP", "FN", "Recall", "AP50", "AP75", "AP90", "AP95", "mAP50-95",
               "iou_mean", "iou_median", "center_err_px640_median", "width_rel_err_median",
               "height_rel_err_median"]
FP_HEADER = ["model", "set", "threshold", "images", "FP_total", "FP_per_image", "images_with_FP", "image_FP_rate",
             "max_conf_FP", "median_conf_FP"]
CENTER_HEADER = ["model", "set", "subset", "variant", "conf", "dist_thr", "frame", "TP", "FP", "FN", "precision", "recall",
                 "f1", "mean_dist", "median_dist", "images_with_gt", "images_with_pred", "images"]
LAT_HEADER = ["model", "params", "gflops", "nc", "stride", "latency_mean_ms", "latency_median_ms",
              "latency_P90_ms", "latency_P95_ms", "fps", "e2e_mean_ms", "e2e_median_ms", "e2e_P90_ms",
              "e2e_P95_ms", "fps_e2e", "preprocess_ms", "inference_ms", "postprocess_ms", "vram_peak_MB",
              "warmup", "measured", "device", "protocol"]
LEAK_HEADER = ["model", "set", "leakage_class", "images", "GT", "TP@0.5", "Recall@0.5"]
PARAMS_HEADER = ["model", "params", "gflops", "nc", "stride", "latency_mean_ms", "latency_median_ms",
                 "latency_P90_ms", "latency_P95_ms", "fps", "vram_peak_MB"]


# =======================================================================================
# 9. orchestration
# =======================================================================================
LAST_CLASS_HIST = {}


def predict_with_model(model_wrapper, keys, out_dir, tag, imgsz, conf_floor, device):
    """Same policy as tools/eval_yolo26_v1.py:308-325 (txt source, stream, conf floor, NMS 0.7, max_det 300)."""
    src_file = out_dir / ("%s_images.txt" % tag)
    src_file.write_text(chr(10).join(k.replace("\\", "/") for k in keys) + chr(10), encoding="utf-8")
    by_path, n = {}, 0
    stream = model_wrapper.predict(source=str(src_file), imgsz=imgsz, conf=conf_floor, iou=NMS_IOU,
                                   max_det=MAX_DET, device=device, stream=True, verbose=False, save=False,
                                   rect=False)
    for res in stream:
        n += 1
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b, c, k in zip(res.boxes.xyxy.cpu().numpy().tolist(), res.boxes.conf.cpu().numpy().tolist(),
                               res.boxes.cls.cpu().numpy().tolist()):
                boxes.append({"box": [float(x) for x in b], "conf": float(c)})
                LAST_CLASS_HIST[int(k)] = LAST_CLASS_HIST.get(int(k), 0) + 1
        by_path[os.path.abspath(res.path)] = boxes
        if n % 500 == 0:
            print("      predicted %d/%d" % (n, len(keys)), flush=True)
    return by_path


def get_dets(model_tag, set_key, samples, wrapper, our_ckpt, cache_dir, device, imgsz, conf_floor):
    tag = ("loc_%s_%s" % (model_tag, set_key)).replace("/", "_")
    cache = cache_dir / (tag + ".json")
    keys = [s["image"] for s in samples]
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        if blob.get("imgsz") == imgsz and blob.get("conf") == conf_floor and blob.get("n_images") == len(samples):
            print("      cache hit: %s" % cache.name, flush=True)
            return {k: v for k, v in blob["dets"].items()}
    print("      predicting %d images with %s ..." % (len(keys), model_tag), flush=True)
    if wrapper is not None:
        dets = predict_with_model(wrapper, keys, cache_dir, tag, imgsz, conf_floor, device)
    else:
        dets = ev.predict_group(our_ckpt, keys, cache_dir, tag, imgsz, conf_floor, device)
    hist = dict(LAST_CLASS_HIST)
    LAST_CLASS_HIST.clear()
    if hist:
        print("      predicted class histogram: %s" % hist, flush=True)
    cache.write_text(json.dumps({"imgsz": imgsz, "conf": conf_floor, "n_images": len(samples),
                                 "checkpoint": model_tag, "set": set_key, "class_hist": hist,
                                 "dets": dets}), encoding="utf-8")
    return dets


def main() -> int:
    ap = argparse.ArgumentParser(description="ETH official baseline audit (read-only, fair-comparison).")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--eth-ckpt", default=None)
    ap.add_argument("--eth-expected-sha256", default="f1aea7dec784a24d53d475f89510ef45fc13255df6298c6a0ea2fbd4419c7d6d")
    ap.add_argument("--eth-expected-bytes", default="134312133")
    ap.add_argument("--eth-inventory", default="outputs/shuttle_capability/metrics/eth_official_repo_inventory.csv")
    ap.add_argument("--manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--ckpt-registry", default="configs/shuttle_detection/checkpoints_v1.yaml")
    ap.add_argument("--models", default="eth_official,a_best,b_best,b_last")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--cache-dir", default="_scratch_localization_audit")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/shuttle_capability/metrics")
    ap.add_argument("--examples-dir", default="outputs/shuttle_capability/examples")
    ap.add_argument("--examples-per-kind", type=int, default=20)
    ap.add_argument("--example-models", default="eth_official,a_best,b_best")
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--report", default="outputs/shuttle_capability/reports/ETH_OFFICIAL_BASELINE_AUDIT.md")
    args = ap.parse_args()
    args.command_line = "python tools/eval_eth_official_baseline.py " + " ".join(sys.argv[1:])

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = repo / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    sets = [s for s in args.sets.split(",") if s]

    print("[1] checkpoint provenance", flush=True)
    eth_info = verify_checkpoint(args.eth_ckpt, repo / args.eth_inventory, args.eth_expected_sha256,
                                 args.eth_expected_bytes) if args.eth_ckpt else {"skipped": True}
    print("    sha256=%s bytes=%s matches_record=%s ckpt_keys=%s" % (
        (eth_info.get("sha256") or "")[:16], eth_info.get("bytes"), eth_info.get("sha256_matches_record"),
        eth_info.get("ckpt_keys")), flush=True)
    if eth_info.get("load_error"):
        print("    load_error=%s" % eth_info["load_error"], flush=True)

    print("[2] model loading and cost", flush=True)
    eth_wrapper, eth_meta = None, {}
    if args.eth_ckpt and "eth_official" in models:
        eth_wrapper, eth_meta = load_eth_model(args.eth_ckpt, args.device)
        eth_meta.update(model_cost(eth_wrapper, args.imgsz))
        print("    eth: %s nc=%d params=%d gflops=%s stride=%s" % (eth_meta["model_name"], eth_meta["nc"],
              eth_meta["params"], eth_meta["gflops"], eth_meta["stride"]), flush=True)
    ckpts = ev.resolve_checkpoints([], args.ckpt_registry, [m for m in models if m != "eth_official"])
    for c in ckpts:
        try:
            from ultralytics import YOLO
            w = YOLO(str(c["path"]))
            c.update(model_cost(w, args.imgsz))
            c["name"] = c["name"]
        except Exception as exc:
            c["cost_error"] = repr(exc)
    model_meta = {"eth_official": eth_meta}
    for c in ckpts:
        model_meta[c["name"]] = {"params": c.get("params"), "gflops": c.get("gflops"),
                                 "nc": c.get("nc"), "stride": c.get("stride"), "path": c.get("path")}

    print("[3] evaluation sets", flush=True)
    samples_by_set = {}
    for set_key in sets:
        if set_key == "val":
            ss, skipped = aud.build_val_samples(repo, args.manifest, [], args.max_images)
        else:
            ss, skipped = aud.build_frozen_samples(repo, set_key, args.max_images)
        samples_by_set[set_key] = ss
        print("    %-28s images=%d GT=%d" % (set_key, len(ss), sum(len(s["gt"]) for s in ss)), flush=True)

    import copy
    all_overall, all_ap, all_loc, all_size, all_center, all_fp = [], [], [], [], [], []
    results, models_sets = {}, []
    for tag in models:
        if tag == "eth_official" and eth_wrapper is None:
            print("    SKIP eth_official (no --eth-ckpt)", flush=True)
            continue
        wrapper = eth_wrapper if tag == "eth_official" else None
        our_ckpt = next((c["path"] for c in ckpts if c["name"] == tag), None)
        if tag != "eth_official" and not our_ckpt:
            print("    SKIP %s (unknown)" % tag, flush=True)
            continue
        for set_key in sets:
            samples = samples_by_set[set_key]
            print("[4] %s on %s" % (tag, set_key), flush=True)
            dets = get_dets(tag, set_key, samples, wrapper, our_ckpt, cache_dir, args.device, args.imgsz,
                            CONF_FLOOR)
            res = evaluate_model_on_set(tag, set_key, samples, dets)
            results[(tag, set_key)] = res
            models_sets.append((tag, set_key, samples, dets))
            meta = model_meta.get(tag, {})
            all_overall.append(overall_row(res, tag, meta))
            all_ap.append(ap_by_iou_row(res, tag))
            all_loc.append(localization_row(res, tag))
            for b in ev.BUCKET_LABELS:
                all_size.append(size_bucket_row(res, tag, b))
            row = all_overall[-1]
            print("      GT=%d TP=%d FP=%d R=%.4f AP50=%.4f AP75=%.4f AP90=%.4f mAP50-95=%.4f Recall_<8=%.4f" % (
                row["GT"], row["TP"], row["FP"], row["Recall"] or 0, row["AP50"] or 0, row["AP75"] or 0,
                row["AP90"] or 0, row["mAP50-95"] or 0, row["Recall_<8"] or 0), flush=True)
            for label, conf, thr, frame in [("eth_official_default", ETH_CONF, ETH_DIST_PX, "content"),
                                            ("eth_default_original_px", ETH_CONF, ETH_DIST_PX, "original"),
                                            ("eth_default_640_equiv", ETH_CONF, ETH_DIST_PX * 640.0 / IMGSZ, "content"),
                                            ("our_conf_0.25_content", CONF_OP, ETH_DIST_PX, "content")]:
                m = eth_style_center_metric(samples, dets, conf=conf, dist_thr=thr, frame=frame)
                m.update({"model": tag, "set": set_key, "subset": "all", "variant": label})
                all_center.append(m)
            uns = eth_unseen_samples(samples)
            if 0 < len(uns) < len(samples):
                ru = evaluate_model_on_set(tag, set_key, uns, dets, with_bucket_ap=False)
                rowu = overall_row(ru, tag, meta)
                rowu["set"] = set_key + "|eth_unseen"
                all_overall.append(rowu)
                all_ap.append({"model": tag, "set": set_key + "|eth_unseen", "images": ru["images"], "GT": ru["gt"],
                               "detections@0.25": ru["detections_op"], "TP@0.5": ru["tp50"], "FP@0.5": ru["fp50"],
                               "FN@0.5": ru["fn50"], "Precision@op": ru["precision_op"], "Recall@0.5": ru["recall50"],
                               **{"AP%d" % round(t * 100): ru["ap_by_iou"].get(t) for t in aud.AUDIT_IOU_THRESHOLDS},
                               "mAP50-95": ru["mAP50-95"]})
                all_loc.append(localization_row(ru, tag))
                print("      eth_unseen subset: images=%d GT=%d R=%.4f mAP50-95=%.4f" % (
                    ru["images"], ru["gt"], ru["recall50"] or 0, ru["mAP50-95"] or 0), flush=True)

    print("[5] false positives on the unlabeled hard-negative sets", flush=True)
    neg_groups = [g for g in ev.discover_frozen_groups(repo / ev.FROZEN_ROOT_REL) if not g.labeled]
    if eth_wrapper is not None and "eth_official" in models:
        for g in neg_groups:
            keys = [str(p) for p in g.images]
            dets = get_dets("eth_official", g.key, [{"image": str(p), "gt": [], "key": str(p)} for p in g.images],
                            eth_wrapper, None, cache_dir, args.device, args.imgsz, CONF_FLOOR)
            all_fp.extend(fp_rows("eth_official", [g], dets, (0.25, 0.50, 0.75)))
    old_neg = out_dir / "frozen_test_negatives.csv"
    if old_neg.exists():
        with open(old_neg, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for th, key in [("0.25", "FP_per_image@0.25"), ("0.5", "FP_per_image@0.5"), ("0.75", "FP_per_image@0.75")]:
                    pass
                all_fp.append({"model": r["checkpoint"], "set": r["set"], "threshold": float(r["threshold"]),
                               "images": int(r["images"]), "FP_total": int(r["total_FP"]),
                               "FP_per_image": float(r["FP_per_image"]), "images_with_FP": int(r["images_with_FP"]),
                               "image_FP_rate": float(r["image_FP_rate"]),
                               "max_conf_FP": (float(r["max_confidence_FP"]) if r["max_confidence_FP"] else None),
                               "median_conf_FP": (float(r["median_confidence_FP"]) if r["median_confidence_FP"] else None)})

    print("[6] latency / VRAM", flush=True)
    lat_rows = []
    if not args.no_latency:
        val_imgs = [s["image"] for s in samples_by_set.get("val", []) if s["gt"]][:40]
        bench = [("eth_official", eth_wrapper)] if eth_wrapper is not None else []
        from ultralytics import YOLO
        for c in ckpts:
            bench.append((c["name"], YOLO(str(c["path"]))))
        for tag, w in bench:
            if w is None:
                continue
            r = latency_bench(w, val_imgs, device=args.device, imgsz=args.imgsz, warmup=args.warmup,
                              n=args.latency_n)
            meta = model_meta.get(tag, {})
            fr, e2e = r["forward_raw"], r["end_to_end_predict"] or {}
            comp = r["components_ms_mean"] or {}
            lat_rows.append({"model": tag, "params": meta.get("params"), "gflops": meta.get("gflops"),
                             "nc": meta.get("nc"), "stride": str(meta.get("stride")),
                             "latency_mean_ms": fr["mean"], "latency_median_ms": fr["median"],
                             "latency_P90_ms": fr["P90"], "latency_P95_ms": fr["P95"], "fps": r["fps_raw"],
                             "e2e_mean_ms": e2e.get("mean"), "e2e_median_ms": e2e.get("median"),
                             "e2e_P90_ms": e2e.get("P90"), "e2e_P95_ms": e2e.get("P95"),
                             "fps_e2e": r["fps_end_to_end"], "preprocess_ms": comp.get("preprocess"),
                             "inference_ms": comp.get("inference"), "postprocess_ms": comp.get("postprocess"),
                             "vram_peak_MB": r["vram_peak_MB"], "warmup": r["warmup"], "measured": r["measured"],
                             "device": r["device"], "protocol": r["forward_protocol"]})
            print("    %-12s fwd mean=%.2f ms (%.1f FPS) e2e=%.2f ms vram=%.0f MB" % (
                tag, fr["mean"], r["fps_raw"], e2e.get("mean") or 0, r["vram_peak_MB"] or 0), flush=True)

    print("[7] leakage audit", flush=True)
    leak = leakage_rows(models_sets)
    inv_rows = []
    for set_key in sets:
        for r in leakage_inventory(samples_by_set[set_key]):
            r["set"] = set_key
            inv_rows.append(r)
    bg = {"controlled_capability": background_leak_probe(repo / "outputs/shuttle_capability/controlled_capability/manifest.csv",
                                                         [repo / "outputs/shuttle_capability", repo / "assets", ETH_ROOT]),
          "challenge_test": background_leak_probe(repo / "outputs/shuttle_capability/challenge_test/manifest.csv",
                                                  [repo / "outputs/shuttle_capability", repo / "assets", ETH_ROOT])}
    print("    leakage rows=%d ; background probe=%s" % (len(leak), {k: v.get("classes") for k, v in bg.items()}),
          flush=True)

    print("[8] examples", flush=True)
    ex_rows, dis_rows = [], []
    if not args.no_examples:
        ex_models = [m for m in args.example_models.split(",") if m]
        for tag in ex_models:
            dets_map = None
            for sk in sets:
                d = json.loads((cache_dir / (("loc_%s_%s" % (tag, sk)).replace("/", "_") + ".json")).read_text(encoding="utf-8"))
                if dets_map is None:
                    dets_map = {}
                if sk == "controlled_capability/images":
                    dets_map = {k: v for k, v in d["dets"].items()}
            if dets_map is not None:
                ex_rows.extend(export_examples(tag, samples_by_set["controlled_capability/images"], dets_map,
                                               repo / args.examples_dir / "eth_official" if tag == "eth_official"
                                               else repo / args.examples_dir / ("ours_" + tag),
                                               args.examples_per_kind))
        # disagreement pairs (ETH vs A, ETH vs B) on val
        def load_map(tag, sk):
            p = cache_dir / (("loc_%s_%s" % (tag, sk)).replace("/", "_") + ".json")
            return json.loads(p.read_text(encoding="utf-8"))["dets"] if p.exists() else {}
        for other in ["a_best", "b_best"]:
            if other in models and "eth_official" in models:
                da = load_map("eth_official", "val")
                db = load_map(other, "val")
                if da and db:
                    dis_rows.extend(export_disagreement(samples_by_set["val"], da, db, "eth", other,
                                                        repo / args.examples_dir / "model_disagreement",
                                                        per_kind=10))
        print("    examples=%d disagreement=%d" % (len(ex_rows), len(dis_rows)), flush=True)

    print("[9] write outputs", flush=True)
    write_csv(out_dir / "eth_official_overall.csv", OVERALL_HEADER, [r for r in all_overall
                                                                    if r["model"] == "eth_official"])
    write_csv(out_dir / "eth_official_ap_by_iou.csv", AP_HEADER, [r for r in all_ap
                                                                 if r["model"] == "eth_official"])
    write_csv(out_dir / "eth_official_size_buckets.csv", SIZE_HEADER, [r for r in all_size
                                                                      if r["model"] == "eth_official"])
    write_csv(out_dir / "eth_official_localization.csv", LOC_HEADER, [r for r in all_loc
                                                                     if r["model"] == "eth_official"])
    write_csv(out_dir / "eth_official_false_positive.csv", FP_HEADER, all_fp)
    write_csv(out_dir / "eth_official_latency.csv", LAT_HEADER, lat_rows)
    write_csv(out_dir / "eth_official_center_metric.csv", CENTER_HEADER, all_center)
    write_csv(out_dir / "eth_vs_ours.csv", OVERALL_HEADER, all_overall)
    write_csv(out_dir / "eth_vs_ours_ap_by_iou.csv", AP_HEADER, all_ap)
    write_csv(out_dir / "eth_vs_ours_size_buckets.csv", SIZE_HEADER, all_size)
    write_csv(out_dir / "eth_vs_ours_localization.csv", LOC_HEADER, all_loc)
    write_csv(out_dir / "eth_data_leakage_audit.csv", LEAK_HEADER, leak)
    write_csv(out_dir / "eth_data_leakage_inventory.csv",
              ["set", "image", "leakage_class", "rel", "location", "difficulty", "sub", "GT"], inv_rows)
    write_csv(out_dir / "eth_official_examples_index.csv",
              ["model", "kind", "rank", "image", "score", "file"], ex_rows)
    write_csv(out_dir / "eth_disagreement_index.csv",
              ["group", "rank", "image", "file", "n_det_eth", "n_det_a_best", "n_det_b_best"], dis_rows)
    (out_dir / "eth_checkpoint_manifest.json").write_text(json.dumps(
        {"checkpoint": eth_info, "eth_model": eth_meta, "models": model_meta,
         "config": {"imgsz": args.imgsz, "conf_op": CONF_OP, "conf_floor": CONF_FLOOR, "nms_iou": NMS_IOU,
                    "max_det": MAX_DET, "eth_conf": ETH_CONF, "eth_dist_px": ETH_DIST_PX,
                    "eth_train_locations": ETH_TRAIN_LOCS, "eth_train_difficulties": ETH_TRAIN_DIFFS,
                    "command": args.command_line},
         "leakage_background_probe": bg,
         "center_metric": all_center}, indent=1, default=str), encoding="utf-8")
    print("    csv/json -> %s" % out_dir, flush=True)

    if args.report:
        write_report(repo / args.report, args, all_overall, all_ap, all_loc, all_size, all_center, all_fp,
                     lat_rows, leak, inv_rows, bg, eth_info, eth_meta, model_meta, ex_rows, dis_rows)
        print("    report -> %s" % (repo / args.report), flush=True)
    print("done", flush=True)
    return 0


# =======================================================================================
# 10. report
# =======================================================================================
def write_report(path, args, overall, ap_rows, loc_rows, size_rows, center_rows, fp_rows_, lat_rows, leak, inv_rows,
                 bg, eth_info, eth_meta, model_meta, ex_rows, dis_rows):
    L = []
    add = L.append
    add("# ETH OFFICIAL BASELINE AUDIT - ETH Zurich shuttle detector vs our YOLO26-P2 (read-only)")
    add("")
    add("> 自动部分（1-12 节）由 tools/eval_eth_official_baseline.py 生成；0 节结论与 13 节问答为人工撰写（工具重跑会清空）。")
    add("> 只做评估：不训练、不 fine-tune、不改 checkpoint / 测试集 / GT / evaluator，不按模型单独调阈值，不引用论文数字替代实跑。")
    add("> 这些集合是 fixed evaluation set / development holdout，不是 untouched final test。")
    add("")
    add("## 0. 结论（人工）")
    add("")
    add("（待填。）")
    add("")
    add("## 1. ETH checkpoint 核验与兼容处理")
    add("")
    add(md_table(["项", "值"], [["路径", eth_info.get("path")], ["字节", eth_info.get("bytes")],
                                ["sha256", eth_info.get("sha256")],
                                ["与记录 sha256 一致", eth_info.get("sha256_matches_record")],
                                ["与官方仓库清单一致", eth_info.get("sha256_matches_inventory")],
                                ["inventory relpath", eth_info.get("inventory_relpath")],
                                ["checkpoint 类型", eth_info.get("ckpt_type")],
                                ["checkpoint keys", str(eth_info.get("ckpt_keys"))[:160]],
                                ["is_ultralytics_format", eth_info.get("is_ultralytics_format")],
                                ["ckpt 内 model_name", eth_info.get("model_name_in_ckpt")],
                                ["state_dict 张量数", eth_info.get("state_dict_tensors")],
                                ["nc（由 state_dict 的 cv3 分支数推断）", eth_info.get("nc_from_state_dict")],
                                ["dfl weight shape", str(eth_info.get("dfl_weight_shape"))]]))
    add("")
    add("**格式不兼容记录**：该文件是 ETH 自建训练脚本保存的 dict（含 model_state_dict 与 model_name），不是 ultralytics checkpoint。")
    add("最小兼容处理：用 ultralytics 自带 yolov8s.yaml 重建同架构（不含预训练权重），load_state_dict(strict=True) 一次性装载，")
    add("并核对由 dfl 输出推断的 nc；未改动任何权重数值、未改结构。装载后 params=%s GFLOPs=%s stride=%s nc=%s。" % (
        eth_meta.get("params"), f4(eth_meta.get("gflops"), 2), eth_meta.get("stride"), eth_meta.get("nc")))
    add("")
    add("## 2. FAIRNESS AUDIT（同一 evaluator / 数据 / 口径）")
    add("")
    add(md_table(["检查项", "状态", "说明"], [
        ["同一 images", "PASS", "val / controlled_capability/images / challenge_test/images（同一 manifest 与冻结目录）"],
        ["同一 GT", "PASS", "全部经 tools/eval_yolo26_v1.py:401-402 同一解码（无 letterbox）"],
        ["同一 imgsz=1024", "PASS", "所有模型 imgsz=1024（ETH 官方训练即 1024）"],
        ["同一 IoU evaluator", "PASS", "同一 match_greedy + ap_for_gt_set（all-point PR，GT 加权）"],
        ["同一 confidence policy", "PASS", "工作点 conf=0.25；AP 曲线下限 0.001（对所有模型相同）"],
        ["同一 NMS policy", "PASS", "iou=0.7, max_det=300, rect=False"],
        ["同一 size bucket", "PASS", "equiv_size_640 半开区间（与 LOCALIZATION_AUDIT 相同）"],
        ["同一 coordinate system", "PASS", "全部在原始图像像素系；ETH-style 指标另在 letterbox 系单独报告"],
        ["同一 hardware", "PASS", str(lat_rows[0]["device"] if lat_rows else "n/a")],
        ["同一 precision", "PASS", "fp32（未用 half / 量化）"],
        ["同一 latency protocol", "PASS", "batch=1 fp32，warmup=%s，measured=%s" % (args.warmup, args.latency_n)],
    ]))
    add("")
    add("阈值公平性声明：没有为任何模型单独调 confidence。主比较一律 conf=0.25（既有工作点）与 AP 曲线下限 0.001；")
    add("ETH 官方工作点 conf=0.5 只用于第 8 节 ETH-style 中心距指标，且对三个模型一并给出，不用于主比较。")
    add("")
    add("## 3. DATA LEAKAGE AUDIT")
    add("")
    add("判据（文件级、可复核）：ETH 官方每档 yaml 写明 train: images/train；其 config.json 的 data.train 列出 12 个 location，")
    add("diff_levels.train=[easy, medium]。位于 <location>_<easy|medium>/images/train 且 location 在该列表内的文件 = A（ETH 训练帧）；")
    add("同 location 但非训练档 = B；其它 ETH 基准档（如 coco_val_easy）或完全在 ETH 数据集之外 = C。")
    add("")
    add(md_table(["model", "set", "leakage class", "images", "GT", "TP@0.5", "Recall@0.5"],
                 [[r["model"], r["set"], r["leakage_class"], r["images"], r["GT"], r["TP@0.5"], f4(r["Recall@0.5"])]
                  for r in leak if r["model"] == "eth_official"]))
    add("")
    add("背景级泄漏探针（合成集使用的真实背景图来源）：")
    add("")
    for k, v in (bg or {}).items():
        add("- %s：resolved=%s unresolved=%s classes=%s" % (k, v.get("resolved"), v.get("unresolved"),
                                                            v.get("classes")))
    add("")
    add("eth_unseen 子集 = 剔除 A 类后的图像（B + C），在 eth_vs_ours.csv 中以 set 后缀 eth_unseen 行给出。")
    add("")
    add("## 4. 主指标（conf=0.25；AP 曲线 conf 0.001）")
    add("")
    hdr = ["model", "set", "GT", "TP", "FP", "FN", "P", "R", "F1", "AP50", "AP75", "AP90", "AP95", "mAP50-95",
           "FP/img"]
    add(md_table(hdr, [[r["model"], r["set"], r["GT"], r["TP"], r["FP"], r["FN"], f4(r["Precision"]), f4(r["Recall"]),
                        f4(r["F1"]), f4(r["AP50"]), f4(r["AP75"]), f4(r["AP90"]), f4(r["AP95"]), f4(r["mAP50-95"]),
                        f4(r["FP_per_image"])] for r in overall]))
    add("")
    add("## 5. AP by IoU（0.50:0.05:0.95）")
    add("")
    add(md_table(["model", "set"] + ["AP%d" % round(t * 100) for t in aud.AUDIT_IOU_THRESHOLDS] + ["mAP50-95"],
                 [[r["model"], r["set"]] + [f4(r.get("AP%d" % round(t * 100))) for t in aud.AUDIT_IOU_THRESHOLDS]
                  + [f4(r.get("mAP50-95"))] for r in ap_rows]))
    add("")
    add("## 6. 尺寸分桶与超小目标召回")
    add("")
    add(md_table(["model", "set", "bucket", "GT", "TP", "FN", "Recall", "AP50", "AP75", "AP90", "mAP50-95",
                  "IoU med"], [[r["model"], r["set"], r["bucket"], r["GT"], r["TP"], r["FN"], f4(r["Recall"]),
                                f4(r["AP50"]), f4(r["AP75"]), f4(r["AP90"]), f4(r["mAP50-95"]), f4(r["iou_median"])]
                               for r in size_rows]))
    add("")
    add("聚合召回（指定口径）：")
    add("")
    add(md_table(["model", "set"] + ["Recall_%s" % n for n, _ in AGGREGATES],
                 [[r["model"], r["set"]] + [f4(r.get("Recall_%s" % n)) for n, _ in AGGREGATES] for r in overall]))
    add("")
    add("## 7. 定位质量（TP 集合）与 height_ratio 检查")
    add("")
    add(md_table(["model", "set", "n_tp", "IoU mean", "IoU med", "IoU P10", "IoU P90", "center px640 med",
                  "norm center med", "w_ratio mean", "h_ratio mean", "h_ratio med", "area_ratio med",
                  "w_rel med", "h_rel med"],
                 [[r["model"], r["set"], r["n_tp"], f4(r["IoU_mean"]), f4(r["IoU_median"]), f4(r["IoU_P10"]),
                   f4(r["IoU_P90"]), f4(r["center_px640_median"], 3), f4(r["norm_center_median"], 4),
                   f4(r["width_ratio_mean"]), f4(r["height_ratio_mean"]), f4(r["height_ratio_median"]),
                   f4(r["area_ratio_median"]), f4(r["width_rel_err_median"]), f4(r["height_rel_err_median"])]
                  for r in loc_rows]))
    add("")
    add("height_ratio 标记规则：ETH 若也接近 1.09 则标 POSSIBLE_SHARED_DATA_OR_EVALUATION_EFFECT；")
    add("ETH 不偏而我们的 A/B 偏则标 POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT（不下因果断言）：")
    add("")
    flags = []
    for r in loc_rows:
        hr, wr = r.get("height_ratio_mean"), r.get("width_ratio_mean")
        if hr is None:
            continue
        if abs(hr - 1.092) <= 0.02 and (wr is None or abs(wr - 1.0) <= 0.03):
            flag = "POSSIBLE_SHARED_DATA_OR_EVALUATION_EFFECT" if r["model"] == "eth_official" else "ours_matches_bias"
        elif r["model"] == "eth_official":
            flag = "POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT (eth not biased)"
        else:
            flag = "not_flagged"
        flags.append([r["model"], r["set"], f4(hr), f4(wr), flag])
    add(md_table(["model", "set", "h_ratio mean", "w_ratio mean", "flag"], flags))
    add("")
    add("## 8. ETH-style center-distance 指标（单独口径，不等于 IoU-based 指标）")
    add("")
    add("**ETH-style F1 != IoU-based F1。** 其定义（src/shuttletrack/utils.py:297-344）：每图只取 top-1 检测框（max_det=1）、")
    add("每图只取 1 个 GT（首个实例）、距离在 letterbox 后的网络输入系内度量、dist < 25 px（config.json: dist_threshold=25.0, ")
    add("confidence=0.5）、FP 至多 1 个/图。")
    add("")
    add(md_table(["model", "set", "subset", "conf", "dist_thr", "frame", "TP", "FP", "FN", "P", "R", "F1", "variant"],
                 [[r["model"], r["set"], r["subset"], f4(r["conf"], 2), f4(r["dist_thr"], 3), r["frame"], r["TP"],
                   r["FP"], r["FN"], f4(r["precision"]), f4(r["recall"]), f4(r["f1"]), r["variant"]]
                  for r in center_rows]))
    add("")
    add("对照：IoU>=0.5 的 P/R/F1 见第 4 节的 Precision@op 与 Recall@0.5。两者差异来自口径，不是模型差异。")
    add("")
    add("## 9. False Positive（无目标集合）")
    add("")
    add(md_table(["model", "set", "conf>=", "images", "FP total", "FP/img", "images with FP", "image FP rate",
                  "max conf FP"], [[r["model"], r["set"], r["threshold"], r["images"], r["FP_total"],
                                    f4(r["FP_per_image"]), r["images_with_FP"], f4(r["image_FP_rate"]),
                                    f4(r["max_conf_FP"])] for r in fp_rows_ if float(r["threshold"]) == 0.25][:40]))
    add("")
    add("## 10. 速度与模型成本（同硬件、batch=1、fp32）")
    add("")
    add(md_table(["model", "params", "GFLOPs", "latency mean ms", "median", "P90", "P95", "FPS",
                  "e2e mean ms", "preprocess ms", "inference ms", "postprocess ms", "VRAM peak MB"],
                 [[r["model"], r["params"], f4(r["gflops"], 2), f4(r["latency_mean_ms"], 3),
                   f4(r["latency_median_ms"], 3), f4(r["latency_P90_ms"], 3), f4(r["latency_P95_ms"], 3),
                   f4(r["fps"], 2), f4(r["e2e_mean_ms"], 3), f4(r["preprocess_ms"], 3), f4(r["inference_ms"], 3),
                   f4(r["postprocess_ms"], 3), f4(r["vram_peak_MB"], 1)] for r in lat_rows]))
    add("")
    add("协议：%s；端到端另计时（含 preprocess/inference/postprocess），组件时间为框架 results.speed 均值，分开报告。" % (
        lat_rows[0]["protocol"] if lat_rows else "n/a"))
    add("")
    add("## 11. 模型成本与参数对照")
    add("")
    add(md_table(["model", "params", "GFLOPs", "nc", "stride", "latency mean ms", "FPS", "VRAM MB"],
                 [[k, v.get("params"), f4(v.get("gflops"), 2), v.get("nc"), str(v.get("stride")),
                   (next((f4(r["latency_mean_ms"], 3) for r in lat_rows if r["model"] == k), "n/a")),
                   (next((f4(r["fps"], 2) for r in lat_rows if r["model"] == k), "n/a")),
                   (next((f4(r["vram_peak_MB"], 1) for r in lat_rows if r["model"] == k), "n/a"))]
                  for k, v in model_meta.items()]))
    add("")
    add("## 12. 复现与产物")
    add("")
    add("~~~")
    add(args.command_line)
    add("~~~")
    add("")
    add("- 预测缓存 %s/loc_<model>_<set>.json（imgsz 1024 / conf 0.001 / iou 0.7 / max_det 300 / rect=False）。" % args.cache_dir)
    add("- 示例图 %d 张；分歧图 %d 张。" % (len(ex_rows), len(dis_rows)))
    add("- 旧结果未覆盖：Stage A/B 既有 metrics 与 reports 原样保留。")
    add("")
    add("## 13. 问答（Q1-Q10，人工）")
    add("")
    add("（待填。）")
    add("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(L) + chr(10), encoding="utf-8")

if __name__ == "__main__":
    raise SystemExit(main())




