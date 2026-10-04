#!/usr/bin/env python3
"""Size-bucketed evaluation for the YOLO26 V1 shuttle detector.

Everything is bucketed by the *unified* equivalent size at 640 input:
    equiv_size_original = sqrt(w_px * h_px)          (original image pixels)
    scale_to_640        = 640 / max(W, H)            (letterbox scale)
    equiv_size_640      = equiv_size_original * scale_to_640
Never bucket by the pixel size at the training imgsz (1024), so 640/960/1024/1280 stay comparable.

Primary metric   : IoU >= 0.5, one-to-one greedy matching by descending confidence.
Secondary metric : center distance <= 25 px in ORIGINAL pixels (ETH paper convention), reported
                   in its own table and never mixed into the primary tables.
FP are NOT given a size bucket (plan A): only overall precision / FP are reported.
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
from dataclasses import dataclass, field
from pathlib import Path

BUCKETS = [(0.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 12.0), (12.0, 16.0),
           (16.0, 24.0), (24.0, 32.0), (32.0, 64.0), (64.0, float("inf"))]
BUCKET_LABELS = ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]
LOW_SAMPLE_GT = 30
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # 0.50 .. 0.95


def bucket_of(eq: float) -> str:
    """Half-open buckets [lo, hi): a value exactly on a boundary joins the upper bucket."""
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        if lo <= eq < hi:
            return lab
    return BUCKET_LABELS[-1]


def equiv_size_640(w_px: float, h_px: float, img_w: float, img_h: float) -> float:
    scale = 640.0 / max(img_w, img_h)
    return math.sqrt(max(w_px, 0.0) * max(h_px, 0.0)) * scale


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def center_dist(a, b) -> float:
    return math.hypot((a[0] + a[2]) / 2.0 - (b[0] + b[2]) / 2.0,
                      (a[1] + a[3]) / 2.0 - (b[1] + b[3]) / 2.0)


def match_greedy(gt_boxes, dets, thr, metric="iou"):
    """One-to-one greedy matching, detections taken in descending confidence.

    gt_boxes : list of boxes
    dets     : list of (conf, box)
    metric   : "iou" -> iou >= thr ; "center" -> center distance <= thr
    Returns dict(gt_hit=[bool], det_tp=[bool], matches=[(det_i, gt_i)]).
    A GT matches at most one detection and a detection matches at most one GT.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][0])
    gt_hit = [False] * len(gt_boxes)
    det_tp = [False] * len(dets)
    matches = []
    for di in order:
        box = dets[di][1]
        best_v, best_j = None, -1
        for gj, gbox in enumerate(gt_boxes):
            if gt_hit[gj]:
                continue
            if metric == "iou":
                v = iou_xyxy(box, gbox)
                ok = v >= thr
            else:
                v = -center_dist(box, gbox)
                ok = center_dist(box, gbox) <= thr
            if ok and (best_v is None or v > best_v):
                best_v, best_j = v, gj
        if best_j >= 0:
            gt_hit[best_j] = True
            det_tp[di] = True
            matches.append((di, best_j))
    return {"gt_hit": gt_hit, "det_tp": det_tp, "matches": matches}


def ap_for_gt_set(gt_boxes, dets, iou_thr):
    """Average precision for one GT set against ALL detections (standard PR-curve, all-point)."""
    npos = len(gt_boxes)
    if npos == 0:
        return None
    m = match_greedy(gt_boxes, dets, iou_thr, metric="iou")
    order = sorted(range(len(dets)), key=lambda i: -dets[i][0])
    tp, fp = [], []
    for di in order:
        if m["det_tp"][di]:
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    ctp, cfp = 0.0, 0.0
    rec, prec = [], []
    for t, f in zip(tp, fp):
        ctp += t
        cfp += f
        rec.append(ctp / npos)
        prec.append(ctp / max(ctp + cfp, 1e-9))
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_samples(samples, conf_op=0.25, center_thr=25.0, iou_thr=0.5):
    """samples: list of {"key", "gt": [{"box", "bucket"}], "dets": [{"box", "conf"}]}."""
    per_bucket = {lab: {"gt": 0, "tp": 0, "fn": 0, "ap50": None, "ap5095": None} for lab in BUCKET_LABELS}
    overall = {"gt": 0, "tp": 0, "fp": 0, "fn": 0}
    sec = {"gt": 0, "tp": 0, "fn": 0}                 # secondary: center distance, overall
    sec_bucket = {lab: {"gt": 0, "tp": 0, "fn": 0} for lab in BUCKET_LABELS}
    ap_acc = {lab: [] for lab in BUCKET_LABELS}       # per-bucket AP50-95 accumulation
    overall_ap = []

    for s in samples:
        gts = s["gt"]
        dets_all = [(d["conf"], d["box"]) for d in s["dets"]]
        dets_op = [(c, b) for c, b in dets_all if c >= conf_op]
        # ---- primary: per-bucket TP/FN at the operating confidence ----
        by_bucket = {}
        for g in gts:
            by_bucket.setdefault(g["bucket"], []).append(g["box"])
        matched_det_idx = set()
        for lab, boxes in by_bucket.items():
            m = match_greedy(boxes, dets_op, iou_thr, metric="iou")
            per_bucket[lab]["gt"] += len(boxes)
            per_bucket[lab]["tp"] += sum(1 for x in m["gt_hit"] if x)
            per_bucket[lab]["fn"] += sum(1 for x in m["gt_hit"] if not x)
            for di, _gj in m["matches"]:
                matched_det_idx.add((id(s), di))
        overall["gt"] += len(gts)
        overall_tp = 0
        if gts:
            m_all = match_greedy([g["box"] for g in gts], dets_op, iou_thr, metric="iou")
            overall_tp = sum(1 for x in m_all["gt_hit"] if x)
        else:
            m_all = {"gt_hit": []}
        overall["tp"] += overall_tp
        overall["fn"] += len(gts) - overall_tp
        overall["fp"] += max(0, len(dets_op) - overall_tp)
        # ---- AP per bucket (GT set = that bucket; detections = all, standard PR curve) ----
        for lab in BUCKET_LABELS:
            boxes = by_bucket.get(lab, [])
            if not boxes:
                continue
            a50 = ap_for_gt_set(boxes, dets_all, 0.5)
            aps = [ap_for_gt_set(boxes, dets_all, t) for t in IOU_THRESHOLDS]
            aps = [a for a in aps if a is not None]
            ap_acc[lab].append((a50, sum(aps) / len(aps) if aps else None, len(boxes)))
        if gts:
            gboxes = [g["box"] for g in gts]
            o50 = ap_for_gt_set(gboxes, dets_all, 0.5)
            ops = [ap_for_gt_set(gboxes, dets_all, t) for t in IOU_THRESHOLDS]
            ops = [a for a in ops if a is not None]
            overall_ap.append((o50, sum(ops) / len(ops) if ops else None, len(gboxes)))
        # ---- secondary: center distance (its own table) ----
        if gts:
            m_sec = match_greedy([g["box"] for g in gts], dets_op, center_thr, metric="center")
            sec["gt"] += len(gts)
            sec["tp"] += sum(1 for x in m_sec["gt_hit"] if x)
            sec["fn"] += sum(1 for x in m_sec["gt_hit"] if not x)
            for gj, g in enumerate(gts):
                sec_bucket[g["bucket"]]["gt"] += 1
                sec_bucket[g["bucket"]]["tp"] += 1 if m_sec["gt_hit"][gj] else 0
                sec_bucket[g["bucket"]]["fn"] += 0 if m_sec["gt_hit"][gj] else 1
        for lab in BUCKET_LABELS:
            per_bucket[lab]["gt"] = per_bucket[lab]["gt"]

    # weighted (GT-count) bucket averages for AP: micro-style over GT
    for lab in BUCKET_LABELS:
        acc = ap_acc[lab]
        if acc:
            n = sum(a[2] for a in acc)
            per_bucket[lab]["ap50"] = sum((a[0] or 0.0) * a[2] for a in acc) / n
            per_bucket[lab]["ap5095"] = sum((a[1] or 0.0) * a[2] for a in acc) / n
    overall_metrics = None
    if overall_ap:
        n = sum(a[2] for a in overall_ap)
        overall_metrics = {
            "mAP50": sum((a[0] or 0.0) * a[2] for a in overall_ap) / n,
            "mAP50-95": sum((a[1] or 0.0) * a[2] for a in overall_ap) / n,
        }

    def _pr(tp, gt, fp=None):
        recall = tp / gt if gt else None
        if fp is None:
            prec = None
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else None
        f1 = (2 * prec * recall / (prec + recall)) if (prec is not None and recall is not None and (prec + recall) > 0) else None
        return prec, recall, f1

    for lab in BUCKET_LABELS:
        d = per_bucket[lab]
        d["precision"] = None            # plan A: FP are not bucketed, so no per-bucket precision
        d["recall"] = d["tp"] / d["gt"] if d["gt"] else None
        d["f1"] = None
        d["low_sample"] = bool(d["gt"] and d["gt"] < LOW_SAMPLE_GT)

    prec, recall, f1 = _pr(overall["tp"], overall["gt"], overall["fp"])
    overall["precision"], overall["recall"], overall["f1"] = prec, recall, f1
    if overall_metrics:
        overall["mAP50"] = overall_metrics["mAP50"]
        overall["mAP50-95"] = overall_metrics["mAP50-95"]
    else:
        overall["mAP50"] = None
        overall["mAP50-95"] = None

    def combined(labels):
        tp = sum(per_bucket[l]["tp"] for l in labels)
        gt = sum(per_bucket[l]["gt"] for l in labels)
        return {"tp": tp, "gt": gt, "recall": (tp / gt) if gt else None}

    sec["recall"] = sec["tp"] / sec["gt"] if sec["gt"] else None
    for lab in BUCKET_LABELS:
        d = sec_bucket[lab]
        d["recall"] = d["tp"] / d["gt"] if d["gt"] else None

    return {
        "buckets": per_bucket,
        "overall": overall,
        "combined_recall_6_16": combined(["6-8", "8-12", "12-16"]),
        "combined_recall_4_16": combined(["4-6", "6-8", "8-12", "12-16"]),
        "secondary_center_distance": {"overall": sec, "buckets": sec_bucket, "threshold_px_original": center_thr},
        "config": {"conf_op": conf_op, "iou_primary": iou_thr, "low_sample_gt": LOW_SAMPLE_GT},
    }


def load_manifest(path: Path, split: str, remaps):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("split") != split:
                continue
            rows.append(r)
    return rows


def remap(p: str, remaps) -> str:
    for a, b in remaps:
        if p.startswith(a):
            out = (b + p[len(a):]).replace("\\", "/")
            while "//" in out:
                out = out.replace("//", "/")
            return out
    return p


# -------------------------------------------------------------------------------------
# Frozen test sets (never used for training / tuning / model selection)
# -------------------------------------------------------------------------------------
FROZEN_ROOT_REL = "outputs/shuttle_capability"
FROZEN_SETS = [
    ("controlled_capability", "synthetic-controlled capability grid (labeled)"),
    ("challenge_test", "hard real-scene challenge set (labeled)"),
    ("synthetic_on_real_bg", "synthetic shuttles composited on real backgrounds (unlabeled)"),
    ("synthetic_3d", "3D-rendered shuttles on clean backgrounds (unlabeled)"),
    ("real_images", "real photos: backgrounds/ (no shuttle) + raw/ (unlabeled)"),
    ("real_video", "real video frames (unlabeled)"),
    ("real_match_frames", "real match frames (unlabeled)"),
    ("real_train", "real photos kept out of training (unlabeled)"),
]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
GROUP_DIR_HINTS = ("images", "frames", "backgrounds", "raw")
NEG_THRESHOLDS_DEFAULT = [0.25, 0.50, 0.75]


@dataclass
class FrozenGroup:
    """One image directory inside a frozen set (a set may have several, e.g. real_images/backgrounds)."""

    set_name: str
    group: str
    image_dir: Path
    label_dir: "Path | None"
    images: list = field(default_factory=list)
    labeled: bool = False

    @property
    def key(self) -> str:
        return self.set_name if not self.group else "%s/%s" % (self.set_name, self.group)

    @property
    def label_by_stem(self) -> dict:
        out = {}
        if self.label_dir is None:
            return out
        for p in sorted(self.label_dir.glob("*.txt")):
            out[p.stem] = p
        return out


def list_images(d: Path) -> list:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def pick_image_dirs(set_dir: Path) -> list:
    """Pick the image directories of a frozen set, deterministically.

    Prefers the well-known sub-directory names (images/frames/backgrounds/raw); if none exist, falls
    back to the set directory itself, and finally to any sub-directory that directly contains images.
    """
    hinted = [set_dir / h for h in GROUP_DIR_HINTS if (set_dir / h).is_dir() and list_images(set_dir / h)]
    if hinted:
        return hinted
    if list_images(set_dir):
        return [set_dir]
    found = []
    for sub in sorted(p for p in set_dir.iterdir() if p.is_dir()):
        if list_images(sub):
            found.append(sub)
    return found


def pick_label_dir(image_dir: Path, images: list) -> "Path | None":
    """Find the label directory for an image directory: <set>/labels, <image_dir>/labels, or beside."""
    stems = {p.stem for p in images}
    candidates = [image_dir.parent / "labels", image_dir / "labels", image_dir]
    for cand in candidates:
        if not cand.is_dir():
            continue
        if any(p.stem in stems for p in cand.glob("*.txt")):
            return cand
    return None


def discover_frozen_groups(root: Path, only=None) -> list:
    """Enumerate <root>/<set>/ image directories as FrozenGroup entries (labeled flag detected)."""
    groups = []
    for set_name, _desc in FROZEN_SETS:
        if only and set_name not in only:
            continue
        set_dir = root / set_name
        if not set_dir.is_dir():
            continue
        for image_dir in pick_image_dirs(set_dir):
            images = list_images(image_dir)
            if not images:
                continue
            group = "" if image_dir == set_dir else image_dir.name
            label_dir = pick_label_dir(image_dir, images)
            labeled = label_dir is not None
            groups.append(FrozenGroup(set_name=set_name, group=group, image_dir=image_dir,
                                      label_dir=label_dir, images=images, labeled=labeled))
    return groups


def load_gt_boxes(image_path, label_path):
    """GT boxes for one image from a YOLO label file.

    Returns [] when there is no label file (hard negative) and None when the image itself is unreadable
    (caller decides whether to skip the sample).
    """
    if not label_path or not os.path.exists(str(label_path)):
        return []
    from PIL import Image
    try:
        with Image.open(str(image_path)) as im:
            W, H = im.size
    except Exception:
        return None
    out = []
    with open(str(label_path), encoding="utf-8", errors="replace") as f:
        for line in f.read().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            w_px, h_px = bw * W, bh * H
            eq = equiv_size_640(w_px, h_px, W, H)
            out.append({"box": [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H],
                        "eq640": eq, "bucket": bucket_of(eq)})
    return out


def percentile(xs, p):
    """Linear-interpolated percentile (same convention as tools/eval_hard_negative.py)."""
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def fp_stats(dets_per_image, thresholds):
    """False-positive statistics for all-negative sets: every detection is a false positive."""
    out = {}
    n_img = len(dets_per_image)
    for th in thresholds:
        fps, imgs_with = [], 0
        for dets in dets_per_image:
            sel = [d["conf"] for d in dets if d["conf"] >= th]
            if sel:
                imgs_with += 1
                fps.extend(sel)
        out[str(th)] = {
            "total_images": n_img,
            "total_FP": len(fps),
            "FP_per_image": (len(fps) / n_img) if n_img else None,
            "images_with_FP": imgs_with,
            "image_FP_rate": (imgs_with / n_img) if n_img else None,
            "max_confidence_FP": max(fps) if fps else None,
            "mean_confidence_FP": statistics.fmean(fps) if fps else None,
            "median_confidence_FP": statistics.median(fps) if fps else None,
            "P95_FP_confidence": percentile(fps, 0.95) if fps else None,
        }
    return out


def load_checkpoint_registry(path) -> dict:
    """Read the unified checkpoint registry: {name: {path, run, epoch, role, note}}."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        doc = yaml.safe_load(text) or {}
    else:
        doc = json.loads(text)
    return doc.get("checkpoints", doc) or {}


def resolve_checkpoints(pairs, registry_path=None, names=None, exists=os.path.exists) -> list:
    """Unified checkpoint resolution used by every stage (A best / B best / B last / future C best).

    * pairs        : repeatable "NAME=PATH" from --ckpt (ad-hoc, wins on name clash)
    * registry_path: optional YAML/JSON registry file (--ckpt-registry)
    * names        : repeatable registry names (--ckpt-name), resolved in the given order
    Returns a list of dicts: {name, path, run, epoch, role, note}; entries whose path is missing on disk
    are returned too (caller skips them with a message) so a placeholder like "c_best" never crashes a run.
    """
    reg = load_checkpoint_registry(registry_path)
    out, seen = [], set()
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit("bad --ckpt (want NAME=PATH): %s" % pair)
        name, path = pair.split("=", 1)
        meta = dict(reg.get(name) or {})
        meta.update({"name": name, "path": path})
        out.append(meta)
        seen.add(name)
    for name in names or []:
        if name in seen:
            continue
        if name not in reg:
            raise SystemExit("unknown --ckpt-name %r (registry: %s)" % (name, ", ".join(sorted(reg))))
        meta = dict(reg[name] or {})
        meta["name"] = name
        if not meta.get("path"):
            print("SKIP %s: registry entry has no path yet (placeholder)" % name, flush=True)
            continue
        out.append(meta)
        seen.add(name)
    return out


def file_sha256(path, limit_bytes=None) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def predict_group(model_path, keys, out_dir, tag, imgsz, conf_floor, device) -> dict:
    """Run the model over `keys` (image paths) and return {abspath: [{box, conf}, ...]}."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    src_file = out_dir / ("%s_images.txt" % tag)
    src_file.write_text(chr(10).join(k.replace("\\", "/") for k in keys) + chr(10), encoding="utf-8")
    by_path = {}
    stream = model.predict(source=str(src_file), imgsz=imgsz, conf=conf_floor, iou=0.7,
                           max_det=300, device=device, stream=True, verbose=False, save=False)
    n = 0
    for res in stream:
        n += 1
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()
            for b, c in zip(xyxy, confs):
                boxes.append({"box": [float(x) for x in b], "conf": float(c)})
        by_path[os.path.abspath(res.path)] = boxes
        if n % 500 == 0:
            print("  predicted %d/%d" % (n, len(keys)), flush=True)
    return by_path


def run_frozen(args, repo: Path, out_dir: Path, remaps) -> int:
    """--split frozen_test: labeled sets -> Recall/AP/buckets; unlabeled sets -> FP/image + image FP rate."""
    only = [s for s in (args.frozen_sets or "").split(",") if s] or None
    neg_thresholds = [float(x) for x in (args.neg_thresholds or "").split(",") if x] or NEG_THRESHOLDS_DEFAULT
    ckpts = resolve_checkpoints(args.ckpt, args.ckpt_registry, args.ckpt_name)
    if not ckpts:
        raise SystemExit("no checkpoint: pass --ckpt NAME=PATH or --ckpt-name NAME (with --ckpt-registry)")
    root = repo / FROZEN_ROOT_REL
    groups = discover_frozen_groups(root, only=only)
    if args.max_images:
        for g in groups:
            g.images = g.images[: args.max_images]
    print("frozen groups: %d" % len(groups), flush=True)
    for g in groups:
        print("  %-34s labeled=%-5s images=%d" % (g.key, g.labeled, len(g.images)), flush=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo / ".yolo_cfg"))
    results = {}
    for meta in ckpts:
        name, path = meta["name"], meta.get("path")
        if not path or not os.path.exists(path):
            print("SKIP missing checkpoint: %s (%s)" % (name, path), flush=True)
            continue
        entry = {"checkpoint": {"name": name, "path": path, "imgsz": args.imgsz,
                                "conf_op": args.conf, "ap_conf": args.ap_conf,
                                "sha256": file_sha256(path), "bytes": os.path.getsize(path),
                                "run": meta.get("run"), "epoch": meta.get("epoch"),
                                "role": meta.get("role"), "note": meta.get("note")},
                 "split": "frozen_test", "sets": {}}
        for g in groups:
            keys = [str(p) for p in g.images]
            tag = ("frozen_%s_%s" % (name, g.key)).replace("/", "_")
            print("=== %s | %s | %d images ===" % (name, g.key, len(keys)), flush=True)
            by_path = predict_group(path, keys, out_dir, tag, args.imgsz, args.ap_conf, args.device)
            if g.labeled:
                samples, skipped = [], 0
                lb = g.label_by_stem
                for p in g.images:
                    gts = load_gt_boxes(p, lb.get(p.stem))
                    if gts is None:
                        skipped += 1
                        continue
                    samples.append({"key": str(p), "gt": gts,
                                    "dets": by_path.get(os.path.abspath(str(p)), [])})
                out = evaluate_samples(samples, conf_op=args.conf, center_thr=args.center_dist,
                                       iou_thr=args.iou)
                out["split"] = "frozen_test"
                out["set"] = g.key
                out["images"] = len(samples)
                out["skipped_unreadable"] = skipped
                out["labeled"] = True
                entry["sets"][g.key] = out
                ov = out["overall"]
                print("  %s: images=%d GT=%d TP=%d FP=%d FN=%d P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f" % (
                    g.key, len(samples), ov["gt"], ov["tp"], ov["fp"], ov["fn"],
                    ov["precision"] or 0.0, ov["recall"] or 0.0, ov["mAP50"] or 0.0,
                    ov["mAP50-95"] or 0.0), flush=True)
            else:
                dets = [by_path.get(os.path.abspath(str(p)), []) for p in g.images]
                stats = fp_stats(dets, neg_thresholds)
                entry["sets"][g.key] = {"labeled": False, "images": len(g.images),
                                        "negatives": stats, "thresholds": neg_thresholds,
                                        "ap_conf": args.ap_conf, "imgsz": args.imgsz}
                for th in neg_thresholds:
                    s = stats[str(th)]
                    print("  %s @%.2f: FP=%d FP/img=%.4f image_FP_rate=%.4f max_conf=%s" % (
                        g.key, th, s["total_FP"], s["FP_per_image"] or 0.0, s["image_FP_rate"] or 0.0,
                        "n/a" if s["max_confidence_FP"] is None else "%.4f" % s["max_confidence_FP"]),
                        flush=True)
        results[name] = entry

    (out_dir / "size_bucket_metrics_frozen_test.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8")

    # per-labeled-set JSON in the val schema, so tools/compare_size_buckets.py gives an N-way compare
    labeled_sets = [g.key for g in groups if g.labeled]
    for set_key in labeled_sets:
        payload = {}
        for name, entry in results.items():
            if set_key in entry["sets"]:
                payload[name] = entry["sets"][set_key]
        (out_dir / ("size_bucket_metrics_frozen_%s.json" % set_key.replace("/", "_"))).write_text(
            json.dumps(payload, indent=1, default=str), encoding="utf-8")

    # ---- unified matrix: one row per (checkpoint, set) ----
    matrix_path = out_dir / "frozen_test_matrix.csv"
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "set", "labeled", "images", "GT", "TP", "FP", "FN", "Precision", "Recall",
                    "F1", "mAP50", "mAP50-95", "Recall_6-16", "Recall_4-16", "FP_per_image@conf",
                    "image_FP_rate@conf", "FP_per_image@0.25", "FP_per_image@0.50", "FP_per_image@0.75",
                    "image_FP_rate@0.25", "image_FP_rate@0.50", "image_FP_rate@0.75"])
        for name, entry in results.items():
            for set_key, s in entry["sets"].items():
                if s.get("labeled"):
                    ov = s["overall"]
                    neg = {"0.25": None, "0.5": None, "0.75": None}
                    row = [name, set_key, True, s["images"], ov["gt"], ov["tp"], ov["fp"], ov["fn"],
                           ov["precision"], ov["recall"], ov["f1"], ov["mAP50"], ov["mAP50-95"],
                           s["combined_recall_6_16"]["recall"], s["combined_recall_4_16"]["recall"],
                           (ov["fp"] / s["images"]) if s["images"] else None,
                           (ov["fp"] / s["images"]) if s["images"] else None]
                else:
                    neg = s["negatives"]
                    row = [name, set_key, False, s["images"], "", "", neg["0.25"]["total_FP"], "", "", "", "", "", "", "", "",
                           neg["0.25"]["FP_per_image"], neg["0.25"]["image_FP_rate"],
                           neg.get("0.25", {}).get("FP_per_image"), neg.get("0.5", {}).get("FP_per_image"),
                           neg.get("0.75", {}).get("FP_per_image"),
                           neg.get("0.25", {}).get("image_FP_rate"), neg.get("0.5", {}).get("image_FP_rate"),
                           neg.get("0.75", {}).get("image_FP_rate")]
                w.writerow(["" if v is None else (round(v, 6) if isinstance(v, float) else v) for v in row])

    # ---- labeled buckets ----
    with open(out_dir / "frozen_test_labeled_buckets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "set", "bucket", "GT", "TP", "FN", "Recall", "AP50", "AP50-95", "low_sample"])
        for name, entry in results.items():
            for set_key, s in entry["sets"].items():
                if not s.get("labeled"):
                    continue
                for lab in BUCKET_LABELS:
                    d = s["buckets"][lab]
                    w.writerow([name, set_key, lab, d["gt"], d["tp"], d["fn"],
                                "" if d["recall"] is None else round(d["recall"], 6),
                                "" if d["ap50"] is None else round(d["ap50"], 6),
                                "" if d["ap5095"] is None else round(d["ap5095"], 6), d["low_sample"]])
                ov = s["overall"]
                w.writerow([name, set_key, "OVERALL", ov["gt"], ov["tp"], ov["fn"],
                            "" if ov["recall"] is None else round(ov["recall"], 6),
                            "" if ov["mAP50"] is None else round(ov["mAP50"], 6),
                            "" if ov["mAP50-95"] is None else round(ov["mAP50-95"], 6), ""])
                for label, key in (("COMBINED_6-16", "combined_recall_6_16"), ("COMBINED_4-16", "combined_recall_4_16")):
                    c = s[key]
                    w.writerow([name, set_key, label, c["gt"], c["tp"], c["gt"] - c["tp"],
                                "" if c["recall"] is None else round(c["recall"], 6), "", "", ""])

    # ---- negatives ----
    with open(out_dir / "frozen_test_negatives.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "set", "threshold", "images", "total_FP", "FP_per_image", "images_with_FP",
                    "image_FP_rate", "max_confidence_FP", "mean_confidence_FP", "median_confidence_FP",
                    "P95_FP_confidence"])
        for name, entry in results.items():
            for set_key, s in entry["sets"].items():
                if s.get("labeled"):
                    continue
                for th in s["thresholds"]:
                    d = s["negatives"][str(th)]
                    w.writerow([name, set_key, th, d["total_images"], d["total_FP"],
                                "" if d["FP_per_image"] is None else round(d["FP_per_image"], 6),
                                d["images_with_FP"],
                                "" if d["image_FP_rate"] is None else round(d["image_FP_rate"], 6),
                                "" if d["max_confidence_FP"] is None else round(d["max_confidence_FP"], 6),
                                "" if d["mean_confidence_FP"] is None else round(d["mean_confidence_FP"], 6),
                                "" if d["median_confidence_FP"] is None else round(d["median_confidence_FP"], 6),
                                "" if d["P95_FP_confidence"] is None else round(d["P95_FP_confidence"], 6)])

    # ---- checkpoint metadata ----
    with open(out_dir / "frozen_test_checkpoints.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "path", "sha256", "bytes", "run", "epoch", "role", "imgsz", "conf_op", "ap_conf"])
        for name, entry in results.items():
            c = entry["checkpoint"]
            w.writerow([name, c["path"], c["sha256"], c["bytes"], c["run"], c["epoch"], c["role"],
                        c["imgsz"], c["conf_op"], c["ap_conf"]])

    print("wrote", out_dir, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--split", default="val", choices=["val", "frozen_test"])
    ap.add_argument("--ckpt", action="append", default=[], help="NAME=PATH, repeatable (ad-hoc)")
    ap.add_argument("--ckpt-registry", default="configs/shuttle_detection/checkpoints_v1.yaml",
                    help="unified checkpoint registry (YAML/JSON); used by every stage")
    ap.add_argument("--ckpt-name", action="append", default=[],
                    help="registry name to evaluate, repeatable (a_best / b_best / b_last / c_best ...)")
    ap.add_argument("--frozen-sets", default="", help="comma list to restrict frozen_test sets (default: all)")
    ap.add_argument("--neg-thresholds", default="0.25,0.50,0.75",
                    help="confidence thresholds for unlabeled frozen sets (FP/image, image FP rate)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25, help="operating confidence for P/R/F1")
    ap.add_argument("--ap-conf", type=float, default=0.001, help="confidence floor for AP curves")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--center-dist", type=float, default=25.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--map", action="append", default=[])
    ap.add_argument("--out-dir", default="outputs/shuttle_capability/metrics")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    remaps = []
    for m in args.map:
        if "=" in m:
            a, b = m.split("=", 1)
            remaps.append((a, b))

    if args.split == "frozen_test":
        return run_frozen(args, repo, out_dir, remaps)

    rows = load_manifest(repo / args.manifest, args.split, remaps)
    if args.max_images:
        rows = rows[: args.max_images]
    print("images in split %s: %d" % (args.split, len(rows)), flush=True)

    # ---- build GT samples (per-box equiv_size_640 from the label file) ----
    from PIL import Image
    samples = []
    for r in rows:
        img = remap(r["image"], remaps)
        lab = remap(r["label"], remaps) if r.get("label") else ""
        gts = []
        if lab and os.path.exists(lab):
            try:
                with Image.open(img) as im:
                    W, H = im.size
            except Exception:
                continue
            with open(lab, encoding="utf-8", errors="replace") as f:
                for line in f.read().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = (float(x) for x in parts[1:5])
                    w_px, h_px = bw * W, bh * H
                    eq = equiv_size_640(w_px, h_px, W, H)
                    gts.append({"box": [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H],
                                "eq640": eq, "bucket": bucket_of(eq)})
        samples.append({"key": img, "gt": gts, "dets": [], "manifest": r})

    got = sum(len(s["gt"]) for s in samples)
    print("GT boxes: %d over %d images" % (got, len(samples)), flush=True)

    results = {}
    _metas = resolve_checkpoints(args.ckpt, args.ckpt_registry, args.ckpt_name)
    ckpts = [(m["name"], m["path"]) for m in _metas if m.get("path")]
    if not ckpts:
        raise SystemExit("no checkpoint: pass --ckpt NAME=PATH or --ckpt-name NAME (with --ckpt-registry)")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo / ".yolo_cfg"))
    from ultralytics import YOLO
    for name, path in ckpts:
        if not os.path.exists(path):
            print("SKIP missing checkpoint:", name, path, flush=True)
            continue
        print("=== predicting with %s (%s) ===" % (name, path), flush=True)
        model = YOLO(path)
        # A .txt source is streamed lazily by ultralytics (LoadImagesAndVideos).
        # A python list would use LoadPilAndNumpy, which loads EVERY image into RAM upfront
        # and blew up with cv2 OutOfMemoryError on a 4413-image val split.
        src_file = out_dir / ("%s_images.txt" % args.split)
        src_file.write_text(chr(10).join(s["key"].replace("\\", "/") for s in samples) + chr(10), encoding="utf-8")
        src = str(src_file)
        by_path = {}
        stream = model.predict(source=src, imgsz=args.imgsz, conf=args.ap_conf, iou=0.7,
                               max_det=300, device=args.device, stream=True, verbose=False, save=False)
        n = 0
        for res in stream:
            n += 1
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().tolist()
                confs = res.boxes.conf.cpu().numpy().tolist()
                for b, c in zip(xyxy, confs):
                    boxes.append({"box": [float(x) for x in b], "conf": float(c)})
            by_path[os.path.abspath(res.path)] = boxes
            if n % 500 == 0:
                print("  predicted %d/%d" % (n, len(samples)), flush=True)
        for s in samples:
            s["dets"] = by_path.get(os.path.abspath(s["key"]), [])
        out = evaluate_samples(samples, conf_op=args.conf, center_thr=args.center_dist, iou_thr=args.iou)
        out["checkpoint"] = {"name": name, "path": path, "imgsz": args.imgsz,
                             "conf_op": args.conf, "ap_conf": args.ap_conf}
        out["split"] = args.split
        out["images"] = len(samples)
        results[name] = out
        ov = out["overall"]
        print("  %s: GT=%d TP=%d FP=%d FN=%d P=%.4f R=%.4f mAP50=%.4f" % (
            name, ov["gt"], ov["tp"], ov["fp"], ov["fn"], ov["precision"] or 0.0, ov["recall"] or 0.0,
            ov["mAP50"] or 0.0), flush=True)

    (out_dir / ("size_bucket_metrics_%s.json" % args.split)).write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8")

    # ---- CSV: one row per (checkpoint, bucket) ----
    with open(out_dir / ("size_bucket_metrics_%s.csv" % args.split), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "bucket", "GT", "TP", "FN", "Recall", "AP50", "AP50-95", "low_sample"])
        for name, out in results.items():
            for lab in BUCKET_LABELS:
                d = out["buckets"][lab]
                w.writerow([name, lab, d["gt"], d["tp"], d["fn"],
                            "" if d["recall"] is None else round(d["recall"], 6),
                            "" if d["ap50"] is None else round(d["ap50"], 6),
                            "" if d["ap5095"] is None else round(d["ap5095"], 6), d["low_sample"]])
            ov = out["overall"]
            w.writerow([name, "OVERALL", ov["gt"], ov["tp"], ov["fn"],
                        "" if ov["recall"] is None else round(ov["recall"], 6),
                        "" if ov["mAP50"] is None else round(ov["mAP50"], 6),
                        "" if ov["mAP50-95"] is None else round(ov["mAP50-95"], 6), ""])
            for label, key in (("COMBINED_6-16", "combined_recall_6_16"), ("COMBINED_4-16", "combined_recall_4_16")):
                c = out[key]
                w.writerow([name, label, c["gt"], c["tp"], c["gt"] - c["tp"],
                            "" if c["recall"] is None else round(c["recall"], 6), "", "", ""])

    names = list(results.keys())
    if len(names) >= 2:
        a, b = names[0], names[1]
        with open(out_dir / ("size_bucket_checkpoint_compare_%s.csv" % args.split), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bucket", "GT", "%s_Recall" % a, "%s_Recall" % b, "delta_recall",
                        "%s_AP50" % a, "%s_AP50" % b, "delta_AP50"])
            for lab in BUCKET_LABELS + ["OVERALL", "COMBINED_6-16", "COMBINED_4-16"]:
                if lab in BUCKET_LABELS or lab == "OVERALL":
                    da, db = results[a]["buckets"].get(lab, {}), results[b]["buckets"].get(lab, {})
                    if lab == "OVERALL":
                        ra, rb = results[a]["overall"]["recall"], results[b]["overall"]["recall"]
                        pa, pb = results[a]["overall"]["mAP50"], results[b]["overall"]["mAP50"]
                        g = results[a]["overall"]["gt"]
                    else:
                        ra, rb = da.get("recall"), db.get("recall")
                        pa, pb = da.get("ap50"), db.get("ap50")
                        g = da.get("gt")
                else:
                    key = "combined_recall_6_16" if lab == "COMBINED_6-16" else "combined_recall_4_16"
                    ra, rb = results[a][key]["recall"], results[b][key]["recall"]
                    pa = pb = None
                    g = results[a][key]["gt"]
                w.writerow([lab, g,
                            "" if ra is None else round(ra, 6), "" if rb is None else round(rb, 6),
                            "" if (ra is None or rb is None) else round(rb - ra, 6),
                            "" if pa is None else round(pa, 6), "" if pb is None else round(pb, 6),
                            "" if (pa is None or pb is None) else round(pb - pa, 6)])

    print("wrote", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())