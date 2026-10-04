#!/usr/bin/env python3
"""probe_eth_on_3d_render.py - run the AUDITED ETH YOLOv8s on the Isaac SIM renders.

Spike only. It does not train, does not touch the network, and does NOT modify
tools/eval_eth_official_baseline.py - it imports that module's loader and constants verbatim so
the inference setup is provably the same one that produced the audit:
    IMGSZ=1024, NMS_IOU=0.7, MAX_DET=300, CONF_FLOOR=0.001, CONF_OP=0.25

Per image it records max_conf_anywhere, best_match_conf / iou / centre distance, and the
detected flags at 0.25 and 0.50, then aggregates per target size:
    size | N | TP@0.25 | Recall@0.25 | median best conf | median best IoU

STATUS: draft, syntax-checked only - the remote host holding the ETH checkpoint was unreachable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import eval_eth_official_baseline as eth  # noqa: E402  (import only; the file is never written to)
import eval_yolo26_v1 as ev  # noqa: E402

EXPECTED_SHA256 = "f1aea7dec784a24d53d475f89510ef45fc13255df6298c6a0ea2fbd4419c7d6d"
EXPECTED_BYTES = 134312133


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center_distance(a, b):
    ca = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def load_gt(label_path, W, H):
    if not os.path.exists(str(label_path)):
        return None
    txt = Path(label_path).read_text(encoding="utf-8").strip()
    if not txt:
        return None
    cx, cy, bw, bh = (float(v) for v in txt.split()[1:5])
    return [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default=str(REPO / "outputs/shuttle_capability/three_d_render_probe"))
    ap.add_argument("--ckpt", default="/home/T7/dgut/robot_sim/eth_official_code/shuttle_detection/runs/final-model/best.pt")
    ap.add_argument("--examples", default=str(REPO / "outputs/shuttle_capability/examples/3d_render_eth_probe"))
    ap.add_argument("--report", default=str(REPO / "outputs/shuttle_capability/reports/THREED_RENDER_ETH_PROBE.md"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--examples-per-size", type=int, default=5)
    ap.add_argument("--skip-sha", action="store_true", help="only for a re-run against the same verified file")
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    if not ckpt.exists():
        raise SystemExit("ETH checkpoint not found: " + str(ckpt))
    sha = sha256_file(ckpt)
    size_b = ckpt.stat().st_size
    print("ckpt {} bytes={} sha256={}".format(ckpt, size_b, sha), flush=True)
    if not a.skip_sha and (sha.lower() != EXPECTED_SHA256 or size_b != EXPECTED_BYTES):
        raise SystemExit("checkpoint is NOT the audited artifact - refusing to use it")

    probe = Path(a.probe_dir)
    records = json.loads((probe / "render_manifest.json").read_text(encoding="utf-8"))
    print("probe images:", len(records), flush=True)

    model, meta = eth.load_eth_model(ckpt, device=a.device)
    print("model meta:", json.dumps(meta)[:300], flush=True)

    rows = []
    for rec in records:
        img_path = probe / "images" / rec["file"]
        img = ev.imread_unicode(str(img_path)) if hasattr(ev, "imread_unicode") else None
        if img is None:
            from PIL import Image
            with Image.open(str(img_path)) as im:
                W, H = im.size
            bgr = np.asarray(Image.open(str(img_path)).convert("RGB"))[:, :, ::-1].copy()
        else:
            bgr = img
            H, W = bgr.shape[:2]
        gt = load_gt(probe / "labels" / (Path(rec["file"]).stem + ".txt"), W, H)
        res = model.predict(bgr, imgsz=eth.IMGSZ, conf=eth.CONF_FLOOR, iou=eth.NMS_IOU,
                            max_det=eth.MAX_DET, verbose=False, device=a.device)[0]
        boxes = res.boxes.xyxy.detach().cpu().numpy() if res.boxes is not None and len(res.boxes) else np.zeros((0, 4))
        confs = res.boxes.conf.detach().cpu().numpy() if res.boxes is not None and len(res.boxes) else np.zeros((0,))
        best_conf, best_iou, best_cd = None, None, None
        for b, c in zip(boxes, confs):
            if gt is None:
                continue
            v = iou_xyxy(list(b), gt)
            if best_iou is None or v > best_iou:
                best_iou, best_conf, best_cd = v, float(c), center_distance(list(b), gt)
        rows.append({
            "file": rec["file"], "target_equiv640": rec["target_equiv640"],
            "measured_equiv640": rec.get("measured_equiv640"),
            "gt_bbox_px": [round(float(v), 2) for v in gt] if gt else None,
            "max_conf_anywhere": float(confs.max()) if confs.size else 0.0,
            "n_pred": int(confs.size),
            "best_match_conf": best_conf, "best_match_iou": (round(best_iou, 4) if best_iou is not None else None),
            "best_match_center_distance": (round(best_cd, 2) if best_cd is not None else None),
            "detected_conf025": bool(best_iou is not None and best_iou >= 0.5 and best_conf >= eth.CONF_OP),
            "detected_conf050": bool(best_iou is not None and best_iou >= 0.5 and best_conf >= 0.50),
            "any_conf025": bool(confs.size and confs.max() >= eth.CONF_OP),
            "any_conf050": bool(confs.size and confs.max() >= 0.50),
        })
        print("  {:<22} tgt={:<5} maxconf={:.4f} bestconf={} iou={}".format(
              rec["file"], rec["target_equiv640"], rows[-1]["max_conf_anywhere"],
              ("%.4f" % best_conf) if best_conf is not None else "-",
              ("%.3f" % best_iou) if best_iou is not None else "-"), flush=True)

    out = probe / "detections.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)

    # ---- aggregate per size ----
    agg = []
    for size in sorted({r["target_equiv640"] for r in rows}):
        sel = [r for r in rows if r["target_equiv640"] == size]
        tp = sum(1 for r in sel if r["detected_conf025"])
        tp50 = sum(1 for r in sel if r["detected_conf050"])
        ious = sorted(r["best_match_iou"] for r in sel if r["best_match_iou"] is not None)
        cfs = sorted(r["best_match_conf"] for r in sel if r["best_match_conf"] is not None)
        med = lambda xs: (xs[len(xs) // 2] if xs else None)
        agg.append({"size_equiv640": size, "N": len(sel), "TP_conf025": tp,
                    "Recall_conf025": round(tp / len(sel), 4) if sel else None,
                    "TP_conf050": tp50, "Recall_conf050": round(tp50 / len(sel), 4) if sel else None,
                    "median_best_conf": med(cfs), "median_best_iou": med(ious),
                    "median_max_conf_anywhere": med(sorted(r["max_conf_anywhere"] for r in sel))})

    lines = ["| size (equiv640) | N | TP@0.25 | Recall@0.25 | TP@0.50 | Recall@0.50 | median best conf | median best IoU | median max conf anywhere |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in agg:
        lines.append("| {} px | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["size_equiv640"], r["N"], r["TP_conf025"], r["Recall_conf025"],
            r["TP_conf050"], r["Recall_conf050"],
            ("%.4f" % r["median_best_conf"]) if r["median_best_conf"] is not None else "n/a",
            ("%.3f" % r["median_best_iou"]) if r["median_best_iou"] is not None else "n/a",
            "%.4f" % r["median_max_conf_anywhere"]))
    table = "\n".join(lines)
    print(table, flush=True)

    rep = Path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join([
        "# 3D-render ETH probe - results",
        "",
        "ckpt: `{}`  bytes={}  sha256=`{}`".format(ckpt, size_b, sha),
        "inference: imgsz={} NMS iou={} max_det={} conf floor={} working point={}".format(
            eth.IMGSZ, eth.NMS_IOU, eth.MAX_DET, eth.CONF_FLOOR, eth.CONF_OP),
        "model: {}".format(json.dumps(meta)),
        "",
        table,
        "",
        "Per-image records: `{}`".format(out),
    ]), encoding="utf-8")
    print("wrote", rep, flush=True)


if __name__ == "__main__":
    main()