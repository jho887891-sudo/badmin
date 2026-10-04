#!/usr/bin/env python3
"""Hard-negative false-positive evaluation (diagnostic only).

Answers one question with numbers: on images that contain NO shuttlecock, how often does the model
hallucinate a shuttle? Reported per confidence threshold as FP/image and image-FP-rate, plus the
confidence distribution of the false positives, and the top-K highest-confidence FP images.

This tool never writes into a training run directory and never needs the GPU being trained on.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path


def percentile(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def collect_images(repo: Path, spec: str, manifest_rel: str):
    """spec: manifest:<source>:<split>  |  dir:<path>"""
    images = []
    if spec.startswith("manifest:"):
        _, source, split = spec.split(":", 2)
        with open(repo / manifest_rel, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("source") == source and r.get("split") == split:
                    images.append(r["image"])
    elif spec.startswith("dir:"):
        d = Path(spec[4:])
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append(str(p))
    else:
        raise SystemExit("bad --images spec: " + spec)
    return images


def label_for(image: str) -> Path:
    s = image.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return Path(s).with_suffix(".txt")


def has_positive_label(image: str) -> bool:
    lp = label_for(image)
    if not lp.exists():
        return False
    for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(line.split()) >= 5:
            return True
    return False


def draw_and_save(image_path: str, boxes, out_path: Path, title: str):
    from PIL import Image, ImageDraw
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for b, c in boxes:
        d.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 0), width=3)
        d.text((b[0] + 4, max(0, b[1] - 14)), "%.3f" % c, fill=(255, 0, 0))
    d.text((4, 4), title, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--images", required=True, help="manifest:<source>:<split>  or  dir:<path>")
    ap.add_argument("--set-name", required=True)
    ap.add_argument("--manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--thresholds", default="0.25,0.50,0.75")
    ap.add_argument("--conf-floor", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs/shuttle_capability/hard_negative_eval")
    ap.add_argument("--metrics-dir", default="outputs/shuttle_capability/metrics")
    args = ap.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    out_dir = repo / args.out_dir
    metrics_dir = repo / args.metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    images = [i for i in collect_images(repo, args.images, args.manifest) if os.path.exists(i)]
    skipped_pos = [i for i in images if has_positive_label(i)]
    images = [i for i in images if i not in set(skipped_pos)]
    print("set=%s images=%d skipped_with_positive_label=%d" % (args.set_name, len(images), len(skipped_pos)), flush=True)

    ckpts = [tuple(c.split("=", 1)) for c in args.ckpt if "=" in c]
    if not ckpts:
        raise SystemExit("no --ckpt NAME=PATH")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(repo / ".yolo_cfg"))
    from ultralytics import YOLO

    all_rows = []
    summary = {}
    src_txt = out_dir / ("%s_images.txt" % args.set_name)
    src_txt.write_text(chr(10).join(i.replace("\\", "/") for i in images) + chr(10), encoding="utf-8")

    for name, ckpt in ckpts:
        if not os.path.exists(ckpt):
            print("SKIP missing checkpoint", name, ckpt, flush=True)
            continue
        print("=== predicting %s with %s ===" % (args.set_name, name), flush=True)
        model = YOLO(ckpt)
        stream = model.predict(source=str(src_txt), imgsz=args.imgsz, conf=args.conf_floor, iou=0.7,
                               max_det=300, device=args.device, stream=True, verbose=False, save=False)
        n = 0
        per_image = []
        for res in stream:
            n += 1
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy().tolist()
                confs = res.boxes.conf.cpu().numpy().tolist()
                for b, c in zip(xyxy, confs):
                    boxes.append(([float(x) for x in b], float(c)))
            per_image.append((os.path.abspath(res.path), boxes))
            if n % 200 == 0:
                print("  %d/%d" % (n, len(images)), flush=True)
        summary[name] = {}
        for th in thresholds:
            fps = []
            imgs_with = 0
            for _path, boxes in per_image:
                sel = [c for _b, c in boxes if c >= th]
                if sel:
                    imgs_with += 1
                    fps.extend(sel)
            total = len(fps)
            summary[name][str(th)] = {
                "total_images": len(per_image),
                "total_FP": total,
                "FP_per_image": (total / len(per_image)) if per_image else None,
                "images_with_FP": imgs_with,
                "image_FP_rate": (imgs_with / len(per_image)) if per_image else None,
                "max_confidence_FP": max(fps) if fps else None,
                "mean_confidence_FP": statistics.fmean(fps) if fps else None,
                "median_confidence_FP": statistics.median(fps) if fps else None,
                "P95_FP_confidence": percentile(fps, 0.95),
            }
        # top-K FP images by their highest-confidence FP box
        ranked = sorted([(max([c for _b, c in bs]) if bs else 0.0, p, bs) for p, bs in per_image],
                        key=lambda t: -t[0])[: args.topk]
        for rank, (conf, path, boxes) in enumerate(ranked, 1):
            oid = Path(path).stem
            dst = out_dir / name / "top20_fp" / ("%s_conf%.3f_%s.jpg" % (name, conf, oid))
            draw_and_save(path, boxes, dst, "%s | %s | top%d conf=%.3f | %d boxes" % (name, args.set_name, rank, conf, len(boxes)))
        for path, boxes in per_image:
            for b, c in boxes:
                all_rows.append({"checkpoint": name, "set": args.set_name, "image_path": path, "confidence": round(c, 6),
                                 "x1": round(b[0], 2), "y1": round(b[1], 2), "x2": round(b[2], 2), "y2": round(b[3], 2),
                                 "prediction_count": len(boxes)})

    # Idempotent write: rows for the (checkpoint, set) pairs being recomputed are replaced, other sets are preserved.
    pred_path = metrics_dir / "hard_negative_predictions.csv"
    fields = ["checkpoint", "set", "image_path", "confidence", "x1", "y1", "x2", "y2", "prediction_count"]
    recomputed = {(name, args.set_name) for name, _ in ckpts}
    kept = []
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if (r["checkpoint"], r["set"]) not in recomputed]
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept + all_rows)
    print("predictions csv: kept %d previous rows, wrote %d new rows for %s" % (len(kept), len(all_rows), args.set_name), flush=True)

    payload = {"set": args.set_name, "images": len(images),
               "skipped_because_positive_label": skipped_pos, "conf_floor": args.conf_floor,
               "imgsz": args.imgsz, "thresholds": thresholds, "results": summary}
    (metrics_dir / ("hard_negative_eval_%s.json" % args.set_name)).write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    with open(metrics_dir / ("hard_negative_eval_%s.csv" % args.set_name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "threshold", "total_images", "total_FP", "FP_per_image", "images_with_FP",
                    "image_FP_rate", "max_confidence_FP", "mean_confidence_FP", "median_confidence_FP", "P95_FP_confidence"])
        for name, per_th in summary.items():
            for th, d in per_th.items():
                w.writerow([name, th] + ["" if d[k] is None else (round(d[k], 6) if isinstance(d[k], float) else d[k])
                                         for k in ("total_images", "total_FP", "FP_per_image", "images_with_FP",
                                                   "image_FP_rate", "max_confidence_FP", "mean_confidence_FP",
                                                   "median_confidence_FP", "P95_FP_confidence")])
    print(json.dumps(payload, indent=1)[:1500], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())