#!/usr/bin/env python3
"""Frozen-checkpoint resolution response probe for a supplied tiny-GT part list.

Reads a part CSV (basename + expected bucket/equiv + optional V2@1024 references),
locates each image/label under --root, runs the FROZEN checkpoint at several
imgsz values, and reports per-GT response (<4 / 4-6 / 6-8 focus) plus latency.

No training, no weight updates. Detector conventions follow tools/eval_yolo26_v1.py
(conf floor for candidates, NMS iou 0.7, max_det 300, rect False, GT via load_gt_boxes).
"""
import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # tools/ : where eval_yolo26_v1.py lives in the repo
import eval_yolo26_v1 as ev  # noqa: E402


def index_images(root: Path):
    idx = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and "images" in p.parts:
            idx.setdefault(p.name, p)
    return idx


def label_for(img: Path):
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts[:-1]) / (img.stem + ".txt")


def read_dets(res):
    out = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy().tolist()
        cf = res.boxes.conf.cpu().numpy().tolist()
        for b, c in zip(xyxy, cf):
            out.append((float(c), [float(x) for x in b]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part-csv", required=True)
    ap.add_argument("--root", default=".", help="root for rel_path rows; unused when rows carry abs_image")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--resolutions", nargs="+", type=int, default=[1024, 1280, 1536])
    ap.add_argument("--conf-floor", type=float, default=0.01)
    ap.add_argument("--conf-op", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--matcher-thr", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--latency-reps", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    root = Path(args.root).resolve()
    weights = Path(args.weights).resolve()
    wsha = ev.file_sha256(weights)
    rows = list(csv.DictReader(open(args.part_csv, encoding="utf-8")))
    print("part rows: %d" % len(rows), flush=True)
    print("weights: %s  %d B  sha256=%s" % (weights, weights.stat().st_size, wsha), flush=True)
    print("root: %s" % root, flush=True)

    need_index = any(not Path((r.get("abs_image") or "").strip()).is_file() for r in rows)
    idx = index_images(root) if need_index else {}
    print("image index: %d files (index built: %s)" % (len(idx), need_index), flush=True)

    imgs, meta = [], []
    missing = []
    picked = []
    for r in rows:
        p = None
        absimg = (r.get("abs_image") or "").strip()
        if absimg and Path(absimg).is_file():
            p = Path(absimg)
        if p is None:
            rel = (r.get("rel_path") or "").strip()
            if rel:
                cand = root / rel
                if cand.exists():
                    p = cand
        if p is None and idx:
            p = idx.get(r["basename"])
        if p is None:
            missing.append(r["basename"])
            continue
        picked.append(str(p.relative_to(root)) if str(p).startswith(str(root)) else str(p))
        imgs.append(p)
        meta.append(r)
    if missing:
        print("MISSING %d: %s" % (len(missing), ",".join(missing[:5])), flush=True)
    if not imgs:
        print("no images located; abort", flush=True)
        return 2
    print("located %d/%d images" % (len(imgs), len(rows)), flush=True)
    if len({p.name for p in imgs}) != len(imgs):
        print("ERROR: duplicate basenames in the part list, keys must be unique", flush=True)
        return 3
    for x in picked:
        print("  pick %s" % x, flush=True)

    model = YOLO(str(weights))
    import platform
    import ultralytics
    report = {"env": {"python": sys.version.split()[0], "platform": platform.platform(),
                      "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                      "cuda": torch.version.cuda},
              "weights": str(weights), "weights_sha256": wsha, "weights_bytes": weights.stat().st_size,
              "root": str(root), "device": args.device, "conf_floor": args.conf_floor,
              "conf_op": args.conf_op, "nms_iou": args.iou, "max_det": args.max_det,
              "matcher_thr": args.matcher_thr, "n_part_rows": len(rows),
              "n_images": len(imgs), "missing": missing, "picked": picked, "resolutions": {}}

    # GT per image (frozen decode)
    gts = {}
    meta_by_name = {m["basename"]: m for m in meta}
    for p in imgs:
        lp = None
        absl = (meta_by_name.get(p.name, {}).get("abs_label") or "").strip()
        if absl and Path(absl).is_file():
            lp = Path(absl)
        if lp is None:
            lp = label_for(p)
        g = ev.load_gt_boxes(str(p), str(lp))
        gts[p.name] = (g if g is not None else [], str(lp))
    n_gt_tiny = sum(1 for p in imgs for b in gts[p.name][0] if b["eq640"] < 8.0)
    n_gt_all = sum(len(gts[p.name][0]) for p in imgs)
    print("GT: %d tiny(<8) / %d all over %d images" % (n_gt_tiny, n_gt_all, len(imgs)), flush=True)

    for R in args.resolutions:
        for _ in range(3):   # warmup, excluded from the timed pass
            model.predict(source=str(imgs[0]), imgsz=R, conf=args.conf_floor, iou=args.iou,
                          max_det=args.max_det, device=args.device, rect=False,
                          verbose=False, save=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        per_image = []
        dets = {}
        t0 = time.perf_counter()
        for p in imgs:
            t1 = time.perf_counter()
            res = model.predict(source=str(p), imgsz=R, conf=args.conf_floor, iou=args.iou,
                                max_det=args.max_det, device=args.device, rect=False,
                                verbose=False, save=False)[0]
            dt = (time.perf_counter() - t1) * 1000.0
            d = read_dets(res)
            dets[p.name] = d
            per_image.append({"image": p.name, "ms": dt, "n_dets": len(d),
                              "n_dets_op": sum(1 for c, _ in d if c >= args.conf_op),
                              "speed": {k: float(v) for k, v in (res.speed or {}).items()},
                              "orig_shape": [int(x) for x in res.orig_shape]})
        total_s = time.perf_counter() - t0

        # steady-state latency: warmup + reps on a fixed image, batch 1
        probe = str(imgs[0])
        for _ in range(3):
            model.predict(source=probe, imgsz=R, conf=args.conf_floor, iou=args.iou,
                          max_det=args.max_det, device=args.device, rect=False,
                          verbose=False, save=False)
        torch.cuda.synchronize()
        lats = []
        speeds = []
        for _ in range(args.latency_reps):
            t1 = time.perf_counter()
            r0 = model.predict(source=probe, imgsz=R, conf=args.conf_floor, iou=args.iou,
                               max_det=args.max_det, device=args.device, rect=False,
                               verbose=False, save=False)[0]
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t1) * 1000.0)
            speeds.append({k: float(v) for k, v in (r0.speed or {}).items()})
        peak = torch.cuda.max_memory_allocated() / (1 << 20)

        # per-GT response
        gt_rows = []
        iou_conf_by_image = {}
        for p in imgs:
            g = gts[p.name][0]
            d = dets[p.name]
            op = [(c, b) for c, b in d if c >= args.conf_op]
            m = ev.match_greedy([x["box"] for x in g], op, args.matcher_thr, metric="iou")
            hit = m["gt_hit"]
            iou_conf = []
            iou_conf_by_image[p.name] = iou_conf
            for j, box in enumerate(g):
                best, bc, nb = 0.0, 0.0, 0
                pairs = []
                for c, db in d:
                    v = ev.iou_xyxy(db, box["box"])
                    pairs.append([c, v])
                    if v > best:
                        best, bc = v, c
                    if v >= 0.3:
                        nb += 1
                iou_conf.append({"gt_index": j, "pairs": sorted(pairs, key=lambda x: -x[0])})
                gt_rows.append({"image": p.name, "gt_index": j, "eq640": box["eq640"],
                                "bucket": box["bucket"], "tiny": box["eq640"] < 8.0,
                                "matched_op": bool(hit[j]), "best_iou_weak": best,
                                "best_conf_at_best_iou": bc, "n_dets_iou30": nb,
                                "n_dets": len(d), "n_dets_op": len(op)})
        report["resolutions"][str(R)] = {
            "per_image": per_image, "gt": gt_rows, "peak_vram_mib": peak,
            "per_image_iou_conf": iou_conf_by_image,
            "latency_ms": {"median": statistics.median(lats), "mean": statistics.fmean(lats),
                           "min": min(lats), "max": max(lats), "reps": len(lats), "batch": 1},
            "latency_speed_median_ms": {k: statistics.median([s.get(k, 0.0) for s in speeds])
                                        for k in ("preprocess", "inference", "postprocess")},
            "pass_speed_median_ms": {k: statistics.median([x["speed"].get(k, 0.0) for x in per_image
                                                           if x["speed"]])
                                     for k in ("preprocess", "inference", "postprocess")},
            "net_input_px": [R, R],
            "per_image_ms_from_pass": {"mean": statistics.fmean([x["ms"] for x in per_image]),
                                       "median": statistics.median([x["ms"] for x in per_image])},
            "pass_total_s": total_s, "img_per_s": len(imgs) / total_s,
            "n_candidates": sum(r["n_dets"] for r in per_image),
            "n_candidates_op": sum(r["n_dets_op"] for r in per_image),
        }
        tiny = [r for r in gt_rows if r["tiny"]]
        lat = report["resolutions"][str(R)]["latency_ms"]
        print("imgsz=%4d  cand=%4d  tinyGT=%2d  tinyHit(op)=%d  medLat=%.2f ms  img/s=%.2f  peakVRAM=%.0f MiB"
              % (R, sum(r["n_dets"] for r in per_image), len(tiny),
                 sum(1 for r in tiny if r["matched_op"]), lat["median"],
                 report["resolutions"][str(R)]["img_per_s"], peak), flush=True)

    # cross-check against any V2@1024 references carried in the part CSV
    ref = report["resolutions"][str(1024)]["gt"] if "1024" in report["resolutions"] else None
    mism = []
    if ref is not None:
        by = {}
        for r in ref:
            by.setdefault(r["image"], []).append(r)
        for m in meta:
            cand = by.get(m["basename"], [])
            want_eq = float(m["equiv_size_640"])
            pick = min(cand, key=lambda r: abs(r["eq640"] - want_eq)) if cand else None
            if pick is None or abs(pick["eq640"] - want_eq) > 1e-6:
                mism.append({"image": m["basename"], "why": "gt_not_found"})
                continue
            ref_hit = m.get("v2_matched_op_1024") or m.get("local_ref_matched_op") or ""
            ref_iou = m.get("v2_best_iou_weak_1024") or m.get("local_ref_best_iou") or ""
            if ref_hit:
                if str(pick["matched_op"]) != ref_hit:
                    mism.append({"image": m["basename"], "why": "matched_op",
                                 "local": ref_hit, "remote": str(pick["matched_op"])})
            if ref_iou:
                want = float(ref_iou)
                if abs(pick["best_iou_weak"] - want) > 0.02:
                    mism.append({"image": m["basename"], "why": "best_iou_weak",
                                 "local": want, "remote": round(pick["best_iou_weak"], 6)})
    report["crosscheck_vs_local_v2_1024"] = {"n_checked": len(meta), "mismatches": mism}
    print("crosscheck@1024 vs local V2 refs: %d checked, %d mismatch" % (len(meta), len(mism)), flush=True)
    for x in mism[:10]:
        print("   ! %s" % json.dumps(x), flush=True)

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("WROTE %s (%d B)" % (args.out, Path(args.out).stat().st_size), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
