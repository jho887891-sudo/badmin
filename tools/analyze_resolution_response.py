#!/usr/bin/env python3
"""Zero-training resolution-response diagnostic.

Question left open by the Tiny Recovery V1 post-mortem: at a HIGHER input resolution, does a strictly sub-stride
tiny object start to produce any spatial response at all? The weights are frozen (the V2 checkpoint is used exactly
as released); only the inference resolution changes, so any difference is a property of the sampling grid, not of
training.

Measured per resolution, on the images of val|eth_unseen that contain at least one <8 px GT:
  * operating-point hits (conf 0.25, greedy IoU 0.5) per bucket
  * the best IoU any detection at conf >= 0.01 achieves on each GT (the "weak response" probe, IoU 0.1/0.3/0.5)
  * latency (median ms/image with warmup) so the cost of the hypothesis is known up front
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_eth_only_v1_errors as err  # noqa: E402
import analyze_tiny_representability as rep  # noqa: E402  (NET_FACTOR, P3_STRIDE, bucket logic)
import eval_yolo26_v1 as ev  # noqa: E402

BUCKETS = ("<4", "4-6", "6-8")
IOU_MARKS = (0.1, 0.3, 0.5)


def tiny_images(dataset_manifest, leakage_inventory, classes=None, max_eq=8.0) -> list:
    """val|eth_unseen images that contain at least one tiny GT, with their GT boxes."""
    classes = classes or err.ETH_UNSEEN_CLASSES
    keep = set()
    for r in err.load_rows(leakage_inventory):
        if str(r.get("leakage_class")) in set(classes):
            keep.add(str(r.get("image")).replace(chr(92), "/").lower())
    out = []
    for r in err.load_rows(dataset_manifest):
        if str(r.get("split")) != "val":
            continue
        img, label = str(r.get("image") or ""), str(r.get("label") or "")
        if not img or not label or img.replace(chr(92), "/").lower() not in keep:
            continue
        gts = ev.load_gt_boxes(img, label)
        tiny = [g for g in gts if float(g["eq640"]) < float(max_eq)]
        if tiny:
            out.append({"image": img, "gts": gts, "tiny": tiny})
    return out


def measure_resolution(model, items, imgsz, op_conf=0.25, op_iou=0.5, weak_conf=0.01, warmup=2, repeats=3) -> dict:
    """Run one resolution and return per-GT rows plus latency."""
    paths = [it["image"] for it in items]
    dets = {}
    t_start = time.perf_counter()
    stream = model.predict(source=paths, imgsz=imgsz, conf=weak_conf, iou=0.7, max_det=300, stream=True,
                           verbose=False, save=False)
    for res in stream:
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()
            boxes = [(float(c), [float(x) for x in b]) for b, c in zip(xyxy, confs)]
        dets[str(res.path)] = boxes
    first_pass_s = time.perf_counter() - t_start
    lat = latency(model, imgsz, warmup=warmup, repeats=repeats)
    lat["per_image_ms_from_pass"] = 1000.0 * first_pass_s / max(1, len(paths))
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    rows = []
    for it in items:
        img = it["image"]
        raw = dets.get(img) or dets.get(str(Path(img).resolve())) or []
        strong = [(c, b) for c, b in raw if c >= float(op_conf)]
        op = ev.match_greedy([g["box"] for g in it["gts"]], strong, op_iou, metric="iou")
        for gj, g in enumerate(it["gts"]):
            best_iou, best_conf = 0.0, None
            for c, b in raw:
                v = ev.iou_xyxy(b, g["box"])
                if v > best_iou:
                    best_iou, best_conf = v, c
            rows.append({
                "image": img, "gt_index": gj, "bucket": g["bucket"], "equiv_size_640": float(g["eq640"]),
                "net_px": float(g["eq640"]) * rep.NET_FACTOR, "imgsz": int(imgsz),
                "matched_op": bool(op["gt_hit"][gj]), "best_iou_weak": round(best_iou, 6),
                "best_conf_weak": (None if best_conf is None else round(best_conf, 6)),
                "detections_conf_ge_weak": len(raw),
            })
    return {"rows": rows, "first_pass_s": first_pass_s, "latency": lat}


def latency(model, imgsz, warmup=2, repeats=3) -> dict:
    """Median single-image latency at one resolution; never fatal (an 8 GB card can OOM at 1536)."""
    import torch
    try:
        with torch.inference_mode():
            x = torch.zeros(1, 3, int(imgsz), int(imgsz), device=next(model.model.parameters()).device)
            for _ in range(max(1, int(warmup))):
                _ = model.model(x)
            torch.cuda.synchronize()
            times = []
            for _ in range(max(1, int(repeats))):
                t0 = time.perf_counter()
                _ = model.model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            del x
        torch.cuda.empty_cache()
        return {"median_ms": statistics.median(times), "min_ms": min(times), "repeats": len(times),
                "imgsz": int(imgsz)}
    except Exception as exc:  # noqa: BLE001 - report instead of dying, the pass timing still exists
        torch.cuda.empty_cache()
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200]), "imgsz": int(imgsz)}


def summarise(rows, imgsz) -> dict:
    sub = [r for r in rows if int(r["imgsz"]) == int(imgsz)]
    out = {"imgsz": int(imgsz), "gt_total": len(sub), "by_bucket": {}}
    for bucket in BUCKETS + ("8+",):
        sel = [r for r in sub if (r["bucket"] == bucket if bucket != "8+" else r["bucket"] not in BUCKETS)]
        if not sel:
            continue
        tp = sum(1 for r in sel if r["matched_op"])
        ious = sorted(float(r["best_iou_weak"]) for r in sel)
        out["by_bucket"][bucket] = {
            "gt": len(sel), "tp": tp, "fn": len(sel) - tp, "recall": tp / len(sel),
            "median_best_iou_weak": ious[len(ious) // 2] if ious else None,
            "max_best_iou_weak": ious[-1] if ious else None,
            "weak_ge_0.1": sum(1 for x in ious if x >= 0.1),
            "weak_ge_0.3": sum(1 for x in ious if x >= 0.3),
            "weak_ge_0.5": sum(1 for x in ious if x >= 0.5),
        }
    sub_stride = [r for r in sub if float(r["net_px"]) < rep.P3_STRIDE and r["bucket"] in BUCKETS]
    ious = sorted(float(r["best_iou_weak"]) for r in sub_stride)
    out["strict_sub_stride"] = {
        "gt": len(sub_stride), "tp": sum(1 for r in sub_stride if r["matched_op"]),
        "max_best_iou_weak": ious[-1] if ious else None,
        "median_best_iou_weak": ious[len(ious) // 2] if ious else None,
        "weak_ge_0.1": sum(1 for x in ious if x >= 0.1),
        "weak_ge_0.3": sum(1 for x in ious if x >= 0.3),
    }
    return out


def compute_branch(summaries) -> dict:
    """The three-way decision frozen before running: resolution vs grid vs a split strategy."""
    base = next((s for s in summaries if s["imgsz"] == 1024), None)
    best = max(summaries, key=lambda s: s["imgsz"])
    if base is None:
        return {"branch": "UNDETERMINED", "why": "1024 baseline missing"}
    def sub_resp(s):
        return (s["by_bucket"].get("<4", {}).get("weak_ge_0.1", 0)
                + s["by_bucket"].get("4-6", {}).get("weak_ge_0.1", 0))
    def mid_resp(s):
        return s["by_bucket"].get("6-8", {}).get("weak_ge_0.1", 0)
    base_sub, hi_sub = sub_resp(base), sub_resp(best)
    base_mid, hi_mid = mid_resp(base), mid_resp(best)
    gained_sub = hi_sub - base_sub
    gained_mid = hi_mid - base_mid
    branch, why = "UNDETERMINED", ""
    if gained_sub >= 3:
        branch = "HIGHER_RESOLUTION_TRAINING"
        why = ("raising imgsz to %d produced %d new weak candidates (IoU >= 0.1) in the sub-stride buckets "
               "(<4 + 4-6: %d -> %d), so the limiter is input scale rather than the sampling grid"
               % (best["imgsz"], gained_sub, base_sub, hi_sub))
    elif gained_mid >= 3 and gained_sub <= 1:
        branch = "P2_PLUS_MID_BUCKET_WEIGHTING"
        why = ("only the 6-8 bucket responded (weak candidates %d -> %d) while the sub-stride buckets stayed at "
               "~0, so the two populations need different treatments" % (base_mid, hi_mid))
    else:
        branch = "P2_STRIDE_4"
        why = ("sub-stride objects still produce essentially no response at %d (weak candidates %d -> %d), so the "
               "binding constraint is the sampling grid itself" % (best["imgsz"], base_sub, hi_sub))
    return {"branch": branch, "why": why, "base_imgsz": base["imgsz"], "max_imgsz": best["imgsz"],
            "sub_stride_weak_ge_0.1": {str(base["imgsz"]): base_sub, str(best["imgsz"]): hi_sub},
            "mid_bucket_weak_ge_0.1": {str(base["imgsz"]): base_mid, str(best["imgsz"]): hi_mid}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Zero-training resolution-response diagnostic (frozen weights)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--leakage-inventory", default="outputs/shuttle_capability/metrics/eth_data_leakage_inventory.csv")
    ap.add_argument("--resolutions", default="1024,1280,1536")
    ap.add_argument("--op-conf", type=float, default=0.25)
    ap.add_argument("--op-iou", type=float, default=0.5)
    ap.add_argument("--weak-conf", type=float, default=0.01)
    ap.add_argument("--device", default="0")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-json", default="outputs/shuttle_capability/metrics/resolution_response_v1.json")
    ap.add_argument("--out-csv", default="outputs/shuttle_capability/metrics/resolution_response_v1.csv")
    ap.add_argument("--decision-json", default="outputs/shuttle_capability/metrics/resolution_response_v1_decision.json")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    from ultralytics import YOLO
    items = tiny_images(repo / a.dataset_manifest, repo / a.leakage_inventory)
    if not items:
        print("[res] no tiny-GT images found")
        return 2
    model = YOLO(str(repo / a.ckpt) if not Path(a.ckpt).is_absolute() else a.ckpt)
    summaries, rows, runs = [], [], {}
    for res in [int(x) for x in str(a.resolutions).split(",") if x.strip()]:
        got = measure_resolution(model, items, res, op_conf=a.op_conf, op_iou=a.op_iou, weak_conf=a.weak_conf,
                                 warmup=a.warmup, repeats=a.repeats)
        rows.extend(got["rows"])
        summaries.append(summarise(got["rows"], res))
        runs[str(res)] = {"latency": got["latency"], "first_pass_s": got["first_pass_s"],
                          "images": len(items)}
        lat_txt = ("%.1f ms/img" % got["latency"]["median_ms"]) if "median_ms" in got["latency"] \
            else ("probe failed: %s" % got["latency"].get("error", "?"))
        print("[res] imgsz=%d done: %d GT rows, tensor probe %s, pass %.1f ms/img (%.1fs for %d images)"
              % (res, len(got["rows"]), lat_txt, got["latency"].get("per_image_ms_from_pass", 0.0),
                 got["first_pass_s"], len(items)))
    decision = compute_branch(summaries)
    rep_out = {
        "scope": "zero-training resolution response with frozen weights; only imgsz changes",
        "checkpoint": a.ckpt, "images_with_tiny_gt": len(items),
        "operating_point": {"conf": a.op_conf, "iou": a.op_iou},
        "weak_probe": {"conf": a.weak_conf, "iou_marks": list(IOU_MARKS)},
        "net_factor": rep.NET_FACTOR, "p3_stride_px": rep.P3_STRIDE,
        "by_resolution": summaries, "runs": runs, "decision": decision,
        "headline": ("%s: %s" % (decision["branch"], decision["why"])),
    }
    Path(repo / a.out_json).parent.mkdir(parents=True, exist_ok=True)
    (repo / a.out_json).write_text(json.dumps(rep_out, indent=1, ensure_ascii=False), encoding="utf-8")
    (repo / a.decision_json).write_text(json.dumps(decision, indent=1, ensure_ascii=False), encoding="utf-8")
    cols = ["imgsz", "image", "gt_index", "bucket", "equiv_size_640", "net_px", "matched_op",
            "best_iou_weak", "best_conf_weak", "detections_conf_ge_weak"]
    with (repo / a.out_csv).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[res] " + rep_out["headline"])
    print("[res] wrote %s, %s and %s" % (a.out_json, a.out_csv, a.decision_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
