#!/usr/bin/env python3
"""Read-only diagnostic: is the tiny-object evidence PRESENT but suppressed, or absent entirely?

The tiny-recovery experiment assumes (design section 3) that after hard-negative learning a tiny shuttlecock's
weak visual evidence is pushed below the decision boundary. That is testable without training:

  * representability: a <8 px (equiv_size_640) box occupies how many pixels in the frozen 1024 network frame, and
    how does that compare with the P3 stride of a plain YOLO26s (8 px)? Objects below one stride cell are the ones
    the head cannot localise independently of the intervention.
  * weak evidence: for every GT in val|eth_unseen, look for the best overlapping detection at a very low
    confidence floor (no greedy constraint, no operating threshold). If most missed tiny GTs still have a
    sub-threshold detection on them, the information exists and a decision-boundary intervention (exposure) can in
    principle recover it; if they have none, the lever is scale/resolution rather than exposure.

Nothing here can change any dataset, recipe or evaluation set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_eth_only_v1_errors as err  # noqa: E402  (canonical GT decode, matcher, statistics)
import eval_yolo26_v1 as ev  # noqa: E402  (canonical IoU)
import eval_eth_official_baseline as eb  # noqa: E402  (imgsz constant)

P3_STRIDE = 8.0
NET_FACTOR = float(eb.IMGSZ) / 640.0
TINY_BUCKETS = ("<4", "4-6", "6-8")


def bucket_of_size(eq640: float) -> str:
    return ev.bucket_of(float(eq640))


def tiny_gt_table(dump_json, dataset_manifest, leakage_inventory, classes=None, op_conf=0.25, op_iou=0.5,
                  weak_conf=0.01, weak_iou=0.1) -> list:
    """Per-GT rows: operating-point match plus the best sub-threshold overlap (weak evidence)."""
    classes = classes or err.ETH_UNSEEN_CLASSES
    dump = json.loads(Path(dump_json).read_text(encoding="utf-8"))
    dets = dump.get("dets") or {}
    keep = set()
    for r in err.load_rows(leakage_inventory):
        if str(r.get("leakage_class")) in set(classes):
            keep.add(str(r.get("image")).replace(chr(92), "/").lower())
    rows = []
    for r in err.load_rows(dataset_manifest):
        if str(r.get("split")) != "val":
            continue
        img, label = str(r.get("image") or ""), str(r.get("label") or "")
        if not img or img.replace(chr(92), "/").lower() not in keep:
            continue
        gts = ev.load_gt_boxes(img, label) if label else []
        raw = dets.get(img) or []
        strong = [(float(d["conf"]), [float(x) for x in d["box"]]) for d in raw if float(d["conf"]) >= float(op_conf)]
        weak = [(float(d["conf"]), [float(x) for x in d["box"]]) for d in raw if float(d["conf"]) >= float(weak_conf)]
        if not gts:
            continue
        op = ev.match_greedy([g["box"] for g in gts], strong, op_iou, metric="iou")
        for gj, g in enumerate(gts):
            best_iou, best_conf = 0.0, None
            for conf, box in weak:
                v = ev.iou_xyxy(box, g["box"])
                if v > best_iou:
                    best_iou, best_conf = v, conf
            rows.append({
                "image": img, "gt_index": gj, "bucket": g["bucket"], "equiv_size_640": float(g["eq640"]),
                "net_px_1024": float(g["eq640"]) * NET_FACTOR,
                "stride_cells": (float(g["eq640"]) * NET_FACTOR) / P3_STRIDE,
                "matched_op": bool(op["gt_hit"][gj]),
                "best_iou_weak": round(best_iou, 6),
                "best_conf_weak": (None if best_conf is None else round(best_conf, 6)),
            })
    return rows


def _stats(rows) -> dict:
    gt = len(rows)
    tp = sum(1 for r in rows if r["matched_op"])
    sizes = sorted(r["net_px_1024"] for r in rows)
    fn = [r for r in rows if not r["matched_op"]]
    return {
        "gt": gt, "tp": tp, "fn": gt - tp,
        "recall": (tp / gt) if gt else None,
        "median_net_px_1024": (sizes[len(sizes) // 2] if sizes else None),
        "max_net_px_1024": (max(sizes) if sizes else None),
        "below_one_stride": sum(1 for x in sizes if x < P3_STRIDE),
        "below_half_stride": sum(1 for x in sizes if x < P3_STRIDE / 2.0),
        "fn_below_one_stride": sum(1 for r in fn if r["net_px_1024"] < P3_STRIDE),
    }


def representability_summary(rows) -> dict:
    out = {"overall": _stats(rows), "by_bucket": {}}
    for bucket in ev.BUCKET_LABELS:
        sub = [r for r in rows if r["bucket"] == bucket]
        if sub:
            out["by_bucket"][bucket] = _stats(sub)
    out["tiny_lt8"] = _stats([r for r in rows if r["equiv_size_640"] < 8.0])
    return out


def weak_evidence_summary(rows, weak_iou=0.1) -> dict:
    def block(sel):
        if not sel:
            return {"n": 0, "with_weak_detection": 0, "share_with_weak_detection": None,
                    "median_best_conf_weak": None, "median_best_iou_weak": None}
        with_weak = [r for r in sel if r["best_iou_weak"] >= weak_iou]
        confs = sorted(r["best_conf_weak"] for r in with_weak if r["best_conf_weak"] is not None)
        ious = sorted(r["best_iou_weak"] for r in with_weak)
        return {
            "n": len(sel), "with_weak_detection": len(with_weak),
            "share_with_weak_detection": len(with_weak) / len(sel),
            "median_best_conf_weak": (confs[len(confs) // 2] if confs else None),
            "median_best_iou_weak": (ious[len(ious) // 2] if ious else None),
            "weak_iou_threshold": weak_iou,
        }
    rows = list(rows)
    return {
        "false_negatives_tiny": block([r for r in rows if not r["matched_op"] and r["equiv_size_640"] < 8.0]),
        "true_positives_tiny": block([r for r in rows if r["matched_op"] and r["equiv_size_640"] < 8.0]),
        "false_negatives_all": block([r for r in rows if not r["matched_op"]]),
        "true_positives_all": block([r for r in rows if r["matched_op"]]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tiny-object representability and weak-evidence diagnostic")
    ap.add_argument("--dump", default="_scratch_localization_audit/loc_eth_real_hardneg_v2_best_val.json")
    ap.add_argument("--model", default="eth_real_hardneg_v2_best")
    ap.add_argument("--dataset-manifest", default="outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv")
    ap.add_argument("--leakage-inventory", default="outputs/shuttle_capability/metrics/eth_data_leakage_inventory.csv")
    ap.add_argument("--op-conf", type=float, default=0.25)
    ap.add_argument("--op-iou", type=float, default=0.5)
    ap.add_argument("--weak-conf", type=float, default=0.01)
    ap.add_argument("--weak-iou", type=float, default=0.1)
    ap.add_argument("--out-csv", default="outputs/shuttle_capability/metrics/tiny_representability_v1.csv")
    ap.add_argument("--out-json", default="outputs/shuttle_capability/metrics/tiny_representability_v1.json")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    dump = repo / a.dump
    if not dump.is_file():
        print("[tiny-diag] V2 prediction dump missing: %s" % dump)
        return 2
    rows = tiny_gt_table(dump, repo / a.dataset_manifest, repo / a.leakage_inventory,
                         op_conf=a.op_conf, op_iou=a.op_iou, weak_conf=a.weak_conf, weak_iou=a.weak_iou)
    rep = {
        "scope": "read-only diagnostic of the tiny-object hypothesis (present-but-suppressed vs absent evidence)",
        "model": a.model, "dump": a.dump, "net_factor_1024_over_640": NET_FACTOR, "p3_stride_px": P3_STRIDE,
        "operating_point": {"conf": a.op_conf, "iou": a.op_iou},
        "weak_evidence_point": {"conf": a.weak_conf, "iou": a.weak_iou},
        "representability": representability_summary(rows),
        "weak_evidence": weak_evidence_summary(rows, a.weak_iou),
    }
    fn_tiny = rep["weak_evidence"]["false_negatives_tiny"]
    tp_tiny = rep["representability"]["tiny_lt8"]
    rep["headline"] = ("%s: tiny (<8 equiv_640) GT %d, TP %d, FN %d; in the 1024 frame a tiny GT is %.2f-%.2f px "
                       "(stride %.0f), and %d of the %d tiny FNs still have a detection with IoU >= %.2f at conf >= "
                       "%.2f (share %s, median conf %s)"
                       % (a.model, tp_tiny["gt"], tp_tiny["tp"], tp_tiny["fn"],
                          min((r["net_px_1024"] for r in rows if r["equiv_size_640"] < 8.0), default=0.0),
                          max((r["net_px_1024"] for r in rows if r["equiv_size_640"] < 8.0), default=0.0),
                          P3_STRIDE, fn_tiny["with_weak_detection"], fn_tiny["n"], a.weak_iou, a.weak_conf,
                          ("%.4f" % fn_tiny["share_with_weak_detection"]) if fn_tiny["share_with_weak_detection"] is not None else "n/a",
                          fn_tiny["median_best_conf_weak"]))
    Path(repo / a.out_json).parent.mkdir(parents=True, exist_ok=True)
    (repo / a.out_json).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    cols = ["image", "gt_index", "bucket", "equiv_size_640", "net_px_1024", "stride_cells", "matched_op",
            "best_iou_weak", "best_conf_weak"]
    with (repo / a.out_csv).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[tiny-diag] " + rep["headline"])
    print("[tiny-diag] wrote %s and %s" % (a.out_json, a.out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
