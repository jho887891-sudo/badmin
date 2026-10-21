#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-02/03: build the synthetic size-bucket fixtures and run the offline protocol.

Two modes:

  --make-fixtures   (re)generate outputs/p7_eval/fixtures/{ground_truth,predictions}.jsonl
                    deterministically (numpy default_rng, fixed seed) and write
                    fixtures.json describing provenance;
  default           evaluate the COMMITTED fixtures and write
                    outputs/p7_eval/size_bucket_eval.json.

No model is loaded, no image is opened, no GPU is touched.  The fixture is a
stochastic SIMULATION of a detector whose recall falls with object size; it
exists so the harness can be scored, not so a detector can be claimed.

Exit codes: 0 = report written; 2 = input rejected by schema validation;
1 = unexpected error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.size_bucket_eval import (  # noqa: E402
    EvalSchemaError,
    EVIDENCE_SYNTHETIC,
    GT_SCHEMA_ID,
    PRED_SCHEMA_ID,
    EVAL_SCHEMA_VERSION,
    evaluate,
    write_report,
)

FIXTURE_DIR = ROOT / "outputs" / "p7_eval" / "fixtures"
GROUND_TRUTH = FIXTURE_DIR / "ground_truth.jsonl"
PREDICTIONS = FIXTURE_DIR / "predictions.jsonl"
REPORT = ROOT / "outputs" / "p7_eval" / "size_bucket_eval.json"
SEED = 20261021
N_FRAMES = 120
IMAGE_SIZE = {"width_px": 1280, "height_px": 720}
CAMERA_ID = "d455_left"
# Simulated recall per bucket.  These are DECLARED fixture parameters, not
# measured detector behaviour: the harness must recover them, nothing more.
RECALL_BY_BUCKET = {"<4px": 0.20, "4-6px": 0.35, "6-10px": 0.60,
                    "10-20px": 0.80, ">20px": 0.95}
FP_RATE_NEGATIVE_FRAME = 0.12


def _bucket(size: float) -> str:
    if size < 4.0:
        return "<4px"
    if size < 6.0:
        return "4-6px"
    if size < 10.0:
        return "6-10px"
    if size < 20.0:
        return "10-20px"
    return ">20px"


def build_fixture_records(seed: int = SEED, n_frames: int = N_FRAMES):
    """Deterministic synthetic GT + predictions (documented in fixtures.json)."""
    rng = np.random.default_rng(seed)
    gt_records, pred_records = [], []
    per_bucket_gt = {}
    per_bucket_tp = {}
    for index in range(n_frames):
        frame_key = "fixture_val_%05d" % index
        timestamp = round(100.0 + index * 0.02, 6)  # 50 Hz
        # log-uniform size 3..32 px so every bucket gets samples
        n_objects = 0 if index % 9 == 8 else int(rng.integers(1, 3))
        objects = []
        for obj_index in range(n_objects):
            size = float(np.exp(rng.uniform(np.log(3.0), np.log(32.0))))
            x1 = float(rng.uniform(20.0, 1240.0))
            y1 = float(rng.uniform(20.0, 680.0))
            w = max(3.0, size)
            h = max(3.0, float(size * rng.uniform(0.85, 1.15)))
            objects.append({"bbox_xyxy_px": [round(x1, 2), round(y1, 2),
                                             round(x1 + w, 2), round(y1 + h, 2)],
                            "size_px": round(float(np.sqrt(w * h)), 4)})
        gt_records.append({
            "schema": GT_SCHEMA_ID, "schema_version": EVAL_SCHEMA_VERSION,
            "frame_key": frame_key, "split": "val", "camera_id": CAMERA_ID,
            "timestamp_s": timestamp, "image_size": IMAGE_SIZE, "labeled": True,
            "objects": objects,
        })
        for obj in objects:
            bucket = _bucket(obj["size_px"])
            per_bucket_gt[bucket] = per_bucket_gt.get(bucket, 0) + 1
            if rng.random() > RECALL_BY_BUCKET[bucket]:
                continue  # simulated miss
            jitter = rng.normal(0.0, 0.35, size=4)
            bbox = [round(float(v), 2) for v in (np.asarray(obj["bbox_xyxy_px"]) + jitter)]
            per_bucket_tp[bucket] = per_bucket_tp.get(bucket, 0) + 1
            pred_records.append({
                "schema": PRED_SCHEMA_ID, "schema_version": EVAL_SCHEMA_VERSION,
                "frame_key": frame_key, "bbox_xyxy_px": bbox,
                "confidence": round(float(min(0.99, max(0.05, 0.5 + 0.4 * rng.random()))), 4),
                "latency_ms": round(float(np.exp(rng.normal(np.log(22.0), 0.35))), 4),
                "image_size": IMAGE_SIZE, "camera_id": CAMERA_ID,
                "timestamp_s": timestamp,
                "pixel_covariance": [[1.0, 0.0], [0.0, 1.0]],
            })
        if not objects and rng.random() < FP_RATE_NEGATIVE_FRAME:
            size = float(rng.uniform(8.0, 24.0))
            x1, y1 = float(rng.uniform(40.0, 1200.0)), float(rng.uniform(40.0, 660.0))
            pred_records.append({
                "schema": PRED_SCHEMA_ID, "schema_version": EVAL_SCHEMA_VERSION,
                "frame_key": frame_key,
                "bbox_xyxy_px": [round(x1, 2), round(y1, 2), round(x1 + size, 2), round(y1 + size, 2)],
                "confidence": round(float(0.1 + 0.3 * rng.random()), 4),
                "latency_ms": round(float(np.exp(rng.normal(np.log(22.0), 0.35))), 4),
                "image_size": IMAGE_SIZE, "camera_id": CAMERA_ID,
                "timestamp_s": timestamp,
                "pixel_covariance": [[1.0, 0.0], [0.0, 1.0]],
            })
    metadata = {
        "seed": seed, "n_frames": n_frames, "n_gt_objects": len(
            [obj for record in gt_records for obj in record["objects"]]),
        "n_predictions": len(pred_records),
        "gt_objects_per_bucket": dict(sorted(per_bucket_gt.items())),
        "simulated_tp_per_bucket": dict(sorted(per_bucket_tp.items())),
        "declared_recall_by_bucket": RECALL_BY_BUCKET,
        "false_positive_rate_on_negative_frames": FP_RATE_NEGATIVE_FRAME,
        "provenance": (
            "GENERATED BY scripts/shuttle_detection/evaluate_size_buckets.py --make-fixtures. "
            "A synthetic simulated detector with size-dependent recall; it is NOT a detector "
            "output and must never be reported as detector accuracy."
        ),
    }
    return gt_records, pred_records, metadata


def write_fixtures() -> None:
    gt_records, pred_records, metadata = build_fixture_records()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for path, records in ((GROUND_TRUTH, gt_records), (PREDICTIONS, pred_records)):
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
                        encoding="utf-8")
    (FIXTURE_DIR / "fixtures.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("wrote", GROUND_TRUTH, len(gt_records), "records")
    print("wrote", PREDICTIONS, len(pred_records), "records")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-fixtures", action="store_true",
                        help="regenerate the committed synthetic fixtures")
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH))
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--provenance", default=EVIDENCE_SYNTHETIC,
                        choices=[EVIDENCE_SYNTHETIC, "REAL_DETECTOR_OUTPUT"])
    parser.add_argument("--split", default="val")
    args = parser.parse_args(argv)
    if args.make_fixtures:
        write_fixtures()
    try:
        report = evaluate(Path(args.ground_truth), Path(args.predictions),
                          provenance=args.provenance, split=args.split)
    except EvalSchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    path = write_report(report, Path(args.report))
    overall = report["matching_rules"][report["primary_matching_rule"]]["overall"]
    print("claim:", report["claim"], "| detection schema:", report["detection_schema_source"])
    for name, block in report["matching_rules"]["iou50"]["per_bucket"].items():
        print("  %-8s gt=%-4d tp=%-4d fp=%-3d fn=%-4d P=%-6s R=%-6s AP=%s"
              % (name, block["gt_objects"], block["tp"], block["fp"], block["fn"],
                 block["precision"], block["recall"], block["ap"]))
    print("  overall  tp=%d fp=%d fn=%d P=%s R=%s AP=%s"
          % (overall["tp"], overall["fp"], overall["fn"], overall["precision"],
             overall["recall"], overall["ap"]))
    print("  latency", report["latency"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
