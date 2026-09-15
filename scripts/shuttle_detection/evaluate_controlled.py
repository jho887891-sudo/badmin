#!/usr/bin/env python3
"""Measure the shuttlecock baseline's controlled capability (spec 03, plan Task 3).

Usage:
    python scripts/shuttle_detection/evaluate_controlled.py \
        --weights outputs/shuttle_detection/training/baseline_synthetic_real/weights/best.pt \
        --manifest manifest_fixed_core_controlled.csv \
        --out outputs/shuttle_detection/capability/controlled

Why this script exists: spec 03 asks where the baseline detector stops working as
the target gets smaller, turns, moves, blurs or becomes occluded, and it asks for
those curves under one measurement protocol. This adapter is the only place that
knows about Ultralytics: it builds the detector callable and hands it to
src/perception/shuttle_detection/evaluate.py, which does the measuring. That split
exists so the evaluator and its tests run with no GPU, no weights and no
Ultralytics -- and so the inference backend can be replaced without touching a
single metric definition.

Outputs written to --out:
    predictions.csv          one row per manifest sample
    size_bucket_metrics.csv  metrics grouped by equivalent_size_px bucket
    pose_metrics.csv         grouped by pose
    position_metrics.csv     grouped by image position
    blur_metrics.csv         grouped by blur
    occlusion_metrics.csv    grouped by occlusion
    background_metrics.csv   grouped by background
    summary.json             parameters, per-dimension provenance, warnings
A metrics file whose manifest column is absent is written with a header and no data
rows, and the reason is listed in summary.json under "not_computed": a curve that
could not be produced has to be visible in the output rather than missing from it.

Exit codes:
    0  the sweep finished (per-sample detector failures are recorded and counted)
    2  the run could not start (missing manifest or weights, no Ultralytics, bad arguments)

The Ultralytics import is inside a function body, never at module scope: on a
machine without it this script still parses, prints its usage and reports the
missing backend as a plain error instead of a traceback.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.dataset_audit import ManifestError  # noqa: E402
from src.perception.shuttle_detection.evaluate import (  # noqa: E402
    ControlledEvaluationConfig,
    evaluate_controlled,
    write_controlled_reports,
)
from src.perception.shuttle_detection.metrics import DEFAULT_IOU_THRESHOLD  # noqa: E402

EXIT_OK = 0
EXIT_UNUSABLE_INPUT = 2

# Values accepted by --expected-class that mean "do not filter on class".
ANY_CLASS = {"none", "any", "all", "-1"}


class DetectorUnavailableError(RuntimeError):
    """The inference backend this script needs is not usable on this host."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure controlled shuttlecock detection capability from a manifest."
    )
    parser.add_argument(
        "--weights", required=True, metavar="PT", help="trained detector checkpoint"
    )
    parser.add_argument(
        "--manifest", required=True, metavar="CSV", help="controlled capability manifest"
    )
    parser.add_argument(
        "--out", required=True, metavar="DIR", help="directory for the report files"
    )
    parser.add_argument(
        "--images",
        default=None,
        metavar="DIR",
        help="directory holding the images; defaults to the manifest's own directory",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument(
        "--conf", type=float, default=0.05, help="detector confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.7, help="detector NMS IoU threshold"
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        dest="match_iou",
        help="IoU at which a prediction counts as a match (metrics, not NMS)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, dest="top_k", help="k for the Top-K hit rate"
    )
    parser.add_argument(
        "--expected-class",
        default=0,
        type=expected_class_argument,
        dest="expected_class",
        metavar="ID",
        help="class id to score, or '" + "/".join(sorted(ANY_CLASS)) + "' to keep every class",
    )
    parser.add_argument(
        "--device", default=None, help="inference device (for example 0 or cpu)"
    )
    return parser.parse_args(argv)


def build_ultralytics_detector(
    weights: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
) -> Any:
    """Return detector(image_path) -> list of {bbox, confidence, class_id}.

    Ultralytics is imported here rather than at module scope so that importing this
    script - or reading its usage - never requires the inference stack.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - inference-only dependency
    except ImportError as error:
        raise DetectorUnavailableError(
            "ultralytics is not installed on this host, so no detector can be built: "
            + str(error)
        ) from error

    model = YOLO(str(weights))

    def detector(image_path: Path) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "source": str(image_path),
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device
        results = model.predict(**kwargs)
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        return [
            {
                "bbox": (float(one[0]), float(one[1]), float(one[2]), float(one[3])),
                "confidence": float(score),
                "class_id": int(class_id),
            }
            for one, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls)
        ]

    return detector


def expected_class_argument(value: str) -> int | None:
    """Parse --expected-class, or fail through argparse with a usage message.

    The class to score is part of the run's identity, so a value that is neither an
    integer nor an explicit "any class" word must be rejected as a bad argument
    rather than raising somewhere inside the sweep.
    """
    text = str(value).strip().lower()
    if text in ANY_CLASS:
        return None
    try:
        return int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected class must be an integer or one of "
            + "/".join(sorted(ANY_CLASS))
            + ", got "
            + repr(value)
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    manifest = Path(args.manifest)
    weights = Path(args.weights)
    if not manifest.is_file():
        print("ERROR: manifest not found: " + str(manifest), file=sys.stderr)
        return EXIT_UNUSABLE_INPUT
    if not weights.is_file():
        print("ERROR: weights not found: " + str(weights), file=sys.stderr)
        return EXIT_UNUSABLE_INPUT

    config = ControlledEvaluationConfig(
        iou_threshold=args.match_iou,
        top_k=args.top_k,
        expected_class=args.expected_class,
        image_root=None if args.images is None else Path(args.images),
    )

    try:
        detector = build_ultralytics_detector(
            weights, args.imgsz, args.conf, args.iou, args.device
        )
    except DetectorUnavailableError as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return EXIT_UNUSABLE_INPUT

    print(
        "evaluate_controlled: "
        + str(manifest)
        + " with "
        + str(weights)
        + " (imgsz="
        + str(args.imgsz)
        + ", conf="
        + str(args.conf)
        + ", match_iou="
        + str(args.match_iou)
        + ", top_k="
        + str(args.top_k)
        + ")"
    )

    try:
        report = evaluate_controlled(manifest, detector, config)
    except (ManifestError, OSError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return EXIT_UNUSABLE_INPUT

    written = write_controlled_reports(report, args.out)
    summary = report.summary

    for warning in summary["warnings"]:
        print(warning, file=sys.stderr)

    print(
        "  samples="
        + str(summary["n_samples"])
        + " evaluable="
        + str(summary["n_evaluable"])
        + " negative="
        + str(summary["n_negative"])
        + " missing_ground_truth="
        + str(summary["n_missing_ground_truth"])
        + " detector_errors="
        + str(summary["n_detector_errors"])
    )
    for dimension in sorted(report.groups):
        entry = summary["dimensions"][dimension]
        print(
            "  "
            + dimension
            + ": "
            + entry["status"]
            + " ("
            + str(entry["n_groups"])
            + " group(s)) -> "
            + str(written[dimension])
        )
    overall = summary["overall"]
    print(
        "  overall: precision="
        + str(overall["precision"])
        + " recall="
        + str(overall["recall"])
        + " mAP50="
        + str(overall["mAP50"])
        + " top1="
        + str(overall["top1_hit_rate"])
        + " topk="
        + str(overall["topk_hit_rate"])
    )
    print("  predictions " + str(written["predictions"]))
    print("  summary " + str(written["summary"]))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
