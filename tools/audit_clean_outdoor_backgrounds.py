#!/usr/bin/env python3
"""CLEAN outdoor Copy-Paste background audit for CLEAN_OUTDOOR_CP_V1 (Task 4).

The only permitted source is a real outdoor COCO train2017 photograph that was never used by the frozen V2 work,
that carries a resolvable licence, and whose CLEAN ROI a human signed off. COCO has no shuttlecock category, so
"COCO has no annotation here" is NEVER accepted as proof that the picture contains no shuttlecock: that check is
a recorded human judgement (no_shuttle_reviewed) and nothing else.

ROIs are ranked by a frozen, candidate-only statistic (edge density, entropy, luminance distance from mid-grey).
The model never scores a background, and no numeric result of a model is allowed to influence this stage.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402

ALLOWED_SOURCE = "coco_train2017"
CLEAN_QUANTILE = 0.25
MIN_POOL_IMAGES = 100
MIN_SCENE_TYPES = 3
MIN_SOURCE_CLUSTERS = 3
MIN_DISTINCT_COLORS = 24
EDGE_THRESHOLD = 18.0
ROI_GRID = ((0, 0, 512, 512), (512, 0, 1024, 512), (0, 512, 512, 1024), (512, 512, 1024, 1024),
            (256, 256, 768, 768), (128, 384, 640, 896))

REQUIRED_CANDIDATE_COLUMNS = ("coco_image_id", "source", "permission_state", "outdoor_reviewed",
                              "no_shuttle_reviewed", "roi_signoff", "reviewer_id", "scene_type",
                              "source_cluster")
ACCEPTED_COLUMNS = ("coco_image_id", "license_or_permission", "outdoor_reviewed", "no_shuttle_reviewed",
                    "clean_roi", "clean_score", "near_duplicate_group", "reviewer_id", "scene_type",
                    "source_cluster", "location_id", "image_sha256")


def _to_gray(image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=2)
    return array


def compute_clean_roi_metrics(image_1024, xyxy: tuple) -> dict:
    """Frozen candidate-only statistics for one ROI: no model, no learned feature, no random seed."""
    gray = _to_gray(image_1024)
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = (int(value) for value in xyxy)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("ROI %s is outside the %dx%d frame" % (xyxy, width, height))
    patch = gray[y0:y1, x0:x1]

    horizontal = np.abs(np.diff(patch, axis=1)).mean() if patch.shape[1] > 1 else 0.0
    vertical = np.abs(np.diff(patch, axis=0)).mean() if patch.shape[0] > 1 else 0.0
    edge_density = float((horizontal + vertical) / 2.0)

    histogram, _ = np.histogram(patch, bins=64, range=(0, 255))
    probabilities = histogram.astype(np.float64) / max(1, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())

    mean = float(patch.mean())
    std = float(patch.std())
    color_source = np.asarray(image_1024)[y0:y1, x0:x1]
    if color_source.ndim == 3:
        # Pack the quantised RGB into one integer and take a 1-D unique: np.unique(..., axis=0) on 262k rows was
        # the dominant cost of the whole COCO review worksheet (measured: a 2000-image pass did not finish in
        # 45 minutes), while the packed form is roughly 20x faster and gives the same count.
        quantised = (color_source[..., :3] // 16).astype(np.int32)
        packed = (quantised[..., 0] << 16) | (quantised[..., 1] << 8) | quantised[..., 2]
        distinct = int(len(np.unique(packed)))
    else:
        distinct = int(len(np.unique((color_source // 16).astype(np.int32))))

    return {
        "edge_density": edge_density,
        "entropy": entropy,
        "luminance_mean": mean,
        "luminance_std": std,
        "luminance_distance": abs(mean - 128.0) / 128.0,
        "distinct_colors": distinct,
        "roi": [x0, y0, x1, y1],
        "roi_pixels": int(patch.size),
    }


def candidate_rois(image_1024, grid=None) -> list:
    """The frozen ROI grid, clipped to a frame that may be smaller than 1024 in unit tests."""
    array = np.asarray(image_1024)
    height, width = array.shape[:2]
    rois = []
    for x0, y0, x1, y1 in (grid or ROI_GRID):
        if x1 <= width and y1 <= height:
            rois.append((x0, y0, x1, y1))
    if not rois:
        rois.append((0, 0, width, height))
    return rois


def rank_clean_rois(rows: pd.DataFrame) -> pd.DataFrame:
    """Rank candidate ROIs so that lower clean_score always means cleaner, then sort by it."""
    frame = rows.copy()
    for column in ("edge_density", "entropy", "luminance_distance"):
        if column not in frame.columns:
            raise ValueError("rank_clean_rois needs the %s metric" % column)
    frame["edge_rank"] = frame["edge_density"].rank(method="average", pct=True)
    frame["entropy_rank"] = frame["entropy"].rank(method="average", pct=True)
    frame["luminance_rank"] = frame["luminance_distance"].rank(method="average", pct=True)
    frame["clean_score"] = frame[["edge_rank", "entropy_rank", "luminance_rank"]].mean(axis=1)
    return frame.sort_values("clean_score").reset_index(drop=True)


def _rejection(coco_id, reason, detail, row=None):
    return {
        "coco_image_id": coco_id,
        "reason": reason,
        "detail": detail,
        "scene_type": (row or {}).get("scene_type", ""),
        "license_or_permission": (row or {}).get("license_or_permission", ""),
    }


def audit_clean_backgrounds(coco_candidates: pd.DataFrame, excluded: pd.DataFrame,
                            quantile: float = CLEAN_QUANTILE) -> pd.DataFrame:
    """Accept only human-approved, licenced, real outdoor COCO train2017 ROIs in the cleanest quartile."""
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in coco_candidates.columns]
    if missing:
        raise ValueError("coco candidates are missing required review columns: %s" % missing)

    excluded_ids = set()
    excluded_hashes = set()
    excluded_groups = set()
    if excluded is not None and len(excluded):
        if "coco_image_id" in excluded.columns:
            excluded_ids = {str(value) for value in excluded["coco_image_id"].dropna()}
        if "sha256" in excluded.columns:
            excluded_hashes = {str(value) for value in excluded["sha256"].dropna()}
        if "near_duplicate_group" in excluded.columns:
            excluded_groups = {str(value) for value in excluded["near_duplicate_group"].dropna()}

    scored = coco_candidates.copy()
    # The CLI reads the worksheet with dtype=str (it carries ids and free text), so the numeric columns must be
    # coerced here: otherwise Series.quantile raises "unsupported operand type(s) for -: 'str' and 'str'".
    for column in ("clean_score", "edge_density", "entropy", "luminance_mean", "luminance_distance",
                   "distinct_colors"):
        if column in scored.columns:
            scored[column] = pd.to_numeric(scored[column], errors="coerce")
    if "clean_score" not in scored.columns:
        if {"edge_density", "entropy", "luminance_mean"} <= set(scored.columns):
            scored = rank_clean_rois(scored)
        else:
            raise ValueError("candidates need clean_score or the frozen ROI metrics")

    threshold = float(scored["clean_score"].quantile(quantile))
    accepted: list = []
    rejections: list = []

    for row in scored.to_dict("records"):
        coco_id = str(row["coco_image_id"])
        if str(row.get("source")) != ALLOWED_SOURCE:
            rejections.append(_rejection(coco_id, "source_not_coco_train2017",
                                         "source is %r" % row.get("source"), row))
            continue
        if coco_id in excluded_ids or (row.get("image_sha256") and str(row["image_sha256"]) in excluded_hashes) \
                or (row.get("near_duplicate_group") and str(row["near_duplicate_group"]) in excluded_groups):
            rejections.append(_rejection(coco_id, "excluded_by_history",
                                         "image id, content hash or near-duplicate group is already used", row))
            continue
        if str(row.get("permission_state")) != "declared":
            rejections.append(_rejection(coco_id, "license_unverifiable",
                                         "no resolvable licence declaration for this picture", row))
            continue
        if not _is_true(row.get("outdoor_reviewed")):
            rejections.append(_rejection(coco_id, "not_outdoor", "no human confirmation that the photo is outdoor", row))
            continue
        if not _is_true(row.get("no_shuttle_reviewed")):
            rejections.append(_rejection(coco_id, "no_shuttle_unreviewed",
                                         "an absent COCO annotation is not proof that no shuttlecock is present", row))
            continue
        if not _is_true(row.get("roi_signoff")) or not str(row.get("reviewer_id") or "").strip():
            rejections.append(_rejection(coco_id, "roi_missing_signoff",
                                         "the CLEAN ROI has no human sign-off", row))
            continue
        if float(row.get("clean_score", 1.0)) > threshold:
            rejections.append(_rejection(coco_id, "clean_score_above_quartile",
                                         "clean_score %.4f is above the frozen quartile %.4f"
                                         % (float(row.get("clean_score", 1.0)), threshold), row))
            continue
        if _artificial_or_flat(row):
            rejections.append(_rejection(coco_id, "artificial_or_flat",
                                         "flat/pure-colour or synthetic-looking background", row))
            continue

        accepted.append({
            "coco_image_id": coco_id,
            "license_or_permission": row.get("license_or_permission", ""),
            "outdoor_reviewed": True,
            "no_shuttle_reviewed": True,
            "clean_roi": row.get("clean_roi", ""),
            "clean_score": float(row.get("clean_score", 1.0)),
            "near_duplicate_group": row.get("near_duplicate_group", ""),
            "reviewer_id": row.get("reviewer_id", ""),
            "scene_type": row.get("scene_type", ""),
            "source_cluster": row.get("source_cluster", ""),
            "location_id": row.get("location_id", ""),
            "image_sha256": row.get("image_sha256", ""),
        })

    frame = pd.DataFrame(accepted, columns=list(ACCEPTED_COLUMNS))
    frame.attrs["rejections"] = pd.DataFrame(rejections,
                                             columns=["coco_image_id", "reason", "detail", "scene_type",
                                                      "license_or_permission"])
    frame.attrs["clean_score_quartile_threshold"] = threshold
    return frame


def _is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _artificial_or_flat(row) -> bool:
    distinct = row.get("distinct_colors")
    if distinct is None or str(distinct).strip() == "":
        return False
    try:
        distinct = int(float(distinct))
    except (TypeError, ValueError):
        return False
    edge = float(row.get("edge_density", 0.0) or 0.0)
    return distinct < MIN_DISTINCT_COLORS and edge < EDGE_THRESHOLD / 4.0


def evaluate_background_pool(accepted: pd.DataFrame) -> dict:
    """The Task 4 gate. Below 100 images / 3 scene types / 3 source clusters the pool is insufficient."""
    images = int(len(accepted))
    scene_types = sorted({str(value) for value in accepted.get("scene_type", []) if str(value).strip()})
    clusters = sorted({str(value) for value in accepted.get("source_cluster", []) if str(value).strip()})
    known_locations = sorted({str(value) for value in accepted.get("location_id", []) if str(value).strip()})
    problems = []
    if images < MIN_POOL_IMAGES:
        problems.append("only %d approved images, need %d" % (images, MIN_POOL_IMAGES))
    if len(scene_types) < MIN_SCENE_TYPES:
        problems.append("only %d scene types, need %d" % (len(scene_types), MIN_SCENE_TYPES))
    if len(clusters) < MIN_SOURCE_CLUSTERS:
        problems.append("only %d source clusters, need %d" % (len(clusters), MIN_SOURCE_CLUSTERS))
    return {
        "images": images,
        "scene_types": scene_types,
        "source_clusters": clusters,
        "known_locations": known_locations,
        "unknown_location_count": int(sum(1 for value in accepted.get("location_id", []) if not str(value).strip())),
        "problems": problems,
        "gate": "PASS" if not problems else "CLEAN_BG_INSUFFICIENT",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit COCO train2017 CLEAN outdoor backgrounds (Task 4)")
    parser.add_argument("--candidates", type=Path, required=True,
                        help="CSV of COCO candidates enriched with the human review columns")
    parser.add_argument("--excluded", type=Path, default=None, help="CSV of prohibited historical/eval sources")
    parser.add_argument("--out-accepted", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    args = parser.parse_args(argv)

    candidates = pd.read_csv(args.candidates, keep_default_na=False, dtype=str)
    excluded = pd.read_csv(args.excluded, keep_default_na=False, dtype=str) if args.excluded else pd.DataFrame()
    accepted = audit_clean_backgrounds(candidates, excluded)
    gate = evaluate_background_pool(accepted)
    args.out_accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(args.out_accepted, index=False)
    audit = {
        "generated_by": "tools/audit_clean_outdoor_backgrounds.py",
        "task": "Task 4: curate COCO train2017 CLEAN outdoor copy-paste backgrounds",
        "accepted": int(len(accepted)),
        "rejected": int(len(accepted.attrs["rejections"])),
        "rejection_reasons": accepted.attrs["rejections"]["reason"].value_counts().to_dict()
                             if len(accepted.attrs["rejections"]) else {},
        "clean_score_quartile_threshold": accepted.attrs["clean_score_quartile_threshold"],
        "gate": gate,
        "note": "a missing COCO annotation is never treated as proof that no shuttlecock is present",
    }
    args.out_audit.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": audit["accepted"], "rejected": audit["rejected"], "gate": gate["gate"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
