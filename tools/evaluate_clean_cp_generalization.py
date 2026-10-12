#!/usr/bin/env python3
"""Canonical evaluation and real-domain FP diagnostics for CLEAN_OUTDOOR_CP_V1 (Task 9).

This adapter never measures anything itself. It pins the frozen evaluation contract (imgsz 1024, operating
confidence 0.25, AP floor 0.001, NMS IoU 0.7, max_det 300, equiv_size_640 = sqrt(w_640 * h_640)), drives the
existing canonical evaluator tools/eval_yolo26_v1.py unchanged, and derives source-aware / location-aware
TP/FP/FN from the very same matcher and AP code the frozen V2 reference used. A second matcher is never
written, and a stored V2 reference must replay identically before any new checkpoint is scored - otherwise the
outcome is EVALUATOR_PARITY_FAIL and every claim stops.

Three honest outcomes are possible and none of them is a number:

* EVALUATED      - a frozen checkpoint and a frozen manifest set exist and the canonical evaluator produced
                   its own output files; only then may the raw denominators be reported.
* PILOT_ONLY     - no independent real outdoor test set was supplied, so outdoor generalization is not
                   claimed, whatever the ETH-set scores say.
* EVALUATOR_PARITY_FAIL / EvalBlocked - the stored V2 reference does not replay, or no checkpoint exists, so
                   nothing is measured.

The false-positive taxonomy produced here is diagnostics only: it carries role EVAL_DIAGNOSTIC_ONLY and
training_use False, and cannot be fed back into any training or candidate-mining pool (plan Task 9).

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 9).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402
from eval_yolo26_v1 import ap_for_gt_set, equiv_size_640, match_greedy  # noqa: E402,F401

REPO = Path(__file__).resolve().parents[1]
NEW_CONFIG_PATH = "configs/eth_real_clean_outdoor_copypaste_v1.yaml"
CANONICAL_EVALUATOR_PATH = "tools/eval_yolo26_v1.py"

# --- the frozen evaluation contract (plan Global Constraint 10 / Task 9 Step 3) ----------------------
IMGSZ = 1024
CONF_OP = 0.25
CONF_FLOOR = 0.001
NMS_IOU = 0.7
MAX_DET = 300
PRIMARY_IOU = 0.5
CENTER_DIST_PX = 25.0
EQUIV_SIZE_NORMALIZED = "sqrt(w_640 * h_640)"
EQUIV_SIZE_FORMULA = "equiv_size_640 = " + EQUIV_SIZE_NORMALIZED
FROZEN_EVAL_VALUES = {"imgsz": IMGSZ, "conf": CONF_OP, "ap_floor": CONF_FLOOR,
                      "nms_iou": NMS_IOU, "max_det": MAX_DET}
FROZEN_EVAL_KEYS = ("imgsz", "conf", "ap_floor", "nms_iou", "max_det")
CANONICAL_PINS = ("iou=0.7", "max_det=300", "conf=conf_floor")

# --- frozen pools, gates and outcomes ----------------------------------------------------------------
NO_TARGET_POOL_SIZE = 496
NO_TARGET_FP_GATE = 3
V2_HISTORICAL_NO_TARGET_FP = 1
OUTDOOR_REAL_OOD_SET = "outdoor_real_ood"
ETH_UNSEEN_SET = "val|eth_unseen"

OUTCOME_EVALUATED = "EVALUATED"
OUTCOME_PILOT_ONLY = "PILOT_ONLY"
OUTCOME_EVALUATOR_PARITY_FAIL = "EVALUATOR_PARITY_FAIL"
OUTCOME_NOT_EXECUTED = "NOT_EXECUTED"

# --- source domains: the frozen val|eth_unseen vocabulary (never guessed, never merged) --------------
SOURCE_REAL = "real"
SOURCE_SYNTHETIC = "synthetic"
REAL_SOURCES = frozenset({"eth_main", "eth_iphone", "coco_bg", "bg_negative"})
SYNTHETIC_SOURCES = frozenset({"synthetic"})

# --- the read-only false-positive taxonomy -----------------------------------------------------------
FP_TAXONOMY_ROLE = "EVAL_DIAGNOSTIC_ONLY"
FP_TAXONOMY_UNREVIEWED = "hardness_unreviewed"
FP_REVIEWED_CATEGORIES = ("white_reflection", "paper_plastic", "feather_leaf", "metal_glint",
                          "bright_signage", "other_reviewed")
FP_KIND_LOCALIZATION = "localization_miss"
FP_KIND_BACKGROUND = "background_false_positive"
LOCALIZATION_IOU_FLOOR = 0.10
PREDICTION_COLUMNS = ("image", "conf", "x1", "y1", "x2", "y2", "matched")
FP_TAXONOMY_COLUMNS = ("image", "conf", "x1", "y1", "x2", "y2", "reviewed_category", "taxonomy",
                       "reviewed", "fp_kind", "taxonomy_role")
TRAINING_POOL_COLUMNS = ("label_path", "is_positive", "repeat_index", "slot_index", "exposure_index")
OUTDOOR_MANIFEST_COLUMNS = ("image", "label", "location", "session", "sha256")

PARITY_FIELDS = ("images", "GT", "TP", "FP", "FN", "mAP50", "mAP50-95")

EVAL_OUTPUT_FILES = ("outdoor_real_ood.csv", "eth_unseen_all.csv", "eth_unseen_real.csv",
                     "eth_unseen_synthetic.csv", "real_no_target_496.csv", "size_buckets_by_source.csv",
                     "negative_error_taxonomy.csv", "eval_integrity.json")


class EvalBlocked(RuntimeError):
    """Raised when an evaluation cannot be executed honestly (missing checkpoint, manifest or metric)."""


class EvalContractViolation(RuntimeError):
    """Raised when the evaluation settings drift from the frozen contract."""


class EvaluatorParityFail(RuntimeError):
    """Raised when a stored V2 reference does not replay identically with the canonical evaluator."""


class EvalIntegrityError(RuntimeError):
    """Raised on a leakage, a merged real/synthetic table or a recyclable evaluation example."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_records(rows) -> list:
    """Normalise a DataFrame or a sequence of mappings into a list of dicts."""
    if rows is None:
        raise EvalIntegrityError("expected rows, got None")
    if isinstance(rows, pd.DataFrame):
        return [dict(row) for row in rows.to_dict("records")]
    records = []
    for row in rows:
        if not hasattr(row, "keys"):
            raise EvalIntegrityError("expected mappings, got %r" % (type(row).__name__,))
        records.append(dict(row))
    return records


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _same(actual, frozen) -> bool:
    """Numeric equality tolerant of YAML/CSV round-trips, but never of a boolean stand-in."""
    if isinstance(actual, bool) or isinstance(frozen, bool):
        return actual is frozen
    try:
        return float(actual) == float(frozen)
    except (TypeError, ValueError):
        return False


# =====================================================================================================
# 1. the frozen evaluation contract
# =====================================================================================================
def frozen_evaluator_cfg() -> dict:
    """The settings this adapter passes to the canonical evaluator, as one auditable mapping."""
    return {"imgsz": IMGSZ, "conf": CONF_OP, "ap_floor": CONF_FLOOR, "nms_iou": NMS_IOU, "max_det": MAX_DET,
            "size_bucket": EQUIV_SIZE_FORMULA}


def load_spec_eval_cfg(config_path=NEW_CONFIG_PATH, repo=REPO) -> dict:
    """Read the frozen 'evaluation' block of the experiment config; a missing block is blocking."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(repo) / path
    if not path.is_file():
        raise EvalBlocked("the frozen experiment config is missing, so the evaluation contract cannot be "
                          "checked: %s (BLOCKED_MISSING_V2_ARTIFACTS)" % path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("evaluation"), dict):
        raise EvalBlocked("the frozen experiment config has no 'evaluation' block: %s" % path)
    return dict(document["evaluation"])


def _normalize_size_formula(text) -> str:
    normalized = str(text).strip().lower().replace(" ", "")
    for prefix in ("equiv_size_640=", "size_bucket=", "equiv_size="):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


def assert_equiv_size_formula() -> str:
    """Check the size-bucket rule numerically against the canonical evaluator, never by restating it."""
    probes = ((10.0, 10.0, 1920, 1200), (6.0, 24.0, 1000, 1000), (0.0, 5.0, 640, 480), (3.0, 7.0, 1920, 1080))
    for width, height, frame_w, frame_h in probes:
        scale = 640.0 / max(frame_w, frame_h)
        expected = math.sqrt(max(width * scale, 0.0) * max(height * scale, 0.0))
        actual = equiv_size_640(width, height, frame_w, frame_h)
        if abs(float(actual) - expected) > 1e-9:
            raise EvalContractViolation(
                "the canonical evaluator's equiv_size_640 is not %s: for (w=%s, h=%s, W=%s, H=%s) it returned "
                "%r, expected %r" % (EQUIV_SIZE_NORMALIZED, width, height, frame_w, frame_h, actual, expected)
            )
    return EQUIV_SIZE_NORMALIZED


def assert_canonical_evaluator_pins() -> dict:
    """Convention-drift alarm: the canonical evaluator must keep predicting with the frozen NMS settings."""
    path = Path(REPO) / CANONICAL_EVALUATOR_PATH
    if not path.is_file():
        raise EvalBlocked("the canonical evaluator is missing: %s" % path)
    source = path.read_text(encoding="utf-8")
    missing = [token for token in CANONICAL_PINS if token not in source]
    if missing:
        raise EvalContractViolation(
            "the canonical evaluator no longer predicts with the frozen settings (missing in source: %s): "
            "re-check the evaluator before scoring anything" % missing
        )
    return {"path": CANONICAL_EVALUATOR_PATH, "sha256": sha256_file(path), "pins": list(CANONICAL_PINS)}


def assert_eval_contract(evaluator_cfg, spec_cfg) -> None:
    """Both the adapter settings and the frozen spec must state the same evaluation contract.

    Frozen: imgsz=1024, operating conf=0.25, AP floor=0.001, NMS IoU=0.7, max_det=300 and
    equiv_size_640 = sqrt(w_640 * h_640). Any missing or drifted field is an error, never a default.
    """
    problems: list = []
    for name, cfg in (("evaluator_cfg", evaluator_cfg), ("spec_cfg", spec_cfg)):
        if not isinstance(cfg, dict):
            problems.append("%s must be a mapping of evaluation settings, got %s"
                            % (name, type(cfg).__name__))
            continue
        for key in FROZEN_EVAL_KEYS:
            if key not in cfg:
                problems.append("%s omits the frozen setting %s=%r" % (name, key, FROZEN_EVAL_VALUES[key]))
            elif not _same(cfg[key], FROZEN_EVAL_VALUES[key]):
                problems.append("%s.%s=%r, frozen value is %r" % (name, key, cfg[key], FROZEN_EVAL_VALUES[key]))
        formula = cfg.get("size_bucket", cfg.get("equiv_size"))
        if formula is None:
            problems.append("%s omits the frozen size-bucket formula (%s)" % (name, EQUIV_SIZE_FORMULA))
        elif _normalize_size_formula(formula) != _normalize_size_formula(EQUIV_SIZE_NORMALIZED):
            problems.append("%s size_bucket=%r, frozen rule is %r" % (name, formula, EQUIV_SIZE_FORMULA))

    if problems:
        raise EvalContractViolation("evaluation contract drift: " + "; ".join(problems))

    assert_equiv_size_formula()
    assert_canonical_evaluator_pins()


# =====================================================================================================
# 2. source domains: real and synthetic are never merged
# =====================================================================================================
def domain_of_source(source) -> str:
    """Map a manifest source onto the frozen real / synthetic domain, refusing anything undeclared."""
    value = str(source if source is not None else "").strip().lower()
    if value in SYNTHETIC_SOURCES:
        return SOURCE_SYNTHETIC
    if value in REAL_SOURCES:
        return SOURCE_REAL
    raise EvalIntegrityError(
        "unknown evaluation source %r: the real and synthetic domains may never be merged implicitly. "
        "Declare the source in REAL_SOURCES or SYNTHETIC_SOURCES first (known real: %s; synthetic: %s)"
        % (source, sorted(REAL_SOURCES), sorted(SYNTHETIC_SOURCES))
    )


def assert_sources_not_merged(rows) -> None:
    """Every source-aware row must belong to exactly one domain; a merged row is refused."""
    records = _as_records(rows)
    for index, row in enumerate(records):
        if "domain" not in row:
            raise EvalIntegrityError("source-aware row %d declares no domain (real|synthetic): %r"
                                     % (index, sorted(row)))
        domain = str(row["domain"]).strip()
        if domain not in (SOURCE_REAL, SOURCE_SYNTHETIC):
            raise EvalIntegrityError(
                "row %d merges domains as %r: real and synthetic - including the tiny buckets - must always "
                "stay in separate rows" % (index, domain)
            )
        if _truthy(row.get("merged")):
            raise EvalIntegrityError("row %d is marked merged: real and synthetic rows are never aggregated"
                                     % index)


def _count(sample: dict, name: str) -> int:
    if name not in sample:
        raise EvalIntegrityError("an evaluation sample must declare %s explicitly; got %r" % (name, sorted(sample)))
    return int(float(sample[name]))


def size_bucket_rows(set_key, samples) -> list:
    """One row per (set, domain, bucket): real and synthetic keep separate denominators.

    Each input sample is one evaluated frame of one bucket and must declare bucket, source, GT, TP and FN
    (images defaults to one per sample unless the caller states otherwise).
    """
    rows: list = []
    for sample in samples:
        bucket = str(sample.get("bucket", "")).strip()
        if bucket not in ev.BUCKET_LABELS:
            raise EvalIntegrityError("bucket %r is not one of the canonical buckets %s" % (bucket, ev.BUCKET_LABELS))
        domain = domain_of_source(sample.get("source"))
        rows.append({"set": str(set_key), "bucket": bucket, "domain": domain,
                     "sources": str(sample.get("source")), "GT": _count(sample, "GT"),
                     "TP": _count(sample, "TP"), "FN": _count(sample, "FN"),
                     "images": int(float(sample.get("images", 1)))})

    merged: dict = {}
    for row in rows:
        key = (row["set"], row["domain"], row["bucket"])
        if key not in merged:
            merged[key] = dict(row)
            merged[key]["GT"] = 0
            merged[key]["TP"] = 0
            merged[key]["FN"] = 0
            merged[key]["images"] = 0
        merged[key]["GT"] += row["GT"]
        merged[key]["TP"] += row["TP"]
        merged[key]["FN"] += row["FN"]
        merged[key]["images"] += row["images"]
        merged[key]["sources"] = ",".join(sorted(set(merged[key]["sources"].split(",")) | {row["sources"]}))
    out = [merged[key] for key in sorted(merged)]
    assert_sources_not_merged(out)
    return out


def partition_by_domain(rows, fields=("GT", "TP", "FN", "images")) -> dict:
    """Sum source-aware rows per domain; a merged input is refused, so no total can hide the split."""
    records = _as_records(rows)
    assert_sources_not_merged(records)
    out = {SOURCE_REAL: {"rows": 0}, SOURCE_SYNTHETIC: {"rows": 0}}
    for row in records:
        domain = str(row["domain"]).strip()
        out[domain]["rows"] += 1
        for field in fields:
            if field in row:
                out[domain][field] = out[domain].get(field, 0) + int(float(row[field]))
    return out


def stratify_eth_unseen(rows) -> dict:
    """Split val|eth_unseen rows into all / real / synthetic, tagging each row with its frozen domain."""
    tagged: list = []
    for row in _as_records(rows):
        record = dict(row)
        record["domain"] = domain_of_source(record.get("source"))
        tagged.append(record)
    assert_sources_not_merged(tagged)
    return {"all": tagged,
            SOURCE_REAL: [row for row in tagged if row["domain"] == SOURCE_REAL],
            SOURCE_SYNTHETIC: [row for row in tagged if row["domain"] == SOURCE_SYNTHETIC]}


# =====================================================================================================
# 3. the independent real outdoor test set
# =====================================================================================================
def assert_outdoor_test_isolation(outdoor_records, training_records) -> dict:
    """The outdoor test set must be disjoint from EVERY training location and session (plus path/hash).

    An outdoor record that cannot name its location, session and content hash cannot prove independence, so
    it is refused rather than accepted on faith. Absence of an outdoor test set is PILOT_ONLY, not a pass.
    """
    outdoor = list(outdoor_records)
    training = list(training_records)
    if not outdoor:
        raise EvalIntegrityError(
            "no outdoor test records were supplied: a missing real outdoor test set is PILOT_ONLY, never "
            "evidence of independence"
        )

    violations: list = []
    train_hashes = {record.sha256 for record in training if record.sha256}
    train_paths = {str(record.path) for record in training if record.path}
    train_locations = {record.location_id for record in training if record.location_id}
    train_sessions = {record.session_id for record in training if record.session_id}

    for record in outdoor:
        if record.role != C.ROLE_FROZEN_EVAL:
            violations.append("outdoor record %s carries role %r: an evaluation image may never carry a "
                              "training role" % (record.source_id, record.role))
        for field in ("location_id", "session_id", "sha256"):
            if not str(getattr(record, field, "") or "").strip():
                violations.append("outdoor record %s does not declare %s, so location/session independence "
                                  "cannot be proven" % (record.source_id, field))
        if record.sha256 and record.sha256 in train_hashes:
            violations.append("outdoor record %s shares the content hash of a training source (sha256 %s)"
                              % (record.source_id, record.sha256))
        if record.path and str(record.path) in train_paths:
            violations.append("outdoor record %s shares path %s with a training source"
                              % (record.source_id, record.path))
        if record.location_id and record.location_id in train_locations:
            violations.append("outdoor record %s reuses training location_id %s"
                              % (record.source_id, record.location_id))
        if record.session_id and record.session_id in train_sessions:
            violations.append("outdoor record %s reuses training session_id %s"
                              % (record.source_id, record.session_id))

    if violations:
        raise EvalIntegrityError("the outdoor test set is not independent: " + "; ".join(violations[:6]))
    return {"status": "PASS", "outdoor_images": len(outdoor), "training_sources": len(training),
            "checked": ["sha256", "path", "location_id", "session_id"],
            "outdoor_locations": sorted({record.location_id for record in outdoor if record.location_id}),
            "outdoor_sessions": sorted({record.session_id for record in outdoor if record.session_id})}


def outdoor_manifest_report(manifest) -> dict:
    """A declared outdoor OOD manifest must be non-empty and prove location/session/hash provenance."""
    path = Path(manifest)
    if not path.is_file():
        raise EvalBlocked("the declared independent outdoor test manifest is missing: %s" % path)
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [column for column in OUTDOOR_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise EvalBlocked(
            "the independent outdoor test manifest must declare %s so location/session independence is "
            "checkable; %s misses %s" % (list(OUTDOOR_MANIFEST_COLUMNS), path, missing)
        )
    if len(frame) == 0:
        raise EvalBlocked("the declared independent outdoor test manifest is empty: %s" % path)
    for column in ("location", "session", "sha256"):
        blank = sum(1 for value in frame[column] if not str(value).strip())
        if blank:
            raise EvalBlocked("outdoor manifest %s has %d rows without %s: independence cannot be proven"
                              % (path, blank, column))
    return {"path": str(path), "images": int(len(frame)), "labeled": True, "sha256": sha256_file(path),
            "locations": sorted(set(frame["location"])), "sessions": sorted(set(frame["session"]))}


# =====================================================================================================
# 4. false-positive counting: detections, never images
# =====================================================================================================
def fp_count(detections_per_image, conf_op=CONF_OP) -> dict:
    """False-positive statistics on an all-negative set: FP is a DETECTION count, never an image count.

    The arithmetic is the canonical evaluator's own tools/eval_yolo26_v1.py::fp_stats; this wrapper only
    names the quantities explicitly so an image count can never be reported as an FP count.
    """
    per_image = [list(dets or []) for dets in detections_per_image]
    stats = ev.fp_stats(per_image, [float(conf_op)])[str(float(conf_op))]
    kept = sum(1 for dets in per_image for det in dets if float(det["conf"]) >= float(conf_op))
    return {"FP": int(stats["total_FP"]), "images": int(stats["total_images"]),
            "images_with_FP": int(stats["images_with_FP"]), "detections": int(kept),
            "detections_all": int(sum(len(dets) for dets in per_image)),
            "FP_per_image": stats["FP_per_image"], "image_FP_rate": stats["image_FP_rate"],
            "max_confidence_FP": stats["max_confidence_FP"],
            "mean_confidence_FP": stats["mean_confidence_FP"],
            "median_confidence_FP": stats["median_confidence_FP"], "conf_op": float(conf_op)}


def no_target_row(set_key, stats) -> dict:
    """One report row for an all-negative frozen pool, keeping FP and image counts as separate columns."""
    return {"set": str(set_key), "kind": "no_target_negative_pool", "images": int(stats["images"]),
            "images_with_FP": int(stats["images_with_FP"]), "FP": int(stats["FP"]),
            "detections": int(stats["detections"]), "FP_per_image": stats["FP_per_image"],
            "image_FP_rate": stats["image_FP_rate"], "max_confidence_FP": stats["max_confidence_FP"],
            "conf_op": stats["conf_op"]}


def assert_fp_count_not_image_count(row, stats) -> None:
    """Refuse a report row whose FP is not the recomputed number of false-positive detections."""
    for key in ("FP", "images", "images_with_FP"):
        if key not in row:
            raise EvalIntegrityError("a no-target report row must carry %s; got %r" % (key, sorted(row)))
    if int(row["FP"]) != int(stats["FP"]):
        raise EvalIntegrityError(
            "FP must be the number of false-positive DETECTIONS (%d), not the number of images (%d) or any "
            "other count: reported FP=%s" % (stats["FP"], stats["images"], row["FP"])
        )
    if int(row["images"]) != int(stats["images"]):
        raise EvalIntegrityError("reported images=%s, but the pool holds %s images"
                                 % (row["images"], stats["images"]))
    if int(row["images_with_FP"]) != int(stats["images_with_FP"]):
        raise EvalIntegrityError("reported images_with_FP=%s, but %s images carry a false positive"
                                 % (row["images_with_FP"], stats["images_with_FP"]))
    if int(row["images_with_FP"]) > int(row["images"]):
        raise EvalIntegrityError("images_with_FP=%s cannot exceed images=%s"
                                 % (row["images_with_FP"], row["images"]))


# =====================================================================================================
# 5. the frozen 496-image no-target guard
# =====================================================================================================
def no_target_pool_report(manifest) -> dict:
    """Count the frozen no-target pool from its manifest; the pool size is never rescaled."""
    path = Path(manifest)
    if not path.is_file():
        raise EvalBlocked("the frozen no-target pool manifest is missing: %s" % path)
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    if "image" not in frame.columns:
        raise EvalIntegrityError("the no-target manifest must list images in an 'image' column; %s has %s"
                                 % (path, sorted(frame.columns)))
    names = [str(value).strip() for value in frame["image"]]
    unique = sorted({name for name in names if name})
    return {"path": str(path), "rows": len(names), "images": len(names), "unique_images": len(unique),
            "duplicate_rows": len(names) - len(unique), "required": NO_TARGET_POOL_SIZE,
            "sha256": sha256_file(path)}


def assert_no_target_496(manifest) -> dict:
    """The frozen guard is exactly 496 unique images; any other pool invalidates the FP<=3 threshold."""
    report = no_target_pool_report(manifest)
    if report["images"] != NO_TARGET_POOL_SIZE or report["unique_images"] != NO_TARGET_POOL_SIZE:
        raise EvalIntegrityError(
            "the frozen no-target guard is exactly %d unique images; %s holds %d rows / %d unique images "
            "(%d duplicate rows). A different pool invalidates the FP<=%d threshold - it is never rescaled."
            % (NO_TARGET_POOL_SIZE, report["path"], report["images"], report["unique_images"],
               report["duplicate_rows"], NO_TARGET_FP_GATE)
        )
    report["gate"] = "PASS"
    return report


def no_target_fp_gate(fp) -> dict:
    """The frozen real no-target gate: FP<=3 over 496 images (historical V2 value 1)."""
    value = int(fp)
    if value < 0:
        raise EvalIntegrityError("a false-positive count cannot be negative: %r" % (fp,))
    return {"FP": value, "pool_images": NO_TARGET_POOL_SIZE, "historical_v2_FP": V2_HISTORICAL_NO_TARGET_FP,
            "gate": "PASS" if value <= NO_TARGET_FP_GATE else "FP_REGRESSION_OR_REAL_HARM",
            "rule": "FP <= %d over the %d frozen no-target images" % (NO_TARGET_FP_GATE, NO_TARGET_POOL_SIZE)}


def assert_no_target_fp_gate(fp) -> dict:
    report = no_target_fp_gate(fp)
    if report["gate"] != "PASS":
        raise EvalIntegrityError(
            "real no-target false positives %d exceed the frozen gate FP<=%d over %d images"
            % (report["FP"], NO_TARGET_FP_GATE, NO_TARGET_POOL_SIZE)
        )
    return report


# =====================================================================================================
# 6. canonical scoring helpers (the only matcher and AP code is the evaluator's own)
# =====================================================================================================
def score_predictions(samples, conf_op=CONF_OP, iou_thr=PRIMARY_IOU) -> dict:
    """Run the canonical evaluator's own matcher/AP code over samples; nothing is re-implemented here."""
    return ev.evaluate_samples(list(samples), conf_op=float(conf_op), center_thr=CENTER_DIST_PX,
                               iou_thr=float(iou_thr))


def score_by_stratum(samples, field, conf_op=CONF_OP, iou_thr=PRIMARY_IOU) -> dict:
    """Source-aware / location-aware TP/FP/FN: the canonical matcher applied per stratum, never a new one."""
    samples = list(samples)
    groups: dict = {}
    unknown = 0
    for sample in samples:
        value = sample.get(field)
        if value in (None, ""):
            unknown += 1
            key = "?"
        else:
            key = str(value).strip()
        groups.setdefault(key, []).append(sample)
    strata = {}
    for key in sorted(groups):
        overall = dict(score_predictions(groups[key], conf_op=conf_op, iou_thr=iou_thr)["overall"])
        overall["images"] = len(groups[key])
        strata[key] = overall
    return {"field": str(field), "images": len(samples), "unknown_count": unknown, "strata": strata}


# =====================================================================================================
# 7. stored V2 reference parity
# =====================================================================================================
def _normalize_reference(payload) -> dict:
    if payload is None:
        return {}
    if isinstance(payload, pd.DataFrame):
        return {str(row["set"]): dict(row) for row in payload.to_dict("records") if "set" in row}
    if not isinstance(payload, dict):
        raise EvaluatorParityFail("a stored reference must be a mapping of set -> metrics, got %s"
                                  % type(payload).__name__)
    out = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise EvaluatorParityFail("reference entry %r must be a mapping of metrics" % (key,))
        out[str(key)] = dict(value)
    return out


def verify_v2_reference_parity(reference, replay, tolerance=1e-9) -> dict:
    """Replay a stored V2 reference without retraining; any mismatch is EVALUATOR_PARITY_FAIL.

    A set that exists on one side only is a mismatch too: a silently missing set would otherwise look like a
    clean replay.
    """
    expected_all = _normalize_reference(reference)
    actual_all = _normalize_reference(replay)
    mismatches: list = []
    for set_key in sorted(expected_all):
        if set_key not in actual_all:
            mismatches.append("%s: the stored reference has no replay entry" % set_key)
            continue
        for field in PARITY_FIELDS:
            expected = expected_all[set_key].get(field)
            actual = actual_all[set_key].get(field)
            if expected is None and actual is None:
                continue
            if expected is None or actual is None:
                mismatches.append("%s.%s: reference=%r replay=%r" % (set_key, field, expected, actual))
                continue
            try:
                equal = abs(float(expected) - float(actual)) <= float(tolerance)
            except (TypeError, ValueError):
                equal = str(expected) == str(actual)
            if not equal:
                mismatches.append("%s.%s: reference=%r replay=%r" % (set_key, field, expected, actual))
    for set_key in sorted(actual_all):
        if set_key not in expected_all:
            mismatches.append("%s: the replay has a set the stored reference does not" % set_key)
    status = "PASS" if not mismatches else OUTCOME_EVALUATOR_PARITY_FAIL
    return {"status": status, "gate": status, "mismatches": mismatches, "sets": sorted(expected_all),
            "fields": list(PARITY_FIELDS), "tolerance": float(tolerance)}


def assert_v2_reference_parity(reference, replay, tolerance=1e-9) -> dict:
    report = verify_v2_reference_parity(reference, replay, tolerance=tolerance)
    if report["status"] != "PASS":
        raise EvaluatorParityFail(
            "the stored V2 reference does not replay identically with the canonical evaluator (%s): every "
            "new claim stops here (EVALUATOR_PARITY_FAIL)" % report["mismatches"][:5]
        )
    return report


# =====================================================================================================
# 8. read-only false-positive taxonomy
# =====================================================================================================
def _review_index(reviewed_hardness) -> dict:
    if not isinstance(reviewed_hardness, pd.DataFrame):
        raise EvalIntegrityError("reviewed_hardness must be a DataFrame, got %s"
                                 % type(reviewed_hardness).__name__)
    for column in ("image", "reviewed_category"):
        if column not in reviewed_hardness.columns:
            raise EvalIntegrityError("reviewed_hardness must carry 'image' and 'reviewed_category'; got %s"
                                     % sorted(reviewed_hardness.columns))
    index: dict = {}
    for record in reviewed_hardness.to_dict("records"):
        image = str(record.get("image", "")).strip()
        if not image:
            raise EvalIntegrityError("reviewed_hardness has a row without an image")
        category = str(record.get("reviewed_category", "")).strip()
        if category not in FP_REVIEWED_CATEGORIES:
            raise EvalIntegrityError(
                "reviewed_category %r is not one of the frozen hardness categories %s: an unreviewed or "
                "invented label is never silently defaulted" % (category, list(FP_REVIEWED_CATEGORIES))
            )
        index[image] = {"reviewed_category": category, "no_shuttle_reviewed": record.get("no_shuttle_reviewed")}
    return index


def classify_fp_errors(predictions, reviewed_hardness) -> pd.DataFrame:
    """Taxonomy of evaluation false positives. Read-only: the result is never a training or mining input.

    predictions        : one row per prediction with image, conf, x1, y1, x2, y2 and matched (bool).
    reviewed_hardness  : image, reviewed_category (frozen enum) and optional no_shuttle_reviewed.
    Returns one row per FALSE-POSITIVE DETECTION (not per image), tagged role EVAL_DIAGNOSTIC_ONLY.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise EvalIntegrityError("predictions must be a DataFrame, got %s" % type(predictions).__name__)
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    if missing:
        raise EvalIntegrityError("prediction frames must carry %s; missing %s"
                                 % (list(PREDICTION_COLUMNS), missing))
    review = _review_index(reviewed_hardness)
    has_gt_iou = "gt_iou_max" in predictions.columns

    rows: list = []
    for record in predictions.to_dict("records"):
        if _truthy(record.get("matched")):
            continue
        image = str(record["image"])
        entry = review.get(image)
        if entry is None or not _truthy(entry.get("no_shuttle_reviewed")):
            category = entry["reviewed_category"] if entry else None
            taxonomy = FP_TAXONOMY_UNREVIEWED
            reviewed = False
        else:
            category = entry["reviewed_category"]
            taxonomy = category
            reviewed = True

        fp_kind = None
        if has_gt_iou:
            raw = record.get("gt_iou_max")
            if raw not in (None, ""):
                value = float(raw)
                fp_kind = FP_KIND_LOCALIZATION if LOCALIZATION_IOU_FLOOR <= value < PRIMARY_IOU else FP_KIND_BACKGROUND

        rows.append({"image": image, "conf": float(record["conf"]), "x1": float(record["x1"]),
                     "y1": float(record["y1"]), "x2": float(record["x2"]), "y2": float(record["y2"]),
                     "reviewed_category": category, "taxonomy": taxonomy, "reviewed": reviewed,
                     "fp_kind": fp_kind, "taxonomy_role": FP_TAXONOMY_ROLE})

    frame = pd.DataFrame(rows, columns=list(FP_TAXONOMY_COLUMNS))
    frame.attrs["role"] = FP_TAXONOMY_ROLE
    frame.attrs["training_use"] = False
    frame.attrs["never_for_training"] = (
        "evaluation false-positive examples are diagnostics only (plan Task 9: 'taxonomy only, never a "
        "training-mining list from eval')"
    )
    frame.attrs["counts"] = {
        "FP": int(len(frame)),
        "images": int(predictions["image"].nunique()),
        "images_with_FP": int(frame["image"].nunique()) if len(frame) else 0,
        "by_taxonomy": frame["taxonomy"].value_counts().to_dict() if len(frame) else {},
        "by_fp_kind": frame["fp_kind"].value_counts().to_dict() if len(frame) else {},
    }
    return frame


def assert_fp_taxonomy_not_training_input(frame) -> None:
    """Refuse anything that would let an evaluation false positive re-enter training."""
    if not isinstance(frame, pd.DataFrame):
        raise EvalIntegrityError("expected the FP taxonomy DataFrame, got %s" % type(frame).__name__)
    attrs = dict(getattr(frame, "attrs", {}) or {})
    if attrs.get("role") != FP_TAXONOMY_ROLE or attrs.get("training_use") is not False:
        raise EvalIntegrityError(
            "this frame is not a marked evaluation-only FP taxonomy (role=%r, training_use=%r): evaluation "
            "examples are diagnostics only and must never be fed back into training"
            % (attrs.get("role"), attrs.get("training_use"))
        )
    wired = [column for column in TRAINING_POOL_COLUMNS if column in frame.columns]
    if wired:
        raise EvalIntegrityError(
            "the FP taxonomy carries training-pool columns %s: evaluation false positives can never be a "
            "training or candidate-mining input (plan Task 9)" % wired
        )


# =====================================================================================================
# 9. one canonical evaluation adapter
# =====================================================================================================
def canonical_eval_command(set_key, manifest, checkpoint, repo=REPO, out_dir=None) -> list:
    """The exact canonical-evaluator invocation; the evaluator itself is never modified or copied."""
    command = [sys.executable, str(Path(repo) / CANONICAL_EVALUATOR_PATH),
               "--repo", str(repo),
               "--split", "val",
               "--manifest", str(manifest),
               "--ckpt", "%s=%s" % (C.RUN_ID, checkpoint),
               "--imgsz", str(IMGSZ),
               "--conf", str(CONF_OP),
               "--ap-conf", str(CONF_FLOOR),
               "--iou", str(PRIMARY_IOU),
               "--center-dist", str(CENTER_DIST_PX)]
    if out_dir is not None:
        command += ["--out-dir", str(out_dir)]
    return command


def _run_command(command, repo) -> dict:
    import subprocess
    completed = subprocess.run([str(part) for part in command], cwd=str(repo), check=False,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise EvalBlocked("the canonical evaluator failed with exit code %d: %s"
                          % (completed.returncode, (completed.stderr or completed.stdout or "")[-800:]))
    return {"returncode": completed.returncode, "stdout_tail": (completed.stdout or "")[-2000:]}


def run_canonical_evaluations(checkpoint, frozen_manifests, repo=REPO, out_dir=None, spec_cfg=None,
                              runner=None) -> dict:
    """Delegate every evaluation to tools/eval_yolo26_v1.py and report the honest outcome.

    checkpoint       : frozen checkpoint path; nothing runs when it does not exist (never a placeholder).
    frozen_manifests : {set key: manifest path}; the key 'outdoor_real_ood' is the independent outdoor test.
    runner           : optional callable(command) -> payload, used to audit the plan without a GPU.

    Returns EVALUATED only when an independent real outdoor test set is declared, otherwise PILOT_ONLY.
    """
    repo = Path(repo)
    spec_cfg = load_spec_eval_cfg(repo=repo) if spec_cfg is None else spec_cfg
    assert_eval_contract(frozen_evaluator_cfg(), spec_cfg)

    if not frozen_manifests:
        raise EvalBlocked("no frozen evaluation manifest was supplied: nothing may be scored")
    resolved = Path(checkpoint) if checkpoint is not None else None
    if resolved is None or not resolved.is_file():
        raise EvalBlocked(
            "refusing to evaluate: no checkpoint at %s. The canonical evaluator is never run without a "
            "frozen checkpoint - report %s instead of inventing a metric." % (checkpoint, OUTCOME_NOT_EXECUTED)
        )

    sets: dict = {}
    outdoor = None
    for set_key in sorted(frozen_manifests):
        manifest = Path(frozen_manifests[set_key])
        if set_key == OUTDOOR_REAL_OOD_SET:
            outdoor = outdoor_manifest_report(manifest)
        elif not manifest.is_file():
            raise EvalBlocked("frozen evaluation manifest is missing: %s (%s)" % (manifest, set_key))
        command = canonical_eval_command(set_key, manifest, resolved, repo=repo, out_dir=out_dir)
        payload = runner(command) if runner is not None else _run_command(command, repo)
        sets[set_key] = {"manifest": str(manifest), "command": command, "payload": payload}

    outdoor_present = OUTDOOR_REAL_OOD_SET in frozen_manifests
    status = OUTCOME_EVALUATED if outdoor_present else OUTCOME_PILOT_ONLY
    blocker = None
    if not outdoor_present:
        blocker = ("no independent real outdoor test set was supplied, so outdoor generalization is not "
                   "claimed whatever the ETH-set scores say: the only honest outcome is %s" % OUTCOME_PILOT_ONLY)
    return {
        "task": "Task 9: Canonical evaluation and real-domain FP diagnostics",
        "canonical_evaluator": CANONICAL_EVALUATOR_PATH,
        "checkpoint": {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)},
        "eval_contract": frozen_evaluator_cfg(),
        "status": status,
        "outdoor_claim_allowed": bool(outdoor_present),
        "outdoor_blocker": blocker,
        "outdoor": outdoor,
        "sets": sets,
        "training_use": "FORBIDDEN",
        "notes": [
            "every metric is the canonical evaluator's own output; this adapter never re-implements GT "
            "decoding, the matcher or the AP curve",
            "independence of a declared outdoor test set must additionally be asserted with "
            "assert_outdoor_test_isolation before any claim",
            "evaluation false positives feed the read-only taxonomy only and are never a training input",
            "a stored V2 reference must be replayed with assert_v2_reference_parity before the new "
            "checkpoint is scored; a mismatch is %s" % OUTCOME_EVALUATOR_PARITY_FAIL,
        ],
    }


# =====================================================================================================
# 10. output writing
# =====================================================================================================
def write_eval_outputs(out_dir, eth_unseen_rows=(), no_target_rows=(), size_bucket_rows_=(),
                       taxonomy=None, outdoor_rows=(), integrity=None) -> dict:
    """Write the Task 9 evaluation outputs; the taxonomy keeps its eval-only marking."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict = {}

    def _write_csv(name, rows):
        path = out_dir / name
        frame = pd.DataFrame(_as_records(rows))
        frame.to_csv(path, index=False)
        written[name] = {"path": str(path), "rows": int(len(frame)), "sha256": sha256_file(path)}

    stratified = stratify_eth_unseen(eth_unseen_rows)
    _write_csv("eth_unseen_all.csv", stratified["all"])
    _write_csv("eth_unseen_real.csv", stratified[SOURCE_REAL])
    _write_csv("eth_unseen_synthetic.csv", stratified[SOURCE_SYNTHETIC])
    if size_bucket_rows_:
        rows = _as_records(size_bucket_rows_)
        assert_sources_not_merged(rows)
        _write_csv("size_buckets_by_source.csv", rows)
    _write_csv("real_no_target_496.csv", no_target_rows)
    _write_csv("outdoor_real_ood.csv", outdoor_rows)
    if taxonomy is not None:
        assert_fp_taxonomy_not_training_input(taxonomy)
        _write_csv("negative_error_taxonomy.csv", taxonomy.to_dict("records"))
    if integrity is not None:
        path = out_dir / "eval_integrity.json"
        path.write_text(json.dumps(integrity, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        written["eval_integrity.json"] = {"path": str(path), "rows": None, "sha256": sha256_file(path)}
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Canonical evaluation and real-domain FP diagnostics (Task 9)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", action="append", default=[],
                        help="NAME=PATH for every frozen evaluation manifest, repeatable "
                             "(the key 'outdoor_real_ood' is the independent outdoor test set)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--dry-run", action="store_true",
                        help="audit the frozen contract and the commands without running the evaluator")
    args = parser.parse_args(argv)

    manifests: dict = {}
    for item in args.manifest:
        if "=" not in item:
            parser.error("--manifest wants NAME=PATH, got %r" % item)
        name, path = item.split("=", 1)
        manifests[name] = Path(path)

    runner = (lambda command: {"command": command, "executed": False}) if args.dry_run else None
    result = run_canonical_evaluations(args.checkpoint, manifests, repo=args.repo, out_dir=args.out_dir,
                                       runner=runner)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval_integrity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "outdoor_claim_allowed": result["outdoor_claim_allowed"],
                      "sets": sorted(result["sets"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
