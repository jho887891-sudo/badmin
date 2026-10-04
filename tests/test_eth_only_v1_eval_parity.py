"""Task 5 evaluator-parity and experiment-consistency tests for the ETH-only YOLO26s 1024 V1 baseline.

The plan requires that the new model is measured with exactly the evaluator conventions already in use,
and that the frozen selection rules are honoured. These tests pin both without needing a GPU:
  * the third-party audit tool must keep importing our evaluator (one shared implementation),
  * the shared constants must equal the frozen contract in configs/eth_only_yolo26s_1024_v1.yaml,
  * the size-bucket / equiv_size_640 definitions must stay the frozen half-open intervals,
  * the registered checkpoint must match the training manifest (epoch, sha256, byte size),
  * the selected epoch must be the internal-validation mAP50-95 argmax (never an evaluation set),
  * the frozen ETH-only split must stay leak-free and location-disjoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import eval_eth_official_baseline as eb  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402
import train_eth_only_v1 as T  # noqa: E402

CONTRACT = REPO / "configs" / "eth_only_yolo26s_1024_v1.yaml"
REGISTRY = REPO / "configs" / "shuttle_detection" / "checkpoints_v1.yaml"
SCRATCH_CKPT = REPO / "_scratch_eth_only_v1" / "best.pt"
MANIFEST = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_full_e50_manifest.json"
RESULTS = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_full_e50_results.csv"


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_entry(name):
    return (yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {})["checkpoints"][name]


# --- one shared evaluator -------------------------------------------------------------

def test_eth_audit_tool_imports_our_evaluator():
    """If this breaks, the two tools could silently use different GT/matching/AP code."""
    assert eb.ev is ev
    assert eb.ev.BUCKET_LABELS == ev.BUCKET_LABELS
    assert eb.ev.equiv_size_640 is ev.equiv_size_640


def test_eth_audit_constants_equal_the_frozen_contract():
    c = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert eb.IMGSZ == c["imgsz"] == 1024
    assert eb.NMS_IOU == float(c["nms_policy"]["iou"]) == 0.7
    assert eb.MAX_DET == int(c["nms_policy"]["max_det"]) == 300
    assert c["nms_policy"]["rect"] is False
    assert eb.CONF_OP == float(c["confidence_policy"]["operating"]) == 0.25
    assert eb.CONF_FLOOR == float(c["confidence_policy"]["ap_floor"]) == 0.001


def test_our_evaluator_prediction_settings_are_locked_in_source():
    """Convention-drift alarm: the SSOT evaluator must keep predicting with these settings."""
    src = (REPO / "tools" / "eval_yolo26_v1.py").read_text(encoding="utf-8")
    assert "iou=0.7" in src and "max_det=300" in src
    assert "conf=conf_floor" in src, "AP curves must start at the AP floor (0.001), not at the operating conf"


# --- frozen metric definitions --------------------------------------------------------

def test_bucket_edges_are_the_frozen_half_open_intervals():
    assert ev.BUCKET_LABELS == ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]
    assert ev.bucket_of(3.999) == "<4" and ev.bucket_of(4.0) == "4-6"
    assert ev.bucket_of(23.999) == "16-24" and ev.bucket_of(24.0) == "24-32"
    assert ev.bucket_of(64.0) == ">64" and ev.bucket_of(1e6) == ">64"


def test_equiv_size_640_is_the_frozen_formula():
    assert ev.equiv_size_640(10.0, 10.0, 1920, 1200) == pytest.approx(10.0 * 640 / 1920)
    assert ev.equiv_size_640(6.0, 24.0, 1000, 1000) == pytest.approx((6.0 * 24.0) ** 0.5 * 0.64)
    assert ev.equiv_size_640(0.0, 5.0, 640, 480) == 0.0


# --- the trained checkpoint and its selection ------------------------------------------

def test_registry_entry_matches_the_trained_checkpoint():
    if not SCRATCH_CKPT.is_file():
        pytest.skip("local scratch copy of the checkpoint is not present (deleted after evaluation)")
    e = _registry_entry("eth_only_v1_best")
    assert e["role"] == "candidate" and int(e["epoch"]) == 21
    assert SCRATCH_CKPT.stat().st_size == 20344069
    assert _sha256(SCRATCH_CKPT) == "7a836a2621686affbdd3f1da7c3a0a57c4b86f14432835be2bc562c2899fc6f2"
    assert e["remote"].endswith("full_e50/weights/best.pt")


def test_selected_epoch_is_the_internal_validation_map50_95_argmax():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    col = man["selection"]["column"]
    assert col == "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[col]))
    assert int(float(best["epoch"])) == int(man["selection"]["epoch"]) == 21
    assert float(best[col]) == pytest.approx(man["selection"]["value"], abs=1e-9)
    assert len(rows) == 50 and man["selection"]["epochs_recorded"] == 50
    assert man["selection"]["selection_scope"] == "internal_validation_only"
    assert man["diagnostic_only"] is False and man["eligible_for_final_report"] is True
    assert man["batch"] == 8 and man["gpu_cap"]["cap_gib"] == 24.0 and man["gpu_cap"]["applied"] is True
    assert man["third_party_loggers"]["WANDB_MODE"] == "disabled"


def test_trained_checkpoint_hash_is_recorded_identically_in_both_artifacts():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    e = _registry_entry("eth_only_v1_best")
    sha = man["best_checkpoint"]["sha256"]
    assert sha == "7a836a2621686affbdd3f1da7c3a0a57c4b86f14432835be2bc562c2899fc6f2"
    note = e["note"].replace(",", "")   # the note formats the byte count as 20,344,069
    assert sha in note and str(man["best_checkpoint"]["bytes"]) in note
    assert man["save_dir"].endswith("runs_eth_only_v1/full_e50")


def test_trainer_refuses_to_use_evaluation_sets_for_selection():
    assert T.EVALUATION_ONLY_SPLITS == ("val|eth_unseen", "external_real_only",
                                        "controlled_capability", "challenge_test")
    m = T.build_manifest({"experiment": "x", "resolved_kwargs": {"epochs": 50}, "contract_epochs": 50})
    assert m["never_used_for_selection"] == list(T.EVALUATION_ONLY_SPLITS)


# --- the frozen split stays leak-free -------------------------------------------------

def test_frozen_eth_only_split_is_leak_free_and_location_disjoint():
    tr = list(csv.DictReader((REPO / "data" / "eth_only_v1_train_manifest.csv").open(encoding="utf-8")))
    va = list(csv.DictReader((REPO / "data" / "eth_only_v1_val_manifest.csv").open(encoding="utf-8")))
    assert len(tr) == 14543 and len(va) == 2920
    assert set(r["location"] for r in tr).isdisjoint(set(r["location"] for r in va))
    assert set(r["sha256"] for r in tr).isdisjoint(set(r["sha256"] for r in va))
    assert set(r["source"] for r in tr) <= {"eth_main", "eth_negative"}
    audit = json.loads((REPO / "outputs" / "shuttle_capability" / "metrics" /
                        "eth_only_v1_data_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS" and audit["problems"] == []
    assert audit["eval_overlap_images"] == 0 and audit["eval_hash_overlap_images"] == 0
    assert audit["val_locations"] == sorted(set(r["location"] for r in va))
    assert audit["negative_fraction"] == 0.1
    positives = sum(1 for r in tr if r["is_negative"] == "False")
    assert positives == 13992
    assert 0.15 <= float(audit["val_positive_fraction"]) <= 0.20
