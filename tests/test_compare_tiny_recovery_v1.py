"""Tests for tools/compare_tiny_recovery_v1.py (the six-label decision matrix, design section 19)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import compare_tiny_recovery_v1 as C  # noqa: E402

V2 = {"recall": 0.2040, "map5095": 0.189727, "tp_lt8": 4, "recall_lt8": 0.0381, "recall_8_16": 0.45,
      "fp_no_target": 1, "precision": 0.8783, "fp_per_image": 0.002016}


def _tiny(**kw):
    base = {"recall": 0.2040, "map5095": 0.189727, "tp_lt8": 4, "recall_lt8": 0.0381, "recall_8_16": 0.45,
            "fp_no_target": 1, "precision": 0.8783, "fp_per_image": 0.002016}
    base.update(kw)
    return base


def test_strong_success_when_tp8_reaches_twelve_with_all_guards():
    out = C.compute_tiny_decision(V2, _tiny(tp_lt8=12, recall=0.20, map5095=0.189, fp_no_target=2))
    assert out["decision"] == "STRONG_SUCCESS"
    assert out["delta_tp_lt8"] == 8 and out["delta_recall"] == pytest.approx(-0.004)


def test_useful_when_tp8_recovers_to_the_v1_level():
    out = C.compute_tiny_decision(V2, _tiny(tp_lt8=8, recall=0.19, map5095=0.185, fp_no_target=2))
    assert out["decision"] == "USEFUL"


def test_no_tiny_gain_when_tp8_stays_below_the_target():
    out = C.compute_tiny_decision(V2, _tiny(tp_lt8=6, recall=0.205, map5095=0.190, fp_no_target=1))
    assert out["decision"] == "NO_TINY_GAIN"
    assert "did not buy tiny recall" in out["why"]


def test_fp_regression_takes_precedence_over_a_tiny_gain():
    out = C.compute_tiny_decision(V2, _tiny(tp_lt8=13, recall=0.205, fp_no_target=4))
    assert out["decision"] == "FP_REGRESSION"


def test_size_tradeoff_is_flagged_even_when_tiny_improves():
    out = C.compute_tiny_decision(V2, _tiny(tp_lt8=12, recall=0.205, fp_no_target=2, recall_8_16=0.41))
    assert out["decision"] == "SIZE_TRADEOFF"
    assert out["delta_recall_8_16"] == pytest.approx(-0.04)


def test_harmful_when_recall_or_map_breach_the_harm_guard():
    assert C.compute_tiny_decision(V2, _tiny(recall=0.17))["decision"] == "HARMFUL"
    assert C.compute_tiny_decision(V2, _tiny(map5095=0.175))["decision"] == "HARMFUL"


def test_the_precedence_order_is_recorded_and_frozen():
    out = C.compute_tiny_decision(V2, _tiny())
    assert out["precedence"] == ["HARMFUL", "FP_REGRESSION", "SIZE_TRADEOFF", "STRONG_SUCCESS", "USEFUL",
                                 "NO_TINY_GAIN"]
    assert out["thresholds"] == {"tp_lt8_min": 8, "tp_lt8_strong": 12, "recall_min": 0.1840,
                                 "fp_no_target_max": 3, "map5095_delta_min": -0.01,
                                 "recall_8_16_delta_min": -0.02}


def test_main_table_has_the_design_metrics_and_signed_deltas():
    rows = C.build_main_table({**V2, "tp_lt8": 8, "recall_lt8": 0.0762, "fp_no_target": 6},
                              V2, _tiny(tp_lt8=12, recall=0.21, fp_no_target=2))
    by = {r["metric"]: r for r in rows}
    assert by["TP<8"]["tiny_recovery"] == 12 and by["TP<8"]["delta_vs_v2"] == 8.0
    assert by["no-target FP"]["v1"] == 6 and by["no-target FP"]["tiny_recovery"] == 2
    assert by["Recall"]["delta_vs_v2"] == pytest.approx(0.006)
    assert [r["metric"] for r in rows][:4] == ["Precision", "Recall", "mAP50-95", "TP<8"]


def test_metrics_from_row_reads_the_canonical_columns():
    got = C.metrics_from_row({"Precision": "0.88", "Recall": "0.204", "mAP50-95": "0.1897", "TP_<8": "12",
                              "Recall_<8": "0.1143", "Recall_8_16": "0.5"})
    assert got == {"precision": 0.88, "recall": 0.204, "map5095": 0.1897, "tp_lt8": 12.0,
                   "recall_lt8": 0.1143, "recall_8_16": 0.5}


def test_metrics_from_row_is_none_safe():
    assert C.metrics_from_row({}) == {"precision": None, "recall": None, "map5095": None, "tp_lt8": None,
                                      "recall_lt8": None, "recall_8_16": None}


def test_no_target_rows_carry_raw_counts_and_intervals(tmp_path):
    cols = ["checkpoint", "set", "threshold", "images", "total_FP", "FP_per_image", "images_with_FP",
            "image_FP_rate", "max_confidence_FP"]
    neg = tmp_path / "neg.csv"
    with neg.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerow({"checkpoint": "tiny", "set": "real_images/backgrounds", "threshold": "0.25", "images": "30",
                    "total_FP": "1", "FP_per_image": "0.0333", "images_with_FP": "1", "image_FP_rate": "0.0333",
                    "max_confidence_FP": "0.4"})
        w.writerow({"checkpoint": "tiny", "set": "real_video/frames", "threshold": "0.25", "images": "150",
                    "total_FP": "0", "FP_per_image": "0.0", "images_with_FP": "0", "image_FP_rate": "0.0",
                    "max_confidence_FP": ""})
    pool = C.A.summarize_false_positives(list(csv.DictReader(neg.open(encoding="utf-8"))), 0.25)
    assert pool["images"] == 180 and pool["fp_total"] == 1
    assert pool["fp_total_ci95"][0] < 1 < pool["fp_total_ci95"][1]
    assert len(pool["fp_per_image_ci95"]) == 2


def test_real_artifacts_are_consistent_when_present():
    decision = REPO / "outputs" / "shuttle_capability" / "metrics" / "tiny_recovery_v1_decision.json"
    if not decision.is_file():
        pytest.skip("tiny recovery evaluation not run yet")
    d = json.loads(decision.read_text(encoding="utf-8"))
    assert d["decision"] in ("USEFUL", "STRONG_SUCCESS", "NO_TINY_GAIN", "FP_REGRESSION", "SIZE_TRADEOFF",
                             "HARMFUL")
    assert d["primary_endpoint"] == "val|eth_unseen TP_<8"
    assert d["cohorts"]["v2"]["tp_lt8"] == 4 and d["cohorts"]["v2"]["fp_no_target"] == 1
    main = REPO / "outputs" / "shuttle_capability" / "metrics" / "tiny_recovery_v1_vs_v2.csv"
    rows = list(csv.DictReader(main.open(encoding="utf-8")))
    assert {r["metric"] for r in rows} >= {"TP<8", "Recall<8", "no-target FP", "Recall 8-16"}
