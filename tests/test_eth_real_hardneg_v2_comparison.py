"""Tests for tools/compare_eth_hardneg_v2.py (Task 5 decision rule and comparison table)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import compare_eth_hardneg_v2 as C  # noqa: E402

DECISION = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_real_hardneg_v2_decision.json"
VS = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_real_hardneg_v2_vs_v1.csv"


def test_success_requires_20pct_fp_reduction_and_recall_drop_under_002():
    v1 = {"fp_per_image": 0.10, "recall": 0.50}
    v2 = {"fp_per_image": 0.079, "recall": 0.49}
    out = C.compute_v2_decision(v1, v2)
    assert out["decision"] == "USEFUL"
    assert out["fp_reduction_relative"] == pytest.approx(0.21, abs=1e-9)
    assert out["delta_recall"] == pytest.approx(-0.01, abs=1e-9)


def test_large_recall_drop_is_harmful_even_if_fp_improves():
    v1 = {"fp_per_image": 0.10, "recall": 0.50}
    v2 = {"fp_per_image": 0.05, "recall": 0.44}
    assert C.compute_v2_decision(v1, v2)["decision"] == "HARMFUL_TRADEOFF"


def test_less_than_10pct_fp_change_is_low_value_when_accuracy_stable():
    v1 = {"fp_per_image": 0.10, "recall": 0.50, "map5095": 0.40}
    v2 = {"fp_per_image": 0.095, "recall": 0.50, "map5095": 0.40}
    assert C.compute_v2_decision(v1, v2)["decision"] == "LOW_VALUE"


def test_recall_drop_between_002_and_005_is_inconclusive():
    v1 = {"fp_per_image": 0.10, "recall": 0.50}
    v2 = {"fp_per_image": 0.06, "recall": 0.47}
    out = C.compute_v2_decision(v1, v2)
    assert out["decision"] == "INCONCLUSIVE"
    assert out["fp_reduction_relative"] == pytest.approx(0.40)


def test_fp_increase_is_never_useful():
    v1 = {"fp_per_image": 0.012, "recall": 0.44}
    v2 = {"fp_per_image": 0.020, "recall": 0.44}
    assert C.compute_v2_decision(v1, v2)["decision"] == "INCONCLUSIVE"


def test_target_boundary_is_inclusive_at_twenty_percent():
    v1 = {"fp_per_image": 0.0125, "recall": 0.44}
    v2 = {"fp_per_image": 0.0100, "recall": 0.44}
    assert C.compute_v2_decision(v1, v2)["decision"] == "USEFUL"


def test_decision_records_the_frozen_thresholds():
    out = C.compute_v2_decision({"fp_per_image": 0.1, "recall": 0.5}, {"fp_per_image": 0.1, "recall": 0.5})
    t = out["thresholds"]
    assert t["fp_reduction_target"] == 0.20 and t["recall_tolerance"] == -0.02
    assert t["recall_harm"] == -0.05 and t["low_value_band"] == 0.10


def _write_csv(path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def test_pooled_fp_uses_the_canonical_summary(tmp_path):
    cols = ["checkpoint", "set", "threshold", "images", "total_FP", "FP_per_image", "images_with_FP",
            "image_FP_rate", "max_confidence_FP"]
    rows = [{"checkpoint": "v1", "set": "real_images/backgrounds", "threshold": "0.25", "images": "30",
             "total_FP": "2", "FP_per_image": "0.0667", "images_with_FP": "2", "image_FP_rate": "0.0667",
             "max_confidence_FP": "0.31"},
            {"checkpoint": "v1", "set": "real_video/frames", "threshold": "0.25", "images": "150",
             "total_FP": "0", "FP_per_image": "0.0", "images_with_FP": "0", "image_FP_rate": "0.0",
             "max_confidence_FP": ""}]
    path = _write_csv(tmp_path / "neg.csv", rows, cols)
    out = C.pooled_fp(path)
    assert out["images"] == 180 and out["fp_total"] == 2
    assert out["fp_per_image"] == pytest.approx(2 / 180)
    assert len(out["fp_per_image_ci95"]) == 2 and out["fp_per_image_ci95"][1] > out["fp_per_image"]


def test_build_comparison_adds_deltas_and_the_pooled_row(tmp_path):
    cols = ["model", "set", "Precision", "Recall", "F1", "AP50", "mAP50-95", "Recall_<8", "FP_per_image"]
    v1 = [{"model": "eth_only_v1_best", "set": "val|eth_unseen", "Precision": "0.69", "Recall": "0.1636",
           "F1": "0.26", "AP50": "0.35", "mAP50-95": "0.1909", "Recall_<8": "0.0762", "FP_per_image": "0.0225"}]
    v2 = [{"model": "eth_real_hardneg_v2_best", "set": "val|eth_unseen", "Precision": "0.72", "Recall": "0.1800",
           "F1": "0.29", "AP50": "0.37", "mAP50-95": "0.2000", "Recall_<8": "0.0900", "FP_per_image": "0.0190"}]
    pool1 = {"images": 496, "fp_total": 6, "fp_per_image": 6 / 496}
    pool2 = {"images": 496, "fp_total": 4, "fp_per_image": 4 / 496}
    rows = C.build_comparison(v1, v2, pool1, pool2)
    assert [r["model"] for r in rows] == ["eth_only_v1_best", "eth_real_hardneg_v2_best",
                                          "eth_only_v1_best", "eth_real_hardneg_v2_best"]
    v2row = [r for r in rows if r["model"] == "eth_real_hardneg_v2_best" and r["set"] == "val|eth_unseen"][0]
    assert "delta_recall +0.01640" in v2row["delta_vs_v1"]
    assert "delta_map5095 +0.00910" in v2row["delta_vs_v1"]
    pooled = [r for r in rows if r["set"].startswith("real_no_target_pooled")]
    assert len(pooled) == 2
    # sign convention: every delta_vs_v1 entry is (v2 - v1), so an FP decrease is negative
    assert pooled[1]["delta_vs_v1"] == "delta_fp_per_image_relative -0.33333"


def test_write_comparison_emits_the_frozen_columns(tmp_path):
    rows = C.build_comparison([], [], {"images": 10, "fp_total": 1, "fp_per_image": 0.1},
                              {"images": 10, "fp_total": 0, "fp_per_image": 0.0})
    path = C.write_comparison(rows, tmp_path / "vs.csv")
    got = list(csv.DictReader(open(path, encoding="utf-8")))
    assert list(got[0].keys()) == list(C.COMPARISON_COLUMNS)
    assert len(got) == 2


def _neg_row(set_name, thr, images, fp, iwf):
    return {"checkpoint": "x", "set": set_name, "threshold": thr, "images": images, "total_FP": fp,
            "FP_per_image": fp / images, "images_with_FP": iwf, "image_FP_rate": iwf / images,
            "max_confidence_FP": "0.5"}


def _table_row(model, set_name, recall, map5095, fp_img):
    return {"model": model, "set": set_name, "Precision": "0.7", "Recall": recall, "F1": "0.3",
            "AP50": "0.35", "mAP50-95": map5095, "Recall_<8": "0.08", "FP_per_image": fp_img}


def test_task5_dry_run_degrades_cleanly_when_the_v2_run_is_missing(tmp_path, capsys):
    """Before the V2 checkpoint exists the pipeline must still run and say so, never crash."""
    neg = _write_csv(tmp_path / "v1neg.csv",
                     [_neg_row("real_images/backgrounds", "0.25", 30, 2, 2),
                      _neg_row("real_images/raw", "0.25", 59, 3, 2),
                      _neg_row("real_video/frames", "0.25", 150, 0, 0),
                      _neg_row("real_match_frames/images", "0.25", 235, 0, 0),
                      _neg_row("real_train/raw", "0.25", 22, 1, 1)],
                     ["checkpoint", "set", "threshold", "images", "total_FP", "FP_per_image",
                      "images_with_FP", "image_FP_rate", "max_confidence_FP"])
    tbl = _write_csv(tmp_path / "v1tbl.csv", [_table_row("eth_only_v1_best", "val|eth_unseen", "0.1636",
                                                         "0.1909", "0.0225")],
                     ["model", "set", "Precision", "Recall", "F1", "AP50", "mAP50-95", "Recall_<8",
                      "FP_per_image"])
    out_csv, out_json = tmp_path / "vs.csv", tmp_path / "dec.json"
    rc = C.main(["--v1-table", str(tbl), "--v2-table", str(tmp_path / "missing.csv"),
                 "--v1-negatives", str(neg), "--v2-negatives", str(tmp_path / "missing_neg.csv"),
                 "--out-csv", str(out_csv), "--out-json", str(out_json)])
    assert rc == 0
    dec = json.loads(out_json.read_text(encoding="utf-8"))
    assert dec["decision"] == "INCONCLUSIVE"
    assert dec["fp_per_image_v1"] == pytest.approx(6 / 496) and dec["fp_per_image_v2"] is None
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert {r["model"] for r in rows} == {"eth_only_v1_best"}


def test_task5_dry_run_end_to_end_with_a_stub_v2(tmp_path):
    """Stub V2 that removes 2 of the 6 FPs with recall -0.01 must be judged USEFUL by the frozen rule."""
    neg = _write_csv(tmp_path / "v1neg.csv",
                     [_neg_row("real_images/backgrounds", "0.25", 30, 2, 2),
                      _neg_row("real_images/raw", "0.25", 59, 3, 2),
                      _neg_row("real_video/frames", "0.25", 150, 0, 0),
                      _neg_row("real_match_frames/images", "0.25", 235, 0, 0),
                      _neg_row("real_train/raw", "0.25", 22, 1, 1)],
                     ["checkpoint", "set", "threshold", "images", "total_FP", "FP_per_image",
                      "images_with_FP", "image_FP_rate", "max_confidence_FP"])
    stub_neg = _write_csv(tmp_path / "v2neg.csv",
                          [_neg_row("real_images/backgrounds", "0.25", 30, 1, 1),
                           _neg_row("real_images/raw", "0.25", 59, 2, 1),
                           _neg_row("real_video/frames", "0.25", 150, 0, 0),
                           _neg_row("real_match_frames/images", "0.25", 235, 0, 0),
                           _neg_row("real_train/raw", "0.25", 22, 1, 1)],
                          ["checkpoint", "set", "threshold", "images", "total_FP", "FP_per_image",
                           "images_with_FP", "image_FP_rate", "max_confidence_FP"])
    cols = ["model", "set", "Precision", "Recall", "F1", "AP50", "mAP50-95", "Recall_<8", "FP_per_image"]
    v1t = _write_csv(tmp_path / "v1tbl.csv",
                     [_table_row("eth_only_v1_best", "val|eth_unseen", "0.1636", "0.1909", "0.0225")], cols)
    v2t = _write_csv(tmp_path / "v2tbl.csv",
                     [_table_row("eth_real_hardneg_v2_best", "val|eth_unseen", "0.1536", "0.1850", "0.0190")],
                     cols)
    out_csv, out_json = tmp_path / "vs.csv", tmp_path / "dec.json"
    assert C.main(["--v1-table", str(v1t), "--v2-table", str(v2t), "--v1-negatives", str(neg),
                   "--v2-negatives", str(stub_neg), "--out-csv", str(out_csv), "--out-json", str(out_json)]) == 0
    dec = json.loads(out_json.read_text(encoding="utf-8"))
    assert dec["decision"] == "USEFUL"
    assert dec["v1_pooled"]["fp_total"] == 6 and dec["v2_pooled"]["fp_total"] == 4
    assert dec["fp_reduction_relative"] == pytest.approx(1 / 3)
    assert dec["delta_recall"] == pytest.approx(-0.01)
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    pooled = [r for r in rows if r["set"].startswith("real_no_target_pooled")]
    assert len(pooled) == 2 and pooled[1]["delta_vs_v1"].endswith("-0.33333")


def test_real_decision_artifact_is_consistent():
    if not DECISION.is_file() or not VS.is_file():
        pytest.skip("V2 comparison artifacts not built yet")
    d = json.loads(DECISION.read_text(encoding="utf-8"))
    assert d["decision"] in ("USEFUL", "HARMFUL_TRADEOFF", "LOW_VALUE", "INCONCLUSIVE")
    assert d["primary_endpoint"] == "pooled_real_no_target_fp_per_image"
    assert d["secondary_endpoint"] == "val|eth_unseen"
    assert d["v1_pooled"]["images"] == 496 and d["v1_pooled"]["fp_total"] == 6
    rows = list(csv.DictReader(VS.open(encoding="utf-8")))
    sets = set(r["set"] for r in rows)
    assert any(s.startswith("real_no_target_pooled") for s in sets)
    assert "val|eth_unseen" in sets
    assert {r["model"] for r in rows} >= {"eth_only_v1_best"}
