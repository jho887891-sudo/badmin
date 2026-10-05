"""Tests for tools/analyze_eth_only_v1_errors.py (Task 3, read-only diagnostic)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import analyze_eth_only_v1_errors as A  # noqa: E402

DIAG = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_error_diagnostic.json"
DIAG_CSV = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_error_diagnostic.csv"
ETH_VS_V1 = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_vs_ours_eth_only_v1.csv"


def test_poisson_ci_matches_the_garwood_values():
    assert A.poisson_ci95(0) == [0.0, 3.689]
    assert A.poisson_ci95(1) == pytest.approx([0.025318, 5.571643], abs=1e-5)
    assert A.poisson_ci95(6) == pytest.approx([2.201894, 13.059474], abs=1e-5)
    assert A.poisson_ci95(17) == pytest.approx([9.903126, 27.218647], abs=1e-5)


def test_poisson_ci_is_monotone_in_the_count():
    lows = [A.poisson_ci95(k)[0] for k in range(1, 8)]
    highs = [A.poisson_ci95(k)[1] for k in range(1, 8)]
    assert lows == sorted(lows) and highs == sorted(highs)


def test_wilson_ci_stays_inside_the_unit_interval():
    lo, hi = A.wilson_ci95(6, 496)
    assert 0.004 < lo < 0.007 and 0.023 < hi < 0.030
    assert A.wilson_ci95(0, 10)[0] == 0.0
    assert A.wilson_ci95(5, 0) == [None, None]


@pytest.fixture()
def fp_rows():
    def row(set_name, thr, images, fp, iwf, maxc=""):
        return {"set": set_name, "threshold": thr, "images": images, "total_FP": fp,
                "images_with_FP": iwf, "max_confidence_FP": maxc}
    return [row("real_images/backgrounds", 0.25, 30, 2, 2, "0.31"),
            row("real_images/backgrounds", 0.5, 30, 0, 0),
            row("real_images/raw", 0.25, 59, 3, 2, "0.57"),
            row("real_video/frames", 0.25, 150, 0, 0),
            row("real_match_frames/images", 0.25, 235, 0, 0),
            row("real_train/raw", 0.25, 22, 1, 1, "0.57"),
            row("synthetic_3d/images", 0.25, 84, 13, 12, "0.78"),
            row("synthetic_on_real_bg/images", 0.25, 160, 17, 15, "0.80")]


def test_summarize_false_positives_pools_the_raw_counts(fp_rows):
    out = A.summarize_false_positives(fp_rows, 0.25)
    assert out["images"] == 496 and out["fp_total"] == 6
    assert out["fp_per_image"] == pytest.approx(6 / 496)
    assert out["per_set"]["real_images/backgrounds"]["fp_total"] == 2
    assert "synthetic_3d/images" not in out["per_set"], "synthetic sets are not part of the real endpoint"
    assert out["fp_total_ci95"] == [2.201894, 13.059474]
    assert out["fp_per_image_ci95"][0] == pytest.approx(2.201894 / 496, abs=1e-6)
    assert out["image_fp_rate"] == pytest.approx(5 / 496)


def test_summarize_false_positives_respects_the_threshold(fp_rows):
    at_half = A.summarize_false_positives(fp_rows, 0.5)
    assert at_half["fp_total"] == 0 and at_half["images"] == 30


def test_summarize_false_positives_can_pool_the_synthetic_diagnostic(fp_rows):
    out = A.summarize_false_positives(fp_rows, 0.25, sets=("synthetic_3d/images", "synthetic_on_real_bg/images"))
    assert out["images"] == 244 and out["fp_total"] == 30


def test_summarize_false_negatives_groups_by_bucket():
    rows = [{"bucket": "<4", "matched": False, "best_iou": 0.1},
            {"bucket": "<4", "matched": False, "best_iou": 0.0},
            {"bucket": "8-12", "matched": True, "best_iou": 0.8},
            {"bucket": "8-12", "matched": False, "best_iou": 0.2},
            {"bucket": "16-24", "matched": True, "best_iou": 0.9}]
    out = A.summarize_false_negatives(rows)
    assert out["gt_total"] == 5 and out["fn_total"] == 3
    assert out["fn_by_bucket"] == {"<4": 2, "8-12": 1}
    assert out["small_lt8_gt"] == 2 and out["small_lt8_fn"] == 2
    assert out["fn_rate_by_bucket"]["8-12"] == pytest.approx(0.5)


def test_compare_eth_tp_v1_fn_finds_the_recoverable_gap():
    eth = [{"image": "/a.jpg", "gt_index": 0, "matched": True},
           {"image": "/a.jpg", "gt_index": 1, "matched": False},
           {"image": "/b.jpg", "gt_index": 0, "matched": True}]
    v1 = [{"image": "/a.jpg", "gt_index": 0, "matched": False},
          {"image": "/a.jpg", "gt_index": 1, "matched": False},
          {"image": "/b.jpg", "gt_index": 0, "matched": True}]
    out = A.compare_eth_tp_v1_fn(eth, v1)
    assert out["common_gt"] == 3 and out["eth_tp_v1_fn"] == 1 and out["v1_tp_eth_fn"] == 0
    assert out["eth_tp_v1_fn_examples"] == ["/a.jpg#0"]
    assert out["eth_tp_total"] == 2 and out["v1_tp_total"] == 1


def test_build_val_match_table_honours_the_operating_confidence(tmp_path):
    """Regression: matching at the dump's AP floor (0.001) would over-count hits vs the published Recall."""
    from PIL import Image
    img = tmp_path / "img.jpg"
    lab = tmp_path / "img.txt"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(img)
    lab.write_text("0 0.5 0.5 0.2 0.2" + chr(10), encoding="utf-8")     # GT box 40,40,60,60
    man = tmp_path / "man.csv"
    man.write_text("image,label,split" + chr(10) + "%s,%s,val" % (img, lab) + chr(10), encoding="utf-8")
    dump = {"imgsz": 1024, "conf": 0.001, "dets": {str(img): [
        {"box": [0, 0, 10, 10], "conf": 0.9},          # high confidence but no overlap
        {"box": [40, 40, 60, 60], "conf": 0.05}]}}     # exact hit, below the operating confidence
    dp = tmp_path / "dump.json"
    dp.write_text(json.dumps(dump), encoding="utf-8")
    at_op = A.build_val_match_table(dp, man, None, None, split="val", conf_thr=0.25)
    at_floor = A.build_val_match_table(dp, man, None, None, split="val", conf_thr=0.01)
    assert len(at_op) == 1 and at_op[0]["matched"] is False
    assert len(at_floor) == 1 and at_floor[0]["matched"] is True


def test_real_diagnostic_matches_the_canonical_evaluator():
    if not DIAG.is_file() or not ETH_VS_V1.is_file():
        pytest.skip("diagnostic or comparison artifacts not built yet")
    diag = json.loads(DIAG.read_text(encoding="utf-8"))
    fp = diag["false_positives_real_no_target"]
    assert fp["images"] == 496 and fp["fp_total"] == 6
    fn = diag["val_eth_unseen_v1"]
    assert fn["gt_total"] == 495 and fn["fn_total"] == 414
    import csv
    rows = list(csv.DictReader(ETH_VS_V1.open(encoding="utf-8")))
    mine = [r for r in rows if r["model"] == "eth_only_v1_best" and r["set"] == "val|eth_unseen"][0]
    assert int(mine["GT"]) == fn["gt_total"]
    assert int(mine["GT"]) - int(mine["TP"]) == fn["fn_total"], "FN must equal GT - TP of the canonical table"
    assert int(mine["TP"]) == 81
    flat = list(csv.DictReader(DIAG_CSV.open(encoding="utf-8")))
    pooled = [r for r in flat if r["scope"] == "real_no_target_pooled"]
    assert float([r for r in pooled if r["metric"] == "fp_total"][0]["value"]) == 6.0
    assert len([r for r in flat if r["scope"] == "real_no_target_per_set"]) == 5 * 4
