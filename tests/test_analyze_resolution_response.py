"""Tests for tools/analyze_resolution_response.py (zero-training resolution diagnostic)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import analyze_resolution_response as R  # noqa: E402

REP = REPO / "outputs" / "shuttle_capability" / "metrics" / "resolution_response_v1.json"
DEC = REPO / "outputs" / "shuttle_capability" / "metrics" / "resolution_response_v1_decision.json"


def _row(bucket, net, iou, matched=False, imgsz=1024):
    return {"imgsz": imgsz, "image": "/i.jpg", "gt_index": 0, "bucket": bucket,
            "equiv_size_640": net / 1.6, "net_px": net, "matched_op": matched,
            "best_iou_weak": iou, "best_conf_weak": 0.05, "detections_conf_ge_weak": 3}


def test_summarise_counts_the_weak_marks_per_bucket():
    rows = [_row("<4", 5.0, 0.0), _row("<4", 5.0, 0.2), _row("<4", 5.0, 0.6, matched=True)]
    s = R.summarise(rows, 1024)
    b = s["by_bucket"]["<4"]
    assert b["gt"] == 3 and b["tp"] == 1 and b["recall"] == pytest.approx(1 / 3)
    assert b["weak_ge_0.1"] == 2 and b["weak_ge_0.3"] == 1 and b["weak_ge_0.5"] == 1
    assert b["max_best_iou_weak"] == 0.6


def test_summarise_isolates_the_strict_sub_stride_subset():
    rows = [_row("<4", 5.0, 0.2), _row("4-6", 9.0, 0.4), _row("6-8", 11.0, 0.6, matched=True)]
    s = R.summarise(rows, 1024)
    sub = s["strict_sub_stride"]
    assert sub["gt"] == 1 and sub["tp"] == 0 and sub["weak_ge_0.1"] == 1
    assert sub["max_best_iou_weak"] == 0.2


def test_summarise_separates_resolutions():
    rows = [_row("<4", 5.0, 0.0, imgsz=1024), _row("<4", 5.0, 0.5, imgsz=1536)]
    a, b = R.summarise(rows, 1024), R.summarise(rows, 1536)
    assert a["by_bucket"]["<4"]["weak_ge_0.1"] == 0
    assert b["by_bucket"]["<4"]["weak_ge_0.1"] == 1


def _sum(imgsz, sub_weak, mid_weak, sub_tp=0):
    return {"imgsz": imgsz,
            "by_bucket": {"<4": {"gt": 24, "tp": sub_tp, "fn": 24 - sub_tp, "recall": 0.0,
                                 "weak_ge_0.1": sub_weak, "weak_ge_0.3": 0, "weak_ge_0.5": 0,
                                 "median_best_iou_weak": 0.0, "max_best_iou_weak": 0.0},
                          "4-6": {"gt": 37, "tp": 0, "fn": 37, "recall": 0.0, "weak_ge_0.1": 0,
                                  "weak_ge_0.3": 0, "weak_ge_0.5": 0, "median_best_iou_weak": 0.0,
                                  "max_best_iou_weak": 0.0},
                          "6-8": {"gt": 44, "tp": 4, "fn": 40, "recall": 0.09, "weak_ge_0.1": mid_weak,
                                  "weak_ge_0.3": 0, "weak_ge_0.5": 0, "median_best_iou_weak": 0.0,
                                  "max_best_iou_weak": 0.6}},
            "strict_sub_stride": {"gt": 43, "tp": sub_tp, "max_best_iou_weak": 0.0,
                                  "median_best_iou_weak": 0.0, "weak_ge_0.1": sub_weak, "weak_ge_0.3": 0}}


def test_branch_higher_resolution_when_the_sub_stride_buckets_respond():
    out = R.compute_branch([_sum(1024, 0, 9), _sum(1536, 4, 11)])
    assert out["branch"] == "HIGHER_RESOLUTION_TRAINING"
    assert out["sub_stride_weak_ge_0.1"] == {"1024": 0, "1536": 4}


def test_branch_p2_plus_weighting_when_only_the_mid_bucket_responds():
    out = R.compute_branch([_sum(1024, 0, 9), _sum(1536, 1, 13)])
    assert out["branch"] == "P2_PLUS_MID_BUCKET_WEIGHTING"


def test_branch_p2_when_nothing_responds():
    out = R.compute_branch([_sum(1024, 0, 9), _sum(1536, 0, 10)])
    assert out["branch"] == "P2_STRIDE_4"
    assert "sampling grid" in out["why"]


def test_branch_is_undetermined_without_the_1024_baseline():
    assert R.compute_branch([_sum(1536, 4, 11)])["branch"] == "UNDETERMINED"


def test_tiny_images_selects_only_val_unseen_images_with_a_tiny_gt(tmp_path):
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.new("RGB", (200, 200), (1, 2, 3)).save(img)
    lab = tmp_path / "a.txt"
    lab.write_text("0 0.1 0.1 0.01 0.01" + chr(10), encoding="utf-8")          # tiny
    big = tmp_path / "b.jpg"
    Image.new("RGB", (200, 200), (4, 5, 6)).save(big)
    lab2 = tmp_path / "b.txt"
    lab2.write_text("0 0.5 0.5 0.3 0.3" + chr(10), encoding="utf-8")           # not tiny
    man = tmp_path / "m.csv"
    man.write_text("image,label,split" + chr(10)
                   + "%s,%s,val" % (img, lab) + chr(10)
                   + "%s,%s,val" % (big, lab2) + chr(10), encoding="utf-8")
    inv = tmp_path / "l.csv"
    inv.write_text("set,image,leakage_class,location,GT" + chr(10)
                   + "val,%s,C_outside_eth_dataset,x,1" % img + chr(10)
                   + "val,%s,A_eth_trained_folder,y,1" % big + chr(10), encoding="utf-8")
    items = R.tiny_images(man, inv)
    assert len(items) == 1 and items[0]["image"] == str(img)
    assert len(items[0]["tiny"]) == 1


def test_real_diagnostic_is_consistent_when_present():
    if not REP.is_file() or not DEC.is_file():
        pytest.skip("resolution response diagnostic not built yet")
    rep = json.loads(REP.read_text(encoding="utf-8"))
    dec = json.loads(DEC.read_text(encoding="utf-8"))
    assert [s["imgsz"] for s in rep["by_resolution"]] == [1024, 1280, 1536]
    assert rep["by_resolution"][0]["strict_sub_stride"]["gt"] == 43
    assert dec["branch"] in ("HIGHER_RESOLUTION_TRAINING", "P2_PLUS_MID_BUCKET_WEIGHTING", "P2_STRIDE_4")
    lat = {int(k): v["latency"]["median_ms"] for k, v in rep["runs"].items()}
    assert lat[1024] < lat[1280] < lat[1536], lat
    rows = list(csv.DictReader((REPO / "outputs" / "shuttle_capability" / "metrics"
                                / "resolution_response_v1.csv").open(encoding="utf-8")))
    assert {int(r["imgsz"]) for r in rows} == {1024, 1280, 1536}
    assert len({(r["imgsz"], r["image"], r["gt_index"]) for r in rows}) == len(rows)
