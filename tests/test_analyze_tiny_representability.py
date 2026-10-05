"""Tests for tools/analyze_tiny_representability.py (read-only tiny-object diagnostic)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import analyze_tiny_representability as D  # noqa: E402

REP = REPO / "outputs" / "shuttle_capability" / "metrics" / "tiny_representability_v1.json"
REP_CSV = REPO / "outputs" / "shuttle_capability" / "metrics" / "tiny_representability_v1.csv"
V2_TABLE = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_vs_ours_eth_hardneg_v2.csv"


def test_net_factor_and_stride_constants():
    assert D.NET_FACTOR == pytest.approx(1024 / 640)
    assert D.P3_STRIDE == 8.0


def test_tiny_gt_table_separates_operating_match_from_weak_overlap(tmp_path):
    from PIL import Image
    img = tmp_path / "a.jpg"
    lab = tmp_path / "a.txt"
    Image.new("RGB", (200, 200), (5, 5, 5)).save(img)
    # two GT boxes: 20x20 px (equiv small) and 60x60 px
    # 0.01 of 200 px = 2 px -> equiv_size_640 = 6.4 (tiny); 0.3 of 200 px = 60 px -> equiv 192
    lab.write_text("0 0.1 0.1 0.01 0.01" + chr(10) + "0 0.7 0.7 0.3 0.3" + chr(10), encoding="utf-8")
    man = tmp_path / "m.csv"
    man.write_text("image,label,split" + chr(10) + "%s,%s,val" % (img, lab) + chr(10), encoding="utf-8")
    inv = tmp_path / "l.csv"
    inv.write_text("set,image,leakage_class,location,GT" + chr(10)
                   + "val,%s,C_outside_eth_dataset,x,2" % img + chr(10), encoding="utf-8")
    dump = {"dets": {str(img): [{"box": [19, 19, 21, 21], "conf": 0.05},      # weak hit on the small GT
                                {"box": [110, 110, 170, 170], "conf": 0.9}]}}  # strong hit on the big GT
    dp = tmp_path / "d.json"
    dp.write_text(json.dumps(dump), encoding="utf-8")
    rows = D.tiny_gt_table(dp, man, inv, classes=("C_outside_eth_dataset",), weak_conf=0.01, weak_iou=0.1)
    assert len(rows) == 2
    small = [r for r in rows if r["equiv_size_640"] < 8][0]
    big = [r for r in rows if r["equiv_size_640"] >= 8][0]
    assert small["matched_op"] is False and small["best_iou_weak"] > 0.9
    assert small["best_conf_weak"] == pytest.approx(0.05)
    assert big["matched_op"] is True and big["best_iou_weak"] > 0.9


def test_representability_flags_sub_stride_objects():
    rows = [{"bucket": "<4", "equiv_size_640": 3.0, "net_px_1024": 4.8, "matched_op": False},
            {"bucket": "6-8", "equiv_size_640": 7.0, "net_px_1024": 11.2, "matched_op": True},
            {"bucket": "<4", "equiv_size_640": 2.5, "net_px_1024": 3.5, "matched_op": False}]
    rep = D.representability_summary(rows)
    assert rep["overall"]["gt"] == 3 and rep["overall"]["tp"] == 1
    assert rep["overall"]["below_one_stride"] == 2   # 4.8 and 3.5 are below the 8 px stride, 11.2 is not
    assert rep["overall"]["below_half_stride"] == 1  # only 3.5 is below half a stride
    assert rep["by_bucket"]["<4"]["recall"] == 0.0
    assert rep["by_bucket"]["6-8"]["recall"] == 1.0
    assert rep["tiny_lt8"]["fn_below_one_stride"] == 2   # the two misses are sub-stride; the TP is not


def test_weak_evidence_share_and_medians():
    def row(matched, iou, conf):
        return {"bucket": "4-6", "equiv_size_640": 5.0, "net_px_1024": 8.0, "matched_op": matched,
                "best_iou_weak": iou, "best_conf_weak": conf}
    rows = [row(False, 0.5, 0.05), row(False, 0.05, 0.02), row(False, 0.2, 0.3), row(True, 0.8, 0.9)]
    out = D.weak_evidence_summary(rows, weak_iou=0.1)
    fn = out["false_negatives_tiny"]
    assert fn["n"] == 3 and fn["with_weak_detection"] == 2
    assert fn["share_with_weak_detection"] == pytest.approx(2 / 3)
    assert fn["median_best_conf_weak"] == pytest.approx(0.3) or fn["median_best_conf_weak"] == pytest.approx(0.05)
    assert out["true_positives_tiny"]["with_weak_detection"] == 1


def test_weak_evidence_is_none_safe_when_empty():
    out = D.weak_evidence_summary([], weak_iou=0.1)
    assert out["false_negatives_tiny"] == {"n": 0, "with_weak_detection": 0,
                                           "share_with_weak_detection": None, "median_best_conf_weak": None,
                                           "median_best_iou_weak": None}


def test_real_diagnostic_is_consistent_with_the_canonical_table():
    if not REP.is_file() or not V2_TABLE.is_file():
        pytest.skip("tiny representability diagnostic not built yet")
    rep = json.loads(REP.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(V2_TABLE.open(encoding="utf-8")))
    v2 = [r for r in rows if r["model"] == "eth_real_hardneg_v2_best" and r["set"] == "val|eth_unseen"][0]
    tiny = rep["representability"]["tiny_lt8"]
    assert tiny["gt"] == int(v2["GT_<8"]) == 105
    assert tiny["tp"] == int(v2["TP_<8"]) == 4
    assert tiny["fn"] == 101 == int(v2["GT_<8"]) - int(v2["TP_<8"])
    lt4 = rep["representability"]["by_bucket"]["<4"]
    mid = rep["representability"]["by_bucket"]["4-6"]
    six = rep["representability"]["by_bucket"]["6-8"]
    assert lt4["recall"] == 0.0 and lt4["below_one_stride"] == lt4["gt"] == 24
    assert mid["recall"] == 0.0 and mid["below_one_stride"] == 19 and mid["gt"] == 37
    assert six["below_one_stride"] == 0 and six["tp"] == 4 and six["gt"] == 44
    weak = rep["weak_evidence"]["false_negatives_tiny"]
    assert weak["n"] == 101 and weak["with_weak_detection"] == 5
    assert weak["share_with_weak_detection"] == pytest.approx(5 / 101)
    assert "2.61" in rep["headline"] or "present" in rep["headline"]


def test_real_csv_has_one_row_per_gt_with_network_sizes():
    if not REP_CSV.is_file():
        pytest.skip("tiny representability diagnostic not built yet")
    rows = list(csv.DictReader(REP_CSV.open(encoding="utf-8")))
    assert len(rows) == 495
    tiny = [r for r in rows if float(r["equiv_size_640"]) < 8.0]
    assert len(tiny) == 105
    for r in tiny:
        assert float(r["net_px_1024"]) == pytest.approx(float(r["equiv_size_640"]) * 1.6, abs=1e-6)
        assert float(r["stride_cells"]) == pytest.approx(float(r["net_px_1024"]) / 8.0, abs=1e-6)
