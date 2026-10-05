"""Tests for tools/analyze_protocol_cost.py (supplementary read-only protocol-cost analysis)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import analyze_protocol_cost as P  # noqa: E402

COST_JSON = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_protocol_cost.json"
COST_CSV = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_protocol_cost.csv"
ETH_VS_V1 = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_vs_ours_eth_only_v1.csv"


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_readers_extract_the_frozen_values(tmp_path):
    recipe = _write(tmp_path / "r.json", {"counts": {"positives_by_location": {"a": 10, "b": 4}}})
    audit = _write(tmp_path / "a.json", {"excluded_eval_by_location": {"b": 4},
                                         "train_location_counts": {"a": 3}})
    assert P.location_inventory(recipe) == {"a": 10, "b": 4}
    assert P.evaluation_consumption(audit) == {"b": 4}
    assert P.v1_training_usage(audit) == {"a": 3}


def test_val_gt_by_location_only_counts_the_fair_subset(tmp_path):
    inv = tmp_path / "i.csv"
    inv.write_text("set,image,leakage_class,location,GT" + chr(10)
                   + "val,/a.jpg,A_eth_trained_folder,ml_3,50" + chr(10)
                   + "val,/b.jpg,B_same_location_not_trained,ml_3,4" + chr(10)
                   + "val,/c.jpg,C_outside_eth_dataset,synthetic,7" + chr(10), encoding="utf-8")
    got = P.val_gt_by_location(inv)
    assert got == {"ml_3": 4, "synthetic": 7}


def test_protocol_cost_table_marks_fully_consumed_locations(tmp_path):
    rows = P.protocol_cost_table({"a": 10, "b": 4}, {"b": 4}, {"a": 3}, {"a": 1, "b": 2},
                                 {"a": {"gt": 1, "fn": 1}, "b": {"gt": 2, "fn": 1}}, {"b": 1})
    by = {r["location"]: r for r in rows}
    assert by["b"]["remaining_trainable"] == 0 and by["b"]["fully_consumed"] is True
    assert by["a"]["remaining_trainable"] == 10 and by["a"]["fully_consumed"] is False
    assert by["b"]["recoverable_fn"] == 1 and by["a"]["v1_fn_rate"] == 1.0


def test_protocol_cost_table_never_returns_negative_remaining():
    rows = P.protocol_cost_table({"a": 5}, {"a": 9}, {}, {}, {}, {})
    assert rows[0]["remaining_trainable"] == 0 and rows[0]["fully_consumed"] is True


def test_summarise_reports_none_not_zero_for_absent_trained_locations():
    rows = P.protocol_cost_table({"a": 10, "b": 4}, {"b": 4}, {"a": 3}, {"b": 2},
                                 {"b": {"gt": 2, "fn": 1}}, {"b": 1})
    s = P.summarise(rows, recoverable_global=5)
    assert s["trained_locations_val_gt"] == 0
    assert s["trained_locations_fn_rate"] is None
    assert s["dropped_locations_fn_rate"] == pytest.approx(0.5)
    assert s["recoverable_share_of_global"] == pytest.approx(1 / 5)


def test_summarise_separates_eth_subset_share_from_the_global_share():
    rows = P.protocol_cost_table({"b": 4}, {"b": 4}, {}, {"b": 2}, {"b": {"gt": 2, "fn": 1}}, {"b": 4})
    s = P.summarise(rows, recoverable_global=10)
    assert s["recoverable_fn_total_in_eth_locations"] == 4
    assert s["recoverable_fn_total_global"] == 10
    assert s["recoverable_share_of_eth_locations"] == 1.0
    assert s["recoverable_share_of_global"] == pytest.approx(0.4)


def test_non_eth_domains_uses_the_fn_table_and_excludes_eth_locations():
    out = P.non_eth_domains({"ml_3": {"gt": 144, "fn": 115}, "synthetic": {"gt": 120, "fn": 117}},
                            {"ml_3": 1004})
    assert out == [{"domain": "synthetic", "val_unseen_gt": 120, "v1_fn": 117, "v1_fn_rate": 0.975}]


def test_real_protocol_cost_closes_the_val_unseen_gt_budget():
    if not COST_JSON.is_file():
        pytest.skip("protocol cost artifacts not built yet")
    d = json.loads(COST_JSON.read_text(encoding="utf-8"))
    s = d["summary"]
    assert s["locations_fully_consumed"] == ["ml_6", "ml_3", "uetlibergstrasse_1"]
    assert s["frames_consumed_total"] == 2765
    assert s["frames_remaining_trainable_total"] == 16913
    assert s["dropped_locations_val_gt"] == 263
    assert s["dropped_locations_fn_rate"] == pytest.approx(0.7642585551330798, abs=1e-9)
    assert s["trained_locations_val_gt"] == 0 and s["trained_locations_fn_rate"] is None
    assert s["recoverable_fn_total_in_eth_locations"] == 187
    assert s["recoverable_fn_total_global"] == 194
    assert s["recoverable_share_of_global"] == pytest.approx(187 / 194, abs=1e-9)
    foreign = sum(x["val_unseen_gt"] for x in s["non_eth_domains"])
    assert foreign == 232 and {x["domain"] for x in s["non_eth_domains"]} == {"synthetic", "iphone_20251015"}
    rows = list(csv.DictReader(ETH_VS_V1.open(encoding="utf-8")))
    gt_total = int([r for r in rows if r["model"] == "eth_only_v1_best" and r["set"] == "val|eth_unseen"][0]["GT"])
    assert s["dropped_locations_val_gt"] + foreign == gt_total == 495
    assert "no trained-location FN rate exists" in d["headline"]


def test_real_protocol_cost_csv_has_one_row_per_location():
    if not COST_CSV.is_file():
        pytest.skip("protocol cost artifacts not built yet")
    rows = list(csv.DictReader(COST_CSV.open(encoding="utf-8")))
    assert len(rows) == 11
    assert {r["location"] for r in rows if r["fully_consumed"] == "True"} == {"ml_3", "ml_6",
                                                                             "uetlibergstrasse_1"}
    assert {r["location"] for r in rows if r["v1_train_frames_used"] != "0"} == {
        "cab_1", "cab_2", "glc_1", "ml_4", "ticino_1", "ticino_2"}
