"""Contract + guard tests for YOLO26S_V2_FULL_ETH_CONTINUATION_V1 (pre-run gates, spec sections 3-5, 9-13, 19-22)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import train_eth_only_v1 as T  # noqa: E402

CONTRACT = REPO / "configs" / "yolo26s_v2_full_eth_cont_v1.yaml"
SPEC = REPO / "docs" / "superpowers" / "specs" / "2026-10-07-yolo26s-v2-full-eth-continuation-v1-spec.md"
MANIFEST = REPO / "data" / "yolo26s_v2_full_eth_cont_v1_train_manifest.csv"
AUDIT = REPO / "outputs" / "shuttle_capability" / "metrics" / "yolo26s_v2_full_eth_cont_v1_data_audit.json"
V2_SHA = "3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7"
V2_BYTES = 20344133


def _contract():
    return T.load_contract(CONTRACT)


def test_spec_is_the_attached_document_verbatim():
    assert SPEC.is_file()
    text = SPEC.read_text(encoding="utf-8")
    assert SPEC.stat().st_size == 20514
    assert "YOLO26S_V2_FULL_ETH_CONTINUATION_V1" in text
    assert "NEXT_ARCHITECTURE_EXPERIMENT = P2_STRIDE_4" in text


def test_contract_pins_the_continuation_from_the_frozen_v2_checkpoint():
    c = _contract()
    assert c["experiment"] == "yolo26s_v2_full_eth_cont_v1"
    assert c["model_sha256"] == V2_SHA and c["model_bytes"] == V2_BYTES
    assert c["continuation_from"] == "eth_real_hardneg_v2_best"
    assert c["model_variant"] == "yolo26s" and c["architecture_change"] == "none" and c["p2_forbidden"] is True
    assert list(c["never_touch"]) == list(T.FROZEN)
    assert c["never_touch_extra"] == ["epochs"]


def test_contract_pins_the_spec_schedule_and_batch_policy():
    c = _contract()
    assert c["imgsz"] == 1024 and c["epochs"] == 20 and c["freeze"] == 0
    assert c["optimizer"] == "AdamW" and float(c["lr0"]) == pytest.approx(3.0e-5)
    assert c["nbs"] == 64 and c["batch"] == 32 and c["batch_fallback"] == [24, 16]
    assert c["save_period"] == 5
    assert c["smoke_epochs"] == 2
    assert c["next_architecture_experiment"] == "P2_STRIDE_4"


def test_resolved_recipe_matches_the_contract_on_every_frozen_knob():
    c = _contract()
    kw = T.resolve_recipe(c, None, epochs=c["epochs"])
    assert kw["imgsz"] == 1024 and kw["epochs"] == 20 and kw["nbs"] == 64
    assert kw["optimizer"] == "AdamW" and float(kw["lr0"]) == pytest.approx(3.0e-5)
    assert kw["freeze"] == 0 and kw["seed"] == 42 and kw["save_period"] == 5
    with pytest.raises(T.RecipeViolationError):
        T.resolve_recipe(c, None, deviating={"lr0": 1.0e-4})
    with pytest.raises(T.RecipeViolationError):
        T.resolve_recipe(c, None, deviating={"nbs": 32})


def test_save_period_is_a_real_ultralytics_training_argument():
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics.cfg import DEFAULT_CFG_DICT
    assert "save_period" in DEFAULT_CFG_DICT
    assert DEFAULT_CFG_DICT["save_period"] == -1   # our contract sets 5, i.e. an explicit override


def test_endpoint_is_fixed_to_epoch_20_and_cannot_drift():
    c = _contract()
    assert T.resolve_endpoint(c, 20)["mode"] == "fixed_epoch"
    assert T.resolve_endpoint(c, 20)["epoch"] == 20
    assert T.resolve_endpoint(c, 20)["use_best_pt_for_primary_comparison"] is False
    with pytest.raises(T.RecipeViolationError):
        T.resolve_endpoint(c, 19)
    smoke = T.resolve_endpoint(c, 2, smoke=True)
    assert smoke["mode"] == "fixed_epoch" and smoke["diagnostic_only"] is True


def test_optimizer_steps_and_accumulation_are_recorded():
    s = T.optimizer_steps(25178, 32, 20, 64)
    assert s["gradient_accumulation"] == 2 and s["nominal_batch"] == 64
    assert s["steps_per_epoch"] == 25178 // 32
    assert s["optimizer_steps"] == (25178 // 32) * 20
    assert T.plan_batch_attempts(32, [24, 16]) == [32, 24, 16]


def test_manifest_marks_internal_val_selection_as_not_used_for_the_primary_comparison():
    c = _contract()
    run = {"experiment": c["experiment"], "smoke": False, "dry_run": False,
           "resolved_kwargs": {"epochs": 20}, "contract_epochs": 20,
           "selection_metric": c["selection_metric"], "endpoint": T.resolve_endpoint(c, 20)}
    m = T.build_manifest(run)
    assert m["eligible_for_final_report"] is True
    assert m["checkpoint_selection"]["used_for_primary_comparison"] is False
    assert m["checkpoint_selection"]["primary_endpoint"]["epoch"] == 20
    assert "legacy_eth_eval" not in " ".join(m["never_used_for_selection"])  # legacy names kept, new label used in reports


def test_decision_criteria_are_the_spec_numbers():
    crit = _contract()["decision_criteria"]
    labels = [x["label"] for x in crit["primary"]]
    assert labels == ["POSITIVE_EXPANSION_STRONG", "POSITIVE_EXPANSION_USEFUL", "POSITIVE_EXPANSION_LOW_EFFECT"]
    strong = crit["primary"][0]["all_of"]
    assert {"delta_recall_min": 0.10} in strong and {"delta_recall_lt8_min": 0.05} in strong
    useful = crit["primary"][1]["any_of"]
    assert {"delta_recall_min": 0.05} in useful and {"delta_recall_lt8_min": 0.03} in useful
    low = crit["primary"][2]["all_of"]
    assert {"delta_recall_max": 0.05} in low and {"delta_recall_lt8_max": 0.03} in low
    assert crit["flags"]["FP_STABLE"]["fp_max"] == 3
    assert crit["flags"]["FP_REGRESSION"]["fp_min"] == 4
    assert crit["flags"]["MAP_REGRESSION"]["delta_map5095_max"] == -0.02
    assert crit["precedence"] == labels and crit["emit_exactly_one_primary"] is True


def test_baseline_and_data_expectations_are_internally_consistent():
    c = _contract()
    b = c["baseline_v2"]
    assert b["recall"] == pytest.approx(0.2040) and b["recall_lt8"] == pytest.approx(0.0381)
    assert b["tp_lt8"] == 4 and b["gt_lt8"] == 105
    assert b["no_target_fp"] == 1 and b["no_target_images"] == 496
    d = c["data"]
    assert d["expected_positive_images"] + d["expected_negative_images"] == d["expected_rows"] == 25178
    assert d["hard_negative_repeat_this_experiment"] == 1


def test_real_manifest_and_audit_when_present():
    if not MANIFEST.is_file() or not AUDIT.is_file():
        pytest.skip("continuation pool not built yet")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    assert audit["GATE_all_pass"] is True and audit["GATE_manifest_complete"] is True
    assert audit["manifest_rows"] == len(rows) == 25178
    assert audit["positive_images"] == 19678 and audit["negative_images"] == 5500
    assert not audit["missing_images"] and not audit["unreadable_images"]
    assert not audit["non_empty_negative_labels"] and not audit["prohibited_source_hits"]
    assert audit["manifest_sha256"] == T.sha256_file(MANIFEST)
    assert {r["difficulty"] for r in rows if r["is_positive"] == "True"} == {"easy", "medium"}
    assert {r["source"] for r in rows} == {"eth_main", "eth_negative"}
    for r in rows[:50]:
        assert r["label_path"].startswith(r["set_dir"] + "/labels/train/")
        assert len(r["sha256"]) == 64
    assert audit["lists"]["internal_val_overlaps_train_pool"] is True
    assert audit["lists"]["internal_val_used_for_selection"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
