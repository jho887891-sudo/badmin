"""Contract tests for ETH_HARDNEG_TINY_RECOVERY_V1 (design 2026-10-05).

The experiment may change exactly one variable - the training exposure of existing real tiny positives - so these
tests pin everything else to the frozen V2 contract and check the pre-registered endpoints, guards and gates.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import train_eth_only_v1 as T  # noqa: E402

V2_CFG = REPO / "configs" / "eth_real_hardneg_yolo26s_v2.yaml"
TINY_CFG = REPO / "configs" / "tiny_recovery_yolo26s_v1.yaml"
RECIPE = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_official_recipe.json"
V2_TRAIN = REPO / "data" / "eth_real_hardneg_v2_train_manifest.csv"
FROZEN_KWARGS = ("imgsz", "epochs", "optimizer", "lr0", "momentum", "weight_decay", "seed", "box", "cls", "dfl",
                 "mosaic", "degrees", "translate", "scale", "shear", "perspective", "copy_paste", "fliplr",
                 "flipud", "mixup", "cutmix", "hsv_h", "hsv_s", "hsv_v", "deterministic", "nbs", "freeze")


def load_tiny(path: Path = TINY_CFG) -> dict:
    return T.load_contract(path)


def _resolved(path: Path) -> dict:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    return T.resolve_recipe(T.load_contract(path), recipe)


def test_tiny_contract_reuses_the_v1_loader_and_freezes_the_recipe():
    cfg = load_tiny()
    assert cfg["experiment"] == "eth_hardneg_tiny_recovery_v1"
    assert list(cfg["never_touch"]) == ["imgsz", "nbs", "optimizer", "lr0"]
    assert {k: _resolved(TINY_CFG)[k] for k in FROZEN_KWARGS} == {k: _resolved(V2_CFG)[k] for k in FROZEN_KWARGS}


def test_tiny_contract_freezes_the_hard_negative_intervention():
    tiny, v2 = load_tiny(), T.load_contract(V2_CFG)
    for key in ("hard_negative_sources", "hard_negative_repeat", "hard_negative_oversample",
                "hard_negative_source_weights", "hard_negative_exclude_list", "hard_negative_expected_unique"):
        assert tiny[key] == v2[key], key


def test_tiny_budget_and_gate_values_are_the_design_values():
    cfg = load_tiny()
    assert float(cfg["tiny_positive_max_equiv_size_640"]) == 8.0
    assert int(cfg["tiny_positive_extra_exposure"]) == 712
    assert int(cfg["tiny_extra_per_unique_image_max"]) == 1
    assert int(cfg["tiny_selection_seed"]) == 42
    assert cfg["tiny_stratify_buckets"] == ["<4", "4-6", "6-8"]
    assert cfg["tiny_require_exact_budget"] is True
    assert cfg["tiny_stop_if_insufficient"] is True
    assert Path(cfg["tiny_source_manifest"]).name == "eth_real_hardneg_v2_train_manifest.csv"


def test_tiny_endpoints_and_guards_match_the_design():
    cfg = load_tiny()
    assert cfg["primary_endpoint"] == "val|eth_unseen_recall_lt8"
    assert int(cfg["primary_min_tp_lt8"]) == 8
    assert int(cfg["strong_tp_lt8"]) == 12
    assert float(cfg["guard_recall_min"]) == 0.1840
    assert int(cfg["guard_no_target_fp_max"]) == 3
    assert float(cfg["guard_map5095_delta_min"]) == -0.01
    assert float(cfg["guard_recall_8_16_delta_min"]) == -0.02
    assert int(cfg["no_target_pool_images"]) == 496
    assert cfg["v2_reference"] == {"recall": 0.2040, "map5095": 0.189727, "tp_lt8": 4,
                                   "fp_no_target": 1, "recall_lt8": 0.0381}
    assert cfg["v1_reference"] == {"tp_lt8": 8, "recall_lt8": 0.0762, "fp_no_target": 6}


def test_tiny_decision_labels_are_the_six_from_the_design():
    assert load_tiny()["decisions"] == ["USEFUL", "STRONG_SUCCESS", "NO_TINY_GAIN", "FP_REGRESSION",
                                        "SIZE_TRADEOFF", "HARMFUL"]


def test_tiny_contract_has_no_duplicate_top_level_keys():
    keys = [line.split(":", 1)[0] for line in TINY_CFG.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith(("#", " ", "-")) and ":" in line]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    assert dupes == [], "duplicate top-level keys would let one value silently shadow another: %s" % dupes


def _tiny_pool():
    rows = list(csv.DictReader(V2_TRAIN.open(encoding="utf-8")))
    pos = [r for r in rows if str(r["is_negative"]).lower() in ("false", "0")]
    tiny = [r for r in pos if float(r["equiv_size_640"]) < 8.0]
    return pos, tiny


def test_the_real_v2_manifest_can_fund_the_712_exposure_budget():
    pos, tiny = _tiny_pool()
    assert len(pos) == 13992
    assert len(tiny) == 5965
    assert len(set(r["sha256"] for r in tiny)) == 5965, "one extra copy per unique image must be possible"
    buckets = Counter("<4" if float(r["equiv_size_640"]) < 4 else
                      ("4-6" if float(r["equiv_size_640"]) < 6 else "6-8") for r in tiny)
    assert dict(buckets) == {"<4": 161, "4-6": 1282, "6-8": 4522}


def test_stratified_allocation_of_712_is_exact_and_proportional():
    _, tiny = _tiny_pool()
    buckets = Counter("<4" if float(r["equiv_size_640"]) < 4 else
                      ("4-6" if float(r["equiv_size_640"]) < 6 else "6-8") for r in tiny)
    total = sum(buckets.values())
    raw = {b: 712 * buckets[b] / total for b in buckets}
    alloc = {b: int(raw[b]) for b in raw}
    for b in sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True)[:712 - sum(alloc.values())]:
        alloc[b] += 1
    assert sum(alloc.values()) == 712
    assert alloc == {"<4": 19, "4-6": 153, "6-8": 540}
    for b in alloc:
        assert alloc[b] <= buckets[b], "allocation must fit inside the eligible pool"
