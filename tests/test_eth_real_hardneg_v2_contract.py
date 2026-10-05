"""V2 contract tests: the ETH hard-negative experiment must reuse the V1 recipe and evaluator exactly.

Rulings applied (2026-10-05): (1) freeze the two verified local pools hard_negatives/hard_negatives2 minus the
7 names in excluded.txt = 89 unique SHA256s; (2) hard_negative_repeat = 8, no other oversampling or source
weighting; (3) the real no-target endpoint is the pooled 496-image FP/image with raw counts and a 95% CI,
keeping per-set breakdowns; (4) reuse tools/train_eth_only_v1.py instead of a duplicate launcher.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import train_eth_only_v1 as T  # noqa: E402

V1_CFG = REPO / "configs" / "eth_only_yolo26s_1024_v1.yaml"
V2_CFG = REPO / "configs" / "eth_real_hardneg_yolo26s_v2.yaml"
RECIPE = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_official_recipe.json"
EXCLUDED = REPO / "outputs" / "shuttle_capability" / "hard_negatives2" / "excluded.txt"

# The plan freezes augmentation/loss/optimizer values that live in the resolved official recipe (Task 1 of V1),
# not in the V1 yaml. Parity is therefore asserted on the RESOLVED kwargs through the same function the
# launcher uses, which is strictly stronger than comparing raw yaml keys and cannot drift silently.
FROZEN_KWARGS = ("imgsz", "epochs", "optimizer", "lr0", "momentum", "weight_decay", "seed", "box", "cls",
                 "dfl", "mosaic", "degrees", "translate", "scale", "shear", "perspective", "copy_paste",
                 "fliplr", "flipud", "mixup", "cutmix", "hsv_h", "hsv_s", "hsv_v", "deterministic",
                 "nbs", "freeze")


def load_v2_config(path: Path = V2_CFG) -> dict:
    """V2 uses the V1 contract loader: no second parser exists in this repository."""
    return T.load_contract(path)


def _resolved(path: Path) -> dict:
    contract = T.load_contract(path)
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    return T.resolve_recipe(contract, recipe)


def test_v2_reuses_the_v1_contract_loader():
    import inspect
    cfg = load_v2_config()
    assert cfg == T.load_contract(V2_CFG), "V2 must resolve through the V1 contract loader"
    src = inspect.getsource(load_v2_config)
    assert "safe_load" not in src and "yaml" not in src, "V2 must not carry a second YAML parser"
    assert cfg["experiment"] == "eth_real_hardneg_yolo26s_v2"
    assert list(cfg["never_touch"]) == ["imgsz", "nbs", "optimizer", "lr0"]


def test_v2_training_recipe_matches_v1_exactly():
    v1 = _resolved(V1_CFG)
    v2 = _resolved(V2_CFG)
    assert {k: v2[k] for k in FROZEN_KWARGS} == {k: v1[k] for k in FROZEN_KWARGS}
    assert v2["imgsz"] == 1024 and v2["epochs"] == 50 and v2["nbs"] == 32 and v2["freeze"] == 0
    assert v2["optimizer"] == "AdamW" and float(v2["lr0"]) == 1e-4


def test_v2_forbids_positive_domain_additions():
    cfg = load_v2_config()
    assert cfg["include_synthetic"] is False
    assert cfg["include_iphone"] is False
    assert cfg["include_d455_positive"] is False
    assert cfg["include_roboflow"] is False
    assert cfg["adds_positives"] is False


def test_hard_negative_sources_are_the_two_verified_pools():
    cfg = load_v2_config()
    assert cfg["hard_negative_sources"] == ["hard_negatives", "hard_negatives2"]
    assert all(isinstance(x, str) and x.strip() for x in cfg["hard_negative_sources"])


def test_repeat_is_eight_and_no_other_weighting_is_enabled():
    cfg = load_v2_config()
    assert int(cfg["hard_negative_repeat"]) == 8
    assert cfg["hard_negative_oversample"] is False
    assert cfg["hard_negative_source_weights"] in (None, {}, [])


def test_the_excluded_list_is_the_frozen_seven_names():
    names = [x.strip() for x in EXCLUDED.read_text(encoding="utf-8").split() if x.strip()]
    assert len(names) == 7 and len(set(names)) == 7
    assert all(n.startswith("hn2_") and n.endswith(".jpg") for n in names)
    cfg = load_v2_config()
    assert str(cfg["hard_negative_exclude_list"]).endswith("hard_negatives2/excluded.txt")


def test_v2_pins_the_same_evaluation_policy_as_v1():
    v1, v2 = load_v2_config(V1_CFG), load_v2_config()
    for key in ("selection_metric", "checkpoint_selection_scope", "confidence_policy", "nms_policy",
                "size_bucket", "inconclusive_abs_diff", "confirmation_range", "actionable_abs_diff"):
        assert v2[key] == v1[key], key
    assert v2["evaluation_only_sets"] == v1["evaluation_only_sets"]
    assert v2["primary_endpoint"].startswith("pooled_real_no_target_fp_per_image")
    assert int(v2["primary_endpoint_images"]) == 496


def test_v2_reuses_the_v1_manifests_and_training_machinery():
    cfg = load_v2_config()
    assert Path(cfg["v1_train_manifest"]).name == "eth_only_v1_train_manifest.csv"
    assert Path(cfg["v1_val_manifest"]).name == "eth_only_v1_val_manifest.csv"
    assert cfg["training_launcher"] == "tools/train_eth_only_v1.py"
    assert cfg["evaluator"] == "tools/eval_yolo26_v1.py"


def test_the_four_rulings_are_recorded_in_the_contract():
    cfg = load_v2_config()
    rulings = cfg["rulings"]
    assert len(rulings) == 4
    joined = json.dumps(rulings, ensure_ascii=False)
    assert "hard_negatives2" in joined and "repeat" in joined.lower() and "95" in joined
