"""Task 1 contract tests: CLEAN_OUTDOOR_CP_V1 must freeze the V2 recipe, the V2 dataset counts and the
pretrained checkpoint hash before any new data or training work starts.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 1, Step 1/2).
These tests are written before tools/clean_cp_contract.py exists, so the first run is RED (ImportError).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import clean_cp_contract as C  # noqa: E402

V2_ARGS = REPO / "configs" / "shuttle_detection" / "eth_real_hardneg_v2_resolved_args.yaml"
V2_TRAIN = REPO / "data" / "eth_real_hardneg_v2_train_manifest.csv"
V2_VAL = REPO / "data" / "eth_real_hardneg_v2_val_manifest.csv"
PRETRAINED = REPO / "assets" / "external" / "_staging" / "F_yolo" / "weights" / "yolo26s.pt"
NEW_CFG = REPO / "configs" / "eth_real_clean_outdoor_copypaste_v1.yaml"


def _v2_contract() -> dict:
    return C.load_v2_contract(V2_ARGS, V2_TRAIN, V2_VAL, PRETRAINED)


def test_recipe_parity_rejects_optimizer_auto():
    """optimizer=auto (or any drift from the frozen V2 kwargs) must block, not warn."""
    contract = _v2_contract()
    frozen = dict(contract["resolved_kwargs"])

    auto = dict(frozen)
    auto["optimizer"] = "auto"
    with pytest.raises(C.ContractBlocked) as exc:
        C.assert_frozen_recipe(auto, contract)
    assert "optimizer" in str(exc.value) and "auto" in str(exc.value)

    lr = dict(frozen)
    lr["lr0"] = 3e-4
    with pytest.raises(C.ContractBlocked):
        C.assert_frozen_recipe(lr, contract)

    missing = dict(frozen)
    missing.pop("weight_decay")
    with pytest.raises(C.ContractBlocked):
        C.assert_frozen_recipe(missing, contract)

    # the untouched copy passes, so the rejections above are caused by the single edit each
    C.assert_frozen_recipe(frozen, contract)


def test_frozen_pretrained_hash(tmp_path):
    """The official yolo26s.pt is frozen by bytes+sha256; any other file blocks."""
    contract = _v2_contract()
    assert contract["pretrained"]["sha256"] == C.FROZEN_PRETRAINED_SHA256
    assert contract["pretrained"]["bytes"] == C.FROZEN_PRETRAINED_BYTES

    impostor = tmp_path / "yolo26s.pt"
    impostor.write_bytes(b"not the official checkpoint")
    with pytest.raises(C.ContractBlocked) as exc:
        C.load_v2_contract(V2_ARGS, V2_TRAIN, V2_VAL, impostor)
    assert "pretrained" in str(exc.value).lower()

    with pytest.raises(C.ContractBlocked):
        C.load_v2_contract(V2_ARGS, V2_TRAIN, V2_VAL, tmp_path / "absent.pt")


def test_v2_counts_13992_551_712():
    """The frozen V2 training composition, derived from the manifest rather than typed in."""
    counts = _v2_contract()["counts"]
    assert counts["train_rows"] == 15255
    assert counts["positives"] == 13992
    assert counts["ordinary_negatives"] == 551
    assert counts["hard_negative_exposures"] == 712
    assert counts["hard_negative_unique"] == 89
    assert counts["repeat_factor"] == 8
    assert counts["positives"] + counts["ordinary_negatives"] + counts["hard_negative_exposures"] == 15255
    # the single row whose role column says official_positives but which carries no box is a documented
    # anomaly of the frozen V2 manifest: it must be counted as a negative and reported, never silently fixed.
    assert counts["role_column_anomalies"] == 1


def test_val_2920_unchanged():
    """The internal validation set is frozen byte-for-byte; it is the only checkpoint-selection set."""
    counts = _v2_contract()["counts"]
    assert counts["val_rows"] == 2920
    assert counts["val_sha256"] == C.FROZEN_V2_VAL_SHA256
    assert counts["val_sha256"] == C.sha256_file(V2_VAL)
    assert counts["val_negative_rows"] == 0
    assert C.FROZEN_V2_VAL_LOCATIONS == ("glc_2", "uetlibergstrasse_2")


def test_no_a0_or_p2():
    """Exactly one model, plain YOLO26s: no A0 arm, no P2 head, no resuming a frozen V2 checkpoint."""
    cfg = yaml.safe_load(NEW_CFG.read_text(encoding="utf-8"))
    C.assert_single_model_config(cfg)

    for mutate in (
        lambda c: c.update({"model_variant": "yolo26s_p2"}),
        lambda c: c.update({"experiment": "A0_no_augmentation"}),
        lambda c: c.update({"resume": True}),
        lambda c: c.update({"models": [{"id": "a"}, {"id": "b"}]}),
    ):
        broken = yaml.safe_load(NEW_CFG.read_text(encoding="utf-8"))
        mutate(broken)
        with pytest.raises(C.ContractBlocked):
            C.assert_single_model_config(broken)


def test_missing_v2_inputs_block(tmp_path):
    """Missing V2 artifacts raise ContractBlocked and are listed in missing_inputs - never defaulted."""
    with pytest.raises(C.ContractBlocked):
        C.load_v2_contract(tmp_path / "no_args.yaml", V2_TRAIN, V2_VAL, PRETRAINED)
    with pytest.raises(C.ContractBlocked):
        C.load_v2_contract(V2_ARGS, tmp_path / "no_train.csv", V2_VAL, PRETRAINED)

    mapping = C.create_contract_map(tmp_path)
    assert mapping["missing_inputs"], "an empty repo root must report every frozen input as missing"
    assert not any(v for k, v in mapping["verified"].items()), "missing inputs must never get a fabricated hash"
    assert mapping["gate"]["status"] == "BLOCKED_MISSING_V2_ARTIFACTS"
    assert mapping["spec_gate"]["status"] == "BLOCKED_MISSING_SPEC"


def test_contract_map_records_real_artifacts():
    """The shipped contract map is produced from the real repository and carries hashes and counts."""
    mapping = C.create_contract_map(REPO)
    verified = mapping["verified"]
    assert verified["v2_train_manifest"]["sha256"] == C.FROZEN_V2_TRAIN_SHA256
    assert verified["v2_val_manifest"]["sha256"] == C.FROZEN_V2_VAL_SHA256
    assert verified["v2_resolved_args"]["sha256"] == C.FROZEN_V2_ARGS_SHA256
    assert verified["pretrained"]["sha256"] == C.FROZEN_PRETRAINED_SHA256
    assert mapping["counts"]["train_rows"] == 15255
    assert mapping["counts"]["val_rows"] == 2920
    assert mapping["gate"]["status"] == "PASS", "every frozen V2 artifact resolved against the real repository"
    assert verified["canonical_evaluator"]["path"] == "tools/eval_yolo26_v1.py"
    # the plan's own precondition is NOT satisfied: the authoritative 2026-10-13 spec was never supplied.
    assert mapping["spec_gate"]["status"] == "BLOCKED_MISSING_SPEC"
    assert "new_spec" in {entry["name"] for entry in mapping["missing_inputs"]}


def test_new_config_passes_the_frozen_recipe():
    """The new frozen YAML must reproduce the V2 resolved kwargs exactly except the declared deltas."""
    contract = _v2_contract()
    cfg = yaml.safe_load(NEW_CFG.read_text(encoding="utf-8"))
    C.assert_frozen_recipe(cfg["resolved_kwargs"], contract)
    assert cfg["experiment"] == C.EXPERIMENT
    assert cfg["run_id"] == C.RUN_ID
    assert cfg["frozen"]["batch"] == 8 and cfg["frozen"]["nbs"] == 32
    assert cfg["counts"]["positives"] == 13992
