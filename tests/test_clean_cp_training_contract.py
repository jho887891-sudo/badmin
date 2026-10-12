"""Task 7 tests: exactly one run, exactly 15,255 slots per epoch, the old exposures untouched, the new
pools capped and replayable by the loader.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 7, Step 1/2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_single_clean_cp_train as B  # noqa: E402

CONFIG = dict(B.default_budget())
CONFIG["epochs"] = 2
FULL_CONFIG = dict(B.default_budget())


def _v2_rows() -> pd.DataFrame:
    records = []
    for index in range(13992):
        records.append({"role": B.ROLE_V2_POSITIVE, "sample_id": "/d/pos/%05d.jpg" % index,
                        "label_path": "/d/pos/%05d.txt" % index, "is_positive": True, "repeat_index": 0,
                        "coco_image_id": None, "background_id": None, "near_duplicate_group": None})
    for index in range(551):
        records.append({"role": B.ROLE_V2_ORDINARY_NEGATIVE, "sample_id": "/d/neg/%03d.jpg" % index,
                        "label_path": "", "is_positive": False, "repeat_index": 0,
                        "coco_image_id": None, "background_id": None, "near_duplicate_group": None})
    for index in range(89):
        for repeat in range(8):
            records.append({"role": B.ROLE_V2_HARD_NEGATIVE, "sample_id": "/d/hn/%03d.jpg" % index,
                            "label_path": "", "is_positive": False, "repeat_index": repeat,
                            "coco_image_id": None, "background_id": None, "near_duplicate_group": None})
    return pd.DataFrame(records, columns=list(B.SLOT_COLUMNS[2:]))


def _cp_rows(count: int = 400) -> pd.DataFrame:
    return pd.DataFrame([{
        "role": B.ROLE_CP_POSITIVE, "sample_id": "/cp/%04d.jpg" % index, "label_path": "/cp/%04d.txt" % index,
        "is_positive": True, "repeat_index": 0, "coco_image_id": 9000000 + (index % 120),
        "background_id": "bg-%03d" % (index % 120), "near_duplicate_group": "nd-%04d" % (index % 120),
    } for index in range(count)], columns=list(B.SLOT_COLUMNS[2:]))


def _new_hn_rows(count: int = 300) -> pd.DataFrame:
    return pd.DataFrame([{
        "role": B.ROLE_NEW_HARD_NEGATIVE, "sample_id": "/hn/%04d.jpg" % index, "label_path": "",
        "is_positive": False, "repeat_index": 0, "coco_image_id": 8000000 + index,
        "background_id": "", "near_duplicate_group": "ndhn-%04d" % index,
    } for index in range(count)], columns=list(B.SLOT_COLUMNS[2:]))


def _slots(epoch: int = 0):
    return B.build_epoch_slots(_v2_rows(), _cp_rows(), _new_hn_rows(), CONFIG, epoch)


def test_single_run_only():
    """A second run, a second arm or step inflation is refused before any slot is built."""
    with pytest.raises(B.BudgetConflict):
        B.build_epoch_slots(_v2_rows(), _cp_rows(), _new_hn_rows(), {**CONFIG, "runs": ["a", "b"]}, 0)
    with pytest.raises(B.BudgetConflict):
        B.build_epoch_slots(_v2_rows(), _cp_rows(), _new_hn_rows(), {**CONFIG, "step_inflation_allowed": True}, 0)


def test_each_epoch_exactly_15255_slots():
    for epoch in (0, 1):
        slots = _slots(epoch)
        assert len(slots) == 15255
        assert slots["slot_index"].tolist() == list(range(15255))


def test_original_positive_exposures_at_least_11194():
    slots = _slots(0)
    positives = int((slots["role"] == B.ROLE_V2_POSITIVE).sum())
    assert positives >= 11194
    report = B.validate_epoch_budget(slots, CONFIG)
    assert report["ok"], report["violations"]


def test_old_ordinary_negative_551_and_old_hardneg_712_retained():
    slots = _slots(0)
    assert int((slots["role"] == B.ROLE_V2_ORDINARY_NEGATIVE).sum()) == 551
    assert int((slots["role"] == B.ROLE_V2_HARD_NEGATIVE).sum()) == 712
    repeats = slots[slots["role"] == B.ROLE_V2_HARD_NEGATIVE]["sample_id"].value_counts()
    assert set(repeats.unique()) == {8}, "each historical hard negative keeps its 8 exposures"


def test_new_outdoor_hardneg_at_most_5pct():
    slots = _slots(0)
    new_hn = int((slots["role"] == B.ROLE_NEW_HARD_NEGATIVE).sum())
    assert new_hn <= 762
    assert new_hn / 15255 <= 0.05
    joint = new_hn + int((slots["role"] == B.ROLE_CP_POSITIVE).sum())
    assert joint <= 2798


def test_new_hardneg_once_per_unique_per_epoch():
    slots = _slots(0)
    new_hn = slots[slots["role"] == B.ROLE_NEW_HARD_NEGATIVE]
    assert not new_hn["sample_id"].duplicated().any()
    assert new_hn["exposure_index"].max() == 0


def test_cp_positive_label_not_empty():
    slots = _slots(0)
    cp = slots[slots["role"] == B.ROLE_CP_POSITIVE]
    assert len(cp) > 0
    assert cp["is_positive"].all()
    assert (cp["label_path"].str.strip() != "").all()


def test_cp_background_never_drawn_as_hardnegative():
    slots = _slots(0)
    poisoned = slots.copy()
    hn_id = poisoned.loc[poisoned["role"] == B.ROLE_NEW_HARD_NEGATIVE, "coco_image_id"].iloc[0]
    poisoned.loc[poisoned["role"] == B.ROLE_CP_POSITIVE, "coco_image_id"] = hn_id
    report = B.validate_epoch_budget(poisoned, CONFIG)
    assert not report["ok"]
    assert any("background and hard negative" in violation for violation in report["violations"])


def test_no_coco_image_id_or_neardup_across_pools():
    cp_rows, hn_rows = _cp_rows(), _new_hn_rows()
    B.assert_pools_disjoint(cp_rows, hn_rows)

    hn_rows_clash = hn_rows.copy()
    hn_rows_clash.loc[0, "coco_image_id"] = cp_rows.loc[0, "coco_image_id"]
    with pytest.raises(B.BudgetConflict):
        B.assert_pools_disjoint(cp_rows, hn_rows_clash)

    hn_rows_group = hn_rows.copy()
    hn_rows_group.loc[0, "near_duplicate_group"] = cp_rows.loc[0, "near_duplicate_group"]
    with pytest.raises(B.BudgetConflict):
        B.assert_pools_disjoint(cp_rows, hn_rows_group)


def test_no_double_shuffle_or_loader_step_inflation():
    """The step count follows the frozen slot count; an epoch is 1,907 batches and 477 optimizer steps."""
    plan = B.step_plan(FULL_CONFIG)
    assert plan["dataloader_batches"] == 1907
    assert plan["gradient_accumulation"] == 4
    assert plan["steps_per_epoch"] == 477
    assert plan["optimizer_steps"] == 23850
    assert FULL_CONFIG["step_inflation_allowed"] is False


def test_sampler_replay_and_actual_exposure_match(tmp_path):
    summary = B.freeze_training_slots(_v2_rows(), _cp_rows(), _new_hn_rows(), CONFIG, tmp_path / "slots.csv")
    assert summary["rows"] == 2 * 15255
    sampler = B.CleanCPIndexSampler(tmp_path / "slots.csv", CONFIG)
    frozen = pd.read_csv(tmp_path / "slots.csv", keep_default_na=False, dtype=str)
    sampler.assert_replay_matches(frozen)
    first = sampler.epoch_sample_ids(0)
    second = sampler.epoch_sample_ids(0)
    assert first == second
    assert len(first) == 15255
    assert first != sampler.epoch_sample_ids(1), "epochs are drawn independently"
    report = sampler.exposure_report()
    assert report["total_slots"] == 2 * 15255
    assert report["by_role"][B.ROLE_V2_HARD_NEGATIVE] == 2 * 712

    mutated = frozen.copy()
    mutated.loc[0, "sample_id"] = "tampered"
    with pytest.raises(B.BudgetConflict):
        sampler.assert_replay_matches(mutated)
