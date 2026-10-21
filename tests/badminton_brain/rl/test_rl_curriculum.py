# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.curriculum import (  # noqa: E402
    CurriculumDefinitionError, CurriculumSplitViolation, default_schedule,
)

SCHEDULE = default_schedule()


def test_validation_and_test_levels_are_never_training_levels():
    training = {lv.name for lv in SCHEDULE.training_levels()}
    assert training == {"L1-near", "L2-mid", "L3-far"}
    for lv in SCHEDULE.eval_levels():
        assert lv.split in ("val", "test")
        assert lv.name not in training
        with pytest.raises(CurriculumSplitViolation):
            SCHEDULE.require_training_level(lv.name)
        with pytest.raises(CurriculumSplitViolation):
            SCHEDULE.require_training_level(lv.name)
    for name in training:
        assert SCHEDULE.require_training_level(name).name == name


def test_eval_levels_cannot_be_used_as_eval_when_they_are_training():
    with pytest.raises(CurriculumSplitViolation):
        SCHEDULE.require_eval_level("L1-near")
    assert SCHEDULE.require_eval_level("V1-edge").split == "val"


def test_sampling_is_deterministic_and_inside_the_declared_ranges():
    for lv in SCHEDULE.levels:
        for seed in (0, 1, 12345, 2**40 + 7):
            a = SCHEDULE.sample(lv.name, seed)
            b = SCHEDULE.sample(lv.name, seed)
            assert a == b
            assert set(a) == set(lv.parameter_names)
            for name, value in a.items():
                assert lv.range_for(name).contains(value), (lv.name, name, value)
    assert SCHEDULE.sample("L1-near", 1) != SCHEDULE.sample("L1-near", 2)


def test_declared_parameter_names_are_shared_by_every_level():
    names = {lv.parameter_names for lv in SCHEDULE.levels}
    assert len(names) == 1
    with pytest.raises(CurriculumDefinitionError):
        SCHEDULE.level("does-not-exist")


def test_schedule_dict_records_the_split_gate():
    doc = SCHEDULE.to_dict()
    assert doc["training_levels"] == ["L1-near", "L2-mid", "L3-far"]
    assert set(doc["eval_levels"]) == {"V1-edge", "V2-tight-reach", "T1-corner", "T2-late"}
    assert all(lv["split"] == "train" for lv in doc["levels"]
               if lv["name"] in doc["training_levels"])
