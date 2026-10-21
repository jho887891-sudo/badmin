# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.seed_ledger import (  # noqa: E402
    BAND_WIDTH, MAX_DERIVATIVES_PER_BASE, SPLITS, LedgerIntegrityError,
    OverlappingSplitRangeError, SeedExhaustionError, SeedLedger, allocate_split_ranges,
    default_band_layout, derive_derivative_seed, derive_seed, split_band,
)

FAMILY = "racket_contact_targeting"
MASTER = 20261022
COUNTS = {"train": 2000, "val": 800, "test": 800}


def build() -> SeedLedger:
    return SeedLedger.derive(task_family=FAMILY, master_seed=MASTER, counts=COUNTS)


def test_splits_are_pairwise_disjoint_over_a_large_n():
    ledger = build()
    sets = {s: set(ledger.seeds_for(s)) for s in SPLITS}
    assert len(sets["train"]) == COUNTS["train"]
    assert len(sets["val"]) == COUNTS["val"]
    assert len(sets["test"]) == COUNTS["test"]
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    # and the disjointness is structural, not luck: each seed is inside its own band
    for split in SPLITS:
        lo, hi = split_band(split)
        assert all(lo <= s < hi for s in sets[split])
    assert max(sets["train"]) < min(sets["val"]) < max(sets["val"]) < min(sets["test"])


def test_derivation_is_deterministic_and_master_seed_sensitive():
    a, b = build(), build()
    assert a.ledger_hash() == b.ledger_hash()
    assert a.seeds_for("train") == b.seeds_for("train")
    other_master = SeedLedger.derive(task_family=FAMILY, master_seed=MASTER + 1, counts=COUNTS)
    assert other_master.ledger_hash() != a.ledger_hash()
    assert other_master.seeds_for("train") != a.seeds_for("train")
    other_family = SeedLedger.derive(task_family="another_family", master_seed=MASTER,
                                     counts=COUNTS)
    assert other_family.ledger_hash() != a.ledger_hash()


def test_scene_derivative_seeds_stay_inside_their_base_window_and_split():
    ledger = build()
    parameters = ["scene-light-a", "scene-light-b", "ball-1", "ball-2"]
    all_derived = {}
    for split in SPLITS:
        lo, hi = split_band(split)
        for index in range(40):
            base = ledger.seed(split, index)
            derived = ledger.derivative_seeds(split, index, parameters)
            assert len(set(derived)) == len(derived)
            for seed in derived:
                assert base <= seed < base + MAX_DERIVATIVES_PER_BASE
                assert lo <= seed < hi
            all_derived[split] = all_derived.get(split, set()) | set(derived)
    assert all_derived["train"] & all_derived["val"] == set()
    assert all_derived["train"] & all_derived["test"] == set()
    assert all_derived["val"] & all_derived["test"] == set()
    for split in SPLITS:
        for index in range(5):
            assert derive_derivative_seed(MASTER, FAMILY, split, index, "ball-1") in (
                ledger.derivative_seeds(split, index, ["ball-1"]))


def test_ledger_hash_changes_when_any_single_seed_changes():
    ledger = build()
    original = ledger.ledger_hash()
    for split in SPLITS:
        for index in (0, 7, 199):
            if index >= len(ledger.seeds_for(split)):
                continue
            old = ledger.seed(split, index)
            bumped = ledger.replace_seed(split, index, old + MAX_DERIVATIVES_PER_BASE)
            bumped.verify()
            assert bumped.ledger_hash() != original
            assert bumped.seed(split, index) == old + MAX_DERIVATIVES_PER_BASE


def test_overlapping_split_ranges_are_refused():
    with pytest.raises(OverlappingSplitRangeError) as excinfo:
        allocate_split_ranges({"train": (0, 100), "val": (50, 150)})
    message = str(excinfo.value)
    assert "overlap" in message and "train" in message and "val" in message
    bad_layout = default_band_layout()
    bad_layout["test"] = bad_layout["val"]
    with pytest.raises(OverlappingSplitRangeError):
        SeedLedger.derive(task_family=FAMILY, master_seed=MASTER, counts={"train": 4, "val": 4,
                                                                         "test": 4},
                          band_layout=bad_layout)


def test_hand_edited_ledger_is_refused_and_hash_mismatch_is_detected(tmp_path):
    ledger = build()
    path = ledger.write(tmp_path / "seed_ledger.json")
    loaded = SeedLedger.load(path)
    assert loaded.ledger_hash() == ledger.ledger_hash()

    doc = ledger.to_dict()
    doc["seeds"]["test"] = list(doc["seeds"]["test"])
    doc["seeds"]["test"][0] = doc["seeds"]["train"][0]  # moves a train seed into test
    with pytest.raises(LedgerIntegrityError):
        SeedLedger.from_dict(doc)

    doc = ledger.to_dict()
    doc["seeds"]["train"] = list(doc["seeds"]["train"])
    doc["seeds"]["train"][0] += MAX_DERIVATIVES_PER_BASE  # legal window, wrong seed
    with pytest.raises(LedgerIntegrityError) as excinfo:
        SeedLedger.from_dict(doc)
    assert "hash mismatch" in str(excinfo.value)


def test_band_exhaustion_and_bad_split_are_refused():
    small = {"train": (0, 256), "val": (256, 512), "test": (512, 768)}
    with pytest.raises(SeedExhaustionError):
        SeedLedger.derive(task_family=FAMILY, master_seed=MASTER,
                          counts={"train": 5, "val": 1, "test": 1}, band_layout=small)
    ok = SeedLedger.derive(task_family=FAMILY, master_seed=MASTER,
                           counts={"train": 4, "val": 4, "test": 4}, band_layout=small)
    assert ok.seeds_for("train")[0] >= 0 and ok.seeds_for("test")[0] >= 512
    with pytest.raises(SeedExhaustionError):
        derive_seed(MASTER, FAMILY, "train", 1 << 34)  # == SLOTS_PER_BAND, one past the last slot
    with pytest.raises(Exception):
        SeedLedger.derive(task_family=FAMILY, master_seed=MASTER, counts={"train": 1, "val": 1})
