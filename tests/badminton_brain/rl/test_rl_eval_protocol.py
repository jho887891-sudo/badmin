# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.baselines import build_baselines  # noqa: E402
from badminton_brain.rl.curriculum import (  # noqa: E402
    CurriculumSplitViolation, default_schedule,
)
from badminton_brain.rl.evaluation import (  # noqa: E402
    EvaluationProtocolError, EpisodeList, LedgerEpisodeMismatchError,
    assert_identical_episodes, check_alignment, episodes_from_ledger, evaluate,
    evaluate_baselines,
)
from badminton_brain.rl.seed_ledger import SeedLedger  # noqa: E402

COUNTS = {"train": 60, "val": 20, "test": 20}
LEDGER = SeedLedger.derive(task_family="racket_contact_targeting", master_seed=20261022,
                           counts=COUNTS)
OTHER = SeedLedger.derive(task_family="racket_contact_targeting", master_seed=20261023,
                          counts=COUNTS)
SCHEDULE = default_schedule()
POLICIES = build_baselines()


def episodes(level="L2-mid", split="train", count=4):
    return episodes_from_ledger(LEDGER, SCHEDULE, split=split, level=level, count=count)


def test_episode_list_is_reproducible_from_the_same_ledger():
    a, b = episodes(), episodes()
    assert a.fingerprint() == b.fingerprint()
    assert_identical_episodes(a, b)
    check_alignment(a, LEDGER)
    other_level = episodes("L3-far")
    with pytest.raises(EvaluationProtocolError):
        assert_identical_episodes(a, other_level)


def test_harness_refuses_a_ledger_that_disagrees_with_the_episode_list():
    ep = episodes()
    with pytest.raises(LedgerEpisodeMismatchError) as excinfo:
        evaluate(POLICIES["tracking-p"], ep, OTHER)
    assert "ledger" in str(excinfo.value).lower()

    forged = EpisodeList(episodes=ep.episodes, ledger_hash=OTHER.ledger_hash(), split=ep.split,
                         level=ep.level)
    with pytest.raises(LedgerEpisodeMismatchError):
        evaluate(POLICIES["tracking-p"], forged, LEDGER)


def test_harness_refuses_an_episode_seed_that_is_not_in_the_split_seed_set():
    ep = episodes()
    foreign = LEDGER.seed("test", 0)
    assert foreign not in LEDGER.seeds_for(ep.split)
    # rebuild the same episodes with one foreign seed (no dataclass magic: explicit rebuild)
    swapped = []
    for episode in ep.episodes:
        swapped.append(type(episode)(**{**episode.__dict__, "seed": foreign,
                                        "episode_id": episode.episode_id + "/foreign"}))
    with pytest.raises(LedgerEpisodeMismatchError):
        evaluate(POLICIES["tracking-p"],
                 EpisodeList(episodes=tuple(swapped), ledger_hash=ep.ledger_hash,
                             split=ep.split, level=ep.level), LEDGER)


def test_level_split_and_seed_split_must_agree():
    with pytest.raises(CurriculumSplitViolation):
        episodes_from_ledger(LEDGER, SCHEDULE, split="train", level="V1-edge", count=2)
    with pytest.raises(CurriculumSplitViolation):
        episodes_from_ledger(LEDGER, SCHEDULE, split="val", level="L1-near", count=2)


def test_both_baselines_run_on_the_identical_episode_list_and_the_counts_add_up():
    ep = episodes("L2-mid", "train", 5)
    results = evaluate_baselines(POLICIES, ep, LEDGER)
    fingerprints = set()
    for name, (report, outcomes) in results.items():
        fingerprints.add(report.episode_fingerprint)
        assert report.episodes_total == 5
        assert report.success + report.timeout + report.safety_reject + report.action_reject == 5
        assert report.success_rate == pytest.approx(report.success / 5.0)
        assert report.synthetic is True
        assert report.status == "NON_RL_BASELINE_NO_LEARNING"
        assert report.ledger_hash == LEDGER.ledger_hash()
        assert len(outcomes) == 5
    assert len(fingerprints) == 1, "the two baselines must share one episode list"


def test_report_json_is_labelled_synthetic_and_contains_no_rl_claim():
    ep = episodes("V2-tight-reach", "val", 3)
    report, outcomes = evaluate(POLICIES["reachability-fsm"], ep, LEDGER)
    doc = report.to_dict(include_outcomes=True, outcomes=outcomes)
    assert doc["synthetic"] is True
    assert doc["dynamics"].startswith("documented kinematic toy")
    assert "NON_RL" in doc["status"]
    assert doc["success"] + doc["timeout"] + doc["safety_reject"] + doc["action_reject"] == 3
    assert len(doc["outcomes"]) == 3
    assert all("shield_action" in row for row in doc["outcomes"])


def test_eval_levels_are_drawn_from_the_matching_seed_split():
    val = episodes("V1-edge", "val", 2)
    test = episodes("T1-corner", "test", 2)
    assert all(s in LEDGER.seeds_for("val") for s in (e.seed for e in val.episodes))
    assert all(s in LEDGER.seeds_for("test") for s in (e.seed for e in test.episodes))
    assert set(e.seed for e in val.episodes) & set(e.seed for e in test.episodes) == set()
