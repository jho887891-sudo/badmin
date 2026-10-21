# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.action_schema import DEFAULT_DT_S, DEFAULT_LIMITS, to_command  # noqa: E402
from badminton_brain.rl.baselines import (  # noqa: E402
    FSM_HOLD_INVALID_PREDICTION, FSM_HOLD_OUT_OF_WORKSPACE, FSM_HOLD_TIMING,
    FSM_HOLD_UNREACHABLE, FSM_TRACK, P7_FSM_EXPECTED_AVAILABLE, LocalReachabilityFSMStub,
    ObservationInterfaceError, ObservationView, build_baselines, intercept_target, p7_fsm_status,
)
from badminton_brain.rl.curriculum import default_schedule  # noqa: E402
from badminton_brain.rl.evaluation import (  # noqa: E402
    SyntheticState, build_observation, episodes_from_ledger, evaluate, run_episode,
)
from badminton_brain.rl.seed_ledger import SeedLedger  # noqa: E402

LEDGER = SeedLedger.derive(task_family="racket_contact_targeting", master_seed=20261022,
                           counts={"train": 40, "val": 20, "test": 20})
SCHEDULE = default_schedule()
POLICIES = build_baselines()


def episode_list(level="L2-mid", split="train", count=4):
    return episodes_from_ledger(LEDGER, SCHEDULE, split=split, level=level, count=count)


def first_view(level="L2-mid", split="train"):
    ep = episode_list(level, split, 1).episodes[0]
    state = SyntheticState(ep.initial_base_position.copy(), ep.initial_joint_position.copy())
    return ObservationView(build_observation(state, ep, 0.0)), ep


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_baselines_emit_actions_the_p8_01_schema_accepts(name):
    policy = POLICIES[name]
    view, ep = first_view()
    action = policy.act(view, joint_position=ep.initial_joint_position)
    command = to_command(action, joint_position=ep.initial_joint_position, dt_s=DEFAULT_DT_S)
    assert command.simulation_only is True
    assert np.all(np.abs(command.base_twist_body) <= np.array([0.60, 0.60, 1.00]) + 1e-12)
    assert np.all(np.abs(command.joint_position_target[:3]) <= 2.60)
    assert set(action) == {"base_linear_velocity", "base_angular_velocity",
                           "joint_position_delta"}


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_baselines_respect_the_rate_limits_and_the_safety_envelope(name):
    episodes = episode_list("L2-mid", "train", 3)
    report, outcomes = evaluate(POLICIES[name], episodes, LEDGER)
    assert report.action_reject == 0
    # the P5-05 envelope (SafetyShield) must never have to reject a baseline command
    assert report.safety_reject == 0
    assert {o.shield_action for o in outcomes} == {"PASS"}
    assert report.success + report.timeout == report.episodes_total


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_step_to_step_changes_stay_inside_the_acceleration_limits(name):
    policy = POLICIES[name]
    ep = episode_list("L3-far", "train", 1).episodes[0]
    state = SyntheticState(ep.initial_base_position.copy(), ep.initial_joint_position.copy())
    previous = None
    for k in range(12):
        view = ObservationView(build_observation(state, ep, 0.02 * k))
        action = policy.act(view, joint_position=state.joint_position, previous_action=previous)
        # to_command enforces the acceleration limits through previous_action
        command = to_command(action, joint_position=state.joint_position,
                             previous_action=previous, dt_s=DEFAULT_DT_S)
        previous = action
        state.base_position = np.array(
            [state.base_position[0] + 0.02 * float(command.base_twist_body[0]),
             state.base_position[1] + 0.02 * float(command.base_twist_body[1]),
             state.base_position[2]])
        state.joint_position = np.asarray(command.joint_position_target, dtype=float)


def test_fsm_stub_maps_the_feasibility_flags_to_fail_closed_states():
    stub = LocalReachabilityFSMStub()
    assert stub.verdict((True, True, True, True)).state == FSM_TRACK
    assert stub.verdict((False, True, True, True)).state == FSM_HOLD_UNREACHABLE
    assert stub.verdict((True, False, True, True)).state == FSM_HOLD_OUT_OF_WORKSPACE
    assert stub.verdict((True, True, False, True)).state == FSM_HOLD_TIMING
    assert stub.verdict((True, True, True, False)).state == FSM_HOLD_INVALID_PREDICTION
    assert all(not stub.verdict(flags).may_act for flags in
               [(False, True, True, True), (True, False, True, True),
                (True, True, False, True), (True, True, True, False)])


def test_fsm_baseline_holds_with_zero_action_when_the_verdict_refuses():
    ep = episode_list("V2-tight-reach", "val", 1).episodes[0]
    assert ep.feasibility_flags[0] is False
    state = SyntheticState(ep.initial_base_position.copy(), ep.initial_joint_position.copy())
    view = ObservationView(build_observation(state, ep, 0.0))
    policy = POLICIES["reachability-fsm"]
    action = policy.act(view, joint_position=state.joint_position)
    assert policy.last_state == FSM_HOLD_UNREACHABLE
    assert np.allclose(action["base_linear_velocity"].value, 0.0)
    assert np.allclose(action["joint_position_delta"].value, 0.0)


def test_tracking_baseline_holds_still_at_zero_error():
    ep = episode_list("L1-near", "train", 1).episodes[0]
    # put the racket exactly on the interpolated target
    state = SyntheticState(np.array([ep.target[0] - 0.30, ep.target[1], 0.0]), np.zeros(6))
    q2 = float(np.arcsin(np.clip((ep.target[2] - 0.60) / 0.40, -1.0, 1.0)))
    state.joint_position = np.array([0.0, 0.0, q2, 0.0, 0.0, 0.0])
    view = ObservationView(build_observation(state, ep, 0.0))
    assert float(np.linalg.norm(intercept_target(view) - view.racket_position)) < 1e-9
    action = POLICIES["tracking-p"].act(view, joint_position=state.joint_position)
    assert np.allclose(action["base_linear_velocity"].value, 0.0)


def test_observation_view_refuses_an_unvalidated_vector():
    with pytest.raises(ObservationInterfaceError):
        ObservationView(np.zeros(3))
    with pytest.raises(Exception):
        ObservationView(np.full(200, np.nan))


def test_p7_fsm_availability_is_recorded_and_the_stub_does_not_claim_p7():
    status = p7_fsm_status()
    assert status["verdict_source"] == "LocalReachabilityFSMStub"
    assert status["importable"] is P7_FSM_EXPECTED_AVAILABLE, (
        "the P7 interaction FSM importability changed: write the real verdict adapter and update "
        "P7_FSM_EXPECTED_AVAILABLE instead of silently switching verdict sources")


def test_baselines_import_no_learning_stack():
    assert "torch" not in sys.modules
    report, _ = evaluate(POLICIES["tracking-p"], episode_list("L1-near", "train", 1), LEDGER)
    assert report.synthetic is True
    assert report.status == "NON_RL_BASELINE_NO_LEARNING"


def test_run_episode_uses_the_shared_harness():
    ep = episode_list("L1-near", "train", 1).episodes[0]
    outcome = run_episode(POLICIES["tracking-p"], ep)
    assert outcome.result in ("success", "timeout", "safety_reject", "action_reject")
    assert outcome.steps <= ep.horizon_steps
