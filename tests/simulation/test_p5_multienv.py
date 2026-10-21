# -*- coding: utf-8 -*-
"""P5-07 / P5-08 tests: multi-environment isolation, determinism and fault containment.

Everything here runs OFFLINE: no Isaac Sim, no GPU, no torch, no pxr. The 'simulator' is
badminton_scene.multienv.OfflineMockSim, whose only physics is "the accepted setpoint becomes the
joint position". These tests therefore prove properties of the HARNESS - isolation, determinism,
validation order, fault containment - and prove NOTHING about Isaac throughput or about 16/64/128
real environments. The management report carries the same caveat.

The independence assertions are exact: a state digest is a sha256 over float.hex() values, so
"env B is unchanged" means bit-identical, not "close to". Two independent sides are kept apart on
purpose: the USD limits used by the rest-pose guard are read from the committed pxr export and
converted here (deg -> rad), never from the module under test.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = REPO_ROOT / "simulation"
if str(SCENE_DIR) not in sys.path:
    sys.path.insert(0, str(SCENE_DIR))

from badminton_scene import multienv as me  # noqa: E402
from badminton_scene import rest_pose  # noqa: E402

FACTS_JSON = (REPO_ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "morph_one"
              / "p5_static" / "arm_joint_facts.json")

#: The RS runtime name set, as literals (not imported from the module under test).
RS_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
            "gripper_joint1", "gripper_joint2")
#: The Stage-0 PiPER pose that must never be written on this arm (joint3 = -1.5 is out of [0, 3.14]).
LEGACY_PIPER_REST = {"joint1": 0.0, "joint2": 1.5, "joint3": -1.5,
                     "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}


def usd_limits():
    """{joint: (lower, upper, unit)} from the pxr export, degrees converted to radians HERE."""
    facts = json.loads(FACTS_JSON.read_text(encoding="utf-8"))
    out = {}
    for joint in facts["joints"]:
        limits = joint.get("limits") or {}
        lower, upper = limits.get("lower"), limits.get("upper")
        if lower is None or upper is None:
            continue  # the Fixed root joint has no travel
        if limits.get("unit") == "degree":
            out[joint["name"]] = (math.radians(lower), math.radians(upper), "rad")
        else:
            out[joint["name"]] = (float(lower), float(upper), limits.get("unit", ""))
    return out


def harness(env_count=2, seed=11, **kwargs):
    return me.build_mock_harness(env_count=env_count, seed=seed, **kwargs)


def drive(h, steps=4):
    for _ in range(steps):
        for env_id in h.env_ids:
            result = h.step(env_id)
            assert result.accepted, result.as_dict()


def digest_of(h, env_id):
    return h.state_digest(env_id)


# --------------------------------------------------------------------------- 1. separation
def test_envs_own_separate_objects_and_seeds():
    h = harness(2)
    assert h.env_count == 2
    envs = [h._env(env_id) for env_id in h.env_ids]
    assert envs[0].controller is not envs[1].controller
    assert envs[0].controller.gate is not envs[1].controller.gate
    assert envs[0].rng is not envs[1].rng
    # one derived seed per env: two envs given the SAME base seed still get different streams
    assert me.derive_seed(5, 0) == me.derive_seed(5, 0)
    assert me.derive_seed(5, 0) != me.derive_seed(5, 1)
    shared = me.MultiEnvHarness(harness(2).sim, env_count=2, seeds=(5, 5))
    assert shared.state_digest(0) != shared.state_digest(1)


def test_hold_in_a_leaves_b_bit_identical():
    h = harness(2)
    drive(h, 3)
    before_b = digest_of(h, 1)
    before_a = digest_of(h, 0)

    h.hold(0)
    assert h.holding[0] is True and h.holding[1] is False
    for _ in range(3):
        assert h.step(0).code == "hold_active"     # refused, recorded, adapter untouched
    assert digest_of(h, 1) == before_b             # bit-identical: the whole point
    assert digest_of(h, 0) != before_a             # A did change (its latch/attempt counter)
    # B is still drivable while A is frozen
    assert h.step(1).accepted is True
    # releasing A does not disturb B either
    frozen_b = digest_of(h, 1)
    h.release(0)
    assert digest_of(h, 1) == frozen_b
    assert h.step(0).accepted is True


def test_reset_in_a_does_not_touch_b():
    h = harness(2)
    drive(h, 3)
    before_b = digest_of(h, 1)
    before_pose_b = h.sim.read_state(1)

    written = h.reset_to_rest(0)
    assert written["code"] == "accepted"
    assert tuple(h.sim.read_state(0)) == h.rest_pose(0) == h.rest_pose_values
    assert digest_of(h, 1) == before_b
    assert h.sim.read_state(1) == before_pose_b
    assert h.sim.reset_count(1) == 0              # B's row was never reset


def test_reset_while_holding_is_refused_and_contained():
    h = harness(2)
    drive(h, 2)
    h.hold(0)
    before_b = digest_of(h, 1)
    out = h.reset_to_rest(0)
    assert out["code"] == "hold_active"
    assert out["written"] is None
    assert h.sim.reset_count(0) == 0
    assert digest_of(h, 1) == before_b


def test_no_state_leak_across_resets_of_every_env():
    h = harness(4)
    drive(h, 3)
    poses_before = {env_id: h.sim.read_state(env_id) for env_id in h.env_ids}
    # reset envs one at a time; each reset must leave the OTHER envs bit-identical
    for env_id in (2, 0, 3, 1):
        others = {other: digest_of(h, other) for other in h.env_ids if other != env_id}
        h.reset_to_rest(env_id)
        for other, before in others.items():
            assert digest_of(h, other) == before, (env_id, other)
        assert h.sim.read_state(env_id) == h.rest_pose(env_id)
        assert h.sim.reset_count(env_id) == 1
    # each env was changed by exactly its own reset, and by nothing else
    assert h.sim.read_state(0) != poses_before[0]


# --------------------------------------------------------------------------- 2. determinism
def test_same_seeds_reproduce_the_same_states():
    a = harness(2, seed=101)
    b = harness(2, seed=101)
    for _ in range(6):
        for env_id in a.env_ids:
            assert a.step(env_id).values == b.step(env_id).values
    assert a.digests() == b.digests()
    # the digests are not trivially constant
    assert len(set(a.digests().values())) == 2


def test_different_seed_changes_only_that_env():
    a = me.build_mock_harness(env_count=2, seed=101)          # seeds (101, 102)
    b = me.MultiEnvHarness(a.sim.__class__(a.sim.joint_names, 2, a.rest_pose_values),
                           env_count=2, seeds=(101, 999))      # env 0 same seed, env 1 changed
    for _ in range(4):
        for env_id in a.env_ids:
            a.step(env_id)
            b.step(env_id)
    assert a.state_digest(0) == b.state_digest(0)            # env 0 identical
    assert a.state_digest(1) != b.state_digest(1)            # env 1 differs
    assert a.sim.read_state(0) == b.sim.read_state(0)


def test_rng_streams_are_per_env():
    h = harness(4, seed=17)
    draws = [[h.random_target(env_id) for _ in range(3)] for env_id in h.env_ids]
    assert len({tuple(values) for row in draws for values in row}) == 12  # no two draws coincide
    # drawing in env 0 does not advance anyone else's stream: after many draws on env 0, the
    # FIRST draw of every other env must still equal the first draw of a pristine harness
    pristine = harness(4, seed=17)
    first = {env_id: pristine.random_target(env_id) for env_id in pristine.env_ids}
    busy = harness(4, seed=17)
    for _ in range(len(draws[0]) * 5):
        busy.random_target(0)
    assert busy.random_target(1) == first[1]
    assert busy.random_target(2) == first[2]
    assert busy.random_target(3) == first[3]
    # and the env-0 stream itself is reproducible draw for draw
    again = harness(4, seed=17)
    assert [again.random_target(0) for _ in range(3)] == list(draws[0])


# --------------------------------------------------------------------------- 3. fault containment
@pytest.mark.parametrize("kind,expected_code", [
    (me.FAULT_NAN_TARGET, "non_finite"),
    (me.FAULT_OUT_OF_LIMIT, "position_limit"),
    (me.FAULT_UNKNOWN_JOINT, "unknown_joint"),
    (me.FAULT_MID_RUN_HOLD, "hold_active"),
])
def test_rejected_fault_is_contained_and_typed(kind, expected_code):
    h = harness(2, seed=23)
    drive(h, 2)
    before_b = digest_of(h, 1)
    writes_before = h.sim.write_count(0)

    h.inject(0, kind)
    result = h.step(0)
    assert result.accepted is False
    assert result.code == expected_code
    # the adapter was never touched for this env: the P5-05 gate refused first
    assert h.sim.write_count(0) == writes_before
    assert digest_of(h, 1) == before_b
    ledger = h.events(0)
    assert ledger and ledger[-1]["code"] == expected_code
    assert all(event["contained"] for event in h.faults(0))
    assert h.faults(1) == [] and h.rejections(1) == []
    # env 0 itself is still alive and drivable (except while HOLD is latched)
    if kind == me.FAULT_MID_RUN_HOLD:
        h.release(0)
    assert h.step(0).accepted is True


def test_adapter_never_receives_a_non_finite_or_out_of_limit_target():
    h = harness(2, seed=31)
    limits = {name: (limit.lower, limit.upper)
              for name, limit in h.controller(0).envelope.limits.items()}
    for kind in (me.FAULT_NAN_TARGET, me.FAULT_OUT_OF_LIMIT, me.FAULT_UNKNOWN_JOINT):
        h.inject(0, kind)
        assert h.step(0).accepted is False
    for env_id in h.env_ids:
        for write in h.sim.writes(env_id):
            for name, value in zip(RS_NAMES, write):
                assert math.isfinite(value), (env_id, name, value)
                lower, upper = limits[name]
                assert lower <= value <= upper, (env_id, name, value, lower, upper)
    assert h.sim.nonfinite_writes == []


def test_adapter_exception_is_contained_and_reported_as_divergence():
    h = harness(2, seed=41)
    drive(h, 2)
    before_b = digest_of(h, 1)
    before_a_pose = h.sim.read_state(0)
    writes_before = h.sim.write_count(0)

    h.inject(0, me.FAULT_ADAPTER_EXCEPTION)
    result = h.step(0)
    assert result.accepted is False and result.code == "adapter_exception"
    assert h.sim.write_count(0) == writes_before      # raised before mutating
    assert h.sim.read_state(0) == before_a_pose
    assert h.write_diverged(0) is True                # the gate latched a setpoint never applied
    assert digest_of(h, 1) == before_b
    resumed = h.resume(0)
    assert resumed["reconciled_divergence"] is True
    assert h.write_diverged(0) is False
    assert tuple(h.sim.read_state(0)) == tuple(h.controller(0).gate.last_valid)
    assert h.step(0).accepted is True                 # recovered


def test_timeout_is_contained_quarantines_and_recovers():
    h = harness(2, seed=51, write_timeout=0.05)
    drive(h, 1)
    before_b = digest_of(h, 1)

    h.inject(0, me.FAULT_TIMEOUT)
    result = h.step(0)
    assert result.accepted is False and result.code == "timeout"
    assert h.status(0) == me.STATUS_QUARANTINED
    assert digest_of(h, 1) == before_b
    event = h.faults(0)[-1]
    assert event["late_write_possible"] is True       # documented, not hidden
    # a quarantined env refuses new commands instead of quietly re-syncing
    with pytest.raises(me.EnvQuarantined):
        h.step(0)
    assert h.step(1).accepted is True                 # the other env is unaffected
    # the abandoned worker is joined before asserting what actually landed
    assert h.wait_for_pending(timeout=5.0) >= 1
    assert h.sim.write_count(0) >= 1                  # the late write did land: see the doc
    h.resume(0)
    assert h.status(0) == me.STATUS_OK
    assert tuple(h.sim.read_state(0)) == tuple(h.controller(0).gate.last_valid)
    assert h.step(0).accepted is True


def test_unknown_fault_kind_and_unknown_env_are_rejected():
    h = harness(2)
    with pytest.raises(me.UnknownFault):
        h.inject(0, "melt_the_arm")
    with pytest.raises(me.UnknownEnv):
        h.step(7)
    with pytest.raises(me.UnknownEnv):
        h.reset_to_rest(7)
    with pytest.raises(me.UnknownEnv):
        h.state_digest(7)
    assert h.env_ids == (0, 1)


# --------------------------------------------------------------------------- 4. restart / resume
def test_restart_and_resume_is_bit_identical_and_isolated():
    h = harness(2, seed=61)
    drive(h, 4)
    h.hold(0)                                   # restart a FROZEN env too, not just a moving one
    before_a, before_b = digest_of(h, 0), digest_of(h, 1)

    event = h.inject(0, me.FAULT_RESTART_RESUME)
    assert event["code"] == "restart_resume"
    assert h.restarts(0) == 1
    assert h.holding[0] is True                 # the HOLD latch survives the restart
    assert digest_of(h, 0) == before_a          # bit-identical resume
    assert digest_of(h, 1) == before_b          # and nothing leaked into env 1
    h.release(0)
    assert h.step(0).accepted is True
    assert h.step(1).accepted is True


def test_restart_does_not_reset_other_envs_counters():
    h = harness(4, seed=71)
    drive(h, 3)
    digests = {env_id: digest_of(h, env_id) for env_id in h.env_ids}
    h.inject(2, me.FAULT_RESTART_RESUME)
    for env_id in h.env_ids:
        if env_id == 2:
            continue
        assert digest_of(h, env_id) == digests[env_id]
    assert h.restarts(2) == 1
    assert [h.restarts(env_id) for env_id in (0, 1, 3)] == [0, 0, 0]


# --------------------------------------------------------------------------- 5. rest pose guard
def test_rest_pose_used_is_inside_the_recorded_usd_limits():
    limits = usd_limits()
    h = harness(3, seed=5)
    rest = h.rest_pose_values
    assert len(rest) == len(RS_NAMES) == len(h.joint_names)
    for name, value in zip(h.joint_names, rest):
        lower, upper, unit = limits[name]
        assert math.isfinite(value)
        assert lower <= value <= upper, (name, value, lower, upper)
        # strictly inside, by a visible margin: a rest pose on a hard stop is not a rest pose
        assert min(value - lower, upper - value) > 0.0, (name, value, lower, upper)
    # every env uses the same model rest pose, and it is the RS one
    for env_id in h.env_ids:
        assert h.rest_pose(env_id) == rest
    assert rest[h.joint_names.index("joint3")] == pytest.approx(1.0)
    assert rest[h.joint_names.index("joint2")] == pytest.approx(0.3)
    # the model table itself agrees with the pxr export to a sane tolerance
    for name, limit in rest_pose.limits_for(rest_pose.MODEL_REBOT_RS).items():
        lower, upper, _unit = limits[name]
        assert abs(limit.lower - lower) < 1e-6
        assert abs(limit.upper - upper) < 1e-6


def test_negative_control_the_piper_rest_pose_is_rejected_on_this_arm():
    """The bug P5B fixed, kept as a live control: -1.5 rad on joint3 must not be writable."""
    h = harness(1, seed=5)
    # the Stage-0 pose, completed with the two finger values the RS arm also has
    legacy = dict(LEGACY_PIPER_REST)
    legacy.update({name: h.rest_pose(0)[h.joint_names.index(name)]
                   for name in ("gripper_joint1", "gripper_joint2")})
    with pytest.raises(rest_pose.RestPoseLimitViolation):
        rest_pose.validate_rest_pose(rest_pose.MODEL_REBOT_RS, legacy)
    limits = usd_limits()
    assert not (limits["joint3"][0] <= legacy["joint3"] <= limits["joint3"][1])
    # and the harness refuses to reset an env to it, before the adapter is touched
    with pytest.raises(rest_pose.RestPoseLimitViolation):
        h.reset(0, legacy)
    assert h.sim.reset_count(0) == 0
    # the valid rest pose, by contrast, is accepted
    assert h.reset_to_rest(0)["code"] == "accepted"
    assert h.sim.reset_count(0) == 1


# --------------------------------------------------------------------------- 6. N=4 batch
def test_n4_batch_keeps_four_distinct_states_under_mixed_faults():
    h = harness(4, seed=81)
    drive(h, 3)
    assert len(set(h.digests().values())) == 4

    before = {env_id: digest_of(h, env_id) for env_id in h.env_ids}
    h.inject(0, me.FAULT_NAN_TARGET)
    assert h.step(0).code == "non_finite"
    h.hold(3)
    assert h.step(3).code == "hold_active"
    h.inject(1, me.FAULT_OUT_OF_LIMIT)
    assert h.step(1).code == "position_limit"

    assert digest_of(h, 2) == before[2]                    # the untouched env is bit-identical
    for env_id in (0, 1, 3):
        assert digest_of(h, env_id) != before[env_id]
        assert len(h.events(env_id)) >= 1
    assert h.events(2) == []
    # contained faults do not spread: only env 2 has an empty ledger, and each ledger is its own
    assert {event["env_id"] for event in h.events(0) + h.events(1) + h.events(3)} <= {0, 1, 3}
    # every env is still individually drivable after the batch-wide fault mix
    h.release(3)
    for env_id in h.env_ids:
        assert h.step(env_id).accepted is True


def test_n4_seeds_are_reproducible_as_a_batch():
    a = harness(4, seed=91)
    b = harness(4, seed=91)
    drive(a, 3)
    drive(b, 3)
    assert a.digests() == b.digests()
    assert a.describe()["rest_pose"] == b.describe()["rest_pose"]
    assert len(set(a.digests().values())) == 4


# --------------------------------------------------------------------------- 7. measurement
def test_offline_mock_measurement_is_labelled_and_not_isaac():
    result = me.measure_offline_mock_throughput(env_count=2, steps=25, seed=3)
    assert result["label"] == me.OFFLINE_LABEL
    assert "OFFLINE-MOCK" in result["label"]
    assert "not Isaac" in result["label"]
    assert result["env_count"] == 2 and result["env_steps"] == 50
    assert result["accepted_env_steps"] == 50
    assert result["wall_seconds"] > 0
    assert result["env_steps_per_second"] > 0
    assert isinstance(result["peak_python_bytes"], int) and result["peak_python_bytes"] > 0
    assert result["isaac_throughput"].startswith("UNKNOWN")
    assert result["gpu_memory"].startswith("UNKNOWN")
    # no environment-count claim anywhere in the payload
    text = json.dumps(result)
    for forbidden in ("128 env", "64 env", "16 env", "capable of 128", "real-time"):
        assert forbidden not in text
