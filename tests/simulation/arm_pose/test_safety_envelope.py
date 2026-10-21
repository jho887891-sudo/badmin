# -*- coding: utf-8 -*-
"""P5-05 - the safety envelope and the HOLD latch: every rejection path, typed and named.

Each rejection must (a) happen, (b) carry the OFFENDING JOINT and VALUE as attributes, and (c) not
be a clamp: a clamped command is indistinguishable from a legitimate one downstream, which is worse
than a refused command. HOLD must freeze the accepted target bit-identically - no recomputation, no
copy-then-drift - and refuse new commands until it is released.

Pure python + the model table: no Isaac, no torch, no numpy.
"""
from __future__ import annotations

import struct

import pytest

from pose_data import REBOT_RS_JOINT_NAMES as NAMES

#: The B601-RS rest pose (P5B), used as a legal starting point for the rate checks.
REST = (0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.025, 0.03575)
DT = 1.0 / 60.0


def _bits(value):
    return struct.pack("<d", float(value))


@pytest.fixture
def safety():
    import badminton_scene.safety as module

    return module


@pytest.fixture
def gate(safety):
    return safety.SafetyGate("rebot_b601_rs", NAMES)


# --------------------------------------------------------------------------- 1. position limits
def test_the_envelope_position_limits_are_the_model_table(safety):
    from badminton_scene.rest_pose import limits_for

    envelope = safety.envelope_for("rebot_b601_rs")
    table = limits_for("rebot_b601_rs")
    assert set(envelope.limits) == set(table) == set(NAMES)
    for name, limit in table.items():
        assert envelope.limits[name].lower == limit.lower
        assert envelope.limits[name].upper == limit.upper
        assert envelope.velocity[name] > 0.0 and envelope.acceleration[name] > 0.0
    # The caps are declared as unverified policy caps, not measurements.
    assert "POLICY CAP" in envelope.source and "unverified" in envelope.source


def test_a_target_outside_a_joint_limit_raises_with_the_joint_and_the_value(gate, safety):
    bad = list(REST)
    bad[2] = -1.5  # joint3 on the RS is [0, 3.14]: the exact P5 F1 defect
    with pytest.raises(safety.PositionLimitViolation) as excinfo:
        gate.accept(tuple(bad))
    error = excinfo.value
    assert error.joint == "joint3"
    assert error.value == -1.5
    assert error.limit == (0.0, 3.14)
    assert error.index == 2
    assert error.code == "position_limit"
    assert gate.last_valid is None, "a rejected target must not become the baseline"
    assert gate.rejected_count == 1


def test_the_limits_are_inclusive_so_a_deliberate_stop_is_allowed(gate):
    """As in reset.py: commanding the exact stop is a request, not a violation."""
    at_lower = list(REST)
    at_lower[3] = -1.57   # joint4 lower stop
    at_lower[6] = 0.0     # finger at its lower stop
    gate.accept(tuple(at_lower))
    assert gate.last_valid[3] == -1.57 and gate.last_valid[6] == 0.0


def test_each_joint_is_checked_against_its_own_limits(gate, safety):
    """A value legal for joint1 is illegal for joint5: the check must not be positional mush."""
    one = list(REST)
    one[0] = 2.0      # inside joint1 [-2.8, 2.8]
    gate.accept(tuple(one))
    five = list(REST)
    five[4] = 2.0     # outside joint5 [-1.57, 1.57]
    with pytest.raises(safety.PositionLimitViolation) as excinfo:
        gate.accept(tuple(five))
    assert excinfo.value.joint == "joint5" and excinfo.value.value == 2.0


# --------------------------------------------------------------------------- 2. non-finite
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_raises_before_any_range_check(gate, safety, poison):
    bad = list(REST)
    bad[4] = poison
    with pytest.raises(safety.NonFiniteCommand) as excinfo:
        gate.accept(tuple(bad))
    error = excinfo.value
    assert error.joint == "joint5"
    # NaN != NaN, so compare the payload's own representation rather than its value.
    assert repr(error.value) == repr(poison), (error.value, poison)
    # NaN would pass a naive range check (NaN < lo is False), so the code must be its own type.
    assert error.code == "non_finite"


def test_a_nan_target_in_a_named_command_is_rejected_too(gate, safety):
    named = {name: value for name, value in zip(NAMES, REST)}
    named["joint2"] = float("nan")
    with pytest.raises(safety.NonFiniteCommand) as excinfo:
        gate.validate_named(named)
    assert excinfo.value.joint == "joint2"


# --------------------------------------------------------------------------- 3. width / names
def test_a_target_with_the_wrong_width_raises(gate, safety):
    with pytest.raises(safety.CommandWidthError) as excinfo:
        gate.accept(REST[:-1])
    assert excinfo.value.value == len(REST) - 1
    with pytest.raises(safety.CommandWidthError):
        gate.accept(())


def test_none_is_not_a_command(gate, safety):
    with pytest.raises(safety.CommandWidthError):
        gate.accept(None)


def test_an_unknown_name_in_a_named_command_raises(gate, safety):
    named = {name: value for name, value in zip(NAMES, REST)}
    named["joint9"] = 0.0
    with pytest.raises(safety.UnknownCommandJoint) as excinfo:
        gate.validate_named(named)
    assert "joint9" in str(excinfo.value) and excinfo.value.joint == "joint9"


def test_a_missing_name_in_a_named_command_raises(gate, safety):
    named = {name: value for name, value in zip(NAMES, REST)}
    named.pop("joint6")
    with pytest.raises(safety.UnknownCommandJoint) as excinfo:
        gate.validate_named(named)
    assert excinfo.value.joint == "joint6"


# --------------------------------------------------------------------------- 4. rate limits
def test_a_step_faster_than_the_velocity_cap_is_rejected(gate, safety):
    gate.accept(REST)
    jumped = list(REST)
    jumped[0] = 1.0  # 1.0 rad in 1/60 s = 60 rad/s, cap is 2 rad/s
    with pytest.raises(safety.VelocityLimitViolation) as excinfo:
        gate.accept(tuple(jumped))
    error = excinfo.value
    assert error.joint == "joint1" and error.value == 1.0 and error.limit == 2.0
    assert gate.last_valid == REST, "the rejected step must not replace the last valid target"


def test_a_small_step_inside_the_velocity_cap_is_accepted(gate):
    gate.accept(REST)
    step = list(REST)
    step[0] = REST[0] + 2.0 * DT * 0.5  # half the cap
    gate.accept(tuple(step))
    assert gate.last_valid[0] == pytest.approx(step[0])


def test_a_step_faster_than_the_acceleration_cap_is_rejected(safety):
    """Three points, so a second difference exists; the velocity of the last step is legal."""
    gate = safety.SafetyGate("rebot_b601_rs", NAMES)
    gate.accept(REST)
    gate.accept(REST)
    kicked = list(REST)
    kicked[0] = 0.01  # 0.6 rad/s (legal) but 36 rad/s^2 (cap 20)
    with pytest.raises(safety.AccelerationLimitViolation) as excinfo:
        gate.accept(tuple(kicked))
    assert excinfo.value.joint == "joint1" and excinfo.value.limit == 20.0
    assert gate.last_valid == REST


def test_the_first_command_has_no_acceleration_to_check(gate):
    """Acceleration is undefined for a single point, not zero - and not a reason to reject."""
    gate.accept(REST)
    assert gate.last_valid == REST


# --------------------------------------------------------------------------- 5. HOLD
def test_hold_freezes_the_previous_target_bit_identically(gate, safety):
    accepted = gate.accept(REST)
    held = gate.hold()
    assert held is accepted, "hold() must freeze the same object, not a copy of the numbers"
    assert gate.held_target is held
    for got, expected in zip(held, REST):
        assert _bits(got) == _bits(expected), (held, REST)
    assert gate.holding is True


def test_hold_refuses_new_commands_until_released(gate, safety):
    gate.accept(REST)
    gate.hold()
    nudged = list(REST)
    nudged[0] = REST[0] + 1e-9  # well inside every limit and every cap: still refused
    with pytest.raises(safety.HoldActive) as excinfo:
        gate.accept(tuple(nudged))
    assert excinfo.value.code == "hold_active"
    assert gate.last_valid == REST
    assert gate.rejected_count == 1
    gate.release()
    assert gate.holding is False
    gate.accept(tuple(nudged))
    assert gate.last_valid[0] == pytest.approx(nudged[0])


def test_hold_before_any_accepted_target_raises(gate, safety):
    with pytest.raises(safety.HoldWithoutTarget) as excinfo:
        gate.hold()
    assert excinfo.value.code == "hold_without_target"


def test_holding_twice_raises(gate, safety):
    gate.accept(REST)
    gate.hold()
    with pytest.raises(safety.HoldActive):
        gate.hold()


def test_release_without_a_hold_raises(gate, safety):
    with pytest.raises(safety.NotHolding) as excinfo:
        gate.release()
    assert excinfo.value.code == "not_holding"


def test_a_rejection_carries_a_structured_form_for_logging(gate, safety):
    bad = list(REST)
    bad[2] = 9.0
    with pytest.raises(safety.PositionLimitViolation) as excinfo:
        gate.accept(tuple(bad))
    event = excinfo.value.as_dict()
    assert event["code"] == "position_limit"
    assert event["joint"] == "joint3" and event["value"] == 9.0
    assert event["index"] == 2
    assert "joint3" in event["message"]
