# -*- coding: utf-8 -*-
"""P8-01: action schema -- strict validation, rate limits and the SIMULATION-ONLY refusal.

Every assertion here runs on a CPU with numpy only.  Nothing in this file can command a
robot: the finest-grained guarantee is that SimulatedCommand.to_real_robot() raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.action_schema import (  # noqa: E402
    ACTION_SCHEMA_VERSION, DEFAULT_LIMITS, OPTIONAL_ACTION_NAMES, REAL_ROBOT_FORBIDDEN_REASON,
    REQUIRED_ACTION_NAMES, SIMULATION_ONLY, TARGET, ActionBoundsError, ActionContextError,
    ActionDTypeError, ActionNonFiniteError, ActionRateLimitError, ActionSchemaError,
    ActionShapeError, ActionUnitError, ActionValue, MissingActionFieldError,
    RealRobotActionForbidden, SimulatedCommand, UnknownActionFieldError, action_dim,
    action_fields, schema_rows, to_command, validate as validate_action,
)
from badminton_brain.status import AssetStatus  # noqa: E402

Q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])


def action(base_lin=(0.4, -0.3), wz=0.5, dq=0.02, rp=None, ro=None):
    out = {
        "base_linear_velocity": ActionValue(np.asarray(base_lin, dtype=float), "m/s"),
        "base_angular_velocity": ActionValue(np.asarray([wz], dtype=float), "rad/s"),
        "joint_position_delta": ActionValue(np.full(6, float(dq)), "rad"),
    }
    if rp is not None:
        out["racket_target_position"] = ActionValue(np.asarray(rp, dtype=float), "m")
    if ro is not None:
        out["racket_target_orientation"] = ActionValue(np.asarray(ro, dtype=float), "1")
    return out


def test_action_dimension_matches_the_documented_placeholder():
    assert action_dim() == 9
    assert action_dim(include_optional=True) == 16
    assert REQUIRED_ACTION_NAMES == ("base_linear_velocity", "base_angular_velocity",
                                     "joint_position_delta")
    assert OPTIONAL_ACTION_NAMES == ("racket_target_position", "racket_target_orientation")
    assert ACTION_SCHEMA_VERSION == "p8-act-v0.1-candidate"


def test_to_command_roundtrip_shape_dtype_and_targets():
    cmd = to_command(action(), joint_position=Q, dt_s=0.02, now_s=3.0)
    assert isinstance(cmd, SimulatedCommand)
    assert cmd.base_twist_body.shape == (3,) and cmd.base_twist_body.dtype == np.float64
    np.testing.assert_allclose(cmd.base_twist_body, [0.4, -0.3, 0.5])
    assert cmd.joint_position_target.shape == (6,)
    np.testing.assert_allclose(cmd.joint_position_target, Q + 0.02)
    assert cmd.simulation_only is True and cmd.target == TARGET == "SIMULATOR_ONLY"
    assert cmd.previous_action_checked is False
    assert cmd.dt_s == pytest.approx(0.02)


def test_optional_racket_target_roundtrip():
    cmd = to_command(action(rp=(-1.0, 0.2, 1.1), ro=(1.0, 0.0, 0.0, 0.0)),
                     joint_position=Q, dt_s=0.02)
    np.testing.assert_allclose(cmd.racket_target_position, [-1.0, 0.2, 1.1])
    np.testing.assert_allclose(cmd.racket_target_orientation, [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ActionSchemaError):
        to_command(action(rp=(-1.0, 0.2, 1.1)), joint_position=Q, dt_s=0.02)


def test_simulation_only_is_enforced_on_the_command_itself():
    cmd = to_command(action(), joint_position=Q, dt_s=0.02)
    cmd.assert_simulation_only()
    with pytest.raises(RealRobotActionForbidden) as ei:
        cmd.to_real_robot()
    assert "SIMULATION-ONLY" in str(ei.value)
    assert "NOT AUTHORISED" in REAL_ROBOT_FORBIDDEN_REASON or "not authorised" in (
        REAL_ROBOT_FORBIDDEN_REASON.lower())
    with pytest.raises(RealRobotActionForbidden):
        SimulatedCommand(base_twist_body=np.zeros(3), joint_position_target=Q, dt_s=0.02,
                         simulation_only=False)
    with pytest.raises(RealRobotActionForbidden):
        SimulatedCommand(base_twist_body=np.zeros(3), joint_position_target=Q, dt_s=0.02,
                         target="REAL_ROBOT")
    assert SIMULATION_ONLY is True


def test_limits_all_declare_their_provenance_as_temp_proxies():
    rows = schema_rows()
    assert len(rows) == len(action_fields()) == 5
    for row in rows:
        assert row["unit"] and row["source"] and row["description"], row["name"]
    for name in ("base_linear_velocity_max", "base_angular_velocity_max", "joint_velocity_max",
                 "base_linear_accel_max", "base_angular_accel_max", "joint_accel_max",
                 "control_dt_max", "joint_position_min", "joint_position_max",
                 "racket_workspace_min", "racket_workspace_max"):
        param = getattr(DEFAULT_LIMITS, name)
        assert param.status is AssetStatus.TEMP_PARAMETERIZED_PROXY, name
        assert param.source.strip(), name


# ------------------------------------------------------------------------ rejection paths
def test_missing_component_is_rejected():
    bad = action()
    del bad["joint_position_delta"]
    with pytest.raises(MissingActionFieldError):
        validate_action(bad)


def test_unknown_component_is_rejected():
    bad = action()
    bad["base_linear_acceleration"] = ActionValue(np.zeros(2), "m/s^2")
    with pytest.raises(UnknownActionFieldError):
        validate_action(bad)


def test_wrong_shape_is_rejected():
    bad = action()
    bad["base_linear_velocity"] = ActionValue(np.zeros(3), "m/s")
    with pytest.raises(ActionShapeError):
        validate_action(bad)


@pytest.mark.parametrize("bad", [np.array(["0", "0"]), np.array([1 + 1j, 0.0]),
                                 np.array([True, False])])
def test_wrong_dtype_is_rejected(bad):
    values = action()
    values["base_linear_velocity"] = ActionValue(bad, "m/s")
    with pytest.raises(ActionDTypeError):
        validate_action(values)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nan_and_inf_are_rejected(bad):
    values = action(base_lin=(0.1, bad))
    with pytest.raises(ActionNonFiniteError):
        validate_action(values)


def test_out_of_bounds_is_rejected_not_clamped():
    with pytest.raises(ActionBoundsError) as ei:
        validate_action(action(base_lin=(0.9, 0.0)))
    assert "not clamped" in str(ei.value)
    with pytest.raises(ActionBoundsError):
        validate_action(action(dq=0.05))          # > 2.0 rad/s * 0.02 s
    with pytest.raises(ActionBoundsError):
        validate_action(action(wz=1.5))           # > 1.0 rad/s proxy


def test_action_bounds_scale_with_dt():
    # 0.05 rad in one step is legal at dt = 0.05 s (<= 2.0 rad/s * 0.05) ...
    validate_action(action(dq=0.05), dt_s=0.05)
    # ... but not at dt = 0.02 s
    with pytest.raises(ActionBoundsError):
        validate_action(action(dq=0.05), dt_s=0.02)


@pytest.mark.parametrize("channel,unit", [
    ("joint_position_delta", "deg"),
    ("base_linear_velocity", "km/h"),
    ("racket_target_position", "mm"),
])
def test_unit_confusion_is_rejected(channel, unit):
    values = action(rp=(-1.0, 0.2, 1.1), ro=(1.0, 0.0, 0.0, 0.0))
    values[channel] = ActionValue(values[channel].value, unit)
    with pytest.raises(ActionUnitError):
        validate_action(values)


def test_bare_array_without_unit_is_rejected():
    values = action()
    values["joint_position_delta"] = np.zeros(6)
    with pytest.raises(ActionSchemaError) as ei:
        validate_action(values)
    assert "ActionValue" in str(ei.value)


def test_rate_limit_rejects_an_in_bounds_jump():
    previous = action(base_lin=(0.0, 0.0), wz=0.0, dq=0.0)
    # 0.05 m/s is inside the absolute envelope but exceeds 0.5 m/s^2 * 0.02 s = 0.01 m/s
    with pytest.raises(ActionRateLimitError) as ei:
        to_command(action(base_lin=(0.05, 0.0), wz=0.0, dq=0.0), joint_position=Q,
                   previous_action=previous, dt_s=0.02)
    assert "rate limit" in str(ei.value)

    with pytest.raises(ActionRateLimitError):
        to_command(action(base_lin=(0.0, 0.0), wz=0.0, dq=-0.04), joint_position=Q,
                   previous_action=action(base_lin=(0.0, 0.0), wz=0.0, dq=0.04), dt_s=0.02)


def test_rate_limit_accepts_a_small_step_and_reports_it_was_checked():
    previous = action(base_lin=(0.0, 0.0), wz=0.0, dq=0.0)
    cmd = to_command(action(base_lin=(0.004, -0.003), wz=0.005, dq=0.001), joint_position=Q,
                     previous_action=previous, dt_s=0.02)
    assert cmd.previous_action_checked is True
    np.testing.assert_allclose(cmd.base_twist_body, [0.004, -0.003, 0.005])


def test_context_errors_are_loud():
    with pytest.raises(ActionContextError):
        to_command(action(), joint_position=np.zeros(5), dt_s=0.02)
    with pytest.raises(ActionBoundsError):
        to_command(action(), joint_position=np.full(6, 3.0), dt_s=0.02)   # > 2.6 rad proxy
    with pytest.raises(ActionContextError):
        to_command(action(), joint_position=Q, dt_s=0.0)
    with pytest.raises(ActionContextError):
        to_command(action(), joint_position=Q, dt_s=0.5)                  # > control_dt_max
    with pytest.raises(ActionNonFiniteError):
        to_command(action(), joint_position=np.array([np.nan, 0, 0, 0, 0, 0]), dt_s=0.02)
