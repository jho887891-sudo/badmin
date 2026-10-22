"""P4-11: real tests for scripts/simulation/p4_command_interface.py.

Everything asserted here runs against the real module and the REAL committed GPU evidence under
outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/.  Nothing is mocked and no result.json is
synthesised: if an evidence directory disappears these tests FAIL (they must never silently skip),
because the whole point of the interface is that it reproduces what the GPU actually ran.

Run:  python -m pytest tests/simulation/test_p4_command_interface.py -q
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import textwrap
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "simulation"
EVIDENCE = REPO_ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "morph_one" / "p2_wc_gates"
GPU_SCRIPT = SCRIPTS / "p4_proxy_wheel_drive.py"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import p4_command_interface as ci  # noqa: E402

R = ci.EFFECTIVE_ROLLING_RADIUS_M          # 0.101459 m, the GPU feed-forward radius
R_GEOM = ci.WHEEL_RADIUS_GEOM_M            # 0.103 m, the P1 v2 torus equator


# ---------------------------------------------------------------------------------------------
# Real-evidence helpers (no mocks)
# ---------------------------------------------------------------------------------------------
def result_of(run_id):
    path = EVIDENCE / run_id / "result.json"
    assert path.is_file(), "missing committed GPU evidence: %s" % path
    return json.loads(path.read_text(encoding="utf-8"))


def frames_of(run_id):
    path = EVIDENCE / run_id / "raw_poses.jsonl"
    assert path.is_file(), "missing committed GPU frames: %s" % path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, "empty frame log: %s" % path
    return [json.loads(line) for line in lines]


def manifest_of(run_id):
    path = EVIDENCE / run_id / "manifest.json"
    assert path.is_file(), "missing committed GPU manifest: %s" % path
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_runs():
    assert EVIDENCE.is_dir(), "GPU evidence directory is gone: %s" % EVIDENCE
    return sorted(p.name for p in EVIDENCE.iterdir() if (p / "result.json").is_file())


# ---------------------------------------------------------------------------------------------
# 1. Sign / axis convention
# ---------------------------------------------------------------------------------------------
def test_positive_speed_needs_positive_omega_on_all_four_and_travels_minus_y():
    command = ci.WheelCommand.from_speed(0.30, R, 0.50, "clamp")
    # straight roll: the SAME sign on all four wheels (the axis fact that P4p_fwd_r02 proved)
    assert len(set(command.wheel_targets_rad_s)) == 1
    assert command.wheel_targets_rad_s == pytest.approx((0.30 / R,) * 4, abs=1e-12)
    assert all(target > 0.0 for target in command.wheel_targets_rad_s)
    # +omega about world +X pushes the chassis along -Y
    assert command.chassis_velocity_m_s() == pytest.approx((0.0, -0.30, 0.0), abs=1e-12)
    assert command.travel_direction == "-Y"
    assert command.audit()["convention"]["roll_axis"].startswith("world Y")


@pytest.mark.parametrize("speed,direction,sign", [(0.30, "-Y", 1.0), (-0.30, "+Y", -1.0)])
def test_red_green_flipping_the_speed_sign_inverts_travel_direction(speed, direction, sign):
    """RED/GREEN: the same call with the sign flipped must invert BOTH targets and travel."""
    command = ci.WheelCommand.from_speed(speed, R, 0.50, "clamp")
    assert command.travel_direction == direction
    assert command.chassis_velocity_m_s()[1] == pytest.approx(-speed, abs=1e-12)
    for target in command.wheel_targets_rad_s:
        assert math.copysign(1.0, target) == sign
        assert abs(target) == pytest.approx(abs(speed) / R, abs=1e-12)


def test_p1_outward_normal_pattern_is_a_motor_channel_not_a_joint_pattern():
    """The +1/+1/-1/-1 outward pattern turns the chassis; it is NOT a straight roll.

    Applying the P1 v2 motor-channel signs to a straight-roll target set yields a mixed-sign
    vector, i.e. a yaw command with psi_dot = -v / half_track.  This is the exact failure that
    made P4p_fwd_r02 yaw 1.915 rad instead of translating.
    """
    speed = 0.30
    straight = ci.WheelCommand.from_speed(speed, R, 0.50, "clamp")
    outward_signs = (1.0, 1.0, -1.0, -1.0)
    mixed = tuple(sign * target for sign, target in zip(outward_signs,
                                                       straight.wheel_targets_rad_s))
    assert len(set(math.copysign(1.0, v) for v in mixed)) == 2, "outward pattern must be mixed"
    as_yaw = ci.WheelCommand.from_yaw_rate(-speed / ci.HALF_TRACK_M, R, 5.0, "clamp")
    assert mixed == pytest.approx(as_yaw.wheel_targets_rad_s, abs=1e-12)
    assert as_yaw.yaw_direction == "-Z"


@pytest.mark.parametrize("run_id,direction,expected_sign", [
    ("P4p_rollneg_r02", "-Y", -1.0),
    ("P4p_rollpos_r02", "+Y", 1.0),
])
def test_gpu_roll_direction_matches_the_interface(run_id, direction, expected_sign):
    """Real GPU run: the recorded wheel targets and chassis dY must obey dY = -omega * r."""
    result = result_of(run_id)
    targets = result["metrics"]["wheel_targets_rad_s_per_wheel"]
    displacement_y = result["metrics"]["chassis_displacement_m"][1]
    assert len(targets) == ci.WHEEL_COUNT
    assert math.copysign(1.0, displacement_y) == expected_sign
    # the interface's prediction for the same targets
    command = ci.WheelCommand.from_speed(targets[0] * R_GEOM, R_GEOM, 5.0, "clamp")
    assert command.wheel_targets_rad_s == pytest.approx(tuple(targets), abs=1e-6)
    assert command.travel_direction == direction
    assert math.copysign(1.0, command.chassis_velocity_m_s()[1]) == expected_sign


def test_gpu_turn_targets_are_rederived_from_a_yaw_rate_within_1e_6():
    """Real GPU run P4p_turn_r03: [+3,+3,-3,-3] came from psi_dot < 0 and yaw went negative."""
    result = result_of("P4p_turn_r03")
    targets = result["metrics"]["wheel_targets_rad_s_per_wheel"]
    frames = frames_of("P4p_turn_r03")
    measured_yaw = frames[-1]["chassis_yaw_rad"] - frames[0]["chassis_yaw_rad"]
    assert measured_yaw < 0.0, "P4p_turn_r03 measured a negative yaw change"

    psi_dot = -targets[0] * R_GEOM / ci.HALF_TRACK_M
    command = ci.WheelCommand.from_yaw_rate(psi_dot, R_GEOM, 5.0, "clamp")
    assert command.yaw_direction == "-Z"
    assert command.yaw_rate_rad_s < 0.0
    assert command.wheel_targets_rad_s == pytest.approx(tuple(targets), abs=1e-6)
    # the x > 0 wheels (wheel1/wheel2, axles at +0.250002 m) spin positively for psi_dot < 0
    assert command.wheel_targets_rad_s[0] > 0.0 and command.wheel_targets_rad_s[1] > 0.0
    assert command.wheel_targets_rad_s[2] < 0.0 and command.wheel_targets_rad_s[3] < 0.0
    # documented UNKNOWN: the achieved yaw rate is far below the kinematic command (scrub)
    achieved = measured_yaw / ((len(frames) - 1) / 120.0)
    assert abs(achieved) < abs(psi_dot)


# ---------------------------------------------------------------------------------------------
# 2. clamp vs reject semantics
# ---------------------------------------------------------------------------------------------
def test_clamp_pins_the_magnitude_and_keeps_the_request():
    command = ci.WheelCommand.from_speed(0.80, R, 0.50, "clamp")
    assert command.clamped is True
    assert command.speed_m_s == pytest.approx(0.50, abs=1e-12)
    assert command.requested_speed_m_s == pytest.approx(0.80, abs=1e-12)
    assert command.wheel_targets_rad_s == pytest.approx((0.50 / R,) * 4, abs=1e-12)
    assert command.audit()["speed_requested_m_s"] == pytest.approx(0.80)


def test_clamp_preserves_the_sign_of_an_over_limit_reverse_command():
    command = ci.WheelCommand.from_speed(-0.80, R, 0.50, "clamp")
    assert command.clamped is True
    assert command.speed_m_s == pytest.approx(-0.50)
    assert command.travel_direction == "+Y"
    assert all(target < 0.0 for target in command.wheel_targets_rad_s)


def test_reject_raises_before_any_target_exists():
    with pytest.raises(ci.SpeedLimitExceededError) as excinfo:
        ci.WheelCommand.from_speed(0.80, R, 0.50, "reject")
    assert excinfo.value.code == "speed_limit_exceeded"
    assert "0.8" in str(excinfo.value) and "0.5" in str(excinfo.value)
    assert isinstance(excinfo.value, ci.CommandValidationError)
    assert not isinstance(excinfo.value, ci.NotFiniteError)


def test_reject_allows_a_command_exactly_at_the_limit():
    command = ci.WheelCommand.from_speed(0.50, R, 0.50, "reject")
    assert command.clamped is False
    assert command.speed_m_s == pytest.approx(0.50)


def test_audit_records_which_mode_was_used():
    assert ci.WheelCommand.from_speed(0.30, R, 0.50, "clamp").audit()["limit_mode"] == "clamp"
    assert ci.WheelCommand.from_speed(0.30, R, 0.50, "reject").audit()["limit_mode"] == "reject"


# ---------------------------------------------------------------------------------------------
# 3. Every rejection path (fail closed, typed, no silent fix)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_speed", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_speed_is_rejected(bad_speed):
    with pytest.raises(ci.NotFiniteError) as excinfo:
        ci.WheelCommand.from_speed(bad_speed, R, 0.50, "clamp")
    assert excinfo.value.code == "not_finite"


@pytest.mark.parametrize("bad_yaw", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_yaw_rate_is_rejected(bad_yaw):
    with pytest.raises(ci.NotFiniteError) as excinfo:
        ci.WheelCommand.from_yaw_rate(bad_yaw, R, 5.0, "clamp")
    assert excinfo.value.code == "not_finite"


@pytest.mark.parametrize("bad_radius", [0.0, -0.101459, -1.0])
def test_non_positive_radius_is_rejected(bad_radius):
    with pytest.raises(ci.InvalidRadiusError) as excinfo:
        ci.WheelCommand.from_speed(0.30, bad_radius, 0.50, "clamp")
    assert excinfo.value.code == "invalid_radius"


@pytest.mark.parametrize("bad_limit", [0.0, -0.50])
def test_zero_or_negative_limit_is_rejected(bad_limit):
    with pytest.raises(ci.InvalidLimitError) as excinfo:
        ci.WheelCommand.from_speed(0.30, R, bad_limit, "clamp")
    assert excinfo.value.code == "invalid_limit"


@pytest.mark.parametrize("bad_limit", [float("nan"), float("inf")])
def test_non_finite_limit_is_rejected(bad_limit):
    with pytest.raises(ci.InvalidLimitError) as excinfo:
        ci.WheelCommand.from_speed(0.30, R, bad_limit, "reject")
    assert excinfo.value.code == "invalid_limit"


@pytest.mark.parametrize("bad_mode", ["", "CLAMP", "ignore", "saturate", None])
def test_unknown_limit_mode_is_rejected(bad_mode):
    with pytest.raises(ci.InvalidLimitModeError) as excinfo:
        ci.WheelCommand.from_speed(0.30, R, 0.50, bad_mode)
    assert excinfo.value.code == "invalid_limit_mode"


def test_yaw_limit_mode_reject_raises():
    with pytest.raises(ci.YawLimitExceededError) as excinfo:
        ci.WheelCommand.from_yaw_rate(-3.0, R, 2.0, "reject")
    assert excinfo.value.code == "yaw_limit_exceeded"


def test_yaw_limit_mode_clamp_pins_the_yaw_rate():
    command = ci.WheelCommand.from_yaw_rate(-3.0, R, 2.0, "clamp")
    assert command.clamped is True
    assert command.yaw_rate_rad_s == pytest.approx(-2.0)
    assert command.requested_yaw_rate_rad_s == pytest.approx(-3.0)


@pytest.mark.parametrize("targets", [(1.0, 2.0, 3.0), (1.0,), (1.0,) * 5, (), "1234", 3.0])
def test_wheel_count_mismatch_is_rejected(targets):
    with pytest.raises(ci.InvalidWheelCountError) as excinfo:
        ci.WheelCommand(wheel_targets_rad_s=targets)
    assert excinfo.value.code == "invalid_wheel_count"


def test_wheel_target_out_of_actuator_range_is_rejected():
    # 50 rad/s is the actuator velocity_limit; 400 rad/s is inside the rpm band and is therefore
    # reported as unit confusion, while 5000 rad/s cannot be any unit at all.
    with pytest.raises(ci.WheelTargetsOutOfRangeError) as excinfo:
        ci.WheelCommand(wheel_targets_rad_s=(5000.0, 5000.0, 5000.0, 5000.0))
    assert excinfo.value.code == "wheel_targets_out_of_range"


def test_joint_velocity_vector_raises_on_a_missing_wheel_joint():
    command = ci.WheelCommand.from_speed(0.30, R, 0.50, "clamp")
    with pytest.raises(ci.InvalidWheelCountError):
        command.joint_velocity_vector(["joint_wheel1", "joint_wheel2", "joint_wheel3"])


def test_joint_velocity_vector_fills_only_the_wheel_joints():
    command = ci.WheelCommand.from_speed(0.30, R, 0.50, "clamp")
    names = ["joint_arm", "joint_wheel1", "joint_other", "joint_wheel2", "joint_wheel3",
             "joint_wheel4"]
    vector = command.joint_velocity_vector(names)
    assert len(vector) == len(names)
    assert vector[0] == 0.0 and vector[2] == 0.0
    assert vector[1] == vector[3] == vector[4] == vector[5] == pytest.approx(0.30 / R)


# ---------------------------------------------------------------------------------------------
# 4. Unit-confusion guards (documented ranges)
# ---------------------------------------------------------------------------------------------
def test_mm_s_speed_is_caught_as_unit_confusion():
    with pytest.raises(ci.UnitConfusionError) as excinfo:
        ci.WheelCommand.from_speed(300.0, R, 0.50, "clamp")
    assert "mm/s" in str(excinfo.value)
    assert "0.300000 m/s" in str(excinfo.value)
    # the corrected value passes
    assert ci.WheelCommand.from_speed(300.0 / 1000.0, R, 0.50, "clamp").speed_m_s == \
        pytest.approx(0.30)


def test_mm_radius_is_caught_as_unit_confusion():
    with pytest.raises(ci.UnitConfusionError) as excinfo:
        ci.WheelCommand.from_speed(0.30, 103.0, 0.50, "clamp")
    assert "millimetre" in str(excinfo.value)


def test_deg_s_yaw_rate_is_caught_as_unit_confusion():
    with pytest.raises(ci.UnitConfusionError) as excinfo:
        ci.WheelCommand.from_yaw_rate(math.radians(1200.0), R, 5.0, "clamp")
    assert "deg/s" in str(excinfo.value)


def test_rpm_wheel_target_is_caught_as_unit_confusion():
    with pytest.raises(ci.UnitConfusionError) as excinfo:
        ci.WheelCommand(wheel_targets_rad_s=(300.0, 300.0, 300.0, 300.0))
    assert "rpm" in str(excinfo.value)


def test_implausible_magnitude_is_rejected_even_in_clamp_mode():
    """A unit guard is fail-closed in BOTH modes: clamping would hide a factor-of-1000 bug."""
    with pytest.raises(ci.UnitConfusionError):
        ci.WheelCommand.from_speed(1.0e6, R, 0.50, "clamp")
    with pytest.raises(ci.UnitConfusionError):
        ci.WheelCommand.from_speed(1.0e6, R, 0.50, "reject")


# ---------------------------------------------------------------------------------------------
# 5. Audit round trip
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [
    lambda: ci.WheelCommand.from_speed(0.30, R, 0.50, "clamp"),
    lambda: ci.WheelCommand.from_speed(0.80, R, 0.50, "clamp"),
    lambda: ci.WheelCommand.from_speed(-0.20, R, 0.50, "reject"),
    lambda: ci.WheelCommand.from_yaw_rate(-1.236, R, 5.0, "clamp"),
    lambda: ci.WheelCommand.from_yaw_rate(2.0, R, 1.0, "clamp"),
])
def test_audit_round_trip(factory):
    command = factory()
    audit = command.audit()
    # JSON-serialisable (must embed in a manifest)
    text = json.dumps(audit)
    assert json.loads(text) == audit
    restored = ci.WheelCommand.from_audit(json.loads(text))
    assert restored == command
    assert restored.wheel_targets_rad_s == command.wheel_targets_rad_s
    assert restored.speed_m_s == command.speed_m_s
    assert restored.clamped == command.clamped


def test_audit_carries_the_manifest_fields_the_gpu_result_uses():
    command = ci.WheelCommand.from_speed(0.30, R, 0.50, "clamp")
    audit = command.audit()
    assert audit["schema"] == ci.AUDIT_SCHEMA
    assert audit["wheel_order"] == list(ci.WHEEL_ORDER)
    assert list(audit["wheel_targets_rad_s_per_wheel"]) == list(command.wheel_targets_rad_s)
    assert audit["convention"]["joint_axis"] == "world +X for all four wheels"
    assert audit["convention"]["p1_outward_normal_signs_is_motor_channel_not_joint"] == \
        [1.0, 1.0, -1.0, -1.0]
    assert audit["unknowns"], "the audit must carry its UNKNOWNs"
    assert json.loads(command.to_json())["speed_effective_m_s"] == pytest.approx(0.30)


def test_audit_round_trip_rejects_a_tampered_audit():
    audit = ci.WheelCommand.from_speed(0.30, R, 0.50, "clamp").audit()
    tampered = dict(audit, schema="p4_11_wheel_command/v0")
    with pytest.raises(ci.CommandValidationError):
        ci.WheelCommand.from_audit(tampered)
    tampered = dict(audit, wheel_order=["wheel1", "wheel2", "wheel3"])
    with pytest.raises(ci.CommandValidationError):
        ci.WheelCommand.from_audit(tampered)
    tampered = dict(audit)
    tampered.pop("wheel_targets_rad_s_per_wheel")
    with pytest.raises(ci.CommandValidationError):
        ci.WheelCommand.from_audit(tampered)


# ---------------------------------------------------------------------------------------------
# 6. GPU cross-check: reproduce what the GPU actually ran
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("run_id", ["P4V_030_r10", "P4V_050_r09", "P4V_clamp_r11"])
def test_gpu_p4v_wheel_targets_rederived_within_1e_6(run_id):
    """Re-derive the GPU-recorded wheel_targets_rad_s_per_wheel for a REAL committed P4V run.

    For the velocity_control profile this field holds the OPEN-LOOP profile target list, which is
    all zeros (the closed loop rebuilds the tensor every step), so the interface must reproduce
    exactly zeros from a zero speed command.  The nonzero case is covered by
    test_gpu_roll_direction_matches_the_interface and
    test_gpu_turn_targets_are_rederived_from_a_yaw_rate_within_1e_6, both real GPU runs.
    """
    result = result_of(run_id)
    recorded = result["metrics"]["wheel_targets_rad_s_per_wheel"]
    assert result["profile"] == "velocity_control"
    assert recorded == [0.0, 0.0, 0.0, 0.0]
    command = ci.WheelCommand.from_speed(0.0, result["metrics"]["velocity_control"]
                                        ["nominal_radius_m"], 0.50, "clamp")
    for index, target in enumerate(recorded):
        assert abs(command.wheel_targets_rad_s[index] - target) <= 1e-6
    assert command.travel_direction == "none"


@pytest.mark.parametrize("run_id", ["P4V_030_r10", "P4V_050_r09", "P4V_clamp_r11", "P4V_030_r09"])
def test_gpu_p4v_clamped_command_and_rolling_radius_rederived_within_1e_6(run_id):
    """The interface must reproduce (a) the recorded post-clamp command and (b) the recorded
    measured rolling radius = |dY| / |dtheta|, both to 1e-6, from the same result.json."""
    result = result_of(run_id)
    control = result["metrics"]["velocity_control"]
    command = ci.WheelCommand.from_speed(control["commanded_speed_m_s"],
                                         control["nominal_radius_m"], 0.50, "clamp")
    assert abs(command.speed_m_s - control["commanded_after_clamp_m_s"]) <= 1e-6
    assert command.clamped == bool(control["clamped"])

    displacement_y = result["metrics"]["chassis_displacement_m"][1]
    assert displacement_y < 0.0, "a positive command must travel along -Y"
    for wheel, recorded_radius in result["metrics"]["measured_rolling_radius_m"].items():
        rederived = abs(displacement_y) / abs(result["metrics"]["wheel_delta_theta_rad"][wheel])
        assert abs(rederived - recorded_radius) <= 1e-6, wheel
    # the achieved steady speed follows the interface's command (documented ~6 % proxy bias)
    assert control["steady_mean_speed_m_s"] == pytest.approx(command.speed_m_s, rel=0.10)
    assert abs(displacement_y) > 1.0, "the chassis really moved, so dY/dtheta is meaningful"


# ---------------------------------------------------------------------------------------------
# 7. Fail-closed integration path (the GPU script's pre-flight)
# ---------------------------------------------------------------------------------------------
def _preflight_parts():
    source = GPU_SCRIPT.read_text(encoding="utf-8")
    begin_marker = "# P4-11 PREFLIGHT BEGIN"
    end_marker = "# P4-11 PREFLIGHT END"
    assert begin_marker in source, "the GPU script lost its P4-11 pre-flight block"
    assert end_marker in source, "the GPU script lost the end of its P4-11 pre-flight block"
    begin = source.index(begin_marker)
    end = source.index(end_marker) + len(end_marker)
    block = textwrap.dedent(source[begin:end])
    return source, block, begin, end


def _run_preflight(**args):
    _source, block, _begin, _end = _preflight_parts()
    printed = []
    namespace = {
        "WheelCommand": ci.WheelCommand,
        "CommandValidationError": ci.CommandValidationError,
        "json": json,
        "ARGS": types.SimpleNamespace(**args),
        "print": lambda *items, **kwargs: printed.append(
            " ".join(str(item) for item in items)),
    }
    exit_code = None
    try:
        exec(compile(block, str(GPU_SCRIPT), "exec"), namespace)
    except SystemExit as exc:  # the real script raises SystemExit(2)
        exit_code = exc.code
    return exit_code, "\n".join(printed)


def test_preflight_runs_before_the_isaac_app_is_created():
    """Structural proof that a rejected command aborts before any simulator work."""
    source, block, begin, end = _preflight_parts()
    launcher_at = source.index("LAUNCHER = AppLauncher(ARGS)")
    assert end < launcher_at, ("the P4-11 pre-flight must appear BEFORE AppLauncher(ARGS); "
                              "otherwise the sim app is created before validation")
    # comments are allowed to MENTION the sim; the executable lines must not touch it
    code_only = "\n".join(line for line in block.splitlines()
                          if not line.strip().startswith("#"))
    for forbidden in ("sim.step", "sim.reset", "author(", "SimulationContext", "makedirs"):
        assert forbidden not in code_only, ("the pre-flight must not touch the sim: found %r"
                                           % forbidden)
    assert "SystemExit(2)" in code_only


def test_preflight_reject_exits_non_zero_without_stepping_the_sim():
    exit_code, output = _run_preflight(run_id="P4I_reject", profile="velocity_control",
                                       velocity_cmd=0.80, speed_limit=0.50,
                                       nominal_radius=R, limit_mode="reject")
    assert exit_code == 2
    payload = json.loads(output.split("GATE_RESULT=ABORTED")[0])
    assert payload["aborted"] is True
    assert payload["reason"] == "SpeedLimitExceededError"
    assert payload["error_code"] == "speed_limit_exceeded"
    assert payload["sim_steps"] == 0 and payload["stage_authored"] is False
    assert payload["root_writes"] == 0
    assert "GATE_RESULT=ABORTED" in output


def test_preflight_clamp_mode_does_not_abort():
    exit_code, output = _run_preflight(run_id="P4I_clamp", profile="velocity_control",
                                       velocity_cmd=0.80, speed_limit=0.50,
                                       nominal_radius=R, limit_mode="clamp")
    assert exit_code is None
    assert output == ""


def test_preflight_rejects_unit_confusion_before_the_sim():
    exit_code, output = _run_preflight(run_id="P4I_mm", profile="velocity_control",
                                       velocity_cmd=300.0, speed_limit=0.50,
                                       nominal_radius=R, limit_mode="clamp")
    assert exit_code == 2
    assert "UnitConfusionError" in output
    assert "GATE_RESULT=ABORTED" in output


def script_defaults():
    """The GPU script's own argparse defaults, read from the source (not duplicated here)."""
    import re

    source = GPU_SCRIPT.read_text(encoding="utf-8")
    defaults = {}
    for flag in ("--velocity-cmd", "--speed-limit", "--nominal-radius", "--limit-mode"):
        pattern = (r"PARSER\.add_argument\(" + re.escape('"' + flag + '"')
                   + r".*?default=([^,]+),")
        match = re.search(pattern, source)
        assert match, "could not read the %s default from the GPU script" % flag
        raw = match.group(1).strip().strip('"')
        defaults[flag] = raw if flag == "--limit-mode" else float(raw)
    return defaults


@pytest.mark.parametrize("run_id", [run for run in gpu_runs() if run.startswith("P4V_")])
def test_committed_gpu_commands_pass_the_preflight(run_id):
    """Every committed P4V command line must survive the new pre-flight unchanged."""
    command_line = manifest_of(run_id)["command"]
    tokens = command_line.split()
    values = dict(script_defaults())     # the manifest omits flags that used the default
    for index, token in enumerate(tokens):
        if token in values:
            values[token] = tokens[index + 1]
    result = result_of(run_id)
    exit_code, output = _run_preflight(
        run_id=run_id, profile=result["profile"],
        velocity_cmd=float(values["--velocity-cmd"]),
        speed_limit=float(values["--speed-limit"]),
        nominal_radius=float(values["--nominal-radius"]),
        limit_mode=values.get("--limit-mode", "clamp"))
    assert exit_code is None, output
    # and the clamp the interface computes agrees with what the GPU recorded
    command = ci.WheelCommand.from_speed(float(values["--velocity-cmd"]),
                                         float(values["--nominal-radius"]),
                                         float(values["--speed-limit"]), "clamp")
    control = result["metrics"]["velocity_control"]
    assert abs(command.speed_m_s - control["commanded_after_clamp_m_s"]) <= 1e-6
    assert command.clamped == bool(control["clamped"])


def test_gpu_script_imports_the_interface_before_the_app():
    source = GPU_SCRIPT.read_text(encoding="utf-8")
    assert "from p4_command_interface import" in source
    assert (source.index("from p4_command_interface import")
            < source.index("LAUNCHER = AppLauncher(ARGS)"))
    assert "--limit-mode" in source
