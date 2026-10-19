# -*- coding: utf-8 -*-
"""F1 regression gate: the B601-RS must hold its rest pose in BOTH the bare arm and the full scene.

F1 background (management/rebot_b601_rs/ARM_MIGRATION_RECORD.md, section E): in the assembled scene
joint3 left +1.0 rad for -0.067 rad within 60 steps, i.e. below its own lower limit of 0, while the
position target was still the rest pose. This test reads the evidence the remote harness produces
and refuses to pass on a curve that drifts, dives or crosses a limit.

The evidence file is committed, so a missing file is a failure, not a skip: without it there is no
proof that the arm holds at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "simulation"))
from src.simulation.rebot_b601_rs.joint_limits import JOINT_LIMITS  # noqa: E402
from badminton_scene.scene_layout import REBOT_RS  # noqa: E402

EVIDENCE = ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "HOLD_rest_pose.json"

#: 5 s of holding is the acceptance criterion for F1.
REQUIRED_HOLD_STEPS = 1200
#: A held joint may sit at most this far from its target (drive compliance under gravity).
MAX_HOLD_ERROR_RAD = 0.05


@pytest.fixture(scope="module")
def ev():
    assert EVIDENCE.is_file(), (
        "missing rest-pose hold evidence; run scripts/simulation/check_arm_rest_pose_hold.py on the GPU host"
    )
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_both_environments_are_covered(ev):
    """The bare arm and the assembled scene must both be recorded, as F1 requires."""
    modes = {run["mode"] for run in ev["runs"]}
    assert {"standalone", "scene"} <= modes, modes


def test_every_run_holds_for_five_seconds(ev):
    for run in ev["runs"]:
        assert run["steps_total"] >= REQUIRED_HOLD_STEPS, run["name"]
        last = run["samples"][-1]
        assert last["step"] >= REQUIRED_HOLD_STEPS, run["name"]
        for name, err in zip(run["joint_names"], last["hold_error_rad"]):
            assert abs(err) <= MAX_HOLD_ERROR_RAD, f"{run['name']}: {name} error {err}"


def test_joint2_and_joint3_never_cross_their_official_limits(ev):
    for run in ev["runs"]:
        if run["mode"] != "full" and run["name"].endswith("gravity_off"):
            continue  # a gravity-free probe is a diagnostic, not an acceptance run
        for sample in run["samples"]:
            for name in ("joint2", "joint3"):
                idx = run["joint_names"].index(name)
                q = sample["joint_pos"][idx]
                lo = JOINT_LIMITS[name].angle.lower
                hi = JOINT_LIMITS[name].angle.upper
                assert lo - 1e-3 <= q <= hi + 1e-3, (
                    f"{run['name']} step {sample['step']}: {name}={q} outside [{lo}, {hi}]"
                )


def test_the_end_effector_does_not_dive(ev):
    """A collapse shows up as the wrist dropping toward the floor; hold the height instead."""
    for run in ev["runs"]:
        if run["name"].endswith("gravity_off"):
            continue
        rel = [s["ee_rel_mount"][2] for s in run["samples"]]
        assert min(rel) > 0.0, f"{run['name']}: wrist went below the mount plane ({min(rel)} m)"
        drop = rel[0] - min(rel)
        assert drop <= 0.05, f"{run['name']}: wrist dropped {drop} m"


def test_the_command_is_recorded_and_equals_the_rest_pose(ev):
    """If the target never becomes the rest pose, a hold test would pass for the wrong reason.

    Per-DOF read-back: root_physx_view.get_dof_position_targets() works and is recorded, together
    with the articulation's joint_pos_target buffer; get_applied_joint_efforts() does NOT exist on
    this IsaacLab build (AttributeError: ArticulationView has no such method), so the applied
    torque is evidenced indirectly by the drive parameters and the measured hold.
    """
    for run in ev["runs"]:
        first = run["samples"][0]
        assert first["joint_pos_target"], run["name"]
        for name, target in zip(run["joint_names"], first["joint_pos_target"]):
            assert abs(target - run["rest_pose"].get(name, 0.0)) <= 1e-3, (
                f"{run['name']}: {name} target is {target}, not the rest pose")
        physx = first.get("physx_dof_position_targets")
        assert isinstance(physx, list) and len(physx) == len(run["joint_names"]), (
            f"{run['name']}: PhysX did not report the DOF targets: {physx}")
        for name, target in zip(run["joint_names"], physx):
            assert abs(target - run["rest_pose"].get(name, 0.0)) <= 1e-3, (
                f"{run['name']}: {name} never reached PhysX (target {target})")
        for name, err in zip(run["joint_names"], first["hold_error_rad"]):
            if not name.startswith("joint"):
                continue
            assert abs(err) <= 0.05, f"{run['name']}: {name} never followed the command ({err})"


def test_drive_parameters_are_the_ones_the_project_intends(ev):
    """k / c and the effort ceiling must be the configured ones, not the converter's degree values.

    The physx_dof_* read-back is authoritative here; the actuator objects stringify their tensors
    in the JSON, so they are only checked for presence.
    """
    expected_k = float(REBOT_RS["stiffness"])
    expected_c = float(REBOT_RS["damping"])
    for run in ev["runs"]:
        assert run.get("actuators"), run["name"]
        assert run["physx_dof_stiffnesses"] == [expected_k] * 8, run["name"]
        assert run["physx_dof_dampings"] == [expected_c] * 8, run["name"]
        forces = run["physx_dof_max_forces"]
        assert forces[:3] == [36.0, 36.0, 36.0], run["name"]
        assert forces[3:6] == [14.0, 14.0, 14.0], run["name"]
