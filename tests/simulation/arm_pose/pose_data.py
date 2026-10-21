# -*- coding: utf-8 -*-
"""Shared data and paths for the P5B reset-pose tests (no pytest fixtures, so tests can import it).

KEEPING THESE AS LITERALS IS THE POINT: they are the independent side of the comparison. If the
production table drifted, a test that read its expectations from the production table would agree
with itself and prove nothing. "The PiPER robot" therefore keeps its Stage-0 name set and values
here, and the RS name set comes from the pxr export, not from the module under test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCENE_DIR = REPO_ROOT / "simulation"

FACTS_JSON = (
    REPO_ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "morph_one" / "p5_static"
    / "arm_joint_facts.json"
)
RESET_PY = SCENE_DIR / "badminton_scene" / "reset.py"
REST_POSE_PY = SCENE_DIR / "badminton_scene" / "rest_pose.py"

#: DOF name sets the two arms present at runtime.
PIPER_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
REBOT_RS_JOINT_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "gripper_joint1", "gripper_joint2",
)

#: The Stage-0 PiPER rest pose, exactly as reset.py used to write it on every arm.
LEGACY_PIPER_REST = {
    "joint1": 0.0, "joint2": 1.5, "joint3": -1.5, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0,
}


def read_facts():
    return json.loads(FACTS_JSON.read_text(encoding="utf-8"))


def read_usd_limits():
    """{joint: (lower, upper, unit)} from the pxr export, angles converted deg -> rad.

    The asset stores revolute limits in DEGREES (UsdPhysics convention); reading them as radians
    would be a 57.3x error. The conversion happens here, once, and the unit travels with the numbers
    so a caller cannot mistake one for the other.
    """
    out = {}
    for joint in read_facts()["joints"]:
        limits = joint.get("limits") or {}
        lower, upper = limits.get("lower"), limits.get("upper")
        if lower is None or upper is None:
            continue  # the Fixed joint (root_joint) has no travel
        if limits.get("unit") == "degree":
            out[joint["name"]] = (math.radians(lower), math.radians(upper), "rad")
        else:
            out[joint["name"]] = (float(lower), float(upper), limits.get("unit", ""))
    return out
