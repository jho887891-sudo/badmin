# -*- coding: utf-8 -*-
"""P5B canary - reset.py may not carry one arm-agnostic rest pose again.

WHY THIS IS A CANARY AND NOT A UNIT TEST
    P5 pinned the defect the other way round: its canary asserted that reset.py's joint3 = -1.5 was
    still outside the B601-RS limits, so fixing reset.py turned the P5 suite RED on purpose
    (management/morph_one_rs/P5_ARM_STATIC_CONSISTENCY.md section 5.1). P5B flips it: this file now
    asserts the fixed shape, and against the unmodified reset.py it fails at the first assertion.
    Recorded reverse control: see management/morph_one_rs/P5B_RESET_POSE.md section 3.

    No import happens here, so the check still works on the old reset.py (which would fail to make
    sense of the new tables) and needs neither torch nor pxr.
"""
from __future__ import annotations

import re

from pose_data import RESET_PY, REST_POSE_PY


def test_reset_py_no_longer_hardcodes_one_arm_agnostic_rest_pose():
    """The literal table and the soft fallback are both gone; the pose now comes from a model."""
    assert RESET_PY.is_file(), "no reset.py at " + str(RESET_PY)
    text = RESET_PY.read_text(encoding="utf-8")

    assert re.search(r"^REST_JOINT\s*=\s*\{", text, re.MULTILINE) is None, (
        "reset.py still defines a module-level REST_JOINT table: one pose for every arm is the "
        "defect (P5 F1). It must come from rest_pose.rest_joint_pos(model, robot.joint_names).")

    for literal in ("-1.5", "1.5"):
        assert literal not in text, (
            "reset.py still contains the PiPER angle %r, which is outside the B601-RS joint3 range "
            "[0, 3.14] rad" % literal)

    assert ".get(name, 0.0)" not in text, (
        "reset.py still falls back to a silent 0.0 for a joint it does not know about")

    assert "from .rest_pose import" in text, (
        "reset.py does not import the per-model tables from rest_pose.py")
    for symbol in ("resolve_model", "rest_joint_pos"):
        assert symbol in text, "reset.py does not use rest_pose." + symbol


def test_the_model_tables_exist_and_are_name_keyed():
    """The pose must be a table per robot, not arithmetic on an index."""
    assert REST_POSE_PY.is_file(), "no rest_pose.py at " + str(REST_POSE_PY)
    text = REST_POSE_PY.read_text(encoding="utf-8")
    for symbol in ("MODEL_PIPER", "MODEL_REBOT_RS", "REST_POSES"):
        assert symbol in text, "rest_pose.py is missing " + symbol
    assert "joint_pos[" not in text and "joint_pos[k]" not in text, (
        "rest_pose.py appears to index joints positionally; the DOF order is not the USD order")
