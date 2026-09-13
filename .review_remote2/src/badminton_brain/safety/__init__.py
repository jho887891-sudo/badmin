# -*- coding: utf-8 -*-
"""Safety layer (ROBOT_BRAIN.md S11/S12): the only producer of executable commands.

T8 deliverable: src/badminton_brain/safety/safety_shield.py (WholeBodyTarget -> SafeCommand).
"""
from .safety_shield import (  # noqa: F401
    ACTION_CONTROLLED_STOP, ACTION_ESTOP_REQUEST, ACTION_HOLD, ACTION_PASS, ACTION_PROJECT,
    ACTION_PROTECTIVE_STOP, SafetyContext, SafetyLimits, SafetyShield,
)

__all__ = ["SafetyContext", "SafetyLimits", "SafetyShield",
           "ACTION_PASS", "ACTION_PROJECT", "ACTION_HOLD", "ACTION_CONTROLLED_STOP",
           "ACTION_PROTECTIVE_STOP", "ACTION_ESTOP_REQUEST"]
