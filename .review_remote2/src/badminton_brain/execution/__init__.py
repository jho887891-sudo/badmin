# -*- coding: utf-8 -*-
"""Execution layer (ROBOT_BRAIN.md S11/S13): SafeCommand -> robot drivers -> Feedback.

The simulation adapter is a pure-numpy command sink: it converts a SafeCommand into the
Morph One four-steer/four-drive targets and the six PiPER joint targets and never imports
Isaac/Kit/Omni (the real environment driver wires it up from outside).
"""
from .sim_adapter import (  # noqa: F401
    ExecutionCommand,
    SimExecutionAdapter,
)

__all__ = ["SimExecutionAdapter", "ExecutionCommand"]
