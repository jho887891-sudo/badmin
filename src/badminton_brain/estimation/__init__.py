# -*- coding: utf-8 -*-
"""Estimation layer modules (ROBOT_BRAIN.md S11/S12).

One responsibility per file: robot localization (EKF over odom + IMU + absolute pose) lives in
robot_localization.py; shuttle state estimation is a separate module owned by its own task and is
deliberately not imported here, so that adding it can never break this import path.
"""
from .robot_localization import (  # noqa: F401
    STATE_DIM, STATE_NAMES, LocalizationConfig, LocalizationNoiseConfig, RobotLocalization, wrap_angle,
)

__all__ = ["RobotLocalization", "LocalizationConfig", "LocalizationNoiseConfig", "wrap_angle",
           "STATE_NAMES", "STATE_DIM"]
