# -*- coding: utf-8 -*-
"""Phase 3 (TDD) - Morph One four-steer/four-drive kinematics tests.

Spec: BADMINTON_ROBOT.md S10 (IK), S11 (steer optimisation), S51/S52 (required tests).
"""
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'simulation'))

from robots.badminton_robot.badminton_robot_cfg import WheelId  # noqa: E402
from robots.badminton_robot.morph_one.kinematics import (  # noqa: E402
    WheelTargets,
    body_twist_to_wheel_targets,
    normalise_angle,
    wheel_targets_to_body_twist,
)

# TEMP wheel centres (REQUIRES_MEASUREMENT in cfg) supplied explicitly for the maths.
WHEEL_POSITIONS = {
    WheelId.FL: (0.25, 0.20),
    WheelId.FR: (0.25, -0.20),
    WheelId.RL: (-0.25, 0.20),
    WheelId.RR: (-0.25, -0.20),
}
R_W = 0.06


class BasicTests(unittest.TestCase):
    def test_normalise_angle_wraps_to_pi(self) -> None:
        self.assertAlmostEqual(normalise_angle(3 * math.pi), math.pi, places=12)
        self.assertAlmostEqual(normalise_angle(-3 * math.pi / 2), math.pi / 2, places=12)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            body_twist_to_wheel_targets([1, 0, 0], WHEEL_POSITIONS, 0.0)
        with self.assertRaises(KeyError):
            body_twist_to_wheel_targets([1, 0, 0], {WheelId.FL: (0.0, 0.0)}, R_W)


class ForwardDriveTests(unittest.TestCase):
    def test_pure_forward_aligns_all_wheels(self) -> None:
        targets = body_twist_to_wheel_targets([0.5, 0.0, 0.0], WHEEL_POSITIONS, R_W)
        for wid in WheelId:
            self.assertAlmostEqual(targets[wid].steer_angle_rad, 0.0, places=12)
            self.assertAlmostEqual(abs(targets[wid].wheel_speed_rad_s), 0.5 / R_W, places=12)

    def test_pure_strafe_turns_wheels_ninety_degrees(self) -> None:
        targets = body_twist_to_wheel_targets([0.0, 0.4, 0.0], WHEEL_POSITIONS, R_W)
        for wid in WheelId:
            self.assertAlmostEqual(abs(targets[wid].steer_angle_rad), math.pi / 2, places=12)
            self.assertAlmostEqual(abs(targets[wid].wheel_speed_rad_s), 0.4 / R_W, places=12)

    def test_spin_tangential_speeds(self) -> None:
        wz = 0.8
        targets = body_twist_to_wheel_targets([0.0, 0.0, wz], WHEEL_POSITIONS, R_W)
        for wid, (xi, yi) in WHEEL_POSITIONS.items():
            vix, viy = -wz * yi, wz * xi
            self.assertAlmostEqual(targets[wid].steer_angle_rad, math.atan2(viy, vix), places=12)
            self.assertAlmostEqual(abs(targets[wid].wheel_speed_rad_s), math.hypot(vix, viy) / R_W, places=12)

    def test_mixed_motion_round_trips_to_chassis_twist(self) -> None:
        twist = np.array([0.35, -0.22, 0.45])
        targets = body_twist_to_wheel_targets(twist, WHEEL_POSITIONS, R_W)
        recovered = wheel_targets_to_body_twist(targets, WHEEL_POSITIONS, R_W)
        np.testing.assert_allclose(recovered, twist, atol=1e-9)

    def test_zero_command_gives_zero_speed(self) -> None:
        targets = body_twist_to_wheel_targets([0.0, 0.0, 0.0], WHEEL_POSITIONS, R_W)
        for wid in WheelId:
            self.assertAlmostEqual(targets[wid].wheel_speed_rad_s, 0.0, places=12)


class SteerOptimisationTests(unittest.TestCase):
    def test_flips_when_current_steer_is_far_away(self) -> None:
        current = {wid: math.pi for wid in WheelId}  # wheels pointing backwards
        targets = body_twist_to_wheel_targets([0.5, 0.0, 0.0], WHEEL_POSITIONS, R_W, current_steer_rad=current)
        for wid in WheelId:
            delta = abs(normalise_angle(targets[wid].steer_angle_rad - current[wid]))
            self.assertLessEqual(delta, math.pi / 2 + 1e-12)
            self.assertLess(targets[wid].wheel_speed_rad_s, 0.0)

    def test_optimisation_is_equivalent(self) -> None:
        twist = [0.3, 0.1, 0.2]
        plain = body_twist_to_wheel_targets(twist, WHEEL_POSITIONS, R_W)
        current = {wid: plain[wid].steer_angle_rad + math.pi for wid in WheelId}
        flipped = body_twist_to_wheel_targets(twist, WHEEL_POSITIONS, R_W, current_steer_rad=current)
        for wid in WheelId:
            a = plain[wid]
            b = flipped[wid]
            # equivalent (theta, speed) and (theta+pi, -speed) representations
            dtheta = abs(normalise_angle(b.steer_angle_rad - a.steer_angle_rad))
            self.assertAlmostEqual(dtheta, math.pi, places=9)
            self.assertAlmostEqual(b.wheel_speed_rad_s, -a.wheel_speed_rad_s, places=9)
            recovered = wheel_targets_to_body_twist(flipped, WHEEL_POSITIONS, R_W)
            np.testing.assert_allclose(recovered, twist, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
