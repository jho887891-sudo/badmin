# -*- coding: utf-8 -*-
"""T2 - robot localization EKF (court frame): acceptance tests.

Spec:
  * docs/architecture/ROBOT_BRAIN.md S11/S12 - estimation layer, court frame only
  * docs/architecture/COORDINATE_SYSTEM.md S3 (court frame), S5.2 (robot pose = x,y,yaw,vx,vy,wz),
    S2.4 (interface quaternion order x,y,z,w)
  * docs/simulation/BADMINTON_ROBOT.md S4/S12 (every number declares its authenticity),
    S29 (reset(env_ids) may only affect the selected environments)
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.estimation.robot_localization import (  # noqa: E402
    RobotLocalization, wrap_angle,
)
from badminton_brain.status import AssetStatus, Param  # noqa: E402
from badminton_brain.types import BrainBoundaryError, UnifiedState  # noqa: E402

DT = 0.01                        # 100 Hz
STEPS = 200                      # 2.0 s
SPEED = 1.5                      # m/s along the body +X axis
YAW_RATE = 0.5                   # rad/s
YAW0 = 0.3                       # rad
START_XY = (-1.60, 0.0)          # ROBOT S31 engineering baseline

# Acceptance bounds, all measured on the synthetic scenarios below (printed by the tests).
ARC_POSITION_TOL = 1e-9          # exact-arc odometry: the model matches the generator
ARC_YAW_TOL = 1e-9
NOISY_POSITION_TOL = 0.06        # m, 2 s of noisy 100 Hz odom + 10 Hz absolute pose updates
NOISY_YAW_TOL = 0.02             # rad


def arc_step(x: float, y: float, yaw: float, v: float, w: float, dt: float):
    """Exact solution of the unicycle model x'=v cos(yaw), y'=v sin(yaw), yaw'=w."""
    if abs(w) < 1e-12:
        return np.array([x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw)]), yaw + w * dt
    return np.array([x + (v / w) * (math.sin(yaw + w * dt) - math.sin(yaw)),
                     y - (v / w) * (math.cos(yaw + w * dt) - math.cos(yaw))]), yaw + w * dt


def straight_line_with_yaw_rate(steps: int, dt: float, v: float = SPEED, w: float = YAW_RATE,
                                yaw0: float = YAW0, start=START_XY) -> np.ndarray:
    """Ground truth poses (steps+1, 3) = (x, y, yaw) in the court frame."""
    poses = np.zeros((steps + 1, 3))
    x, y = float(start[0]), float(start[1])
    yaw = yaw0
    poses[0] = (x, y, yaw)
    for k in range(steps):
        p, yaw = arc_step(x, y, yaw, v, w, dt)
        x, y = float(p[0]), float(p[1])
        poses[k + 1] = (x, y, wrap_angle(yaw))
    return poses


def body_increments(poses: np.ndarray) -> np.ndarray:
    """Odometry increments (steps, 3) = (dx, dy, dyaw) in the robot base frame."""
    out = np.zeros((len(poses) - 1, 3))
    for k in range(len(poses) - 1):
        dxw = poses[k + 1, 0] - poses[k, 0]
        dyw = poses[k + 1, 1] - poses[k, 1]
        c, s = math.cos(poses[k, 2]), math.sin(poses[k, 2])
        out[k] = (c * dxw + s * dyw, -s * dxw + c * dyw, wrap_angle(poses[k + 1, 2] - poses[k, 2]))
    return out


def ekf_at(pose_xyyaw, num_envs: int = 1) -> RobotLocalization:
    """EKF pinned to a known pose (tiny covariance) so predictions are exact-by-construction."""
    ekf = RobotLocalization(num_envs)
    ekf.initialize(np.tile(np.asarray(pose_xyyaw, dtype=float), (num_envs, 1)),
                   covariance=np.tile(np.diag([1e-12] * 6), (num_envs, 1, 1)))
    return ekf


    def test_per_env_dt_and_one_dimensional_imu_arrays(self) -> None:
        """dt and the IMU input must be per-environment: the batch is N independent filters."""
        ekf = RobotLocalization(2)
        ekf.initialize(np.zeros((2, 3)), covariance=np.tile(np.eye(6) * 1e-12, (2, 1, 1)))
        ekf.predict(np.array([0.01, 0.05]),                     # per-env dt
                    np.array([[0.01, 0.0, 0.0], [0.0, 0.0, 0.02]]),
                    np.array([0.5, -0.5]))                      # 1-D gyro array
        state = ekf.state
        self.assertAlmostEqual(state[0, 0], 0.01, places=12)
        self.assertAlmostEqual(state[0, 3], 1.0, places=12)     # 0.01 m over 10 ms
        self.assertAlmostEqual(state[0, 2], 0.005, places=12)   # 0.5 rad/s over 10 ms
        self.assertAlmostEqual(state[1, 0], 0.0, places=12)
        self.assertAlmostEqual(state[1, 2], -0.025, places=12)  # -0.5 rad/s over 50 ms


class CourtFrameConventionTests(unittest.TestCase):
    def test_body_increment_is_rotated_into_court_frame(self) -> None:
        # yaw = +90 deg: the robot body +X axis points to court +Y.
        ekf = ekf_at((0.0, 0.0, math.pi / 2.0))
        ekf.predict(DT, np.array([[1.0, 0.0, 0.0]]), None)
        self.assertAlmostEqual(ekf.state[0, 0], 0.0, places=12)
        self.assertAlmostEqual(ekf.state[0, 1], 1.0, places=12)
        # yaw = 180 deg: the robot faces court -X.
        ekf = ekf_at((0.0, 0.0, math.pi))
        ekf.predict(DT, np.array([[1.0, 0.0, 0.0]]), None)
        self.assertAlmostEqual(ekf.state[0, 0], -1.0, places=12)
        self.assertAlmostEqual(ekf.state[0, 1], 0.0, places=12)

    def test_straight_line_keeps_lateral_offset_zero(self) -> None:
        ekf = ekf_at(START_XY + (0.0,))
        odom = np.tile(np.array([SPEED * DT, 0.0, 0.0]), (STEPS, 1))
        for k in range(STEPS):
            ekf.predict(DT, odom[k:k + 1], np.array([[0.0]]))
        state = ekf.state[0]
        self.assertAlmostEqual(state[0], START_XY[0] + SPEED * STEPS * DT, places=10)
        self.assertAlmostEqual(state[1], 0.0, places=12)
        self.assertAlmostEqual(state[2], 0.0, places=12)
        self.assertAlmostEqual(state[3], SPEED, places=10)
        self.assertAlmostEqual(state[4], 0.0, places=12)


class BoundedErrorTests(unittest.TestCase):
    def test_curved_arc_pose_error_is_bounded(self) -> None:
        """Straight line + constant yaw rate, exact-arc odometry and IMU yaw rate."""
        truth = straight_line_with_yaw_rate(STEPS, DT)
        odom = body_increments(truth)
        for use_imu in (False, True):
            ekf = ekf_at(truth[0])
            for k in range(STEPS):
                imu = np.array([[YAW_RATE]]) if use_imu else None
                ekf.predict(DT, odom[k:k + 1], imu)
            est = ekf.state[0]
            pos_err = float(np.linalg.norm(est[:2] - truth[-1, :2]))
            yaw_err = float(abs(wrap_angle(est[2] - truth[-1, 2])))
            print(f"[arc use_imu={use_imu}] position_error={pos_err:.3e} m  yaw_error={yaw_err:.3e} rad  "
                  f"vx={est[3]:.6f} vy={est[4]:.6f} wz={est[5]:.6f}")
            self.assertLess(pos_err, ARC_POSITION_TOL, f"use_imu={use_imu}")
            self.assertLess(yaw_err, ARC_YAW_TOL, f"use_imu={use_imu}")
            self.assertLess(abs(est[3] - SPEED) / SPEED, 1e-3)   # arc chord is slightly shorter
            self.assertLess(abs(est[5] - YAW_RATE), 1e-9)

    def test_noisy_scenario_error_is_bounded_and_beats_dead_reckoning(self) -> None:
        rng = np.random.default_rng(20260913)
        truth = straight_line_with_yaw_rate(STEPS, DT)
        odom = body_increments(truth)
        odom_noisy = odom.copy()
        odom_noisy[:, :2] += rng.normal(0.0, 0.002, (STEPS, 2))
        odom_noisy[:, 2] += rng.normal(0.0, 0.002, STEPS)
        wz_noisy = YAW_RATE + rng.normal(0.0, 0.005, STEPS)          # gyro @ 100 Hz
        meas_cov = np.diag([0.01 ** 2, 0.01 ** 2, 0.005 ** 2])       # absolute pose @ 10 Hz

        ekf = ekf_at(truth[0])
        dead_reckoning = truth[0].copy()
        for k in range(STEPS):
            if k % 10 == 0:
                measured = truth[k] + rng.normal(0.0, [0.01, 0.01, 0.005])
                ekf.update_pose_measurement(measured.reshape(1, 3), meas_cov.reshape(1, 3, 3))
            ekf.predict(DT, odom_noisy[k:k + 1], np.array([[wz_noisy[k]]]))
            dead_reckoning[0] += (math.cos(dead_reckoning[2]) * odom_noisy[k, 0]
                                  - math.sin(dead_reckoning[2]) * odom_noisy[k, 1])
            dead_reckoning[1] += (math.sin(dead_reckoning[2]) * odom_noisy[k, 0]
                                  + math.cos(dead_reckoning[2]) * odom_noisy[k, 1])
            dead_reckoning[2] = wrap_angle(dead_reckoning[2] + odom_noisy[k, 2])

        est = ekf.state[0]
        pos_err = float(np.linalg.norm(est[:2] - truth[-1, :2]))
        yaw_err = float(abs(wrap_angle(est[2] - truth[-1, 2])))
        dr_pos_err = float(np.linalg.norm(dead_reckoning[:2] - truth[-1, :2]))
        dr_yaw_err = float(abs(wrap_angle(dead_reckoning[2] - truth[-1, 2])))
        print(f"[noisy] ekf_position_error={pos_err:.4f} m (dead reckoning {dr_pos_err:.4f} m) | "
              f"ekf_yaw_error={yaw_err:.4f} rad (dead reckoning {dr_yaw_err:.4f} rad) | "
              f"trace(P)={float(np.trace(ekf.covariance[0])):.4e}")
        self.assertLess(pos_err, NOISY_POSITION_TOL)
        self.assertLess(yaw_err, NOISY_YAW_TOL)
        self.assertLess(pos_err, dr_pos_err)
        self.assertLess(yaw_err, dr_yaw_err)
        self.assertTrue(np.all(np.isfinite(ekf.state)))

    def test_imu_yaw_rate_is_used_when_present_otherwise_odom_yaw(self) -> None:
        # odom reports zero yaw increment, the gyro reports 1.0 rad/s
        ekf = ekf_at((0.0, 0.0, 0.0))
        ekf.predict(DT, np.array([[0.0, 0.0, 0.0]]), np.array([[1.0]]))
        self.assertAlmostEqual(ekf.state[0, 2], 1.0 * DT, places=12)
        # without an IMU reading the wheel yaw increment is the fallback
        ekf = ekf_at((0.0, 0.0, 0.0))
        ekf.predict(DT, np.array([[0.0, 0.0, 0.25]]), None)
        self.assertAlmostEqual(ekf.state[0, 2], 0.25, places=12)


class BatchTests(unittest.TestCase):
    def test_batch_envs_are_independent(self) -> None:
        ekf = RobotLocalization(3)
        odom = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        ekf.predict(DT, odom, None)
        state = ekf.state
        self.assertEqual(state.shape, (3, 6))
        self.assertAlmostEqual(state[0, 0], START_XY[0] + 1.0, places=12)
        self.assertAlmostEqual(state[0, 1], 0.0, places=12)
        self.assertAlmostEqual(state[0, 2], 0.0, places=12)
        self.assertAlmostEqual(state[1, 0], START_XY[0], places=12)
        self.assertAlmostEqual(state[1, 1], 2.0, places=12)
        self.assertAlmostEqual(state[2, 0], START_XY[0], places=12)
        self.assertAlmostEqual(state[2, 2], 1.0, places=12)
        self.assertNotEqual(state[0, 2], state[2, 2])


class CovarianceTests(unittest.TestCase):
    def test_covariance_stays_symmetric_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(5)
        ekf = RobotLocalization(4)
        for k in range(300):
            ekf.predict(DT, rng.normal(0.0, 0.01, (4, 3)), rng.normal(0.0, 0.01, (4, 1)))
            if k % 10 == 0:
                meas = ekf.state[:, :3] + rng.normal(0.0, 0.005, (4, 3))
                ekf.update_pose_measurement(meas, np.tile(np.diag([1e-4, 1e-4, 1e-5]), (4, 1, 1)))
        P = ekf.covariance
        self.assertTrue(np.array_equal(P, np.transpose(P, (0, 2, 1))), "P must be exactly symmetric")
        min_eig = float(np.min(np.linalg.eigvalsh(P)))
        print(f"[covariance] min_eigenvalue={min_eig:.3e} "
              f"max_trace={float(np.max(np.trace(P, axis1=1, axis2=2))):.4e}")
        self.assertGreater(min_eig, -1e-10)
        self.assertTrue(np.all(np.isfinite(P)))
        self.assertLess(float(np.max(np.trace(P, axis1=1, axis2=2))), 100.0)

    def test_pose_update_moves_toward_measurement_and_shrinks_covariance(self) -> None:
        ekf = RobotLocalization(1)
        ekf.initialize(np.array([[START_XY[0], START_XY[1], 0.0]]),
                       covariance=np.tile(np.diag([1.0, 1.0, 0.5, 0.1, 0.1, 0.1]), (1, 1, 1)))
        before_trace = float(np.trace(ekf.covariance[0]))
        ekf.update_pose_measurement(np.array([[0.5, -0.25, 0.1]]),
                                    np.diag([1e-4, 1e-4, 1e-6]).reshape(1, 3, 3))
        state = ekf.state[0]
        self.assertLess(abs(state[0] - 0.5), 0.05)
        self.assertLess(abs(state[1] + 0.25), 0.05)
        self.assertLess(abs(wrap_angle(state[2] - 0.1)), 0.05)
        self.assertLess(float(np.trace(ekf.covariance[0])), before_trace)

    def test_yaw_innovation_takes_the_short_way_across_pi(self) -> None:
        prior_yaw = math.pi - 0.05
        ekf = RobotLocalization(1)
        ekf.initialize(np.array([[0.0, 0.0, prior_yaw]]),
                       covariance=np.tile(np.diag([1e-6, 1e-6, 0.4, 0.1, 0.1, 0.1]), (1, 1, 1)))
        measured_yaw = -math.pi + 0.05
        ekf.update_pose_measurement(np.array([[0.0, 0.0, measured_yaw]]),
                                    np.diag([1e-6, 1e-6, 1e-6]).reshape(1, 3, 3))
        yaw = float(ekf.state[0, 2])
        self.assertLess(abs(wrap_angle(yaw - measured_yaw)), 0.04, "must converge to the measurement")
        self.assertLess(abs(wrap_angle(yaw - prior_yaw)), 0.15, "must move the short way (~0.1 rad)")
        self.assertLessEqual(abs(yaw), math.pi + 1e-12, "yaw state stays in (-pi, pi]")


class ResetIsolationTests(unittest.TestCase):
    def test_reset_only_touches_the_selected_envs(self) -> None:
        ekf = RobotLocalization(4)
        rng = np.random.default_rng(11)
        for _ in range(20):
            ekf.predict(DT, rng.normal(0.0, 0.01, (4, 3)), None)
        state_before = ekf.state.copy()
        cov_before = ekf.covariance.copy()
        ekf.reset([1, 3])
        state_after, cov_after = ekf.state, ekf.covariance
        untouched = [0, 2]
        self.assertTrue(np.array_equal(state_after[untouched], state_before[untouched]),
                        "reset(env_ids) must not change unselected environments element-wise")
        self.assertTrue(np.array_equal(cov_after[untouched], cov_before[untouched]),
                        "reset(env_ids) must not change unselected covariances element-wise")
        for env in (1, 3):
            self.assertFalse(np.array_equal(state_after[env], state_before[env]))
            self.assertEqual(state_after[env, 3:].tolist(), [0.0, 0.0, 0.0])
            self.assertTrue(np.allclose(state_after[env, :2], np.array(START_XY)))

    def test_reset_all_and_empty_env_ids(self) -> None:
        ekf = RobotLocalization(2)
        ekf.predict(DT, np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]), None)
        moved = ekf.state.copy()
        ekf.reset([])
        self.assertTrue(np.array_equal(ekf.state, moved), "reset([]) must be a no-op")
        ekf.reset(None)
        self.assertTrue(np.allclose(ekf.state[:, :2], np.array([START_XY, START_XY])))
        self.assertTrue(np.allclose(ekf.state[:, 3:], 0.0))


class ContractAndAuthenticityTests(unittest.TestCase):
    def test_base_pose_matches_unified_state_contract(self) -> None:
        ekf = RobotLocalization(3)
        ekf.predict(DT, np.array([[0.1, 0.0, 0.2]] * 3), None)
        pose = ekf.base_pose()
        twist = ekf.base_twist()
        self.assertEqual(pose.shape, (3, 7))          # x, y, z, qx, qy, qz, qw (COORDINATE_SYSTEM S2.4)
        self.assertEqual(twist.shape, (3, 6))
        self.assertTrue(np.allclose(np.linalg.norm(pose[:, 3:7], axis=1), 1.0, atol=1e-12))
        self.assertTrue(np.allclose(pose[:, 3:5], 0.0))          # planar yaw only
        yaw_from_quat = 2.0 * np.arctan2(pose[:, 5], pose[:, 6])
        self.assertTrue(np.allclose(yaw_from_quat, ekf.state[:, 2], atol=1e-12))
        self.assertTrue(np.allclose(twist[:, 5], ekf.state[:, 5]))
        self.assertTrue(np.allclose(twist[:, 2:5], 0.0))         # planar base: no vz/wx/wy
        state = UnifiedState(base_pose=pose, base_twist=twist,
                             joint_pos=np.zeros((3, 6)), joint_vel=np.zeros((3, 6)),
                             racket_contact_pose=np.zeros((3, 7)), racket_contact_twist=np.zeros((3, 6)),
                             shuttle_position=np.zeros((3, 3)), shuttle_velocity=np.zeros((3, 3)),
                             timestamp=0.0)
        self.assertEqual(state.frame, 'court')
        self.assertEqual(state.base_pose.shape, (3, 7))

    def test_every_numeric_parameter_declares_its_origin(self) -> None:
        params = RobotLocalization(1).parameters()
        self.assertTrue(params)
        for name, param in params.items():
            self.assertIsInstance(param, Param, name)
            self.assertTrue(str(param.source).strip(), name)
            if param.status in (AssetStatus.REQUIRES_MEASUREMENT, AssetStatus.UNKNOWN,
                                AssetStatus.REQUIRES_CALIBRATION):
                self.assertIsNone(param.value, name)
        # sensor noise is not identified on the real robot yet -> temporary proxy, never "measured"
        self.assertEqual(params['odom_translation_noise_m'].status,
                         AssetStatus.TEMP_PARAMETERIZED_PROXY)
        self.assertIsNotNone(params['odom_translation_noise_m'].value)
        self.assertEqual(params['odom_translation_noise_measured_m'].status,
                         AssetStatus.REQUIRES_MEASUREMENT)
        self.assertIsNone(params['odom_translation_noise_measured_m'].value)

    def test_invalid_inputs_are_rejected(self) -> None:
        ekf = RobotLocalization(2)
        with self.assertRaises(BrainBoundaryError):
            ekf.predict(DT, np.zeros((2, 2)), None)                        # odom must be (N, 3)
        with self.assertRaises(BrainBoundaryError):
            ekf.predict(DT, np.zeros((3, 3)), None)                        # wrong batch size
        with self.assertRaises(BrainBoundaryError):
            ekf.predict(DT, np.array([[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]), None)
        with self.assertRaises(BrainBoundaryError):
            ekf.predict(-DT, np.zeros((2, 3)), None)                       # dt must be positive
        with self.assertRaises(BrainBoundaryError):
            ekf.update_pose_measurement(np.zeros((2, 3)), np.zeros((2, 3, 3)) - 1.0)   # not PSD
        with self.assertRaises(BrainBoundaryError):
            ekf.update_pose_measurement(np.zeros((1, 3)), np.zeros((1, 3, 3)))        # batch mismatch
        with self.assertRaises(BrainBoundaryError):
            ekf.reset([5])                                                  # env id out of range
        self.assertTrue(np.all(np.isfinite(ekf.state)))


    def test_yaw_quaternion_matches_the_simulation_frame_convention(self) -> None:
        """COORDINATE_SYSTEM.md S2.4: the interface order is (x, y, z, w); the frame module of the
        simulation stores (w, x, y, z) internally - check the two agree on the robot forward axis."""
        try:
            sys.path.insert(0, str(ROOT))
            from simulation.robots.badminton_robot.frames.robot_frames import quat_to_matrix
        except Exception as exc:                                   # pragma: no cover - env dependent
            self.skipTest(f"simulation frame module unavailable: {exc}")
        yaw = 0.7
        ekf = RobotLocalization(1)
        ekf.initialize(np.array([[0.0, 0.0, yaw]]), covariance=np.eye(6) * 1e-12)
        qx, qy, qz, qw = ekf.base_pose()[0][3:7]
        forward = quat_to_matrix(np.array([qw, qx, qy, qz])) @ np.array([1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(forward, [math.cos(yaw), math.sin(yaw), 0.0], atol=1e-12))


if __name__ == '__main__':
    unittest.main(verbosity=2)
