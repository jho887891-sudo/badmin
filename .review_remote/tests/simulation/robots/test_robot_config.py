# -*- coding: utf-8 -*-
"""Phase 1 (TDD/RED) - BadmintonRobot config schema tests.

Spec: docs/simulation/BADMINTON_ROBOT.md  (S12 TEMP policy, S32 cfg, S33 validation, S49 tests)
Frame names: docs/architecture/COORDINATE_SYSTEM.md (S2.5 canonical frames, S2.6 T_A_B)
"""
from __future__ import annotations
import dataclasses
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'simulation'))

from robots.badminton_robot.badminton_robot_cfg import (  # noqa: E402
    AssetStatus,
    BadmintonRobotCfg,
    DriveMode,
    MorphOneCfg,
    MorphOneGeometry,
    MorphOneMassProperties,
    Param,
    PiperCfg,
    RacketCfg,
    StereoCameraCfg,
    STEER_DRIVE_JOINT_NAMES,
    WheelId,
    make_default_robot_cfg,
)
from robots.badminton_robot.validation.robot_validator import (  # noqa: E402
    ValidationError,
    validate_robot_cfg,
)


class AssetStatusTests(unittest.TestCase):
    def test_status_enum_covers_spec_section_4(self) -> None:
        expected = {
            'VERIFIED_OFFICIAL', 'VERIFIED_MEASURED', 'DERIVED_FROM_MEASUREMENT',
            'TRACEABLE_REFERENCE', 'TEMP_PARAMETERIZED_PROXY', 'UNKNOWN',
            'REQUIRES_MEASUREMENT', 'REQUIRES_CALIBRATION',
        }
        self.assertEqual({s.name for s in AssetStatus}, expected)

    def test_param_requires_status_and_keeps_source(self) -> None:
        p = Param(value=None, status=AssetStatus.REQUIRES_MEASUREMENT, source='hardware not measured')
        self.assertIsNone(p.value)
        self.assertIs(p.status, AssetStatus.REQUIRES_MEASUREMENT)
        self.assertTrue(p.source)
        with self.assertRaises(ValueError):
            Param(value=None, status=AssetStatus.REQUIRES_MEASUREMENT)  # missing source

    def test_temp_value_must_be_explicitly_marked(self) -> None:
        ok = Param(value=0.30, status=AssetStatus.TEMP_PARAMETERIZED_PROXY, source='engineering baseline')
        self.assertEqual(ok.value, 0.30)
        with self.assertRaises(ValueError):
            Param(value=0.30, status=AssetStatus.REQUIRES_MEASUREMENT, source='x')  # value without measurement


class MorphOneCfgTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = make_default_robot_cfg()

    def test_topology_is_four_steer_four_drive(self) -> None:
        self.assertEqual({w.name for w in WheelId}, {'FL', 'FR', 'RL', 'RR'})
        self.assertEqual(len(STEER_DRIVE_JOINT_NAMES), 8)
        for wid in WheelId:
            self.assertIn(f'{wid.value.lower()}_steer', STEER_DRIVE_JOINT_NAMES)
            self.assertIn(f'{wid.value.lower()}_drive', STEER_DRIVE_JOINT_NAMES)

    def test_morph_one_defaults_are_never_fake_official_values(self) -> None:
        m: MorphOneCfg = self.cfg.morph_one
        self.assertIs(m.asset_status, AssetStatus.TEMP_PARAMETERIZED_PROXY)
        for field in ('length_m', 'width_m', 'wheel_radius_m', 'wheel_width_m'):
            p: Param = getattr(m.geometry, field)
            self.assertIn(p.status, (AssetStatus.REQUIRES_MEASUREMENT, AssetStatus.TEMP_PARAMETERIZED_PROXY))
        for field in ('total_mass_kg', 'com_robot', 'inertia_robot'):
            p = getattr(m.mass_properties, field)
            self.assertIs(p.status, AssetStatus.REQUIRES_MEASUREMENT)
            self.assertIsNone(p.value)

    def test_wheel_positions_stay_unknown_until_measured(self) -> None:
        wpos = self.cfg.morph_one.geometry.wheel_positions_robot
        self.assertIs(wpos.status, AssetStatus.REQUIRES_MEASUREMENT)
        self.assertIsNone(wpos.value)

    def test_steering_limits_present(self) -> None:
        m = self.cfg.morph_one
        self.assertLess(m.max_steer_angle_rad.value, math.pi + 1e-9)
        self.assertGreater(m.max_steer_rate_rad_s.value, 0.0)
        self.assertGreater(m.max_wheel_speed_rad_s.value, 0.0)


class PiperCfgTests(unittest.TestCase):
    def test_frozen_joint_order_and_count(self) -> None:
        p: PiperCfg = make_default_robot_cfg().piper
        self.assertEqual(p.joint_names, ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'))
        self.assertEqual(p.num_joints, 6)

    def test_mount_transform_is_temp_or_unmeasured_but_composable(self) -> None:
        p = make_default_robot_cfg().piper
        self.assertIn(p.mount_translation.status,
                      (AssetStatus.TEMP_PARAMETERIZED_PROXY, AssetStatus.REQUIRES_MEASUREMENT))
        self.assertEqual(len(p.mount_translation.value), 3)
        self.assertAlmostEqual(sum(v * v for v in p.mount_quaternion.value), 1.0, places=9)


class RacketCfgTests(unittest.TestCase):
    def test_face_normal_and_unmeasured_mount(self) -> None:
        r: RacketCfg = make_default_robot_cfg().racket
        self.assertEqual(tuple(r.face_normal_local), (1.0, 0.0, 0.0))
        self.assertIs(r.t_link6_tcp.status, AssetStatus.REQUIRES_MEASUREMENT)
        self.assertIsNone(r.t_link6_tcp.value)
        self.assertIs(r.t_tcp_contact.status, AssetStatus.REQUIRES_MEASUREMENT)


class StereoCameraCfgTests(unittest.TestCase):
    def test_baseline_and_left_right_ordering(self) -> None:
        c: StereoCameraCfg = make_default_robot_cfg().stereo_camera
        self.assertAlmostEqual(c.baseline_m.value, 0.29, places=9)
        self.assertAlmostEqual(c.left_y_m - c.right_y_m, c.baseline_m.value, places=9)
        self.assertGreater(c.left_y_m, c.right_y_m)
        self.assertIs(c.asset_status, AssetStatus.TEMP_PARAMETERIZED_PROXY)


class DriveModeTests(unittest.TestCase):
    def test_two_supported_modes(self) -> None:
        self.assertEqual({m.name for m in DriveMode},
                         {'BODY_TWIST_ACTUATOR', 'STEER_DRIVE_WHEEL_MODEL'})
        self.assertIs(make_default_robot_cfg().drive_mode, DriveMode.BODY_TWIST_ACTUATOR)


class ValidatorTests(unittest.TestCase):
    def test_development_mode_allows_temp_but_reports_warnings(self) -> None:
        cfg = make_default_robot_cfg()
        report = validate_robot_cfg(cfg, mode='development')
        self.assertTrue(report.ok)
        self.assertTrue(report.warnings)
        self.assertTrue(any('TEMP' in w or 'REQUIRES_MEASUREMENT' in w for w in report.warnings))

    def test_final_mode_fails_on_temp_and_unmeasured(self) -> None:
        cfg = make_default_robot_cfg()
        with self.assertRaises(ValidationError) as ctx:
            validate_robot_cfg(cfg, mode='final')
        self.assertIn('final', str(ctx.exception).lower())

    def test_rejects_bad_quaternion(self) -> None:
        cfg = make_default_robot_cfg()
        cfg.piper.mount_quaternion = Param(value=(1.0, 1.0, 0.0, 0.0),
                                           status=AssetStatus.TEMP_PARAMETERIZED_PROXY, source='bad')
        with self.assertRaises(ValidationError):
            validate_robot_cfg(cfg, mode='development')

    def test_rejects_wrong_joint_count(self) -> None:
        cfg = make_default_robot_cfg()
        cfg.piper.joint_names = ('joint1', 'joint2')
        with self.assertRaises(ValidationError):
            validate_robot_cfg(cfg, mode='development')

    def test_rejects_nonpositive_baseline(self) -> None:
        cfg = make_default_robot_cfg()
        cfg.stereo_camera.baseline_m = Param(value=0.0, status=AssetStatus.TEMP_PARAMETERIZED_PROXY, source='bad')
        with self.assertRaises(ValidationError):
            validate_robot_cfg(cfg, mode='development')

    def test_initial_base_pose_is_engineering_config_not_hardcoded(self) -> None:
        cfg = make_default_robot_cfg()
        self.assertEqual(tuple(cfg.initial_base_pose_xyz), (-1.60, 0.0, 0.0))
        self.assertIn('engineering', cfg.initial_base_pose_source.lower())


if __name__ == '__main__':
    unittest.main()
