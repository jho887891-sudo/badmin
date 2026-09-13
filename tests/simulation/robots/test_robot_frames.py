# -*- coding: utf-8 -*-
"""Phase 2 (TDD) - robot frame tree tests.

Spec: BADMINTON_ROBOT.md S7 (frame chain), S41 (whole-body pose), S50 (frame tests), S28 (env_origin).
Naming: docs/architecture/COORDINATE_SYSTEM.md S2.5/S2.6.
"""
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'simulation'))

from robots.badminton_robot.badminton_robot_cfg import make_default_robot_cfg  # noqa: E402
from robots.badminton_robot.frames.robot_frames import (  # noqa: E402
    MissingTransformError,
    Transform,
    compose_racket_contact_court,
    court_from_world,
    make_robot_frame_tree,
    quat_from_rpy,
    world_from_court,
)


def quat_equal(a, b, tol=1e-9) -> bool:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return abs(abs(float(np.dot(a, b))) - 1.0) < tol


class TransformTests(unittest.TestCase):
    def test_identity_and_round_trip(self) -> None:
        T = Transform.identity()
        p = np.array([0.3, -0.2, 1.1])
        np.testing.assert_allclose(T.apply(p), p, atol=1e-12)
        np.testing.assert_allclose(T.compose(T.inverse()).matrix(), np.eye(4), atol=1e-12)

    def test_compose_matches_manual_matrix_product(self) -> None:
        A = Transform.from_xyz_quat([1.0, 0.0, 0.0], quat_from_rpy(0.0, 0.0, math.pi / 3))
        B = Transform.from_xyz_quat([0.2, 0.3, 0.4], quat_from_rpy(0.2, -0.1, 0.5))
        np.testing.assert_allclose(A.compose(B).matrix(), A.matrix() @ B.matrix(), atol=1e-12)

    def test_round_trip_error_is_tiny(self) -> None:
        T = Transform.from_xyz_quat([-0.4, 0.7, 1.3], quat_from_rpy(0.3, 0.2, -1.1))
        err = float(np.max(np.abs(T.compose(T.inverse()).matrix() - np.eye(4))))
        self.assertLess(err, 1e-12)

    def test_rejects_non_normalized_quaternion(self) -> None:
        with self.assertRaises(ValueError):
            Transform.from_xyz_quat([0, 0, 0], [1.0, 1.0, 0.0, 0.0])


class RobotFrameTreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = make_default_robot_cfg()
        self.base_pose = ([ -1.60, 0.0, 0.0 ], quat_from_rpy(0.0, 0.0, 0.0))
        self.link6_in_piper = ([0.10, 0.0, 0.45], quat_from_rpy(0.0, 0.0, 0.0))
        self.t_link6_tcp = ([0.0, 0.0, 0.02], quat_from_rpy(0.0, 0.0, 0.0))
        self.t_tcp_contact = ([0.12, 0.0, 0.34], quat_from_rpy(0.0, 0.0, 0.0))
        self.tree = make_robot_frame_tree(
            self.cfg,
            base_pose_xyz_quat=self.base_pose,
            link6_pose_in_piper_xyz_quat=self.link6_in_piper,
            t_link6_tcp_xyz_quat=self.t_link6_tcp,
            t_tcp_contact_xyz_quat=self.t_tcp_contact,
        )

    def test_canonical_frames_exist(self) -> None:
        for name in ('court', 'robot_base', 'piper_base', 'piper_link6',
                     'racket_tcp', 'racket_contact', 'racket_contact_frame',
                     'camera_center', 'camera_left', 'camera_right', 'camera_rig'):
            self.assertIn(name, self.tree.frames, msg=name)

    def test_court_to_robot_base(self) -> None:
        T = self.tree.get_transform('robot_base', 'court')
        np.testing.assert_allclose(T.translation, [-1.60, 0.0, 0.0], atol=1e-12)

    def test_robot_base_to_piper_base_uses_cfg_mount(self) -> None:
        T = self.tree.get_transform('piper_base', 'robot_base')
        np.testing.assert_allclose(T.translation, self.cfg.piper.mount_translation.value, atol=1e-12)

    def test_link6_to_racket_tcp_to_contact_chain(self) -> None:
        tcp = self.tree.get_transform('racket_tcp', 'piper_link6')
        np.testing.assert_allclose(tcp.translation, self.t_link6_tcp[0], atol=1e-12)
        contact = self.tree.get_transform('racket_contact', 'racket_tcp')
        np.testing.assert_allclose(contact.translation, self.t_tcp_contact[0], atol=1e-12)

    def test_racket_contact_frame_is_alias_of_racket_contact(self) -> None:
        a = self.tree.get_transform('racket_contact', 'court')
        b = self.tree.get_transform('racket_contact_frame', 'court')
        np.testing.assert_allclose(a.matrix(), b.matrix(), atol=1e-12)

    def test_camera_left_right_baseline_and_ordering(self) -> None:
        left = self.tree.get_transform('camera_left', 'robot_base').translation
        right = self.tree.get_transform('camera_right', 'robot_base').translation
        self.assertAlmostEqual(float(left[1] - right[1]), self.cfg.stereo_camera.baseline_m.value, places=12)
        self.assertGreater(float(left[1]), float(right[1]))
        center = self.tree.get_transform('camera_center', 'robot_base').translation
        np.testing.assert_allclose(center, self.cfg.stereo_camera.center_offset_robot_xyz, atol=1e-12)

    def test_unknown_frame_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.tree.get_transform('does_not_exist', 'court')

    def test_whole_body_racket_contact_composition_matches_spec_41(self) -> None:
        T = compose_racket_contact_court(
            self.cfg, base_pose_xyz_quat=self.base_pose,
            link6_pose_in_piper_xyz_quat=self.link6_in_piper,
            t_link6_tcp_xyz_quat=self.t_link6_tcp,
            t_tcp_contact_xyz_quat=self.t_tcp_contact)
        manual = (Transform.from_xyz_quat(*self.base_pose)
                  .compose(Transform.from_xyz_quat(self.cfg.piper.mount_translation.value, self.cfg.piper.mount_quaternion.value))
                  .compose(Transform.from_xyz_quat(*self.link6_in_piper))
                  .compose(Transform.from_xyz_quat(*self.t_link6_tcp))
                  .compose(Transform.from_xyz_quat(*self.t_tcp_contact)))
        np.testing.assert_allclose(T.matrix(), manual.matrix(), atol=1e-12)

    def test_missing_measured_adapter_transform_is_explicit(self) -> None:
        cfg = make_default_robot_cfg()
        with self.assertRaises(MissingTransformError) as ctx:
            make_robot_frame_tree(cfg, base_pose_xyz_quat=self.base_pose,
                                  link6_pose_in_piper_xyz_quat=self.link6_in_piper)
        self.assertIn('REQUIRES_MEASUREMENT', str(ctx.exception))


class EnvOriginTests(unittest.TestCase):
    def test_court_from_world_subtracts_env_origin(self) -> None:
        env_origin = [15.0, -15.0, 0.0]
        world = [14.0, -15.5, 1.2]
        court = court_from_world(world, env_origin)
        np.testing.assert_allclose(court, [-1.0, -0.5, 1.2], atol=1e-12)

    def test_world_from_court_round_trip(self) -> None:
        env_origin = [30.0, -15.0, 0.0]
        court = [-1.6, 0.25, 0.8]
        back = court_from_world(world_from_court(court, env_origin), env_origin)
        np.testing.assert_allclose(back, court, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
