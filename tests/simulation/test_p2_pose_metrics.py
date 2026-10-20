"""Independent quaternion-order tests that require no Isaac Sim installation."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'simulation'))
from p2_pose_metrics import Pose, angle, mount_pose_error, world_pose_error


class PoseMetricsTest(unittest.TestCase):
    def test_identity_both_sources(self):
        usd = Pose.from_usd_wxyz((0, 0, 0), (1, 0, 0, 0))
        physx = Pose.from_isaac_body_xyzw((0, 0, 0), (0, 0, 0, 1))
        self.assertEqual(world_pose_error(usd, physx),
                         {'world_position_error_m': 0, 'world_rotation_error_rad': 0})

    def test_known_nonidentity_yaw_and_roll(self):
        for (axis, degrees) in [('yaw', 90), ('roll', 60)]:
            c, s = math.cos(math.radians(degrees / 2)), math.sin(math.radians(degrees / 2))
            q = (c, 0, 0, s) if axis == 'yaw' else (c, s, 0, 0)
            usd = Pose.from_usd_wxyz((1, 2, 3), q)
            physx = Pose.from_isaac_body_xyzw((1, 2, 3), (q[1], q[2], q[3], q[0]))
            result = world_pose_error(usd, physx)
            self.assertLess(result['world_rotation_error_rad'], 1e-7)
            self.assertAlmostEqual(result['world_position_error_m'], 0)

    def test_mount_in_chassis_local_frame_with_world_yaw(self):
        c, s = math.sqrt(.5), math.sqrt(.5)
        chassis = Pose.from_usd_wxyz((0, 0, 0), (c, 0, 0, s))
        base = Pose.from_isaac_body_xyzw((0.05, 0.12, 0.2), (0, 0, s, c))
        error = mount_pose_error(chassis, base, (0.12, -0.05, 0.2))
        self.assertLess(error['mount_position_error_m'], 1e-12)
        self.assertLess(error['mount_rotation_error_rad'], 1e-7)

    def test_world_offset_is_not_hidden_by_mount_consistency(self):
        chassis = Pose.from_isaac_body_xyzw((0, 0, .0379), (0, 0, 0, 1))
        base = Pose.from_isaac_body_xyzw((.12, -.05, .2379), (0, 0, 0, 1))
        self.assertLess(mount_pose_error(chassis, base, (.12, -.05, .2))['mount_position_error_m'], 1e-12)
        usd_chassis = Pose.from_usd_wxyz((0, 0, 0), (1, 0, 0, 0))
        self.assertAlmostEqual(world_pose_error(usd_chassis, chassis)['world_position_error_m'], .0379)

    def test_archived_p2_reset0_has_good_mount_but_bad_absolute_pose(self):
        # Evidence: e947bad1 / p2_b0853ef/p2_joint-local.json (reset0).
        authored = Pose.from_usd_wxyz((0, 0, 0), (1, 0, 0, 0))
        q = (0.007430156, -0.000311807, 0.00041042, 0.999972224)
        chassis = Pose.from_isaac_body_xyzw((.000044573, .000191213, .037923977), q)
        base = Pose.from_isaac_body_xyzw((.11996226, -.052677333, .237234399), q)
        absolute = world_pose_error(authored, chassis)
        mount = mount_pose_error(chassis, base, (.119999997, -.050000001, .200000003))
        self.assertAlmostEqual(absolute['world_position_error_m'], .037924485239, places=7)
        self.assertAlmostEqual(absolute['world_rotation_error_rad'], .014896162793, places=6)
        self.assertLess(mount['mount_position_error_m'], 1e-6)
        self.assertEqual(mount['mount_rotation_error_rad'], 0)

    def test_signed_quaternion_invariance(self):
        self.assertEqual(angle((1, 0, 0, 0), (-1, 0, 0, 0)), 0)

    def test_invalid_quaternion_and_nonfinite(self):
        with self.assertRaises(ValueError):
            Pose.from_usd_wxyz((0, 0, 0), (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            Pose.from_isaac_body_xyzw((0, 0, float('nan')), (0, 0, 0, 1))


if __name__ == '__main__':
    unittest.main()
