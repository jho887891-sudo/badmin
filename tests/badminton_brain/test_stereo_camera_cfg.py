# -*- coding: utf-8 -*-
"""CPU-only tests for the Isaac Lab stereo camera configuration builder."""
from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from simulation.badminton_scene.stereo_camera_cfg import (  # noqa: E402
    build_stereo_camera_sensor_specs,
)
from simulation.badminton_scene.scene_layout import CAMERA  # noqa: E402
from badminton_brain.perception.stereo_geometry import forward_camera_rotation  # noqa: E402


class FakeCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeCameraCfg(FakeCfg):
    class OffsetCfg(FakeCfg):
        pass


class FakePinholeCameraCfg(FakeCfg):
    pass


class FakeCamera:
    pass


def fake_isaaclab_modules():
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    sim = types.ModuleType("isaaclab.sim")
    sim.PinholeCameraCfg = FakePinholeCameraCfg
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.Camera = FakeCamera
    sensors.CameraCfg = FakeCameraCfg
    isaaclab.sim = sim
    isaaclab.sensors = sensors
    return {
        "isaaclab": isaaclab,
        "isaaclab.sim": sim,
        "isaaclab.sensors": sensors,
    }


class StereoCameraConfigTests(unittest.TestCase):
    def test_builds_left_and_right_rgbd_cameras_from_frozen_rig_layout(self):
        with patch.dict(sys.modules, fake_isaaclab_modules()):
            specs = build_stereo_camera_sensor_specs(
                width=1280, height=720, update_period=1.0 / 60.0
            )

        self.assertEqual([spec[0] for spec in specs], ["camera_left", "camera_right"])
        left = specs[0][1]
        right = specs[1][1]
        self.assertIs(specs[0][2], FakeCamera)
        self.assertIs(specs[1][2], FakeCamera)

        self.assertEqual(left.prim_path, "/World/envs/env_0/CameraLeft")
        self.assertEqual(right.prim_path, "/World/envs/env_0/CameraRight")
        for cfg in (left, right):
            self.assertEqual(cfg.width, 1280)
            self.assertEqual(cfg.height, 720)
            self.assertAlmostEqual(cfg.update_period, 1.0 / 60.0)
            self.assertEqual(cfg.data_types, ["rgb", "distance_to_image_plane"])
            self.assertEqual(cfg.offset.convention, "world")
            self.assertEqual(cfg.spawn.focal_length, 24.0)
            self.assertEqual(cfg.spawn.clipping_range, (0.05, 20.0))

        self.assertEqual(left.offset.pos, (-1.4, 0.145, 1.2))
        self.assertEqual(right.offset.pos, (-1.4, -0.145, 1.2))
        pitch_rad = math.radians(float(CAMERA["pitch_deg"]))
        expected_rot = (0.0, math.sin(pitch_rad / 2.0), 0.0, math.cos(pitch_rad / 2.0))
        for cfg in (left, right):
            for actual, expected in zip(cfg.offset.rot, expected_rot):
                self.assertAlmostEqual(actual, expected, places=12)

        # Isaac Lab convention="world" uses +X forward / +Z up. Its Y rotation
        # must point the optical axis in the same direction as the frozen court
        # geometry helper, which defines the signed pitch convention.
        theta = 2.0 * math.atan2(left.offset.rot[1], left.offset.rot[3])
        camera_forward_world = (math.cos(theta), 0.0, -math.sin(theta))
        geometry_forward_world = forward_camera_rotation(float(CAMERA["pitch_deg"]))[:, 2]
        for actual, expected in zip(camera_forward_world, geometry_forward_world):
            self.assertAlmostEqual(actual, float(expected), places=12)

    def test_rejects_invalid_resolution_before_importing_isaaclab(self):
        with self.assertRaisesRegex(ValueError, "at least 32"):
            build_stereo_camera_sensor_specs(width=0, height=720)

    def test_rejects_negative_update_period(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_stereo_camera_sensor_specs(update_period=-1.0)


if __name__ == "__main__":
    unittest.main()
