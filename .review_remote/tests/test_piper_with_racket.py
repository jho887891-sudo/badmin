#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import sys
sys.path.insert(0, str(ROOT / "tools"))
CONFIG_PATH = ROOT / "configs" / "racket.yaml"

from build_piper_with_racket import (  # noqa: E402
    compose_link6_tcp_transform,
    load_mount_config,
    validate_mount_config,
)


def measured_mount_cfg():
    cfg = load_mount_config(CONFIG_PATH)
    mount = cfg["piper_attachment"]["mount_transform"]
    mount.update(
        {
            "status": "MEASURED",
            "translation_m": [0.012, -0.003, 0.041],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    )
    return cfg


class PiperMountValidationTests(unittest.TestCase):
    def test_default_mount_is_explicitly_unmeasured(self) -> None:
        cfg = load_mount_config(CONFIG_PATH)
        with self.assertRaisesRegex(ValueError, "mount_transform"):
            validate_mount_config(cfg, final=True)

    def test_measured_identity_orientation_transform(self) -> None:
        cfg = measured_mount_cfg()
        validate_mount_config(cfg, final=True)
        T = compose_link6_tcp_transform(cfg)
        self.assertEqual(T[0][:3], (1.0, 0.0, 0.0))
        self.assertEqual(T[1][:3], (0.0, 1.0, 0.0))
        self.assertEqual(T[2][:3], (0.0, 0.0, 1.0))
        self.assertAlmostEqual(T[0][3], 0.012, places=12)
        self.assertAlmostEqual(T[1][3], -0.003, places=12)
        self.assertAlmostEqual(T[2][3], 0.041, places=12)

    def test_quaternion_is_normalized_and_rotates_axes(self) -> None:
        cfg = measured_mount_cfg()
        s = math.sqrt(0.5)
        cfg["piper_attachment"]["mount_transform"]["quaternion_xyzw"] = [s, 0.0, 0.0, s]
        T = compose_link6_tcp_transform(cfg)
        self.assertAlmostEqual(T[0][0], 1.0, places=12)
        self.assertAlmostEqual(T[1][2], -1.0, places=12)
        self.assertAlmostEqual(T[2][1], 1.0, places=12)

    def test_rejects_zero_quaternion(self) -> None:
        cfg = measured_mount_cfg()
        cfg["piper_attachment"]["mount_transform"]["quaternion_xyzw"] = [0.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "quaternion"):
            validate_mount_config(cfg, final=True)

    def test_config_points_at_frozen_no_gripper_piper_asset(self) -> None:
        cfg = load_mount_config(CONFIG_PATH)
        a = cfg["piper_attachment"]
        self.assertTrue(a["piper_usd_path"].endswith("piper_no_gripper.usd"))
        self.assertEqual(a["piper_destination_prim_path"], "/Robot/Piper")
        self.assertEqual(a["piper_link6_prim_path"], "/Robot/Piper/Geometry/link6")
        self.assertEqual(a["racket_root_prim_path"], "/Robot/Racket")


@unittest.skipUnless(importlib.util.find_spec("pxr") is not None, "pxr/USD not installed in this Python")
class OptionalPiperWrapperUsdTests(unittest.TestCase):
    def test_wrapper_module_exposes_usd_builder(self) -> None:
        import build_piper_with_racket as m
        self.assertTrue(callable(m.build_piper_with_racket))


if __name__ == "__main__":
    unittest.main()
