"""Check wheel heights against the actual pre-fix USD evidence and new frame math.

No Isaac/pxr requirement: historical authored USDA was committed at 459cb12.
"""
from pathlib import Path
import re
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "simulation"))
from p2_chassis_frame_math import (
    wheel_bottom_world_z, historical_nested_wheel_bottom_world_z,
)


def usd_translation(block):
    hit = re.search(r"xformOp:translate = \(([^)]+)\)", block)
    if not hit:
        raise AssertionError("No authored USD translation")
    return tuple(float(v.strip()) for v in hit.group(1).split(","))


class ChassisWorldFrameTests(unittest.TestCase):
    def test_correctly_nested_wheel_clearance_matrix(self):
        cases = ((.300, -.034), (.334, 0), (.350, .016))
        for lift, expected in cases:
            with self.subTest(lift=lift):
                self.assertAlmostEqual(wheel_bottom_world_z(lift), expected, places=9)

    def test_historical_double_lift_values(self):
        cases = ((.300, .2256), (.334, .2936), (.350, .3256))
        for lift, expected in cases:
            with self.subTest(lift=lift):
                self.assertAlmostEqual(historical_nested_wheel_bottom_world_z(lift),
                                       expected, places=9)

    def test_historical_usda_proves_double_lift_without_physx(self):
        base = (REPO / "outputs" / "simulation" / "rebot_b601_rs" /
                "morph_one" / "p2_real_chassis")
        for run_id, lift in (("RC_lift0300_r01", .300), ("RC_lift0334_r01", .334)):
            with self.subTest(run_id=run_id):
                authored = (base / run_id / "authored_chassis.usda").read_text()
                chassis_part = authored.split('def Xform "Chassis"', 1)[1]
                wheel_part = authored.split('def Xform "wheel1"', 1)[1]
                chassis_z = usd_translation(chassis_part)[2]
                wheel_local_z = usd_translation(wheel_part)[2]
                observed_bottom = chassis_z + wheel_local_z - .09
                self.assertAlmostEqual(observed_bottom,
                                       historical_nested_wheel_bottom_world_z(lift),
                                       places=6)
                self.assertGreater(observed_bottom, 0.20)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            wheel_bottom_world_z(.334, wheel_half_height_m=0)
        with self.assertRaises(ValueError):
            wheel_bottom_world_z(float("nan"))


if __name__ == "__main__":
    unittest.main()
