"""P3 offline negative tests: Z omission / wrong direction must never PASS."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "simulation"))
from p3_motion_gate import directional_displacement_gate as check


class P3DirectedTranslationGate(unittest.TestCase):
    def test_z_10cm_realized(self):
        outcome = check((0, 0, 0.10), (0, 0, .4), (0, 0, .500000024))
        self.assertTrue(outcome["pass"])

    def test_z_10cm_missing_must_fail(self):
        outcome = check((0, 0, .10), (0, 0, .4), (0, 0, .4))
        self.assertFalse(outcome["pass"])

    def test_z_command_cannot_be_satisfied_by_x_motion(self):
        outcome = check((0, 0, .10), (0, 0, .4), (.12, 0, .4))
        self.assertFalse(outcome["pass"])

    def test_xy_command_wrong_direction_fails(self):
        outcome = check((.3, 0, 0), (0, 0, .4), (-.3, 0, .4))
        self.assertFalse(outcome["pass"])
        self.assertLess(outcome["achieved_ratio"], 0)

    def test_xy_command_meets_90_percent(self):
        self.assertTrue(check((.3, 0, 0), (0, 0, .4),
                              (.299999984, 0, .4))["pass"])

    def test_compound_command_xyz_contributes(self):
        outcome = check((.20, -.15, .05), (0, 0, .4), (.2, -.15, .45))
        self.assertTrue(outcome["pass"])
        self.assertAlmostEqual(outcome["achieved_ratio"], 1)

    def test_compound_off_axis_magnitude_fails(self):
        outcome = check((.20, -.15, .05), (0, 0, .4), (0, .30, .4))
        self.assertFalse(outcome["pass"])

    def test_missing_any_component_fails(self):
        with self.assertRaises(ValueError):
            check((.2, .1), (0, 0, 0), (0, 0, 0))

    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):
            check((.2, float("nan"), 0), (0, 0, 0), (0, 0, 0))

    def test_zero_command_not_applicable(self):
        self.assertIsNone(check((0, 0, 0), (0, 0, .4), (0, 0, .4)))

    def test_just_under_threshold_fails(self):
        self.assertFalse(check((.1, 0, 0), (0, 0, .4),
                               (.089, 0, .4))["pass"])

    def test_bad_ratio_fails(self):
        with self.assertRaises(ValueError):
            check((.1, 0, 0), (0, 0, 0), (.1, 0, 0), ratio=1.1)


if __name__ == "__main__":
    unittest.main()
