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
CONFIG_PATH = ROOT / "configs" / "shuttlecock.yaml"

from build_shuttlecock import (  # noqa: E402
    build_physics_description,
    build_usd,
    compute_mass_properties,
    load_config,
    validate_config,
)


class ShuttleConfigAndPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)
        validate_config(cls.cfg)
        cls.desc = build_physics_description(cls.cfg)
        cls.mass = compute_mass_properties(cls.cfg)

    def test_bwf_ranges_and_feather_count_are_preserved(self) -> None:
        ref = self.cfg["reference"]
        bwf = ref["bwf"]
        self.assertEqual(ref["feathers_count"], 16)
        self.assertEqual(bwf["feather_length_range_m"], [0.062, 0.070])
        self.assertEqual(bwf["skirt_tip_diameter_range_m"], [0.058, 0.068])
        self.assertEqual(bwf["cork_diameter_range_m"], [0.025, 0.028])
        self.assertEqual(bwf["total_mass_range_kg"], [0.00474, 0.00550])

    def test_reference_mass_distribution_matches_literature_profile(self) -> None:
        masses = self.cfg["reference"]["mass_distribution"]
        self.assertAlmostEqual(masses["total_mass_kg"], 0.005, places=12)
        self.assertAlmostEqual(masses["cork_mass_kg"], 0.003, places=12)
        self.assertAlmostEqual(masses["skirt_mass_kg"], 0.002, places=12)
        self.assertAlmostEqual(masses["cork_mass_kg"] + masses["skirt_mass_kg"], masses["total_mass_kg"], places=12)

    def test_skirt_tip_diameter_is_derived_from_30_cm2_reference_area(self) -> None:
        area = self.cfg["reference"]["selected_geometry"]["reference_cross_section_area_m2"]
        expected_diameter = 2.0 * math.sqrt(area / math.pi)
        self.assertAlmostEqual(self.desc["skirt_tip_radius_m"] * 2.0, expected_diameter, places=12)
        self.assertGreaterEqual(expected_diameter, 0.058)
        self.assertLessEqual(expected_diameter, 0.068)

    def test_cork_proxy_is_bwf_compliant_hemisphere(self) -> None:
        cork = self.desc["cork"]
        self.assertAlmostEqual(cork["radius_m"] * 2.0, 0.0265, places=12)
        self.assertAlmostEqual(cork["z_min_m"], -0.01325, places=12)
        self.assertAlmostEqual(cork["z_max_m"], 0.0, places=12)

    def test_skirt_collision_is_open_shell_not_solid_cone(self) -> None:
        segments = self.desc["skirt_collision_segments"]
        self.assertEqual(len(segments), 16)
        for seg in segments:
            self.assertGreater(seg["root_inner_radius_m"], 0.0)
            self.assertGreater(seg["tip_inner_radius_m"], seg["root_inner_radius_m"])
            self.assertGreater(seg["root_outer_radius_m"], seg["root_inner_radius_m"])
            self.assertGreater(seg["tip_outer_radius_m"], seg["tip_inner_radius_m"])

    def test_mass_properties_are_finite_positive_and_cork_biased(self) -> None:
        self.assertAlmostEqual(self.mass["mass_kg"], 0.005, places=12)
        com_z = self.mass["center_of_mass_m"][2]
        self.assertTrue(math.isfinite(com_z))
        self.assertGreater(com_z, self.mass["component_centers_m"]["cork"][2])
        self.assertLess(com_z, self.mass["component_centers_m"]["skirt"][2])
        self.assertLess(abs(com_z), 0.02)
        for value in self.mass["diagonal_inertia_kg_m2"]:
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_axisymmetric_inertia_has_equal_transverse_components(self) -> None:
        ix, iy, iz = self.mass["diagonal_inertia_kg_m2"]
        self.assertAlmostEqual(ix, iy, places=15)
        self.assertNotAlmostEqual(ix, iz, places=12)

    def test_contact_pairs_are_explicitly_unfitted_not_fake_numbers(self) -> None:
        contact = self.cfg["contact_pair_parameters"]
        self.assertEqual(contact["status"], "REQUIRES_PAIR_CALIBRATION")
        self.assertIsNone(contact["shuttle_ground"])
        self.assertIsNone(contact["shuttle_net"])
        self.assertIsNone(contact["shuttle_racket"])

    def test_visual_source_is_open_license_reference(self) -> None:
        visual = self.cfg["visual"]
        self.assertEqual(visual["mode"], "EXTERNAL_REFERENCE")
        self.assertTrue(visual["required_for_final"])
        self.assertEqual(visual["author"], "game_travel")
        self.assertIn("CC Attribution", visual["license"])
        self.assertIn("795e4cca2e544d7ab86a8e5c7ded2362", visual["source_url"])

    def test_description_contains_no_nonfinite_numbers(self) -> None:
        def walk(value):
            if isinstance(value, dict):
                for v in value.values():
                    yield from walk(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    yield from walk(v)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                yield float(value)

        for number in walk(self.desc):
            self.assertTrue(math.isfinite(number), number)


class ShuttleValidationTests(unittest.TestCase):
    def test_rejects_mass_split_mismatch(self) -> None:
        cfg = load_config(CONFIG_PATH)
        cfg["reference"]["mass_distribution"]["skirt_mass_kg"] = 0.003
        with self.assertRaisesRegex(ValueError, "mass split"):
            validate_config(cfg)

    def test_rejects_selected_geometry_outside_bwf_range(self) -> None:
        cfg = load_config(CONFIG_PATH)
        cfg["reference"]["selected_geometry"]["cork_diameter_m"] = 0.040
        with self.assertRaisesRegex(ValueError, "BWF"):
            validate_config(cfg)

    def test_rejects_drag_k_inconsistent_with_aerodynamic_length(self) -> None:
        cfg = load_config(CONFIG_PATH)
        cfg["reference"]["aerodynamics"]["quadratic_drag_k_per_m"] = 0.2
        with self.assertRaisesRegex(ValueError, "1 / aerodynamic_length"):
            validate_config(cfg)


@unittest.skipUnless(importlib.util.find_spec("pxr") is not None, "pxr/USD not installed in this Python")
class OptionalUsdAuthoringTests(unittest.TestCase):
    def test_temp_debug_usd_contains_required_prims(self) -> None:
        from pxr import Usd

        cfg = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "shuttlecock.usda"
            build_usd(cfg, out, allow_temp_visual=True)
            stage = Usd.Stage.Open(str(out))
            self.assertIsNotNone(stage)
            for path in [
                "/Shuttlecock",
                "/Shuttlecock/Visual",
                "/Shuttlecock/Physics",
                "/Shuttlecock/Physics/CorkCollider",
                "/Shuttlecock/Physics/SkirtColliders",
                "/Shuttlecock/Frames",
                "/Shuttlecock/Frames/shuttle_com",
                "/Shuttlecock/Frames/cork_tip",
                "/Shuttlecock/Frames/skirt_axis",
            ]:
                self.assertTrue(stage.GetPrimAtPath(path).IsValid(), path)


if __name__ == "__main__":
    unittest.main()
