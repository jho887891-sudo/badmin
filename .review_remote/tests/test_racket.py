#!/usr/bin/env python3
from __future__ import annotations

import copy
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

from build_racket import (  # noqa: E402
    build_physics_description,
    build_usd,
    compute_mass_properties,
    load_config,
    tcp_to_contact_transform,
    validate_config,
)


def measured_cfg():
    cfg = load_config(CONFIG_PATH)
    m = cfg["measured_unit"]
    m.update(
        {
            "status": "MEASURED",
            "installed_mass_kg": 0.0912,
            "center_of_mass_from_tcp_m": [0.0, 0.0, 0.287],
            "diagonal_inertia_kg_m2": [0.00321, 0.00318, 0.000145],
            "handle_size_x_m": 0.030,
            "handle_size_y_m": 0.026,
            "handle_length_m": 0.205,
            "head_outer_length_m": 0.292,
            "head_outer_width_m": 0.228,
            "stringbed_length_m": 0.265,
            "stringbed_width_m": 0.205,
            "frame_tube_diameter_m": 0.010,
            "stringbed_thickness_m": 0.003,
        }
    )
    return cfg


class RacketReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)
        validate_config(cls.cfg, final=False)

    def test_bwf_legal_envelope_is_exact(self) -> None:
        bwf = self.cfg["reference"]["bwf"]
        self.assertAlmostEqual(bwf["max_overall_length_m"], 0.680, places=12)
        self.assertAlmostEqual(bwf["max_overall_width_m"], 0.230, places=12)
        self.assertAlmostEqual(bwf["max_stringed_area_length_m"], 0.280, places=12)
        self.assertAlmostEqual(bwf["max_stringed_area_width_m"], 0.220, places=12)

    def test_traceable_product_fields_match_victor_reference(self) -> None:
        p = self.cfg["reference"]["product"]
        self.assertEqual(p["manufacturer"], "VICTOR")
        self.assertEqual(p["model"], "THRUSTER Onigiri")
        self.assertEqual(p["item_code"], "TK-ONIGIRI")
        self.assertAlmostEqual(p["overall_length_m"], 0.675, places=12)
        self.assertAlmostEqual(p["shaft_diameter_m"], 0.0064, places=12)
        self.assertEqual(p["weight_class"], "4U")
        self.assertEqual(p["unstrung_mass_range_kg"], [0.0800, 0.0849])

    def test_default_config_refuses_to_claim_unmeasured_unit_is_final(self) -> None:
        with self.assertRaisesRegex(ValueError, "measured_unit"):
            validate_config(self.cfg, final=True)

    def test_final_visual_is_external_cc_attribution_asset(self) -> None:
        visual = self.cfg["visual"]
        self.assertEqual(visual["mode"], "EXTERNAL_REFERENCE")
        self.assertTrue(visual["required_for_final"])
        self.assertEqual(visual["author"], "game_travel")
        self.assertIn("CC Attribution", visual["license"])
        self.assertIn("795e4cca2e544d7ab86a8e5c7ded2362", visual["source_url"])


class RacketMeasuredPhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = measured_cfg()
        validate_config(self.cfg, final=True)
        self.desc = build_physics_description(self.cfg)
        self.mass = compute_mass_properties(self.cfg)

    def test_measured_geometry_respects_bwf_envelope(self) -> None:
        self.assertAlmostEqual(self.desc["overall_length_m"], 0.675, places=12)
        self.assertLessEqual(self.desc["head"]["outer_width_m"], 0.230)
        self.assertLessEqual(self.desc["stringbed"]["length_m"], 0.280)
        self.assertLessEqual(self.desc["stringbed"]["width_m"], 0.220)

    def test_head_center_is_derived_from_total_length_not_hand_tuned(self) -> None:
        head = self.desc["head"]
        expected = 0.675 - 0.292 / 2.0
        self.assertAlmostEqual(head["center_z_m"], expected, places=12)
        self.assertAlmostEqual(head["top_z_m"], 0.675, places=12)
        self.assertAlmostEqual(head["bottom_z_m"], 0.675 - 0.292, places=12)

    def test_frame_is_closed_segmented_ellipse(self) -> None:
        segs = self.desc["head"]["frame_segments"]
        self.assertEqual(len(segs), self.cfg["physics_proxy"]["frame_segments"])
        for i, seg in enumerate(segs):
            nxt = segs[(i + 1) % len(segs)]
            self.assertEqual(tuple(seg["p1_m"]), tuple(nxt["p0_m"]))
            self.assertGreater(seg["radius_m"], 0.0)

    def test_stringbed_is_independent_thin_ellipse(self) -> None:
        s = self.desc["stringbed"]
        self.assertGreaterEqual(len(s["outline_yz_m"]), 16)
        self.assertGreater(s["thickness_m"], 0.0)
        self.assertLess(s["length_m"], self.desc["head"]["outer_length_m"])
        self.assertLess(s["width_m"], self.desc["head"]["outer_width_m"])

    def test_tcp_to_contact_frame_uses_stringbed_center(self) -> None:
        T = tcp_to_contact_transform(self.cfg)
        self.assertEqual(len(T), 4)
        self.assertAlmostEqual(T[0][3], 0.0, places=12)
        self.assertAlmostEqual(T[1][3], 0.0, places=12)
        self.assertAlmostEqual(T[2][3], self.desc["stringbed"]["center_z_m"], places=12)
        self.assertEqual(T[0][:3], (1.0, 0.0, 0.0))
        self.assertEqual(T[1][:3], (0.0, 1.0, 0.0))
        self.assertEqual(T[2][:3], (0.0, 0.0, 1.0))

    def test_mass_properties_are_explicit_measured_values(self) -> None:
        self.assertAlmostEqual(self.mass["mass_kg"], 0.0912, places=12)
        self.assertEqual(self.mass["center_of_mass_m"], (0.0, 0.0, 0.287))
        self.assertEqual(self.mass["diagonal_inertia_kg_m2"], (0.00321, 0.00318, 0.000145))
        self.assertEqual(self.mass["provenance"], "MEASURED_UNIT")

    def test_face_normal_is_positive_local_x(self) -> None:
        self.assertEqual(self.desc["frames"]["face_normal_local"], (1.0, 0.0, 0.0))


class RacketValidationTests(unittest.TestCase):
    def test_rejects_head_width_over_bwf_limit(self) -> None:
        cfg = measured_cfg()
        cfg["measured_unit"]["head_outer_width_m"] = 0.240
        with self.assertRaisesRegex(ValueError, "BWF"):
            validate_config(cfg, final=True)

    def test_rejects_stringbed_over_bwf_limit(self) -> None:
        cfg = measured_cfg()
        cfg["measured_unit"]["stringbed_width_m"] = 0.225
        with self.assertRaisesRegex(ValueError, "BWF"):
            validate_config(cfg, final=True)

    def test_rejects_geometry_that_cannot_fit_handle_shaft_and_head(self) -> None:
        cfg = measured_cfg()
        cfg["measured_unit"]["handle_length_m"] = 0.500
        with self.assertRaisesRegex(ValueError, "handle"):
            validate_config(cfg, final=True)

    def test_rejects_nonpositive_inertia(self) -> None:
        cfg = measured_cfg()
        cfg["measured_unit"]["diagonal_inertia_kg_m2"][2] = 0.0
        with self.assertRaisesRegex(ValueError, "inertia"):
            validate_config(cfg, final=True)


@unittest.skipUnless(importlib.util.find_spec("pxr") is not None, "pxr/USD not installed in this Python")
class OptionalRacketUsdTests(unittest.TestCase):
    def test_usd_contains_required_physics_and_frame_prims(self) -> None:
        from pxr import Usd

        cfg = measured_cfg()
        cfg["visual"]["required_for_final"] = False
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "racket.usda"
            build_usd(cfg, out, allow_temp_visual=True)
            stage = Usd.Stage.Open(str(out))
            self.assertIsNotNone(stage)
            for path in [
                "/Racket",
                "/Racket/Visual",
                "/Racket/Physics",
                "/Racket/Physics/HandleCollider",
                "/Racket/Physics/ShaftCollider",
                "/Racket/Physics/HeadFrameColliders",
                "/Racket/Physics/StringBedCollider",
                "/Racket/Frames/racket_tcp",
                "/Racket/Frames/racket_contact_frame",
            ]:
                self.assertTrue(stage.GetPrimAtPath(path).IsValid(), path)


if __name__ == "__main__":
    unittest.main()
