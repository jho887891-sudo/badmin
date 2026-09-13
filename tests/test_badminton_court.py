#!/usr/bin/env python3
"""Tests for the parameterized badminton court generator.

These tests are intentionally split into:
1) pure-Python geometry tests, runnable outside Isaac Sim;
2) an optional USD stage test, enabled automatically when ``pxr`` exists.

Run:
    python test_badminton_court.py -v
or:
    pytest -q test_badminton_court.py
"""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "court.yaml"

from build_badminton_court import (  # noqa: E402
    build_geometry,
    build_usd,
    load_config,
    validate_config,
)


class CourtGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(CONFIG_PATH)
        validate_config(cls.cfg)
        cls.geo = build_geometry(cls.cfg)

    def test_project_court_dimensions(self) -> None:
        court = self.cfg["court"]
        self.assertAlmostEqual(court["length_m"], 13.40, places=9)
        self.assertAlmostEqual(court["width_m"], 6.10, places=9)
        self.assertAlmostEqual(court["singles_width_m"], 5.18, places=9)
        self.assertAlmostEqual(court["line_width_m"], 0.04, places=9)

    def test_outer_boundary_lines_are_at_project_coordinates(self) -> None:
        lines = {line["name"]: line for line in self.geo["lines"]}
        self.assertAlmostEqual(lines["Baseline_Robot"]["center"][0], -6.70, places=9)
        self.assertAlmostEqual(lines["Baseline_Opponent"]["center"][0], 6.70, places=9)
        self.assertAlmostEqual(lines["DoublesSideline_Left"]["center"][1], 3.05, places=9)
        self.assertAlmostEqual(lines["DoublesSideline_Right"]["center"][1], -3.05, places=9)

    def test_singles_and_service_lines_are_at_project_coordinates(self) -> None:
        lines = {line["name"]: line for line in self.geo["lines"]}
        self.assertAlmostEqual(lines["SinglesSideline_Left"]["center"][1], 2.59, places=9)
        self.assertAlmostEqual(lines["SinglesSideline_Right"]["center"][1], -2.59, places=9)
        self.assertAlmostEqual(lines["ShortService_Robot"]["center"][0], -1.98, places=9)
        self.assertAlmostEqual(lines["ShortService_Opponent"]["center"][0], 1.98, places=9)
        self.assertAlmostEqual(lines["DoublesLongService_Robot"]["center"][0], -5.94, places=9)
        self.assertAlmostEqual(lines["DoublesLongService_Opponent"]["center"][0], 5.94, places=9)

    def test_center_lines_only_cover_service_courts(self) -> None:
        lines = {line["name"]: line for line in self.geo["lines"]}
        robot = lines["CenterLine_Robot"]
        opponent = lines["CenterLine_Opponent"]

        expected_length = 6.70 - 1.98
        self.assertAlmostEqual(robot["size"][0], expected_length, places=9)
        self.assertAlmostEqual(opponent["size"][0], expected_length, places=9)
        self.assertAlmostEqual(robot["center"][0], -(6.70 + 1.98) / 2.0, places=9)
        self.assertAlmostEqual(opponent["center"][0], (6.70 + 1.98) / 2.0, places=9)

    def test_floor_visual_and_ground_collider_have_top_surface_at_z_zero(self) -> None:
        floor = self.geo["floor_visual"]
        ground = self.geo["ground_collider"]
        floor_top = floor["center"][2] + floor["size"][2] / 2.0
        ground_top = ground["center"][2] + ground["size"][2] / 2.0
        self.assertAlmostEqual(floor_top, 0.0, places=9)
        self.assertAlmostEqual(ground_top, 0.0, places=9)

    def test_net_has_finite_width_and_never_creates_ground_to_net_wall(self) -> None:
        net = self.geo["net"]
        segments = net["segments"]
        self.assertGreater(len(segments), 1)

        min_bottom = min(seg["z_bottom"] for seg in segments)
        max_abs_y = max(abs(seg["center"][1]) + seg["size"][1] / 2.0 for seg in segments)

        self.assertGreaterEqual(min_bottom, 1.524 - 0.760 - 1e-9)
        self.assertGreater(min_bottom, 0.70)
        self.assertLessEqual(max_abs_y, 3.05 + 1e-9)

    def test_net_top_sags_to_center_and_rises_to_posts(self) -> None:
        profile = self.geo["net"]["profile_samples"]
        center = min(profile, key=lambda p: abs(p["y"]))
        left_edge = max(profile, key=lambda p: p["y"])
        right_edge = min(profile, key=lambda p: p["y"])

        self.assertAlmostEqual(center["top_z"], 1.524, places=6)
        self.assertAlmostEqual(left_edge["top_z"], 1.55, places=6)
        self.assertAlmostEqual(right_edge["top_z"], 1.55, places=6)

    def test_post_geometry_matches_project_definition(self) -> None:
        posts = {p["name"]: p for p in self.geo["posts"]}
        self.assertAlmostEqual(posts["NetPost_Left"]["center"][1], 3.05, places=9)
        self.assertAlmostEqual(posts["NetPost_Right"]["center"][1], -3.05, places=9)
        self.assertAlmostEqual(posts["NetPost_Left"]["height_m"], 1.55, places=9)
        self.assertAlmostEqual(posts["NetPost_Right"]["height_m"], 1.55, places=9)

    def test_semantic_frames_exist_at_expected_locations(self) -> None:
        frames = self.geo["frames"]
        self.assertEqual(frames["court"], (0.0, 0.0, 0.0))
        self.assertEqual(frames["net_center"], (0.0, 0.0, 0.0))
        self.assertEqual(frames["net_post_left"], (0.0, 3.05, 0.0))
        self.assertEqual(frames["net_post_right"], (0.0, -3.05, 0.0))
        self.assertEqual(frames["robot_home"], (-1.60, 0.0, 0.0))

    def test_geometry_contains_no_nonfinite_values(self) -> None:
        def walk(value):
            if isinstance(value, dict):
                for v in value.values():
                    yield from walk(v)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    yield from walk(v)
            elif isinstance(value, (int, float)):
                yield float(value)

        for value in walk(self.geo):
            self.assertTrue(math.isfinite(value), msg=f"non-finite geometry value: {value}")


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_ground_to_net_invisible_wall_configuration(self) -> None:
        cfg = load_config(CONFIG_PATH)
        cfg["net"]["vertical_depth_m"] = 1.50
        with self.assertRaisesRegex(ValueError, "net bottom"):
            validate_config(cfg)

    def test_rejects_nonpositive_line_width(self) -> None:
        cfg = load_config(CONFIG_PATH)
        cfg["court"]["line_width_m"] = 0.0
        with self.assertRaisesRegex(ValueError, "line_width_m"):
            validate_config(cfg)


@unittest.skipUnless(importlib.util.find_spec("pxr") is not None, "pxr/USD not installed in this Python")
class OptionalUsdAuthoringTests(unittest.TestCase):
    def test_generated_stage_contains_required_prims(self) -> None:
        from pxr import Usd

        cfg = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "badminton_court.usda"
            build_usd(cfg, out)

            stage = Usd.Stage.Open(str(out))
            self.assertIsNotNone(stage)

            required = [
                "/BadmintonCourt",
                "/BadmintonCourt/Visual/Floor",
                "/BadmintonCourt/Visual/CourtLines",
                "/BadmintonCourt/Visual/Net",
                "/BadmintonCourt/Visual/Posts",
                "/BadmintonCourt/Physics/GroundCollider",
                "/BadmintonCourt/Physics/NetColliders",
                "/BadmintonCourt/Physics/PostColliders",
                "/BadmintonCourt/Frames/court",
                "/BadmintonCourt/Frames/net_center",
            ]
            for prim_path in required:
                self.assertTrue(stage.GetPrimAtPath(prim_path).IsValid(), prim_path)


if __name__ == "__main__":
    unittest.main()
