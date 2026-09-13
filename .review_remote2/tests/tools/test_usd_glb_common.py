# -*- coding: utf-8 -*-
"""Shared GLB->USD helper tests (behaviour pinning for both extractors)."""
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))

from usd_glb_common import (  # noqa: E402
    GlbData, author_mesh_from_glb, iter_mesh_nodes, make_material_from_glb,
    parse_glb, read_accessor, set_orient_compat,
)

GLB = ROOT / 'assets/external/_staging/D_racket_shuttle/original/badminton_racket_and_shuttlecock_low_poly.glb'


@unittest.skipUnless(GLB.exists(), 'staged GLB not present')
class ParseTests(unittest.TestCase):
    def test_parse_glb_reads_json_and_binary(self) -> None:
        glb = parse_glb(GLB)
        self.assertIsInstance(glb, GlbData)
        self.assertEqual(glb.json['asset']['version'], '2.0')
        self.assertEqual(len(glb.json['nodes']), 17)
        self.assertEqual(len(glb.json['meshes']), 6)
        self.assertGreater(len(glb.binary), 1000)

    def test_read_accessor_returns_vec3_floats(self) -> None:
        glb = parse_glb(GLB)
        pos = read_accessor(glb, glb.json['meshes'][0]['primitives'][0]['attributes']['POSITION'])
        self.assertEqual(len(pos[0]), 3)
        self.assertTrue(all(math.isfinite(v) for v in pos[0]))
        self.assertGreater(len(pos), 100)

    def test_iter_mesh_nodes_filters_first_racket(self) -> None:
        glb = parse_glb(GLB)
        names = [n for n, _ in iter_mesh_nodes(glb, include=('Obj_Racket', 'Obj_Strings'),
                                               exclude=('Obj_Racket.001', 'Obj_Strings.001'))]
        self.assertEqual(names, ['Obj_Racket_0', 'Obj_Strings_0'])

    def test_iter_mesh_nodes_selects_shuttle(self) -> None:
        glb = parse_glb(GLB)
        names = [n for n, _ in iter_mesh_nodes(glb, include=('Obj_Feather', 'Obj_Cork'))]
        self.assertEqual(names, ['Obj_Feather_0', 'Obj_Cork_0'])


class OrientCompatTests(unittest.TestCase):
    def test_sets_orientation_on_fresh_op_without_type_error(self) -> None:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        cyl = UsdGeom.Cylinder.Define(stage, '/C')
        op = cyl.AddOrientOp()
        set_orient_compat(op, [1.0, 0.0, 0.0, 0.0])
        q = op.Get()
        self.assertAlmostEqual(float(q.GetReal()), 1.0, places=6)

    def test_rotates_plus_z_to_requested_direction(self) -> None:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        cyl = UsdGeom.Cylinder.Define(stage, '/C2')
        op = cyl.AddOrientOp()
        w = math.cos(math.pi / 4); y = math.sin(math.pi / 4)
        set_orient_compat(op, [w, 0.0, y, 0.0])
        q = op.Get()
        qw, qx, qy, qz = float(q.GetReal()), *[float(v) for v in q.GetImaginary()]
        vx = 2 * (qx * qz + qw * qy)
        vy = 2 * (qy * qz - qw * qx)
        vz = 1 - 2 * (qx * qx + qy * qy)
        self.assertAlmostEqual(vx, 1.0, places=5)
        self.assertAlmostEqual(vy, 0.0, places=5)
        self.assertAlmostEqual(vz, 0.0, places=5)


@unittest.skipUnless(GLB.exists(), 'staged GLB not present')
class AuthoringTests(unittest.TestCase):
    def test_author_mesh_matches_glb_triangle_count_and_bbox(self) -> None:
        from pxr import Usd, UsdGeom
        glb = parse_glb(GLB)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/R')
        prim = author_mesh_from_glb(stage, '/R/Feather', glb, 0)
        counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
        self.assertEqual(len(counts), 1920)
        pts = np.asarray([tuple(p) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()], dtype=float)
        self.assertAlmostEqual(float(pts[:, 2].min()), 0.0006, places=3)
        self.assertAlmostEqual(float(pts[:, 2].max()), 0.0780, places=3)
        self.assertAlmostEqual(float(pts[:, 0].max()), 0.0309, places=3)

    def test_material_is_data_driven_and_texture_optional(self) -> None:
        from pxr import Usd, UsdGeom, UsdShade
        glb = parse_glb(GLB)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/R2')
        tex_dir = ROOT / 'outputs' / 'tmp_tex'
        white = make_material_from_glb(stage, '/R2/Looks/White', glb, 0)
        strings = make_material_from_glb(stage, '/R2/Looks/Strings', glb, 1,
                                         texture_dir=tex_dir, texture_tag='strings')
        d0 = UsdShade.Material(white).ComputeSurfaceSource()[0].GetInput('diffuseColor')
        self.assertFalse(d0.HasConnectedSource())
        d1 = UsdShade.Material(strings).ComputeSurfaceSource()[0].GetInput('diffuseColor')
        self.assertTrue(d1.HasConnectedSource())
        self.assertTrue((tex_dir / 'strings.png').exists())


if __name__ == '__main__':
    unittest.main()