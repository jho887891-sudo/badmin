#!/usr/bin/env python3
"""Extract ONE racket from the CC-BY GLB into a standalone USD visual.

Refactored (TDD/REFACTOR step) onto tools/usd_glb_common.py.
Frame mapping (configs/racket.yaml): origin at racket_tcp (handle end), +Z handle->head,
+X = face normal toward the opponent.  The GLB string-bed plane is X-Z (normal +Y), so
points/normals are rotated -90 deg about Z: (x, y, z) -> (y, -x, z).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from usd_glb_common import (  # noqa: E402
    author_mesh_from_glb, iter_mesh_nodes, make_material_from_glb, parse_glb,
)

INCLUDE = ('Obj_Racket', 'Obj_Strings')
EXCLUDE = ('Obj_Racket.001', 'Obj_Strings.001')


def rot(point):
    x, y, z = point
    return (y, -x, z)


def main() -> int:
    glb_path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    glb = parse_glb(glb_path)
    nodes = iter_mesh_nodes(glb, include=INCLUDE, exclude=EXCLUDE)
    print('extracted meshes:', nodes)

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/RacketVisual')
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute('project:sourceAsset', Sdf.ValueTypeNames.String, custom=True).Set(glb_path.name)
    root.GetPrim().CreateAttribute('project:frameConvention', Sdf.ValueTypeNames.String, custom=True).Set(
        'origin at racket_tcp (handle end); +Z handle to head; +X face normal toward opponent')

    used = []
    for _, mesh_index in nodes:
        for prim in glb.json['meshes'][mesh_index]['primitives']:
            mat = prim.get('material')
            if mat is not None and mat not in used:
                used.append(mat)
    materials = {}
    for m in sorted(used):
        name = str(glb.json['materials'][m].get('name') or ('Mat%d' % m)).replace(' ', '_')
        materials[m] = make_material_from_glb(stage, '/RacketVisual/Looks/%s' % name, glb, m,
                                              texture_dir=out.parent / 'textures',
                                              texture_tag='racket_%s_basecolor' % name.lower())
        print('  material[%d] %s' % (m, name))

    total = 0
    for name, mesh_index in nodes:
        mat_idx = glb.json['meshes'][mesh_index]['primitives'][0].get('material')
        mesh = author_mesh_from_glb(stage, '/RacketVisual/%s' % name, glb, mesh_index,
                                    material=materials.get(mat_idx), point_transform=rot)
        counts = UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Get()
        pts = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
        total += len(counts)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        print('  %-18s tris=%-5d x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]'
              % (name, len(counts), min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
    stage.GetRootLayer().Save()
    print('total tris (one racket):', total)
    print('wrote:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
