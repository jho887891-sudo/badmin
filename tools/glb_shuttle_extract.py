#!/usr/bin/env python3
"""Extract ONLY the shuttlecock from the CC-BY GLB into a standalone USD visual.

Refactored (TDD/REFACTOR step) onto tools/usd_glb_common.py - one shared GLB->USD path.
Source : assets/external/_staging/D_racket_shuttle/original/*.glb
Output : assets/third_party/shuttlecock_visual.usd
Frame  : origin at cork flat face centre, +Z toward the skirt (configs/shuttlecock.yaml).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from usd_glb_common import (  # noqa: E402
    author_mesh_from_glb, iter_mesh_nodes, make_material_from_glb, parse_glb,
)

INCLUDE = ('Obj_Feather', 'Obj_Cork')


def main() -> int:
    glb_path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    glb = parse_glb(glb_path)
    nodes = iter_mesh_nodes(glb, include=INCLUDE)
    print('extracted meshes:', nodes)

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/ShuttlecockVisual')
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute('project:sourceAsset', Sdf.ValueTypeNames.String, custom=True).Set(glb_path.name)
    root.GetPrim().CreateAttribute('project:frameConvention', Sdf.ValueTypeNames.String, custom=True).Set(
        'origin at cork flat face centre; +Z from cork toward skirt; metres')

    used = []
    for _, mesh_index in nodes:
        for prim in glb.json['meshes'][mesh_index]['primitives']:
            mat = prim.get('material')
            if mat is not None and mat not in used:
                used.append(mat)
    materials = {}
    for m in sorted(used):
        name = str(glb.json['materials'][m].get('name') or ('Mat%d' % m)).replace(' ', '_')
        materials[m] = make_material_from_glb(stage, '/ShuttlecockVisual/Looks/%s' % name, glb, m,
                                              texture_dir=out.parent / 'textures',
                                              texture_tag='mat%d_basecolor' % m)
        print('  material[%d] %s' % (m, name))

    total = 0
    for name, mesh_index in nodes:
        mat_idx = glb.json['meshes'][mesh_index]['primitives'][0].get('material')
        mesh = author_mesh_from_glb(stage, '/ShuttlecockVisual/%s' % name, glb, mesh_index,
                                    material=materials.get(mat_idx))
        counts = UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Get()
        pts = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
        total += len(counts)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        print('  %-16s tris=%-5d x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]'
              % (name, len(counts), min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
    stage.GetRootLayer().Save()
    print('total tris (shuttle only):', total)
    print('wrote:', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
