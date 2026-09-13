#!/usr/bin/env python3
"""Extract ONE racket from the CC-BY GLB into a standalone USD visual.

Source : assets/external/_staging/D_racket_shuttle/original/*.glb
Output : assets/third_party/racket_visual.usd

Frame mapping (configs/racket.yaml): origin at racket_tcp (handle end), +Z handle->head,
+X = racket face normal toward the opponent.  The GLB racket has its string-bed plane in
the X-Z plane, i.e. the face normal is +Y, so points/normals are rotated -90 deg about Z:
    (x, y, z) -> (y, -x, z)
No rescale is applied: final scaling must follow the measured physical racket.
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path

GLB = Path(sys.argv[1])
OUT = Path(sys.argv[2])
TEXDIR = OUT.parent / 'textures'

raw = GLB.read_bytes()
magic, version, length = struct.unpack_from('<III', raw, 0)
assert magic == 0x46546C67, 'not a GLB'
off, js, binbuf = 12, None, None
while off < length:
    clen, ctype = struct.unpack_from('<II', raw, off)
    chunk = raw[off + 8: off + 8 + clen]
    if ctype == 0x4E4F534A:
        js = json.loads(chunk.decode('utf-8'))
    elif ctype == 0x004E4942:
        binbuf = chunk
    off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)

COMP = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2), 5123: ('H', 2), 5125: ('I', 4), 5126: ('f', 4)}
NUM = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}

def read_accessor(idx):
    acc = js['accessors'][idx]
    bv = js['bufferViews'][acc['bufferView']]
    fmt, size = COMP[acc['componentType']]
    n = NUM[acc['type']]
    stride = bv.get('byteStride') or size * n
    base = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    return [struct.unpack_from('<' + fmt * n, binbuf, base + i * stride) for i in range(acc['count'])]

# --- pick the FIRST racket only (Obj_Racket + its Obj_Strings), skip Obj_Racket.001 ---
want, skip = [], False
for n in js['nodes']:
    nm = n.get('name') or ''
    if nm == 'Obj_Racket.001':
        skip = True
    if nm == 'Obj_Racket':
        skip = False
    if 'mesh' in n and not skip and nm.startswith('Obj_Racket'):
        want.append((nm.replace('.', '_'), n['mesh']))
    if 'mesh' in n and not skip and nm.startswith('Obj_Strings'):
        want.append((nm.replace('.', '_'), n['mesh']))
print('extracted meshes:', want)

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt

def rot(p):
    x, y, z = p
    return (y, -x, z)

used = []
for _, mi in want:
    for prim in js['meshes'][mi]['primitives']:
        m = prim.get('material')
        if m is not None and m not in used:
            used.append(m)
used.sort()

OUT.parent.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.CreateNew(str(OUT))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
root = UsdGeom.Xform.Define(stage, '/RacketVisual')
stage.SetDefaultPrim(root.GetPrim())
root.GetPrim().CreateAttribute('project:sourceAsset', Sdf.ValueTypeNames.String, custom=True).Set(GLB.name)
root.GetPrim().CreateAttribute('project:frameConvention', Sdf.ValueTypeNames.String, custom=True).Set(
    'origin at racket_tcp (handle end); +Z handle to head; +X face normal toward opponent')

mat_paths = {}
for mi_idx in used:
    spec = js['materials'][mi_idx]
    pbr = spec.get('pbrMetallicRoughness', {}) or {}
    name = str(spec.get('name') or ('Mat%d' % mi_idx)).replace(' ', '_')
    base = pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])
    path = '/RacketVisual/Looks/%s' % name
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + '/PreviewSurface')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(pbr.get('roughnessFactor', 1.0)))
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(pbr.get('metallicFactor', 1.0)))
    bct = pbr.get('baseColorTexture')
    if bct:
        tex = js['textures'][bct['index']]
        img = js['images'][tex['source']]
        bv = js['bufferViews'][img['bufferView']]
        blob = binbuf[bv.get('byteOffset', 0): bv.get('byteOffset', 0) + bv['byteLength']]
        TEXDIR.mkdir(parents=True, exist_ok=True)
        tex_file = TEXDIR / ('racket_%s_basecolor.png' % name.lower())
        tex_file.write_bytes(blob)
        rd = UsdShade.Shader.Define(stage, path + '/stReader')
        rd.CreateIdAttr('UsdPrimvarReader_float2')
        rd.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tx = UsdShade.Shader.Define(stage, path + '/tex')
        tx.CreateIdAttr('UsdUVTexture')
        tx.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(tex_file))
        tx.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), 'result')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), 'rgb')
        print('  texture ->', tex_file, len(blob), 'bytes')
    else:
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in base[:3]]))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    mat_paths[mi_idx] = mat

total = 0
for name, mi in want:
    for prim in js['meshes'][mi]['primitives']:
        attrs = prim['attributes']
        pts = [rot(p) for p in read_accessor(attrs['POSITION'])]
        flat = [c[0] for c in read_accessor(prim['indices'])]
        total += len(flat) // 3
        mesh = UsdGeom.Mesh.Define(stage, '/RacketVisual/%s' % name)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(flat) // 3)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(flat))
        if 'NORMAL' in attrs:
            nrm = [rot(n) for n in read_accessor(attrs['NORMAL'])]
            mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm]))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if 'TEXCOORD_0' in attrs:
            pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in read_accessor(attrs['TEXCOORD_0'])]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat_paths[prim.get('material', used[0])])
        if js['materials'][prim.get('material', used[0])].get('doubleSided'):
            mesh.CreateDoubleSidedAttr(True)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        print('  %-18s tris=%-5d x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]' % (name, len(flat)//3, min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
stage.GetRootLayer().Save()
print('total tris (one racket):', total)
print('wrote:', OUT)