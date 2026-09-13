# -*- coding: utf-8 -*-
"""Shared GLB -> USD helpers (one implementation for both extractors).

Driven by tests/tools/test_usd_glb_common.py (TDD: RED -> GREEN -> REFACTOR).
Codifies ISSUE-005: USD 25.11 orient ops may be Quatf or Quatd, so the setter must
match the attribute type instead of guessing; customData takes Vt arrays, not lists.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
COMPONENT = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2), 5123: ('H', 2), 5125: ('I', 4), 5126: ('f', 4)}
NUM_COMPONENTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


@dataclass
class GlbData:
    json: Dict = field(default_factory=dict)
    binary: bytes = b''
    source: Path = Path('.')


def parse_glb(path) -> GlbData:
    """Read a .glb container into its JSON chunk and binary buffer."""
    raw = Path(path).read_bytes()
    magic, version, length = struct.unpack_from('<III', raw, 0)
    if magic != GLB_MAGIC:
        raise ValueError('not a GLB container')
    off, js, binbuf = 12, None, None
    while off < length:
        clen, ctype = struct.unpack_from('<II', raw, off)
        chunk = raw[off + 8: off + 8 + clen]
        if ctype == CHUNK_JSON:
            js = json.loads(chunk.decode('utf-8'))
        elif ctype == CHUNK_BIN:
            binbuf = chunk
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    if js is None:
        raise ValueError('GLB has no JSON chunk')
    return GlbData(js, binbuf or b'', Path(path))


def read_accessor(glb: GlbData, index: int) -> List[Tuple[float, ...]]:
    """Decode a glTF accessor into a list of tuples (handles byteStride)."""
    acc = glb.json['accessors'][index]
    view = glb.json['bufferViews'][acc['bufferView']]
    fmt, size = COMPONENT[acc['componentType']]
    n = NUM_COMPONENTS[acc['type']]
    stride = view.get('byteStride') or size * n
    base = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    return [struct.unpack_from('<' + fmt * n, glb.binary, base + i * stride) for i in range(acc['count'])]


def iter_mesh_nodes(glb: GlbData, *, include: Sequence[str], exclude: Sequence[str] = ()) -> List[Tuple[str, int]]:
    """Return (sanitised_node_name, mesh_index) for mesh nodes matching the filters."""
    out: List[Tuple[str, int]] = []
    for node in glb.json.get('nodes', []):
        name = node.get('name') or ''
        if 'mesh' not in node:
            continue
        if exclude and any(name.startswith(prefix) for prefix in exclude):
            continue
        if include and not any(name.startswith(prefix) for prefix in include):
            continue
        out.append((name.replace('.', '_'), int(node['mesh'])))
    return out


def set_orient_compat(op, quaternion_wxyz: Sequence[float]) -> None:
    """Author an orient xformOp using the precision the stage actually created.

    USD 25.11 hands back either Quatf or Quatd for a freshly added xformOp:orient;
    writing the other precision raises Tf.ErrorException (ISSUE-005).
    """
    from pxr import Gf, Sdf

    w, x, y, z = [float(v) for v in quaternion_wxyz]
    if op.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Quatf:
        op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    else:
        op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def author_mesh_from_glb(stage, prim_path: str, glb: GlbData, mesh_index: int, *,
                         material=None, point_transform: Optional[Callable] = None):
    """Author a USD mesh from one glTF mesh (first primitive) and return the schema object."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    spec = glb.json['meshes'][mesh_index]
    prim_spec = spec['primitives'][0]
    attrs = prim_spec['attributes']
    pts = read_accessor(glb, attrs['POSITION'])
    if point_transform is not None:
        pts = [tuple(point_transform(list(p))) for p in pts]
    flat = [c[0] for c in read_accessor(glb, prim_spec['indices'])]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(flat) // 3)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(flat))
    if 'NORMAL' in attrs:
        nrm = read_accessor(glb, attrs['NORMAL'])
        if point_transform is not None:
            nrm = [tuple(point_transform(list(n))) for n in nrm]
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in n]) for n in nrm]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if 'TEXCOORD_0' in attrs:
        uv = read_accessor(glb, attrs['TEXCOORD_0'])
        pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(u[0]), float(u[1])) for u in uv]))
    mat_idx = prim_spec.get('material')
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    elif mat_idx is not None:
        pass
    if mat_idx is not None and glb.json['materials'][mat_idx].get('doubleSided'):
        mesh.CreateDoubleSidedAttr(True)
    return mesh


def make_material_from_glb(stage, prim_path: str, glb: GlbData, material_index: int,
                           texture_dir: Optional[Path] = None, texture_tag: str = 'tex'):
    """Author a UsdPreviewSurface from a glTF material; textures only when referenced."""
    from pxr import Gf, Sdf, UsdShade

    spec = glb.json['materials'][material_index]
    pbr = spec.get('pbrMetallicRoughness', {}) or {}
    mat = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, prim_path + '/PreviewSurface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(float(pbr.get('roughnessFactor', 1.0)))
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(float(pbr.get('metallicFactor', 1.0)))
    base = pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])
    bct = pbr.get('baseColorTexture')
    if bct is not None:
        tex = glb.json['textures'][bct['index']]
        img = glb.json['images'][tex['source']]
        view = glb.json['bufferViews'][img['bufferView']]
        blob = glb.binary[view.get('byteOffset', 0): view.get('byteOffset', 0) + view['byteLength']]
        ext = '.png' if str(img.get('mimeType', '')).endswith('png') else '.jpg'
        target_dir = Path(texture_dir) if texture_dir is not None else Path('.')
        target_dir.mkdir(parents=True, exist_ok=True)
        tex_file = target_dir / (texture_tag + ext)
        tex_file.write_bytes(blob)
        reader = UsdShade.Shader.Define(stage, prim_path + '/stReader')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tex_node = UsdShade.Shader.Define(stage, prim_path + '/tex')
        tex_node.CreateIdAttr('UsdUVTexture')
        tex_node.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(tex_file))
        tex_node.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_node.ConnectableAPI(), 'rgb')
    else:
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[float(v) for v in base[:3]]))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return mat
