#!/usr/bin/env python3
"""Extract ONLY the shuttlecock from the CC-BY GLB and author a standalone USD visual.

Source : assets/external/_staging/D_racket_shuttle/original/*.glb
Output : assets/third_party/shuttlecock_visual.usd  (+ extracted base-colour texture)

Convention (matches configs/shuttlecock.yaml):
    origin at cork flat face centre, +Z from cork toward the feather skirt, metres.
The rackets in the GLB are dropped; visual proportions are preserved (no rescale).
"""
from __future__ import annotations
import json, struct, sys
from pathlib import Path

GLB = Path(sys.argv[1])
OUT = Path(sys.argv[2])
TEXDIR = OUT.parent / "textures"

# ---------- read GLB ----------
raw = GLB.read_bytes()
magic, version, length = struct.unpack_from("<III", raw, 0)
assert magic == 0x46546C67, "not a GLB"
off, js, binbuf = 12, None, None
while off < length:
    clen, ctype = struct.unpack_from("<II", raw, off)
    chunk = raw[off + 8: off + 8 + clen]
    if ctype == 0x4E4F534A:
        js = json.loads(chunk.decode("utf-8"))
    elif ctype == 0x004E4942:
        binbuf = chunk
    off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)

COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NUM = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

def read_accessor(idx):
    acc = js["accessors"][idx]
    bv = js["bufferViews"][acc["bufferView"]]
    fmt, size = COMP[acc["componentType"]]
    n = NUM[acc["type"]]
    stride = bv.get("byteStride") or size * n
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = []
    for i in range(acc["count"]):
        out.append(struct.unpack_from("<" + fmt * n, binbuf, base + i * stride))
    return out

# ---------- which nodes are the shuttle ----------
name_to_node = {n.get("name"): i for i, n in enumerate(js["nodes"])}
shuttle_root = None
for i, n in enumerate(js["nodes"]):
    if n.get("name") == "Gp_Shuttle":
        shuttle_root = i
assert shuttle_root is not None, "Gp_Shuttle not found"

want_meshes, keep_nodes = [], []
def walk(i):
    n = js["nodes"][i]
    keep_nodes.append(n.get("name"))
    if "mesh" in n:
        want_meshes.append((n.get("name"), n["mesh"]))
    for c in n.get("children", []):
        walk(c)
walk(shuttle_root)
print("kept nodes:", keep_nodes, "| meshes:", [m[1] for m in want_meshes])

# ---------- materials (data-driven from the GLB) ----------
used_mats = []
for _, mi in want_meshes:
    for prim in js["meshes"][mi]["primitives"]:
        m = prim.get("material")
        if m is not None and m not in used_mats:
            used_mats.append(m)
used_mats.sort()

from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt

def _tex_index(mat_idx):
    pbr = js["materials"][mat_idx].get("pbrMetallicRoughness", {}) or {}
    bct = pbr.get("baseColorTexture")
    return bct.get("index") if bct else None

def _extract_texture(tex_index, tag):
    tex = js["textures"][tex_index]
    img = js["images"][tex["source"]]
    bv = js["bufferViews"][img["bufferView"]]
    blob = binbuf[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
    ext = ".png" if img.get("mimeType", "").endswith("png") else ".jpg"
    TEXDIR.mkdir(parents=True, exist_ok=True)
    out = TEXDIR / ("%s%s" % (tag, ext))
    out.write_bytes(blob)
    print("texture ->", out, len(blob), "bytes")
    return out

OUT.parent.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.CreateNew(str(OUT))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
root = UsdGeom.Xform.Define(stage, "/ShuttlecockVisual")
stage.SetDefaultPrim(root.GetPrim())
root.GetPrim().CreateAttribute("project:sourceAsset", Sdf.ValueTypeNames.String, custom=True).Set(GLB.name)
root.GetPrim().CreateAttribute("project:frameConvention", Sdf.ValueTypeNames.String, custom=True).Set(
    "origin at cork flat face centre; +Z from cork toward skirt; metres")

mat_paths = {}
for mi_idx in used_mats:
    spec = js["materials"][mi_idx]
    pbr = spec.get("pbrMetallicRoughness", {}) or {}
    name = spec.get("name") or ("Mat%d" % mi_idx)
    base = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    rough = float(pbr.get("roughnessFactor", 1.0))
    metal = float(pbr.get("metallicFactor", 1.0))
    path = "/ShuttlecockVisual/Looks/%s" % str(name).replace(" ", "_")
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    tex_index = _tex_index(mi_idx)
    if tex_index is not None:
        tex_file = _extract_texture(tex_index, "mat%d_basecolor" % mi_idx)
        rd = UsdShade.Shader.Define(stage, path + "/stReader")
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex_file))
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in base[:3]]))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    mat_paths[mi_idx] = mat
    print("material[%d] '%s' -> %s  base=%s rough=%.2f metal=%.2f tex=%s" % (
        mi_idx, name, path, [round(float(v),3) for v in base[:3]], rough, metal, tex_index))

tot_tris = 0
for name, mi in want_meshes:
    for pi, prim in enumerate(js["meshes"][mi]["primitives"]):
        attrs = prim["attributes"]
        pts = read_accessor(attrs["POSITION"])
        normals = read_accessor(attrs["NORMAL"]) if "NORMAL" in attrs else None
        uvs = read_accessor(attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs else None
        flat = [c[0] for c in read_accessor(prim["indices"])]
        tot_tris += len(flat) // 3
        path = "/ShuttlecockVisual/%s" % name
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(flat) // 3)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(flat))
        if normals:
            mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals]))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if uvs:
            pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in uvs]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat_paths[prim.get("material", used_mats[0])])
        if js["materials"][prim.get("material", used_mats[0])].get("doubleSided"):
            mesh.CreateDoubleSidedAttr(True)
        # report
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        print("  %-16s tris=%-5d x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]" % (
            name, len(flat) // 3, min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))

stage.GetRootLayer().Save()
print("total tris (shuttle only):", tot_tris)
print("wrote:", OUT)
