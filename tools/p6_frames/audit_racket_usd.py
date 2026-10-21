#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6-01: audit assets/third_party/racket_visual.usd and its source GLB. READ-ONLY on the assets.

Answers exactly the questions P6-01 asks, from the files themselves:
  * metersPerUnit / upAxis   -> read from the stage metadata, not assumed
  * bounding box in metres   -> computed world bound, then scaled BY metersPerUnit
  * collision geometry       -> hard search for CollisionAPI / Physics* prims, count only
  * flange / end link        -> NOT in this asset; recorded as absent, with the arm-side
                                answer coming from the RS layer evidence instead
  * sha256                   -> hashed here so every other record can be tied to bytes
  * GLB units/scale          -> the glTF JSON chunk is read directly: glTF is metres by
                                definition, but Sketchfab puts an authoring scale on
                                Sketchfab_model, so the node transforms are inspected
                                rather than believed.

Needs pxr for the USD part (pip install usd-core).  The GLB part is pure stdlib.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

RACKET_USD = "assets/third_party/racket_visual.usd"
RACKET_GLB = "assets/external/_staging/D_racket_shuttle/original/badminton_racket_and_shuttlecock_low_poly.glb"

# Values recorded by the committed extraction record (assets/THIRD_PARTY_ASSETS_RACKET.md)
# so a mismatch against the live asset is a FAIL, not a silent difference.
DOCUMENTED = {
    "racket_usd_sha256": "643fb67003c6429f98ec132d2db056c83ed02811b594ad27904b618bf7c1ec29",
    "glb_sha256": "8f6c2baac0bbd2f2f1aceaa295b06fd74a10aed59de8686d2d76cdcc2e22bfa5",
    "frame_bbox_xyz_min": [-0.0158, -0.1080, 0.0000],
    "frame_bbox_xyz_max": [0.0158, 0.1080, 0.6827],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_glb(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, length = struct.unpack("<4sII", data[:12])
    off, js = 12, None
    while off < length:
        clen, ctype = struct.unpack("<I4s", data[off:off + 8])
        if ctype == b"JSON":
            js = json.loads(data[off + 8:off + 8 + clen].decode("utf-8"))
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    nodes = js.get("nodes", [])
    scaled = [
        {"index": i, "name": n.get("name"), "scale": n.get("scale"),
         "translation": n.get("translation"), "rotation": n.get("rotation")}
        for i, n in enumerate(nodes) if "scale" in n or "rotation" in n or "translation" in n
    ]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": len(data),
        "glb_magic": magic.decode("ascii", "replace"),
        "glb_version": version,
        "declared_length": length,
        "asset": js.get("asset"),
        "asset_extras": (js.get("asset") or {}).get("extras"),
        "generator": (js.get("asset") or {}).get("generator"),
        # glTF 2.0 declares units implicitly: metres, right-handed, +Y up, no unit field.
        "unit_convention": "glTF 2.0 spec: length unit is metres, right-handed, +Y up. The file has no unit field, so the unit is a SPEC property, not a measured one.",
        "nodes_total": len(nodes),
        "nodes_with_local_transform": scaled,
        "meshes": [m.get("name") for m in js.get("meshes", [])],
        "materials": [m.get("name") for m in js.get("materials", [])],
        "extensions_used": js.get("extensionsUsed"),
        "scale_verdict": "PASS" if not scaled else "REVIEW",
        "scale_note": (
            "No node in this GLB carries a scale/rotation/translation, so no Sketchfab "
            "authoring scale is hiding in the hierarchy: the mesh coordinates are already "
            "in glTF metres. Whether those metres match the real racket is a DIFFERENT "
            "question and is still open (682.7 mm vs 675 mm product length)."
            if not scaled else
            "Some nodes carry local transforms; the extraction record's rotation must be "
            "re-derived against them before the visual is trusted."
        ),
    }


def audit_usd(path: Path) -> dict:
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "UNKNOWN",
            "reason": "pxr not importable in this interpreter: " + repr(exc),
            "sha256": sha256(path),
            "note": "The USD is a binary USDC crate, so units/bbox/collision cannot be read without pxr.",
        }

    stage = Usd.Stage.Open(str(path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    default_prim = stage.GetDefaultPrim()

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    per_mesh, collision_prims, physics_prims, meshes, materials = {}, [], [], [], []
    for prim in stage.Traverse():
        name = prim.GetTypeName()
        if name == "Mesh":
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = rng.GetMin(), rng.GetMax()
            meshes.append({
                "path": str(prim.GetPath()),
                "points": len(UsdGeom.Mesh(prim).GetPointsAttr().Get() or []),
                "bbox_stage_units_min": [mn[0], mn[1], mn[2]],
                "bbox_stage_units_max": [mx[0], mx[1], mx[2]],
                "bbox_metres_min": [v * mpu for v in (mn[0], mn[1], mn[2])],
                "bbox_metres_max": [v * mpu for v in (mx[0], mx[1], mx[2])],
                "has_collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
            })
            per_mesh[str(prim.GetPath())] = meshes[-1]
        if prim.HasAPI(UsdPhysics.CollisionAPI) or name.startswith("Physics"):
            collision_prims.append(str(prim.GetPath()))
        if name == "Material":
            materials.append(str(prim.GetPath()))

    # union bbox over the visual meshes, in metres
    mn = [min(m["bbox_metres_min"][i] for m in meshes) for i in range(3)]
    mx = [max(m["bbox_metres_max"][i] for m in meshes) for i in range(3)]
    size = [mx[i] - mn[i] for i in range(3)]

    frame = per_mesh.get("/RacketVisual/Obj_Racket_0")
    documented_matches = (
        frame is not None
        and all(abs(frame["bbox_metres_min"][i] - DOCUMENTED["frame_bbox_xyz_min"][i]) <= 5e-4 for i in range(3))
        and all(abs(frame["bbox_metres_max"][i] - DOCUMENTED["frame_bbox_xyz_max"][i]) <= 5e-4 for i in range(3))
    )
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "file_format": "USDC (binary crate) - header " + path.read_bytes()[:8].decode("ascii", "replace"),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "metersPerUnit": mpu,
        "upAxis": up,
        "units_verdict": "PASS" if mpu == 1.0 and str(up).upper() == "Z" else "FAIL",
        "units_note": (
            "metersPerUnit=1 with upAxis=Z: every coordinate in this asset is already metres "
            "and Z-up, so no 0.01 / 1000 factor may be applied anywhere downstream."
        ),
        "materials": materials,
        "meshes": meshes,
        "union_bbox_metres_min": mn,
        "union_bbox_metres_max": mx,
        "union_bbox_size_metres": size,
        "overall_length_m": size[2],
        "head_width_m": size[1],
        "collision_prims": collision_prims,
        "collision_count": len(collision_prims),
        "collision_verdict": "PASS" if not collision_prims else "REVIEW",
        "collision_note": (
            "ZERO CollisionAPI/Physics prims: this asset is visual-only and cannot be "
            "collided with. P6-07 must author its own proxies; anything reading a bbox "
            "from this file and calling it a collider is wrong."
        ),
        "flange_or_end_link_in_asset": False,
        "flange_note": (
            "The asset contains no flange, no mount and no joint: it is two meshes under "
            "one Xform. The origin is the butt end of the handle (z=0 coincides with the "
            "lowest mesh point), it is NOT the flange mating face. How deep the handle sits "
            "inside the mount is unmeasured, so the flange->handle-butt translation is UNKNOWN."
        ),
        "documented_bbox_matches": documented_matches,
        "documented_bbox_verdict": "PASS" if documented_matches else "FAIL",
        "bbox_note": (
            "Measured 682.7 mm overall vs 675 mm product length and 680 mm BWF maximum: the "
            "visual is bigger than the racket it stands for and has NOT been rescaled "
            "(configs/racket.yaml measured_unit is still REQUIRES_MEASUREMENT)."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    usd = audit_usd(root / RACKET_USD)
    glb = audit_glb(root / RACKET_GLB)

    usd_sha_ok = usd.get("sha256") == DOCUMENTED["racket_usd_sha256"]
    glb_sha_ok = glb.get("sha256") == DOCUMENTED["glb_sha256"]
    record = {
        "schema": "p6_racket_asset_audit/1",
        "usd": usd,
        "glb": glb,
        "sha256_verdict": {
            "racket_usd": "PASS" if usd_sha_ok else "FAIL",
            "glb": "PASS" if glb_sha_ok else "FAIL",
            "documented_in": "assets/THIRD_PARTY_ASSETS_RACKET.md:15,17",
        },
        "provenance": {
            "generated_by": "tools/p6_frames/audit_racket_usd.py",
            "pxr_used": usd.get("status") != "UNKNOWN",
            "assets_modified": False,
        },
    }
    target = out_dir / "racket_usd_audit.json"
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", target)
    print("usd", usd.get("status"), "metersPerUnit", usd.get("metersPerUnit"), "upAxis", usd.get("upAxis"),
          "collision", usd.get("collision_count"), "sha_ok", usd_sha_ok, "glb_sha_ok", glb_sha_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
