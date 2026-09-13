#!/usr/bin/env python3
"""Build a traceable badminton-racket physics asset for Isaac Sim.

The module intentionally separates three kinds of information:

1. regulatory limits from BWF,
2. manufacturer-published reference-product fields,
3. measurements from the exact physical racket installed on the robot.

Final physics authoring requires category (3). Missing physical-unit values are
not silently replaced by guessed constants.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat4 = Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]


def load_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read racket.yaml") from exc
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"racket config must be a YAML mapping: {p}")
    return copy.deepcopy(data)


def _mapping(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid mapping: {key}")
    return value


def _number(mapping: Mapping[str, Any], key: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{key} must be > 0")
    if nonnegative and value < 0.0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _pair(value: Any, name: str) -> Tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-element sequence")
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
        raise ValueError(f"{name} must satisfy 0 < min <= max")
    return lo, hi


def _vec3(value: Any, name: str) -> Vec3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-element sequence")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite values")
    return result  # type: ignore[return-value]


def _measured_values(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    measured = _mapping(cfg, "measured_unit")
    if measured.get("status") != "MEASURED":
        raise ValueError("measured_unit.status must be MEASURED for final physical geometry")
    return measured


def validate_config(cfg: Mapping[str, Any], *, final: bool = False) -> None:
    ref = _mapping(cfg, "reference")
    bwf = _mapping(ref, "bwf")
    product = _mapping(ref, "product")
    proxy = _mapping(cfg, "physics_proxy")
    visual = _mapping(cfg, "visual")
    measured = _mapping(cfg, "measured_unit")

    max_len = _number(bwf, "max_overall_length_m", positive=True)
    max_width = _number(bwf, "max_overall_width_m", positive=True)
    max_sb_len = _number(bwf, "max_stringed_area_length_m", positive=True)
    max_sb_width = _number(bwf, "max_stringed_area_width_m", positive=True)

    product_len = _number(product, "overall_length_m", positive=True)
    shaft_d = _number(product, "shaft_diameter_m", positive=True)
    if product_len > max_len:
        raise ValueError("reference product length exceeds BWF legal maximum")
    if shaft_d >= max_width:
        raise ValueError("reference product shaft diameter is not plausible")
    _pair(product.get("unstrung_mass_range_kg"), "reference product unstrung mass range")

    frame_segments = proxy.get("frame_segments")
    outline_segments = proxy.get("stringbed_outline_segments")
    if not isinstance(frame_segments, int) or isinstance(frame_segments, bool) or frame_segments < 12:
        raise ValueError("physics_proxy.frame_segments must be an integer >= 12")
    if not isinstance(outline_segments, int) or isinstance(outline_segments, bool) or outline_segments < 16:
        raise ValueError("physics_proxy.stringbed_outline_segments must be an integer >= 16")

    if visual.get("mode") != "EXTERNAL_REFERENCE":
        raise ValueError("visual.mode must be EXTERNAL_REFERENCE")
    if not isinstance(visual.get("local_asset_path"), str) or not visual.get("local_asset_path"):
        raise ValueError("visual.local_asset_path must be a non-empty string")

    if final and measured.get("status") != "MEASURED":
        raise ValueError("measured_unit is incomplete: exact physical-racket measurements are required for final authoring")

    if measured.get("status") == "MEASURED":
        mass = _number(measured, "installed_mass_kg", positive=True)
        com = _vec3(measured.get("center_of_mass_from_tcp_m"), "center_of_mass_from_tcp_m")
        inertia = _vec3(measured.get("diagonal_inertia_kg_m2"), "diagonal_inertia_kg_m2")
        if any(v <= 0.0 for v in inertia):
            raise ValueError("diagonal inertia must contain three positive values")

        hx = _number(measured, "handle_size_x_m", positive=True)
        hy = _number(measured, "handle_size_y_m", positive=True)
        handle_len = _number(measured, "handle_length_m", positive=True)
        head_len = _number(measured, "head_outer_length_m", positive=True)
        head_width = _number(measured, "head_outer_width_m", positive=True)
        sb_len = _number(measured, "stringbed_length_m", positive=True)
        sb_width = _number(measured, "stringbed_width_m", positive=True)
        frame_d = _number(measured, "frame_tube_diameter_m", positive=True)
        sb_thickness = _number(measured, "stringbed_thickness_m", positive=True)

        if head_width > max_width + 1e-12:
            raise ValueError("measured head width exceeds BWF maximum overall width")
        if sb_len > max_sb_len + 1e-12 or sb_width > max_sb_width + 1e-12:
            raise ValueError("measured stringbed dimensions exceed BWF maximum")
        if head_len >= product_len:
            raise ValueError("measured head length must be smaller than overall racket length")
        head_bottom = product_len - head_len
        if handle_len >= head_bottom:
            raise ValueError("measured handle leaves no positive-length shaft before the head")
        if sb_len >= head_len or sb_width >= head_width:
            raise ValueError("measured stringbed must fit strictly inside the measured head")
        if frame_d >= min(head_len - sb_len, head_width - sb_width):
            raise ValueError("measured frame tube diameter is inconsistent with head/stringbed dimensions")
        if sb_thickness >= frame_d:
            raise ValueError("stringbed collider thickness must be smaller than frame tube diameter")
        if hx >= head_width or hy >= head_width:
            raise ValueError("measured handle size is inconsistent with racket width")
        if not (0.0 <= com[2] <= product_len):
            raise ValueError("measured center of mass must lie along the racket's physical length")
        if mass > 1.0:
            raise ValueError("installed racket mass is implausibly large; check kg vs g units")


def _ellipse_point(center_z: float, half_width: float, half_length: float, theta: float) -> Vec3:
    return (0.0, half_width * math.cos(theta), center_z + half_length * math.sin(theta))


def build_physics_description(cfg: Mapping[str, Any], *, allow_temp: bool = False) -> Dict[str, Any]:
    del allow_temp  # kept for interface stability; final geometry still requires measured data.
    validate_config(cfg, final=True)
    measured = _measured_values(cfg)
    product = cfg["reference"]["product"]
    proxy = cfg["physics_proxy"]

    total_length = float(product["overall_length_m"])
    shaft_d = float(product["shaft_diameter_m"])
    handle_len = float(measured["handle_length_m"])
    head_len = float(measured["head_outer_length_m"])
    head_width = float(measured["head_outer_width_m"])
    sb_len = float(measured["stringbed_length_m"])
    sb_width = float(measured["stringbed_width_m"])
    frame_radius = float(measured["frame_tube_diameter_m"]) / 2.0
    head_center_z = total_length - head_len / 2.0
    head_bottom_z = total_length - head_len

    frame_count = int(proxy["frame_segments"])
    points = [
        _ellipse_point(head_center_z, head_width / 2.0, head_len / 2.0, 2.0 * math.pi * i / frame_count)
        for i in range(frame_count)
    ]
    frame_segments = [
        {
            "index": i,
            "p0_m": points[i],
            "p1_m": points[(i + 1) % frame_count],
            "radius_m": frame_radius,
        }
        for i in range(frame_count)
    ]

    outline_count = int(proxy["stringbed_outline_segments"])
    stringbed_outline = [
        (sb_width / 2.0 * math.cos(2.0 * math.pi * i / outline_count),
         head_center_z + sb_len / 2.0 * math.sin(2.0 * math.pi * i / outline_count))
        for i in range(outline_count)
    ]

    return {
        "overall_length_m": total_length,
        "handle": {
            "size_x_m": float(measured["handle_size_x_m"]),
            "size_y_m": float(measured["handle_size_y_m"]),
            "length_m": handle_len,
            "center_m": (0.0, 0.0, handle_len / 2.0),
        },
        "shaft": {
            "diameter_m": shaft_d,
            "radius_m": shaft_d / 2.0,
            "z_min_m": handle_len,
            "z_max_m": head_bottom_z,
            "length_m": head_bottom_z - handle_len,
            "center_m": (0.0, 0.0, (handle_len + head_bottom_z) / 2.0),
        },
        "head": {
            "outer_length_m": head_len,
            "outer_width_m": head_width,
            "center_z_m": head_center_z,
            "bottom_z_m": head_bottom_z,
            "top_z_m": total_length,
            "frame_segments": frame_segments,
        },
        "stringbed": {
            "length_m": sb_len,
            "width_m": sb_width,
            "center_z_m": head_center_z,
            "thickness_m": float(measured["stringbed_thickness_m"]),
            "outline_yz_m": tuple((float(y), float(z)) for y, z in stringbed_outline),
        },
        "frames": {
            "racket_tcp_m": (0.0, 0.0, 0.0),
            "racket_contact_frame_m": (0.0, 0.0, head_center_z),
            "face_normal_local": (1.0, 0.0, 0.0),
        },
    }


def compute_mass_properties(cfg: Mapping[str, Any], *, allow_temp: bool = False) -> Dict[str, Any]:
    del allow_temp
    validate_config(cfg, final=True)
    measured = _measured_values(cfg)
    return {
        "mass_kg": float(measured["installed_mass_kg"]),
        "center_of_mass_m": _vec3(measured["center_of_mass_from_tcp_m"], "center_of_mass_from_tcp_m"),
        "diagonal_inertia_kg_m2": _vec3(measured["diagonal_inertia_kg_m2"], "diagonal_inertia_kg_m2"),
        "provenance": "MEASURED_UNIT",
    }


def tcp_to_contact_transform(cfg: Mapping[str, Any], *, allow_temp: bool = False) -> Mat4:
    desc = build_physics_description(cfg, allow_temp=allow_temp)
    z = float(desc["stringbed"]["center_z_m"])
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _config_sha256(cfg: Mapping[str, Any]) -> str:
    import json
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _set_orient(op, quat) -> None:
    """Set an orient xformOp with the precision the stage actually created.

    USD 25.11 creates freshly added xformOp:orient attributes as Quatf while
    Gf.Quatd is the double-precision variant; writing the wrong type raises
    Tf.ErrorException. Match the attribute type instead of guessing.
    """
    from pxr import Gf, Sdf

    attr = op.GetAttr()
    if attr.GetTypeName() == Sdf.ValueTypeNames.Quatf:
        op.Set(Gf.Quatf(float(quat.GetReal()), Gf.Vec3f(*[float(v) for v in quat.GetImaginary()])))
    else:
        op.Set(quat)


def _quat_from_z_to_direction(direction: Vec3):
    from pxr import Gf

    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1e-15:
        raise ValueError("cannot orient frame segment with zero length")
    x, y, z = dx / norm, dy / norm, dz / norm
    dot = max(-1.0, min(1.0, z))
    if dot > 1.0 - 1e-12:
        return Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    if dot < -1.0 + 1e-12:
        return Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))
    ax, ay, az = -y, x, 0.0  # z_axis cross direction
    an = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / an, ay / an, az / an
    angle = math.acos(dot)
    return Gf.Quatd(math.cos(angle / 2.0), Gf.Vec3d(ax, ay, az) * math.sin(angle / 2.0))


def _author_stringbed_mesh(stage, path: str, outline_yz, thickness: float):
    from pxr import Gf, UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh.Define(stage, path)
    n = len(outline_yz)
    x0, x1 = -thickness / 2.0, thickness / 2.0
    points = [Gf.Vec3f(x0, float(y), float(z)) for y, z in outline_yz] + [
        Gf.Vec3f(x1, float(y), float(z)) for y, z in outline_yz
    ]
    counts = []
    indices = []
    counts.append(n)
    indices.extend(reversed(range(n)))
    counts.append(n)
    indices.extend(range(n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        indices.extend([i, j, n + j, n + i])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def build_usd(cfg: Mapping[str, Any], output_path: str | Path, *, allow_temp_visual: bool = False) -> Path:
    validate_config(cfg, final=True)
    desc = build_physics_description(cfg)
    mass = compute_mass_properties(cfg)

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pxr/USD is required for USD authoring; run this under Isaac Sim Python") from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Racket")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("schema_version", str(cfg.get("schema_version", "1.0")))
    root.GetPrim().SetCustomDataByKey("config_sha256", _config_sha256(cfg))
    root.GetPrim().SetCustomDataByKey("frame_convention", str(cfg["frames"]["convention"]))

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(mass["mass_kg"])
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*mass["center_of_mass_m"]))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*mass["diagonal_inertia_kg_m2"]))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    visual_xf = UsdGeom.Xform.Define(stage, "/Racket/Visual")
    visual = cfg["visual"]
    visual_path = Path(str(visual["local_asset_path"]))
    if visual_path.exists():
        visual_xf.GetPrim().GetReferences().AddReference(str(visual_path))
    elif bool(visual.get("required_for_final", True)) and not allow_temp_visual:
        raise FileNotFoundError(
            f"final visual asset not found: {visual_path}. Extract the CC Attribution racket mesh first, "
            "or pass --allow-temp-visual only for a debug USD."
        )
    else:
        temp = UsdGeom.Xform.Define(stage, "/Racket/Visual/TEMP_Debug")
        temp.GetPrim().SetCustomDataByKey("TEMP", True)
        temp.GetPrim().SetCustomDataByKey("reason", "External visual mesh not present")

    UsdGeom.Xform.Define(stage, "/Racket/Physics")

    h = desc["handle"]
    handle = UsdGeom.Cube.Define(stage, "/Racket/Physics/HandleCollider")
    handle.CreateSizeAttr(1.0)
    hx = handle.AddTranslateOp()
    hx.Set(Gf.Vec3d(*h["center_m"]))
    hs = handle.AddScaleOp()
    hs.Set(Gf.Vec3d(h["size_x_m"], h["size_y_m"], h["length_m"]))
    UsdPhysics.CollisionAPI.Apply(handle.GetPrim())

    s = desc["shaft"]
    shaft = UsdGeom.Cylinder.Define(stage, "/Racket/Physics/ShaftCollider")
    shaft.CreateAxisAttr(UsdGeom.Tokens.z)
    shaft.CreateRadiusAttr(s["radius_m"])
    shaft.CreateHeightAttr(s["length_m"])
    shaft.AddTranslateOp().Set(Gf.Vec3d(*s["center_m"]))
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())

    UsdGeom.Xform.Define(stage, "/Racket/Physics/HeadFrameColliders")
    for seg in desc["head"]["frame_segments"]:
        p0 = seg["p0_m"]
        p1 = seg["p1_m"]
        mid = tuple((a + b) / 2.0 for a, b in zip(p0, p1))
        direction = tuple(b - a for a, b in zip(p0, p1))
        length = math.sqrt(sum(v * v for v in direction))
        cyl = UsdGeom.Cylinder.Define(stage, f"/Racket/Physics/HeadFrameColliders/seg_{seg['index']:02d}")
        cyl.CreateAxisAttr(UsdGeom.Tokens.z)
        cyl.CreateRadiusAttr(seg["radius_m"])
        cyl.CreateHeightAttr(length)
        cyl.AddTranslateOp().Set(Gf.Vec3d(*mid))
        _set_orient(cyl.AddOrientOp(), _quat_from_z_to_direction(direction))
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

    sb = desc["stringbed"]
    _author_stringbed_mesh(stage, "/Racket/Physics/StringBedCollider", sb["outline_yz_m"], sb["thickness_m"])

    frames = UsdGeom.Xform.Define(stage, "/Racket/Frames")
    del frames
    UsdGeom.Xform.Define(stage, "/Racket/Frames/racket_tcp")
    contact = UsdGeom.Xform.Define(stage, "/Racket/Frames/racket_contact_frame")
    contact.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, sb["center_z_m"]))
    contact.GetPrim().SetCustomDataByKey("face_normal_local", Vt.FloatArray([1.0, 0.0, 0.0]))

    stage.GetRootLayer().Save()
    return out


def _summary(cfg: Mapping[str, Any]) -> str:
    ref = cfg["reference"]["product"]
    measured = cfg["measured_unit"]
    lines = [
        "Badminton Racket Asset",
        f"  reference product: {ref['manufacturer']} {ref['model']} ({ref['item_code']})",
        f"  published length: {float(ref['overall_length_m']) * 1000.0:.1f} mm",
        f"  published shaft diameter: {float(ref['shaft_diameter_m']) * 1000.0:.1f} mm",
        f"  published unstrung weight class: {ref['weight_class']} {ref['unstrung_mass_range_kg']} kg",
        f"  measured unit status: {measured.get('status')}",
    ]
    if measured.get("status") == "MEASURED":
        desc = build_physics_description(cfg)
        mass = compute_mass_properties(cfg)
        lines.extend(
            [
                f"  installed mass: {mass['mass_kg'] * 1000.0:.3f} g",
                f"  head outer: {desc['head']['outer_length_m']*1000.0:.2f} x {desc['head']['outer_width_m']*1000.0:.2f} mm",
                f"  stringbed: {desc['stringbed']['length_m']*1000.0:.2f} x {desc['stringbed']['width_m']*1000.0:.2f} mm",
                f"  contact center z: {desc['stringbed']['center_z_m']*1000.0:.2f} mm",
            ]
        )
    else:
        lines.append("  final physics: BLOCKED until the exact installed racket is measured")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="racket.yaml")
    parser.add_argument("--output", default="racket.usd")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--final", action="store_true", help="require exact measured-unit fields during validation")
    parser.add_argument("--allow-temp-visual", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg, final=args.final)
    if args.print_summary:
        print(_summary(cfg))
    if not args.validate_only:
        build_usd(cfg, args.output, allow_temp_visual=args.allow_temp_visual)


if __name__ == "__main__":
    main()
