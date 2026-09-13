#!/usr/bin/env python3
"""Build a traceable badminton shuttlecock physics asset for Isaac Sim.

The pure-Python geometry and mass-property functions can be tested without USD.
USD authoring is imported lazily and must run in an environment that provides
Pixar USD's ``pxr`` package, such as Isaac Sim Python.

Final visual policy:
- Default: reference an externally prepared CC Attribution shuttlecock mesh.
- If that asset is absent, final authoring fails instead of silently replacing it.
- ``--allow-temp-procedural-visual`` enables an explicitly TEMP debug visual.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def load_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read shuttlecock.yaml") from exc
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"shuttlecock config must be a YAML mapping: {p}")
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
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0.0 or hi <= lo:
        raise ValueError(f"{name} must satisfy 0 < min < max")
    return lo, hi


def _vec3(value: Any, name: str) -> Vec3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-element sequence")
    vec = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in vec):
        raise ValueError(f"{name} must contain finite values")
    return vec  # type: ignore[return-value]


def validate_config(cfg: Mapping[str, Any]) -> None:
    ref = _mapping(cfg, "reference")
    bwf = _mapping(ref, "bwf")
    selected = _mapping(ref, "selected_geometry")
    masses = _mapping(ref, "mass_distribution")
    aero = _mapping(ref, "aerodynamics")
    proxy = _mapping(cfg, "physics_proxy")
    visual = _mapping(cfg, "visual")
    contact = _mapping(cfg, "contact_pair_parameters")
    runtime = _mapping(cfg, "aerodynamics_runtime")
    frames = _mapping(cfg, "frames")

    feathers = ref.get("feathers_count")
    if feathers != 16:
        raise ValueError("reference.feathers_count must be 16 for the BWF feathered-shuttle profile")

    feather_range = _pair(bwf.get("feather_length_range_m"), "BWF feather length range")
    tip_range = _pair(bwf.get("skirt_tip_diameter_range_m"), "BWF skirt tip diameter range")
    cork_range = _pair(bwf.get("cork_diameter_range_m"), "BWF cork diameter range")
    mass_range = _pair(bwf.get("total_mass_range_kg"), "BWF total mass range")

    feather_length = _number(selected, "feather_length_m", positive=True)
    cork_diameter = _number(selected, "cork_diameter_m", positive=True)
    area = _number(selected, "reference_cross_section_area_m2", positive=True)
    tip_diameter = 2.0 * math.sqrt(area / math.pi)
    if not (feather_range[0] <= feather_length <= feather_range[1]):
        raise ValueError("selected feather length must lie inside the BWF range")
    if not (cork_range[0] <= cork_diameter <= cork_range[1]):
        raise ValueError("selected cork diameter must lie inside the BWF range")
    if not (tip_range[0] <= tip_diameter <= tip_range[1]):
        raise ValueError("area-derived skirt tip diameter must lie inside the BWF range")

    total = _number(masses, "total_mass_kg", positive=True)
    cork_mass = _number(masses, "cork_mass_kg", positive=True)
    skirt_mass = _number(masses, "skirt_mass_kg", positive=True)
    if not (mass_range[0] <= total <= mass_range[1]):
        raise ValueError("selected total mass must lie inside the BWF range")
    if abs((cork_mass + skirt_mass) - total) > 1e-12:
        raise ValueError("mass split must sum exactly to total_mass_kg")

    if aero.get("model") != "AERODYNAMIC_LENGTH":
        raise ValueError("reference.aerodynamics.model must be AERODYNAMIC_LENGTH in v0.1")
    length = _number(aero, "aerodynamic_length_m", positive=True)
    k = _number(aero, "quadratic_drag_k_per_m", positive=True)
    if not math.isclose(k, 1.0 / length, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("quadratic_drag_k_per_m must equal 1 / aerodynamic_length_m")

    if proxy.get("cork_shape") != "HEMISPHERE_CONVEX_HULL":
        raise ValueError("physics_proxy.cork_shape must be HEMISPHERE_CONVEX_HULL")
    if proxy.get("skirt_shape") != "OPEN_FRUSTUM_SHELL_SEGMENTS":
        raise ValueError("physics_proxy.skirt_shape must be OPEN_FRUSTUM_SHELL_SEGMENTS")
    for key, minimum in (("cork_rings", 2), ("cork_radial_segments", 8), ("skirt_collision_segments", 4)):
        value = proxy.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"physics_proxy.{key} must be integer >= {minimum}")
    _number(proxy, "skirt_shell_thickness_m", positive=True)

    if visual.get("mode") != "EXTERNAL_REFERENCE":
        raise ValueError("visual.mode must be EXTERNAL_REFERENCE for the final asset")
    if visual.get("required_for_final") is not True:
        raise ValueError("visual.required_for_final must be true")
    for key in ("local_asset_path", "source_url", "source_uid", "title", "author", "license"):
        if not isinstance(visual.get(key), str) or not str(visual.get(key)).strip():
            raise ValueError(f"visual.{key} must be a non-empty string")

    if contact.get("status") != "REQUIRES_PAIR_CALIBRATION":
        raise ValueError("contact_pair_parameters.status must be REQUIRES_PAIR_CALIBRATION")
    for key in ("shuttle_ground", "shuttle_net", "shuttle_racket"):
        if contact.get(key) is not None:
            raise ValueError(f"{key} must stay null until pair-specific calibration exists")

    gravity = _vec3(runtime.get("gravity_mps2"), "aerodynamics_runtime.gravity_mps2")
    _vec3(runtime.get("wind_mps"), "aerodynamics_runtime.wind_mps")
    _number(runtime, "rk4_internal_dt_s", positive=True)
    if gravity[2] >= 0.0:
        raise ValueError("gravity z must be negative in Court Frame")

    convention = frames.get("convention")
    if not isinstance(convention, str) or "+Z" not in convention:
        raise ValueError("frames.convention must document +Z shuttle axis")
    radius = cork_diameter / 2.0
    if not math.isclose(_number(frames, "cork_tip_z_m"), -radius, abs_tol=1e-12):
        raise ValueError("frames.cork_tip_z_m must equal -cork radius for hemisphere proxy")
    if not math.isclose(_number(frames, "skirt_tip_z_m"), feather_length, abs_tol=1e-12):
        raise ValueError("frames.skirt_tip_z_m must equal selected feather length")


def config_sha256(cfg: Mapping[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _skirt_shell_segment(index: int, count: int, root_radius: float, tip_radius: float, length: float, thickness: float) -> Dict[str, float | int]:
    half_angle = math.pi / count
    center_angle = 2.0 * math.pi * index / count
    root_inner = max(root_radius - thickness / 2.0, 1e-6)
    root_outer = root_radius + thickness / 2.0
    tip_inner = max(tip_radius - thickness / 2.0, root_inner + 1e-6)
    tip_outer = tip_radius + thickness / 2.0
    return {
        "index": index,
        "center_angle_rad": center_angle,
        "half_angle_rad": half_angle,
        "root_inner_radius_m": root_inner,
        "root_outer_radius_m": root_outer,
        "tip_inner_radius_m": tip_inner,
        "tip_outer_radius_m": tip_outer,
        "z_root_m": 0.0,
        "z_tip_m": length,
    }


def build_physics_description(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    validate_config(cfg)
    ref = cfg["reference"]
    selected = ref["selected_geometry"]
    proxy = cfg["physics_proxy"]
    cork_radius = float(selected["cork_diameter_m"]) / 2.0
    area = float(selected["reference_cross_section_area_m2"])
    tip_radius = math.sqrt(area / math.pi)
    length = float(selected["feather_length_m"])
    thickness = float(proxy["skirt_shell_thickness_m"])
    count = int(proxy["skirt_collision_segments"])

    return {
        "cork": {
            "radius_m": cork_radius,
            "z_min_m": -cork_radius,
            "z_max_m": 0.0,
            "shape": proxy["cork_shape"],
        },
        "skirt_root_radius_m": cork_radius,
        "skirt_tip_radius_m": tip_radius,
        "skirt_length_m": length,
        "skirt_collision_segments": [
            _skirt_shell_segment(i, count, cork_radius, tip_radius, length, thickness)
            for i in range(count)
        ],
        "frame": {
            "origin_m": (0.0, 0.0, 0.0),
            "cork_tip_m": (0.0, 0.0, -cork_radius),
            "skirt_axis_point_m": (0.0, 0.0, length),
        },
    }


def _thin_frustum_shell_mass_properties(mass: float, r0: float, r1: float, length: float) -> Dict[str, float]:
    """Axisymmetric thin conical-frustum shell with uniform surface density.

    Surface slope is constant, so the common ``sqrt(1 + (dr/dz)^2)`` factor
    cancels from all normalized axial moments.  The integration weight is
    therefore proportional to r(z) dz.
    """
    a = (r1 - r0) / length
    w0 = r0 * length + 0.5 * a * length**2
    w1 = 0.5 * r0 * length**2 + (a / 3.0) * length**3
    w2 = (r0 / 3.0) * length**3 + 0.25 * a * length**4
    if abs(a) < 1e-15:
        r3 = r0**3 * length
    else:
        r3 = (r1**4 - r0**4) / (4.0 * a)

    zc = w1 / w0
    mean_z2 = w2 / w0
    mean_r2 = r3 / w0
    iz = mass * mean_r2
    ix_origin = mass * (mean_z2 + 0.5 * mean_r2)
    ix_cm = ix_origin - mass * zc**2
    return {"zc": zc, "ix": ix_cm, "iy": ix_cm, "iz": iz}


def compute_mass_properties(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    validate_config(cfg)
    desc = build_physics_description(cfg)
    masses = cfg["reference"]["mass_distribution"]
    mc = float(masses["cork_mass_kg"])
    ms = float(masses["skirt_mass_kg"])
    total = float(masses["total_mass_kg"])
    r = float(desc["cork"]["radius_m"])
    r0 = float(desc["skirt_root_radius_m"])
    r1 = float(desc["skirt_tip_radius_m"])
    length = float(desc["skirt_length_m"])

    # Uniform solid hemisphere. The flat face is z=0 and the nose is z=-R.
    zc_cork = -3.0 * r / 8.0
    ix_cork = (83.0 / 320.0) * mc * r**2
    iy_cork = ix_cork
    iz_cork = (2.0 / 5.0) * mc * r**2

    skirt = _thin_frustum_shell_mass_properties(ms, r0, r1, length)
    zc_skirt = skirt["zc"]
    zc = (mc * zc_cork + ms * zc_skirt) / total

    ix = ix_cork + mc * (zc_cork - zc) ** 2 + skirt["ix"] + ms * (zc_skirt - zc) ** 2
    iy = iy_cork + mc * (zc_cork - zc) ** 2 + skirt["iy"] + ms * (zc_skirt - zc) ** 2
    iz = iz_cork + skirt["iz"]

    values = (ix, iy, iz)
    if not all(math.isfinite(v) and v > 0.0 for v in values):
        raise ValueError("computed inertia must be finite and positive")

    return {
        "mass_kg": total,
        "center_of_mass_m": (0.0, 0.0, zc),
        "diagonal_inertia_kg_m2": values,
        "component_centers_m": {
            "cork": (0.0, 0.0, zc_cork),
            "skirt": (0.0, 0.0, zc_skirt),
        },
        "method": "uniform solid hemisphere cork + uniform thin frustum shell skirt + parallel axis theorem",
    }


def geometry_summary(cfg: Mapping[str, Any]) -> str:
    desc = build_physics_description(cfg)
    mass = compute_mass_properties(cfg)
    aero = cfg["reference"]["aerodynamics"]
    return "\n".join(
        [
            "Badminton Shuttlecock v0.1",
            f"  mass: {mass['mass_kg'] * 1000.0:.3f} g",
            f"  cork diameter: {desc['cork']['radius_m'] * 2000.0:.3f} mm",
            f"  skirt tip diameter: {desc['skirt_tip_radius_m'] * 2000.0:.3f} mm",
            f"  feather/skirt length: {desc['skirt_length_m'] * 1000.0:.3f} mm",
            f"  skirt collider segments: {len(desc['skirt_collision_segments'])}",
            f"  COM z from cork flat face: {mass['center_of_mass_m'][2] * 1000.0:.3f} mm",
            "  inertia diag [kg m^2]: " + ", ".join(f"{x:.9e}" for x in mass["diagonal_inertia_kg_m2"]),
            f"  aerodynamic length L: {float(aero['aerodynamic_length_m']):.3f} m",
            f"  quadratic drag k: {float(aero['quadratic_drag_k_per_m']):.9f} 1/m",
            f"  visual: {cfg['visual']['title']} / {cfg['visual']['license']}",
            f"  contact pairs: {cfg['contact_pair_parameters']['status']}",
            f"  config sha256: {config_sha256(cfg)}",
        ]
    )


def _import_pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "USD authoring requires pxr. Run with Isaac Sim's Python environment."
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics


def _closed_mesh(stage, path: str, points: List[Vec3], faces: List[List[int]], *, collision: bool, visible: bool = True):
    Gf, _, _, UsdGeom, UsdPhysics = _import_pxr()
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    counts = [len(face) for face in faces]
    indices: List[int] = []
    for face in faces:
        indices.extend(face)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if not visible:
        UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        try:
            mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        except AttributeError:  # compatibility with some USD builds
            mesh_api.CreateApproximationAttr().Set("convexHull")
    return mesh.GetPrim()


def _hemisphere_mesh(radius: float, rings: int, segments: int) -> Tuple[List[Vec3], List[List[int]]]:
    points: List[Vec3] = [(0.0, 0.0, -radius)]
    ring_indices: List[List[int]] = []
    for i in range(1, rings + 1):
        theta = (math.pi / 2.0) * i / rings
        rr = radius * math.sin(theta)
        z = -radius * math.cos(theta)
        ring: List[int] = []
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            ring.append(len(points))
            points.append((rr * math.cos(phi), rr * math.sin(phi), z))
        ring_indices.append(ring)

    faces: List[List[int]] = []
    first = ring_indices[0]
    for j in range(segments):
        faces.append([0, first[(j + 1) % segments], first[j]])
    for a, b in zip(ring_indices[:-1], ring_indices[1:]):
        for j in range(segments):
            faces.append([a[j], a[(j + 1) % segments], b[(j + 1) % segments], b[j]])
    cap_center = len(points)
    points.append((0.0, 0.0, 0.0))
    last = ring_indices[-1]
    for j in range(segments):
        faces.append([cap_center, last[j], last[(j + 1) % segments]])
    return points, faces


def _frustum_shell_segment_mesh(seg: Mapping[str, Any]) -> Tuple[List[Vec3], List[List[int]]]:
    c = float(seg["center_angle_rad"])
    h = float(seg["half_angle_rad"])
    a0, a1 = c - h, c + h
    z0, z1 = float(seg["z_root_m"]), float(seg["z_tip_m"])
    ri0, ro0 = float(seg["root_inner_radius_m"]), float(seg["root_outer_radius_m"])
    ri1, ro1 = float(seg["tip_inner_radius_m"]), float(seg["tip_outer_radius_m"])

    def p(r: float, a: float, z: float) -> Vec3:
        return (r * math.cos(a), r * math.sin(a), z)

    points = [
        p(ri0, a0, z0), p(ro0, a0, z0), p(ro0, a1, z0), p(ri0, a1, z0),
        p(ri1, a0, z1), p(ro1, a0, z1), p(ro1, a1, z1), p(ri1, a1, z1),
    ]
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    return points, faces


def _author_frame(stage, path: str, translation: Vec3) -> None:
    Gf, _, _, UsdGeom, _ = _import_pxr()
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))


def _resolve_visual_path(cfg: Mapping[str, Any], output: Path) -> Path:
    configured = Path(str(cfg["visual"]["local_asset_path"]))
    if configured.is_absolute():
        return configured
    # First try relative to current working directory/repository, then output dir.
    cwd_candidate = (Path.cwd() / configured).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (output.parent / configured).resolve()


def _author_temp_visual(stage, desc: Mapping[str, Any]) -> None:
    Gf, Sdf, _, UsdGeom, _ = _import_pxr()
    visual = UsdGeom.Xform.Define(stage, "/Shuttlecock/Visual/TEMP_ProceduralDebug")
    prim = visual.GetPrim()
    prim.CreateAttribute("project:TEMP", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    prim.CreateAttribute("project:warning", Sdf.ValueTypeNames.String, custom=True).Set(
        "Debug visual only; replace with configured CC Attribution shuttle mesh before final use."
    )

    r = float(desc["cork"]["radius_m"])
    sphere = UsdGeom.Sphere.Define(stage, "/Shuttlecock/Visual/TEMP_ProceduralDebug/Cork")
    sphere.CreateRadiusAttr(r)
    xs = UsdGeom.Xformable(sphere.GetPrim())
    xs.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -r / 2.0))
    xs.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.5))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.70, 0.50)])

    cone = UsdGeom.Cone.Define(stage, "/Shuttlecock/Visual/TEMP_ProceduralDebug/SkirtEnvelope")
    cone.CreateAxisAttr(UsdGeom.Tokens.z)
    cone.CreateRadiusAttr(float(desc["skirt_tip_radius_m"]))
    cone.CreateHeightAttr(float(desc["skirt_length_m"]))
    xc = UsdGeom.Xformable(cone.GetPrim())
    xc.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(desc["skirt_length_m"]) / 2.0))
    cone.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.95)])
    cone.CreateDisplayOpacityAttr([0.25])


def build_usd(cfg: Mapping[str, Any], output_path: str | Path, *, allow_temp_visual: bool = False) -> Path:
    validate_config(cfg)
    desc = build_physics_description(cfg)
    mass = compute_mass_properties(cfg)
    Gf, Sdf, Usd, UsdGeom, UsdPhysics = _import_pxr()

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/Shuttlecock")
    stage.SetDefaultPrim(root.GetPrim())
    rp = root.GetPrim()
    rp.CreateAttribute("project:schemaVersion", Sdf.ValueTypeNames.String, custom=True).Set(str(cfg.get("schema_version", "1.0")))
    rp.CreateAttribute("project:configSha256", Sdf.ValueTypeNames.String, custom=True).Set(config_sha256(cfg))
    rp.CreateAttribute("project:frameConvention", Sdf.ValueTypeNames.String, custom=True).Set(str(cfg["frames"]["convention"]))
    rp.CreateAttribute("project:contactCalibrationStatus", Sdf.ValueTypeNames.String, custom=True).Set(str(cfg["contact_pair_parameters"]["status"]))

    UsdPhysics.RigidBodyAPI.Apply(rp)
    mass_api = UsdPhysics.MassAPI.Apply(rp)
    mass_api.CreateMassAttr(float(mass["mass_kg"]))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*mass["center_of_mass_m"]))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*mass["diagonal_inertia_kg_m2"]))

    UsdGeom.Scope.Define(stage, "/Shuttlecock/Visual")
    UsdGeom.Scope.Define(stage, "/Shuttlecock/Physics")
    UsdGeom.Scope.Define(stage, "/Shuttlecock/Physics/SkirtColliders")
    UsdGeom.Scope.Define(stage, "/Shuttlecock/Frames")

    visual_path = _resolve_visual_path(cfg, output)
    if visual_path.exists():
        ext = stage.DefinePrim("/Shuttlecock/Visual/External", "Xform")
        ext.GetReferences().AddReference(str(visual_path))
        ext.CreateAttribute("project:sourceUrl", Sdf.ValueTypeNames.String, custom=True).Set(str(cfg["visual"]["source_url"]))
        ext.CreateAttribute("project:author", Sdf.ValueTypeNames.String, custom=True).Set(str(cfg["visual"]["author"]))
        ext.CreateAttribute("project:license", Sdf.ValueTypeNames.String, custom=True).Set(str(cfg["visual"]["license"]))
    elif allow_temp_visual:
        _author_temp_visual(stage, desc)
    else:
        raise FileNotFoundError(
            "Final shuttle visual asset is missing: "
            f"{visual_path}. Prepare the configured CC Attribution shuttle-only USD, "
            "or use --allow-temp-procedural-visual only for explicit debug builds."
        )

    points, faces = _hemisphere_mesh(
        float(desc["cork"]["radius_m"]),
        int(cfg["physics_proxy"]["cork_rings"]),
        int(cfg["physics_proxy"]["cork_radial_segments"]),
    )
    _closed_mesh(stage, "/Shuttlecock/Physics/CorkCollider", points, faces, collision=True, visible=False)

    for seg in desc["skirt_collision_segments"]:
        points, faces = _frustum_shell_segment_mesh(seg)
        _closed_mesh(
            stage,
            f"/Shuttlecock/Physics/SkirtColliders/Segment_{int(seg['index']):02d}",
            points,
            faces,
            collision=True,
            visible=False,
        )

    _author_frame(stage, "/Shuttlecock/Frames/shuttle_com", tuple(mass["center_of_mass_m"]))
    _author_frame(stage, "/Shuttlecock/Frames/cork_tip", tuple(desc["frame"]["cork_tip_m"]))
    _author_frame(stage, "/Shuttlecock/Frames/skirt_axis", tuple(desc["frame"]["skirt_axis_point_m"]))

    stage.GetRootLayer().Save()
    return output


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("shuttlecock.yaml"))
    p.add_argument("--output", type=Path, default=Path("shuttlecock.usda"))
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--print-summary", action="store_true")
    p.add_argument("--allow-temp-procedural-visual", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    if args.print_summary:
        print(geometry_summary(cfg))
    if not args.validate_only:
        out = build_usd(cfg, args.output, allow_temp_visual=args.allow_temp_procedural_visual)
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
