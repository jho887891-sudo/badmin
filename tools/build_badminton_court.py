#!/usr/bin/env python3
"""Build a parameterized badminton court USD asset.

The geometry calculations are pure Python and can be unit-tested without Isaac Sim.
USD authoring is loaded lazily and therefore must run under an environment that
provides Pixar USD's ``pxr`` package (for example Isaac Sim's Python).

Typical usage inside Isaac Sim Python:

    python build_badminton_court.py \
        --config court.yaml \
        --output badminton_court.usd

Validation only (works in ordinary Python with PyYAML):

    python build_badminton_court.py --config court.yaml --validate-only
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Geometry = Dict[str, Any]


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config and return an independent mutable dictionary."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "PyYAML is required to read court.yaml. "
            "Install PyYAML or run inside the project's Isaac Sim Python."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"court config must be a YAML mapping: {config_path}")

    return copy.deepcopy(data)


def _require_mapping(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid mapping: {key}")
    return value


def _require_number(mapping: Mapping[str, Any], key: str, *, positive: bool = False) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{key} must be > 0")
    return value


def _require_rgb(mapping: Mapping[str, Any], key: str) -> Tuple[float, float, float]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{key} must be a 3-element RGB sequence")
    rgb = tuple(float(v) for v in value)
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in rgb):
        raise ValueError(f"{key} values must be finite and within [0, 1]")
    return rgb  # type: ignore[return-value]


def validate_config(cfg: Mapping[str, Any]) -> None:
    """Validate geometry invariants before any USD prim is authored."""
    court = _require_mapping(cfg, "court")
    floor = _require_mapping(cfg, "surrounding_floor")
    lines = _require_mapping(cfg, "lines")
    net = _require_mapping(cfg, "net")
    appearance = _require_mapping(cfg, "appearance")
    physics = _require_mapping(cfg, "physics")
    frames = _require_mapping(cfg, "frames")

    length = _require_number(court, "length_m", positive=True)
    width = _require_number(court, "width_m", positive=True)
    singles_width = _require_number(court, "singles_width_m", positive=True)
    line_width = _require_number(court, "line_width_m", positive=True)
    short_x = _require_number(court, "short_service_x_m", positive=True)
    doubles_long_x = _require_number(court, "doubles_long_service_x_m", positive=True)

    if court.get("line_reference") != "CENTERLINE":
        raise ValueError("court.line_reference must be CENTERLINE for Robot Brain v0.1")

    half_length = length / 2.0
    if singles_width >= width:
        raise ValueError("singles_width_m must be smaller than court width_m")
    if short_x >= half_length:
        raise ValueError("short_service_x_m must be inside the half court")
    if doubles_long_x <= short_x or doubles_long_x >= half_length:
        raise ValueError("doubles_long_service_x_m must lie between short service and baseline")

    floor_length = _require_number(floor, "length_m", positive=True)
    floor_width = _require_number(floor, "width_m", positive=True)
    _require_number(floor, "visual_thickness_m", positive=True)
    _require_number(floor, "collider_thickness_m", positive=True)
    if floor_length < length or floor_width < width:
        raise ValueError("surrounding floor must fully contain the court")

    surface = cfg.get("court_surface")
    if surface is not None:
        surface = _require_mapping(cfg, "court_surface")
        _require_number(surface, "thickness_m", positive=True)
        cof = _require_number(surface, "measured_reference_cof")
        if not 0.0 < cof <= 1.0:
            raise ValueError("court_surface.measured_reference_cof must be within (0, 1]")
        shock = _require_number(surface, "shock_absorption")
        if not 0.0 <= shock <= 1.0:
            raise ValueError("court_surface.shock_absorption must be within [0, 1]")

    _require_number(lines, "visual_thickness_m", positive=True)

    net_width = _require_number(net, "width_m", positive=True)
    center_height = _require_number(net, "center_height_m", positive=True)
    post_height = _require_number(net, "post_height_m", positive=True)
    vertical_depth = _require_number(net, "vertical_depth_m", positive=True)
    _require_number(net, "visual_thickness_m", positive=True)
    _require_number(net, "collider_thickness_m", positive=True)
    _tape_key = "top_tape_width_m" if "top_tape_width_m" in net else "top_tape_height_m"
    _require_number(net, _tape_key, positive=True)
    _require_number(net, "top_tape_thickness_m", positive=True)
    if "mesh_size_m" in net:
        _require_number(net, "mesh_size_m", positive=True)
    _require_number(net, "post_radius_m", positive=True)
    min_bottom = _require_number(net, "minimum_allowed_bottom_z_m")

    segments = net.get("segments")
    if not isinstance(segments, int) or isinstance(segments, bool) or segments < 2:
        raise ValueError("net.segments must be an integer >= 2")
    if segments % 2 != 0:
        raise ValueError("net.segments must be even so y=0 is a profile sample")
    if net.get("sag_profile") != "QUADRATIC":
        raise ValueError("net.sag_profile must be QUADRATIC in Scene v0.1")

    if abs(net_width - width) > 1e-9:
        raise ValueError("net.width_m must match court.width_m")

    net_bottom_at_center = center_height - vertical_depth
    if net_bottom_at_center < min_bottom:
        raise ValueError(
            "net bottom would be below minimum_allowed_bottom_z_m; "
            "this risks recreating the ground-to-net invisible wall"
        )
    if post_height < center_height:
        raise ValueError("net.post_height_m must be >= net.center_height_m")

    for key in ("floor_rgb", "court_line_rgb", "net_rgb", "top_tape_rgb", "post_rgb"):
        _require_rgb(appearance, key)
    opacity = _require_number(appearance, "net_opacity")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("appearance.net_opacity must be within [0, 1]")

    for frame_key in ("robot_home_xyz_m", "opponent_home_xyz_m"):
        value = frames.get(frame_key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
            raise ValueError(f"frames.{frame_key} must be a 3-element sequence")
        if not all(math.isfinite(float(v)) for v in value):
            raise ValueError(f"frames.{frame_key} must contain finite values")

    for material_name in ("ground_material", "net_material"):
        material = _require_mapping(physics, material_name)
        enabled = material.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"physics.{material_name}.enabled must be boolean")
        if enabled:
            for key in ("static_friction", "dynamic_friction"):
                value = _require_number(material, key)
                if value < 0.0:
                    raise ValueError(f"physics.{material_name}.{key} must be >= 0")
            restitution = material.get("restitution")
            if restitution is not None:
                restitution = float(restitution)
                if restitution < 0.0 or restitution > 1.0:
                    raise ValueError(f"physics.{material_name}.restitution must be within [0, 1]")


def _line(
    name: str,
    center: Vec3,
    size: Vec3,
    rgb: Tuple[float, float, float],
) -> Dict[str, Any]:
    return {
        "name": name,
        "center": center,
        "size": size,
        "rgb": rgb,
    }


def _quadratic_net_top(y: float, half_width: float, center_height: float, post_height: float) -> float:
    """Symmetric parabola: center sag, post height at both edges."""
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    normalized = abs(y) / half_width
    return center_height + (post_height - center_height) * normalized * normalized


def build_geometry(cfg: Mapping[str, Any]) -> Geometry:
    """Build deterministic court geometry in Court Frame without importing USD."""
    validate_config(cfg)

    court = cfg["court"]
    floor_cfg = cfg["surrounding_floor"]
    lines_cfg = cfg["lines"]
    net_cfg = cfg["net"]
    frames_cfg = cfg["frames"]
    appearance = cfg["appearance"]

    length = float(court["length_m"])
    width = float(court["width_m"])
    singles_width = float(court["singles_width_m"])
    line_width = float(court["line_width_m"])
    short_x = float(court["short_service_x_m"])
    doubles_long_x = float(court["doubles_long_service_x_m"])

    half_length = length / 2.0
    half_width = width / 2.0
    singles_half_width = singles_width / 2.0
    line_z = float(lines_cfg["visual_thickness_m"]) / 2.0
    line_thickness_z = float(lines_cfg["visual_thickness_m"])
    line_rgb = tuple(float(v) for v in appearance["court_line_rgb"])

    floor_visual_thickness = float(floor_cfg["visual_thickness_m"])
    floor_collider_thickness = float(floor_cfg["collider_thickness_m"])
    floor_size_xy = (float(floor_cfg["length_m"]), float(floor_cfg["width_m"]))

    surface_cfg = cfg.get("court_surface") or {}
    mat_thickness = float(surface_cfg.get("thickness_m", 0.0) or 0.0)
    floor_visual = {
        "name": "Floor",
        "center": (0.0, 0.0, -mat_thickness - floor_visual_thickness / 2.0),
        "size": (floor_size_xy[0], floor_size_xy[1], floor_visual_thickness),
        "rgb": tuple(float(v) for v in appearance["floor_rgb"]),
    }
    court_mat = None
    if mat_thickness > 0.0:
        court_mat = {
            "name": "CourtMat",
            "center": (0.0, 0.0, -mat_thickness / 2.0),
            "size": (length, width, mat_thickness),
            "rgb": tuple(float(v) for v in appearance["floor_rgb"]),
            "surface_type": str(surface_cfg.get("type", "UNKNOWN")),
        }
    ground_collider = {
        "name": "GroundCollider",
        "center": (0.0, 0.0, -floor_collider_thickness / 2.0),
        "size": (floor_size_xy[0], floor_size_xy[1], floor_collider_thickness),
    }

    # Court-line coordinates follow the project's centerline convention.
    lines: List[Dict[str, Any]] = [
        _line(
            "Baseline_Robot",
            (-half_length, 0.0, line_z),
            (line_width, width + line_width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "Baseline_Opponent",
            (half_length, 0.0, line_z),
            (line_width, width + line_width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "DoublesSideline_Left",
            (0.0, half_width, line_z),
            (length + line_width, line_width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "DoublesSideline_Right",
            (0.0, -half_width, line_z),
            (length + line_width, line_width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "SinglesSideline_Left",
            (0.0, singles_half_width, line_z),
            (length, line_width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "SinglesSideline_Right",
            (0.0, -singles_half_width, line_z),
            (length, line_width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "ShortService_Robot",
            (-short_x, 0.0, line_z),
            (line_width, width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "ShortService_Opponent",
            (short_x, 0.0, line_z),
            (line_width, width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "DoublesLongService_Robot",
            (-doubles_long_x, 0.0, line_z),
            (line_width, width, line_thickness_z),
            line_rgb,
        ),
        _line(
            "DoublesLongService_Opponent",
            (doubles_long_x, 0.0, line_z),
            (line_width, width, line_thickness_z),
            line_rgb,
        ),
    ]

    center_line_length = half_length - short_x
    center_line_offset = (half_length + short_x) / 2.0
    lines.extend(
        [
            _line(
                "CenterLine_Robot",
                (-center_line_offset, 0.0, line_z),
                (center_line_length, line_width, line_thickness_z),
                line_rgb,
            ),
            _line(
                "CenterLine_Opponent",
                (center_line_offset, 0.0, line_z),
                (center_line_length, line_width, line_thickness_z),
                line_rgb,
            ),
        ]
    )

    net_width = float(net_cfg["width_m"])
    net_half_width = net_width / 2.0
    center_height = float(net_cfg["center_height_m"])
    post_height = float(net_cfg["post_height_m"])
    vertical_depth = float(net_cfg["vertical_depth_m"])
    n_segments = int(net_cfg["segments"])
    visual_thickness = float(net_cfg["visual_thickness_m"])
    collider_thickness = float(net_cfg["collider_thickness_m"])
    _tape_key = "top_tape_width_m" if "top_tape_width_m" in net_cfg else "top_tape_height_m"
    tape_height = float(net_cfg[_tape_key])
    mesh_size = float(net_cfg.get("mesh_size_m", 0.0) or 0.0)
    tape_thickness = float(net_cfg["top_tape_thickness_m"])
    dy = net_width / n_segments

    # Profile samples include both posts and exact center (segments is even).
    profile_samples: List[Dict[str, float]] = []
    for i in range(n_segments + 1):
        y = -net_half_width + i * dy
        top_z = _quadratic_net_top(y, net_half_width, center_height, post_height)
        profile_samples.append({"y": y, "top_z": top_z})

    segments: List[Dict[str, Any]] = []
    top_tape_segments: List[Dict[str, Any]] = []
    for i in range(n_segments):
        y0 = -net_half_width + i * dy
        y1 = y0 + dy
        y_mid = (y0 + y1) / 2.0
        top_z = _quadratic_net_top(y_mid, net_half_width, center_height, post_height)
        bottom_z = top_z - vertical_depth
        z_center = (top_z + bottom_z) / 2.0

        segments.append(
            {
                "name": f"NetSegment_{i:02d}",
                "center": (0.0, y_mid, z_center),
                "size": (collider_thickness, dy, vertical_depth),
                "visual_size": (visual_thickness, dy, vertical_depth),
                "z_top": top_z,
                "z_bottom": bottom_z,
                "rgb": tuple(float(v) for v in appearance["net_rgb"]),
                "opacity": float(appearance["net_opacity"]),
            }
        )
        top_tape_segments.append(
            {
                "name": f"TopTape_{i:02d}",
                "center": (0.0, y_mid, top_z - tape_height / 2.0),
                "size": (tape_thickness, dy, tape_height),
                "rgb": tuple(float(v) for v in appearance["top_tape_rgb"]),
            }
        )

    post_radius = float(net_cfg["post_radius_m"])
    posts = [
        {
            "name": "NetPost_Left",
            "center": (0.0, net_half_width, post_height / 2.0),
            "radius_m": post_radius,
            "height_m": post_height,
            "rgb": tuple(float(v) for v in appearance["post_rgb"]),
        },
        {
            "name": "NetPost_Right",
            "center": (0.0, -net_half_width, post_height / 2.0),
            "radius_m": post_radius,
            "height_m": post_height,
            "rgb": tuple(float(v) for v in appearance["post_rgb"]),
        },
    ]

    robot_home = tuple(float(v) for v in frames_cfg["robot_home_xyz_m"])
    opponent_home = tuple(float(v) for v in frames_cfg["opponent_home_xyz_m"])
    frames: Dict[str, Vec3] = {
        "court": (0.0, 0.0, 0.0),
        "net_center": (0.0, 0.0, 0.0),
        "net_post_left": (0.0, net_half_width, 0.0),
        "net_post_right": (0.0, -net_half_width, 0.0),
        "robot_home": robot_home,  # type: ignore[assignment]
        "opponent_home": opponent_home,  # type: ignore[assignment]
    }

    return {
        "floor_visual": floor_visual,
        "court_mat": court_mat,
        "ground_collider": ground_collider,
        "lines": lines,
        "net": {
            "mesh_size_m": mesh_size,
            "segments": segments,
            "top_tape_segments": top_tape_segments,
            "profile_samples": profile_samples,
        },
        "posts": posts,
        "frames": frames,
    }


def config_sha256(cfg: Mapping[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def geometry_summary(cfg: Mapping[str, Any], geometry: Mapping[str, Any]) -> str:
    net_segments = geometry["net"]["segments"]
    min_bottom = min(seg["z_bottom"] for seg in net_segments)
    max_top = max(seg["z_top"] for seg in net_segments)
    return "\n".join(
        [
            "Badminton Court Geometry",
            f"  court: {cfg['court']['length_m']:.3f} m x {cfg['court']['width_m']:.3f} m",
            f"  singles width: {cfg['court']['singles_width_m']:.3f} m",
            f"  line width: {cfg['court']['line_width_m']:.3f} m",
            f"  net segments: {len(net_segments)}",
            f"  net bottom min: {min_bottom:.3f} m",
            f"  net top max: {max_top:.3f} m",
            f"  lines: {len(geometry['lines'])}",
            f"  config sha256: {config_sha256(cfg)}",
        ]
    )


def _import_pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - requires Isaac/USD
        raise RuntimeError(
            "USD authoring requires the 'pxr' package. "
            "Run this script with Isaac Sim's Python environment."
        ) from exc
    return Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _author_cube(stage, path: str, center: Vec3, size: Vec3, rgb=None, opacity: float = 1.0):
    Gf, _, _, UsdGeom, _, _ = _import_pxr()
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
    xformable.AddScaleOp().Set(Gf.Vec3f(*size))
    if rgb is not None:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        cube.CreateDisplayOpacityAttr([float(opacity)])
    return cube.GetPrim()


def _author_cylinder(stage, path: str, center: Vec3, radius_m: float, height_m: float, rgb=None):
    Gf, _, _, UsdGeom, _, _ = _import_pxr()
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(radius_m))
    cylinder.CreateHeightAttr(float(height_m))
    xformable = UsdGeom.Xformable(cylinder.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
    if rgb is not None:
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    return cylinder.GetPrim()


def _make_invisible(prim) -> None:
    _, _, _, UsdGeom, _, _ = _import_pxr()
    UsdGeom.Imageable(prim).MakeInvisible()


def _apply_collision(prim) -> None:
    _, _, _, _, UsdPhysics, _ = _import_pxr()
    UsdPhysics.CollisionAPI.Apply(prim)


def _author_physics_material(stage, path: str, spec: Mapping[str, Any]):
    """Create a USD physics material only when calibrated values are enabled."""
    if not bool(spec["enabled"]):
        return None

    _, _, _, _, UsdPhysics, UsdShade = _import_pxr()
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(spec["static_friction"]))
    api.CreateDynamicFrictionAttr(float(spec["dynamic_friction"]))
    if spec.get("restitution") is not None:
        api.CreateRestitutionAttr(float(spec["restitution"]))
    return material


def _bind_physics_material(prim, material) -> None:
    if material is None:
        return
    _, _, _, _, _, UsdShade = _import_pxr()
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def _author_frame(stage, path: str, translation: Vec3) -> None:
    Gf, _, _, UsdGeom, _, _ = _import_pxr()
    frame = UsdGeom.Xform.Define(stage, path)
    frame.AddTranslateOp().Set(Gf.Vec3d(*translation))


def build_usd(cfg: Mapping[str, Any], output_path: str | Path) -> Path:
    """Author the court as a USD asset and return the written path."""
    validate_config(cfg)
    geometry = build_geometry(cfg)

    Gf, Sdf, Usd, UsdGeom, _, _ = _import_pxr()

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/BadmintonCourt")
    stage.SetDefaultPrim(root.GetPrim())

    root_prim = root.GetPrim()
    root_prim.CreateAttribute(
        "project:schemaVersion",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(str(cfg.get("schema_version", "1.0")))
    root_prim.CreateAttribute(
        "project:coordinateConvention",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set("CourtFrame: origin=net_center_ground,+X=robot_to_opponent,+Y=left,+Z=up")
    root_prim.CreateAttribute(
        "project:configSha256",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(config_sha256(cfg))

    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Visual")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Visual/CourtLines")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Visual/Net")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Visual/Posts")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Physics")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Physics/NetColliders")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Physics/PostColliders")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Physics/Materials")
    UsdGeom.Scope.Define(stage, "/BadmintonCourt/Frames")

    floor = geometry["floor_visual"]
    _author_cube(
        stage,
        "/BadmintonCourt/Visual/Floor",
        floor["center"],
        floor["size"],
        rgb=floor["rgb"],
    )

    if geometry.get("court_mat"):
        mat = geometry["court_mat"]
        _author_cube(
            stage,
            "/BadmintonCourt/Visual/CourtMat",
            mat["center"],
            mat["size"],
            rgb=mat["rgb"],
        )

    for line in geometry["lines"]:
        _author_cube(
            stage,
            f"/BadmintonCourt/Visual/CourtLines/{line['name']}",
            line["center"],
            line["size"],
            rgb=line["rgb"],
        )

    for seg in geometry["net"]["segments"]:
        visual_size = seg["visual_size"]
        _author_cube(
            stage,
            f"/BadmintonCourt/Visual/Net/{seg['name']}",
            seg["center"],
            visual_size,
            rgb=seg["rgb"],
            opacity=seg["opacity"],
        )

    for tape in geometry["net"]["top_tape_segments"]:
        _author_cube(
            stage,
            f"/BadmintonCourt/Visual/Net/{tape['name']}",
            tape["center"],
            tape["size"],
            rgb=tape["rgb"],
        )

    for post in geometry["posts"]:
        _author_cylinder(
            stage,
            f"/BadmintonCourt/Visual/Posts/{post['name']}",
            post["center"],
            post["radius_m"],
            post["height_m"],
            rgb=post["rgb"],
        )

    ground_material = _author_physics_material(
        stage,
        "/BadmintonCourt/Physics/Materials/GroundMaterial",
        cfg["physics"]["ground_material"],
    )
    net_material = _author_physics_material(
        stage,
        "/BadmintonCourt/Physics/Materials/NetMaterial",
        cfg["physics"]["net_material"],
    )

    ground = geometry["ground_collider"]
    ground_prim = _author_cube(
        stage,
        "/BadmintonCourt/Physics/GroundCollider",
        ground["center"],
        ground["size"],
    )
    _apply_collision(ground_prim)
    _make_invisible(ground_prim)
    _bind_physics_material(ground_prim, ground_material)

    for seg in geometry["net"]["segments"]:
        collider_prim = _author_cube(
            stage,
            f"/BadmintonCourt/Physics/NetColliders/{seg['name']}",
            seg["center"],
            seg["size"],
        )
        _apply_collision(collider_prim)
        _make_invisible(collider_prim)
        _bind_physics_material(collider_prim, net_material)

    # Post collision uses the same geometry as the visible post.
    for post in geometry["posts"]:
        collider_prim = _author_cylinder(
            stage,
            f"/BadmintonCourt/Physics/PostColliders/{post['name']}",
            post["center"],
            post["radius_m"],
            post["height_m"],
        )
        _apply_collision(collider_prim)
        _make_invisible(collider_prim)
        _bind_physics_material(collider_prim, net_material)

    for name, translation in geometry["frames"].items():
        _author_frame(
            stage,
            f"/BadmintonCourt/Frames/{name}",
            translation,
        )

    stage.GetRootLayer().Save()
    return output


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=here / "court.yaml",
        help="Path to court YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "badminton_court.usd",
        help="Output USD/USDA/USDC path.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config/geometry without importing pxr or writing USD.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print deterministic geometry summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    geometry = build_geometry(cfg)

    if args.print_summary or args.validate_only:
        print(geometry_summary(cfg, geometry))

    if args.validate_only:
        return 0

    output = build_usd(cfg, args.output)
    print(f"Wrote USD: {output}")
    print(geometry_summary(cfg, geometry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
