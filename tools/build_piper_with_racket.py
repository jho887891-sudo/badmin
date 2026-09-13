#!/usr/bin/env python3
"""Compose the frozen PiPER no-gripper USD with a measured racket asset."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

Mat4 = Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]


def load_mount_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read racket.yaml") from exc
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"mount config must be a YAML mapping: {p}")
    return copy.deepcopy(data)


def _vec(value: Any, n: int, name: str) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != n:
        raise ValueError(f"{name} must be a {n}-element sequence")
    result = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def validate_mount_config(cfg: Mapping[str, Any], *, final: bool = False) -> None:
    attach = cfg.get("piper_attachment")
    if not isinstance(attach, Mapping):
        raise ValueError("missing piper_attachment mapping")
    for key in ("piper_usd_path", "piper_destination_prim_path", "piper_link6_prim_path", "racket_usd_path", "racket_root_prim_path"):
        if not isinstance(attach.get(key), str) or not attach.get(key):
            raise ValueError(f"piper_attachment.{key} must be a non-empty string")
    mount = attach.get("mount_transform")
    if not isinstance(mount, Mapping):
        raise ValueError("missing piper_attachment.mount_transform mapping")
    if final and mount.get("status") != "MEASURED":
        raise ValueError("mount_transform is incomplete: measured T_link6_tcp is required for final authoring")
    if mount.get("status") == "MEASURED":
        _vec(mount.get("translation_m"), 3, "mount translation")
        q = _vec(mount.get("quaternion_xyzw"), 4, "mount quaternion")
        qn = math.sqrt(sum(v * v for v in q))
        if qn <= 1e-12:
            raise ValueError("mount quaternion norm must be non-zero")


def _normalized_quaternion_xyzw(cfg: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    validate_mount_config(cfg, final=True)
    q = _vec(cfg["piper_attachment"]["mount_transform"]["quaternion_xyzw"], 4, "mount quaternion")
    n = math.sqrt(sum(v * v for v in q))
    if n <= 1e-12:
        raise ValueError("mount quaternion norm must be non-zero")
    return tuple(v / n for v in q)  # type: ignore[return-value]


def compose_link6_tcp_transform(cfg: Mapping[str, Any]) -> Mat4:
    validate_mount_config(cfg, final=True)
    t = _vec(cfg["piper_attachment"]["mount_transform"]["translation_m"], 3, "mount translation")
    x, y, z, w = _normalized_quaternion_xyzw(cfg)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), t[0]),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), t[1]),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), t[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def build_piper_with_racket(
    cfg: Mapping[str, Any],
    output_path: str | Path,
    *,
    piper_usd_path: str | Path | None = None,
    racket_usd_path: str | Path | None = None,
) -> Path:
    validate_mount_config(cfg, final=True)
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pxr/USD is required for USD authoring; run this under Isaac Sim Python") from exc

    attach = cfg["piper_attachment"]
    piper_asset = Path(piper_usd_path or attach["piper_usd_path"])
    racket_asset = Path(racket_usd_path or attach["racket_usd_path"])
    if not piper_asset.exists():
        raise FileNotFoundError(f"frozen PiPER USD not found: {piper_asset}")
    if not racket_asset.exists():
        raise FileNotFoundError(f"racket USD not found: {racket_asset}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())

    piper_dst = str(attach["piper_destination_prim_path"])
    piper = UsdGeom.Xform.Define(stage, piper_dst)
    piper.GetPrim().GetReferences().AddReference(str(piper_asset))

    racket_dst = str(attach["racket_root_prim_path"])
    racket = UsdGeom.Xform.Define(stage, racket_dst)
    racket.GetPrim().GetReferences().AddReference(str(racket_asset))

    joint = UsdPhysics.FixedJoint.Define(stage, "/Robot/Joints/racket_fixed_joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(str(attach["piper_link6_prim_path"]))])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(racket_dst)])

    t = _vec(attach["mount_transform"]["translation_m"], 3, "mount translation")
    x, y, z, w = _normalized_quaternion_xyzw(cfg)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*t))
    joint.CreateLocalRot0Attr(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))

    joint.GetPrim().SetCustomDataByKey("mount_convention", str(attach["mount_transform"].get("convention", "T_link6_tcp")))

    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(out))
    if reopened is None:
        raise RuntimeError(f"failed to reopen authored stage: {out}")
    if not reopened.GetPrimAtPath(str(attach["piper_link6_prim_path"])).IsValid():
        raise ValueError(
            f"composed PiPER stage does not contain configured link6 prim {attach['piper_link6_prim_path']}; "
            "check the frozen PiPER asset hierarchy before using the wrapper"
        )
    if not reopened.GetPrimAtPath(racket_dst).IsValid():
        raise ValueError(f"composed stage does not contain racket root prim {racket_dst}")
    return out


def _summary(cfg: Mapping[str, Any]) -> str:
    attach = cfg["piper_attachment"]
    mount = attach["mount_transform"]
    lines = [
        "PiPER + Racket Attachment",
        f"  PiPER asset: {attach['piper_usd_path']}",
        f"  link6 prim: {attach['piper_link6_prim_path']}",
        f"  racket asset: {attach['racket_usd_path']}",
        f"  mount status: {mount.get('status')}",
    ]
    if mount.get("status") == "MEASURED":
        lines.append(f"  T_link6_tcp: {compose_link6_tcp_transform(cfg)}")
    else:
        lines.append("  final wrapper: BLOCKED until the physical adapter transform is measured")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="racket.yaml")
    parser.add_argument("--output", default="piper_with_racket.usd")
    parser.add_argument("--piper-usd")
    parser.add_argument("--racket-usd")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    cfg = load_mount_config(args.config)
    validate_mount_config(cfg, final=args.final)
    if args.print_summary:
        print(_summary(cfg))
    if not args.validate_only:
        build_piper_with_racket(cfg, args.output, piper_usd_path=args.piper_usd, racket_usd_path=args.racket_usd)


if __name__ == "__main__":
    main()
