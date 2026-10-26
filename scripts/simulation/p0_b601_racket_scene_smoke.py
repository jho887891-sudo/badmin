#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 Isaac Lab smoke test: fixed-base B601-RS + racket visual/collision proxy.

This is a SIMULATION-ONLY smoke test. It does not launch hardware, create a shuttle,
claim a measured racket mount, or validate physical contact forces.

Example (explicit test-only identity mount):
  ./isaaclab.sh -p scripts/simulation/p0_b601_racket_scene_smoke.py \
    --headless --robot-usd /absolute/path/to/ReBot_Arm_RS.usda \
    --test-identity-mount --steps 120 \
    --output outputs/simulation/rebot_b601_rs/morph_one/p6_frames/evidence/p0_scene_smoke.json

The identity transform is permitted only with --test-identity-mount. For a calibrated
mount, supply --mount-status MEASURED, --mount-translation-m X Y Z,
--mount-quaternion-wxyz W X Y Z, and --mount-source with traceable measurement evidence.
The output path is exclusive-create: an existing report is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_mount_record(args):
    if args.test_identity_mount:
        if any((args.mount_status, args.mount_translation_m,
                args.mount_quaternion_wxyz, args.mount_source)):
            raise ValueError("--test-identity-mount cannot be combined with explicit mount fields")
        return {
            "status": "SIMULATION_ASSUMPTION",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "source": "TEST ONLY: explicit identity mount for scene-load smoke test; not physical calibration",
        }
    required = {
        "mount-status": args.mount_status,
        "mount-translation-m": args.mount_translation_m,
        "mount-quaternion-wxyz": args.mount_quaternion_wxyz,
        "mount-source": args.mount_source,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "supply all calibrated/simulation mount fields or explicitly use --test-identity-mount; "
            "missing: " + ", ".join(missing)
        )
    return {
        "status": args.mount_status,
        "translation_m": list(args.mount_translation_m),
        "quaternion_wxyz": list(args.mount_quaternion_wxyz),
        "source": args.mount_source,
    }


def write_report_exclusive(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def parse_args(argv=None):
    # AppLauncher only configures Kit; the application is not launched until after
    # the input paths and the explicit mount evidence have been validated.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", required=True, help="verified B601-RS USD/USDA asset path")
    parser.add_argument("--racket-usd", default=str(ROOT / "assets/third_party/racket_visual.usd"))
    parser.add_argument(
        "--collision-proxy",
        default=str(ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/racket_collision_proxy.json"),
    )
    parser.add_argument("--output", required=True, help="new JSON evidence path; existing files are never overwritten")
    parser.add_argument("--steps", type=int, default=120, help="simulation steps after reset (default: 120)")
    parser.add_argument("--test-identity-mount", action="store_true",
                        help="explicitly opt into an identity SIMULATION_ASSUMPTION; never a physical calibration")
    parser.add_argument("--mount-status", choices=("MEASURED", "SIMULATION_ASSUMPTION"))
    parser.add_argument("--mount-translation-m", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--mount-quaternion-wxyz", nargs=4, type=float, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--mount-source", help="traceable measurement or simulation-assumption provenance")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = getattr(args, "device", "cuda:0") or "cuda:0"
    robot_usd = Path(args.robot_usd).expanduser().resolve()
    racket_usd = Path(args.racket_usd).expanduser().resolve()
    collision_proxy = Path(args.collision_proxy).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if output.exists():
        raise FileExistsError("refusing to overwrite existing evidence: " + str(output))
    for label, path in (("robot USD", robot_usd), ("racket USD", racket_usd),
                        ("collision proxy JSON", collision_proxy)):
        if not path.is_file():
            raise FileNotFoundError("%s does not exist: %s" % (label, path))

    from badminton_scene.racket_mount_adapter import (
        attach_racket_visual,
        gripper_end_prim_path,
        load_racket_collision_proxy,
        racket_mount_prim_path,
        resolve_racket_visual_path,
        validate_mount_transform,
    )
    mount = make_mount_record(args)
    validate_mount_transform(mount)
    proxy_doc = load_racket_collision_proxy(str(collision_proxy))
    if resolve_racket_visual_path(str(racket_usd)) != racket_usd:
        raise RuntimeError("racket asset path resolution mismatch")

    report = {
        "schema": "p0_b601_racket_scene_smoke/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
        "scope": "Isaac Lab single-arm fixed-base scene-load and short-step smoke test",
        "not_claimed": [
            "physical racket mount calibration",
            "Morph One chassis integration",
            "racket rigid-body mass or inertia",
            "validated shuttle-racket contact dynamics",
            "hardware deployment or real robot movement",
        ],
        "inputs": {
            "robot_usd": str(robot_usd),
            "robot_usd_sha256": sha256_file(robot_usd),
            "racket_usd": str(racket_usd),
            "racket_usd_sha256": sha256_file(racket_usd),
            "collision_proxy_json": str(collision_proxy),
            "collision_proxy_sha256": sha256_file(collision_proxy),
            "mount_transform": mount,
            "mount_transform_is_physical_calibration": mount["status"] == "MEASURED",
            "steps_requested": args.steps,
            "device": device,
        },
        "expected_collision_proxy_primitives": int(proxy_doc["primitive_count"]),
        "checks": {},
    }

    app = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(args).app

        import omni.usd
        import torch
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdGeom, UsdPhysics, Gf

        from badminton_scene.robot_cfg import REBOT_RS_PRIM_PATH, spawn_rebot_rs_directly
        from badminton_scene.rs_physics import rebot_rs_simulation_cfg
        from badminton_scene.scene_layout import REBOT_RS, SIMULATION

        # Override only the asset path for this invocation. No repository config is mutated.
        REBOT_RS["usd"] = str(robot_usd)
        dt = float(SIMULATION["physics_dt"])
        sim = sim_utils.SimulationContext(
            rebot_rs_simulation_cfg(dt=dt, device=device)
        )
        robot = spawn_rebot_rs_directly(translation=tuple(REBOT_RS["mount_world"]))

        stage = omni.usd.get_context().get_stage()
        arm_root = stage.GetPrimAtPath(REBOT_RS_PRIM_PATH)
        gripper_path = gripper_end_prim_path(REBOT_RS_PRIM_PATH)
        gripper = stage.GetPrimAtPath(gripper_path)
        if not arm_root.IsValid():
            raise RuntimeError("B601-RS root prim was not authored: " + REBOT_RS_PRIM_PATH)
        if not gripper.IsValid():
            raise RuntimeError("B601-RS gripper_end prim was not authored: " + gripper_path)

        visual_path = attach_racket_visual(
            stage,
            arm_prim_path=REBOT_RS_PRIM_PATH,
            mount_transform=mount,
            asset_path=str(racket_usd),
            include_collision=True,
            collision_proxy_path=str(collision_proxy),
        )
        mount_path = racket_mount_prim_path(REBOT_RS_PRIM_PATH)
        proxy_root = stage.GetPrimAtPath(mount_path + "/CollisionProxy")
        colliders = [
            prim for prim in Usd.PrimRange(proxy_root)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        capsule_count = sum(1 for prim in colliders if prim.IsA(UsdGeom.Capsule))
        box_count = sum(1 for prim in colliders if prim.IsA(UsdGeom.Cube))

        sim.reset()
        q = torch.tensor(
            [[REBOT_RS["rest_joint_pos"].get(name, 0.0) for name in robot.joint_names]],
            dtype=torch.float32,
            device=robot.device,
        )
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        for _ in range(args.steps):
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

        joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
        if not all(math.isfinite(float(v)) for v in joint_pos):
            raise RuntimeError("non-finite joint position after simulation steps")

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        gripper_world_xf = cache.GetLocalToWorldTransform(gripper)
        gripper_world = gripper_world_xf.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        visual_prim = stage.GetPrimAtPath(visual_path)
        mount_prim = stage.GetPrimAtPath(mount_path)
        visual_world_xf = cache.GetLocalToWorldTransform(visual_prim)
        mount_world_xf = cache.GetLocalToWorldTransform(mount_prim)
        visual_world = visual_world_xf.Transform(Gf.Vec3d(0.0, 0.0, 0.0))

        # A reference to /RacketVisual must preserve its authored root frame. Compare
        # origin and three local basis probes against RacketMount to detect an
        # unexpected reference transform or an accidental unit/scale change.
        probes = (
            Gf.Vec3d(0.0, 0.0, 0.0),
            Gf.Vec3d(0.1, 0.0, 0.0),
            Gf.Vec3d(0.0, 0.1, 0.0),
            Gf.Vec3d(0.0, 0.0, 0.1),
        )
        mount_visual_probe_error_m = max(
            math.dist(
                tuple(mount_world_xf.Transform(point)),
                tuple(visual_world_xf.Transform(point)),
            )
            for point in probes
        )

        link6_path = gripper_path.rsplit("/", 1)[0]
        link6_prim = stage.GetPrimAtPath(link6_path)
        collider_body_ancestors = {}
        for prim in colliders:
            ancestor = prim
            body_path = None
            while ancestor and ancestor.IsValid():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    body_path = str(ancestor.GetPath())
                    break
                ancestor = ancestor.GetParent()
            collider_body_ancestors[str(prim.GetPath())] = body_path
        unique_body_ancestors = sorted({value for value in collider_body_ancestors.values()})

        checks = {
            "arm_root_prim_valid": bool(arm_root.IsValid()),
            "gripper_end_prim_valid": bool(gripper.IsValid()),
            "link6_prim_valid": bool(link6_prim.IsValid()),
            "link6_has_rigid_body": bool(link6_prim.HasAPI(UsdPhysics.RigidBodyAPI)),
            "racket_visual_prim_valid": bool(visual_prim.IsValid()),
            "racket_mount_prim_valid": bool(mount_prim.IsValid()),
            "collision_proxy_prim_valid": bool(proxy_root.IsValid()),
            "collision_primitive_count": len(colliders),
            "expected_collision_primitive_count": int(proxy_doc["primitive_count"]),
            "capsule_count": capsule_count,
            "box_count": box_count,
            "collision_proxy_rigid_body_ancestor_paths": unique_body_ancestors,
            "all_collision_shapes_bound_to_link6": bool(
                link6_prim.IsValid()
                and link6_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                and unique_body_ancestors == [str(link6_prim.GetPath())]
            ),
            "visual_mount_alignment_max_probe_error_m": mount_visual_probe_error_m,
            "visual_aligned_with_mount": mount_visual_probe_error_m <= 1e-5,
            "joint_count": len(robot.joint_names),
            "joint_names": list(robot.joint_names),
            "joint_positions_rad_or_m": [float(v) for v in joint_pos],
            "simulation_steps_completed": args.steps,
            "gripper_end_world_position_m": [float(v) for v in gripper_world],
            "racket_visual_origin_world_position_m": [float(v) for v in visual_world],
            "mount_status_on_mount_prim": mount_prim.GetCustomDataByKey("badminton_mount_status"),
            "mount_status_on_visual_prim": visual_prim.GetCustomDataByKey("badminton_mount_status"),
            "racket_has_separate_rigid_body": bool(mount_prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        }
        report["checks"] = checks
        expected = int(proxy_doc["primitive_count"])
        if len(colliders) != expected:
            raise RuntimeError("collision proxy count mismatch: got %d, expected %d" % (len(colliders), expected))
        if capsule_count + box_count != expected:
            raise RuntimeError("unexpected collision primitive type distribution")
        if len(robot.joint_names) < 8:
            raise RuntimeError("expected the B601-RS 8-DOF articulation, got %d joints" % len(robot.joint_names))
        if not checks["link6_has_rigid_body"]:
            raise RuntimeError("link6 does not carry UsdPhysics.RigidBodyAPI; proxy inheritance is not verified")
        if not checks["all_collision_shapes_bound_to_link6"]:
            raise RuntimeError(
                "collision shapes do not all resolve to link6 as their nearest rigid-body ancestor: "
                + repr(unique_body_ancestors)
            )
        if not checks["visual_aligned_with_mount"]:
            raise RuntimeError(
                "referenced racket visual root does not match RacketMount transform; max probe error %.6g m"
                % mount_visual_probe_error_m
            )
        if checks["racket_has_separate_rigid_body"]:
            raise RuntimeError("racket mount unexpectedly has a separate RigidBodyAPI")
        if checks["mount_status_on_mount_prim"] != mount["status"]:
            raise RuntimeError("mount evidence label missing from RacketMount prim")
        if checks["mount_status_on_visual_prim"] != mount["status"]:
            raise RuntimeError("mount evidence label missing from RacketVisual prim")
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
        report["traceback"] = traceback.format_exc()
    finally:
        if app is not None:
            app.close()
        write_report_exclusive(output, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("WROTE", output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
