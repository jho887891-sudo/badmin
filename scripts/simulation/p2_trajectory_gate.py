"""Offline P2 trajectory verifier; never substitutes for Isaac/PhysX execution.

Reads a single run's manifest.json and full-step raw_poses.jsonl, and produces
result.json with explicit boolean gates. Requires PR #15 p2_pose_metrics.py.
Designed to fail closed if telemetry, contact evidence or a step is missing.

  python scripts/simulation/p2_trajectory_gate.py \
    --manifest /path/manifest.json \
    --trajectory /path/raw_poses.jsonl \
    --result /path/result.json
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys

from p2_pose_metrics import Pose, length, mount_pose_error, world_pose_error


class InvalidEvidence(ValueError):
    pass


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          InvalidEvidence(f"Non-finite JSON constant: {value}")))


def finite_tree(value, path="root"):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not isfinite(value):
            raise InvalidEvidence(f"Non-finite number at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")


def body_pose(data):
    layout = data["quat_layout"]
    position = data["position_m"]
    raw = data["raw_quaternion"]
    if not data.get("source_api"):
        raise InvalidEvidence("body pose missing source_api")
    if layout == "xyzw":
        return Pose.from_isaac_body_xyzw(position, raw)
    if layout == "wxyz":
        return Pose.from_usd_wxyz(position, raw)
    raise InvalidEvidence(f"Unknown quaternion layout: {layout}")


def audit(manifest: dict, trajectory: list[dict]) -> dict:
    if not isinstance(manifest, dict) or not trajectory:
        raise InvalidEvidence("Empty or non-object manifest / trajectory")
    finite_tree(manifest, "manifest")
    finite_tree(trajectory, "trajectory")
    try:
        dt = float(manifest["physics_dt_s"])
        count = int(manifest.get("settle_steps", 60))
        authored = manifest["authored_world"]
        mount = manifest["expected_mount"]
        expected_bodies = tuple(authored.keys())
        expected_joints = tuple(manifest["joint_names"])
        if not (0 < dt <= 1 and 1 <= count <= 100000):
            raise InvalidEvidence("Invalid dt or number of steps")
        if "Chassis" not in expected_bodies or "Arm/base_link" not in expected_bodies:
            raise InvalidEvidence("Required Chassis and Arm/base_link missing")
        if count != manifest.get("settle_steps", 60):
            raise InvalidEvidence("settle_steps must be an integer")
        # Do not silently accept anonymous evidence without a reproducible source.
        for field in ("run_id", "commit", "scene_variant"):
            if not isinstance(manifest[field], str) or not manifest[field].strip():
                raise InvalidEvidence(f"Missing manifest.{field}")
        if len(expected_bodies) != len(set(expected_bodies)):
            raise InvalidEvidence("Duplicate authored bodies")
    except (KeyError, ValueError, TypeError) as exc:
        raise InvalidEvidence(f"Invalid manifest: {exc}") from exc

    frames = {}
    observed_contacts = True
    no_unauthorized = True
    velocities_complete = True
    try:
        for record in trajectory:
            event = record.get("event")
            if event not in ("reset0", "post_step"):
                continue   # write-before/after entries are retained but not step snapshots
            step = record["step"]
            if type(step) is not int or step < 0:
                raise InvalidEvidence("Step must be a nonnegative integer")
            if step in frames:
                raise InvalidEvidence(f"Duplicate primary step {step}")
            if step == 0 and event != "reset0":
                raise InvalidEvidence("Step0 must be reset0")
            if step > 0 and event != "post_step":
                raise InvalidEvidence("Steps>=1 must be post_step")
            bodies = record["bodies"]
            joints = record["joints"]
            for name in expected_bodies:
                data = bodies[name]
                body_pose(data)
                linear = data.get("linear_velocity_m_s")
                angular = data.get("angular_velocity_rad_s")
                if linear is None or angular is None or len(linear) != 3 or len(angular) != 3:
                    velocities_complete = False
            for name in expected_joints:
                joint = joints[name]
                if not isinstance(joint.get("position_rad"), (int, float)) or not isinstance(
                        joint.get("velocity_rad_s"), (int, float)):
                    velocities_complete = False
            status = record.get("contact_status", "unavailable")
            if status != "measured":
                observed_contacts = False
            else:
                count_unsafe = record.get("unauthorized_contacts")
                if type(count_unsafe) is not int or count_unsafe < 0:
                    observed_contacts = False
                elif count_unsafe > 0:
                    no_unauthorized = False
            frames[step] = record
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise InvalidEvidence(f"Malformed primary frame: {exc}") from exc

    full_sequence = set(frames) == set(range(count + 1))
    if not full_sequence:
        missing = sorted(set(range(count + 1)) - set(frames))
        extras = sorted(set(frames) - set(range(count + 1)))
        raise InvalidEvidence(f"Missing/extra step snapshots: missing={missing[:15]}, extra={extras[:15]}")

    world_p = []
    world_r = []
    for name in expected_bodies:
        spec = authored[name]
        reference = Pose.from_usd_wxyz(spec["position_m"], spec["quat_wxyz"])
        observed = body_pose(frames[0]["bodies"][name])
        errors = world_pose_error(reference, observed)
        world_p.append(errors["world_position_error_m"])
        world_r.append(errors["world_rotation_error_rad"])
    t0_chassis = body_pose(frames[0]["bodies"]["Chassis"])
    t0_base = body_pose(frames[0]["bodies"]["Arm/base_link"])
    local = mount_pose_error(t0_chassis, t0_base,
                             mount["position_m"], mount["quat_wxyz"])

    max_drift = max(length(tuple(
        a - b for a, b in zip(body_pose(frames[index]["bodies"]["Chassis"]).position_m,
                             t0_chassis.position_m)))
                    for index in range(count + 1))
    pos_tol = float(manifest.get("position_tolerance_m", 0.001))
    rot_tol = float(manifest.get("rotation_tolerance_rad", 0.001))
    if pos_tol <= 0 or rot_tol <= 0:
        raise InvalidEvidence("Invalid tolerance")
    metrics = {
        "max_t0_world_position_error_m": max(world_p),
        "max_t0_world_rotation_error_rad": max(world_r),
        "t0_mount_position_error_m": local["mount_position_error_m"],
        "t0_mount_rotation_error_rad": local["mount_rotation_error_rad"],
        "max_chassis_settle_drift_m": max_drift,
        "last_chassis_settle_drift_m": length(tuple(
            a - b for a, b in zip(body_pose(frames[count]["bodies"]["Chassis"]).position_m,
                                 t0_chassis.position_m))),
        "step_count": count,
    }
    gates = {
        "t0_world_position_within_tol": metrics["max_t0_world_position_error_m"] <= pos_tol,
        "t0_world_orientation_within_tol": metrics["max_t0_world_rotation_error_rad"] <= rot_tol,
        "t0_mount_position_within_tol": metrics["t0_mount_position_error_m"] <= pos_tol,
        "t0_mount_orientation_within_tol": metrics["t0_mount_rotation_error_rad"] <= rot_tol,
        "settle_drift_within_tol": metrics["max_chassis_settle_drift_m"] <= pos_tol,
        "velocity_telemetry_complete": velocities_complete,
        "contact_evidence_complete": observed_contacts,
        "no_unauthorized_contacts": no_unauthorized,
        "no_non_finite": True,
        "step_sequence_complete": full_sequence,
    }
    return {"gate_result": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates, "metrics": metrics, "source_run_id": manifest["run_id"],
            "position_tolerance_m": pos_tol,
            "rotation_tolerance_rad": rot_tol,
            "evidence_type": "offline_telemetry_audit_not_physx_reexecution"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.result.exists():
        parser.error("Refuse to overwrite existing result JSON")
    try:
        manifest = load_json(args.manifest)
        trajectory = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
            InvalidEvidence(f"Non-finite JSON constant {x}")))
                      for line in args.trajectory.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        verdict = audit(manifest, trajectory)
    except (InvalidEvidence, ValueError, KeyError, OSError, TypeError) as exc:
        verdict = {"gate_result": "FAIL", "gates": {"evidence_schema_valid": False},
                   "error": f"{type(exc).__name__}: {exc}",
                   "evidence_type": "offline_telemetry_audit_not_physx_reexecution"}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    with args.result.open("x", encoding="utf-8") as fp:
        json.dump(verdict, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.write("\n")
    print(json.dumps(verdict, ensure_ascii=False))
    return 0 if verdict["gate_result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
