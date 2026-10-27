#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-closed external acceptance gate for a P0 B601-RS racket smoke JSON.

Pure Python. Runs OUTSIDE Isaac Sim, so native Kit exit_group(0) cannot turn a
missing or failed report into a PASS. This gate validates what a JSON claims,
not the physical truth of the measurements: target-host provenance is separate.

Usage:
  python tools/p0/verify_racket_smoke.py --report /absolute/path/to/new.json

Exit 0 = all required smoke report fields pass; exit 2 = any missing/failed
field or unreadable report. It never edits the evidence file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA = "p0_b601_racket_scene_smoke/1"
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def _finite_number(value: Any) -> bool:
    return (type(value) in (int, float)
            and math.isfinite(value))


def _valid_vec3(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 3
            and all(_finite_number(x) for x in value))


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def verify_report(path: Path | str) -> dict[str, Any]:
    """Verify the formal 120-step scene-load test, never accepting partial JSON."""
    p = Path(path).expanduser().resolve()
    result: dict[str, Any] = {
        "schema": "p0_racket_smoke_verdict/1",
        "report_path": str(p),
        "report_sha256": None,
        "status": "FAIL",
        "failures": [],
        "verified_contract": (
            "fixed-base simulated B601-RS / racket proxy; "
            "not physical mount or shuttle-racket contact verification"
        ),
    }
    fail = result["failures"]
    try:
        raw = p.read_bytes()
        result["report_sha256"] = hashlib.sha256(raw).hexdigest()
        doc = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        fail.append(f"report unreadable or invalid: {type(exc).__name__}: {exc}")
        return result

    if not isinstance(doc, dict):
        fail.append("top-level report must be a JSON object")
        return result

    inputs = _dict(doc.get("inputs"))
    checks = _dict(doc.get("checks"))

    def require(name: str, passes: bool) -> None:
        if not passes:
            fail.append(name)

    require("schema mismatch", doc.get("schema") == SCHEMA)
    require("reported status is not PASS", doc.get("status") == "PASS")
    require("inputs object missing", isinstance(doc.get("inputs"), dict))
    require("checks object missing", isinstance(doc.get("checks"), dict))

    for name in ("robot_usd_sha256", "racket_usd_sha256", "collision_proxy_sha256"):
        digest = inputs.get(name)
        require("missing or invalid " + name,
                isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest) is not None)

    require("requested steps must be 120", _exact_int(inputs.get("steps_requested"), 120))
    require("completed steps must be 120", _exact_int(checks.get("simulation_steps_completed"), 120))
    require("expected collision proxy count must be 43",
            _exact_int(doc.get("expected_collision_proxy_primitives"), 43))
    require("collision primitive count must be 43",
            _exact_int(checks.get("collision_primitive_count"), 43))
    require("checked expected collision count must be 43",
            _exact_int(checks.get("expected_collision_primitive_count"), 43))
    require("capsule count must be 32",
            _exact_int(checks.get("capsule_count"), 32))
    require("box count must be 11",
            _exact_int(checks.get("box_count"), 11))

    for key in (
        "arm_root_prim_valid", "gripper_end_prim_valid",
        "link6_prim_valid", "link6_has_rigid_body",
        "racket_visual_prim_valid", "racket_mount_prim_valid",
        "collision_proxy_prim_valid", "all_collision_shapes_bound_to_link6",
        "visual_aligned_with_mount",
    ):
        require(f"{key} must be true", checks.get(key) is True)

    rigid_paths = checks.get("collision_proxy_rigid_body_ancestor_paths")
    require("all collider body ancestors must resolve to one link6",
            isinstance(rigid_paths, list)
            and len(rigid_paths) == 1
            and isinstance(rigid_paths[0], str)
            and rigid_paths[0].endswith("/link6"))

    err = checks.get("visual_mount_alignment_max_probe_error_m")
    require("visual mount alignment error must be finite <= 1e-5 m",
            _finite_number(err) and 0.0 <= err <= 1e-5)

    require("racket must not have separate rigid body",
            checks.get("racket_has_separate_rigid_body") is False)

    joint_names = checks.get("joint_names")
    joint_values = checks.get("joint_positions_rad_or_m")
    require("joint count must be 8", _exact_int(checks.get("joint_count"), 8))
    require("joint names must have 8 unique elements",
            isinstance(joint_names, list) and len(joint_names) == 8
            and all(isinstance(name, str) and bool(name) for name in joint_names)
            and len(set(joint_names)) == 8)
    require("joint positions must be 8 finite numbers",
            isinstance(joint_values, list) and len(joint_values) == 8
            and all(_finite_number(v) for v in joint_values))

    for key in ("gripper_end_world_position_m", "racket_visual_origin_world_position_m"):
        require(f"{key} must be a finite 3D position", _valid_vec3(checks.get(key)))

    mount = _dict(inputs.get("mount_transform"))
    require("mount is not explicitly SIMULATION_ASSUMPTION",
            mount.get("status") == "SIMULATION_ASSUMPTION")
    require("mount must not be classified as physical calibration",
            inputs.get("mount_transform_is_physical_calibration") is False)
    for key in ("mount_status_on_mount_prim", "mount_status_on_visual_prim"):
        require(f"{key} is not SIMULATION_ASSUMPTION",
                checks.get(key) == "SIMULATION_ASSUMPTION")

    result["status"] = "PASS" if not fail else "FAIL"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="JSON emitted by p0_b601_racket_scene_smoke.py")
    args = parser.parse_args(argv)
    verdict = verify_report(args.report)
    print(json.dumps(verdict, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if verdict["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
