# -*- coding: utf-8 -*-
"""Fail-closed contract tests for the host-written P0 Isaac Lab JSON report."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
VERIFIER = ROOT / "tools/p0/verify_racket_smoke.py"


def _verifier():
    assert VERIFIER.is_file(), "missing required P0 authoritative JSON report verifier"
    spec = importlib.util.spec_from_file_location("verify_racket_smoke", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def good_report():
    return {
        "schema": "p0_b601_racket_scene_smoke/1",
        "status": "PASS",
        "inputs": {
            "mount_transform": {"status": "SIMULATION_ASSUMPTION"},
            "mount_transform_is_physical_calibration": False,
            "robot_usd_sha256": "a" * 64,
            "racket_usd_sha256": "b" * 64,
            "collision_proxy_sha256": "c" * 64,
            "steps_requested": 120,
        },
        "expected_collision_proxy_primitives": 43,
        "checks": {
            "arm_root_prim_valid": True,
            "gripper_end_prim_valid": True,
            "link6_prim_valid": True,
            "link6_has_rigid_body": True,
            "racket_visual_prim_valid": True,
            "racket_mount_prim_valid": True,
            "collision_proxy_prim_valid": True,
            "collision_primitive_count": 43,
            "expected_collision_primitive_count": 43,
            "capsule_count": 32,
            "box_count": 11,
            "collision_proxy_rigid_body_ancestor_paths": [
                "/World/envs/env_0/Robot/RebotB601RS/Geometry/base_link/link1/link2/link3/link4/link5/link6"
            ],
            "all_collision_shapes_bound_to_link6": True,
            "visual_mount_alignment_max_probe_error_m": 0.0,
            "visual_aligned_with_mount": True,
            "joint_count": 8,
            "joint_names": [
                "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                "gripper_joint1", "gripper_joint2"
            ],
            "joint_positions_rad_or_m": [0.0] * 8,
            "simulation_steps_completed": 120,
            "gripper_end_world_position_m": [0.0, 0.0, 0.3],
            "racket_visual_origin_world_position_m": [0.0, 0.0, 0.3],
            "mount_status_on_mount_prim": "SIMULATION_ASSUMPTION",
            "mount_status_on_visual_prim": "SIMULATION_ASSUMPTION",
            "racket_has_separate_rigid_body": False,
        },
    }


def run_verification(tmp_path, report=None):
    path = tmp_path / "p0.json"
    if report is not None:
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return _verifier().verify_report(path)


def test_complete_host_report_passes(tmp_path):
    verdict = run_verification(tmp_path, good_report())
    assert verdict["status"] == "PASS"
    assert len(verdict["report_sha256"]) == 64
    assert verdict["failures"] == []


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(status="FAIL"),
    lambda r: r["checks"].update(simulation_steps_completed=0),
    lambda r: r["checks"].update(collision_primitive_count=42),
    lambda r: r["checks"].update(capsule_count=31),
    lambda r: r["checks"].update(all_collision_shapes_bound_to_link6=False),
    lambda r: r["checks"].update(collision_proxy_rigid_body_ancestor_paths=["/World/Other"]),
    lambda r: r["checks"].update(visual_mount_alignment_max_probe_error_m=0.01),
    lambda r: r["checks"].update(visual_mount_alignment_max_probe_error_m=float("nan")),
    lambda r: r["checks"].update(visual_aligned_with_mount=False),
    lambda r: r["checks"].update(racket_has_separate_rigid_body=True),
    lambda r: r["inputs"]["mount_transform"].update(status="UNKNOWN"),
    lambda r: r["inputs"].update(mount_transform_is_physical_calibration=True),
    lambda r: r["checks"].pop("collision_primitive_count"),
    lambda r: r["checks"].update(mount_status_on_visual_prim=None),
    lambda r: r["checks"].update(joint_names=["joint1"]),
])
def test_invalid_host_report_never_passes(tmp_path, mutation):
    report = good_report()
    mutation(report)
    verdict = run_verification(tmp_path, report)
    assert verdict["status"] == "FAIL"
    assert verdict["failures"]


def test_missing_report_must_fail_even_if_process_returned_zero(tmp_path):
    verdict = run_verification(tmp_path, None)
    assert verdict["status"] == "FAIL"
    assert verdict["failures"]


def test_corrupt_report_must_fail(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text('{"status": "PASS"', encoding="utf-8")
    verdict = _verifier().verify_report(path)
    assert verdict["status"] == "FAIL"


def test_cli_fails_closed(tmp_path):
    verifier = _verifier()
    path = tmp_path / "missing.json"
    assert verifier.main(["--report", str(path)]) != 0
    good = tmp_path / "good.json"
    good.write_text(json.dumps(good_report()), encoding="utf-8")
    assert verifier.main(["--report", str(good)]) == 0
