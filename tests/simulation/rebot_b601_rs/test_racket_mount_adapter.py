# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene.racket_mount_adapter import (  # noqa: E402
    RacketMountError,
    gripper_end_prim_path,
    load_racket_collision_proxy,
    racket_mount_prim_path,
    validate_mount_transform,
)


def test_unknown_mount_is_rejected_even_when_no_values_are_provided():
    with pytest.raises(RacketMountError, match="status must be"):
        validate_mount_transform({
            "status": "UNKNOWN",
            "translation_m": None,
            "quaternion_wxyz": None,
            "source": "",
        })


def test_unknown_mount_cannot_smuggle_in_an_identity_matrix():
    with pytest.raises(RacketMountError, match="status must be"):
        validate_mount_transform({
            "status": "UNKNOWN",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "source": "unverified identity",
        })


def test_explicit_simulation_assumption_is_accepted_and_labelled():
    value = validate_mount_transform({
        "status": "SIMULATION_ASSUMPTION",
        "translation_m": [0.01, -0.02, 0.03],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "source": "unit-test-only assumed mount",
    })
    assert value.status == "SIMULATION_ASSUMPTION"
    assert value.translation_m == (0.01, -0.02, 0.03)


def test_non_unit_quaternion_is_rejected_not_silently_normalized():
    with pytest.raises(RacketMountError, match="unit length"):
        validate_mount_transform({
            "status": "MEASURED",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [2.0, 0.0, 0.0, 0.0],
            "source": "test fixture",
        })


def test_measured_mount_requires_source_provenance():
    with pytest.raises(RacketMountError, match="provenance"):
        validate_mount_transform({
            "status": "MEASURED",
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "source": "",
        })


def test_gripper_and_racket_paths_are_under_the_b601_rs_asset():
    arm = "/World/envs/env_0/Robot/RebotB601RS"
    gripper = gripper_end_prim_path(arm)
    assert gripper.endswith("/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end")
    assert racket_mount_prim_path(arm) == gripper + "/RacketMount"


def test_committed_collision_proxy_has_43_valid_engineering_primitives():
    doc = load_racket_collision_proxy()
    assert doc["schema"] == "p6_racket_collision_proxy/1"
    assert "NOT a manufacturer collision model" in doc["kind"]
    assert doc["primitive_count"] == 43
    assert len(doc["primitives"]) == 43
    assert {item["type"] for item in doc["primitives"]} == {"capsule", "box"}


def test_collision_proxy_loader_rejects_unknown_primitive_type(tmp_path):
    import json

    bad = {
        "schema": "p6_racket_collision_proxy/1",
        "kind": "ENGINEERING_PROXY — NOT a manufacturer collision model",
        "primitive_count": 1,
        "primitives": [{"id": "bad_shape", "type": "sphere", "geometry": {}}],
    }
    path = tmp_path / "bad_proxy.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RacketMountError, match="unsupported collision primitive"):
        load_racket_collision_proxy(str(path))
