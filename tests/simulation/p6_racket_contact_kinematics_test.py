# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from rebot_b601_rs.kinematics.fk import Chain, DOF_ORDER  # noqa: E402
from rebot_b601_rs.kinematics.racket_contact import (  # noqa: E402
    CalibrationRequiredError,
    ContactKinematicsError,
    JointLimitError,
    RacketContactKinematics,
    TransformEvidence,
    racket_stringbed_center_proxy,
    validate_se3,
)

P6 = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
PROXY = P6 / "racket_collision_proxy.json"


class FakeChain:
    def __init__(self, pose, limits=None):
        self._pose = np.asarray(pose, dtype=float)
        self._limits = limits or {"joint1": True}

    def within_limits(self, q):
        return dict(self._limits)

    def pose(self, q, link="gripper_end"):
        assert link == "gripper_end"
        return self._pose.copy()


def evidence(name, translation, status="MEASURED", source="test fixture"):
    T = np.eye(4)
    T[:3, 3] = translation
    return TransformEvidence(name=name, matrix=T, status=status, source=source)


def unknown(name):
    return TransformEvidence(name=name, matrix=None, status="UNKNOWN", source="not calibrated")


def test_unknown_mount_is_rejected_instead_of_assuming_identity():
    model = RacketContactKinematics(
        FakeChain(np.eye(4)),
        gripper_end_from_racket_root=unknown("gripper_end_from_racket_root"),
        racket_root_from_contact=evidence("racket_root_from_contact", [0, 0, 0]),
    )
    with pytest.raises(CalibrationRequiredError, match="UNKNOWN"):
        model.compute(
            {"joint1": 0.0},
            world_from_arm_root=evidence("world_from_arm_root", [0, 0, 0]),
        )


def test_simulation_assumptions_need_explicit_opt_in():
    mount = evidence(
        "gripper_end_from_racket_root", [0.0, 0.0, 0.0],
        status="SIMULATION_ASSUMPTION", source="unit-test-only identity assumption",
    )
    model = RacketContactKinematics(
        FakeChain(np.eye(4)), gripper_end_from_racket_root=mount,
        racket_root_from_contact=evidence("racket_root_from_contact", [0, 0, 0]),
    )
    with pytest.raises(CalibrationRequiredError, match="simulation-only"):
        model.compute(
            {"joint1": 0.0},
            world_from_arm_root=evidence("world_from_arm_root", [0, 0, 0]),
        )


def test_transform_chain_composes_in_the_declared_frame_order():
    T_arm_gripper = np.eye(4)
    T_arm_gripper[:3, 3] = [1.0, 0.0, 0.0]
    model = RacketContactKinematics(
        FakeChain(T_arm_gripper),
        gripper_end_from_racket_root=evidence("gripper_end_from_racket_root", [0, 1, 0]),
        racket_root_from_contact=evidence("racket_root_from_contact", [0, 0, 2]),
    )
    result = model.compute(
        {"joint1": 0.0},
        world_from_arm_root=evidence("world_from_arm_root", [10, 0, 0]),
    )
    assert result.position_m == pytest.approx((11.0, 1.0, 2.0))
    assert result.status == "MEASURED"
    assert result.frame_name == "racket_contact_frame"


def test_proxy_contact_point_is_derived_from_the_committed_stringbed_bbox():
    doc = {
        "schema": "p6_racket_collision_proxy/1",
        "generated_by": "unit-test proxy",
        "root_frame": {"name": "racket_root", "mesh_local_to_world": "identity"},
        "source_meshes": [{
            "role": "string bed (visual only)",
            "bbox_min_m": [-0.001, -0.10073, 0.44391],
            "bbox_max_m": [0.001, 0.10073, 0.67526],
            "sha256": "test-sha",
        }],
    }
    proxy = racket_stringbed_center_proxy(doc)
    assert proxy.status == "PROXY_DERIVED"
    assert proxy.matrix[:3, 3] == pytest.approx([0.0, 0.0, 0.559585])
    assert "not a measured sweet spot" in proxy.source


def test_proxy_geometry_is_rejected_unless_explicitly_enabled():
    proxy = racket_stringbed_center_proxy({
        "schema": "p6_racket_collision_proxy/1",
        "generated_by": "unit-test proxy",
        "root_frame": {"name": "racket_root", "mesh_local_to_world": "identity"},
        "source_meshes": [{
            "role": "string bed (visual only)",
            "bbox_min_m": [-0.001, -0.1, 0.44],
            "bbox_max_m": [0.001, 0.1, 0.68],
            "sha256": "test-sha",
        }],
    })
    model = RacketContactKinematics(
        FakeChain(np.eye(4)),
        gripper_end_from_racket_root=evidence("mount", [0, 0, 0]),
        racket_root_from_contact=proxy,
    )
    with pytest.raises(CalibrationRequiredError, match="proxy-derived"):
        model.compute(
            {"joint1": 0.0},
            world_from_arm_root=evidence("world", [0, 0, 0]),
        )


def test_invalid_se3_is_rejected():
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ContactKinematicsError, match="orthonormal"):
        validate_se3(bad, "bad")


def test_joint_limit_violation_is_rejected_before_fk():
    model = RacketContactKinematics(
        FakeChain(np.eye(4), {"joint1": False}),
        gripper_end_from_racket_root=evidence("mount", [0, 0, 0]),
        racket_root_from_contact=evidence("contact", [0, 0, 0]),
    )
    with pytest.raises(JointLimitError, match="joint1"):
        model.compute(
            {"joint1": 999.0},
            world_from_arm_root=evidence("world", [0, 0, 0]),
        )


def test_real_p6_chain_and_committed_proxy_compose_with_explicit_simulation_labels():
    chain = Chain()
    q = {name: 0.0 for name in DOF_ORDER}
    proxy = racket_stringbed_center_proxy(PROXY)
    model = RacketContactKinematics(
        chain,
        gripper_end_from_racket_root=evidence(
            "gripper_end_from_racket_root", [0.0, 0.0, 0.0],
            status="SIMULATION_ASSUMPTION", source="test-only identity mount",
        ),
        racket_root_from_contact=proxy,
    )
    world = evidence(
        "world_from_arm_root", [0.0, 0.0, 0.0],
        status="SIMULATION_STATE", source="test-only identity world pose",
    )
    result = model.compute(
        q, world_from_arm_root=world,
        allow_simulation_assumptions=True,
        allow_proxy_geometry=True,
    )
    expected = chain.pose(q, link="gripper_end") @ proxy.matrix
    assert result.matrix_world_contact == pytest.approx(expected, abs=1e-9)
    assert result.status == "PROXY_DERIVED"
    assert result.input_statuses == ("SIMULATION_STATE", "SIMULATION_ASSUMPTION", "PROXY_DERIVED")
    assert result.frame_name == "racket_stringbed_center_proxy"
