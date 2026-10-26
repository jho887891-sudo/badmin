# -*- coding: utf-8 -*-
"""USD-core stage test for racket visual and collision-proxy authoring.

This exercises OpenUSD authoring APIs, not Isaac Sim/PhysX runtime contact dynamics.
The mount values below are explicitly test-only assumptions.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

pxr = pytest.importorskip("pxr", reason="requires OpenUSD Python bindings (pxr)")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene.racket_mount_adapter import (  # noqa: E402
    attach_racket_visual,
    gripper_end_prim_path,
    racket_mount_prim_path,
)


def test_racket_visual_and_43_collision_shapes_are_authored_with_valid_usd_types():
    stage = Usd.Stage.CreateInMemory()
    arm_path = "/World/envs/env_0/Robot/RebotB601RS"
    gripper_path = gripper_end_prim_path(arm_path)
    UsdGeom.Xform.Define(stage, gripper_path)

    # Deliberately an explicit test assumption, not a physical calibration.
    mount = {
        "status": "SIMULATION_ASSUMPTION",
        "translation_m": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "source": "unit test only; not a measured B601-RS racket mount",
    }
    visual_path = attach_racket_visual(
        stage,
        arm_prim_path=arm_path,
        mount_transform=mount,
        include_collision=True,
    )

    mount_path = racket_mount_prim_path(arm_path)
    assert visual_path == mount_path + "/RacketVisual"
    assert stage.GetPrimAtPath(visual_path).IsValid()
    assert stage.GetPrimAtPath(mount_path).GetCustomDataByKey("badminton_mount_status") == "SIMULATION_ASSUMPTION"
    assert stage.GetPrimAtPath(visual_path).GetCustomDataByKey("badminton_mount_status") == "SIMULATION_ASSUMPTION"

    colliders = [
        prim for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    assert len(colliders) == 43
    assert all(not prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in colliders)

    # The JSON endpoints describe the complete outer extent. USD capsule.height
    # excludes the hemispherical caps, so the authored cylinder spine is shorter.
    handle = UsdGeom.Capsule(stage.GetPrimAtPath(
        mount_path + "/CollisionProxy/racket_handle"
    ))
    radius = float(handle.GetRadiusAttr().Get())
    height = float(handle.GetHeightAttr().Get())
    assert radius == pytest.approx(0.015845)
    assert height == pytest.approx(0.15669 - 2.0 * radius)
    extent = handle.GetExtentAttr().Get()
    assert len(extent) == 2
    assert float(extent[0][2]) == pytest.approx(-(height / 2.0 + radius))
    assert float(extent[1][2]) == pytest.approx(height / 2.0 + radius)

    # Explicitly authored double precision avoids a Vec3f/Vec3d mismatch at runtime.
    bed = stage.GetPrimAtPath(
        mount_path + "/CollisionProxy/racket_string_bed_slab_07"
    )
    scale = bed.GetAttribute("xformOp:scale")
    assert scale.IsValid()
    assert str(scale.GetTypeName()) == "double3"
    assert scale.Get() is not None
