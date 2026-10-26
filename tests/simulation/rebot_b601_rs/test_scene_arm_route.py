# -*- coding: utf-8 -*-
"""The formal scene's arm route must be explicit - and must not claim multi-env support it lacks.

MultiEnvArm supports 128 environments, but that is a separate research assembly. create_scene is the
formal robot scene and still assembles the B601-RS with num_envs == 1 only; asking for more must raise
instead of silently degrading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene.assembly_route import resolve_scene_arm_route  # noqa: E402


def test_single_env_b601_rs_uses_the_direct_assembly():
    assert resolve_scene_arm_route(include_robot=True, arm_model="rebot_b601_rs", num_envs=1) == "direct"


@pytest.mark.parametrize("num_envs", [2, 4, 16, 64, 128])
def test_multi_env_b601_rs_through_create_scene_raises(num_envs):
    with pytest.raises(NotImplementedError) as exc:
        resolve_scene_arm_route(include_robot=True, arm_model="rebot_b601_rs", num_envs=num_envs)
    msg = str(exc.value)
    assert "num_envs" in msg and str(num_envs) in msg
    assert "MultiEnvArm" in msg, "the error must point at the supported alternative"


def test_no_arm_scene_ignores_num_envs():
    for n in (1, 2, 128):
        assert resolve_scene_arm_route(include_robot=False, arm_model="rebot_b601_rs",
                                       num_envs=n) == "scene_registered"


def test_legacy_piper_path_is_unchanged():
    assert resolve_scene_arm_route(include_robot=True, arm_model="piper", num_envs=1) == "scene_registered"
    assert resolve_scene_arm_route(include_robot=True, arm_model="piper", num_envs=8) == "scene_registered"


def test_create_scene_actually_uses_the_route_helper():
    """Anti-drift: the guard is worthless if create_scene keeps its own private copy of the rule."""
    src = (ROOT / "simulation" / "badminton_scene" / "scene_cfg.py").read_text(encoding="utf-8")
    assert "resolve_scene_arm_route(" in src
    assert "include_robot and ARM_MODEL ==" not in src, "an inline duplicate of the route rule came back"
