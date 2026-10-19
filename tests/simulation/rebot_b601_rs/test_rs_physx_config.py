# -*- coding: utf-8 -*-
"""Pure-CPU regression for the TGS external-forces-per-iteration configuration.

This checks config construction and all three B601-RS simulation call sites.
The actual joint velocity improvement must additionally be demonstrated in GPU runs;
the assertions here cannot replace numerical PhysX evidence.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene.rs_physics import rebot_rs_simulation_cfg


def test_b601_rs_physx_config_selects_verified_tgs_settings(monkeypatch):
    class FakeCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    sim = types.ModuleType("isaaclab.sim")
    sim.SimulationCfg = FakeCfg
    isaaclab.sim = sim
    physx_package = types.ModuleType("isaaclab_physx")
    physx_package.__path__ = []
    physx = types.ModuleType("isaaclab_physx.physics")
    physx.PhysxCfg = FakeCfg
    physx_package.physics = physx
    for name, obj in (("isaaclab", isaaclab),
                      ("isaaclab.sim", sim),
                      ("isaaclab_physx", physx_package),
                      ("isaaclab_physx.physics", physx)):
        monkeypatch.setitem(sys.modules, name, obj)

    cfg = rebot_rs_simulation_cfg(dt=1.0 / 240.0, device="cuda:0")
    assert cfg.dt == pytest.approx(1.0 / 240.0)
    assert cfg.device == "cuda:0"
    assert cfg.physics.solver_type == 1
    assert cfg.physics.enable_external_forces_every_iteration is True
    # Do not introduce unrelated gains, solver iterations or other hard-coded assumptions.
    assert set(cfg.physics.__dict__) == {
        "solver_type", "enable_external_forces_every_iteration"}


@pytest.mark.parametrize("relative_path,function_name", [
    ("simulation/badminton_scene/scene_cfg.py", "create_scene"),
    ("simulation/badminton_scene/multienv_arm.py", "spawn"),
    ("scripts/simulation/check_arm_rest_pose_hold.py", None),
])
def test_every_formal_arm_entrypoint_uses_the_same_physx_config(relative_path, function_name):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    if function_name is None:
        nodes = [tree]
    else:
        nodes = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == function_name]
        assert len(nodes) == 1, (relative_path, function_name)
    calls = [n for root in nodes for n in ast.walk(root)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "rebot_rs_simulation_cfg"]
    assert len(calls) == 1, relative_path + " did not select the verified TGS configuration"
