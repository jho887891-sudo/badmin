# -*- coding: utf-8 -*-
"""robot_cfg.py regression tests that need no Isaac Sim.

The historical PiPER branch used to assign 'p' and then read '_p', so selecting
BADMINTON_ARM_MODEL=piper raised NameError at import time - and no test ever selected it.
These tests stub the isaaclab modules, import the real module from disk, and exercise both branches
at the configuration level (no simulation involved).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCENE_DIR = ROOT / "simulation" / "badminton_scene"
sys.path.insert(0, str(ROOT / "simulation"))


def _stub_isaaclab():
    """Minimal stand-ins for the three isaaclab symbols robot_cfg imports at module scope."""
    class _Cfg:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

        def replace(self, **kw):
            clone = type(self)(**{k: getattr(self, k) for k in self.__dict__})
            for k, v in kw.items():
                setattr(clone, k, v)
            return clone

    # ArticulationCfg.InitialStateCfg(...) is called directly, so the nested name must exist
    _Cfg.InitialStateCfg = _Cfg

    class _UsdFileCfg(_Cfg):
        def func(self, prim_path, cfg, **kwargs):
            self.spawned = (prim_path, kwargs)

    isaaclab = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    sim.UsdFileCfg = _UsdFileCfg
    sim.RigidBodyPropertiesCfg = _Cfg
    sim.ArticulationRootPropertiesCfg = _Cfg
    actuators = types.ModuleType("isaaclab.actuators")
    actuators.ImplicitActuatorCfg = _Cfg
    assets = types.ModuleType("isaaclab.assets")
    assets.ArticulationCfg = _Cfg
    assets.Articulation = _Cfg
    isaaclab.sim = sim
    isaaclab.actuators = actuators
    isaaclab.assets = assets
    sys.modules.update({"isaaclab": isaaclab, "isaaclab.sim": sim,
                        "isaaclab.actuators": actuators, "isaaclab.assets": assets})


@pytest.fixture(scope="module")
def robot_cfg_module():
    _stub_isaaclab()
    pkg = types.ModuleType("bs_scene")
    pkg.__path__ = [str(SCENE_DIR)]
    sys.modules["bs_scene"] = pkg
    spec = importlib.util.spec_from_file_location("bs_scene.robot_cfg", SCENE_DIR / "robot_cfg.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bs_scene.robot_cfg"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_piper_mount_pose_matches_the_scene_layout(robot_cfg_module):
    """Regression: the piper mount used to raise NameError ('p' assigned, '_p' read)."""
    from badminton_scene.scene_layout import PIPER

    pos, rot = robot_cfg_module.piper_mount_pose()
    assert pos == tuple(float(v) for v in PIPER["mount_world"])
    assert rot == (1.0, 0.0, 0.0, 0.0)


def test_piper_arm_cfg_applies_the_mount_without_raising(robot_cfg_module, tmp_path, monkeypatch):
    """Exercise the whole historical branch end-to-end at config level."""
    from badminton_scene.scene_layout import PIPER

    fake = tmp_path / "piper_cfg.py"
    fake.write_text(
        "class _State:\n"
        "    pos = None\n"
        "    rot = None\n"
        "    joint_pos = {}\n\n"
        "class _Spawn:\n"
        "    usd_path = None\n\n"
        "class _Cfg:\n"
        "    def __init__(self):\n"
        "        self.prim_path = None\n"
        "        self.init_state = _State()\n"
        "        self.spawn = _Spawn()\n\n"
        "    def replace(self, **kw):\n"
        "        new = _Cfg()\n"
        "        for k, v in kw.items():\n"
        "            setattr(new, k, v)\n"
        "        new.init_state = _State()\n"
        "        new.spawn = _Spawn()\n"
        "        return new\n\n"
        "PIPER_CFG = _Cfg()\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    cfg = robot_cfg_module._piper_articulation_cfg()
    assert cfg.init_state.pos == tuple(float(v) for v in PIPER["mount_world"])
    assert cfg.init_state.rot == (1.0, 0.0, 0.0, 0.0)
    assert cfg.init_state.joint_pos == dict(PIPER["rest_joint_pos"])
    assert cfg.spawn.usd_path == robot_cfg_module._PIPER_USD


def test_arm_model_selector_accepts_known_values(robot_cfg_module):
    assert robot_cfg_module.resolve_arm_model(None) == "rebot_b601_rs"
    assert robot_cfg_module.resolve_arm_model("PiPER") == "piper"
    assert robot_cfg_module.resolve_arm_model(" rebot_b601_rs ") == "rebot_b601_rs"


def test_arm_model_selector_rejects_unknown_values(robot_cfg_module):
    with pytest.raises(ValueError) as exc:
        robot_cfg_module.resolve_arm_model("b601_dm")
    assert "b601_dm" in str(exc.value)
    assert "rebot_b601_rs" in str(exc.value) and "piper" in str(exc.value)


def test_rebot_spawn_cfg_points_at_the_converted_asset(robot_cfg_module):
    from badminton_scene.scene_layout import REBOT_RS

    cfg = robot_cfg_module.rebot_rs_spawn_cfg()
    assert cfg.usd_path == str(REBOT_RS["usd"])
    assert cfg.articulation_props.fix_root_link is False


def test_default_module_state_is_the_formal_arm(robot_cfg_module):
    assert robot_cfg_module.ARM_MODEL == "rebot_b601_rs"
    assert robot_cfg_module.ROBOT_ARTICULATION_CFG.prim_path == "{ENV_REGEX_NS}/Robot/RebotB601RS"
