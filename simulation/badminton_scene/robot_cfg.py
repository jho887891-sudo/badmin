# -*- coding: utf-8 -*-
"""Robot assembly config - PiPER (Stage0-verified) mounted on Morph placeholder.

ENGINEERING_V0_1:
  robot_base world = (-1.60, 0, 0); heading +X (yaw 0)
  Morph placeholder static visual; PiPER keeps Stage-0 fixed-base articulation.
"""
import sys
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from .scene_layout import PIPER, MORPH_PLACEHOLDER, ROBOT

_PIPER_SCRIPTS = "/home/T7/ojh/robot_sim/simulation/piper_stage0"
if _PIPER_SCRIPTS not in sys.path:
    sys.path.insert(0, _PIPER_SCRIPTS)
from piper_cfg import PIPER_CFG  # noqa: E402  (Stage-0 verified articulation cfg)

ROBOT_ARTICULATION_CFG: ArticulationCfg = PIPER_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot/PiPER",
)
# base pose in env-local (== world, origin at net centre)
_p = PIPER["mount_world"]
ROBOT_ARTICULATION_CFG.init_state.pos = (_p[0], _p[1], _p[2])
ROBOT_ARTICULATION_CFG.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # yaw 0 facing +X
ROBOT_ARTICULATION_CFG.init_state.joint_pos = dict(PIPER["rest_joint_pos"])
ROBOT_ARTICULATION_CFG.spawn.usd_path = "/home/T7/ojh/robot_sim/assets/piper_stage0/assets/piper_with_racket.usd"

MORPH = MORPH_PLACEHOLDER
