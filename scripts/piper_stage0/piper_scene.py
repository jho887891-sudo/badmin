# Shared single-env scene builder for PiPER Stage-0 tests.
# Import ONLY after SimulationContext exists / AppLauncher started (isaaclab modules).
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from piper_cfg import PIPER_CFG

SIM_DT = 1.0 / 240.0


@configclass
class PiperSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/PiPER")


def create_piper_scene(num_envs: int = 1):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0"))
    scene = InteractiveScene(PiperSceneCfg(num_envs=num_envs, env_spacing=8.0))
    robot = scene.articulations["robot"]
    sim.reset()
    return sim, scene, robot
