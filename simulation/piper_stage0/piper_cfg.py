# Shared Isaac Lab configuration for the AgileX PiPER 6-DOF arm (no gripper).
# Badminton project Stage 0. Mirror lives at scripts/piper_stage0/ (local SSOT)
# and projects/piper_stage0/ on the remote dev machine.
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

PIPER_STAGE0_DIR = Path(__file__).resolve().parents[1]
# Canonical USD produced by t01_convert_urdf.py from the AgileX URDF
# (assets/piper_isaac_sim/piper_description/urdf/piper_no_gripper_description.urdf)
PIPER_USD_PATH = str(PIPER_STAGE0_DIR / "assets" / "piper_no_gripper.usd")

# URDF zero config places joint2 (limit [0, 3.14]) and joint3 (limit [-2.967, 0])
# exactly on their limit boundaries. Stage 0 uses a mid-range rest pose to avoid
# limit-contact effects while validating stability and position control.
PIPER_REST_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 1.5,
    "joint3": -1.5,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
}

# Moderate PD gains for Stage 0 (light links, total arm mass ~4 kg). Runs at 240 Hz.
# NOTE: revisits when adding payload/racket or a chassis later.
PIPER_ARM_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["joint[1-6]"],
    stiffness=150.0,
    damping=15.0,
    effort_limit_sim=100.0,
)

# self-collisions disabled for clean Stage-0 control validation; enable later with
# real motion/collision studies.
PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=PIPER_REST_JOINT_POS),
    actuators={"piper_arm": PIPER_ARM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=1.0,
)