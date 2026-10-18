# -*- coding: utf-8 -*-
"""Badminton Robot Scene v0.1 - SINGLE SOURCE OF TRUTH for geometry.

Every coordinate/size/physical definition used by the scene, tests and
evidence scripts MUST be read from this module. ENGINEERING_V0_1 values are
frozen by spec; assets marked TEMP_PLACEHOLDER must never be presented as
real hardware measurements.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any

Vec3 = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# World frame (spec section 2) - right-handed, origin = net-centre under court
# +X robot side -> opponent side, +Y = left when facing +X, +Z up.
# Robot side: X<0 ; opponent side: X>0 ; net: X=0
# ---------------------------------------------------------------------------
WORLD_ORIGIN: Vec3 = (0.0, 0.0, 0.0)

COURT = dict(
    length=13.40,          # full court X extent total
    half_length=6.70,      # X = +-6.70 back boundaries
    doubles_width=6.10,    # Y = +-3.05 outer doubles lines
    singles_width=5.18,    # Y = +-2.59 singles lines
    half_doubles=3.05,
    half_singles=2.59,
    line_width=0.040,
    line_z=0.001,          # offset above ground to avoid z-fighting
    back_x=6.70,           # 两端后场边界 X=+-6.70
    short_service_x=1.98,  # 短发球线 X=+-1.98
    long_service_x=5.94,   # 双打长发球线 X=+-5.94 (6.70-0.76)
    center_y=0.0,
)

NET = dict(
    x=0.0,
    width=6.10,            # spans Y [-3.05, +3.05]
    half_width=3.05,
    depth=0.76,            # net depth (real net structural depth, spec value)
    center_top_z=1.524,
    side_top_z=1.55,
    post_height=1.55,
    collision_thickness_x=0.02,
    collision_top_z=1.55,
    collision_bottom_z=0.764,
)

ROBOT = dict(
    # ENGINEERING_V0_1 fixed robot pose
    world_pos=(-1.60, 0.00, 0.00),
    yaw=0.0,               # facing +X
    front_range=(-0.60, 1.40),   # world X approx covered in front 1-3 m
)

MORPH_PLACEHOLDER = dict(
    # ENGINEERING_V0_1 TEMP_PLACEHOLDER (spec 7) - NOT official Morph One data
    source="TEMP_PLACEHOLDER",
    length_x=0.70,
    width_y=0.55,
    height_z=0.25,
    center_world=(-1.60, 0.0, 0.125),   # base centre z = h/2
    top_z=0.25,
)

REBOT_RS = dict(
    # Formal arm since the 2026-10 migration. Mixing these numbers with the PiPER block below is a
    # defect: the two arms have different joint ranges, link lengths and masses.
    source="TEMP_PARAMETERIZED_PROXY",
    usd="/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs"
        "/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda",
    # mount relative to robot_base origin (robot_base at world (-1.6,0,0))
    mount_rel=(0.0, 0.0, 0.30),
    mount_world=(-1.60, 0.0, 0.30),
    # Engineering baseline rest pose INSIDE the official URDF ranges. Note the PiPER rest pose
    # (joint2=1.5, joint3=-1.5) is NOT reachable on this arm: joint2 and joint3 are [0, 3.14] here.
    # Chosen from geometry (outputs/simulation/rebot_b601_rs/rest_pose_probe.json): no link below
    # the base plane, link6 sits 0.497 m above the mount.
    rest_joint_pos={"joint1": 0.0, "joint2": 0.3, "joint3": 1.0,
                    "joint4": 0.0, "joint5": 0.0, "joint6": 0.0},
    # The URDF converter's own drive values, kept because they are reproducible. KNOWN DEFECT: at
    # these gains the rest pose does NOT hold - within 60 steps joint3 drifts to -0.067 rad, i.e.
    # below its own lower limit of 0 (outputs/simulation/rebot_b601_rs/VERIFY_scene_load.json).
    # Raising stiffness to 5000 changed the result by less than 3 mrad, so the drives themselves,
    # not the gain choice, are the suspect. Tracked as open item F1 in
    # management/rebot_b601_rs/ARM_MIGRATION_RECORD.md.
    stiffness=100.0,
    damping=1.0,
)

PIPER = dict(
    # mount relative to robot_base origin (robot_base at world (-1.6,0,0))
    mount_rel=(0.0, 0.0, 0.30),
    mount_world=(-1.60, 0.0, 0.30),
    # Stage-0 verified safe/rest pose (reused, do not re-guess)
    rest_joint_pos={"joint1": 0.0, "joint2": 1.5, "joint3": -1.5,
                    "joint4": 0.0, "joint5": 0.0, "joint6": 0.0},
)

RACKET = dict(
    # ENGINEERING_V0_1 TEMP_PLACEHOLDER mass/geometry; BWF legal limits
    source="TEMP_PLACEHOLDER",
    total_length=0.675,        # <= 0.680
    head_width_max=0.225,      # <= 0.230
    head_height_approx=0.290,
    mass=0.10,                 # TEMP placeholder
    tcp_to_contact_approx=0.50,  # spec approx; recomputed from geometry
    # racket local frame: +X_racket face normal -> opponent; +Z_racket handle->head
)

CAMERA = dict(
    # ENGINEERING_V0_1 camera rig placeholder (no sensor/render)
    rig_height=1.20,
    rig_rel_x=0.20,
    rig_world=(-1.40, 0.0, 1.20),
    pitch_deg=-4.0,
    baseline=0.29,
    left_y=0.145,
    right_y=-0.145,
)

PERCEPTION_ROI = dict(         # spec 15 (robot-relative X_rel [1,3] etc.)
    rel_x=(1.0, 3.0),
    rel_y=(-1.0, 1.0),
    world_x=(-0.60, 1.40),
    world_y=(-1.0, 1.0),
    z=(0.3, 2.8),
    color=(0.2, 0.5, 1.0, 0.25),
)

CANDIDATE_STRIKE_ROI = dict(   # spec 16 robot side only
    world_x=(-0.75, -0.10),
    world_y=(-0.90, 0.90),
    z=(0.70, 1.80),
    color=(1.0, 0.5, 0.1, 0.3),
)

SPAWN_REGION = dict(           # spec 21 opponent forecourt
    world_x=(0.80, 1.40),
    world_y=(-0.80, 0.80),
    z=(1.60, 2.40),
    color=(1.0, 0.25, 0.2, 0.3),
)

TARGET_ZONE = dict(            # spec 22 opponent court landing zone
    world_x=(2.00, 4.50),
    world_y=(-2.00, 2.00),
    z=(0.0, 0.10),
    z_ground=True,
    color=(0.1, 0.9, 0.4, 0.25),
)

CANONICAL_INCOMING = dict(     # spec 20 CANONICAL_INCOMING_001
    name="CANONICAL_INCOMING_001",
    pos=(1.20, 0.00, 1.80),
    lin_vel=(-3.00, 0.00, 1.50),
    ang_vel=(0.0, 0.0, 0.0),
    gravity_on=True,
    drag_off=True,
)

SHUTTLE = dict(
    mass=0.0051,
    base_diameter_m=0.028,     # spec ~25-28 mm base
    physics_collision_radius=0.014,   # cork/base proxy radius
    visual_model="simple",     # SimpleShuttle v0.1
)

SIMULATION = dict(
    physics_dt=1.0 / 240.0,
    render_dt=1.0 / 60.0,
    env_spacing=15.0,          # ENGINEERING_V0_1 (>= full court length)
    gravity=9.81,
)

# per-env world placement offset used by vectorised clones (global x tiling)
def env_origin(i: int) -> Vec3:
    return (float(i) * SIMULATION["env_spacing"], 0.0, 0.0)

@dataclass
class SceneLayout:
    """Snapshot-able layout container (scene_layout.json writer uses asdict)."""
    world_origin: Vec3 = WORLD_ORIGIN
    court: Dict[str, Any] = field(default_factory=lambda: COURT.copy())
    net: Dict[str, Any] = field(default_factory=lambda: NET.copy())
    robot: Dict[str, Any] = field(default_factory=lambda: ROBOT.copy())
    morph_placeholder: Dict[str, Any] = field(default_factory=lambda: MORPH_PLACEHOLDER.copy())
    piper: Dict[str, Any] = field(default_factory=lambda: PIPER.copy())
    racket: Dict[str, Any] = field(default_factory=lambda: RACKET.copy())
    camera: Dict[str, Any] = field(default_factory=lambda: CAMERA.copy())
    perception_roi: Dict[str, Any] = field(default_factory=lambda: PERCEPTION_ROI.copy())
    candidate_strike_roi: Dict[str, Any] = field(default_factory=lambda: CANDIDATE_STRIKE_ROI.copy())
    spawn_region: Dict[str, Any] = field(default_factory=lambda: SPAWN_REGION.copy())
    target_zone: Dict[str, Any] = field(default_factory=lambda: TARGET_ZONE.copy())
    canonical_incoming: Dict[str, Any] = field(default_factory=lambda: CANONICAL_INCOMING.copy())
    shuttle: Dict[str, Any] = field(default_factory=lambda: SHUTTLE.copy())
    simulation: Dict[str, Any] = field(default_factory=lambda: SIMULATION.copy())

def layout_to_dict() -> Dict[str, Any]:
    return asdict(SceneLayout())
