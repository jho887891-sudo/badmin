# -*- coding: utf-8 -*-
"""Badminton Robot Scene v0.1 - scene assembly (InteractiveScene + raw USD statics).

Static geometry (court/net/ROI/placeholders) authored with raw pxr prims to
avoid spawner path pitfalls; robot/shuttle registered as InteractiveScene
physics assets for vectorized cloning. Physics 240 Hz (ENGINEERING_V0_1).
"""
from __future__ import annotations
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf

from .scene_layout import (COURT, NET, CAMERA, PIPER, MORPH_PLACEHOLDER,
                           PERCEPTION_ROI, CANDIDATE_STRIKE_ROI, SPAWN_REGION,
                           TARGET_ZONE, SIMULATION)
from .court_cfg import court_lines_geometry, GROUND_VISUAL, LINE_COLOR, LINE_HEIGHT_Z
from .net_cfg import (net_collision_box, net_visual_band_box, net_post_positions,
                      NET_VISUAL, POST_RADIUS)
from .robot_cfg import ROBOT_ARTICULATION_CFG, MORPH
from .shuttle_cfg import SHUTTLE_CFG

SIM_DT = SIMULATION["physics_dt"]
_MAT_COUNTER = [0]


def _stage():
    return omni.usd.get_context().get_stage()


def _material(rgb):
    st = _stage()
    _MAT_COUNTER[0] += 1
    name = "mat_" + str(_MAT_COUNTER[0])
    path = "/World/_materials/" + name
    mat = UsdShade.Material.Define(st, path)
    sh = UsdShade.Shader.Define(st, path + "/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants)


def _box(path, size, center, color, collision=False):
    st = _stage()
    xf = UsdGeom.Xform.Define(st, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube = UsdGeom.Cube.Define(st, path + "/Cube")
    cube.AddScaleOp().Set(Gf.Vec3f(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))
    UsdGeom.Gprim(cube).CreateDisplayColorAttr([Gf.Vec3f(*color[:3])])
    mat = _material(color[:3])
    _bind(cube.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return path


def _cyl(path, radius, height, center, color):
    st = _stage()
    xf = UsdGeom.Xform.Define(st, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    cyl = UsdGeom.Cylinder.Define(st, path + "/Cylinder")
    cyl.GetRadiusAttr().Set(float(radius))
    cyl.GetHeightAttr().Set(float(height))
    UsdGeom.Gprim(cyl).CreateDisplayColorAttr([Gf.Vec3f(*color[:3])])
    mat = _material(color[:3])
    _bind(cyl.GetPrim(), mat)
    return path


def _roi(path, d):
    x0, x1 = d["world_x"]; y0, y1 = d["world_y"]; z0, z1 = d["z"]
    _box(path, (x1 - x0, y1 - y0, z1 - z0),
         ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0), d["color"])


def add_court_statics(prefix):
    c = COURT
    _box(prefix + "/Court/CourtFloor", (c["length"], c["doubles_width"], 0.004),
         (0.0, 0.0, 0.002), GROUND_VISUAL["color"])
    for (x, y, la, wa) in court_lines_geometry():
        nm = "L_X%.3f_Y%.3f" % (x, y)
        nm = nm.replace("-", "m").replace(".", "p")
        _box(prefix + "/Court/OuterLines/" + nm, (la, wa, 0.004), (x, y, LINE_HEIGHT_Z), LINE_COLOR)
    nb = net_visual_band_box()
    _box(prefix + "/Court/Net/NetVisual", (nb["size_x"], nb["size_y"], nb["size_z"]),
         (0.0, nb["y_center"], nb["z_center"]), (0.95, 0.95, 0.95))
    ncb = net_collision_box()
    _box(prefix + "/Court/Net/NetCollision", (ncb["size_x"], ncb["size_y"], ncb["size_z"]),
         (0.0, ncb["y_center"], ncb["z_center"]), (1.0, 0.9, 0.9), collision=True)
    posts = net_post_positions()
    _cyl(prefix + "/Court/Net/PostL", POST_RADIUS, NET["post_height"], posts[0], NET_VISUAL["post_color"])
    _cyl(prefix + "/Court/Net/PostR", POST_RADIUS, NET["post_height"], posts[1], NET_VISUAL["post_color"])
    _roi(prefix + "/Court/PerceptionROI", PERCEPTION_ROI)
    _roi(prefix + "/Court/CandidateStrikeROI", CANDIDATE_STRIKE_ROI)
    _roi(prefix + "/Court/SpawnRegion", SPAWN_REGION)
    _roi(prefix + "/Court/TargetZone", TARGET_ZONE)
    add_robot_visuals(prefix)


def add_robot_visuals(prefix):
    m = MORPH
    _box(prefix + "/Robot/MorphOnePlaceholder",
         (m["length_x"], m["width_y"], m["height_z"]), m["center_world"], (0.25, 0.35, 0.45))
    rig_x = CAMERA["rig_world"][0]; rig_z = CAMERA["rig_world"][2]
    _cyl(prefix + "/Robot/CameraRig/Mast", 0.02, rig_z, (rig_x, 0.0, rig_z / 2.0), (0.5, 0.5, 0.5))
    for nm, y in (("CameraLeftFrame", CAMERA["left_y"]), ("CameraRightFrame", -CAMERA["right_y"])):
        _box(prefix + "/Robot/CameraRig/" + nm, (0.05, 0.05, 0.05), (rig_x, y, rig_z), (0.95, 0.95, 0.25))
    _box(prefix + "/Robot/ComputeBoxPlaceholder", (0.20, 0.18, 0.08), (-1.60, 0.12, 0.29), (0.3, 0.3, 0.3))
    _box(prefix + "/Robot/BatteryPlaceholder", (0.30, 0.16, 0.08), (-1.60, -0.14, 0.29), (0.4, 0.4, 0.5))
    _box(prefix + "/Robot/EStopPlaceholder", (0.05, 0.05, 0.03), (-1.45, 0.22, 0.33), (1.0, 0.1, 0.1))


@configclass
class BadmintonSceneCfg(InteractiveSceneCfg):
    robot: object = ROBOT_ARTICULATION_CFG
    shuttle: object = SHUTTLE_CFG


def create_scene(num_envs: int = 1):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0"))
    scene = InteractiveScene(BadmintonSceneCfg(num_envs=num_envs,
                                               env_spacing=SIMULATION["env_spacing"]))
    gp = sim_utils.GroundPlaneCfg()
    gp.func("/World/Ground", gp)
    light = sim_utils.DistantLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light, translation=(0.0, 0.0, 10.0))
    add_court_statics("/World/envs/env_0")
    sim.reset()
    robot = scene.articulations["robot"]
    shuttle = scene.rigid_objects["shuttle"]
    return sim, scene, robot, shuttle
