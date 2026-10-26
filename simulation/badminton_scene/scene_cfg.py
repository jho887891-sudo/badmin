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

import torch
from .scene_layout import (CAMERA, PIPER, MORPH_PLACEHOLDER,
                           PERCEPTION_ROI, CANDIDATE_STRIKE_ROI, SPAWN_REGION,
                           TARGET_ZONE, SIMULATION, REBOT_RS)
from .robot_cfg import (ARM_MODEL, MORPH, ROBOT_ARTICULATION_CFG,
                        REBOT_RS_PRIM_PATH, spawn_rebot_rs_directly)
from .assembly_route import resolve_scene_arm_route
from .rs_physics import rebot_rs_simulation_cfg
from .shuttle_cfg import SHUTTLE_CFG

SIM_DT = SIMULATION["physics_dt"]
_MAT_COUNTER = [0]


def _stage():
    return omni.usd.get_context().get_stage()


COURT_ASSET_USD = "/home/T7/ojh/robot_sim/assets/court/badminton_court.usd"


def add_court_asset(prefix):
    """Reference the generated parameterized court asset (configs/court.yaml)."""
    st = _stage()
    path = prefix + "/CourtAsset"
    xf = UsdGeom.Xform.Define(st, path)
    prim = xf.GetPrim()
    prim.GetReferences().AddReference(COURT_ASSET_USD)
    return prim


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
    """Court ONLY: visuals and colliders come from the generated asset.

    configs/court.yaml -> assets/court/badminton_court.usd (tools/build_badminton_court.py).
    No hand-drawn court / ROI / placeholder primitives are authored here.
    """
    add_court_asset(prefix)


@configclass
class BadmintonSceneCfg(InteractiveSceneCfg):
    """Full scene: court asset + robot + shuttle (used only when explicitly enabled)."""
    robot: object = ROBOT_ARTICULATION_CFG
    shuttle: object = SHUTTLE_CFG


@configclass
class ShuttleOnlySceneCfg(InteractiveSceneCfg):
    """Shuttle without the scene-registered robot (the formal arm is spawned directly)."""
    shuttle: object = SHUTTLE_CFG


@configclass
class CourtOnlySceneCfg(InteractiveSceneCfg):
    """Default scene: the user's court asset only (no robot, no shuttle, no ROIs,
    no placeholders, no ground). A camera may still be attached for rendering."""


def create_scene(num_envs: int = 1, contact_cfg=None, extra_sensors=None, pre_axes=None, statics_mode="first",
                 include_robot: bool = False, include_shuttle: bool = False, include_ground: bool = False,
                 include_racket: bool = False, racket_mount_transform=None, racket_visual_path=None):
    # F1: the converted B601-RS gets no drive torque when registered through InteractiveScene, so the
    # formal arm is spawned and bound directly (verified to hold the rest pose). See
    # management/rebot_b601_rs/ARM_MIGRATION_RECORD.md section E/F1.
    # The route (and the num_envs guard) lives in a pure helper so it can be unit tested without
    # Isaac Sim: see tests/simulation/rebot_b601_rs/test_scene_arm_route.py.
    _direct_arm = resolve_scene_arm_route(
        include_robot=include_robot, arm_model=ARM_MODEL, num_envs=num_envs) == "direct"
    if include_racket:
        if not include_robot or not _direct_arm:
            raise ValueError("include_racket requires include_robot=True with the formal B601-RS direct-articulation route")
        # Validate the mount and visual asset before creating SimulationContext or touching the GPU.
        from .racket_mount_adapter import validate_mount_transform, resolve_racket_visual_path
        validate_mount_transform(racket_mount_transform)
        racket_visual_path = str(resolve_racket_visual_path(racket_visual_path))
    # Only the formal B601-RS path opts into the TGS velocity-consistency fix.
    _sim_cfg = (rebot_rs_simulation_cfg(dt=SIM_DT) if _direct_arm
                else sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0"))
    sim = sim_utils.SimulationContext(_sim_cfg)
    if _direct_arm:
        _mount = REBOT_RS["mount_world"]
        _origin0 = SIMULATION["env_spacing"] * 0.0  # env_0 origin x offset is 0; see env_origin helper
        _direct_robot = spawn_rebot_rs_directly(
            translation=(_mount[0] + _origin0, _mount[1], _mount[2]))
        _cfg_cls = ShuttleOnlySceneCfg if include_shuttle else CourtOnlySceneCfg
    else:
        _direct_robot = None
        _cfg_cls = BadmintonSceneCfg if (include_robot or include_shuttle) else CourtOnlySceneCfg
    scene = InteractiveScene(_cfg_cls(num_envs=num_envs, env_spacing=SIMULATION["env_spacing"]))
    if include_racket:
        from .racket_mount_adapter import attach_racket_visual
        attach_racket_visual(
            stage=_stage(),
            arm_prim_path=REBOT_RS_PRIM_PATH,
            mount_transform=racket_mount_transform,
            asset_path=racket_visual_path,
        )
    # large static ground collider covering the whole vectorised env grid
    _eo = scene.env_origins.cpu().float()
    _cx = float((_eo[:, 0].min() + _eo[:, 0].max()) / 2.0)
    _cy = float((_eo[:, 1].min() + _eo[:, 1].max()) / 2.0)
    _half = float(max(120.0, 0.6 * (float(_eo[:, 0].max() - _eo[:, 0].min()) + float(_eo[:, 1].max() - _eo[:, 1].min()))))
    if include_ground:
        _g = _box("/World/Ground", (2.0 * _half, 2.0 * _half, 0.40), (_cx, _cy, -0.203),
                  (0.10, 0.10, 0.10), collision=True)
        # _box returns a prim PATH; UsdGeom.Imageable needs the prim itself (passing the string raises
        # Boost.Python.ArgumentError). This path is only reached with include_ground=True, which the
        # court-only default never exercised.
        UsdGeom.Imageable(_stage().GetPrimAtPath(_g)).MakeInvisible()
    light = sim_utils.DistantLightCfg(intensity=700.0, color=(0.8, 0.8, 0.8))
    light.func("/World/Light", light, translation=(0.0, 0.0, 10.0))
    dome = sim_utils.DomeLightCfg(intensity=350.0, color=(0.85, 0.88, 0.95))
    dome.func("/World/DomeLight", dome)
    _envs_statics = range(num_envs) if statics_mode == "all" else range(min(1, num_envs))
    for _i in _envs_statics:
        add_court_statics("/World/envs/env_%d" % _i)
    contact_sensor = None
    if contact_cfg is not None:
        from isaaclab.sensors import ContactSensor
        contact_sensor = ContactSensor(contact_cfg)
        scene.sensors["shuttle_contact"] = contact_sensor
    extra = {}
    if extra_sensors:
        for _name, _cfg, _cls in extra_sensors:
            _obj = _cls(_cfg)
            scene.sensors[_name] = _obj
            extra[_name] = _obj
    if pre_axes:
        for _nm, _ln, _th in pre_axes:
            _p = "/World/Axes/" + _nm
            _box(_p + "/X", (_ln, _th, _th), (_ln / 2.0, 0.0, 0.0), (1.0, 0.05, 0.05))
            _box(_p + "/Y", (_th, _ln, _th), (0.0, _ln / 2.0, 0.0), (0.05, 1.0, 0.05))
            _box(_p + "/Z", (_th, _th, _ln), (0.0, 0.0, _ln / 2.0), (0.05, 0.2, 1.0))
            _xf = UsdGeom.Xform.Define(_stage(), _p)
            _xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            # An xformOp's precision is fixed when the op is created: an op added here wants GfQuatf while a
            # spawner-created op wants GfQuatd. Setting the wrong one raises "Type mismatch for ...: expected
            # 'GfQuatf', got 'GfQuatd'" (and the mirror image), so match the op's own declared precision.
            _orient = _xf.AddOrientOp()
            if _orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                _orient.Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
            else:
                _orient.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    sim.reset()
    if _direct_arm:
        _jp = torch.tensor(
            [[REBOT_RS["rest_joint_pos"].get(nm, 0.0) for nm in _direct_robot.joint_names]],
            dtype=torch.float32, device=_direct_robot.device)
        _direct_robot.write_joint_state_to_sim(_jp, torch.zeros_like(_jp))
        _direct_robot.set_joint_position_target(_jp)
        _direct_robot.write_data_to_sim()
        scene.direct_articulations = [_direct_robot]
        robot = _direct_robot
    else:
        robot = scene.articulations.get("robot") if include_robot else None
    shuttle = scene.rigid_objects.get("shuttle") if include_shuttle else None
    return sim, scene, robot, shuttle, contact_sensor, extra


def step_env(scene, sim, dt=None):
    """One simulation step with data sync + buffer update.

    Articulations spawned directly (F1 workaround) are not part of InteractiveScene, so they are
    written before the step and refreshed after it.
    """
    _dt = SIM_DT if dt is None else dt
    _direct = getattr(scene, "direct_articulations", ())
    scene.write_data_to_sim()
    for _art in _direct:
        _art.write_data_to_sim()
    sim.step()
    scene.update(_dt)
    for _art in _direct:
        _art.update(_dt)
