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