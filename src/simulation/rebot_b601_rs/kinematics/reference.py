# -*- coding: utf-8 -*-
"""Independent reference implementations for the P6 dynamic FK (cross-check only).

Two paths, neither of which reuses fk.py's arithmetic:

  1. reference_pxr(q): builds an in-memory USD stage whose prim hierarchy is the
     real arm hierarchy, authors each link's local transform with pxr's OWN math
     (Gf.Matrix4d / Gf.Rotation / Gf.Quatd, i.e. C++ quaternion->matrix and a
     C++ 4x4 inverse), then asks UsdGeom.XformCache for the world pose.  The
     parent-chain accumulation is therefore USD's, not ours.
  2. reference_quat(q): pure python, rotation carried as Hamilton quaternions and
     composed by quaternion product instead of by 3x3 matrix product, with the
     SE(3) inverse taken as (conjugate quaternion, -R^T t).

Both consume the same committed anchors as fk.py -- the point is to check the
ALGEBRA and the ORDER of composition, not to re-derive the anchors.  The
anchors themselves are checked against the pxr-authored hop list in
p6_dynamic_fk_test.py.

No Isaac, no GPU, no files written.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import numpy as np

from .fk import (
    DOF_ORDER,
    JOINT_PARENT_CHILD,
    ANCHORS_JSON,
    JOINT_CHAIN_JSON,
    joint_motion,
    load_committed_anchors,
)

LINK_PARENT = {child: parent for (parent, child) in JOINT_PARENT_CHILD.values()}
LINK_PARENT["base_link"] = "arm_root"
LINK_PARENT["gripper_end"] = "link6"


# ---------------------------------------------------------------------------
# 1. pxr / USD path
# ---------------------------------------------------------------------------
def reference_pxr(q: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """World poses via a real UsdGeom.XformCache over a real USD prim hierarchy."""
    from pxr import Gf, Usd, UsdGeom  # local import: this path must stay optional

    import json

    raw = json.loads(ANCHORS_JSON.read_text(encoding="utf-8"))
    chain_raw = json.loads(JOINT_CHAIN_JSON.read_text(encoding="utf-8"))
    src = {j["prim"]: (j["local_pos0_m"], j["local_rot0_wxyz"], j["local_pos1_m"], j["local_rot1_wxyz"])
           for j in raw["joints"]}
    for j in chain_raw["joints"]:
        if j["prim"] not in src:
            src[j["prim"]] = (j["local_pos0"], j["local_rot0_wxyz"], j["local_pos1"], j["local_rot1_wxyz"])
    ge = {f["prim_name"]: f for f in chain_raw["named_frames"]}["gripper_end"]
    src["gripper_end"] = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    ge_pose = (ge["xformOp:translate"], ge["xformOp:orient_wxyz"])

    stage = Usd.Stage.CreateInMemory("p6_dynamic_fk_reference.usda")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    def make_xform(path: str, M: "Gf.Matrix4d"):
        prim = UsdGeom.Xform.Define(stage, path)
        op = prim.AddTransformOp()
        op.Set(M)
        return prim

    def gf_pose(pos, quat_wxyz):
        """pxr's own quaternion -> matrix, straight from the authored wxyz."""
        w, x, y, z = [float(v) for v in quat_wxyz]
        M = Gf.Matrix4d(1.0)
        M.SetTransform(Gf.Rotation(Gf.Quatd(w, x, y, z)), Gf.Vec3d(*[float(v) for v in pos]))
        return M

    def gf_joint(name: str, value: float):
        p0, q0, p1, q1 = src[name]
        X0 = gf_pose(p0, q0)
        X1 = gf_pose(p1, q1)
        if name in DOF_ORDER and name.startswith("gripper"):
            motion = Gf.Matrix4d(1.0)
            motion.SetTranslate(Gf.Vec3d(0.0, 0.0, float(value)))
        elif name in DOF_ORDER:
            motion = Gf.Matrix4d(1.0)
            motion.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(float(value))))
        else:
            motion = Gf.Matrix4d(1.0)
        if name == "gripper_end":
            X0 = gf_pose(*ge_pose)
            X1 = Gf.Matrix4d(1.0)
        # USD/Gf stores a transform TRANSPOSED (point maps as p_row * M, so the
        # translation sits in the last row).  A column-vector product
        #   T = X0 * M(q) * inv(X1)
        # therefore reads, in Gf order,
        #   inv(X1) * M(q) * X0
        # and that reversed order is exactly what this line asks Gf for.
        return X1.GetInverse() * motion * X0

    make_xform("/arm", Gf.Matrix4d(1.0))
    make_xform("/arm/base_link", gf_joint("root_joint", 0.0))
    paths = {"arm_root": "/arm", "base_link": "/arm/base_link"}
    for name in DOF_ORDER:
        parent, child = JOINT_PARENT_CHILD[name]
        paths[child] = paths[parent] + "/" + child
        make_xform(paths[child], gf_joint(name, q[name]))
    paths["gripper_end"] = paths["link6"] + "/gripper_end"
    make_xform(paths["gripper_end"], gf_joint("gripper_end", 0.0))

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for name in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6",
                 "gripper_left", "gripper_right", "gripper_end"):
        M = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(paths[name]))
        # Gf stores the transform transposed (row-vector convention), so the
        # column-convention pose fk.py speaks is M^T.
        T = np.eye(4)
        for i in range(4):
            for j in range(4):
                T[i, j] = M[j][i]
        out[name] = T
    out["arm_root"] = np.eye(4)
    return out


# ---------------------------------------------------------------------------
# 2. pure-python quaternion path
# ---------------------------------------------------------------------------
def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _qrot(q, v):
    """Rotate v by unit quaternion q (w,x,y,z) using q * (0,v) * q^-1."""
    return _qmul(_qmul(q, np.array([0.0, v[0], v[1], v[2]])), np.array([q[0], -q[1], -q[2], -q[3]]))[1:]


def _qmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _mat_quat(R):
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def _compose(qa, ta, qb, tb):
    """(qa, ta) * (qb, tb): rotation qa*qb, translation ta + qa*tb."""
    return _qmul(qa, qb), ta + _qrot(qa, tb)


def _inverse(q, t):
    qi = np.array([q[0], -q[1], -q[2], -q[3]])
    return qi, -_qrot(qi, t)


def reference_quat(q: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """World poses with rotations composed as Hamilton quaternion products."""
    import json

    raw = json.loads(ANCHORS_JSON.read_text(encoding="utf-8"))
    chain = json.loads(JOINT_CHAIN_JSON.read_text(encoding="utf-8"))
    src = {j["prim"]: (j["local_pos0_m"], j["local_rot0_wxyz"], j["local_pos1_m"], j["local_rot1_wxyz"])
           for j in raw["joints"]}
    for j in chain["joints"]:
        if j["prim"] not in src:
            src[j["prim"]] = (j["local_pos0"], j["local_rot0_wxyz"], j["local_pos1"], j["local_rot1_wxyz"])
    ge = {f["prim_name"]: f for f in chain["named_frames"]}["gripper_end"]
    src["gripper_end"] = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

    def norm(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    def joint_rt(name, value):
        p0, q0, p1, q1 = src[name]
        q0n, q1n = norm(q0), norm(q1)
        if name in DOF_ORDER and name.startswith("gripper"):
            qm, tm = np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, float(value)])
        elif name in DOF_ORDER:
            qm, tm = np.array([math.cos(value / 2.0), 0.0, 0.0, math.sin(value / 2.0)]), np.zeros(3)
        else:
            qm, tm = np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)
        R, t = _compose(q0n, np.asarray(p0, dtype=float), qm, tm)
        Ri, ti = _inverse(q1n, np.asarray(p1, dtype=float))
        return _compose(R, t, Ri, ti)

    # NOTE: the joint frame composes parent-joint-frame first, i.e. this path
    # multiplies right-to-left through the SAME tree but with translations
    # carried separately from rotations -- a different arithmetic order from
    # fk.py, which multiplies whole 4x4 matrices.
    poses = {"arm_root": (np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))}
    poses["base_link"] = joint_rt("root_joint", 0.0)
    for name in DOF_ORDER:
        parent, child = JOINT_PARENT_CHILD[name]
        qp, tp = poses[parent]
        ql, tl = joint_rt(name, q[name])
        poses[child] = _compose(qp, tp, ql, tl)
    q6, t6 = poses["link6"]
    poses["gripper_end"] = _compose(q6, t6, *_compose(
        norm(ge["xformOp:orient_wxyz"]), np.asarray(ge["xformOp:translate"], dtype=float),
        np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)))

    out = {}
    for name, (qq, tt) in poses.items():
        T = np.eye(4)
        T[:3, :3] = _qmat(qq / np.linalg.norm(qq))
        T[:3, 3] = tt
        out[name] = T
    return out
