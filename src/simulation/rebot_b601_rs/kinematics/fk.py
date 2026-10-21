# -*- coding: utf-8 -*-
"""P6 dynamic forward kinematics for the ReBot B601-RS 8-DOF chain.

WHAT THIS FILE IS
-----------------
The P6 frame work stopped at q = 0: mounted_chain_v2.json composes the anchors
with every joint coordinate at rest.  This module makes the joints MOVE.  One
joint is one USD PhysicsJoint, and a USD PhysicsJoint pins its joint frame twice:

    X0 = (physics:localPos0, physics:localRot0)      the joint frame in body0
    X1 = (physics:localPos1, physics:localRot1)      the joint frame in body1
    T_body0_body1(q) = X0 * M(q) * inv(X1)

M(q) is the joint coordinate expressed in the JOINT frame: a rotation about the
joint axis for a revolute joint, a translation along it for a prismatic one.
Every axis in this asset is authored as "Z" in the joint frame, so

    revolute : M(q) = Rz(q)
    prismatic: M(q) = Tz(q)

The five 180 deg body1 flips (localRot1 = (0, -1, 0, 0) wxyz) are NOT a special
case to patch: they are what inv(X1) does, and they are why the recorded local
pose of a child link at q = 0 is not simply X0.

WHAT IS *NOT* MODELLED HERE (see management/morph_one_rs/P6_DYNAMIC_FK.md)
-------------------------------------------------------------------------
* No verified TCP / flange transform: gripper_end is an AUTHORED xformOp frame,
  not a calibrated tool centre point.
* No measured racket face: the racket mount edges (13/14/15) are still UNKNOWN.
* No D455 extrinsics, no collision geometry, no dynamics (mass, inertia,
  friction, compliance, gearbox backlash).  This is kinematics only.

FRAME
-----
All poses returned here are in the ARM ROOT frame (/ReBot_Arm_RS), which the P3
evidence records as identical to the chassis/retarget frame (edge 2 is still
UNKNOWN).  A base pose can be passed to change the frame; nothing else assumes
one.

UNITS: metre / radian.  Quaternions leave this module as wxyz (USD convention);
any other convention must be declared at the boundary (quat_to_matrix order=...).

Pure numpy.  No pxr, no Isaac, no GPU, no network.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
P6_FRAMES = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
ANCHORS_JSON = P6_FRAMES / "joint_anchors.json"
JOINT_CHAIN_JSON = P6_FRAMES / "evidence" / "robot_usd_joint_chain.json"

#: The 8 driven coordinates of the B601-RS arm, in the order this module uses
#: for a bare 8-vector.  THIS IS NOT THE USD STAGE PRIM ORDER and it is not
#: derived from it: the stage also carries root_joint (fixed, no coordinate) and
#: the caller is expected to address joints BY NAME.  See coerce_q().
DOF_ORDER = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
)

#: joint name -> (parent link, child link)
JOINT_PARENT_CHILD = {
    "joint1": ("base_link", "link1"),
    "joint2": ("link1", "link2"),
    "joint3": ("link2", "link3"),
    "joint4": ("link3", "link4"),
    "joint5": ("link4", "link5"),
    "joint6": ("link5", "link6"),
    "gripper_joint1": ("link6", "gripper_left"),
    "gripper_joint2": ("link6", "gripper_right"),
}

#: link name -> the joint that moves it (None for the chain root).
LINK_JOINT = {child: j for j, (parent, child) in JOINT_PARENT_CHILD.items()}
LINK_JOINT["base_link"] = "root_joint"
LINK_JOINT["arm_root"] = None

LINK_NAMES = (
    "arm_root",
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_left",
    "gripper_right",
    "gripper_end",
)

#: (lower, upper) in SI: rad for the revolutes, METRE for the jaws.
#: Source: joint_anchors.json / evidence robot_usd_joint_chain.json limits_*_as_authored.
#: NOTE the jaw limits are authored in metres while the layer's unit field says
#: "degree" -- that field is wrong; the numbers are metres (0..0.05, 0..0.0715).
LIMITS = {
    "joint1": (-2.7999999539821165, 2.7999999539821165),
    "joint2": (0.0, 3.1400000406473483),
    "joint3": (0.0, 3.1400000406473483),
    "joint4": (-1.5700001075901369, 1.5700001075901369),
    "joint5": (-1.5700001075901369, 1.5700001075901369),
    "joint6": (-3.1400000406473483, 3.1400000406473483),
    "gripper_joint1": (0.0, 0.05),
    "gripper_joint2": (0.0, 0.0715),
}

JOINT_TYPE = {name: ("prismatic" if name.startswith("gripper") else "revolute") for name in DOF_ORDER}


class ConventionError(ValueError):
    """A quaternion was handed over without a declared, matching convention."""


class QError(ValueError):
    """q was not a complete, correctly named 8-vector."""


# ---------------------------------------------------------------------------
# SO(3) / SE(3) primitives.  Quaternion convention is always EXPLICIT.
# ---------------------------------------------------------------------------
def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ConventionError("zero-norm quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_to_matrix(quat: Sequence[float], order: str = "wxyz") -> np.ndarray:
    """Quaternion -> rotation matrix with the convention declared by the caller.

    order="wxyz"  USD / GfQuatf order (real part first)
    order="xyzw"  Hamilton / SciPy / ROS order -- the SAME four numbers mean a
                  different rotation, so the swap is never implicit.
    """
    q = np.asarray(quat, dtype=float).reshape(4)
    if order == "wxyz":
        return quat_wxyz_to_matrix(q)
    if order == "xyzw":
        return quat_wxyz_to_matrix(np.array([q[3], q[0], q[1], q[2]], dtype=float))
    raise ConventionError(f"unknown quaternion order {order!r}; declare 'wxyz' (USD) or 'xyzw'")


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion (w, x, y, z) with w >= 0."""
    m = np.asarray(R, dtype=float)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues rotation about an axis (normalised internally)."""
    a = np.asarray(axis, dtype=float).reshape(3)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=float)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def pose_from(pos: Sequence[float], R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Closed-form SE(3) inverse: R^T and -R^T p (no generic 4x4 solve)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in radians (numerically safe)."""
    c = (float(np.trace(np.asarray(R, dtype=float))) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle), radians."""
    R = np.asarray(R, dtype=float)
    ang = rotation_angle(R)
    if ang < 1e-12:
        # first order: the skew part is 2*sin(ang)*axis
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if abs(math.pi - ang) < 1e-6:
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        axis = A[:, i] / math.sqrt(max(A[i, i], 1e-300))
        return axis / np.linalg.norm(axis) * ang
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return axis * ang


# ---------------------------------------------------------------------------
# Anchors -> per-joint constant frames
# ---------------------------------------------------------------------------
def _pose(pos, R) -> np.ndarray:
    """(translation, 3x3 rotation) -> 4x4 pose."""
    return pose_from(pos, R)


def load_committed_anchors(quat_order: str = "wxyz") -> Dict[str, dict]:
    """Read the committed anchors and return per-joint {X0, X1, axis, type}.

    quat_order must describe the SOURCE.  Both committed files declare wxyz
    (USD convention); quat_order="xyzw" exists only so the convention test can
    show that a swapped read is a DIFFERENT pose, not a rounding difference.
    """
    anchors = json.loads(ANCHORS_JSON.read_text(encoding="utf-8"))
    chain = json.loads(JOINT_CHAIN_JSON.read_text(encoding="utf-8"))

    declared = anchors.get("units", {}).get("quaternion", "wxyz (USD convention)")
    if "wxyz" not in declared:
        raise ConventionError(f"anchor file declares an unexpected quaternion order: {declared!r}")
    declared_chain = chain.get("quaternion_convention", {}).get("this_record")
    if declared_chain != "wxyz":
        raise ConventionError(f"joint-chain record declares {declared_chain!r}, expected 'wxyz'")

    def wrap(name, pos, rot, axis, jtype, source):
        return {
            "X0": _pose(pos, quat_to_matrix(rot, quat_order)),
            "X1": np.eye(4),
            "axis": axis,
            "type": jtype,
            "source": source,
        }

    out: Dict[str, dict] = {}
    for joint in anchors["joints"]:
        out[joint["prim"]] = {
            "X0": _pose(joint["local_pos0_m"], quat_to_matrix(joint["local_rot0_wxyz"], quat_order)),
            "X1": _pose(joint["local_pos1_m"], quat_to_matrix(joint["local_rot1_wxyz"], quat_order)),
            "axis": None if joint["physics_axis"] is None else str(joint["physics_axis"]),
            "type": "fixed" if joint["usd_type"].endswith("FixedJoint") else "revolute",
            "source": joint["source_lines"],
        }

    for joint in chain["joints"]:
        if joint["prim"] in out:
            continue
        out[joint["prim"]] = {
            "X0": _pose(joint["local_pos0"], quat_to_matrix(joint["local_rot0_wxyz"], quat_order)),
            "X1": _pose(joint["local_pos1"], quat_to_matrix(joint["local_rot1_wxyz"], quat_order)),
            "axis": joint["axis"],
            "type": "prismatic" if joint["type"].endswith("PrismaticJoint") else "revolute",
            "source": {"file": "payloads/Physics/physics.usda", "locator": joint["prim"]},
        }

    # gripper_end is an authored UsdGeom frame on link6, not a joint
    named = {f["prim_name"]: f for f in chain["named_frames"]}
    ge = named["gripper_end"]
    out["gripper_end"] = wrap(
        "gripper_end",
        ge["xformOp:translate"],
        ge["xformOp:orient_wxyz"],
        None,
        "fixed",
        {"file": "ReBot_Arm_RS/payloads (composed stage)", "locator": "gripper_end xformOps"},
    )
    return out


def coerce_q(q, dof_order: Sequence[str] = DOF_ORDER) -> Dict[str, float]:
    """Normalise q to {joint name: value}.

    A Mapping is the supported form: joints are addressed BY NAME, so the order
    of the USD stage (or of anyone's JSON) is irrelevant.  A bare sequence of
    len(DOF_ORDER) is accepted only as the documented local order.
    """
    if isinstance(q, Mapping):
        missing = [n for n in dof_order if n not in q]
        extra = [n for n in q if n not in dof_order]
        if missing or extra:
            raise QError(f"q must name exactly the 8 driven joints; missing={missing} extra={extra}")
        out = {n: float(q[n]) for n in dof_order}
    else:
        arr = np.asarray(q, dtype=float).ravel()
        if arr.size != len(dof_order):
            raise QError(f"q must have {len(dof_order)} values, got {arr.size}")
        out = {n: float(v) for n, v in zip(dof_order, arr)}
    for n, v in out.items():
        if not math.isfinite(v):
            raise QError(f"q[{n}] is not finite: {v}")
    return out


def joint_motion(name: str, q: float) -> np.ndarray:
    """M(q) in the joint frame.  Every authored axis in this asset is Z.

    A fixed joint (root_joint, gripper_end) has no coordinate: M(q) is the
    identity for every q, including q = 0.
    """
    kind = JOINT_TYPE.get(name)
    if kind is None:
        return np.eye(4)
    if kind == "revolute":
        return pose_from([0.0, 0.0, 0.0], axis_angle_matrix([0.0, 0.0, 1.0], q))
    return pose_from([0.0, 0.0, q], np.eye(3))


# ---------------------------------------------------------------------------
# FK
# ---------------------------------------------------------------------------
class Chain:
    """The 8-DOF B601-RS kinematic chain, loaded once from the committed anchors."""

    def __init__(self, anchors: Dict[str, dict] | None = None):
        self.anchors = anchors if anchors is not None else load_committed_anchors()
        T_root = self.anchors["root_joint"]["X0"] @ se3_inverse(self.anchors["root_joint"]["X1"])
        if float(np.max(np.abs(T_root - np.eye(4)))) > 1e-9:
            raise ConventionError("root_joint is not the identity; the arm-root frame assumption is void")

    # -- single joint -----------------------------------------------------
    def joint_local(self, name: str, q: float) -> np.ndarray:
        """X0 * M(q) * inv(X1): the body0 -> body1 transform of one joint."""
        a = self.anchors[name]
        return a["X0"] @ joint_motion(name, q) @ se3_inverse(a["X1"])

    def joint_world_frames(self, q, base: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        """World pose of every joint frame AND every link, keyed by name.

        The joint frame of joint j is the frame M(q) acts in:
        T_world_jointframe = T_world_body0 * X0.  Its z column is the true world
        axis, which is what the Jacobian needs.
        """
        qn = coerce_q(q)
        T_world = {"arm_root": np.eye(4) if base is None else np.asarray(base, dtype=float)}
        T_world["base_link"] = T_world["arm_root"] @ self.joint_local("root_joint", 0.0)
        joint_frames: Dict[str, np.ndarray] = {}
        for name in DOF_ORDER:
            parent, child = JOINT_PARENT_CHILD[name]
            T_parent = T_world[parent]
            joint_frames[name] = T_parent @ self.anchors[name]["X0"]
            T_world[child] = T_parent @ self.joint_local(name, qn[name])
        T_world["gripper_end"] = T_world["link6"] @ self.joint_local("gripper_end", 0.0)
        out = dict(T_world)
        out.update({"joint_frame:" + k: v for k, v in joint_frames.items()})
        return out

    # -- the public call --------------------------------------------------
    def fk(self, q, base: np.ndarray | None = None, links: Sequence[str] | None = None) -> Dict[str, np.ndarray]:
        """Forward kinematics: q (8 named values) -> {link: 4x4 world pose}."""
        frames = self.joint_world_frames(q, base=base)
        names = LINK_NAMES if links is None else links
        return {n: frames[n] for n in names}

    def pose(self, q, link: str = "gripper_end", base: np.ndarray | None = None) -> np.ndarray:
        return self.fk(q, base=base, links=(link,))[link]

    def within_limits(self, q, tol: float = 1e-9) -> Dict[str, bool]:
        qn = coerce_q(q)
        return {n: (LIMITS[n][0] - tol) <= v <= (LIMITS[n][1] + tol) for n, v in qn.items()}

    # -- Jacobian ---------------------------------------------------------
    def jacobian(self, q, link: str = "gripper_end") -> np.ndarray:
        """Analytic 6x8 geometric Jacobian of the given link, in the arm-root frame.

        Rows 0..2: d(position)/d(q)   m/rad  (m/m for the jaws)
        Rows 3..5: d(rotation vector)/d(q)   rad/rad (rad/m for the jaws)

        Column j is [z_w x (p_link - p_joint); z_w] for a revolute joint and
        [z_w; 0] for a prismatic one, with z_w the joint axis in the world frame.
        """
        qn = coerce_q(q)
        frames = self.joint_world_frames(qn)
        T_link = frames[link]
        p_link = T_link[:3, 3]
        J = np.zeros((6, len(DOF_ORDER)))
        ancestors = self._ancestor_joints(link)
        for col, name in enumerate(DOF_ORDER):
            if name not in ancestors:
                continue
            T_j = frames["joint_frame:" + name]
            z = T_j[:3, 2] / np.linalg.norm(T_j[:3, 2])
            if JOINT_TYPE[name] == "revolute":
                J[:3, col] = np.cross(z, p_link - T_j[:3, 3])
                J[3:, col] = z
            else:
                J[:3, col] = z
        return J

    def _ancestor_joints(self, link: str) -> set:
        if link == "arm_root":
            return set()
        if link == "base_link":
            return {"root_joint"}
        if link == "gripper_end":
            return {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}
        path = []
        node = link
        while node in LINK_JOINT and LINK_JOINT[node] is not None:
            j = LINK_JOINT[node]
            path.append(j)
            if j not in JOINT_PARENT_CHILD:  # root_joint: the chain root above base_link
                break
            node = JOINT_PARENT_CHILD[j][0]
        return set(path)

    def jacobian_central(self, q, link: str = "gripper_end", h: float = 1e-6) -> np.ndarray:
        """Central-difference Jacobian of the same map, step h (rad; m for jaws).

        Position rows use the plain central difference; rotation rows use the
        SO(3) log of R(q+h) * R(q-h)^T, i.e. the derivative of the rotation
        vector at q, exact to O(h^2) with no Euler-angle singularity.
        """
        qn = coerce_q(q)
        J = np.zeros((6, len(DOF_ORDER)))
        for col, name in enumerate(DOF_ORDER):
            plus = dict(qn)
            minus = dict(qn)
            plus[name] = qn[name] + h
            minus[name] = qn[name] - h
            Tp = self.pose(plus, link)
            Tm = self.pose(minus, link)
            J[:3, col] = (Tp[:3, 3] - Tm[:3, 3]) / (2.0 * h)
            J[3:, col] = so3_log(Tp[:3, :3] @ Tm[:3, :3].T) / (2.0 * h)
        return J


def q_from_names(**kwargs) -> Dict[str, float]:
    """Keyword sugar: q_from_names(joint1=0.1, joint2=0.2, ...)."""
    return coerce_q(kwargs)
