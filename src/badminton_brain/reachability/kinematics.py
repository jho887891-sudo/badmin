# -*- coding: utf-8 -*-
"""P7-09: a documented analytic forward kinematics for the racket-face point.

WHY A PROXY TABLE AND NOT THE REAL URDF
---------------------------------------
The B601-RS / PiPER URDF is NOT in this repository (assets live on the resource
host, see AGENTS.md) and no unit has been measured.  Every number in
PROXY_KINEMATIC_TABLE is therefore an ENGINEERING_V0_1 PROXY: it produces a
kinematic chain of roughly the right size and joint layout so that the
reachability/IK *interface* can be built and tested, and it MUST NOT be read as
the kinematic model of the real arm.

Consequence, stated once and carried everywhere:
  * a converged IK result on this table proves the SOLVER works;
  * it proves NOTHING about whether the real PiPER can reach a real shuttle;
  * reachability.assess_reachability() therefore keeps an UNKNOWN path alive for
    everything that depends on unverified geometry.

Convention: standard (distal) DH, all joints revolute, metres and radians.
    T_i = Rz(q_i) . Tz(d_i) . Tx(a_i) . Rx(alpha_i)
    T_0_face = T_1 . T_2 . ... . T_6 . T_tool
The tool frame is the RACKET FACE CENTRE, 0.50 m along +z of link6 - the same
0.50 m as badminton_scene/racket_cfg.py RACKET_GEO["head_center_from_tcp"], so
the arm model and the scene asset agree about where the face is.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "DOF", "JOINT_NAMES", "PROXY_KINEMATIC_TABLE", "PROXY_SOURCE",
    "JOINT_LOWER_RAD", "JOINT_UPPER_RAD", "LIMIT_SOURCE",
    "FACE_FROM_LINK6_M", "MAX_FACE_REACH_M",
    "dh_transform", "forward_kinematics_face", "face_position", "geometric_jacobian",
]

DOF = 6
JOINT_NAMES: Tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

PROXY_SOURCE = (
    "ENGINEERING_V0_1 PROXY. The PiPER/B601-RS URDF is not in this repository and no unit "
    "has been measured. Link lengths below are placeholders chosen to match the published "
    "PiPER class geometry (about 1.25 m to the racket face centre); they are NOT measured "
    "and NOT verified. Replace with URDF-derived values before any real reachability claim."
)

#: (d_i [m], a_i [m], alpha_i [rad]) per joint.  PROXY - see PROXY_SOURCE.
PROXY_KINEMATIC_TABLE: Tuple[Tuple[float, float, float], ...] = (
    (0.123, 0.000, math.pi / 2.0),
    (0.000, 0.285, 0.0),
    (0.000, 0.020, -math.pi / 2.0),
    (0.250, 0.000, math.pi / 2.0),
    (0.000, 0.000, -math.pi / 2.0),
    (0.088, 0.000, 0.0),
)

#: Distance link6 origin -> racket face centre, along +z of link6.
FACE_FROM_LINK6_M = 0.50

#: Joint limits, radians, from the PiPER URDF limits quoted by
#: simulation/piper_stage0/piper_cfg.py (joint2 in [0, 3.14], joint3 in [-2.967, 0]);
#: the remaining four are the published PiPER joint ranges.  NOT measured on a unit.
JOINT_LOWER_RAD: Tuple[float, ...] = (-2.618, 0.0, -2.967, -1.745, -1.220, -2.094)
JOINT_UPPER_RAD: Tuple[float, ...] = (2.618, 3.141592653589793, 0.0, 1.745, 1.220, 2.094)

LIMIT_SOURCE = (
    "URDF-DECLARED (PiPER): joint2 [0, 3.14] and joint3 [-2.967, 0] are quoted in "
    "simulation/piper_stage0/piper_cfg.py; the other four are the published PiPER ranges. "
    "Not verified against a physical unit or the USD stage."
)

#: Largest distance from the base origin to the face centre this proxy can produce.
MAX_FACE_REACH_M = (sum(abs(a) + abs(d) for d, a, _ in PROXY_KINEMATIC_TABLE)
                    + FACE_FROM_LINK6_M)


def dh_transform(d: float, a: float, alpha: float, q: float) -> np.ndarray:
    """Standard (distal) DH transform, 4x4 homogeneous."""
    cq, sq = math.cos(q), math.sin(q)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [cq, -sq * ca, sq * sa, a * cq],
            [sq, cq * ca, -cq * sa, a * sq],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _check_q(q: Sequence[float]) -> np.ndarray:
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.shape != (DOF,):
        raise ValueError(f"joint vector must have {DOF} entries, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("joint vector contains NaN/Inf")
    return arr


def forward_kinematics_face(q: Sequence[float]) -> np.ndarray:
    """4x4 pose of the racket face centre in the arm base frame."""
    arr = _check_q(q)
    T = np.eye(4, dtype=float)
    for index, (d, a, alpha) in enumerate(PROXY_KINEMATIC_TABLE):
        T = T @ dh_transform(d, a, alpha, float(arr[index]))
    tool = np.eye(4, dtype=float)
    tool[2, 3] = FACE_FROM_LINK6_M
    return T @ tool


def face_position(q: Sequence[float]) -> np.ndarray:
    """Position (3,) of the racket face centre in the arm base frame."""
    return forward_kinematics_face(q)[:3, 3].copy()


def geometric_jacobian(q: Sequence[float], *, step: float = 1e-6) -> np.ndarray:
    """6x6 geometric Jacobian of the face pose, by CENTRAL DIFFERENCES on the DH FK.

    Documented approximation: central differences make the Jacobian second-order
    accurate in step; step is FIXED (never adaptive) so the same joint vector
    always yields the same matrix and the tests stay deterministic.
    """
    arr = _check_q(q)
    jac = np.zeros((6, DOF), dtype=float)
    for index in range(DOF):
        plus = arr.copy()
        minus = arr.copy()
        plus[index] += step
        minus[index] -= step
        Tp = forward_kinematics_face(plus)
        Tm = forward_kinematics_face(minus)
        jac[:3, index] = (Tp[:3, 3] - Tm[:3, 3]) / (2.0 * step)
        Rrel = Tp[:3, :3] @ Tm[:3, :3].T
        jac[3:, index] = np.array(
            [
                Rrel[2, 1] - Rrel[1, 2],
                Rrel[0, 2] - Rrel[2, 0],
                Rrel[1, 0] - Rrel[0, 1],
            ]
        ) / (4.0 * step)
    return jac
