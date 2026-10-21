# -*- coding: utf-8 -*-
"""P6 dynamic FK: emit the validation numbers quoted in P6_DYNAMIC_FK.md.

Runs the numpy FK, the two independent reference paths and the Jacobian checks,
and writes outputs/simulation/rebot_b601_rs/morph_one/p6_fk/fk_validation.json.

    python scripts/simulation/p6_dynamic_fk_validate.py

Read-only on the asset; writes only the one committed JSON under outputs/.
No Isaac, no GPU, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from rebot_b601_rs.kinematics.fk import (  # noqa: E402
    DOF_ORDER,
    Chain,
    matrix_to_quat_wxyz,
    rotation_angle,
    so3_log,
)
from rebot_b601_rs.kinematics.reference import reference_pxr, reference_quat  # noqa: E402

OUT = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_fk/fk_validation.json"
PXR_REST_POSITION_M = (0.301710169999, 2.62633e-07, 0.217746720407)
PXR_REST_QUAT_WXYZ = (1.0, -6.9738e-08, -1.7116e-08, 0.0)

Q_ZERO = {n: 0.0 for n in DOF_ORDER}
BLOCKS = {
    "A_mixed_signs": dict(zip(DOF_ORDER, [-0.7, 0.9, -1.2, 0.4, -0.9, 1.7, 0.02, 0.03])),
    "B_all_positive_jaws_open": dict(zip(DOF_ORDER, [0.3, 0.45, 0.6, 0.5, 0.35, 0.2, 0.05, 0.0715])),
    "C_near_limits_inside": dict(zip(DOF_ORDER, [2.79, 3.13, 3.13, -1.56, 1.56, -3.13, 0.049, 0.07])),
}


def main() -> int:
    chain = Chain()
    out = {
        "schema": "p6_dynamic_fk_validation/1",
        "task": "P6 item C3 -- dynamic forward kinematics",
        "dof_order": list(DOF_ORDER),
        "frames": "arm root (/ReBot_Arm_RS) == chassis frame after the P3 retarget",
        "units": "metre, radian; quaternion wxyz (USD)",
        "zero_pose": {},
        "nonzero_pose": {},
        "jacobian": {},
        "reference_check": {},
    }

    T0 = chain.pose(Q_ZERO, "gripper_end")
    out["zero_pose"] = {
        "computed_position_m": [float(v) for v in T0[:3, 3]],
        "computed_quat_wxyz": [float(v) for v in matrix_to_quat_wxyz(T0[:3, :3])],
        "pxr_committed_position_m": list(PXR_REST_POSITION_M),
        "pxr_committed_quat_wxyz": list(PXR_REST_QUAT_WXYZ),
        "max_abs_position_delta_m": float(np.max(np.abs(T0[:3, 3] - np.asarray(PXR_REST_POSITION_M)))),
        "geodesic_angle_rad": float(
            rotation_angle(
                T0[:3, :3]
                @ np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
            )
        ),
        "det_R": float(np.linalg.det(T0[:3, :3])),
        "note": "residual = float32 anchors in the USD layer, not a convention error",
    }

    for label, q in BLOCKS.items():
        T = chain.fk(q)
        J = chain.jacobian(q)
        Jc = chain.jacobian_central(q, h=1e-6)
        # J vs a forward difference of fk() itself
        T0q = chain.pose(q, "gripper_end")
        h = 1e-6
        Jfd = np.zeros_like(J)
        for col, name in enumerate(DOF_ORDER):
            qp = dict(q)
            qp[name] = q[name] + h
            Tp = chain.pose(qp, "gripper_end")
            Jfd[:3, col] = (Tp[:3, 3] - T0q[:3, 3]) / h
            Jfd[3:, col] = so3_log(Tp[:3, :3] @ T0q[:3, :3].T) / h
        pxr = reference_pxr(q)
        quat = reference_quat(q)
        out["nonzero_pose"][label] = {
            "q": {k: float(v) for k, v in q.items()},
            "within_limits": chain.within_limits(q),
            "gripper_end_position_m": [float(v) for v in T["gripper_end"][:3, 3]],
            "gripper_end_quat_wxyz": [float(v) for v in matrix_to_quat_wxyz(T["gripper_end"][:3, :3])],
            "min_det_R": float(min(np.linalg.det(M[:3, :3]) for M in T.values())),
            "max_det_R": float(max(np.linalg.det(M[:3, :3]) for M in T.values())),
            "max_orthonormality_error": float(
                max(np.max(np.abs(M[:3, :3] @ M[:3, :3].T - np.eye(3))) for M in T.values())
            ),
        }
        out["jacobian"][label] = {
            "analytic_vs_central_max_abs": float(np.max(np.abs(J - Jc))),
            "analytic_vs_forward_diff_max_abs": float(np.max(np.abs(J - Jfd))),
            "central_step": 1e-6,
            "forward_diff_step": h,
        }
        out["reference_check"][label] = {
            "pxr_xformcache_max_position_delta_m": float(
                max(np.max(np.abs(T[k][:3, 3] - pxr[k][:3, 3])) for k in pxr)
            ),
            "pxr_xformcache_max_geodesic_angle_rad": float(
                max(rotation_angle(T[k][:3, :3] @ pxr[k][:3, :3].T) for k in pxr)
            ),
            "quaternion_algebra_max_abs_matrix_delta": float(
                max(np.max(np.abs(T[k] - quat[k])) for k in quat if k in T)
            ),
        }

    # joint6 is a pure roll of the tool frame (the end frame is on its axis)
    q6 = dict(Q_ZERO)
    q6["joint6"] = 0.2
    T6 = chain.pose(q6, "gripper_end")
    p6 = reference_pxr(q6)["gripper_end"]
    out["joint6_is_a_pure_roll"] = {
        "numpy_delta_position_m": float(np.linalg.norm(T6[:3, 3] - T0[:3, 3])),
        "numpy_delta_angle_rad": float(rotation_angle(T6[:3, :3] @ T0[:3, :3].T)),
        "pxr_delta_position_m": float(np.linalg.norm(p6[:3, 3] - reference_pxr(Q_ZERO)["gripper_end"][:3, 3])),
        "pxr_delta_angle_rad": float(
            rotation_angle(p6[:3, :3] @ reference_pxr(Q_ZERO)["gripper_end"][:3, :3].T)
        ),
        "why": "the authored gripper_end frame is (0,0,0.16621) on link6, which lies ON the joint6 axis",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
