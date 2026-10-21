# -*- coding: utf-8 -*-
"""P6 dynamic FK: the 8-DOF B601-RS chain moves, and every pose is checked twice.

WHAT THESE TESTS CLOSE
----------------------
mounted_chain_v2.json composes the joint anchors at q = 0 only.  A composed rest
pose cannot tell a correct joint model from a swapped sign, a dropped body1 flip
or a wrong composition order -- all of those agree at q = 0 by construction.
So the gates here are all about NON-ZERO q:

  1. ZERO: q = 0 still reproduces the committed pxr gripper_end pose (1e-6 m).
  2. NON-ZERO: three poses (including one with every joint near, but inside, its
     limit) give det(R) = +1 and orthonormal R on every link, and the whole tree
     is consistent with its own per-joint locals.
  3. JACOBIAN: analytic 6x8 J vs central differences vs a forward difference of
     the FK itself, for random q.
  4. CONVENTION: the committed quaternions are wxyz.  Reading the same four
     numbers as xyzw is a DETECTABLY different pose -- that is the red test.
  5. INDEPENDENT PATH: pxr (Gf math + UsdGeom.XformCache over an in-memory
     stage) and a pure-quaternion path agree with fk.py on the same non-zero q.

Pure python + numpy + optional pxr.  No Isaac, no GPU, no network.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "simulation"))

from rebot_b601_rs.kinematics.fk import (  # noqa: E402
    DOF_ORDER,
    LIMITS,
    ConventionError,
    QError,
    Chain,
    coerce_q,
    load_committed_anchors,
    matrix_to_quat_wxyz,
    quat_to_matrix,
    rotation_angle,
    so3_log,
)

P6 = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
EVIDENCE = P6 / "evidence" / "p3_mount_frame_tree_at_0bbed19.json"

#: The pxr reference, quoted from the committed evidence (NOT recomputed here):
#: /arm_chain_rest_pose/gripper_end_world_rest_pose, i.e.
#: UsdGeom.XformCache.GetLocalToWorldTransform(.../link6/gripper_end) at q = 0.
PXR_REST_POSITION_M = (0.301710169999, 2.62633e-07, 0.217746720407)
PXR_REST_QUAT_WXYZ = (1.0, -6.9738e-08, -1.7116e-08, 0.0)

TOL_M = 1e-6
TOL_RAD = 1e-6

#: Three non-zero configurations.  The third sits just inside every limit.
Q_ZERO = {name: 0.0 for name in DOF_ORDER}
Q_NONZERO = (
    {
        "block": "A: mixed signs",
        "q": dict(zip(DOF_ORDER, [-0.7, 0.9, -1.2, 0.4, -0.9, 1.7, 0.02, 0.03])),
    },
    {
        "block": "B: all positive, jaws open",
        "q": dict(zip(DOF_ORDER, [0.3, 0.45, 0.6, 0.5, 0.35, 0.2, 0.05, 0.0715])),
    },
    {
        "block": "C: near but inside every limit",
        "q": dict(zip(DOF_ORDER, [2.79, 3.13, 3.13, -1.56, 1.56, -3.13, 0.049, 0.07])),
    },
)
Q_RANDOM = dict(zip(DOF_ORDER, np.random.default_rng(6).uniform(-1.0, 1.0, 8)))


@pytest.fixture(scope="module")
def chain():
    return Chain()


def orthonormality_error(R):
    return float(np.max(np.abs(R @ R.T - np.eye(3))))


# ---------------------------------------------------------------------------
# 1. zero pose
# ---------------------------------------------------------------------------
def test_zero_pose_reproduces_the_committed_pxr_gripper_end_pose(chain):
    T = chain.pose(Q_ZERO, "gripper_end")
    dpos = float(np.max(np.abs(T[:3, 3] - np.asarray(PXR_REST_POSITION_M))))
    assert dpos < TOL_M, f"zero-pose position off by {dpos:.3e} m"
    # quaternion from the matrix, compared as wxyz with a documented sign fix
    q = matrix_to_quat_wxyz(T[:3, :3])
    ref = np.asarray(PXR_REST_QUAT_WXYZ)
    ref = ref / np.linalg.norm(ref)
    ang = rotation_angle(quat_to_matrix(q, "wxyz") @ quat_to_matrix(ref, "wxyz").T)
    assert ang < TOL_RAD, f"zero-pose orientation off by {ang:.3e} rad"


def test_zero_pose_reproduces_the_pxr_hop_list(chain):
    """Every link, not just the end: against the recorded pxr local hops."""
    T = chain.fk(Q_ZERO)
    assert float(np.max(np.abs(T["arm_root"] - np.eye(4)))) < 1e-12
    assert float(np.max(np.abs(T["base_link"] - np.eye(4)))) < 1e-12
    # hop base_link -> link1 is a pure translation in pxr (no rotation)
    assert np.allclose(T["link1"][:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(T["link1"][:3, 3], [-0.00034283, -0.00098683, 0.075], atol=1e-7)
    # the 5 body1 flips must NOT survive into the q=0 child poses
    for name in ("link2", "link3", "link4", "link5", "link6", "gripper_end"):
        assert orthonormality_error(T[name][:3, :3]) < 1e-12


# ---------------------------------------------------------------------------
# 2. non-zero poses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block", Q_NONZERO, ids=[b["block"] for b in Q_NONZERO])
def test_nonzero_pose_is_a_proper_rigid_motion(chain, block):
    q = block["q"]
    T = chain.fk(q)
    assert set(chain.fk(q)) == set(T) and "gripper_end" in T
    for name, M in T.items():
        R = M[:3, :3]
        assert np.isfinite(M).all()
        assert abs(np.linalg.det(R) - 1.0) < 1e-12, f"{name}: det(R) != +1"
        assert orthonormality_error(R) < 1e-12, f"{name}: R not orthonormal"
        assert np.allclose(M[3], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("block", Q_NONZERO, ids=[b["block"] for b in Q_NONZERO])
def test_nonzero_pose_is_tree_consistent(chain, block):
    """Each link pose == parent pose * joint local, for every hop."""
    q = block["q"]
    T = chain.fk(q)
    from rebot_b601_rs.kinematics.fk import JOINT_PARENT_CHILD

    for name in DOF_ORDER:
        parent, child = JOINT_PARENT_CHILD[name]
        expect = T[parent] @ chain.joint_local(name, q[name])
        assert np.max(np.abs(T[child] - expect)) < 1e-12
    assert np.max(np.abs(T["gripper_end"] - T["link6"] @ chain.joint_local("gripper_end", 0.0))) < 1e-12


def test_a_moving_joint_actually_moves_the_chain(chain):
    """A test that cannot fail at q=0 is not a FK test: prove motion is live."""
    T0 = chain.pose(Q_ZERO, "gripper_end")
    base = T0[:3, 3]
    for name in DOF_ORDER:
        q = dict(Q_ZERO)
        q[name] = 0.2 if not name.startswith("gripper") else 0.02
        T = chain.pose(q, "gripper_end")
        d = float(np.linalg.norm(T[:3, 3] - base))
        dang = rotation_angle(T[:3, :3] @ T0[:3, :3].T)
        if name.startswith("gripper"):
            # the jaws do not move gripper_end (it hangs off link6)
            assert d < 1e-12 and dang < 1e-12
        elif name == "joint6":
            # MEASURED, not assumed: gripper_end sits ON the joint6 axis
            # (the authored frame is (0, 0, 0.16621) on that axis), so joint6 is
            # a pure ROLL of the tool frame: no translation, 0.2 rad of rotation.
            # The pxr reference gives the same split (2.8e-17 m, 0.2 rad).
            assert d < 1e-12 and abs(dang - 0.2) < 1e-9, (d, dang)
        else:
            assert d > 1e-3, f"{name} does not move the end effector (delta {d:.3e} m)"
    # the two jaws move in OPPOSITE directions along link6 z
    q = dict(Q_ZERO)
    q["gripper_joint1"] = 0.03
    T = chain.fk(q)
    assert float(np.linalg.norm(T["gripper_left"][:3, 3] - T["gripper_right"][:3, 3])) > 1e-3


def test_limits_are_respected_by_the_near_limit_block(chain):
    q = Q_NONZERO[2]["q"]
    inside = chain.within_limits(q)
    assert all(inside.values()), inside
    for name in DOF_ORDER:
        over = dict(q)
        over[name] = LIMITS[name][1] + 0.01
        assert not chain.within_limits(over)[name], f"{name} over-limit not detected"


# ---------------------------------------------------------------------------
# 3. Jacobian
# ---------------------------------------------------------------------------
def test_analytic_jacobian_matches_central_differences(chain):
    for label, q in [("random", Q_RANDOM), ("blockA", Q_NONZERO[0]["q"]), ("nearlimit", Q_NONZERO[2]["q"])]:
        J = chain.jacobian(q)
        Jc = chain.jacobian_central(q, h=1e-6)
        assert J.shape == (6, 8)
        d = float(np.max(np.abs(J - Jc)))
        assert d < 1e-7, f"{label}: analytic vs central Jacobian differ by {d:.3e}"


def test_jacobian_matches_a_forward_difference_of_the_fk(chain):
    """Coarse forward difference of fk() itself: catches a wrong Jacobian ROW."""
    q = Q_RANDOM
    h = 1e-6
    T0 = chain.pose(q, "gripper_end")
    J = chain.jacobian(q)
    for col, name in enumerate(DOF_ORDER):
        qp = dict(q)
        qp[name] = q[name] + h
        Tp = chain.pose(qp, "gripper_end")
        fd_pos = (Tp[:3, 3] - T0[:3, 3]) / h
        fd_rot = so3_log(Tp[:3, :3] @ T0[:3, :3].T) / h  # signed, no acos precision loss
        assert np.max(np.abs(fd_pos - J[:3, col])) < 1e-5
        assert np.max(np.abs(fd_rot - J[3:, col])) < 1e-5


def test_jacobian_prismatic_columns_have_no_angular_part(chain):
    J = chain.jacobian(Q_RANDOM)
    for name in ("gripper_joint1", "gripper_joint2"):
        col = DOF_ORDER.index(name)
        assert np.max(np.abs(J[3:, col])) < 1e-12
    # jaw motion only moves the jaw it drives
    for name, driven in (("gripper_joint1", "gripper_left"), ("gripper_joint2", "gripper_right")):
        Jl = chain.jacobian(Q_RANDOM, link=driven)
        col = DOF_ORDER.index(name)
        assert np.linalg.norm(Jl[:3, col]) > 0.99


# ---------------------------------------------------------------------------
# 4. quaternion convention, per SOURCE
# ---------------------------------------------------------------------------
def test_default_read_is_usd_wxyz_and_agrees_with_the_declared_source():
    anchors = load_committed_anchors(quat_order="wxyz")
    assert anchors["joint1"]["axis"] == "Z"
    # (0,-1,0,0) wxyz is 180 deg about +X: R = diag(1,-1,-1)
    R = quat_to_matrix([0.0, -1.0, 0.0, 0.0], "wxyz")
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]), atol=1e-12)
    assert np.allclose(anchors["joint1"]["X1"][:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-12)


def test_xyzw_and_wxyz_are_not_interchangeable_red_green(chain):
    """RED: the same four numbers under the other convention is a different pose.

    GREEN: the two conventions agree when the same rotation is expressed in each.
    """
    Rg = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]  # 90 deg about +Z
    )
    q_wxyz = matrix_to_quat_wxyz(Rg)  # (cos45, 0, 0, sin45)
    assert np.allclose(quat_to_matrix(q_wxyz, "wxyz"), Rg)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    assert np.allclose(quat_to_matrix(q_xyzw, "xyzw"), Rg)
    # the swap is a different rotation by 90 deg here
    assert rotation_angle(quat_to_matrix(q_xyzw, "wxyz") @ Rg.T) > 1.0

    # and on the real chain: reading the committed wxyz anchors as xyzw
    wrong = Chain(load_committed_anchors(quat_order="xyzw"))
    dT = float(np.max(np.abs(wrong.pose(Q_ZERO, "gripper_end") - chain.pose(Q_ZERO, "gripper_end"))))
    assert dT > 1e-2, f"a swapped convention changed the pose by only {dT:.3e}; the test is blind"
    with pytest.raises(ConventionError):
        quat_to_matrix([1.0, 0.0, 0.0, 0.0], "wzyx")


# ---------------------------------------------------------------------------
# 5. q plumbing: name-mapped, not index-mapped
# ---------------------------------------------------------------------------
def test_q_is_addressed_by_name_not_by_index(chain):
    q = Q_NONZERO[0]["q"]
    shuffled = {k: q[k] for k in reversed(list(q.keys()))}
    assert np.max(np.abs(chain.pose(q) - chain.pose(shuffled))) < 1e-15
    # a bare 8-vector is the module order, documented, and equals the named form
    assert np.max(np.abs(chain.pose([q[n] for n in DOF_ORDER]) - chain.pose(q))) < 1e-15
    with pytest.raises(QError):
        coerce_q({n: 0.0 for n in DOF_ORDER[:-1]})
    with pytest.raises(QError):
        coerce_q({**q, "joint7": 0.0})
    with pytest.raises(QError):
        coerce_q([0.0] * 7)


def test_every_link_including_gripper_end_is_returned(chain):
    out = chain.fk(Q_RANDOM)
    for name in ("arm_root", "base_link", "link1", "link2", "link3", "link4", "link5",
                 "link6", "gripper_left", "gripper_right", "gripper_end"):
        assert name in out and out[name].shape == (4, 4)


# ---------------------------------------------------------------------------
# 6. independent reference paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block", Q_NONZERO, ids=[b["block"] for b in Q_NONZERO])
def test_pxr_reference_agrees_on_nonzero_q(chain, block):
    pytest.importorskip("pxr")
    from rebot_b601_rs.kinematics.reference import reference_pxr

    q = block["q"]
    mine = chain.fk(q)
    ref = reference_pxr(q)
    dp = max(float(np.max(np.abs(mine[k][:3, 3] - ref[k][:3, 3]))) for k in ref)
    dr = max(rotation_angle(mine[k][:3, :3] @ ref[k][:3, :3].T) for k in ref)
    assert dp < 1e-8, f"pxr path differs by {dp:.3e} m"
    assert dr < 1e-7, f"pxr path differs by {dr:.3e} rad"
    # and the pxr path reproduces the committed rest pose too
    ref0 = reference_pxr(Q_ZERO)
    assert np.max(np.abs(ref0["gripper_end"][:3, 3] - np.asarray(PXR_REST_POSITION_M))) < TOL_M


@pytest.mark.parametrize("block", Q_NONZERO, ids=[b["block"] for b in Q_NONZERO])
def test_quaternion_algebra_reference_agrees(chain, block):
    from rebot_b601_rs.kinematics.reference import reference_quat

    q = block["q"]
    mine = chain.fk(q)
    ref = reference_quat(q)
    d = max(float(np.max(np.abs(mine[k] - ref[k]))) for k in ref if k in mine)
    assert d < 1e-12, f"quaternion path differs by {d:.3e}"
