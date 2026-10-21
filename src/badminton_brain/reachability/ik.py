# -*- coding: utf-8 -*-
"""P7-09: the IK INTERFACE, and one real solver behind it.

THE CONTRACT (this is the part that matters)
--------------------------------------------
An IK solver in this repository may do exactly two things when asked for a joint
vector:

  1. return an IKResult that says converged and reports the RESIDUAL it achieved,
     measured with the same FK the solver used; or
  2. raise IkNotImplemented, saying it cannot solve at all.

It may NOT return an approximate pose dressed up as a solution.  A planner that
receives a 3 cm pose error without being told is a planner that swings a racket
3 cm off the shuttle and blames perception.  So:
  * a solver that fails to reach the tolerance raises IkNotConverged (never a
    silent best-effort), and the caller sees the residual in the exception;
  * a placeholder solver raises IkNotImplemented;
  * every IKResult carries solver id, iterations, iteration budget and tolerances,
    so the result can be audited without knowing which solver ran.

DampedLeastSquaresIk
--------------------
Minimum-norm damped least squares on the numerical geometric Jacobian of
kinematics.py:
    e      = [ p_goal - p(q) ; w_rot * rot_error(R_goal, R(q)) ]   (6,)
    dq     = J^T (J J^T + lambda^2 I)^-1 e,   J the 6x6 geometric Jacobian
    q      <- clamp(q + clip(dq), joint limits)
Convergence (both must hold):
    ||p_goal - p(q)||  <= tol_position_m        (default 1e-4 m)
    ||rot_error||      <= tol_rotation_rad      (default 1e-3 rad)
Stall (max |dq| below the step floor for a whole iteration) also raises.

ASSUMPTIONS, ALL UNVERIFIED ON HARDWARE
  * the DH table is the ENGINEERING_V0_1 PROXY of kinematics.py, not the unit;
  * the Jacobian is central-difference numeric, step fixed at 1e-6 rad;
  * joint limits are the URDF-declared PiPER ranges, not measured;
  * no self-collision, no torque limit, no velocity limit is modelled here
    (reachability.py adds the margin and collision proxy checks).
Therefore convergence proves the SOLVER, not the robot.  See the UNKNOWNs
section of management/morph_one_rs/P7_REACHABILITY_AND_FSM.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .kinematics import (DOF, JOINT_LOWER_RAD, JOINT_UPPER_RAD, face_position,
                         forward_kinematics_face, geometric_jacobian)

__all__ = [
    "IK_SOLVER_ID", "DEFAULT_TOL_POSITION_M", "DEFAULT_TOL_ROTATION_RAD",
    "IkError", "IkNotImplemented", "IkNotConverged", "InvalidIkTarget",
    "IkTarget", "IkResult", "IkSolver", "UnavailableIk", "DampedLeastSquaresIk",
    "rotation_error_vector",
]

IK_SOLVER_ID = "damped_least_squares_numeric_jacobian/v1"
DEFAULT_TOL_POSITION_M = 1e-4
DEFAULT_TOL_ROTATION_RAD = 1e-3


class IkError(RuntimeError):
    """Base class for every IK refusal.  Never raised in place of a converged result."""


class IkNotImplemented(IkError):
    """This solver cannot produce joint values at all (interface placeholder)."""

    def __init__(self, solver: str, detail: str) -> None:
        self.solver = solver
        self.detail = detail
        super().__init__(f"IK_NOT_IMPLEMENTED[{solver}]: {detail}")


class IkNotConverged(IkError):
    """The solver ran and did not reach its documented tolerance."""

    def __init__(self, residual_position_m: float, residual_rotation_rad: float,
                 iterations: int, tol_position_m: float, tol_rotation_rad: float) -> None:
        self.residual_position_m = float(residual_position_m)
        self.residual_rotation_rad = float(residual_rotation_rad)
        self.iterations = int(iterations)
        self.tol_position_m = float(tol_position_m)
        self.tol_rotation_rad = float(tol_rotation_rad)
        super().__init__(
            "IK_NOT_CONVERGED: residual %.6f m (tol %.6f) / %.6f rad (tol %.6f) after %d "
            "iterations" % (self.residual_position_m, self.tol_position_m,
                            self.residual_rotation_rad, self.tol_rotation_rad,
                            self.iterations))


class InvalidIkTarget(IkError):
    """The target pose itself is unusable (shape, NaN, non-orthonormal rotation)."""


def _as_matrix3(value: object, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise InvalidIkTarget(f"{what} must be (3,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise InvalidIkTarget(f"{what} contains NaN/Inf")
    if not np.allclose(arr @ arr.T, np.eye(3), atol=1e-6):
        raise InvalidIkTarget(f"{what} is not orthonormal")
    return arr


@dataclass(frozen=True)
class IkTarget:
    """A racket-face pose target in the ARM BASE frame."""

    position_m: Sequence[float]
    rotation: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        pos = np.asarray(self.position_m, dtype=float).reshape(-1)
        if pos.shape != (3,):
            raise InvalidIkTarget(f"position_m must be (3,), got {pos.shape}")
        if not np.all(np.isfinite(pos)):
            raise InvalidIkTarget("position_m contains NaN/Inf")
        rot = _as_matrix3(self.rotation, "rotation")
        object.__setattr__(self, "position_m", pos)
        object.__setattr__(self, "rotation", rot)

    def as_matrix4(self) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(self.rotation)
        T[:3, 3] = np.asarray(self.position_m)
        return T


@dataclass(frozen=True)
class IkResult:
    """A CONVERGED solution plus the numbers needed to audit it."""

    joints_rad: Tuple[float, ...]
    residual_position_m: float
    residual_rotation_rad: float
    iterations: int
    tolerance_position_m: float
    tolerance_rotation_rad: float
    solver: str = IK_SOLVER_ID
    converged: bool = True

    def __post_init__(self) -> None:
        if not self.converged:
            # A non-converged result must never be constructible: that is the
            # whole point of the interface (raise instead of approximating).
            raise ValueError("IkResult may only hold a converged solution; raise "
                             "IkNotConverged instead of returning a best effort")

    def as_dict(self) -> dict:
        return {
            "solver": self.solver,
            "joints_rad": [float(v) for v in self.joints_rad],
            "residual_position_m": float(self.residual_position_m),
            "residual_rotation_rad": float(self.residual_rotation_rad),
            "iterations": int(self.iterations),
            "tolerance_position_m": float(self.tolerance_position_m),
            "tolerance_rotation_rad": float(self.tolerance_rotation_rad),
            "converged": bool(self.converged),
        }


def rotation_error_vector(R_current: np.ndarray, R_goal: np.ndarray) -> np.ndarray:
    """SO(3) log-ish error, space frame: 0.5 * vee(R_goal R_cur^T - R_cur R_goal^T).

    Magnitude equals sin(angle); it is small-angle equivalent to the axis-angle
    error, which is all the DLS loop and the tolerance need.
    """
    R = np.asarray(R_goal, dtype=float) @ np.asarray(R_current, dtype=float).T
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * skew


class IkSolver:
    """The interface every IK implementation must satisfy (not a runtime ABC)."""

    solver_id = "abstract"

    def solve(self, target: IkTarget, seed_rad: Sequence[float]) -> IkResult:  # pragma: no cover
        raise IkNotImplemented(self.solver_id, "no solver installed")


class UnavailableIk(IkSolver):
    """Interface placeholder: refuses honestly instead of returning a pose.

    This is the ONLY acceptable stand-in until the real URDF kinematics and a
    measured TCP exist.  It exists so the rest of the stack can be wired and
    tested: everything downstream must degrade to UNKNOWN / HOLD, never to an
    approximate swing.
    """

    solver_id = "unavailable/placeholder"

    def __init__(self, detail: str = "no IK solver is installed for this arm") -> None:
        self.detail = detail

    def solve(self, target: IkTarget, seed_rad: Sequence[float]) -> IkResult:
        raise IkNotImplemented(self.solver_id, self.detail)


class DampedLeastSquaresIk(IkSolver):
    """Levenberg-style damped least squares on the numeric geometric Jacobian."""

    solver_id = IK_SOLVER_ID

    def __init__(self, *, max_iterations: int = 200,
                 tolerance_position_m: float = DEFAULT_TOL_POSITION_M,
                 tolerance_rotation_rad: float = DEFAULT_TOL_ROTATION_RAD,
                 damping: float = 1e-3, max_step_rad: float = 0.20,
                 rotation_weight: float = 0.25, step_floor_rad: float = 1e-10) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        for name, value in (("tolerance_position_m", tolerance_position_m),
                            ("tolerance_rotation_rad", tolerance_rotation_rad),
                            ("damping", damping), ("max_step_rad", max_step_rad),
                            ("rotation_weight", rotation_weight)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        self.max_iterations = int(max_iterations)
        self.tolerance_position_m = float(tolerance_position_m)
        self.tolerance_rotation_rad = float(tolerance_rotation_rad)
        self.damping = float(damping)
        self.max_step_rad = float(max_step_rad)
        self.rotation_weight = float(rotation_weight)
        self.step_floor_rad = float(step_floor_rad)

    def _residuals(self, q: np.ndarray, target: IkTarget) -> Tuple[np.ndarray, float, float]:
        T = forward_kinematics_face(q)
        p_err = np.asarray(target.position_m) - T[:3, 3]
        r_err = rotation_error_vector(T[:3, :3], np.asarray(target.rotation))
        return np.concatenate([p_err, self.rotation_weight * r_err]), \
            float(np.linalg.norm(p_err)), float(np.linalg.norm(r_err))

    def solve(self, target: IkTarget, seed_rad: Sequence[float]) -> IkResult:
        if not isinstance(target, IkTarget):
            raise InvalidIkTarget("target must be an IkTarget")
        seed = np.asarray(seed_rad, dtype=float).reshape(-1)
        if seed.shape != (DOF,):
            raise InvalidIkTarget(f"seed must have {DOF} entries, got {seed.shape}")
        if not np.all(np.isfinite(seed)):
            raise InvalidIkTarget("seed contains NaN/Inf")

        lower = np.asarray(JOINT_LOWER_RAD, dtype=float)
        upper = np.asarray(JOINT_UPPER_RAD, dtype=float)
        q = np.clip(seed, lower, upper)
        residual_p, residual_r = float("inf"), float("inf")
        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            err, residual_p, residual_r = self._residuals(q, target)
            if (residual_p <= self.tolerance_position_m
                    and residual_r <= self.tolerance_rotation_rad):
                return IkResult(
                    joints_rad=tuple(float(v) for v in q),
                    residual_position_m=residual_p,
                    residual_rotation_rad=residual_r,
                    iterations=iteration,
                    tolerance_position_m=self.tolerance_position_m,
                    tolerance_rotation_rad=self.tolerance_rotation_rad,
                    solver=self.solver_id,
                )
            jac = geometric_jacobian(q)
            jac_w = jac.copy()
            jac_w[3:, :] *= self.rotation_weight
            JJt = jac_w @ jac_w.T + (self.damping ** 2) * np.eye(6)
            dq = jac_w.T @ np.linalg.solve(JJt, err)
            peak = float(np.max(np.abs(dq)))
            if peak > self.max_step_rad:
                dq *= self.max_step_rad / peak
                peak = self.max_step_rad
            if peak < self.step_floor_rad:
                break  # stalled at a singularity / against the limits
            q = np.clip(q + dq, lower, upper)
        _, residual_p, residual_r = self._residuals(q, target)
        raise IkNotConverged(residual_p, residual_r, iteration,
                             self.tolerance_position_m, self.tolerance_rotation_rad)
