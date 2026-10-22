# -*- coding: utf-8 -*-
"""P7-11: TWO kinematics backends behind ONE interface, and never a silent mix.

WHY THIS FILE EXISTS
--------------------
P7-09 built the reachability/IK *interface* on an ENGINEERING_V0_1 PROXY DH table
(src/badminton_brain/reachability/kinematics.py).  P6 then produced a VERIFIED
analytic FK for the real 8-DOF B601-RS chain
(src/simulation/rebot_b601_rs/kinematics/fk.py on branch p6/dynamic-fk, PR #40:
T = X0 * M(q) * inv(X1), zero pose matches pxr to 1.85e-9 m).  The two live on
different branches, so this module does NOT vendor a copy of the FK: it loads the
real module at import time and reports WHICH backend answered every question.

    KinematicsBackend            pose(q) / jacobian(q) / limits / provenance
      ProxyDhBackend             6-DOF proxy DH (P7-09), frame = racket face centre
      VerifiedFkBackend          8-DOF verified P6 FK, frame = gripper_end

SELECTION IS EXPLICIT AND FAILS LOUDLY
--------------------------------------
select_backend("verified") raises VerifiedFkUnavailable when the P6 module cannot
be found.  It NEVER falls back to the proxy: a fallback would make an unverified
number look verified, which is the whole failure mode this task is about.  The
same holds inside the harness: a run carries exactly one backend id, every verdict
records it, and a run that sees two different ids raises BackendMixError.

WHAT "VERIFIED" MEANS HERE, AND WHAT IT DOES NOT
------------------------------------------------
VerifiedFkBackend answers for the point P6 actually verified: the authored
gripper_end frame on link6.  The RACKET FACE and the racket MOUNT are still
UNKNOWN in the P6 frame contract (edges 13/14/15 of frame_tree.json), so a
"verified racket face" does not exist.  The backend therefore reports
frame_name="gripper_end" and target_kind="ARM_END_FRAME", and the reachability
verdict is about THAT point - not about a racket face.  See
management/morph_one_rs/P7_CLOSED_LOOP_WIRING.md.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ik import (DEFAULT_TOL_POSITION_M, DEFAULT_TOL_ROTATION_RAD, IkError,
                 IkNotConverged, IkNotImplemented, IkResult, IkSolver, IkTarget,
                 InvalidIkTarget, rotation_error_vector)
from .kinematics import (DOF, JOINT_LOWER_RAD, JOINT_NAMES, JOINT_UPPER_RAD,
                         PROXY_SOURCE, forward_kinematics_face, geometric_jacobian)
from .reachability import (KinematicLimits, P5_VELOCITY_CAP_RAD_S, ReachabilityDecision,
                           ReachabilityReason, ReachabilityVerdict, assess_reachability)

__all__ = [
    "BACKEND_PROXY", "BACKEND_VERIFIED", "BACKEND_CHOICES",
    "P6_FK_RELPATH", "P6_PROXY_RELPATH", "P6_ROOT_ENV", "P6_FK_PATH_ENV",
    "BackendError", "BackendUnavailable", "VerifiedFkUnavailable", "BackendMixError",
    "KinematicsBackend", "ProxyDhBackend", "VerifiedFkBackend",
    "BackendDampedLeastSquaresIk", "select_backend", "kinematic_limits_for",
    "resolve_p6_root", "sha256_file",
]

BACKEND_PROXY = "proxy_dh"
BACKEND_VERIFIED = "verified_p6_fk"
BACKEND_CHOICES: Tuple[str, ...] = (BACKEND_PROXY, BACKEND_VERIFIED)

#: Where the P6 artifacts live, relative to a P6 checkout root.
P6_FK_RELPATH = "src/simulation/rebot_b601_rs/kinematics/fk.py"
P6_PROXY_RELPATH = "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/racket_collision_proxy.json"

#: Environment overrides.  There is NO implicit fallback: if the variable is set
#: and the path is wrong, that is an error, not a hint.
P6_ROOT_ENV = "BADMIN_P6_ROOT"
P6_FK_PATH_ENV = "BADMIN_P6_FK_PATH"


class BackendError(RuntimeError):
    """Base class for every backend refusal."""


class BackendUnavailable(BackendError):
    """The requested backend cannot be constructed; nothing will be substituted."""


class VerifiedFkUnavailable(BackendUnavailable):
    """The verified P6 FK module was expected but is not importable."""


class BackendMixError(BackendError):
    """Two different backends answered inside one run/report - forbidden."""


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    # src/badminton_brain/reachability/backends.py -> parents[3] == repository root
    return Path(__file__).resolve().parents[3]


def resolve_p6_root(explicit_root: Optional[os.PathLike | str] = None,
                    extra_roots: Sequence[os.PathLike | str] = ()) -> Optional[Path]:
    """First root that actually carries the P6 FK module, or None.

    Order: explicit_root -> $BADMIN_P6_ROOT -> extra_roots -> this repository.
    A candidate only counts if <root>/src/.../kinematics/fk.py exists, so a stale
    or empty directory cannot shadow the real checkout.
    """
    candidates: List[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))
    env_root = os.environ.get(P6_ROOT_ENV)
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(Path(root) for root in extra_roots)
    candidates.append(_repo_root())
    for root in candidates:
        if (root / P6_FK_RELPATH).is_file():
            return root.resolve()
    return None


def _load_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise VerifiedFkUnavailable(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KinematicsBackend:
    """The interface both backends satisfy.  Not a runtime ABC on purpose."""

    backend_id = "abstract"
    source_class = "ABSTRACT"
    frame_name = "unknown"
    target_kind = "UNKNOWN"
    dof = 0
    joint_names: Tuple[str, ...] = ()
    lower_rad: Tuple[float, ...] = ()
    upper_rad: Tuple[float, ...] = ()
    limit_source = ""
    detail = ""

    def pose(self, q: Sequence[float]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def position(self, q: Sequence[float]) -> np.ndarray:
        return np.asarray(self.pose(q), dtype=float)[:3, 3].copy()

    def jacobian(self, q: Sequence[float]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _check_q(self, q: Sequence[float]) -> np.ndarray:
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape != (self.dof,):
            raise InvalidIkTarget(f"{self.backend_id}: q must have {self.dof} entries, "
                                  f"got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise InvalidIkTarget(f"{self.backend_id}: q contains NaN/Inf")
        return arr

    def provenance(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_id,
            "source_class": self.source_class,
            "frame_name": self.frame_name,
            "target_kind": self.target_kind,
            "dof": int(self.dof),
            "joint_names": list(self.joint_names),
            "limit_source": self.limit_source,
            "detail": self.detail,
        }


class ProxyDhBackend(KinematicsBackend):
    """The P7-09 ENGINEERING_V0_1 proxy DH: 6 DOF, frame = racket FACE centre."""

    backend_id = BACKEND_PROXY
    source_class = "PROXY_DH_ENGINEERING_V0_1"
    frame_name = "racket_face_proxy"
    target_kind = "RACKET_FACE_PROXY"
    dof = DOF
    joint_names = JOINT_NAMES
    lower_rad = JOINT_LOWER_RAD
    upper_rad = JOINT_UPPER_RAD
    limit_source = PROXY_SOURCE

    def __init__(self) -> None:
        self.detail = "P7-09 proxy DH table, 6 revolute joints, tool = face centre 0.50 m along +z of link6"

    def pose(self, q: Sequence[float]) -> np.ndarray:
        return forward_kinematics_face(self._check_q(q))

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        return geometric_jacobian(self._check_q(q))


class VerifiedFkBackend(KinematicsBackend):
    """The P6 VERIFIED 8-DOF B601-RS FK, loaded from the real module - never copied.

    The module is located by resolve_p6_root() / $BADMIN_P6_FK_PATH and executed
    from its own file, so the committed anchors and the committed FK code are the
    ones that answer.  If it is not there, the constructor raises
    VerifiedFkUnavailable and NO proxy number is produced in its place.
    """

    backend_id = BACKEND_VERIFIED
    source_class = "VERIFIED_ASSET_KINEMATICS"
    frame_name = "gripper_end"
    target_kind = "ARM_END_FRAME"

    def __init__(self, *, root: Optional[os.PathLike | str] = None,
                 fk_path: Optional[os.PathLike | str] = None,
                 extra_roots: Sequence[os.PathLike | str] = ()) -> None:
        path = Path(fk_path) if fk_path is not None else None
        if path is None:
            env_path = os.environ.get(P6_FK_PATH_ENV)
            if env_path:
                path = Path(env_path)
        if path is None:
            root_dir = resolve_p6_root(root, extra_roots)
            if root_dir is None:
                raise VerifiedFkUnavailable(
                    "VERIFIED_FK_UNAVAILABLE: the P6 verified FK module was expected but no "
                    f"checkout was found.  Looked for {P6_FK_RELPATH!r} under "
                    f"$%s, the explicit root, and this repository.  The proxy DH is NOT "
                    "substituted: run with a P6 checkout (git worktree of p6/dynamic-fk) or "
                    "ask for the proxy backend explicitly." % P6_ROOT_ENV)
            path = root_dir / P6_FK_RELPATH
        if not path.is_file():
            raise VerifiedFkUnavailable(
                f"VERIFIED_FK_UNAVAILABLE: {path} does not exist; refusing to substitute "
                "the proxy DH behind a 'verified' request")
        try:
            module = _load_module_from_path(path, "badminton_p6_verified_fk")
        except Exception as exc:  # noqa: BLE001 - any import failure is loud
            raise VerifiedFkUnavailable(
                f"VERIFIED_FK_UNAVAILABLE: importing {path} failed: "
                f"{type(exc).__name__}: {exc}") from exc
        for attribute in ("Chain", "DOF_ORDER", "LIMITS"):
            if not hasattr(module, attribute):
                raise VerifiedFkUnavailable(
                    f"VERIFIED_FK_UNAVAILABLE: {path} has no {attribute!r}; it is not the "
                    "P6 FK module this adapter was written against")
        self.module = module
        self.fk_path = path
        self.fk_sha256 = sha256_file(path)
        self.chain = module.Chain()
        self.dof = len(module.DOF_ORDER)
        self.joint_names = tuple(module.DOF_ORDER)
        limits = module.LIMITS
        self.lower_rad = tuple(float(limits[name][0]) for name in self.joint_names)
        self.upper_rad = tuple(float(limits[name][1]) for name in self.joint_names)
        self.limit_source = (
            "P6 joint_anchors.json / evidence robot_usd_joint_chain.json "
            "limits_*_as_authored (read from the loaded module's LIMITS table)")
        self.detail = (
            "P6 verified FK T = X0*M(q)*inv(X1) on the 8-DOF B601-RS chain (6 revolute + 2 "
            "prismatic jaws); pose and jacobian for the authored gripper_end frame; the "
            "racket FACE frame is UNKNOWN (P6 frame_tree edges 13/14/15)")

    def _check_q(self, q: Sequence[float]) -> np.ndarray:
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.shape != (self.dof,):
            raise InvalidIkTarget(f"{self.backend_id}: q must have {self.dof} entries, "
                                  f"got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise InvalidIkTarget(f"{self.backend_id}: q contains NaN/Inf")
        return arr

    def pose(self, q: Sequence[float]) -> np.ndarray:
        return np.asarray(self.chain.pose(self._check_q(q), link=self.frame_name), dtype=float)

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        return np.asarray(self.chain.jacobian(self._check_q(q), link=self.frame_name),
                          dtype=float)

    def provenance(self) -> Dict[str, Any]:
        out = super().provenance()
        out.update({
            "fk_module_path": str(self.fk_path),
            "fk_module_sha256": self.fk_sha256,
            "note": "the racket face / racket mount frame is NOT verified in P6; this backend "
                    "answers for gripper_end only",
        })
        return out


class BackendDampedLeastSquaresIk(IkSolver):
    """Damped least squares on ANY KinematicsBackend (proxy 6-DOF or verified 8-DOF).

    The maths is the documented P7-09 DLS loop, but the FK and the Jacobian come
    from the backend, so the residual reported here is measured with the SAME
    kinematics that produced the solution - a converged result on the verified
    backend is a verified-chain residual, not a proxy one.
    """

    def __init__(self, backend: KinematicsBackend, *, max_iterations: int = 200,
                 tolerance_position_m: float = DEFAULT_TOL_POSITION_M,
                 tolerance_rotation_rad: float = DEFAULT_TOL_ROTATION_RAD,
                 damping: float = 1e-3, max_step_rad: float = 0.20,
                 rotation_weight: float = 0.25, step_floor_rad: float = 1e-10) -> None:
        if not isinstance(backend, KinematicsBackend):
            raise TypeError("backend must be a KinematicsBackend")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        self.backend = backend
        self.solver_id = f"dls/{backend.backend_id}/v1"
        self.max_iterations = int(max_iterations)
        self.tolerance_position_m = float(tolerance_position_m)
        self.tolerance_rotation_rad = float(tolerance_rotation_rad)
        self.damping = float(damping)
        self.max_step_rad = float(max_step_rad)
        self.rotation_weight = float(rotation_weight)
        self.step_floor_rad = float(step_floor_rad)

    def _residuals(self, q: np.ndarray, target: IkTarget) -> Tuple[np.ndarray, float, float]:
        T = np.asarray(self.backend.pose(q), dtype=float)
        p_err = np.asarray(target.position_m, dtype=float) - T[:3, 3]
        r_err = rotation_error_vector(T[:3, :3], np.asarray(target.rotation, dtype=float))
        err = np.concatenate([p_err, self.rotation_weight * r_err])
        return err, float(np.linalg.norm(p_err)), float(np.linalg.norm(r_err))

    def solve(self, target: IkTarget, seed_rad: Sequence[float]) -> IkResult:
        if not isinstance(target, IkTarget):
            raise InvalidIkTarget("target must be an IkTarget")
        seed = np.asarray(seed_rad, dtype=float).reshape(-1)
        if seed.shape != (self.backend.dof,):
            raise InvalidIkTarget(f"seed must have {self.backend.dof} entries (backend "
                                  f"{self.backend.backend_id}), got {seed.shape}")
        if not np.all(np.isfinite(seed)):
            raise InvalidIkTarget("seed contains NaN/Inf")
        lower = np.asarray(self.backend.lower_rad, dtype=float)
        upper = np.asarray(self.backend.upper_rad, dtype=float)
        q = np.clip(seed, lower, upper)
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
            jac = np.asarray(self.backend.jacobian(q), dtype=float)
            jac_w = jac.copy()
            jac_w[3:, :] *= self.rotation_weight
            JJt = jac_w @ jac_w.T + (self.damping ** 2) * np.eye(6)
            dq = jac_w.T @ np.linalg.solve(JJt, err)
            peak = float(np.max(np.abs(dq)))
            if peak > self.max_step_rad:
                dq *= self.max_step_rad / peak
                peak = self.max_step_rad
            if peak < self.step_floor_rad:
                break
            q = np.clip(q + dq, lower, upper)
        _, residual_p, residual_r = self._residuals(q, target)
        raise IkNotConverged(residual_p, residual_r, iteration,
                             self.tolerance_position_m, self.tolerance_rotation_rad)


def kinematic_limits_for(backend: KinematicsBackend, *,
                         joint_margin_rad: Optional[float] = None,
                         velocity_cap_rad_s: Optional[Sequence[float]] = None,
                         **overrides: Any) -> KinematicLimits:
    """KinematicLimits sized to the backend's DOF, with its own joint stops.

    The workspace box and the sigma threshold keep the P7-09 defaults so a
    proxy-vs-verified comparison changes ONLY the kinematics and the stops.
    Joint margin: the proxy keeps the 0.05 rad default; the verified chain carries
    two PRISMATIC jaws whose stroke is 0.05 m / 0.0715 m, where a 0.05 "rad" margin
    would be meaningless, so the caller passes a declared value (0.01 by default
    for mixed units) and it is reported.
    """
    dof = int(backend.dof)
    if velocity_cap_rad_s is None:
        caps = []
        for name in backend.joint_names:
            caps.append(0.05 if name.startswith("gripper") else P5_VELOCITY_CAP_RAD_S)
    else:
        caps = [float(v) for v in velocity_cap_rad_s]
    if len(caps) != dof:
        raise ValueError(f"velocity_cap_rad_s must have {dof} entries, got {len(caps)}")
    margin = (0.05 if backend.backend_id == BACKEND_PROXY else 0.01) \
        if joint_margin_rad is None else float(joint_margin_rad)
    return KinematicLimits(
        lower_rad=tuple(float(v) for v in backend.lower_rad),
        upper_rad=tuple(float(v) for v in backend.upper_rad),
        joint_margin_rad=margin,
        velocity_cap_rad_s=tuple(caps),
        dof=dof,
        joint_names=tuple(backend.joint_names),
        **overrides,
    )


def select_backend(name: str, *, root: Optional[os.PathLike | str] = None,
                   fk_path: Optional[os.PathLike | str] = None,
                   extra_roots: Sequence[os.PathLike | str] = ()) -> KinematicsBackend:
    """Build the backend the caller ASKED for, or raise.  Never substitutes."""
    key = str(name).strip()
    if key == BACKEND_PROXY:
        return ProxyDhBackend()
    if key == BACKEND_VERIFIED:
        return VerifiedFkBackend(root=root, fk_path=fk_path, extra_roots=extra_roots)
    raise BackendUnavailable(f"unknown backend {name!r}; choose one of {BACKEND_CHOICES}")
