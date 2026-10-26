# -*- coding: utf-8 -*-
"""Fail-closed B601-RS -> racket stringbed-centre proxy kinematics.

This module composes transforms; it does not calibrate them. Unknown mount/base
transforms remain UNKNOWN and are rejected. Proxy geometry is never labelled as
a measured physical sweet spot.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


KNOWN_STATUSES = frozenset({
    "MEASURED",
    "SIMULATION_ASSUMPTION",
    "SIMULATION_STATE",
    "PROXY_DERIVED",
    "UNKNOWN",
})
SIMULATION_STATUSES = frozenset({"SIMULATION_ASSUMPTION", "SIMULATION_STATE"})


class ContactKinematicsError(ValueError):
    """Base error for invalid or unsupported contact-pose requests."""


class CalibrationRequiredError(ContactKinematicsError):
    """A required transform is absent, UNKNOWN, or not allowed for this run."""


class JointLimitError(ContactKinematicsError):
    """The requested joint vector violates the FK chain's declared limits."""


def validate_se3(matrix: Sequence[Sequence[float]], name: str = "transform",
                 atol: float = 1e-6) -> np.ndarray:
    """Validate a column-vector SE(3) matrix without silently repairing it."""
    T = np.asarray(matrix, dtype=float)
    if T.shape != (4, 4):
        raise ContactKinematicsError("%s must be a 4x4 matrix, got %s" % (name, T.shape))
    if not np.isfinite(T).all():
        raise ContactKinematicsError("%s contains NaN or Inf" % name)
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ContactKinematicsError("%s has an invalid homogeneous bottom row" % name)
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol, rtol=0.0):
        raise ContactKinematicsError("%s rotation is not orthonormal" % name)
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > atol:
        raise ContactKinematicsError("%s rotation determinant must be +1, got %.9g" % (name, det))
    return T.copy()


@dataclass(frozen=True)
class TransformEvidence:
    """One named T_A_B transform plus the evidence class that permits its use.

    matrix is T_A_B: it maps points expressed in frame B into frame A.
    UNKNOWN must carry no matrix. Known values must carry a source string.
    """
    name: str
    matrix: Optional[np.ndarray]
    status: str
    source: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("transform name must be non-empty")
        if self.status not in KNOWN_STATUSES:
            raise ValueError("unsupported transform status %r" % self.status)
        if self.status == "UNKNOWN":
            if self.matrix is not None:
                raise ValueError("UNKNOWN transform %s must not carry numeric values" % self.name)
            return
        if self.matrix is None:
            raise ValueError("%s transform %s requires a matrix" % (self.status, self.name))
        if not self.source.strip():
            raise ValueError("%s transform %s requires provenance" % (self.status, self.name))
        object.__setattr__(self, "matrix", validate_se3(self.matrix, self.name))

    def resolve(self, *, allow_simulation_assumptions: bool = False,
                allow_proxy_geometry: bool = False) -> np.ndarray:
        if self.status == "UNKNOWN" or self.matrix is None:
            raise CalibrationRequiredError(
                "%s is UNKNOWN; refusing to invent an identity or guessed transform" % self.name)
        if self.status in SIMULATION_STATUSES and not allow_simulation_assumptions:
            raise CalibrationRequiredError(
                "%s is %s; simulation-only use must be explicitly enabled" %
                (self.name, self.status))
        if self.status == "PROXY_DERIVED" and not allow_proxy_geometry:
            raise CalibrationRequiredError(
                "%s is proxy-derived geometry, not a measured physical contact frame" % self.name)
        return self.matrix.copy()


@dataclass(frozen=True)
class RacketContactPose:
    """World pose result with evidence labels retained alongside the numbers."""
    matrix_world_contact: np.ndarray
    status: str
    sources: tuple[str, ...]
    input_statuses: tuple[str, ...]
    frame_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "matrix_world_contact",
            validate_se3(self.matrix_world_contact, "matrix_world_contact"))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "input_statuses", tuple(self.input_statuses))

    @property
    def position_m(self) -> tuple[float, float, float]:
        p = self.matrix_world_contact[:3, 3]
        return (float(p[0]), float(p[1]), float(p[2]))


def racket_stringbed_center_proxy(source: str | Path | Mapping[str, Any]) -> TransformEvidence:
    """Derive a *visual bounding-box centre proxy*, never a physical sweet spot.

    The input is the committed p6_racket_collision_proxy/1 JSON or its parsed
    mapping. The selected source mesh must explicitly be the visual-only string bed.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        doc = json.loads(path.read_text(encoding="utf-8"))
        source_label = str(path.as_posix())
    else:
        doc = source
        source_label = str(doc.get("generated_by", "provided proxy document"))

    if doc.get("schema") != "p6_racket_collision_proxy/1":
        raise ContactKinematicsError("unexpected racket proxy schema")
    root = doc.get("root_frame") or {}
    if root.get("name") != "racket_root":
        raise ContactKinematicsError("proxy root frame must be racket_root")
    if not str(root.get("mesh_local_to_world", "")).lower().startswith("identity"):
        raise ContactKinematicsError("proxy mesh is not declared in the racket_root frame")

    meshes = doc.get("source_meshes") or []
    candidates = [(i, m) for i, m in enumerate(meshes)
                  if "string bed" in str(m.get("role", "")).lower()]
    if len(candidates) != 1:
        raise ContactKinematicsError(
            "expected exactly one visual-only string-bed source mesh, got %d" % len(candidates))
    index, mesh = candidates[0]
    lo = np.asarray(mesh.get("bbox_min_m"), dtype=float)
    hi = np.asarray(mesh.get("bbox_max_m"), dtype=float)
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ContactKinematicsError("string-bed bounding box must contain finite 3D metre values")
    if np.any(hi <= lo):
        raise ContactKinematicsError("string-bed bounding box is degenerate or reversed")

    T = np.eye(4)
    T[:3, 3] = (lo + hi) / 2.0
    mesh_sha = str(mesh.get("sha256", "UNRECORDED"))
    return TransformEvidence(
        name="racket_root_from_stringbed_center_proxy",
        matrix=T,
        status="PROXY_DERIVED",
        source=(
            "%s: source_meshes[%d] visual-only string-bed bbox midpoint; mesh_sha256=%s; "
            "not a measured sweet spot" % (source_label, index, mesh_sha)
        ),
    )


class RacketContactKinematics:
    """Compose the B601-RS FK with explicitly supplied base and racket transforms.

    T_world_arm_root and T_gripper_end_racket_root are required inputs. The
    committed project records both as unresolved for physical hardware. A caller
    may opt into simulation assumptions and/or the visual string-bed proxy, but
    the returned result preserves those evidence labels.
    """

    def __init__(self, chain: Any, *,
                 gripper_end_from_racket_root: TransformEvidence,
                 racket_root_from_contact: TransformEvidence) -> None:
        self.chain = chain
        self.gripper_end_from_racket_root = gripper_end_from_racket_root
        self.racket_root_from_contact = racket_root_from_contact

    def compute(self, q: Any, *, world_from_arm_root: TransformEvidence,
                allow_simulation_assumptions: bool = False,
                allow_proxy_geometry: bool = False) -> RacketContactPose:
        if not hasattr(self.chain, "within_limits") or not hasattr(self.chain, "pose"):
            raise TypeError("chain must expose within_limits(q) and pose(q, link='gripper_end')")
        within = self.chain.within_limits(q)
        violations = [name for name, valid in within.items() if not valid]
        if violations:
            raise JointLimitError("joint vector is outside limits: %s" % ", ".join(violations))

        T_world_arm = world_from_arm_root.resolve(
            allow_simulation_assumptions=allow_simulation_assumptions,
            allow_proxy_geometry=allow_proxy_geometry)
        T_gripper_racket = self.gripper_end_from_racket_root.resolve(
            allow_simulation_assumptions=allow_simulation_assumptions,
            allow_proxy_geometry=allow_proxy_geometry)
        T_racket_contact = self.racket_root_from_contact.resolve(
            allow_simulation_assumptions=allow_simulation_assumptions,
            allow_proxy_geometry=allow_proxy_geometry)

        T_arm_gripper = validate_se3(
            self.chain.pose(q, link="gripper_end"), "arm_root_from_gripper_end")
        T_world_contact = validate_se3(
            T_world_arm @ T_arm_gripper @ T_gripper_racket @ T_racket_contact,
            "world_from_racket_contact")

        evidence = (world_from_arm_root, self.gripper_end_from_racket_root,
                    self.racket_root_from_contact)
        statuses = tuple(item.status for item in evidence)
        if "PROXY_DERIVED" in statuses:
            result_status = "PROXY_DERIVED"
        elif any(status in SIMULATION_STATUSES for status in statuses):
            result_status = "SIMULATION_ASSUMPTION"
        else:
            result_status = "MEASURED"
        frame_name = (
            "racket_stringbed_center_proxy"
            if self.racket_root_from_contact.status == "PROXY_DERIVED"
            else "racket_contact_frame"
        )
        return RacketContactPose(
            matrix_world_contact=T_world_contact,
            status=result_status,
            sources=tuple(item.source for item in evidence),
            input_statuses=statuses,
            frame_name=frame_name,
        )


__all__ = [
    "CalibrationRequiredError", "ContactKinematicsError", "JointLimitError",
    "RacketContactKinematics", "RacketContactPose", "TransformEvidence",
    "racket_stringbed_center_proxy", "validate_se3",
]
