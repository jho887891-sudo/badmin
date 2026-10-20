"""Source-aware, dependency-free SE(3) pose checks for P2 PhysX diagnostics.

USD/Gf quaternion values are stored as wxyz. On the recorded Isaac Lab 3.0 beta2
build body_quat_w arrives as xyzw. Never infer a source's order by minimizing
an internal joint residual. The caller must declare each data source.

This module does no simulation and cannot certify runtime PhysX correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite, sqrt
from typing import Sequence


def _vec(values: Sequence[float], n: int, label: str) -> tuple[float, ...]:
    if len(values) != n:
        raise ValueError(f"{label} must have {n} components, got {len(values)}")
    result = tuple(float(v) for v in values)
    if not all(isfinite(v) for v in result):
        raise ValueError(f"{label} contains nonfinite components")
    return result


def _normalize(q: Sequence[float]) -> tuple[float, float, float, float]:
    values = _vec(q, 4, "quaternion")
    magnitude = sqrt(sum(v * v for v in values))
    if magnitude <= 1e-15:
        raise ValueError("zero-length quaternion")
    return tuple(v / magnitude for v in values)  # type: ignore[return-value]


def qmul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = _normalize(a)
    v, p, q, r = _normalize(b)
    return (w*v-x*p-y*q-z*r, w*p+x*v+y*r-z*q,
            w*q-x*r+y*v+z*p, w*r+x*q-y*p+z*v)


def qconj(q: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = _normalize(q)
    return (w, -x, -y, -z)


def rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a vector by a wxyz quaternion in a right-handed world frame."""
    w, x, y, z = _normalize(q)
    vx, vy, vz = _vec(v, 3, "vector")
    tx, ty, tz = (2 * (y*vz-z*vy), 2 * (z*vx-x*vz), 2 * (x*vy-y*vx))
    return (vx+w*tx+(y*tz-z*ty),
            vy+w*ty+(z*tx-x*tz),
            vz+w*tz+(x*ty-y*tx))


def angle(a: Sequence[float], b: Sequence[float]) -> float:
    """Shortest relative rotation in radians, invariant to q and -q."""
    a = _normalize(a)
    b = _normalize(b)
    dot = min(1.0, abs(sum(x*y for x, y in zip(a, b))))
    return 2.0 * acos(dot)


def length(v: Sequence[float]) -> float:
    return sqrt(sum(x*x for x in _vec(v, 3, "vector")))


@dataclass(frozen=True)
class Pose:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    source_api: str

    @classmethod
    def from_usd_wxyz(cls, position: Sequence[float], quat: Sequence[float]) -> "Pose":
        return cls(_vec(position, 3, "USD position"), _normalize(quat), "USD/Gf.Quat(wxyz)")

    @classmethod
    def from_isaac_body_xyzw(cls, position: Sequence[float], quat: Sequence[float]) -> "Pose":
        q = _vec(quat, 4, "Isaac body quaternion xyzw")
        return cls(_vec(position, 3, "Isaac position"),
                   _normalize((q[3], q[0], q[1], q[2])),
                   "IsaacLab.body_quat_w(xyzw, validated build)")


def world_pose_error(authored: Pose, measured: Pose) -> dict[str, float]:
    """Absolute world pose error, unlike a relative FixedJoint residual."""
    delta = tuple(x - y for x, y in zip(measured.position_m, authored.position_m))
    return {"world_position_error_m": length(delta),
            "world_rotation_error_rad": angle(authored.quaternion_wxyz,
                                               measured.quaternion_wxyz)}


def mount_pose_error(chassis: Pose, base: Pose,
                     expected_local_position: Sequence[float],
                     expected_local_quaternion_wxyz: Sequence[float] = (1, 0, 0, 0)) -> dict[str, float]:
    """Check Chassis -> arm-base transform against its USD FixedJoint authority."""
    local = rotate(qconj(chassis.quaternion_wxyz),
                   tuple(x-y for x, y in zip(base.position_m, chassis.position_m)))
    target = _vec(expected_local_position, 3, "mount offset")
    local_q = qmul(qconj(chassis.quaternion_wxyz), base.quaternion_wxyz)
    return {"mount_position_error_m": length(tuple(x-y for x, y in zip(local, target))),
            "mount_rotation_error_rad": angle(local_q, expected_local_quaternion_wxyz)}
