# -*- coding: utf-8 -*-
"""Robot frame tree and SE(3) transform algebra (Phase 2).

Spec: BADMINTON_ROBOT.md S7 (frame chain), S28 (env_origin is simulator-only),
      S41 (whole-body racket-contact pose), S50 (round-trip tests).
Names: docs/architecture/COORDINATE_SYSTEM.md S2.5 canonical frames, S2.6 T_A_B naming.

Convention: quaternions are (w, x, y, z) and unit-norm. Transform stores
T_A_B, i.e. it maps points expressed in B into frame A (p_A = T_A_B p_B).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..badminton_robot_cfg import BadmintonRobotCfg


class MissingTransformError(RuntimeError):
    """Raised when a transform required by the chain is still unmounted/unmeasured."""


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic (yaw-pitch-roll) euler angles to unit quaternion (w,x,y,z)."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
    return q / np.linalg.norm(q)


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


class Transform:
    """Rigid transform T_A_B (B -> A)."""

    __slots__ = ("translation", "quaternion")

    def __init__(self, translation: Sequence[float], quaternion: Sequence[float]) -> None:
        t = np.asarray(translation, dtype=float).reshape(3)
        q = np.asarray(quaternion, dtype=float).reshape(4)
        norm = float(np.linalg.norm(q))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError(f"quaternion must be unit-norm, got |q|={norm}")
        self.translation = t
        self.quaternion = q / norm

    @staticmethod
    def identity() -> "Transform":
        return Transform([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def from_xyz_quat(*args) -> "Transform":
        """Accept either (xyz, quat) or a single (xyz, quat) pair."""
        if len(args) == 2:
            xyz, quat = args
        elif len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) == 2:
            xyz, quat = args[0]
        else:
            raise TypeError("from_xyz_quat expects (xyz, quat) or one (xyz, quat) pair")
        return Transform(xyz, quat)

    def matrix(self) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = quat_to_matrix(self.quaternion)
        out[:3, 3] = self.translation
        return out

    def compose(self, other: "Transform") -> "Transform":
        """self @ other  (apply `other` first, then `self`)."""
        m = self.matrix() @ other.matrix()
        return Transform(m[:3, 3], matrix_to_quat(m[:3, :3]))

    def inverse(self) -> "Transform":
        R = quat_to_matrix(self.quaternion)
        Rt = R.T
        return Transform(-Rt @ self.translation, matrix_to_quat(Rt))

    def apply(self, point: Sequence[float]) -> np.ndarray:
        p = np.asarray(point, dtype=float).reshape(3)
        return quat_to_matrix(self.quaternion) @ p + self.translation

    def __matmul__(self, other: "Transform") -> "Transform":
        return self.compose(other)


class FrameTree:
    """Named frames with parent links; transforms resolved by composition."""

    def __init__(self) -> None:
        self.frames: Dict[str, Tuple[str, Transform]] = {}
        self._alias_target: Dict[str, str] = {}

    def add(self, name: str, parent: str, transform: Transform) -> None:
        self.frames[name] = (parent, transform)

    def add_alias(self, alias: str, target: str) -> None:
        """Alias kept so both spec name sets resolve to one frame (no duplicate truth)."""
        self.frames[alias] = ("__alias__", Transform.identity())

        self._alias_target[alias] = target

    def path_to_root(self, frame: str) -> Iterable[str]:
        if frame not in self.frames:
            raise KeyError(f"unknown frame '{frame}' (canonical frames: {sorted(self.frames)})")
        chain = []
        cur = frame
        while True:
            parent, _ = self.frames[cur]
            chain.append(cur)
            if parent in (None, "", "__alias__"):
                break
            cur = parent
        return chain

    def get_transform(self, frame: str, base: str) -> Transform:
        """Return T_base_frame."""
        if frame not in self.frames:
            raise KeyError(f"unknown frame '{frame}'")
        if base not in self.frames:
            raise KeyError(f"unknown base '{base}'")
        if frame in self._alias_target:
            return self.get_transform(self._alias_target[frame], base)
        if base in self._alias_target:
            return self.get_transform(frame, self._alias_target[base]).inverse()
        chain = list(self.path_to_root(frame))
        if base not in chain:
            base_chain = list(self.path_to_root(base))
            common = next((f for f in chain if f in base_chain), None)
            if common is None:
                raise KeyError(f"frames '{frame}' and '{base}' are not connected")
        else:
            common = base
        up = Transform.identity()
        cur = frame
        while cur != common:
            parent, xf = self.frames[cur]
            if parent == "__alias__":
                return self.get_transform(self._alias_target[cur], base)
            up = xf.compose(up)
            cur = parent
        if common == base:
            return up
        down = Transform.identity()
        cur = base
        while cur != common:
            parent, xf = self.frames[cur]
            if parent == "__alias__":
                return self.get_transform(frame, self._alias_target[cur]).inverse()
            down = xf.compose(down)
            cur = parent
        return down.inverse().compose(up)


def _resolve(name: str, override, param) -> Tuple[Sequence[float], Sequence[float]]:
    if override is not None:
        return (override[0], override[1])
    if param is None or param.value is None:
        raise MissingTransformError(
            f"{name} is REQUIRES_MEASUREMENT (still {param.status.value if param else 'UNSET'}); "
            "measure it and pass an explicit transform"
        )
    value = param.value
    if isinstance(value, Mapping):
        return (value["translation_m"], value["quaternion_xyzw"][3:] + value["quaternion_xyzw"][:3])
    raise MissingTransformError(f"{name}: unsupported param layout {type(value)!r}")


def make_robot_frame_tree(
    cfg: BadmintonRobotCfg,
    *,
    base_pose_xyz_quat: Tuple[Sequence[float], Sequence[float]],
    link6_pose_in_piper_xyz_quat: Tuple[Sequence[float], Sequence[float]],
    t_link6_tcp_xyz_quat: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    t_tcp_contact_xyz_quat: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> FrameTree:
    """Build the chain court -> robot_base -> piper_base -> link6 -> tcp -> contact (S7 / S41)."""
    tree = FrameTree()
    tree.add("court", None, Transform.identity())
    tree.add("robot_base", "court", Transform.from_xyz_quat(base_pose_xyz_quat))
    tree.add("piper_base", "robot_base",
             Transform(cfg.piper.mount_translation.value, cfg.piper.mount_quaternion.value))
    tree.add("piper_link6", "piper_base", Transform.from_xyz_quat(link6_pose_in_piper_xyz_quat))

    t_link6_tcp = _resolve("racket.t_link6_tcp", t_link6_tcp_xyz_quat, cfg.racket.t_link6_tcp)
    t_tcp_contact = _resolve("racket.t_tcp_contact", t_tcp_contact_xyz_quat, cfg.racket.t_tcp_contact)
    tree.add("racket_tcp", "piper_link6", Transform.from_xyz_quat(t_link6_tcp))
    tree.add("racket_contact", "racket_tcp", Transform.from_xyz_quat(t_tcp_contact))
    tree.add_alias("racket_contact_frame", "racket_contact")

    cam = cfg.stereo_camera
    pitch = math.radians(float(cam.pitch_deg))
    cam_center = Transform(list(cam.center_offset_robot_xyz), quat_from_rpy(0.0, pitch, 0.0))
    tree.add("camera_center", "robot_base", cam_center)
    tree.add_alias("camera_rig", "camera_center")
    half = float(cam.baseline_m.value) / 2.0
    tree.add("camera_left", "camera_center", Transform([0.0, half, 0.0], [1.0, 0.0, 0.0, 0.0]))
    tree.add("camera_right", "camera_center", Transform([0.0, -half, 0.0], [1.0, 0.0, 0.0, 0.0]))
    # REP-103 optical frame: +Z forward, +X right, +Y down (rotation about X by -90 deg).
    body_to_optical = Transform([0.0, 0.0, 0.0], quat_from_rpy(-math.pi / 2.0, 0.0, 0.0))
    tree.add("camera_left_optical", "camera_left", body_to_optical)
    tree.add("camera_right_optical", "camera_right", body_to_optical)
    # IMU placement is not specified by any source yet: identity TEMP link, never a measured claim.
    tree.add("imu_link", "robot_base", Transform.identity())
    return tree


def compose_racket_contact_court(
    cfg: BadmintonRobotCfg,
    *,
    base_pose_xyz_quat: Tuple[Sequence[float], Sequence[float]],
    link6_pose_in_piper_xyz_quat: Tuple[Sequence[float], Sequence[float]],
    t_link6_tcp_xyz_quat: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    t_tcp_contact_xyz_quat: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Transform:
    """T_court_contact = T_court_base . T_base_piper . T_piper_link6 . T_link6_tcp . T_tcp_contact (S41)."""
    return (Transform.from_xyz_quat(base_pose_xyz_quat)
            .compose(Transform(cfg.piper.mount_translation.value, cfg.piper.mount_quaternion.value))
            .compose(Transform.from_xyz_quat(link6_pose_in_piper_xyz_quat))
            .compose(Transform.from_xyz_quat(
                _resolve("racket.t_link6_tcp", t_link6_tcp_xyz_quat, cfg.racket.t_link6_tcp)))
            .compose(Transform.from_xyz_quat(
                _resolve("racket.t_tcp_contact", t_tcp_contact_xyz_quat, cfg.racket.t_tcp_contact))))


def court_from_world(world_xyz: Sequence[float], env_origin_xyz: Sequence[float]) -> np.ndarray:
    """S28: simulator world -> court frame (env_origin is stripped here and nowhere else)."""
    return np.asarray(world_xyz, dtype=float).reshape(3) - np.asarray(env_origin_xyz, dtype=float).reshape(3)


def world_from_court(court_xyz: Sequence[float], env_origin_xyz: Sequence[float]) -> np.ndarray:
    """Court frame -> simulator world (used only for placement/teleport)."""
    return np.asarray(court_xyz, dtype=float).reshape(3) + np.asarray(env_origin_xyz, dtype=float).reshape(3)


__all__ = [
    "Transform", "FrameTree", "MissingTransformError", "quat_from_rpy",
    "quat_to_matrix", "matrix_to_quat", "make_robot_frame_tree",
    "compose_racket_contact_court", "court_from_world", "world_from_court",
]
