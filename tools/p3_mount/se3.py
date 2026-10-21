#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal SE(3)/SO(3) helpers for the P3-01 mount frame contract.

Pure python + numpy, quaternions are **wxyz** (USD / GfQuatf convention).
Kept in its own module so the extractor, the fixture generator and the test all
share ONE implementation of the algebra instead of three copies.
"""
from __future__ import annotations

import math

import numpy as np

IDENTITY_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]


def quat_normalize(q):
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        raise ValueError("zero quaternion")
    return q / n


def quat_to_matrix(q):
    """wxyz -> 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def matrix_to_quat(m):
    """3x3 rotation matrix -> wxyz (canonical: w >= 0)."""
    m = np.asarray(m, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = quat_normalize(q)
    return q if q[0] >= 0.0 else -q


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def make_transform(pos, quat_wxyz):
    out = np.eye(4)
    out[:3, :3] = quat_to_matrix(quat_wxyz)
    out[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return out


def transform_from_matrix(rot, pos):
    out = np.eye(4)
    out[:3, :3] = np.asarray(rot, dtype=float).reshape(3, 3)
    out[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    return out


def compose(*transforms):
    out = np.eye(4)
    for t in transforms:
        out = out @ np.asarray(t, dtype=float).reshape(4, 4)
    return out


def pose_of(transform):
    t = np.asarray(transform, dtype=float).reshape(4, 4)
    p = t[:3, 3].copy()
    q = matrix_to_quat(t[:3, :3])
    return [round(float(v), 12) for v in p], [round(float(v), 12) for v in q]


def position_delta(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def rotation_angle(rot_a, rot_b=None):
    """Geodesic angle in radians between two rotation matrices (or against identity)."""
    ra = np.asarray(rot_a, dtype=float).reshape(3, 3)
    rb = np.eye(3) if rot_b is None else np.asarray(rot_b, dtype=float).reshape(3, 3)
    r = ra @ rb.T
    cos = (float(np.trace(r)) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, cos))))


def quat_angle_rad(q_a, q_b=None):
    return rotation_angle(quat_to_matrix(q_a), None if q_b is None else quat_to_matrix(q_b))


def determinant(rot):
    return float(np.linalg.det(np.asarray(rot, dtype=float).reshape(3, 3)))
