"""Convert [x_px, y_px, vx_px/s, vy_px/s] covariance into pixel-only uncertainty.

A 4x4 kinematic covariance contains mixed units; its full eigenvalues cannot
be interpreted as spatial uncertainty. Use only the 2x2 xy block for ellipses.
"""
from __future__ import annotations

import math
import numpy as np

# Joint 95% probability region of a 2D Gaussian (chi-square, 2 degrees of freedom).
CHI2_2D_95 = 5.991464547107979


def position_uncertainty(state_covariance: np.ndarray) -> dict:
    """Return position sigma and a joint 95% xy ellipse; axes are radii in px."""
    matrix = np.asarray(state_covariance, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("state covariance must be finite 4x4 [x,y,vx,vy]")
    matrix = 0.5 * (matrix + matrix.T)
    position = matrix[:2, :2]
    velocity = matrix[2:, 2:]
    eigvals, eigvecs = np.linalg.eigh(position)
    if eigvals[0] < -1e-8:
        raise ValueError("xy covariance must be positive semidefinite")
    eigvals = np.maximum(eigvals, 0.0)
    minor_sigma, major_sigma = np.sqrt(eigvals)
    direction = eigvecs[:, 1]
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    return {
        "position_covariance_px2": position.tolist(),
        "velocity_covariance_px2_s2": velocity.tolist(),
        "position_sigma_major_px": float(major_sigma),
        "position_sigma_minor_px": float(minor_sigma),
        "ellipse95": {
            "confidence": 0.95,
            "chi2_df2": CHI2_2D_95,
            "major_radius_px": float(math.sqrt(CHI2_2D_95) * major_sigma),
            "minor_radius_px": float(math.sqrt(CHI2_2D_95) * minor_sigma),
            "angle_deg_image_xy": angle,
        },
    }
