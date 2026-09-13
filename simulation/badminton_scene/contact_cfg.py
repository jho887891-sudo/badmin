# -*- coding: utf-8 -*-
"""Contact report configuration & event taxonomy for scene v0.1."""
CONTACT_TYPES = ["SHUTTLE_RACKET", "SHUTTLE_NET", "SHUTTLE_GROUND"]
CONTACT_FIELDS = [
    "timestamp", "physics_step", "env_id", "contact_type", "body_a", "body_b",
    "force_x", "force_y", "force_z",
    "shuttle_px", "shuttle_py", "shuttle_pz",
    "shuttle_vx_before", "shuttle_vy_before", "shuttle_vz_before",
    "shuttle_vx_after", "shuttle_vy_after", "shuttle_vz_after",
]
