# -*- coding: utf-8 -*-
"""Racket asset geometry config - ENGINEERING_V0_1 TEMP_PLACEHOLDER."""
from .scene_layout import RACKET

# Racket local frame: +X face normal toward opponent, +Z handle->head.
# tcp sits at PiPER link6 mount origin. contact at head sweet-spot centre.
# Values below are the ENGINEERING_V0_1 model; recomputed from actual geometry
# in the racket builder and written to evidence/racket_frames.json.
RACKET_GEO = dict(
    total_length=0.675,
    head_width=0.225,
    head_height=0.290,
    handle_len=0.20,
    shaft_len=0.185,
    head_center_from_tcp=0.50,   # approx, recomputed in builder
    handle_radius=0.012,
    shaft_radius=0.008,
    paddle_thickness=0.008,      # thin collision paddle
    mass=0.10,
)
