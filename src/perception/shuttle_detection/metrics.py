#!/usr/bin/env python3
"""Detection-matching and capability metrics for the shuttlecock detector.

Why this module exists: spec 03 measures one thing -- where the baseline detector's
capability starts to fall away as the target changes -- and every one of those
curves is a function of the same two primitives: how big the target is, and whether
a predicted box is the target. If each reporting script recomputed them, two curves
in the same report could disagree about what "8-12 px" or "hit" means, and the
disagreement would be invisible. Both primitives are therefore declared here once,
straight from docs/superpowers/specs/03_CONTROLLED_CAPABILITY_SPEC.md.

Size (spec 03 section 2): the shuttlecock's bounding box changes aspect ratio as its
3D pose changes, so neither width nor height alone is a stable measure of how large
the target looks. The geometric mean is the single scale:

    equivalent_size_px = sqrt(bbox_width_px * bbox_height_px)

Buckets (spec 03 section 3) are lower-inclusive and upper-exclusive, so 4.0 is in
"4-6" and 8.0 is in "8-12". Edge values therefore never fall into a gap and never
belong to two buckets; a size sweep always produces a curve with no hole in it.

This module is standard library only on purpose: the capability sweep has to run on
the training host, next to the weights, without reshaping that host's Python
environment.

Run this module's tests with:
    python tests/perception/shuttle_detection/test_metrics.py -v
"""
from __future__ import annotations

import math
from typing import Any

# Spec 03 section 3, in the spec's order. The bucket a size falls into is decided
# by SIZE_BUCKET_EDGES; this tuple is the vocabulary every report writes.
SIZE_BUCKETS: tuple[str, ...] = ("<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", ">32")

# Upper edges, exclusive: a value equal to an edge belongs to the bucket that the
# edge opens, not the one it closes.
SIZE_BUCKET_EDGES: tuple[float, ...] = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def _require_number(value: Any, name: str) -> float:
    """Return value as a finite float, or explain why it is not one.

    A manifest column that has been left blank, mistyped or written as text must
    produce an error the caller can report, not a NumberFormatException traceback
    in the middle of a sweep and not a silently mis-bucketed sample.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(name + " must be a real number, got " + repr(value))
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(name + " must be finite, got " + repr(value))
    return number


def equivalent_size_px(width_px: float, height_px: float) -> float:
    """The primary target-size scalar: sqrt(width_px * height_px).

    Both extents are in pixels and describe the same bounding box (spec 03
    section 2). The geometric mean is used rather than the area or the larger
    extent because the shuttlecock's box aspect ratio changes with its 3D pose: a
    cocked shuttlecock seen head-on is tall and narrow, seen from the side it is
    short and wide, and the geometric mean reports those as similar-sized targets
    while the area would over-weight the wide pose and the max extent would
    discard the other axis entirely.

    A negative extent is rejected: the square root of a negative product is not a
    small target, it is a corrupt manifest row.
    """
    width = _require_number(width_px, "width_px")
    height = _require_number(height_px, "height_px")
    if width < 0.0 or height < 0.0:
        raise ValueError(
            "bounding box extents must not be negative, got "
            + repr(width_px)
            + "x"
            + repr(height_px)
        )
    return math.sqrt(width * height)


def size_bucket(eq: float) -> str:
    """Bucket an equivalent size in pixels into the spec 03 section 3 vocabulary.

    Edges are lower-inclusive and upper-exclusive: 3.999 is "<4", 4.0 is "4-6",
    5.0 is "4-6", 6.0 is "6-8", 8.0 is "8-12", and 32.0 is ">32".

    NaN and infinity are rejected rather than bucketed. Every comparison against
    NaN is False, so a NaN would fall through to the final return and be reported
    as a measured ">32 px" target -- the most flattering possible reading of a
    broken manifest row.
    """
    value = _require_number(eq, "equivalent_size_px")
    for index, edge in enumerate(SIZE_BUCKET_EDGES):
        if value < edge:
            return SIZE_BUCKETS[index]
    return SIZE_BUCKETS[-1]
