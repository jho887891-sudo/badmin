#!/usr/bin/env python3
"""Target-size metrics for the controlled shuttlecock capability test.

Why this module exists: spec 03 sections 2 and 3 define the primary size scalar
that every capability curve is plotted against, and the eight buckets the curves
are binned into. The bucket edges are the whole point -- a curve labelled "4-6 px"
is a claim about which samples are inside it -- so this test pins every edge, both
sides of it. The edges are lower-inclusive and upper-exclusive, which is what makes
5.0 land in "4-6" and 8.0 land in "8-12" as the plan requires: a size sweep must
never fall into a gap between two buckets, and no sample may appear in two.

Run:
    python tests/perception/shuttle_detection/test_metrics.py -v
    pytest tests/perception/shuttle_detection/test_metrics.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.metrics import (  # noqa: E402
    SIZE_BUCKET_EDGES,
    SIZE_BUCKETS,
    equivalent_size_px,
    size_bucket,
)

# Spec 03 section 3, in order. The list itself is part of the contract: a ninth
# bucket added here would silently split every existing capability curve.
EXPECTED_BUCKETS = ("<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", ">32")

# (value, bucket) for every edge of spec 03 section 3, edge value included in the
# bucket that starts at it: 4.0 -> "4-6", 6.0 -> "6-8", ... 32.0 -> ">32".
EDGE_CASES = (
    (4.0, "4-6"),
    (6.0, "6-8"),
    (8.0, "8-12"),
    (12.0, "12-16"),
    (16.0, "16-24"),
    (24.0, "24-32"),
    (32.0, ">32"),
)


# --------------------------------------------------------------------------- #
# equivalent_size_px = sqrt(width_px * height_px)  (spec 03 section 2)
# --------------------------------------------------------------------------- #


def test_equivalent_size_of_a_square_is_its_side():
    assert equivalent_size_px(6, 6) == 6.0


def test_equivalent_size_of_the_plan_example():
    # plan Task 1 step 1: sqrt(4 * 9) = 6.0
    assert equivalent_size_px(4, 9) == 6.0


def test_equivalent_size_of_a_rectangle_is_the_geometric_mean():
    assert equivalent_size_px(2.0, 8.0) == 4.0
    assert equivalent_size_px(3.0, 12.0) == 6.0


def test_equivalent_size_is_symmetric_in_width_and_height():
    assert equivalent_size_px(5.0, 20.0) == equivalent_size_px(20.0, 5.0)


def test_equivalent_size_of_a_zero_extent_is_zero():
    assert equivalent_size_px(0.0, 7.0) == 0.0


def test_equivalent_size_of_a_fractional_box():
    assert equivalent_size_px(2.5, 10.0) == pytest.approx(5.0)


def test_equivalent_size_rejects_a_negative_extent():
    # A negative extent is not a small target, it is a broken manifest row; the
    # geometric mean of one negative number is not a real number and must not be
    # silently returned as a float.
    with pytest.raises(ValueError):
        equivalent_size_px(-1.0, 9.0)
    with pytest.raises(ValueError):
        equivalent_size_px(9.0, -1.0)


def test_equivalent_size_rejects_a_non_number():
    with pytest.raises(TypeError):
        equivalent_size_px("4", 9)


# --------------------------------------------------------------------------- #
# size_bucket (spec 03 section 3)
# --------------------------------------------------------------------------- #


def test_bucket_vocabulary_is_exactly_the_spec_list():
    assert SIZE_BUCKETS == EXPECTED_BUCKETS
    assert SIZE_BUCKET_EDGES == (4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


@pytest.mark.parametrize("value,expected", EDGE_CASES)
def test_bucket_at_every_edge_value(value, expected):
    # The plan's boundary rule: the edge value belongs to the bucket above it.
    assert size_bucket(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    (
        (4.0, "4-6"),
        (6.0, "6-8"),
        (8.0, "8-12"),
        (12.0, "12-16"),
        (16.0, "16-24"),
        (24.0, "24-32"),
        (32.0, ">32"),
    ),
)
def test_bucket_just_below_every_edge_value(value, expected):
    # One ULP-scale step below an edge stays in the bucket below it, so a value
    # that is merely near an edge is not promoted into it.
    assert size_bucket(math.nextafter(value, 0.0)) != expected


def test_bucket_of_the_plan_examples():
    # plan Task 1 step 1
    assert size_bucket(5.0) == "4-6"
    assert size_bucket(8.0) == "8-12"


@pytest.mark.parametrize(
    "value,expected",
    (
        (0.0, "<4"),
        (3.999, "<4"),
        (5.0, "4-6"),
        (7.5, "6-8"),
        (10.0, "8-12"),
        (15.0, "12-16"),
        (20.0, "16-24"),
        (30.0, "24-32"),
        (40.0, ">32"),
        (1e6, ">32"),
    ),
)
def test_bucket_interior_values(value, expected):
    assert size_bucket(value) == expected


def test_bucket_covers_the_whole_number_line():
    # A sweep spanning every edge must produce a bucket for every sample: an
    # unmapped value would show up as a gap in a capability curve rather than as
    # an error.
    values = [0.0, 0.001]
    for edge in SIZE_BUCKET_EDGES:
        values.extend([edge - 0.001, edge, edge + 0.001])
    values.extend([100.0, 1000.0])
    for value in values:
        assert size_bucket(value) in EXPECTED_BUCKETS


def test_bucket_is_monotone_in_size():
    # Walking up the size axis must never walk back down the bucket list; if it
    # did, a capability curve would be non-monotone for a reason that has nothing
    # to do with the detector.
    order = {name: index for index, name in enumerate(SIZE_BUCKETS)}
    previous = -1
    for index in range(0, 4000):
        value = index / 100.0
        current = order[size_bucket(value)]
        assert current >= previous
        previous = current


def test_bucket_accepts_integer_input():
    assert size_bucket(5) == "4-6"


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_bucket_rejects_a_non_finite_size(value):
    # NaN is the dangerous one: every comparison against it is False, so a naive
    # implementation returns the highest bucket and reports a broken manifest row
    # as a measured very-large target.
    with pytest.raises(ValueError):
        size_bucket(value)


def test_bucket_rejects_a_non_number():
    with pytest.raises(TypeError):
        size_bucket("5.0")
    with pytest.raises(TypeError):
        size_bucket(None)
