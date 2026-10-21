# -*- coding: utf-8 -*-
"""P8-02: reward terms -- offline computability on synthetic frames, masked != zero.

Purity, masking and the corrupted-frame policy are all asserted here.  No simulator, no
training loop, no optimiser: these are pure functions over recorded frames.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.reward_terms import (  # noqa: E402
    REWARD_TERMS, REWARD_TERM_NAMES, Frame, RewardParams, RewardWeights, TermResult,
    collision_penalty, contact_success, distance_to_intercept, frame_fingerprint,
    instability_penalty, joint_limit_penalty, offline_computability_report, timeout_penalty,
    total_reward, velocity_matching,
)


def full_frame():
    """Fully recorded synthetic frame: every term must be computable on it."""
    return Frame(
        t_s=1.0,
        shuttle_position=np.array([0.0, 0.0, 1.0]),
        shuttle_velocity=np.array([0.0, 0.0, 3.0]),
        racket_position=np.array([0.0, 0.0, 0.8]),
        racket_velocity=np.array([0.0, 0.0, 2.0]),
        intercept_point=np.array([0.0, 0.0, 0.5]),
        contact_detected=True,
        collision_impulse_ns=0.2,
        joint_limit_excess_rad=0.0,
        base_tilt_rad=0.1,
    )


def test_every_term_is_computable_offline_on_a_complete_frame():
    frame = full_frame()
    results = {name: fn(frame) for name, fn in (
        ("distance_to_intercept", distance_to_intercept),
        ("contact_success", contact_success),
        ("velocity_matching", velocity_matching),
        ("collision_penalty", collision_penalty),
        ("joint_limit_penalty", joint_limit_penalty),
        ("instability_penalty", instability_penalty),
        ("timeout_penalty", timeout_penalty),
    )}
    assert all(isinstance(r, TermResult) for r in results.values())
    assert {r.name for r in results.values()} == set(REWARD_TERM_NAMES)
    assert not any(r.masked for r in results.values())
    assert results["distance_to_intercept"].value == pytest.approx(-1.0)   # 0.5 m / 0.5 m
    assert results["contact_success"].value == pytest.approx(1.0)
    assert results["velocity_matching"].value == pytest.approx(-1.0)       # 1.0 m/s / 1.0
    assert results["collision_penalty"].value == pytest.approx(0.0)        # below threshold
    assert results["joint_limit_penalty"].value == pytest.approx(0.0)
    assert results["instability_penalty"].value == pytest.approx(0.0)
    assert results["timeout_penalty"].value == pytest.approx(0.0)


def test_total_reward_sums_the_weighted_terms():
    breakdown = total_reward(full_frame())
    assert breakdown.complete is True
    assert breakdown.masked_terms == ()
    expected = 1.0 * (-1.0) + 5.0 * 1.0 + 0.5 * (-1.0) + 2.0 * 0.0 + 0.0 + 0.0 + 0.0
    assert breakdown.total == pytest.approx(expected) == pytest.approx(3.5)
    weighted = breakdown.weighted(RewardWeights.temp_proxy())
    assert weighted["contact_success"] == pytest.approx(5.0)
    assert weighted["distance_to_intercept"] == pytest.approx(-1.0)
    assert "NOT tuned" in breakdown.weights_source


def test_masked_is_not_zero_even_when_the_totals_coincide():
    no_sensor = dataclasses.replace(full_frame(), contact_detected=None)
    measured_miss = dataclasses.replace(full_frame(), contact_detected=False)

    masked_breakdown = total_reward(no_sensor)
    zero_breakdown = total_reward(measured_miss)

    masked_term = masked_breakdown.by_name()["contact_success"]
    zero_term = zero_breakdown.by_name()["contact_success"]

    assert masked_term.masked is True and masked_term.value is None
    assert "not recorded" in masked_term.reason
    assert zero_term.masked is False and zero_term.value == pytest.approx(0.0)
    assert "measured" in zero_term.reason

    assert masked_breakdown.masked_terms == ("contact_success",)
    assert zero_breakdown.masked_terms == ()
    assert masked_breakdown.complete is False and zero_breakdown.complete is True
    # the numeric totals coincide -- which is exactly why the mask must be reported
    assert masked_breakdown.total == pytest.approx(zero_breakdown.total) == pytest.approx(-1.5)


def test_missing_data_never_becomes_a_silent_zero():
    empty = Frame()
    breakdown = total_reward(empty)
    assert breakdown.total == pytest.approx(0.0)
    assert breakdown.complete is False
    assert set(breakdown.masked_terms) == set(REWARD_TERM_NAMES)
    for result in breakdown.results:
        assert result.masked is True
        assert result.value is None
        assert result.reason  # every mask explains itself


def test_term_result_rejects_a_nan_and_an_unlabelled_none():
    with pytest.raises(ValueError):
        TermResult.measured("distance_to_intercept", float("nan"))
    with pytest.raises(ValueError):
        TermResult(name="x", value=None, masked=False)
    with pytest.raises(ValueError):
        TermResult(name="x", value=1.0, masked=True)


def test_partial_records_mask_only_their_own_term():
    frame = dataclasses.replace(full_frame(), shuttle_velocity=None)
    breakdown = total_reward(frame)
    assert breakdown.masked_terms == ("velocity_matching",)
    assert breakdown.by_name()["distance_to_intercept"].masked is False
    assert "shuttle_velocity" in breakdown.by_name()["velocity_matching"].reason


def test_corrupt_frames_raise_instead_of_masking():
    with pytest.raises(ValueError):
        distance_to_intercept(dataclasses.replace(full_frame(), shuttle_position=np.zeros(2)))
    with pytest.raises(ValueError):
        velocity_matching(dataclasses.replace(
            full_frame(), shuttle_velocity=np.array([0.0, np.nan, 1.0])))
    with pytest.raises(ValueError):
        collision_penalty(dataclasses.replace(full_frame(), collision_impulse_ns=float("inf")))


def test_timeout_boundary_and_zero_measurement():
    assert timeout_penalty(Frame(t_s=5.999)).value == pytest.approx(0.0)
    assert timeout_penalty(Frame(t_s=6.0)).value == pytest.approx(-1.0)
    assert timeout_penalty(Frame(t_s=6.0)).masked is False
    assert timeout_penalty(Frame()).masked is True


def test_terms_are_pure_functions_of_the_recorded_frame():
    frame = full_frame()
    before = frame_fingerprint(frame)
    arrays_before = {f.name: None if getattr(frame, f.name) is None
                     else np.array(getattr(frame, f.name), copy=True)
                     for f in dataclasses.fields(Frame)}
    first = [fn(frame) for _, fn in ((s.name, s.fn) for s in REWARD_TERMS)]
    second = [fn(frame) for _, fn in ((s.name, s.fn) for s in REWARD_TERMS)]
    assert [r.value for r in first] == [r.value for r in second]
    assert frame_fingerprint(frame) == before
    for name, original in arrays_before.items():
        if original is None:
            continue
        np.testing.assert_array_equal(np.asarray(getattr(frame, name)), original)
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.contact_detected = False


def test_offline_computability_report_makes_every_mask_visible():
    frames = (full_frame(),
              dataclasses.replace(full_frame(), contact_detected=None),
              Frame())
    rows = offline_computability_report(frames)
    assert tuple(r.term for r in rows) == REWARD_TERM_NAMES
    by_term = {r.term: r for r in rows}
    # the empty frame records nothing, so it masks every term except the ones it can answer
    assert by_term["contact_success"].computable_frames == 1
    assert by_term["contact_success"].masked_frames == 2
    assert by_term["contact_success"].all_computable is False
    assert by_term["distance_to_intercept"].computable_frames == 2
    assert by_term["distance_to_intercept"].masked_frames == 1
    assert by_term["timeout_penalty"].computable_frames == 2
    assert by_term["timeout_penalty"].masked_frames == 1
    for row in rows:
        assert row.per_frame[2].startswith("masked: ")
        assert row.per_frame[0].startswith("-") or row.per_frame[0].startswith("0") \
            or "(" in row.per_frame[0]


def test_weights_and_params_are_labelled_placeholders():
    weights = RewardWeights.temp_proxy()
    params = RewardParams.temp_proxy()
    assert "PLACEHOLDER" in weights.weight_source
    assert "TEMP_PARAMETERIZED_PROXY" in params.param_source
    for spec in REWARD_TERMS:
        assert hasattr(weights, spec.weight_key), spec.name
        assert spec.unit, spec.name
        assert spec.description, spec.name
    assert len(set(REWARD_TERM_NAMES)) == len(REWARD_TERM_NAMES) == 7
    assert all(spec.fn(full_frame()) .name == spec.name for spec in REWARD_TERMS)
