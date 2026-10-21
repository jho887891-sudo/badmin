# -*- coding: utf-8 -*-
"""P8-01: observation schema -- shape/dtype round-trip and every strict rejection path.

These tests are CPU-only, numpy-only and simulator-free: they prove the encoder refuses bad
data instead of training on it.  No training loop exists to call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.observation_schema import (  # noqa: E402
    CLOCK_TOLERANCE_S, FIELD_SLICES, OBSERVATION_BY_NAME, OBSERVATION_DIM, OBSERVATION_FIELDS,
    OBSERVATION_SCHEMA_VERSION, ObservationBatchError, ObservationBoundsError,
    ObservationDTypeError, ObservationNonFiniteError, ObservationSchemaError,
    ObservationShapeError, ObservationStalenessError, ObservationUnitError, ObservationValue,
    MissingFieldError, UnknownFieldError, build_observation, field_names, observation_dim,
    schema_rows, validate,
)

NOW = 10.0


def flags(*values):
    return np.asarray(values, dtype=np.int8)


def good_values(now_s: float = NOW):
    """A complete, in-bounds, fresh observation mapping (synthetic, no simulator needed)."""
    def v(value, unit, ts=None):
        return ObservationValue(np.asarray(value, dtype=float), unit, now_s if ts is None else ts)

    return {
        "base_position": v([0.0, 0.0, 0.0], "m"),
        "base_orientation": v([1.0, 0.0, 0.0, 0.0], "1"),
        "base_linear_velocity": v([0.05, -0.02, 0.0], "m/s"),
        "base_angular_velocity": v([0.0, 0.0, 0.1], "rad/s"),
        "joint_position": v([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], "rad"),
        "joint_velocity": v([0.01, 0.02, -0.03, 0.04, -0.05, 0.06], "rad/s"),
        "racket_position": v([-1.0, 0.0, 1.0], "m"),
        "racket_orientation": v([0.0, 1.0, 0.0, 0.0], "1"),
        "racket_linear_velocity": v([0.2, 0.0, -0.3], "m/s"),
        "shuttle_position": v([0.0, 0.5, 1.5], "m"),
        "shuttle_velocity": v([[0.0, 0.0, -1.0], [0.0, 0.0, -1.1], [0.0, 0.0, -1.2]], "m/s"),
        "shuttle_covariance": v([0.01] + [0.0] * 8, "m^2"),
        "time_to_hit_s": v([0.35], "s"),
        "feasibility_flags": ObservationValue(flags(1, 1, 1, 1), "1", now_s),
        "prev_base_linear_command": v([0.04, -0.01], "m/s"),
        "prev_base_angular_command": v([0.05], "rad/s"),
        "prev_joint_delta_command": v([0.0] * 6, "rad"),
    }


def batched(values, rows: int):
    out = {}
    for name, item in values.items():
        arr = np.asarray(item.value)  # keep the channel dtype (the flag channel is integer)
        out[name] = ObservationValue(np.repeat(arr[None, ...], rows, axis=0), item.unit,
                                     item.timestamp_s)
    return out


# ---------------------------------------------------------------------------- happy paths
def test_dimension_and_layout_are_declared_consistently():
    assert observation_dim() == OBSERVATION_DIM == 70
    assert len(OBSERVATION_FIELDS) == len(FIELD_SLICES) == 17
    total = 0
    for field in OBSERVATION_FIELDS:
        sl = FIELD_SLICES[field.name]
        assert sl.start == total
        assert sl.stop - sl.start == field.width
        total = sl.stop
    assert total == OBSERVATION_DIM
    assert OBSERVATION_SCHEMA_VERSION == "p8-obs-v0.1-candidate"


def test_build_roundtrip_shape_dtype_and_values():
    values = good_values()
    obs = build_observation(values, now_s=NOW)
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float64
    for name, item in values.items():
        sl = FIELD_SLICES[name]
        expected = np.asarray(item.value, dtype=float).reshape(-1)
        np.testing.assert_allclose(obs[sl], expected)
    # history is flattened frame-major: (history, 3) -> 9 values, oldest first
    np.testing.assert_allclose(obs[FIELD_SLICES["shuttle_velocity"]],
                               np.asarray(values["shuttle_velocity"].value).reshape(-1))


def test_batched_roundtrip_matches_unbatched_rows():
    single = build_observation(good_values(), now_s=NOW)
    batch = build_observation(batched(good_values(), 4), now_s=NOW)
    assert batch.shape == (4, OBSERVATION_DIM)
    for row in batch:
        np.testing.assert_allclose(row, single)
    report = validate(batched(good_values(), 4), now_s=NOW)
    assert report.batch_size == 4
    assert report.dim == OBSERVATION_DIM
    assert report.training_status == "PREPARATION_ONLY_NO_TRAINING"


def test_unit_aliases_are_accepted_but_units_are_not_ignored():
    values = good_values()
    values["joint_position"] = ObservationValue(values["joint_position"].value, "radians", NOW)
    np.testing.assert_allclose(build_observation(values, now_s=NOW),
                               build_observation(good_values(), now_s=NOW))


# ------------------------------------------------------------------------ rejection paths
def test_missing_key_is_rejected():
    values = good_values()
    del values["racket_position"]
    with pytest.raises(MissingFieldError) as ei:
        build_observation(values, now_s=NOW)
    assert "racket_position" in str(ei.value)


def test_unknown_key_is_rejected():
    values = good_values()
    values["racket_postion"] = ObservationValue(np.zeros(3), "m", NOW)  # typo
    with pytest.raises(UnknownFieldError):
        build_observation(values, now_s=NOW)


def test_wrong_shape_is_rejected():
    values = good_values()
    values["base_position"] = ObservationValue(np.zeros(2), "m", NOW)
    with pytest.raises(ObservationShapeError):
        build_observation(values, now_s=NOW)


def test_missing_history_is_rejected_and_explained():
    values = good_values()
    values["shuttle_velocity"] = ObservationValue(np.zeros(3), "m/s", NOW)
    with pytest.raises(ObservationShapeError) as ei:
        build_observation(values, now_s=NOW)
    assert "history=3" in str(ei.value)


@pytest.mark.parametrize("bad", [
    np.array(["0", "0", "0"]),                     # strings
    np.array([1 + 2j, 0, 0]),                      # complex
    np.array([True, False, True]),                 # bool in a float channel
])
def test_wrong_dtype_is_rejected(bad):
    values = good_values()
    values["base_position"] = ObservationValue(bad, "m", NOW)
    with pytest.raises(ObservationDTypeError):
        build_observation(values, now_s=NOW)


def test_flag_channel_rejects_a_probability():
    values = good_values()
    values["feasibility_flags"] = ObservationValue(np.array([1.0, 0.5, 1.0, 1.0]), "1", NOW)
    with pytest.raises(ObservationDTypeError) as ei:
        build_observation(values, now_s=NOW)
    assert "flag" in str(ei.value)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_are_rejected(bad):
    values = good_values()
    bad_pos = np.zeros(3)
    bad_pos[1] = bad
    values["shuttle_position"] = ObservationValue(bad_pos, "m", NOW)
    with pytest.raises(ObservationNonFiniteError):
        build_observation(values, now_s=NOW)


def test_out_of_bounds_is_rejected():
    values = good_values()
    values["shuttle_position"] = ObservationValue(np.array([100.0, 0.0, 1.0]), "m", NOW)
    with pytest.raises(ObservationBoundsError):
        build_observation(values, now_s=NOW)

    values = good_values()
    values["joint_position"] = ObservationValue(np.full(6, 3.0), "rad", NOW)  # > 2.6 proxy
    with pytest.raises(ObservationBoundsError):
        build_observation(values, now_s=NOW)


@pytest.mark.parametrize("channel,unit,needle", [
    ("joint_position", "deg", "degrees"),
    ("shuttle_position", "mm", "millimetres"),
    ("racket_position", "cm", "centimetres"),
    ("base_linear_velocity", "km/h", "unit confusion"),
])
def test_unit_confusion_is_rejected(channel, unit, needle):
    values = good_values()
    values[channel] = ObservationValue(values[channel].value, unit, NOW)
    with pytest.raises(ObservationUnitError) as ei:
        build_observation(values, now_s=NOW)
    assert needle in str(ei.value)


def test_stale_timestamp_is_rejected():
    values = good_values()
    values["shuttle_position"] = ObservationValue(np.zeros(3), "m", NOW - 0.5)  # max_age 0.15
    with pytest.raises(ObservationStalenessError) as ei:
        build_observation(values, now_s=NOW)
    assert "stale" in str(ei.value)


def test_future_timestamp_is_rejected():
    values = good_values()
    values["joint_position"] = ObservationValue(np.zeros(6), "rad", NOW + CLOCK_TOLERANCE_S + 0.1)
    with pytest.raises(ObservationStalenessError) as ei:
        build_observation(values, now_s=NOW)
    assert "future" in str(ei.value)


def test_missing_timestamp_is_rejected():
    values = good_values()
    values["joint_velocity"] = ObservationValue(np.zeros(6), "rad/s", None)
    with pytest.raises(ObservationStalenessError):
        build_observation(values, now_s=NOW)


def test_staleness_check_requires_a_reference_time():
    with pytest.raises(ObservationSchemaError) as ei:
        build_observation(good_values(), now_s=None)
    assert "now_s is required" in str(ei.value)
    # explicitly opting out is allowed and reported
    report = validate(good_values(), now_s=None, enforce_staleness=False)
    assert report.staleness_enforced is False


def test_bare_array_without_unit_is_rejected():
    values = good_values()
    values["base_position"] = np.zeros(3)
    with pytest.raises(ObservationSchemaError) as ei:
        build_observation(values, now_s=NOW)
    assert "ObservationValue" in str(ei.value)


def test_mixed_batching_is_rejected():
    values = good_values()
    values["base_position"] = ObservationValue(np.zeros((4, 3)), "m", NOW)
    with pytest.raises(ObservationBatchError):
        build_observation(values, now_s=NOW)


def test_inconsistent_batch_sizes_are_rejected():
    values = batched(good_values(), 4)
    values["joint_position"] = ObservationValue(np.zeros((3, 6)), "rad", NOW)
    with pytest.raises(ObservationBatchError):
        build_observation(values, now_s=NOW)


# --------------------------------------------------------------- validate() on arrays
def test_validate_array_reports_and_rejects_structural_damage():
    obs = build_observation(good_values(), now_s=NOW)
    report = validate(obs)
    assert report.dim == OBSERVATION_DIM and report.batch_size is None
    assert set(report.checks) == {"dim", "dtype", "finite", "bounds"}

    with pytest.raises(ObservationDTypeError):
        validate(obs.astype(np.float32))
    with pytest.raises(ObservationShapeError):
        validate(np.zeros(OBSERVATION_DIM - 1))
    broken = obs.copy()
    broken[0] = np.nan
    with pytest.raises(ObservationNonFiniteError):
        validate(broken)
    illegal = obs.copy()
    illegal[FIELD_SLICES["shuttle_position"]] = 500.0
    with pytest.raises(ObservationBoundsError):
        validate(illegal)


def test_validator_does_not_repair_or_clamp():
    values = good_values()
    values["shuttle_position"] = ObservationValue(np.array([0.0, 0.0, 99.0]), "m", NOW)
    with pytest.raises(ObservationBoundsError):
        validate(values, now_s=NOW)  # mapping path is strict too


def test_schema_declares_unit_bounds_source_and_staleness_for_every_channel():
    rows = schema_rows()
    assert len(rows) == len(field_names())
    units = set()
    for row in rows:
        assert row["unit"], row["name"]
        assert row["source"], row["name"]
        assert row["max_age_s"] > 0.0, row["name"]
        assert row["bounds"] is not None, row["name"]
        lo, hi = row["bounds"]
        assert len(lo) == len(hi) == OBSERVATION_BY_NAME[row["name"]].elements
        assert all(a < b for a, b in zip(lo, hi)), row["name"]
        units.add(row["unit"])
    assert {"m", "m/s", "rad", "rad/s", "s", "m^2", "1"}.issubset(units)
