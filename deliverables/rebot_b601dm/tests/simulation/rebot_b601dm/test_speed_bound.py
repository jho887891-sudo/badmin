"""Task 7: the speed bound, the duty cycle, and the experiment record.

A racket-head speed with no bound attached is not a result. The bound comes from the real motors and the
geometry, and it is written here as a test rather than as prose so that it cannot drift away from the numbers
the rest of the work uses.

Reference values, all from the frozen sources:
  reach 0.767 m                      README_zh.md:176, B601-DM maximum
  DM4340 no-load 52.5 rpm = 5.4980 rad/s, DM4310 200 rpm = 20.9440 rad/s
  official speed guidance 70% of maximum   Performance_Testing_zh.md:87

The bound is omega * reach, an UPPER BOUND: it ignores joint coupling, which lowers it, and it stops at the
flange, so a racket of length L raises the true figure rather than lowering it. It is quoted to show the SIZE
of the discrepancy the shipped asset had, not as a prediction.
"""
from __future__ import annotations

import pytest

from src.simulation.rebot_b601dm.joint_limits import rad_s_from_rpm
from src.simulation.rebot_b601dm.speed_bound import (
    DEFAULT_DUTY_CYCLE,
    DutyCycle,
    REACH_M,
    bound_at_fraction,
    experiment_record,
    limit_variant_factor,
    tip_speed_bound,
)

DM4340_NO_LOAD_RAD_S = rad_s_from_rpm(52.5)
DM4310_NO_LOAD_RAD_S = rad_s_from_rpm(200.0)


class TestTheGeometry:
    def test_the_reach_is_the_specified_one(self):
        assert REACH_M == pytest.approx(0.767)


class TestTheBoundMatchesTheRealMotors:
    def test_the_dm4340_bound(self):
        assert tip_speed_bound(DM4340_NO_LOAD_RAD_S) == pytest.approx(4.22, rel=0.02)

    def test_the_dm4310_bound(self):
        assert tip_speed_bound(DM4310_NO_LOAD_RAD_S) == pytest.approx(16.06, rel=0.02)

    def test_the_official_seventy_percent_guidance(self):
        assert tip_speed_bound(0.70 * DM4310_NO_LOAD_RAD_S) == pytest.approx(11.24, rel=0.02)

    def test_a_custom_reach_is_honoured(self):
        # both sides must use the precise constant; mixing it with a rounded literal cannot hold at 1e-9
        assert tip_speed_bound(DM4310_NO_LOAD_RAD_S, reach_m=0.5) == pytest.approx(DM4310_NO_LOAD_RAD_S * 0.5, rel=1e-12)

    def test_a_negative_speed_is_refused(self):
        with pytest.raises(ValueError):
            tip_speed_bound(-1.0)


class TestTheShippedAssetWasAnOrderOfMagnitudeOut:
    def test_the_shipped_limits_would_have_permitted_about_153_metres_per_second(self):
        shipped = 200.0  # the rad/s figure the shipped asset enforced
        assert tip_speed_bound(shipped) == pytest.approx(153.4, rel=0.01)

    def test_the_real_bound_is_the_smaller_one_by_about_ten_times(self):
        shipped = tip_speed_bound(200.0)
        real = tip_speed_bound(DM4310_NO_LOAD_RAD_S)
        assert shipped / real == pytest.approx(9.55, rel=0.02)


class TestTheLimitVariantIsPartOfTheIdentity:
    def test_the_two_variants_are_the_ones_the_patch_produces(self):
        assert limit_variant_factor("real") == 1.0
        assert limit_variant_factor("recommended_70") == 0.70

    def test_an_unknown_variant_is_refused(self):
        with pytest.raises(ValueError):
            limit_variant_factor("whatever")

    def test_the_bound_scales_with_the_variant(self):
        full = bound_at_fraction(1.0)
        reduced = bound_at_fraction(0.70)
        assert reduced == pytest.approx(0.70 * full, rel=1e-9)


class TestTheDutyCycleRespectsTheOfficialTesting:
    def test_the_default_is_inside_the_published_recommendations(self):
        d = DEFAULT_DUTY_CYCLE
        assert d.payload_kg < 1.5
        assert d.reach_fraction < 0.70
        assert d.speed_fraction <= 0.70

    def test_an_overloaded_duty_cycle_is_refused(self):
        with pytest.raises(ValueError):
            DutyCycle(payload_kg=2.5, reach_fraction=0.7, speed_fraction=0.7)

    def test_reaching_beyond_the_recommended_radius_is_refused(self):
        with pytest.raises(ValueError):
            DutyCycle(payload_kg=1.0, reach_fraction=1.0, speed_fraction=0.7)

    def test_the_official_failure_modes_are_carried_as_evidence(self):
        d = DEFAULT_DUTY_CYCLE
        text = " ".join(d.evidence).lower()
        assert "motor 2" in text
        assert "90" in text or "thermal" in text

    def test_a_conforming_duty_cycle_is_accepted(self):
        d = DutyCycle(payload_kg=1.2, reach_fraction=0.5, speed_fraction=0.5)
        assert d.is_compliant


class TestTheExperimentRecordCarriesEverythingNeeded:
    def test_the_record_names_the_limit_variant_and_torque_convention_and_duty_cycle(self):
        rec = experiment_record(limit_variant="real", torque_convention="rated",
                                duty_cycle=DEFAULT_DUTY_CYCLE, measured_peak_m_s=3.1)
        assert rec["limit_variant"] == "real"
        assert rec["torque_convention"] == "rated"
        assert rec["duty_cycle"]["payload_kg"] < 1.5
        assert rec["measured_peak_m_s"] == pytest.approx(3.1)

    def test_the_record_states_the_bound_it_must_not_exceed(self):
        rec = experiment_record(limit_variant="real", torque_convention="rated",
                                duty_cycle=DEFAULT_DUTY_CYCLE, measured_peak_m_s=3.1)
        assert rec["bound_m_s"] == pytest.approx(bound_at_fraction(1.0), rel=1e-9)
        assert rec["inside_bound"] is True

    def test_a_speed_above_the_bound_is_flagged_not_hidden(self):
        """Above the bound means the asset is still wrong, not that the arm is fast."""
        rec = experiment_record(limit_variant="real", torque_convention="rated",
                                duty_cycle=DEFAULT_DUTY_CYCLE, measured_peak_m_s=99.0)
        assert rec["inside_bound"] is False
        assert "asset" in rec["verdict"].lower() or "wrong" in rec["verdict"].lower()

    def test_the_record_lists_the_known_optimistic_biases(self):
        rec = experiment_record(limit_variant="real", torque_convention="rated",
                                duty_cycle=DEFAULT_DUTY_CYCLE, measured_peak_m_s=3.1)
        joined = " ".join(rec["known_biases"]).lower()
        assert "mass" in joined
        assert "racket" in joined

    def test_the_record_names_the_rackets_unknown_inertia(self):
        rec = experiment_record(limit_variant="real", torque_convention="rated",
                                duty_cycle=DEFAULT_DUTY_CYCLE, measured_peak_m_s=3.1)
        assert "UNKNOWN" in rec["racket_mass_status"]
