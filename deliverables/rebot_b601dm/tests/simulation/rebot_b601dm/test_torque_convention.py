"""Task 3: the torque convention must be declared, and the default must be the conservative one.

The trap this closes: 27 and 7 N.m are the PEAK torques of the DM4340 and DM4310, while the rated figures are
9 and 3 N.m. A swing that quietly assumed peak would report a capability roughly three times what the motor
can sustain - and the official thermal testing says this arm cannot sustain load at all, terminating every
run on motor-2 overheating.

Two different places, deliberately handled differently:
  - `resolve_torque_limit` REFUSES to guess. Passing None is an error, not a default.
  - An experiment carries a convention that defaults to RATED, and that convention is part of the experiment
    record, so a reader can always tell which one produced a number.
"""
from __future__ import annotations

import pytest

from src.simulation.rebot_b601dm.joint_limits import Provenance
from src.simulation.rebot_b601dm.torque_convention import (
    DEFAULT_TORQUE_CONVENTION,
    ExperimentTorque,
    TorqueConvention,
    resolve_torque_limit,
)


class TestTheResolverRefusesToGuess:
    def test_an_unlabelled_call_is_refused(self):
        with pytest.raises(ValueError, match="convention"):
            resolve_torque_limit("joint1", convention=None)

    def test_an_unknown_convention_is_refused(self):
        with pytest.raises(ValueError):
            resolve_torque_limit("joint1", convention="whenever")

    def test_a_joint_with_no_real_torque_is_refused(self):
        """The gripper force has no hardware source; returning a number would be invention."""
        with pytest.raises(ValueError):
            resolve_torque_limit("gripper_joint1", convention=TorqueConvention.RATED)


class TestTheTwoValuesAreKeptApart:
    def test_peak_returns_the_peak_torque(self):
        assert resolve_torque_limit("joint1", convention="peak") == pytest.approx(27.0)
        assert resolve_torque_limit("joint4", convention="peak") == pytest.approx(7.0)

    def test_rated_returns_the_rated_torque(self):
        assert resolve_torque_limit("joint1", convention="rated") == pytest.approx(9.0)
        assert resolve_torque_limit("joint4", convention="rated") == pytest.approx(3.0)

    def test_peak_is_three_times_rated_for_both_motors(self):
        """If this ratio ever changes, the table or the convention enum has been mis-edited."""
        for joint in ("joint1", "joint2", "joint3"):
            assert resolve_torque_limit(joint, convention="peak") == pytest.approx(
                3 * resolve_torque_limit(joint, convention="rated")
            )
        assert resolve_torque_limit("joint4", convention="peak") == pytest.approx(
            7.0 / 3.0 * resolve_torque_limit("joint4", convention="rated")
        )

    def test_strings_and_enum_members_are_interchangeable(self):
        assert resolve_torque_limit("joint1", convention="peak") == resolve_torque_limit(
            "joint1", convention=TorqueConvention.PEAK
        )


class TestTheProjectDefaultIsTheSafeOne:
    def test_the_default_convention_is_rated_not_peak(self):
        assert DEFAULT_TORQUE_CONVENTION is TorqueConvention.RATED

    def test_an_experiment_that_says_nothing_gets_rated(self):
        exp = ExperimentTorque()
        assert exp.convention is TorqueConvention.RATED
        assert exp.limit_for("joint1") == pytest.approx(9.0)

    def test_an_experiment_can_choose_peak_and_that_choice_is_visible(self):
        exp = ExperimentTorque(convention="peak")
        assert exp.limit_for("joint1") == pytest.approx(27.0)
        assert exp.convention is TorqueConvention.PEAK


class TestTheConventionIsPartOfTheRecord:
    def test_the_experiment_record_names_the_convention(self):
        """A measured speed with no convention attached cannot be interpreted."""
        record = ExperimentTorque(convention="peak").as_record()
        assert record["torque_convention"] == "peak"
        assert "peak_torque_nm" in record or "torque_nm" in record

    def test_the_record_carries_the_reason_the_choice_matters(self):
        record = ExperimentTorque(convention="rated").as_record()
        assert "peak" in record["note"].lower()
        assert "rated" in record["note"].lower()

    def test_the_gripper_is_listed_as_unresolved_rather_than_omitted(self):
        record = ExperimentTorque(convention="rated").as_record()
        assert "gripper_joint1" in record["unresolved"]