"""Task 1: the authoritative B601-DM joint-limit table must carry provenance for every value.

Why this test exists at all: the inspection found the shipped simulation asset enforcing joint speeds 9.09x
and 9.55x the real motors, because the real rpm figures were used as rad/s. A project that can make that
mistake needs the provenance of each number to be structural, not a comment someone might read.
"""
from __future__ import annotations

import math

import pytest

from src.simulation.rebot_b601dm.joint_limits import (
    JOINT_LIMITS,
    MOTOR_ASSIGNMENT,
    Provenance,
    limits_for,
    rad_s_from_rpm,
)


class TestProvenanceIsStructural:
    def test_every_revolute_joint_declares_a_provenance_for_each_limit_kind(self):
        for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            spec = JOINT_LIMITS[name]
            assert spec.velocity.provenance in set(Provenance)
            assert spec.torque.provenance in set(Provenance)
            assert spec.angle.provenance in set(Provenance)

    def test_every_claimed_provenance_carries_a_source_string(self):
        for name, spec in JOINT_LIMITS.items():
            for kind in ("angle", "velocity", "torque"):
                limit = getattr(spec, kind)
                if limit.provenance != Provenance.UNKNOWN:
                    assert limit.source, f"{name}.{kind} claims {limit.provenance} with no source"

    def test_sim_only_values_are_never_claimable_as_real(self):
        """The whole point: a SIM_ONLY number must not be usable as a capability claim."""
        for name, spec in JOINT_LIMITS.items():
            if spec.velocity.provenance == Provenance.SIM_ONLY:
                with pytest.raises(ValueError):
                    limits_for(name, require_real=True)


class TestTheInspectionFindingsAreEncoded:
    def test_arm_joints_use_the_motors_the_bom_specifies(self):
        assert MOTOR_ASSIGNMENT["joint1"] == "DM4340P"
        assert MOTOR_ASSIGNMENT["joint2"] == "DM4340P"
        assert MOTOR_ASSIGNMENT["joint3"] == "DM4340P"
        for name in ("joint4", "joint5", "joint6"):
            assert MOTOR_ASSIGNMENT[name] == "DM4310"

    def test_real_velocities_are_the_motor_no_load_speeds_and_are_not_the_sim_values(self):
        """DM4340 52.5 rpm and DM4310 200 rpm, against sim values of 50 and 200 rad/s."""
        for name in ("joint1", "joint2", "joint3"):
            spec = JOINT_LIMITS[name]
            assert spec.velocity.value == pytest.approx(rad_s_from_rpm(52.5), rel=1e-9)
            assert spec.velocity.value == pytest.approx(5.498, rel=1e-3)
            assert spec.velocity.sim_value == pytest.approx(50.0)
        for name in ("joint4", "joint5", "joint6"):
            spec = JOINT_LIMITS[name]
            assert spec.velocity.value == pytest.approx(rad_s_from_rpm(200.0), rel=1e-9)
            assert spec.velocity.value == pytest.approx(20.944, rel=1e-3)
            assert spec.velocity.sim_value == pytest.approx(200.0)

    def test_the_sim_overshoot_is_recorded_and_is_about_nine_times(self):
        for name in ("joint1", "joint2", "joint3"):
            assert JOINT_LIMITS[name].velocity.sim_overshoot == pytest.approx(9.09, rel=0.02)
        for name in ("joint4", "joint5", "joint6"):
            assert JOINT_LIMITS[name].velocity.sim_overshoot == pytest.approx(9.55, rel=0.02)

    def test_torque_is_peak_and_rated_torque_is_kept_separately(self):
        """27 and 7 N.m are PEAK. Rated are 9 and 3. Conflating them is the trap."""
        for name in ("joint1", "joint2", "joint3"):
            assert JOINT_LIMITS[name].torque.value == pytest.approx(27.0)
            assert JOINT_LIMITS[name].torque.rated == pytest.approx(9.0)
            assert JOINT_LIMITS[name].torque.is_peak is True
        for name in ("joint4", "joint5", "joint6"):
            assert JOINT_LIMITS[name].torque.value == pytest.approx(7.0)
            assert JOINT_LIMITS[name].torque.rated == pytest.approx(3.0)

    def test_reduction_ratios_match_the_official_table(self):
        assert JOINT_LIMITS["joint1"].reduction_ratio == 40
        assert JOINT_LIMITS["joint4"].reduction_ratio == 10


class TestAngleLimitsStayUnknown:
    def test_angle_limits_are_unknown_because_no_hardware_source_exists(self):
        """Neither repository states a joint angle range. If someone later proves one, this test changes with
        a source; until then the honest tag is UNKNOWN."""
        for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            assert JOINT_LIMITS[name].angle.provenance == Provenance.UNKNOWN

    def test_the_angle_values_are_still_recorded_from_the_urdf(self):
        assert JOINT_LIMITS["joint1"].angle.lower == pytest.approx(-2.8)
        assert JOINT_LIMITS["joint1"].angle.upper == pytest.approx(2.8)
        assert JOINT_LIMITS["joint2"].angle.upper == pytest.approx(0.0)
        assert JOINT_LIMITS["joint4"].angle.lower == pytest.approx(-1.87)
        assert JOINT_LIMITS["joint6"].angle.upper == pytest.approx(3.14)


class TestGripperIsRecordedHonestly:
    def test_gripper_force_is_unknown_and_marked_so(self):
        spec = JOINT_LIMITS["gripper_joint1"]
        assert spec.torque.provenance == Provenance.UNKNOWN
        assert MOTOR_ASSIGNMENT["gripper_joint1"] == "DM4310"


class TestValuesMatchTheShippedArtifacts:
    """Stronger than transcribing values into a table: read them back out of the actual repositories.

    These tests skip when the repositories are absent so the suite still runs on a machine that has not
    downloaded them, but on a machine that HAS them the table cannot silently drift from the assets.
    """

    @staticmethod
    def _repo(name):
        from pathlib import Path
        for base in (Path("_scratch_rebot"), Path(".")):
            p = base / name
            if p.is_dir():
                return p
        return None

    def test_sim_velocity_values_match_physx_usda(self):
        repo = self._repo("reBot-Isaacsim")
        if repo is None:
            pytest.skip("reBot-Isaacsim not present locally")
        import re
        text = (repo / "usd/reBot_B601_DM/payloads/Physics/physx.usda").read_text(errors="replace")
        degrees = sorted({float(m) for m in re.findall(r"maxJointVelocity = ([0-9.]+)", text)})
        expected_deg = sorted({
            50.0 * 180.0 / math.pi,      # joint1-3 shipped sim value
            200.0 * 180.0 / math.pi,     # joint4-6 shipped sim value
            15.0 * 180.0 / math.pi,      # gripper
        })
        for got, want in zip(degrees, expected_deg):
            assert got == pytest.approx(want, rel=1e-5), f"asset says {got} deg/s, table expects {want}"

    def test_angle_values_match_the_urdf(self):
        repo = self._repo("reBot-Isaacsim")
        if repo is None:
            pytest.skip("reBot-Isaacsim not present locally")
        import xml.etree.ElementTree as ET
        root = ET.parse(repo / "urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf").getroot()
        got = {}
        for j in root.findall("joint"):
            lim = j.find("limit")
            if lim is not None and lim.get("lower") is not None:
                got[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            assert got[name][0] == pytest.approx(JOINT_LIMITS[name].angle.lower), name
            assert got[name][1] == pytest.approx(JOINT_LIMITS[name].angle.upper), name

    def test_torque_values_match_the_urdf_effort(self):
        repo = self._repo("reBot-Isaacsim")
        if repo is None:
            pytest.skip("reBot-Isaacsim not present locally")
        import xml.etree.ElementTree as ET
        root = ET.parse(repo / "urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf").getroot()
        for j in root.findall("joint"):
            name = j.get("name")
            lim = j.find("limit")
            if name in JOINT_LIMITS and lim is not None and lim.get("effort"):
                assert float(lim.get("effort")) == pytest.approx(JOINT_LIMITS[name].torque.sim_value), name


class TestStrictModeRefusesPartlyRealJoints:
    def test_arm_joints_are_refused_because_their_angles_are_unknown(self):
        """The honest state, enforced: the velocities and torques are real, the angle ranges are not."""
        for name in ("joint1", "joint4"):
            with pytest.raises(ValueError, match="angle"):
                limits_for(name, require_real=True)

    def test_a_hypothetical_fully_real_joint_is_returned(self):
        spec = limits_for("joint1")
        assert spec.velocity.value == pytest.approx(5.498, rel=1e-3)
        assert spec.velocity.provenance == Provenance.CONFIRMED_REAL


def test_rad_s_from_rpm_converts_correctly():
    assert rad_s_from_rpm(60.0) == pytest.approx(2 * math.pi, rel=1e-12)
    assert rad_s_from_rpm(0.0) == pytest.approx(0.0)