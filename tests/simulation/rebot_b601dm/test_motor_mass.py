"""Task 4: the motor masses are TEMP, and the reason they were NOT substituted.

The CAD export gives motor3, motor4, motor5 and motor7 a mass of 99.0 g, identical to seven digits across two
different motor models, and assigns motor2 256.9 g. The official masses are ~362 g for the DM4340 and ~300 g
for the DM4310, so the wrist motors are modelled at about a third of their real mass - and motor1 and motor6
are absent from the export entirely.

The obvious move is to substitute the official figures. This file records why that was NOT done: the inertia
tensors were computed for 99 g, so changing mass alone would produce a model whose mass and inertia disagree,
which corresponds to no real object and is worse for a dynamics measurement than a uniformly light one.
"""
from __future__ import annotations

import pytest

from src.simulation.rebot_b601dm.motor_mass import (
    CAD_MOTOR_MASSES_G,
    OFFICIAL_MOTOR_MASSES_G,
    MISSING_MOTORS,
    modelled_arm_mass_kg,
    motor_mass_audit,
    specified_arm_mass_kg,
)

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("_scratch_rebot/reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.csv").is_file(),
    reason="reBot-Isaacsim not present locally",
)


class TestThePlaceholderIsQuantifiedNotGuessed:
    def test_four_motor_rows_share_one_mass_to_seven_digits(self):
        # the values are 99.0319... g, printed as 99.0 by %.1f; the tolerance has to allow for that
        shared = {n: m for n, m in CAD_MOTOR_MASSES_G.items() if m == pytest.approx(99.03, abs=0.05)}
        assert set(shared) == {"motor3", "motor4", "motor5", "motor7"}

    def test_motor2_is_different_and_closer_to_a_real_motor(self):
        assert CAD_MOTOR_MASSES_G["motor2"] == pytest.approx(256.88, abs=0.1)
        assert CAD_MOTOR_MASSES_G["motor2"] > CAD_MOTOR_MASSES_G["motor3"] * 2

    def test_the_official_masses_are_recorded(self):
        assert OFFICIAL_MOTOR_MASSES_G["DM4340P"] == pytest.approx(362.0)
        assert OFFICIAL_MOTOR_MASSES_G["DM4310"] == pytest.approx(300.0)

    def test_two_motors_are_absent_from_the_export(self):
        assert set(MISSING_MOTORS) == {"motor1", "motor6"}
        assert MISSING_MOTORS["motor1"] == "DM4340P"
        assert MISSING_MOTORS["motor6"] == "DM4310"


class TestTheDeficitIsQuantified:
    def test_the_modelled_arm_is_substantially_lighter_than_specified(self):
        modelled = modelled_arm_mass_kg()
        specified = specified_arm_mass_kg()
        assert modelled == pytest.approx(2.7445, abs=0.01)
        assert specified == pytest.approx(4.5, abs=0.01)
        assert specified - modelled == pytest.approx(1.7555, abs=0.02)

    def test_the_motors_explain_almost_all_of_it(self):
        audit = motor_mass_audit()
        assert audit["explained_fraction"] > 0.90, "the motors should account for most of the deficit"

    def test_the_wrist_motors_are_the_worst_off(self):
        audit = motor_mass_audit()
        for row in ("motor4", "motor5", "motor7"):
            assert audit["rows"][row]["ratio"] < 0.40, row


class TestSubstitutionWasRefusedAndTheReasonIsRecorded:
    def test_the_audit_refuses_to_produce_substituted_masses(self):
        """Mass without matching inertia would be an inconsistent model, so no substitution is offered."""
        audit = motor_mass_audit()
        assert audit["substituted"] is False
        assert "inertia" in audit["refusal_reason"].lower()

    def test_the_audit_names_what_would_be_needed_to_fix_it(self):
        audit = motor_mass_audit()
        assert audit["requires"]
        joined = " ".join(audit["requires"]).lower()
        assert "inertia" in joined
        assert "cad" in joined or "mass" in joined

    def test_every_row_carries_a_status(self):
        audit = motor_mass_audit()
        for row, info in audit["rows"].items():
            assert info["status"] in {"TEMP_PARAMETERIZED_PROXY", "UNKNOWN"}
            assert info["source"]


class TestTheDirectionOfTheErrorIsStated:
    def test_the_audit_says_which_way_a_speed_measurement_would_be_wrong(self):
        """An under-massed arm swings faster than the real one, so a measured speed is optimistic."""
        audit = motor_mass_audit()
        assert "optimistic" in audit["error_direction"].lower()


def test_temp_readme_is_current():
    """A hand-edited or stale TEMP_README would silently misstate what is TEMP."""
    import subprocess
    import sys
    from pathlib import Path

    path = Path("management/rebot_b601dm/TEMP_README.md")
    if not path.is_file():
        pytest.skip("TEMP_README not generated yet")
    before = path.read_text(encoding="utf-8")
    subprocess.run(
        [sys.executable, "tools/generate_b601dm_temp_readme.py"],
        check=True, capture_output=True,
    )
    after = path.read_text(encoding="utf-8")
    assert before == after, "TEMP_README.md is stale; regenerate it"
