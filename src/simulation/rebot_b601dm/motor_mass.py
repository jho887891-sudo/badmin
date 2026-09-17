"""Motor masses in the B601-DM model: quantified, and deliberately NOT substituted.

FINDING. The CAD export gives motor3, motor4, motor5 and motor7 a mass of 99.0 g, identical to seven digits
across two different motor models, and gives motor2 256.9 g. The official masses are ~362 g for the DM4340
and ~300 g for the DM4310. motor1 and motor6 do not appear at all.

Quantified, the modelled arm is **2.7445 kg** against a specified **4.5 kg** (README_zh.md:177), and the motor
deficit of 1633 g explains **93%** of the missing 1755 g.

WHY SUBSTITUTION WAS REFUSED, which is the point of this module. The obvious fix is to write 362 and 300 into
the five rows that exist. That would be wrong: the inertia tensors were computed for 99 g. Changing mass
alone produces a model whose mass and inertia disagree, corresponding to no real object - and for a dynamics
measurement that is WORSE than a uniformly light arm, because a uniformly light arm at least behaves like
something, while an inconsistent one behaves like nothing.

So the values stay as they are, tagged, with the error direction stated: an under-massed arm swings faster
than the real one, so a racket-head speed measured now would be **optimistic**.
"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

URDF_REL = "_scratch_rebot/reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.urdf"
CSV_REL = "_scratch_rebot/reBot-Isaacsim/urdf/reBot_B601_DM/urdf/reBot_B601_DM.csv"
REPO_REL = "_scratch_rebot/reBot-Isaacsim"

# Measured from the CAD export. Kept here so the audit is reproducible without the repository present.
CAD_MOTOR_MASSES_G = {
    "motor2": 256.880819154806,
    "motor3": 99.031978192153,
    "motor4": 99.0319853065598,
    "motor5": 99.0319769260336,
    "motor7": 99.0319790497617,
}

# SeeedStudio damiao_series table, Weight (g) column.
OFFICIAL_MOTOR_MASSES_G = {"DM4340P": 362.0, "DM4310": 300.0}

# BOM assignment: reBot-DevArm/hardware/reBot_B601_DM/readme.md:141-142 lists 3x DM4340P and 4x DM4310.
# The CSV index matches the URDF link index, verified by summing each group against the link mass.
MOTOR_ASSIGNMENT = {
    "motor1": "DM4340P",   # absent from the CSV
    "motor2": "DM4340P",
    "motor3": "DM4340P",
    "motor4": "DM4310",
    "motor5": "DM4310",
    "motor6": "DM4310",    # absent from the CSV
    "motor7": "DM4310",    # in gripper_link
}
MISSING_MOTORS = {"motor1": "DM4340P", "motor6": "DM4310"}

SPECIFIED_ARM_MASS_KG = 4.5  # README_zh.md:177, "约 4.5 kg"


def _urdf_path() -> Path:
    return Path(URDF_REL)


def modelled_arm_mass_kg() -> float:
    """Sum of the URDF link masses."""
    root = ET.parse(_urdf_path()).getroot()
    total = 0.0
    for link in root.findall("link"):
        ine = link.find("inertial")
        m = ine.find("mass") if ine is not None else None
        if m is not None:
            total += float(m.get("value"))
    return total


def specified_arm_mass_kg() -> float:
    return SPECIFIED_ARM_MASS_KG


def motor_mass_audit() -> dict:
    """Everything known about the motor masses, and the explicit refusal to guess the rest."""
    modelled_motors_g = sum(CAD_MOTOR_MASSES_G.values())
    real_motors_g = sum(OFFICIAL_MOTOR_MASSES_G[m] for m in MOTOR_ASSIGNMENT.values())
    arm_deficit_g = (SPECIFIED_ARM_MASS_KG - modelled_arm_mass_kg()) * 1000.0
    motor_deficit_g = real_motors_g - modelled_motors_g

    rows = {}
    for name, model in MOTOR_ASSIGNMENT.items():
        want = OFFICIAL_MOTOR_MASSES_G[model]
        have = CAD_MOTOR_MASSES_G.get(name)
        rows[name] = {
            "model": model,
            "modelled_g": have,
            "official_g": want,
            "ratio": (have / want) if have is not None else 0.0,
            "status": "UNKNOWN" if have is None else "TEMP_PARAMETERIZED_PROXY",
            "source": ("absent from " + CSV_REL) if have is None
                      else (CSV_REL + " (CAD export)"),
        }

    return {
        "rows": rows,
        "modelled_motors_g": modelled_motors_g,
        "real_motors_g": real_motors_g,
        "motor_deficit_g": motor_deficit_g,
        "modelled_arm_mass_kg": modelled_arm_mass_kg(),
        "specified_arm_mass_kg": SPECIFIED_ARM_MASS_KG,
        "arm_deficit_g": arm_deficit_g,
        "explained_fraction": motor_deficit_g / arm_deficit_g,
        "substituted": False,
        "refusal_reason": (
            "the inertia tensors were computed for the modelled masses, so writing the official mass into a "
            "row would leave mass and inertia disagreeing. A model that is uniformly light behaves like a "
            "lighter object; a model with a tripled mass and unchanged inertia behaves like no object at all, "
            "which is worse for a dynamics measurement."
        ),
        "requires": [
            "per-motor rigid-body mass AND inertia tensors from the vendor CAD, so both can be corrected together",
            "the mounting position and orientation of motor1 and motor6, which are absent from the export",
            "a decision on whether the official actuator weight includes the driver board and cabling",
        ],
        "error_direction": (
            "the modelled arm is lighter than the real one, and lighter distal links accelerate faster under "
            "the same torque, so a racket-head speed measured on this asset would be OPTIMISTIC - it would "
            "overstate what the real arm can do."
        ),
    }
