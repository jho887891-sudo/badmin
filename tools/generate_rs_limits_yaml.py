# -*- coding: utf-8 -*-
"""Render the B601-RS joint-limit table to YAML so both copies can be diffed by eye.

GENERATED: edit src/simulation/rebot_b601_rs/joint_limits.py and re-run this script.
tests/simulation/rebot_b601_rs/test_limits_yaml_consistency.py fails if the two drift apart.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.rebot_b601_rs import joint_limits as jl  # noqa: E402

OUT = ROOT / "configs" / "simulation" / "rebot_b601_rs_joint_limits.yaml"


def _limit_block(limit, indent: str) -> list[str]:
    lines = [indent + "provenance: " + limit.provenance.value,
             indent + "usable_for_capability: " + ("true" if limit.usable_for_capability else "false"),
             indent + "source: " + repr(limit.source)]
    if limit.lower is not None:
        lines.append(indent + "lower: " + repr(limit.lower))
    if limit.upper is not None:
        lines.append(indent + "upper: " + repr(limit.upper))
    if limit.value is not None:
        lines.append(indent + "value: " + repr(limit.value))
    return lines


def render() -> str:
    out: list[str] = [
        "# reBot B601-RS joint limits - the FORMAL arm of this project since the 2026-10 migration.",
        "#",
        "# GENERATED from src/simulation/rebot_b601_rs/joint_limits.py; do not hand-edit.",
        "#",
        "# Every value carries provenance. OFFICIAL_URDF means: real in the official model, NOT a",
        "# hardware measurement, so it may not back a capability claim. CONFIRMED_REAL means an",
        "# official Seeed product page states it and the URDF agrees.",
        "#",
        "# This table is NOT the B601-DM table and NOT the PiPER table: different motors, different",
        "# joint ranges, different masses. Cross-arm copying is a defect, not a shortcut.",
        "",
        "robot: " + jl.ROBOT_NAME,
        "model: " + repr(jl.MODEL),
        "identity: " + jl.IDENTITY,
        "gripper_fingers_hardware_coupled: " + ("true" if jl.GRIPPER_FINGERS_HARDWARE_COUPLED else "false"),
        "",
        "spec:",
    ]
    for k, v in jl.SPEC.items():
        out.append("  " + k + ": " + repr(v))
    out += ["", "asset:"]
    for k, v in jl.ASSET.items():
        out.append("  " + k + ": " + repr(v))
    out += ["", "motors:"]
    for name, motor in jl.MOTORS.items():
        out += ["  " + name + ":",
                "    model: " + repr(motor.model),
                "    peak_torque_nm: " + repr(motor.peak_torque_nm),
                "    reduction_ratio: " + repr(motor.reduction_ratio),
                "    provenance: " + motor.provenance.value,
                "    source: " + repr(motor.source)]
    out += ["", "link_masses_kg:"]
    for k, v in jl.LINK_MASSES_KG.items():
        out.append("  " + k + ": " + repr(v))
    out += ["", "joints:"]
    for name in jl.JOINT_ORDER:
        spec = jl.JOINT_LIMITS[name]
        out += ["  " + name + ":",
                "    kind: " + spec.kind,
                "    motor: " + spec.motor,
                "    urdf_velocity: " + repr(spec.urdf_velocity),
                "    urdf_effort: " + repr(spec.urdf_effort),
                "    angle:"]
        out += _limit_block(spec.angle, "      ")
        out += ["    velocity:"]
        out += _limit_block(spec.velocity, "      ")
        out += ["    torque:"]
        out += _limit_block(spec.torque, "      ")
    out.append("")
    return "\n".join(out)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(), encoding="utf-8")
    print("wrote " + str(OUT) + " (" + str(len(render().splitlines())) + " lines)")


if __name__ == "__main__":
    main()
