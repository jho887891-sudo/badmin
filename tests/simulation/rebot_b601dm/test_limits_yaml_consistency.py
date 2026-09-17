"""The generated YAML and the Python table must not drift apart."""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from src.simulation.rebot_b601dm.joint_limits import JOINT_LIMITS, MOTOR_ASSIGNMENT, Provenance

YAML_PATH = Path("configs/simulation/rebot_b601dm_joint_limits.yaml")


@pytest.fixture(scope="module")
def cfg():
    if not YAML_PATH.is_file():
        pytest.skip("generated YAML not present; run tools/generate_b601dm_limits_yaml.py")
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def test_yaml_declares_the_candidate_identity(cfg):
    """Task 0 outcome: PiPER keeps the project identity while this arm is evaluated."""
    assert cfg["identity"] == "test_bench_candidate_arm"


def test_every_joint_in_the_table_is_in_the_yaml(cfg):
    assert set(cfg["joints"]) == set(JOINT_LIMITS)


def test_motor_assignment_agrees(cfg):
    for name, spec in JOINT_LIMITS.items():
        assert cfg["joints"][name]["motor"] == MOTOR_ASSIGNMENT[name]


def test_values_and_provenance_agree(cfg):
    for name, spec in JOINT_LIMITS.items():
        for kind in ("angle", "velocity", "torque"):
            limit = getattr(spec, kind)
            block = cfg["joints"][name][kind]
            assert block["provenance"] == limit.provenance.value, f"{name}.{kind}"
            if limit.value is None:
                assert block["value"] is None, f"{name}.{kind}"
            else:
                assert block["value"] == pytest.approx(limit.value, rel=1e-6), f"{name}.{kind}"


def test_sim_values_are_preserved_not_overwritten(cfg):
    """The shipped numbers must stay visible; that is how the 9x error was found."""
    for name in ("joint1", "joint4"):
        spec = JOINT_LIMITS[name]
        block = cfg["joints"][name]["velocity"]
        assert block["sim_value"] == pytest.approx(spec.velocity.sim_value)
        assert block["sim_overshoot"] == pytest.approx(spec.velocity.sim_overshoot, rel=1e-3)
        assert block["value"] != block["sim_value"]


def test_the_yaml_names_the_unknowns(cfg):
    unknown = [n for n, s in JOINT_LIMITS.items() if s.angle.provenance == Provenance.UNKNOWN]
    assert len(unknown) >= 6, "the six arm joints have no hardware angle source"
    for name in ("joint1", "joint4"):
        assert cfg["joints"][name]["angle"]["provenance"] == "UNKNOWN"