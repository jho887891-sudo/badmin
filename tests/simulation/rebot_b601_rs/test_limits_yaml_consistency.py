# -*- coding: utf-8 -*-
"""The generated YAML and the Python table must not drift apart."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.simulation.rebot_b601_rs import joint_limits as jl  # noqa: E402

YAML_PATH = ROOT / "configs" / "simulation" / "rebot_b601_rs_joint_limits.yaml"


@pytest.fixture(scope="module")
def cfg():
    if not YAML_PATH.is_file():
        pytest.skip("generated YAML not present; run tools/generate_rs_limits_yaml.py")
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def test_yaml_declares_the_formal_identity(cfg):
    assert cfg["identity"] == "formal_arm"
    assert cfg["model"] == jl.MODEL


def test_every_joint_in_the_table_is_in_the_yaml(cfg):
    assert set(cfg["joints"]) == set(jl.JOINT_LIMITS)


def test_motor_assignment_agrees(cfg):
    for name, spec in jl.JOINT_LIMITS.items():
        assert cfg["joints"][name]["motor"] == jl.MOTOR_ASSIGNMENT[name]


def test_values_and_provenance_agree(cfg):
    for name, spec in jl.JOINT_LIMITS.items():
        for kind in ("angle", "velocity", "torque"):
            limit = getattr(spec, kind)
            block = cfg["joints"][name][kind]
            assert block["provenance"] == limit.provenance.value, name + "." + kind
            assert block["usable_for_capability"] == limit.usable_for_capability, name + "." + kind
            if limit.value is None:
                assert "value" not in block, name + "." + kind
            else:
                assert block["value"] == pytest.approx(limit.value, rel=1e-9), name + "." + kind


def test_urdf_numbers_are_preserved_verbatim(cfg):
    for name, spec in jl.JOINT_LIMITS.items():
        assert cfg["joints"][name]["urdf_velocity"] == pytest.approx(spec.urdf_velocity)
        assert cfg["joints"][name]["urdf_effort"] == pytest.approx(spec.urdf_effort)


def test_asset_pin_is_in_the_yaml(cfg):
    assert cfg["asset"]["commit"] == jl.ASSET["commit"]
    assert cfg["asset"]["urdf_sha256"] == jl.ASSET["urdf_sha256"]
