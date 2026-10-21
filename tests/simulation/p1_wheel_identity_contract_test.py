"""Shape/unit validation for the Morph One wheel identity contract (P1-09).

The contract is the machine-readable gate P4 reads.  These tests pin the two
things that must never regress:

  * schema/shape/units of wheel_identity_contract.json;
  * the anti-promotion rule - a candidate radius may not be frozen, and every
    UNKNOWN must name the concrete missing measurement.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "simulation", "p1_wheel_identity_contract.py")
EVIDENCE_DIR = os.path.join(
    REPO_ROOT, "outputs", "simulation", "rebot_b601_rs", "morph_one", "p1_wheel_identity"
)
COMMITTED = os.path.join(EVIDENCE_DIR, "wheel_identity_contract.json")


def load_module():
    spec = importlib.util.spec_from_file_location("p1_wheel_identity_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    if not os.path.isfile(SCRIPT):
        pytest.skip("contract script missing")
    return load_module()


@pytest.fixture(scope="module")
def contract(module):
    inputs, provenance = module.load_inputs(EVIDENCE_DIR)
    if inputs["comparison"] is None:
        pytest.skip("recorded comparison.json missing")
    built = module.build(EVIDENCE_DIR, inputs, provenance)
    module.validate(built)
    return built


def test_units_and_frames(contract):
    assert contract["units"] == {"length": "mm", "angle": "deg"}
    assert contract["frames"]["chassis_frame"] == "NOT FROZEN", "the chassis frame must stay UNKNOWN"
    assert contract["frames"]["usd_stage_up_axis"] == "Z", "recorded stage upAxis is Z and must be cited as-is"


def test_four_wheels_with_positive_geometry(contract):
    assert set(contract["wheels"]) == {"435700", "759491", "773650", "871375"}
    for eid, block in contract["wheels"].items():
        size = block["bbox_mm"]["usd_point_cloud_bbox_size"]
        assert all(value and value > 0 for value in size), "%s: bbox must be positive mm" % eid
        assert abs(size[0] - 48.791061) < 1e-3, "%s: thin axis changed" % eid
        radial = block["radial_extent_mm"]
        assert radial["r_max_mm"] > radial["r_min_mm"] > 0, "%s: radial ordering" % eid
        assert radial["outer_diameter_mm"] == pytest.approx(2.0 * radial["r_max_mm"], rel=1e-9)
        assert 0.0 <= block["principal_axes"][
            "angle_smallest_extent_axis_to_nearest_cad_axis_deg"
        ] <= 1.0, "%s: smallest-extent axis must be within 1 deg of a CAD axis" % eid


def test_radius_is_never_promoted(contract):
    radius = contract["quantities"]["wheel_radius"]
    assert radius["verdict"] == "UNKNOWN"
    assert radius["candidate"]["label"] == "max circular feature radius candidate"
    assert "missing_measurement" in radius and radius["missing_measurement"]
    text = json.dumps(radius["forbidden"], ensure_ascii=False)
    assert "206" in text and "diameter" in text, "the 206 mm diameter must stay explicitly forbidden"


def test_unknowns_carry_a_missing_measurement(contract):
    for name, quantity in contract["quantities"].items():
        assert quantity["verdict"] in {"PASS", "PARTIAL", "FAIL", "UNKNOWN"}
        if quantity["verdict"] == "UNKNOWN":
            assert quantity.get("missing_measurement"), name


def test_proxy_box_gap_is_thirteen_millimetres(contract):
    height = contract["quantities"]["loaded_rolling_height"]
    assert height["verdict"] == "UNKNOWN"
    assert height["candidate_rejected"]["value_m"] == pytest.approx(0.334)
    assert height["proxy_vs_feature_delta_mm"] == pytest.approx(13.0, abs=1e-6)
    assert "13 mm" in height["delta_statement"]


def test_p4_usage_contract(contract):
    usage = contract["p4_usage_contract"]
    assert usage["may_consume_today_parameterised_only"], "P4 must be told what it may consume"
    assert usage["forbidden_as_frozen_truth"], "P4 must be told what stays forbidden"
    forbidden = " ".join(usage["forbidden_as_frozen_truth"])
    for needle in ("0.334", "0.090", "206", "inertia", "upAxis"):
        assert needle in forbidden, "forbidden list must mention %s" % needle


def test_committed_contract_is_not_stale():
    """The committed JSON must equal a fresh build (regression against hand edits)."""
    if not os.path.isfile(COMMITTED):
        pytest.skip("contract JSON not built yet")
    completed = subprocess.run(
        [sys.executable, SCRIPT, "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, "contract JSON is stale:\n%s\n%s" % (
        completed.stdout,
        completed.stderr,
    )
