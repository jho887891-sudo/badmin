"""Schema, unit and residual-bound tests for the P1 wheel geometry v2 record.

Pure python: no pxr, no numpy, no CAD kernel, no network.  The checks that
matter are (a) the record says which fields are verified vs candidate vs
UNKNOWN, (b) the chassis-frame numbers really are the CAD numbers pushed
through the documented 0.001 / +90 deg x mapping, and (c) the residual bounds
the report claims actually hold.

Run:  python -m pytest tests/simulation/p1_wheel_geometry_v2_test.py -q
"""
from __future__ import annotations

import json
import math
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORD = os.path.join(
    REPO_ROOT,
    "outputs",
    "simulation",
    "rebot_b601_rs",
    "morph_one",
    "p1_wheel_identity",
    "wheel_geometry_v2.json",
)
EIDS = ("435700", "759491", "773650", "871375")
SCALE = 0.001
VERDICTS = {"verified", "candidate", "UNKNOWN"}


def rot_x(vec, degrees):
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    x, y, z = vec
    return [x, c * y - s * z, s * y + c * z]


@pytest.fixture(scope="module")
def record():
    with open(RECORD, encoding="utf-8") as handle:
        return json.load(handle)


def test_schema_and_frames(record):
    assert record["schema"] == "badmin.morph_one.wheel_geometry_v2/1"
    assert record["status_policy"]
    assert "cad" in record["frames"] and "chassis" in record["frames"]
    assert record["units"]["length"] == "mm"
    assert record["units"]["length_chassis"] == "m"
    assert record["units"]["area"] == "mm^2"


def test_every_field_is_labelled(record):
    """No bare number: every value carries verdict/unit/frame/residual."""
    checked = 0
    for eid in EIDS:
        wheel = record["wheels"][eid]
        for name in ("axle_line_cad", "axle_line_chassis", "outward_axle_direction", "rolling_surface"):
            field = wheel[name]
            assert field["verdict"] in VERDICTS, (eid, name, field.get("verdict"))
            assert field.get("residual") is not None, (eid, name)
            checked += 1
        assert wheel["axle_line_cad"]["unit"] == "mm"
        assert wheel["axle_line_chassis"]["unit"] == "m"
    assert checked == 4 * 4


def test_chassis_point_is_the_mapped_cad_point(record):
    """Independent re-derivation: chassis = R_x(90) . (0.001 * CAD mm)."""
    for eid in EIDS:
        wheel = record["wheels"][eid]
        cad = wheel["axle_line_cad"]["point_mm"]
        expected = rot_x([v * SCALE for v in cad], 90.0)
        got = wheel["axle_line_chassis"]["point_m"]
        for k in range(3):
            assert abs(expected[k] - got[k]) < 1e-9, (eid, k, expected, got)


def test_roundtrip_residual_is_zero(record):
    for eid in EIDS:
        residual = record["wheels"][eid]["axle_line_chassis"]["residual"]
        assert max(abs(v) for v in residual["transform_roundtrip_mm"]) < 1e-6
        assert residual["path_residual_mm"] < 0.01


def test_axle_direction_is_chassis_x_and_unsigned(record):
    for eid in EIDS:
        wheel = record["wheels"][eid]
        direction = wheel["axle_line_cad"]["direction_unit"]
        # parallel to x: the cross product with x-hat must vanish
        cross = [0.0, direction[2], -direction[1]]
        assert math.sqrt(sum(v * v for v in cross)) < 1e-6, (eid, direction)
        # the stored sign is not a spin direction; it flips on the left wheels
        stored = wheel["axle_line_cad"]["stored_brep_axis_sign"]
        side = math.copysign(1.0, wheel["axle_line_chassis"]["point_m"][0])
        assert stored * side > 0.0, (eid, stored, side)


def test_outward_direction_and_spin_sign_agree(record):
    for eid in EIDS:
        wheel = record["wheels"][eid]
        centre_x = wheel["axle_line_chassis"]["point_m"][0]
        outward = wheel["outward_axle_direction"]["direction_unit_chassis"]
        spin = wheel["outward_axle_direction"]["spin_sign"]
        assert spin in (-1, 1)
        assert math.copysign(1.0, outward[0]) == math.copysign(1.0, centre_x)
        assert abs(abs(outward[0]) - 1.0) < 1e-12
        assert abs(outward[1]) < 1e-12 and abs(outward[2]) < 1e-12
        assert spin == int(math.copysign(1.0, centre_x))


def test_track_and_wheelbase_match_the_point_set(record):
    xs = [record["wheels"][e]["axle_line_chassis"]["point_m"][0] for e in EIDS]
    ys = [record["wheels"][e]["axle_line_chassis"]["point_m"][1] for e in EIDS]
    aggregates = record["chassis_aggregates"]
    assert abs(aggregates["track_width"]["value_m"] - (max(xs) - min(xs))) < 1e-9
    assert abs(aggregates["wheelbase"]["value_m"] - (max(ys) - min(ys))) < 1e-9
    assert aggregates["track_width"]["value_m"] == pytest.approx(0.500004, abs=1e-6)
    assert aggregates["wheelbase"]["value_m"] == pytest.approx(0.594, abs=1e-6)
    for name in ("track_width", "wheelbase"):
        assert aggregates[name]["verdict"] == "candidate"
        assert aggregates[name]["uncertainty_m"] > 0.0


def test_wheel_centres_share_one_plane(record):
    zs = [record["wheels"][e]["axle_line_chassis"]["point_m"][2] for e in EIDS]
    assert max(zs) - min(zs) < 1e-9
    assert abs(min(zs) - (-0.244)) < 1e-6
    height = record["chassis_aggregates"]["unloaded_rolling_height_from_geometry"]
    assert abs(height["value_m"] - (-min(zs) + 0.103)) < 1e-9
    assert height["residual"]["difference_m"] == pytest.approx(0.013, abs=1e-9)
    assert height["verdict"] == "candidate"


def test_rolling_surface_is_a_torus_not_a_cylinder(record):
    for eid in EIDS:
        surface = record["wheels"][eid]["rolling_surface"]
        classification = surface["classification"]
        assert classification["verdict"] == "verified"
        assert "TOROIDAL_SURFACE" in classification["value"]
        residual = classification["residual"]
        # the equator of the torus reproduces the measured 103 mm circle
        assert abs(residual["equator_residual_mm"]) < 1e-6
        assert residual["equator_recompute_mm"] == pytest.approx(103.0, abs=1e-6)
        # and no cylinder of that radius exists in the solid
        assert residual["max_cylinder_radius_in_solid_mm"] < 92.5
        assert residual["cylinder_minus_103_mm"] <= -10.0
        tread = surface["tread_geometry"]
        assert tread["radial_band_mm"][1] == pytest.approx(103.0, abs=1e-6)
        assert tread["radial_band_mm"][0] == pytest.approx(92.5, abs=1e-6)
        assert tread["axial_band_width_mm"] == pytest.approx(41.0, abs=1e-6)
        assert tread["face_count"] == 4
        assert tread["area_mm2"] > 1000.0
        assert tread["minor_parameter_band_deg"][1] < 90.0
        rejected = [item["hypothesis"] for item in surface["rejected_hypotheses"]]
        assert any("adjacent cylinder" in text for text in rejected)


def test_face_group_table_covers_every_face(record):
    for eid in EIDS:
        table = record["wheels"][eid]["rolling_surface"]["face_group_table_mm2"]
        assert sum(row["face_count"] for row in table) == 190
        assert any(row["touches_r103_circle"] for row in table)
        assert all(row["axial_width"] if "axial_width" in row else True for row in table)


def test_mirrored_pairs_residual_bound(record):
    identity = record["part_identity"]
    assert "mirrored" in identity["verdict"]
    assert identity["max_mirror_band_residual_mm"] <= 0.01
    for eid in EIDS:
        row = identity["per_wheel"][eid]
        assert row["tread_torus_face_count"] == 4
        assert row["face_count"] == 190


def test_p4_contract_separates_use_from_truth(record):
    contract = record["p4_consumption_contract_v2"]
    allowed = contract["may_use_as_TEST_PROXY_parameters"]
    forbidden = contract["forbidden_as_real_values"]
    assert allowed and forbidden and contract["gates_for_p4"]
    for item in allowed:
        assert item["verdict"] in VERDICTS
        assert item["constraint"]
    joined = " ".join(forbidden)
    for token in ("loaded_rolling_height", "lift = 0.334", "friction", "inertia", "steering"):
        assert token in joined, token
    # a candidate radius must never be advertised as a frozen truth
    assert not any(item.get("verdict") == "verified" for item in allowed)


def test_unknowns_name_the_missing_measurement(record):
    unknowns = record["unknowns"]
    assert len(unknowns) >= 5
    for item in unknowns:
        assert item["quantity"]
        assert item["missing_measurement"]
