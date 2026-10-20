"""Tests for the STEP -> USD part mapping.

The matching logic takes plain dicts, so these tests need neither pxr nor the real
CAD files: they pin down the grading rules that decide what may be called
VERIFIED.  They prove that

* an equal number of solids and meshes never becomes a bijection by itself,
* duplicate name keys are downgraded to CANDIDATE even when geometry agrees,
* the literal name NONE can never reach VERIFIED,
* parts with different name keys are not paired just because their boxes match,
* a curved part whose mesh bulges past the vertex bbox is still accepted, but a
  mesh that leaves the control-point hull is rejected.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation.map_step_solids_to_usd_meshes import (  # noqa: E402
    GRADE_CANDIDATE,
    GRADE_STRONG_CANDIDATE,
    match_parts,
    name_key,
    prim_core,
    sandwich,
)


def solid(eid, name, lower_min, lower_max, upper_min=None, upper_max=None):
    return {
        "eid": eid,
        "name": name,
        "lower_min": lower_min,
        "lower_max": lower_max,
        "upper_min": lower_min if upper_min is None else upper_min,
        "upper_max": lower_max if upper_max is None else upper_max,
    }


def mesh_row(path, core, minimum, maximum, points=100):
    return {"path": path, "core": core, "min": minimum, "max": maximum, "points": points}


def test_name_key_survives_the_converter_re_encoding():
    # the converter turns the full-width slash into a stray lowercase u
    assert name_key("M1.1-00V3 主体模块-1/GB\u2571T 70.1-2008[M5\u00d712]-3-solid1") == name_key(
        "M1100V31GBuT7012008M5123solid1"
    )
    assert name_key("NONE") == "NONE"
    # the trailing "solidN" marker keeps its digit, which is what distinguishes parts
    assert name_key("part-solid1") != name_key("part-solid2")


def test_prim_core_strips_hash_and_dedup_suffixes():
    assert (
        prim_core("tn__M1100V31M1105V33solid1_zMGn99EKMW0Ei0uy2yCy0Z2h3V1qAt5")
        == "M1100V31M1105V33solid1"
    )
    assert prim_core("NONE_34") == "NONE"
    assert prim_core("tn__M1105V33solid1_1") == "M1105V33solid1"
    assert prim_core("M1105V33solid1") == "M1105V33solid1"
    assert prim_core("NONE") == "NONE"


def test_sandwich_accepts_identical_boxes():
    result = sandwich([0, 0, 0], [10, 20, 30], [0, 0, 0], [10, 20, 30], [0, 0, 0], [10, 20, 30])
    assert result["ok"] is True and result["cost"] == 0.0


def test_sandwich_accepts_a_curved_part_bulging_past_the_vertex_box():
    # real case: vertex box 48.7 x 180.8 x 157.8, control hull 49 x 206 x 206
    lower_min, lower_max = [0.0, 0.0, 0.0], [48.7, 180.8, 157.8]
    upper_min, upper_max = [-0.1, -0.1, -0.1], [48.9, 206.0, 206.0]
    mesh_min, mesh_max = [0.0, 0.0, 0.0], [48.79, 205.99, 205.99]
    result = sandwich(lower_min, lower_max, upper_min, upper_max, mesh_min, mesh_max)
    assert result["ok"] is True
    assert max(result["exceeds_vertex_bbox_mm"]) > 40.0


def test_sandwich_allows_legitimate_min_side_bulge():
    # Vertex box must be contained in the mesh, not the other way around.
    result = sandwich(
        [10, 10, 10], [20, 20, 20],
        [0, 0, 0], [30, 30, 30],
        [5, 5, 5], [25, 25, 25],
    )
    assert result["ok"] is True
    assert result["below_lower_bound_mm"] == [0.0, 0.0, 0.0]


def test_sandwich_rejects_missing_the_min_side_of_a_brep_vertex_bbox():
    # The previous implementation reversed this inequality and accepted it.
    result = sandwich(
        [0, 0, 0], [10, 10, 10],
        [-10, -10, -10], [20, 20, 20],
        [5, 0, 0], [10, 10, 10],
    )
    assert result["ok"] is False
    assert result["below_lower_bound_mm"][0] == 5


def test_sandwich_rejects_a_mesh_that_escapes_the_control_hull():
    result = sandwich([0, 0, 0], [10, 10, 10], [0, 0, 0], [12, 12, 12], [0, 0, 0], [40, 10, 10])
    assert result["ok"] is False
    assert max(result["above_upper_bound_mm"]) > 20.0


def test_sandwich_rejects_a_mesh_that_sits_inside_the_lower_bound():
    result = sandwich([0, 0, 0], [10, 10, 10], [0, 0, 0], [10, 10, 10], [3, 0, 0], [7, 10, 10])
    assert result["ok"] is False


def test_unique_key_and_geometry_are_strong_candidate_not_proven_identity():
    result = match_parts(
        [solid(1, "PART-A-1-solid1", [0, 0, 0], [10, 20, 30])],
        [mesh_row("/p/one/Mesh", "PARTA1solid1", [0, 0, 0], [10, 20, 30])],
    )
    assert result["grade_counts"] == {GRADE_STRONG_CANDIDATE: 1}
    assert result["bijection_proven"] is False
    assert "identity NOT proven" in result["pairs"][0]["reasons"][-1]


def test_duplicate_keys_are_candidate_and_paired_by_position():
    solids = [
        solid(1, "DUPPART", [0, 0, 0], [10, 10, 10]),
        solid(2, "DUPPART", [100, 0, 0], [110, 10, 10]),
    ]
    meshes = [
        mesh_row("/p/a/Mesh", "DUPPART", [100, 0, 0], [110, 10, 10]),
        mesh_row("/p/b/Mesh", "DUPPART", [0, 0, 0], [10, 10, 10]),
    ]
    result = match_parts(solids, meshes)
    assert result["grade_counts"] == {GRADE_CANDIDATE: 2}
    assert result["bijection_proven"] is False
    by_eid = {pair["step_eid"]: pair for pair in result["pairs"]}
    assert by_eid[1]["usd_path"] == "/p/b/Mesh"
    assert by_eid[2]["usd_path"] == "/p/a/Mesh"


def test_literal_none_names_can_never_be_verified():
    result = match_parts(
        [solid(9, "NONE", [0, 0, 0], [5, 5, 5])],
        [mesh_row("/p/none/Mesh", "NONE", [0, 0, 0], [5, 5, 5])],
    )
    assert result["grade_counts"] == {GRADE_CANDIDATE: 1}
    assert result["bijection_proven"] is False
    assert result["pairs_without_name_evidence"] == 1


def test_different_keys_are_not_paired_even_when_boxes_match():
    result = match_parts(
        [solid(3, "SOMEPART", [0, 0, 0], [10, 10, 10])],
        [mesh_row("/p/other/Mesh", "OTHERPART", [0, 0, 0], [10, 10, 10])],
    )
    assert result["pairs"] == []
    assert len(result["unmatched_solids"]) == 1
    assert len(result["unmatched_meshes"]) == 1
    assert result["unmatched_solids"][0]["closest_mesh_key"] == "OTHERPART"


def test_geometry_rejection_is_reported_not_silently_dropped():
    result = match_parts(
        [solid(4, "KEEPPART", [0, 0, 0], [10, 10, 10])],
        [mesh_row("/p/big/Mesh", "KEEPPART", [0, 0, 0], [500, 10, 10])],
    )
    assert result["pairs"] == []
    assert result["rejected_pair_count"] >= 1
    assert result["rejected_pairs"][0]["step_eid"] == 4


def test_identical_unique_names_do_not_prove_bijection():
    result = match_parts(
        [solid(1, "BOLT-1", [0, 0, 0], [4, 4, 4])],
        [mesh_row("/bolt/Mesh", "BOLT1", [0, 0, 0], [4, 4, 4])],
    )
    assert result["grade_counts"] == {GRADE_STRONG_CANDIDATE: 1}
    assert result["bijection_proven"] is False




def test_matching_maximizes_pairs_instead_of_greedy_lowest_cost(monkeypatch):
    # S1 has two valid partners; S2 can only use M1. Greedy global
    # cost picks S1-M1 and strands S2; an augmenting path recovers both.
    import scripts.simulation.map_step_solids_to_usd_meshes as mapper

    def fake_sandwich(lower_min, lower_max, upper_min, upper_max, mesh_min, mesh_max):
        eid = int(lower_min[0])
        mid = int(mesh_min[0])
        valid = (eid, mid) in {(1, 1), (1, 2), (2, 1)}
        return {"ok": valid, "exceeds_vertex_bbox_mm": [0, 0, 0],
                "below_lower_bound_mm": [0, 0, 0],
                "above_upper_bound_mm": [0, 0, 0], "cost": 0 if valid else 100}

    def fake_cost(lower_min, lower_max, upper_min, upper_max, mesh_min, mesh_max):
        return {(1, 1): 0.0, (1, 2): 2.0, (2, 1): 1.0, (2, 2): 100.0}[
            (int(lower_min[0]), int(mesh_min[0]))]

    monkeypatch.setattr(mapper, "sandwich", fake_sandwich)
    monkeypatch.setattr(mapper, "pair_cost", fake_cost)
    solids = [solid(1, "DUP", [1, 0, 0], [10, 10, 10]),
              solid(2, "DUP", [2, 0, 0], [10, 10, 10])]
    meshes = [mesh_row("/m1", "DUP", [1, 0, 0], [10, 10, 10]),
              mesh_row("/m2", "DUP", [2, 0, 0], [10, 10, 10])]
    result = mapper.match_parts(solids, meshes)
    assert len(result["pairs"]) == 2
    assert {p["step_eid"]: p["usd_path"] for p in result["pairs"]} == {
        1: "/m2", 2: "/m1"}
    assert result["bijection_proven"] is False


