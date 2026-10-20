"""Tests for the mapping supersession tool.

The tool exists to attribute a change between two mappings to either the
containment-inequality fix or the pairing-algorithm change, so the tests pin down
exactly that: the two rules must disagree in the documented direction, and the
cause labels must follow the evidence rather than the ordering of the rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulation.verify_mapping_supersession import (  # noqa: E402
    audited_rule_ok,
    compare,
    old_rule_ok,
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


def mesh_row(path, core, minimum, maximum):
    return {"path": path, "core": core, "min": minimum, "max": maximum}


def pair(eid, name, path, grade="STRONG_CANDIDATE"):
    return {"step_eid": eid, "step_name": name, "usd_path": path, "grade": grade}


def test_pre_audit_rule_rejected_a_mesh_bulging_below_the_vertex_box():
    """The fix in action: a curved part whose mesh reaches below the vertex box."""
    s = solid(1, "PART", [0.0, 0.0, 0.0], [48.7, 180.8, 157.8],
              [-8.0, -8.0, -8.0], [52.0, 210.0, 210.0])
    # 6 mm below the vertex box is outside the 2.5 mm tolerance, so the pre-audit
    # formula rejects it while the audited one does not care about that direction
    m = mesh_row("/p/Mesh", "PART", [-6.0, -6.0, -6.0], [48.79, 205.99, 205.99])
    assert old_rule_ok(s, m) is False   # old formula penalised the negative bulge
    assert audited_rule_ok(s, m) is True


def test_audited_rule_rejects_a_mesh_that_fails_to_cover_the_vertex_box():
    """The other half of the fix: the pre-audit formula let this through."""
    s = solid(2, "PART", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0],
              [0.0, 0.0, 0.0], [20.0, 20.0, 20.0])
    # the mesh starts 3 mm inside the vertex box, so it does not cover the part
    m = mesh_row("/p/Mesh", "PART", [3.0, 0.0, 0.0], [20.0, 10.0, 10.0])
    assert old_rule_ok(s, m) is True
    assert audited_rule_ok(s, m) is False


def test_identical_boxes_are_admissible_under_both_rules():
    s = solid(3, "PART", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
    m = mesh_row("/p/Mesh", "PART", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
    assert old_rule_ok(s, m) and audited_rule_ok(s, m)


def test_compare_marks_kept_pairs_as_unchanged():
    s = solid(1, "PART", [0, 0, 0], [10, 10, 10])
    m = mesh_row("/p/Mesh", "PART", [0, 0, 0], [10, 10, 10])
    result = compare([s], [m], [pair(1, "PART", "/p/Mesh")], [pair(1, "PART", "/p/Mesh")],
                     reproduced_old={1: "/p/Mesh"})
    assert result["cause_counts"] == {"UNCHANGED": 1}
    assert result["reproduced_old_rule_agreement_ratio"] == 1.0


def test_compare_attributes_a_new_pair_to_the_inequality_fix():
    s = solid(7, "PART", [0, 0, 0], [10, 10, 10], [-8, -8, -8], [12, 12, 12])
    m = mesh_row("/p/Mesh", "PART", [-6, -6, -6], [10, 10, 10])
    result = compare([s], [m], [], [pair(7, "PART", "/p/Mesh")], reproduced_old={})
    row = result["rows"][0]
    assert row["cause"] == "INEQUALITY_FIX"
    assert row["new_partner_feasible_under_old_rule"] is False
    assert row["candidate_count_old_rule"] == 0


def test_compare_attributes_a_partner_swap_to_the_matching_algorithm():
    s = solid(8, "DUP", [0, 0, 0], [10, 10, 10])
    first = mesh_row("/p/a/Mesh", "DUP", [0, 0, 0], [10, 10, 10])
    second = mesh_row("/p/b/Mesh", "DUP", [0, 0, 0], [10, 10, 10])
    result = compare(
        [s], [first, second],
        [pair(8, "DUP", "/p/a/Mesh")], [pair(8, "DUP", "/p/b/Mesh")],
        reproduced_old={8: "/p/a/Mesh"},
    )
    row = result["rows"][0]
    assert row["cause"] == "MATCHING_ALGORITHM"
    assert row["new_partner_feasible_under_old_rule"] is True
    assert row["old_partner_feasible_under_audited_rule"] is True


def test_compare_reports_lost_pairs_instead_of_hiding_them():
    s = solid(9, "PART", [0, 0, 0], [10, 10, 10])
    result = compare([s], [], [pair(9, "PART", "/p/Mesh")], [], reproduced_old={9: "/p/Mesh"})
    assert result["cause_counts"] == {"LOST": 1}
    assert result["rows"][0]["new_usd_path"] is None
