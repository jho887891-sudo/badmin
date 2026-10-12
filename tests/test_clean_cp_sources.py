"""Task 2b tests: source-role bookkeeping must make a leak or a double-used COCO image impossible to miss.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 2, Step 1/2).
Written before the SourceRecord/audit_source_roles implementation exists, so the first run is RED.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import clean_cp_contract as C  # noqa: E402


def _rec(source_id, role, **kw):
    payload = dict(source_id=source_id, role=role, sha256="sha-" + source_id, path="/data/" + source_id + ".jpg",
                   origin="fixture", license_or_permission="declared", qa_status="APPROVED")
    payload.update(kw)
    return C.SourceRecord(**payload)


def test_eval_source_role_never_allowed():
    """A frozen evaluation image may never be handed a training role, by path or by content."""
    frozen = [_rec("eval-1", C.ROLE_FROZEN_EVAL, sha256="sha-shared", path="/data/eval/f1.jpg")]
    records = [
        _rec("bg-1", C.ROLE_COCO_CLEAN_BG),
        _rec("bg-2", C.ROLE_COCO_CLEAN_BG, sha256="sha-shared", path="/data/other/g2.jpg"),
    ]
    audit = C.audit_source_roles(records, frozen)
    assert audit["status"] == "INVALID_OR_LEAK"
    assert any("content" in v["kind"] or "sha" in v["kind"] for v in audit["violations"])

    records_path = [_rec("bg-3", C.ROLE_COCO_CLEAN_BG, sha256="sha-fresh", path="/data/eval/f1.jpg")]
    audit_path = C.audit_source_roles(records_path, frozen)
    assert audit_path["status"] == "INVALID_OR_LEAK"

    record_with_eval_role = [_rec("x", C.ROLE_FROZEN_EVAL)]
    audit_role = C.audit_source_roles(record_with_eval_role, frozen)
    assert audit_role["status"] == "INVALID_OR_LEAK"
    assert any(v["kind"] == "eval_role_in_training" for v in audit_role["violations"])


def test_same_coco_id_cannot_have_two_new_roles():
    """One COCO image may be a clean background or a hard negative, never both."""
    records = [
        _rec("bg", C.ROLE_COCO_CLEAN_BG, coco_image_id=555),
        _rec("hn", C.ROLE_COCO_OUTDOOR_HARDNEG, coco_image_id=555),
    ]
    audit = C.audit_source_roles(records, [])
    assert audit["status"] == "INVALID_OR_LEAK"
    assert any(v["kind"] == "coco_id_in_two_new_roles" and v["coco_image_id"] == 555 for v in audit["violations"])

    clean = [
        _rec("eth", C.ROLE_ETH_TRAIN_FOREGROUND),
        _rec("bg", C.ROLE_COCO_CLEAN_BG, coco_image_id=556),
        _rec("hn", C.ROLE_COCO_OUTDOOR_HARDNEG, coco_image_id=557),
        _rec("old", C.ROLE_V2_HISTORICAL_NEGATIVE, coco_image_id=558),
    ]
    ok = C.audit_source_roles(clean, [])
    assert ok["status"] == "PASS"
    assert ok["role_counts"][C.ROLE_COCO_CLEAN_BG] == 1


def test_unknown_location_not_counted_as_distinct_location():
    """COCO metadata cannot prove a location: blank location_id must never inflate the location count."""
    records = [
        _rec("a", C.ROLE_COCO_CLEAN_BG, coco_image_id=1, location_id=""),
        _rec("b", C.ROLE_COCO_CLEAN_BG, coco_image_id=2, location_id=None),
        _rec("c", C.ROLE_COCO_CLEAN_BG, coco_image_id=3, location_id="farm-road"),
        _rec("d", C.ROLE_ETH_TRAIN_FOREGROUND, coco_image_id=None, location_id="cab_1"),
    ]
    audit = C.audit_source_roles(records, [])
    assert audit["status"] == "PASS"
    assert audit["distinct_locations"] == {"cab_1", "farm-road"}
    assert audit["unknown_location_count"] == 2


def test_same_source_id_twice_is_a_violation():
    """Two rows claiming the same source_id would double-count one picture: that is a leak, not a warning."""
    records = [_rec("bg", C.ROLE_COCO_CLEAN_BG, coco_image_id=9), _rec("bg", C.ROLE_COCO_CLEAN_BG, coco_image_id=10)]
    audit = C.audit_source_roles(records, [])
    assert audit["status"] == "INVALID_OR_LEAK"
    assert any(v["kind"] == "duplicate_source_id" for v in audit["violations"])


def test_roles_must_be_from_the_frozen_enum():
    """An invented role string must be rejected: the plan fixes the role enum for the audit."""
    audit = C.audit_source_roles([_rec("x", "SOMETHING_ELSE")], [])
    assert audit["status"] == "INVALID_OR_LEAK"
    assert any(v["kind"] == "unknown_role" for v in audit["violations"])
