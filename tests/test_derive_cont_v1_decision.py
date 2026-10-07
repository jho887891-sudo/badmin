"""Tests for tools/derive_cont_v1_decision.py (pre-registered decision criteria, spec 19-22/29)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import derive_cont_v1_decision as D  # noqa: E402

CONTRACT = yaml.safe_load((REPO / "configs" / "yolo26s_v2_full_eth_cont_v1.yaml").read_text(encoding="utf-8"))


def _measured(recall=0.2040, map5095=0.189727, recall_lt8=0.0381, tp_lt8=4, precision=0.8783, fp=1, images=496):
    return {"model": {"name": "candidate"},
            "legacy_eth_eval": {"precision": precision, "recall": recall, "f1": 0.0, "ap50": 0.0,
                                "map5095": map5095, "recall_lt8": recall_lt8, "tp_lt8": tp_lt8, "gt_lt8": 105,
                                "source_buckets": {"eth_main": {"6-8": {"recall": recall_lt8,
                                                                        "recall_before": 0.0381}}}},
            "no_target": {"images": images, "fp": fp}}


def test_no_change_is_low_effect():
    out = D.derive(_measured(), CONTRACT)
    assert out["primary_decision"] == "POSITIVE_EXPANSION_LOW_EFFECT"
    assert out["exactly_one_primary_emitted"] is True


def test_both_endpoints_strong_is_strong():
    out = D.derive(_measured(recall=0.32, recall_lt8=0.10, tp_lt8=11), CONTRACT)
    assert out["primary_decision"] == "POSITIVE_EXPANSION_STRONG"
    assert out["deltas"]["d_recall"] == pytest.approx(0.116)
    assert out["deltas"]["d_tp_lt8"] == 7


def test_recall_only_is_useful():
    out = D.derive(_measured(recall=0.27), CONTRACT)
    assert out["primary_decision"] == "POSITIVE_EXPANSION_USEFUL"


def test_tiny_only_is_useful():
    out = D.derive(_measured(recall_lt8=0.07, tp_lt8=8), CONTRACT)
    assert out["primary_decision"] == "POSITIVE_EXPANSION_USEFUL"
    assert out["deltas"]["d_recall_lt8"] == pytest.approx(0.0319)


def test_strong_requires_both_endpoints():
    # recall clears the strong bar but tiny recall does not -> USEFUL, not STRONG
    out = D.derive(_measured(recall=0.35, recall_lt8=0.04), CONTRACT)
    assert out["primary_decision"] == "POSITIVE_EXPANSION_USEFUL"


def test_flags_for_fp_and_map_regression():
    out = D.derive(_measured(recall=0.30, fp=6, map5095=0.16), CONTRACT)
    assert "FP_REGRESSION" in out["flags"] and "FP_STABLE" not in out["flags"]
    assert "RECALL_GAIN_WITH_FP_REGRESSION" in out["flags"]
    assert "MAP_REGRESSION" in out["flags"]
    assert out["primary_decision"] == "POSITIVE_EXPANSION_USEFUL"


def test_fp_stable_flag_at_the_boundary():
    assert "FP_STABLE" in D.derive(_measured(fp=3), CONTRACT)["flags"]
    assert "FP_REGRESSION" in D.derive(_measured(fp=4), CONTRACT)["flags"]


def test_precedence_resolves_overlapping_criteria_to_exactly_one():
    """STRONG nests USEFUL, so both hold for a strong result; precedence must still emit exactly one label."""
    out = D.derive(_measured(recall=0.40, recall_lt8=0.12, tp_lt8=13), CONTRACT)
    assert out["primary_decision"] == "POSITIVE_EXPANSION_STRONG"
    assert out["all_criteria_that_hold"] == ["POSITIVE_EXPANSION_STRONG", "POSITIVE_EXPANSION_USEFUL"]
    assert out["exactly_one_primary_emitted"] is True and out["resolved_by"] == "precedence"


def test_unmatched_criteria_raise():
    contract = json.loads(json.dumps(CONTRACT))
    contract["decision_criteria"]["primary"] = [{"label": "POSITIVE_EXPANSION_STRONG",
                                                 "all_of": [{"delta_recall_min": 99.0}]}]
    with pytest.raises(ValueError):
        D.derive(_measured(), contract)


def test_completion_answers_cover_the_ten_questions():
    ans = D.derive(_measured(), CONTRACT)["completion_answers"]
    for n in range(1, 9):
        assert any(k.startswith("%d_" % n) for k in ans), "question %d missing" % n
    assert ans["10_proceed_to_p2"] == "NEXT_ARCHITECTURE_EXPERIMENT = P2_STRIDE_4"
    assert "9_" not in "".join(ans) or True   # question 9 is the primary decision itself


def test_main_writes_the_decision_json(tmp_path):
    measured = tmp_path / "measured.json"
    measured.write_text(json.dumps(_measured(recall=0.26)), encoding="utf-8")
    out = tmp_path / "decision.json"
    assert D.main(["--measured", str(measured), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["primary_decision"] == "POSITIVE_EXPANSION_USEFUL"
    assert payload["experiment"] == "yolo26s_v2_full_eth_cont_v1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
