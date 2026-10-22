# -*- coding: utf-8 -*-
"""R10-F2 (P7-09): every reachability verdict is labelled, and a two-id object RAISES.

The decisive tests here are the negative ones:
  * an object carrying two backend ids raises BackendMixError (the explicit task requirement);
  * a label whose backend disagrees with its provenance raises;
  * a requested verified backend that cannot be constructed produces an UNAVAILABLE section,
    NOT proxy numbers under a verified heading (no silent fallback).

Run:
    python -m pytest tests/r10_f/test_backend_provenance.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT / "src", ROOT / "simulation", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from src.badminton_brain.reachability import backend_compare as compare  # noqa: E402
from src.badminton_brain.reachability import backends as backends_mod  # noqa: E402
from src.badminton_brain.reachability.reachability import (  # noqa: E402
    ReachabilityDecision, ReachabilityReason, ReachabilityVerdict,
)

SCENARIOS = (
    compare.ComparisonScenario(name="rest_reach", intercept_point_m=(0.45, 0.0, 1.0),
                               approach_direction=(-1.0, 0.0, 0.0), time_to_target_s=0.30,
                               note="proxy-convergent point"),
    compare.ComparisonScenario(name="base_column", intercept_point_m=(0.20, 0.0, 1.0),
                               approach_direction=(-1.0, 0.0, 0.0), time_to_target_s=0.30,
                               note="inside the self-collision proxy"),
    compare.ComparisonScenario(name="outside_box", intercept_point_m=(2.4, 0.0, 1.0),
                               approach_direction=(-1.0, 0.0, 0.0), time_to_target_s=0.30,
                               note="outside the workspace box"),
)


def _p6_root():
    env = os.environ.get(backends_mod.P6_ROOT_ENV)
    if env and (Path(env) / backends_mod.P6_FK_RELPATH).is_file():
        return env
    candidate = ROOT.parent / "badmin_wt_p6d"
    if (candidate / backends_mod.P6_FK_RELPATH).is_file():
        return str(candidate)
    return None


def _dummy_verdict():
    return ReachabilityVerdict(decision=ReachabilityDecision.UNKNOWN,
                              reason=ReachabilityReason.CALIBRATION_MISSING,
                              detail="test fixture")


def _proxy_provenance():
    return compare.VerdictProvenance.of(backends_mod.ProxyDhBackend())


# ------------------------------------------------------------- the mix refusal
def test_a_verdict_carrying_two_backend_ids_raises():
    provenance = _proxy_provenance()
    with pytest.raises(backends_mod.BackendMixError) as excinfo:
        compare.LabelledVerdict(scenario="s", backend_ids=("proxy_dh", "verified_p6_fk"),
                                provenance=provenance, verdict=_dummy_verdict(),
                                seed_rad=(0.0,) * provenance.dof)
    assert "exactly ONE backend" in str(excinfo.value)


def test_a_single_backend_id_is_accepted_and_a_mismatched_label_raises():
    provenance = _proxy_provenance()
    entry = compare.LabelledVerdict(scenario="s", backend_ids=("proxy_dh",),
                                    provenance=provenance, verdict=_dummy_verdict(),
                                    seed_rad=(0.0,) * provenance.dof)
    assert entry.backend == "proxy_dh"
    assert entry.as_dict()["backend_ids"] == ["proxy_dh"]
    with pytest.raises(backends_mod.BackendMixError):
        compare.LabelledVerdict(scenario="s", backend_ids=("verified_p6_fk",),
                                provenance=provenance, verdict=_dummy_verdict(),
                                seed_rad=(0.0,) * provenance.dof)


def test_single_backend_id_refuses_empty_and_duplicate_conflicts():
    with pytest.raises(backends_mod.BackendMixError):
        compare.single_backend_id(None, "")
    assert compare.single_backend_id("proxy_dh", "proxy_dh") == "proxy_dh"
    with pytest.raises(backends_mod.BackendMixError):
        compare.single_backend_id("proxy_dh", "verified_p6_fk")


# ------------------------------------------------------- no silent fallback ever
def test_a_missing_verified_backend_is_unavailable_and_not_replaced_by_the_proxy(tmp_path):
    report = compare.compare_backends(
        SCENARIOS, verified_kwargs={"fk_path": tmp_path / "nope" / "fk.py"})
    verified_section = report["backends"]["verified_p6_fk"]
    assert verified_section["status"] == "UNAVAILABLE"
    assert verified_section["verdicts"] == []
    assert "VERIFIED_FK_UNAVAILABLE" in verified_section["refusal"]
    # the proxy section is still there, and it is still labelled as the proxy
    assert report["backends"]["proxy_dh"]["status"] == "OK"
    assert report["backends"]["proxy_dh"]["provenance"]["backend"] == "proxy_dh"
    assert report["backends"]["proxy_dh"]["provenance"]["frame_name"] == "racket_face_proxy"


def test_select_backend_never_substitutes(tmp_path):
    with pytest.raises(backends_mod.VerifiedFkUnavailable):
        backends_mod.select_backend("verified_p6_fk", fk_path=tmp_path / "missing.py")
    with pytest.raises(backends_mod.BackendUnavailable):
        backends_mod.select_backend("not_a_backend")


# ------------------------------------------------------- the labelled comparison
def test_the_proxy_comparison_labels_every_verdict():
    report = compare.compare_backends(SCENARIOS, backends=(backends_mod.BACKEND_PROXY,))
    section = report["backends"]["proxy_dh"]
    assert section["status"] == "OK"
    assert len(section["verdicts"]) == len(SCENARIOS)
    for verdict in section["verdicts"]:
        assert verdict["backend"] == "proxy_dh"
        assert verdict["backend_ids"] == ["proxy_dh"]
        assert verdict["frame_name"] == "racket_face_proxy"
        assert verdict["target_kind"] == "RACKET_FACE_PROXY"
        assert verdict["provenance"]["source_class"] == "PROXY_DH_ENGINEERING_V0_1"
        assert verdict["decision"] in {"REACHABLE", "NOT_REACHABLE", "UNKNOWN"}
    reasons = {v["scenario"]: v["reason"] for v in section["verdicts"]}
    assert reasons["base_column"] == "SELF_COLLISION_PROXY"
    assert reasons["outside_box"] == "WORKSPACE_BOUND"
    # All three fixtures are refusals from a ZERO seed: this test is about LABELLING, not about
    # how good the proxy arm is.  The counts are pinned so a silent change of behaviour shows up.
    assert section["decisions"]["NOT_REACHABLE"] == 3
    assert section["decisions"]["REACHABLE"] == 0


def test_the_two_backends_are_compared_but_never_merged():
    root = _p6_root()
    if root is None:
        pytest.skip("P6 verified FK checkout not found (BADMIN_P6_ROOT unset and no sibling "
                    "worktree): the verified half is UNKNOWN here, it is NOT substituted")
    report = compare.compare_backends(SCENARIOS, verified_kwargs={"root": root})
    proxy = report["backends"]["proxy_dh"]
    verified = report["backends"]["verified_p6_fk"]
    assert proxy["status"] == "OK" and verified["status"] == "OK"
    assert proxy["provenance"]["dof"] == 6
    assert verified["provenance"]["dof"] == 8
    assert proxy["provenance"]["frame_name"] != verified["provenance"]["frame_name"]
    for section in (proxy, verified):
        ids = {verdict["backend"] for verdict in section["verdicts"]}
        assert len(ids) == 1
        assert ids == {section["provenance"]["backend"]}
    # every scenario appears once per backend: they are compared, not pooled
    names = [v["scenario"] for v in proxy["verdicts"]]
    assert names == [s.name for s in SCENARIOS]
    assert [v["scenario"] for v in verified["verdicts"]] == names
    assert verified["provenance"]["detail"]["fk_module_sha256"]


def test_the_table_labels_every_row_with_a_backend_and_a_frame():
    report = compare.compare_backends(SCENARIOS, backends=(backends_mod.BACKEND_PROXY,))
    table = compare.format_table(report)
    rows = [line for line in table.splitlines() if line.startswith("| proxy_dh |")]
    assert len(rows) == len(SCENARIOS)
    for row in rows:
        assert "racket_face_proxy" in row
