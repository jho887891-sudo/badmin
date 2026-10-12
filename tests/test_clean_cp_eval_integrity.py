"""Task 9 tests: the canonical evaluation adapter must pin the frozen evaluation contract, keep the real and
synthetic domains (and their tiny size buckets) apart, refuse any outdoor claim without an independent real
outdoor test set, and never let an evaluation false-positive example become training input.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 9, Step 1/2).
Written before tools/evaluate_clean_cp_generalization.py exists, so the first run is RED (ImportError).

Nothing in this file reads an evaluation image, runs the real evaluator or invents a metric: the fixtures are
synthetic frames and stored-reference payloads, and the one injected runner is a lambda.
"""
from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import clean_cp_contract as C  # noqa: E402
import eval_yolo26_v1 as ev  # noqa: E402
import evaluate_clean_cp_generalization as E  # noqa: E402

NEW_CFG = REPO / "configs" / "eth_real_clean_outdoor_copypaste_v1.yaml"


def _csv(path: Path, columns, rows) -> Path:
    """Write a small manifest fixture; the adapter only ever parses these files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _spec_eval_cfg() -> dict:
    return dict(yaml.safe_load(NEW_CFG.read_text(encoding="utf-8"))["evaluation"])


def _record(source_id, role, **kw) -> C.SourceRecord:
    payload = dict(source_id=source_id, role=role, sha256="sha-" + source_id, path="/data/" + source_id + ".jpg",
                   origin="fixture", license_or_permission="declared", qa_status="APPROVED",
                   location_id="cab_1", session_id="eth_cab_1")
    payload.update(kw)
    return C.SourceRecord(**payload)


def _prediction_frame() -> pd.DataFrame:
    """Two unmatched boxes on one reviewed image plus one unmatched box on an unreviewed image."""
    return pd.DataFrame([
        {"image": "a.jpg", "conf": 0.91, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "matched": False},
        {"image": "a.jpg", "conf": 0.44, "x1": 50.0, "y1": 50.0, "x2": 60.0, "y2": 60.0, "matched": False},
        {"image": "b.jpg", "conf": 0.80, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "matched": True},
        {"image": "c.jpg", "conf": 0.60, "x1": 5.0, "y1": 5.0, "x2": 15.0, "y2": 15.0, "matched": False},
    ])


def test_eval_imgsz_conf_apfloor_nms_and_maxdet_fixed():
    """Any drift of imgsz / conf / AP floor / NMS IoU / max_det - in either config - is a hard error."""
    spec = _spec_eval_cfg()
    evaluator = E.frozen_evaluator_cfg()
    E.assert_eval_contract(evaluator, spec)
    assert (evaluator["imgsz"], evaluator["conf"], evaluator["ap_floor"]) == (1024, 0.25, 0.001)
    assert (evaluator["nms_iou"], evaluator["max_det"]) == (0.7, 300)

    for key, wrong in (("imgsz", 640), ("conf", 0.5), ("ap_floor", 0.05), ("nms_iou", 0.5), ("max_det", 100)):
        drifted = dict(evaluator)
        drifted[key] = wrong
        with pytest.raises(E.EvalContractViolation) as exc:
            E.assert_eval_contract(drifted, spec)
        assert key in str(exc.value), "the failure must name the drifted setting"

        spec_drift = dict(spec)
        spec_drift[key] = wrong
        with pytest.raises(E.EvalContractViolation):
            E.assert_eval_contract(evaluator, spec_drift)

    incomplete = dict(evaluator)
    incomplete.pop("max_det")
    with pytest.raises(E.EvalContractViolation):
        E.assert_eval_contract(incomplete, spec)
    with pytest.raises(E.EvalContractViolation):
        E.assert_eval_contract(evaluator, {"imgsz": 1024})

    # the size-bucket rule is checked numerically against the canonical evaluator, never restated by hand
    assert E.assert_equiv_size_formula() == "sqrt(w_640 * h_640)"
    assert ev.equiv_size_640(10.0, 20.0, 1920, 1080) == pytest.approx(
        ((10.0 * 640.0 / 1920.0) * (20.0 * 640.0 / 1920.0)) ** 0.5)


def test_real_synthetic_tiny_cannot_merge():
    """Real and synthetic rows - including the tiny buckets - keep separate denominators in every table."""
    rows = E.size_bucket_rows("val|eth_unseen", [
        {"bucket": "<4", "source": "eth_main", "GT": 3, "TP": 1, "FN": 2},
        {"bucket": "4-6", "source": "eth_iphone", "GT": 2, "TP": 1, "FN": 1},
    ])
    rows += E.size_bucket_rows("val|eth_unseen", [
        {"bucket": "<4", "source": "synthetic", "GT": 20, "TP": 5, "FN": 15},
    ])
    E.assert_sources_not_merged(rows)
    assert {row["domain"] for row in rows} == {E.SOURCE_REAL, E.SOURCE_SYNTHETIC}
    tiny = [row for row in rows if row["bucket"] == "<4"]
    assert sorted(row["GT"] for row in tiny) == [3, 20], "tiny real and tiny synthetic never share a denominator"

    part = E.partition_by_domain(rows)
    assert part[E.SOURCE_REAL]["rows"] == 2 and part[E.SOURCE_REAL]["GT"] == 5
    assert part[E.SOURCE_SYNTHETIC]["rows"] == 1 and part[E.SOURCE_SYNTHETIC]["GT"] == 20

    for merged_label in ("real+synthetic", "mixed", "all", ""):
        with pytest.raises(E.EvalIntegrityError):
            E.size_bucket_rows("val|eth_unseen",
                               [{"bucket": "<4", "source": merged_label, "GT": 23, "TP": 6, "FN": 17}])
        with pytest.raises(E.EvalIntegrityError):
            E.assert_sources_not_merged(rows + [{"set": "val|eth_unseen", "bucket": "<8", "domain": merged_label,
                                                 "GT": 25, "TP": 7, "FN": 18}])

    with pytest.raises(E.EvalIntegrityError):
        E.assert_sources_not_merged(rows + [{"set": "val|eth_unseen", "bucket": "<4", "GT": 25}])


def test_outdoor_test_independent_from_all_training_locations_and_sessions():
    """The outdoor test set must be disjoint from every training location, session, path and content hash."""
    training = [
        _record("v2-pos", C.ROLE_ETH_TRAIN_FOREGROUND, location_id="cab_1", session_id="eth_cab_1"),
        _record("cp-bg", C.ROLE_COCO_CLEAN_BG, coco_image_id=11, location_id=None, session_id=None),
        _record("new-hn", C.ROLE_COCO_OUTDOOR_HARDNEG, coco_image_id=12, location_id=None, session_id=None),
        _record("v2-neg", C.ROLE_V2_HISTORICAL_NEGATIVE, coco_image_id=13, location_id=None, session_id=None),
    ]
    outdoor = [_record("ood-1", C.ROLE_FROZEN_EVAL, location_id="zurich_lake", session_id="ood_lake_2026-10-13",
                       sha256="sha-ood-1", path="/data/ood/1.jpg")]
    report = E.assert_outdoor_test_isolation(outdoor, training)
    assert report["status"] == "PASS"
    assert report["outdoor_images"] == 1 and report["training_sources"] == len(training)

    for field, value in (("location_id", "cab_1"), ("session_id", "eth_cab_1"),
                         ("sha256", "sha-v2-pos"), ("path", "/data/v2-pos.jpg")):
        clash = [dataclasses.replace(outdoor[0], **{field: value})]
        with pytest.raises(E.EvalIntegrityError) as exc:
            E.assert_outdoor_test_isolation(clash, training)
        assert field in str(exc.value)

    wrong_role = [dataclasses.replace(outdoor[0], role=C.ROLE_COCO_OUTDOOR_HARDNEG)]
    with pytest.raises(E.EvalIntegrityError):
        E.assert_outdoor_test_isolation(wrong_role, training)

    for blanked in ({"location_id": None}, {"session_id": None}, {"sha256": None}):
        unknown = [dataclasses.replace(outdoor[0], **blanked)]
        with pytest.raises(E.EvalIntegrityError):
            E.assert_outdoor_test_isolation(unknown, training)

    with pytest.raises(E.EvalIntegrityError):
        E.assert_outdoor_test_isolation([], training)


def test_test_fp_examples_not_recycled_to_training():
    """classify_fp_errors is a read-only taxonomy: it is per detection, marked eval-only, unusable as a pool."""
    taxonomy = E.classify_fp_errors(_prediction_frame(), pd.DataFrame([
        {"image": "a.jpg", "reviewed_category": "white_reflection", "no_shuttle_reviewed": True},
    ]))
    assert len(taxonomy) == 3, "one row per false-positive detection, never one per image"
    assert set(taxonomy["taxonomy"]) == {"white_reflection", E.FP_TAXONOMY_UNREVIEWED}
    assert taxonomy.attrs["role"] == E.FP_TAXONOMY_ROLE
    assert taxonomy.attrs["training_use"] is False
    assert E.FP_TAXONOMY_ROLE not in C.TRAINING_ROLES
    E.assert_fp_taxonomy_not_training_input(taxonomy)

    stripped = taxonomy.copy()
    stripped.attrs = {}
    with pytest.raises(E.EvalIntegrityError):
        E.assert_fp_taxonomy_not_training_input(stripped)

    wired = taxonomy.copy()
    wired["label_path"] = "/train/x.txt"
    wired.attrs = dict(taxonomy.attrs)
    with pytest.raises(E.EvalIntegrityError) as exc:
        E.assert_fp_taxonomy_not_training_input(wired)
    assert "label_path" in str(exc.value)

    with pytest.raises(E.EvalIntegrityError):
        E.classify_fp_errors(_prediction_frame(), pd.DataFrame([
            {"image": "a.jpg", "reviewed_category": "sparkly", "no_shuttle_reviewed": True}]))

    no_matching_column = _prediction_frame().drop(columns=["matched"])
    with pytest.raises(E.EvalIntegrityError):
        E.classify_fp_errors(no_matching_column, pd.DataFrame(
            columns=["image", "reviewed_category", "no_shuttle_reviewed"]))


def test_no_real_outdoor_test_returns_PILOT_ONLY(tmp_path):
    """Without a real outdoor test set the honest outcome is PILOT_ONLY, and a missing checkpoint is refused."""
    manifest = _csv(tmp_path / "val_manifest.csv", ["image", "label", "split"],
                    [{"image": "/d/a.jpg", "label": "/d/a.txt", "split": "val"}])

    with pytest.raises(E.EvalBlocked) as exc:
        E.run_canonical_evaluations(tmp_path / "absent.pt", {"val": manifest}, runner=lambda command: {})
    assert "checkpoint" in str(exc.value)

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder checkpoint; this test never loads or runs the real evaluator")
    result = E.run_canonical_evaluations(checkpoint, {"val": manifest}, runner=lambda command: {})
    assert result["status"] == "PILOT_ONLY"
    assert result["outdoor_claim_allowed"] is False
    assert result["outdoor_blocker"]
    assert E.OUTDOOR_REAL_OOD_SET not in result["sets"]
    assert result["eval_contract"]["nms_iou"] == 0.7 and result["eval_contract"]["max_det"] == 300
    command = result["sets"]["val"]["command"]
    assert Path(command[1]).name == "eval_yolo26_v1.py"
    assert command[command.index("--imgsz") + 1] == str(E.IMGSZ)
    assert command[command.index("--ap-conf") + 1] == str(E.CONF_FLOOR)

    ood = _csv(tmp_path / "outdoor_real_ood.csv", ["image", "label", "location", "session", "sha256"],
               [{"image": "/ood/1.jpg", "label": "/ood/1.txt", "location": "zurich_lake",
                 "session": "ood_lake_2026-10-13", "sha256": "a" * 64}])
    claimed = E.run_canonical_evaluations(checkpoint, {"val": manifest, E.OUTDOOR_REAL_OOD_SET: ood},
                                          runner=lambda command: {})
    assert claimed["status"] == "EVALUATED" and claimed["outdoor_claim_allowed"] is True
    assert claimed["outdoor"]["images"] == 1 and claimed["outdoor"]["locations"] == ["zurich_lake"]

    empty = _csv(tmp_path / "empty_ood.csv", ["image", "label", "location", "session", "sha256"], [])
    with pytest.raises(E.EvalBlocked):
        E.run_canonical_evaluations(checkpoint, {"val": manifest, E.OUTDOOR_REAL_OOD_SET: empty},
                                    runner=lambda command: {})
    unlabeled = _csv(tmp_path / "unlabeled_ood.csv", ["image", "location"],
                     [{"image": "/ood/2.jpg", "location": "zurich_lake"}])
    with pytest.raises(E.EvalBlocked):
        E.run_canonical_evaluations(checkpoint, {"val": manifest, E.OUTDOOR_REAL_OOD_SET: unlabeled},
                                    runner=lambda command: {})
    with pytest.raises(E.EvalBlocked):
        E.run_canonical_evaluations(checkpoint, {"val": tmp_path / "missing.csv"}, runner=lambda command: {})


def test_same_canonical_tp_matcher_as_v2():
    """One shared matcher/AP implementation, and a stored V2 reference must replay identically."""
    assert E.ev is ev
    assert E.match_greedy is ev.match_greedy
    assert E.ap_for_gt_set is ev.ap_for_gt_set
    assert E.equiv_size_640 is ev.equiv_size_640
    assert (E.IMGSZ, E.CONF_OP, E.CONF_FLOOR, E.NMS_IOU, E.MAX_DET, E.PRIMARY_IOU) == (1024, 0.25, 0.001, 0.7, 300, 0.5)

    gt = (0.0, 0.0, 10.0, 10.0)
    dets = [(0.90, [0.0, 0.0, 10.0, 10.0]), (0.80, [100.0, 100.0, 110.0, 110.0])]
    canonical = ev.match_greedy([gt], dets, E.PRIMARY_IOU, metric="iou")
    scored = E.score_predictions([{"key": "a.jpg", "gt": [{"box": gt, "bucket": "4-6"}],
                                   "dets": [{"box": box, "conf": conf} for conf, box in dets]}])
    assert scored["overall"]["gt"] == sum(canonical["gt_hit"]) == 1
    assert scored["overall"]["tp"] == 1 and scored["overall"]["fp"] == 1 and scored["overall"]["fn"] == 0

    # fixture payloads copied from the stored V2 reference (outputs/shuttle_capability/metrics/
    # eth_vs_ours_eth_hardneg_v2.csv); they are not a new measurement and no image is read.
    reference = {"val|eth_unseen": {"images": 1648, "GT": 495, "TP": 101, "FP": 14, "FN": 394,
                                    "mAP50-95": 0.189727}}
    replay = {"val|eth_unseen": dict(reference["val|eth_unseen"])}
    assert E.verify_v2_reference_parity(reference, replay)["status"] == "PASS"
    E.assert_v2_reference_parity(reference, replay)

    drifted = {"val|eth_unseen": dict(reference["val|eth_unseen"], TP=102, FN=393)}
    report = E.verify_v2_reference_parity(reference, drifted)
    assert report["status"] == E.OUTCOME_EVALUATOR_PARITY_FAIL
    assert any("TP" in mismatch for mismatch in report["mismatches"])
    with pytest.raises(E.EvaluatorParityFail):
        E.assert_v2_reference_parity(reference, drifted)
    assert E.verify_v2_reference_parity(reference, {})["status"] == E.OUTCOME_EVALUATOR_PARITY_FAIL
    assert E.verify_v2_reference_parity({}, replay)["status"] == E.OUTCOME_EVALUATOR_PARITY_FAIL


def test_fp_count_not_image_count():
    """On an all-negative set FP is a detection count: reporting the image count instead is refused."""
    detections = [
        [{"box": [0.0, 0.0, 5.0, 5.0], "conf": 0.91}, {"box": [9.0, 9.0, 14.0, 14.0], "conf": 0.30}],
        [{"box": [1.0, 1.0, 6.0, 6.0], "conf": 0.55}],
        [{"box": [2.0, 2.0, 7.0, 7.0], "conf": 0.40}, {"box": [3.0, 3.0, 8.0, 8.0], "conf": 0.10}],
    ]
    stats = E.fp_count(detections, conf_op=0.25)
    assert stats["FP"] == 4 and stats["images"] == 3 and stats["images_with_FP"] == 3
    assert stats["detections"] == 4 and stats["detections_all"] == 5, "the AP floor detection is filtered out"
    assert stats["FP"] != stats["images"]
    assert stats["FP_per_image"] == pytest.approx(4.0 / 3.0)
    assert stats["image_FP_rate"] == pytest.approx(1.0)

    row = E.no_target_row("real_no_target_496", stats)
    assert row["FP"] == 4 and row["images"] == 3 and row["images_with_FP"] == 3
    E.assert_fp_count_not_image_count(row, stats)

    conflated = dict(row)
    conflated["FP"] = row["images"]
    with pytest.raises(E.EvalIntegrityError) as exc:
        E.assert_fp_count_not_image_count(conflated, stats)
    assert "image" in str(exc.value)

    inconsistent = dict(row, images_with_FP=4)
    with pytest.raises(E.EvalIntegrityError):
        E.assert_fp_count_not_image_count(inconsistent, stats)


def test_no_target_496_guard(tmp_path):
    """The frozen no-target pool is exactly 496 unique images and the gate is FP<=3 (V2 historical: 1)."""
    def manifest(count: int, duplicate: bool = False) -> Path:
        rows = [{"image": "/none/target_%04d.jpg" % index} for index in range(count)]
        if duplicate:
            rows[-1] = dict(rows[0])
        return _csv(tmp_path / ("no_target_%d%s.csv" % (count, "_dup" if duplicate else "")), ["image"], rows)

    report = E.assert_no_target_496(manifest(496))
    assert report["images"] == 496 and report["required"] == 496 and report["unique_images"] == 496
    assert report["gate"] == "PASS"

    with pytest.raises(E.EvalIntegrityError) as exc:
        E.assert_no_target_496(manifest(495))
    assert "496" in str(exc.value)
    with pytest.raises(E.EvalIntegrityError):
        E.assert_no_target_496(manifest(496, duplicate=True))
    with pytest.raises(E.EvalBlocked):
        E.assert_no_target_496(tmp_path / "absent.csv")

    assert E.NO_TARGET_POOL_SIZE == 496 and E.NO_TARGET_FP_GATE == 3
    gate = E.no_target_fp_gate(3)
    assert gate["gate"] == "PASS" and gate["historical_v2_FP"] == E.V2_HISTORICAL_NO_TARGET_FP == 1
    E.assert_no_target_fp_gate(3)
    with pytest.raises(E.EvalIntegrityError):
        E.assert_no_target_fp_gate(4)
    with pytest.raises(E.EvalIntegrityError):
        E.no_target_fp_gate(-1)


# -----------------------------------------------------------------------------------------------------------
# Task 10: decision state machine, denominators and provenance report
# -----------------------------------------------------------------------------------------------------------

import report_single_clean_cp_v1 as R  # noqa: E402


def _ood(recall=0.35, mean_ap=0.25, status="PRESENT"):
    return {"status": status, "recall_at_conf_0_25": recall, "map50_95": mean_ap}


def test_leak_invalid_takes_precedence():
    """A leak invalidates the run no matter how good the numbers look."""
    perfect = {"real_no_target_496": {"count": 0, "denominator": 496}, "outdoor_real_ood": _ood(0.99, 0.99)}
    assert R.decide_clean_cp_v1({"violations": [{"kind": "eval_content_hash"}]}, perfect) == R.INVALID_OR_LEAK
    assert R.decide_clean_cp_v1({"status": "INVALID_OR_LEAK"}, perfect) == R.INVALID_OR_LEAK
    assert R.decide_clean_cp_v1({"violations": [{"kind": "coco_id_in_two_new_roles"}],
                                 "gate": "BLOCKED_COCO_INVENTORY"}, perfect) == R.INVALID_OR_LEAK
    assert R.decide_clean_cp_v1({}, perfect) == R.OUTDOOR_CAPABILITY_VERIFIED


def test_missing_real_ood_never_success():
    """Without an independent real outdoor test set the best possible outcome is PILOT_ONLY."""
    metrics = {"eth_unseen_all": {"count": 400, "denominator": 495},
               "real_no_target_496": {"count": 0, "denominator": 496}}
    assert R.decide_clean_cp_v1({}, metrics) == R.PILOT_ONLY
    assert R.decide_clean_cp_v1({"gate": "BLOCKED_TRAINING_PREFLIGHT"}, metrics) == R.PILOT_ONLY
    assert R.decide_clean_cp_v1({}, {**metrics, "outdoor_real_ood": {"status": "MISSING"}}) == R.PILOT_ONLY
    assert R.decide_clean_cp_v1({"gate": "FOREGROUND_INSUFFICIENT"},
                                {**metrics, "outdoor_real_ood": _ood(0.99, 0.99)}) == R.PILOT_ONLY


def test_fp_more_than_three_is_harm():
    """The 496-image frozen no-target pool: more than three false positives is real harm."""
    for count, expected in ((0, R.PILOT_ONLY), (1, R.PILOT_ONLY), (3, R.PILOT_ONLY),
                            (4, R.FP_REGRESSION_OR_REAL_HARM), (9, R.FP_REGRESSION_OR_REAL_HARM)):
        metrics = {"real_no_target_496": {"count": count, "denominator": 496}}
        assert R.decide_clean_cp_v1({}, metrics) == expected, count
    assert R.DEFAULT_THRESHOLDS["historical_v2_fp"] == 1


def test_oood_recall_and_map_both_required():
    """Both the recall floor and the mAP floor must be met; one of them is never enough."""
    base = {"real_no_target_496": {"count": 1, "denominator": 496}}
    assert R.decide_clean_cp_v1({}, {**base, "outdoor_real_ood": _ood(0.30, 0.20)}) == R.OUTDOOR_CAPABILITY_VERIFIED
    assert R.decide_clean_cp_v1({}, {**base, "outdoor_real_ood": _ood(0.29, 0.90)}) == R.BELOW_TARGET
    assert R.decide_clean_cp_v1({}, {**base, "outdoor_real_ood": _ood(0.90, 0.19)}) == R.BELOW_TARGET
    with pytest.raises(ValueError):
        R.decide_clean_cp_v1({}, {**base, "outdoor_real_ood": {"status": "PRESENT", "recall_at_conf_0_25": 0.5}})


def test_report_never_claims_copy_paste_caused_gain():
    """A single-arm design cannot attribute the effect; the report must not even imply that it can."""
    report = R.render_report({"decision": R.PILOT_ONLY,
                              "gates": {"foreground": "FOREGROUND_INSUFFICIENT", "source contract": "PASS"},
                              "metrics": {}})
    assert "PILOT_ONLY" in report and "descriptive only" in report
    R.assert_no_causal_language(report)

    for claim in ("Copy-Paste caused the mAP gain over V2.",
                  "the improvement comes from the new hard negatives",
                  "the gain is due to copy-paste",
                  "proves that copy-paste works"):
        with pytest.raises(ValueError):
            R.assert_no_causal_language(claim)


def test_all_results_have_denominators():
    R.assert_denominators({"real_no_target_496": {"count": 1, "denominator": 496, "per_image": 0.002}})
    with pytest.raises(ValueError):
        R.assert_denominators({"real_no_target_496": {"per_image": 0.002}})
    with pytest.raises(ValueError):
        R.assert_denominators({"eth_unseen_all": {"rate": 0.8}})

    report = R.render_report({"decision": R.BELOW_TARGET,
                              "metrics": {"real_no_target_496": {"count": 4, "denominator": 496, "per_image": 0.008}}})
    assert "raw 4 / 496" in report
    assert "not measured" in report, "an unmeasured set is reported as unmeasured, not as zero"


def test_no_artifact_listed_as_trained_without_run_evidence():
    with pytest.raises(ValueError):
        R.assert_run_evidence({})
    with pytest.raises(ValueError):
        R.assert_run_evidence({"status": "COMPLETED", "manifest": "m.json", "epochs_recorded": 50})
    with pytest.raises(ValueError):
        R.assert_run_evidence({"status": "NOT_EXECUTED", "checkpoint_sha256": "x", "manifest": "m",
                               "epochs_recorded": 50})
    with pytest.raises(ValueError):
        R.assert_run_evidence({"status": "COMPLETED", "checkpoint_sha256": "x", "manifest": "m",
                               "epochs_recorded": 0})
    R.assert_run_evidence({"status": "COMPLETED", "checkpoint_sha256": "x", "manifest": "m", "epochs_recorded": 50})


def test_artifact_manifest_hashes_every_deliverable(tmp_path):
    manifest = R.build_artifact_manifest([C.NEW_CONFIG_PATH, "does/not/exist.json"], repo_root=REPO)
    assert manifest["artifacts"][0]["path"] == C.NEW_CONFIG_PATH
    assert manifest["artifacts"][0]["sha256"] == C.sha256_file(REPO / C.NEW_CONFIG_PATH)
    assert manifest["artifacts"][0]["bytes"] > 0
    assert manifest["missing"] == ["does/not/exist.json"]


def test_collect_gates_reads_every_audit_and_never_defaults_to_pass(tmp_path):
    """The final decision is driven by the audit files themselves, and an unreadable file is not a pass."""
    (tmp_path / "repo_contract_map.json").write_text('{"gate": {"status": "PASS"}}', encoding="utf-8")
    (tmp_path / "foreground_audit.json").write_text('{"gate": "FOREGROUND_INSUFFICIENT"}', encoding="utf-8")
    (tmp_path / "coco_provenance_audit.json").write_text('{"gate": {"status": "BLOCKED_COCO_INVENTORY"}}',
                                                         encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    gates = R.collect_gates(tmp_path)
    assert gates["repo_contract_map.json"] == "PASS"
    assert gates["foreground_audit.json"] == "FOREGROUND_INSUFFICIENT"
    assert gates["coco_provenance_audit.json"] == "BLOCKED_COCO_INVENTORY"
    assert gates["broken.json"] == "UNREADABLE"
    assert R.decide_clean_cp_v1({"gates": gates}, {}) == R.PILOT_ONLY

    only_pass = tmp_path / "good"
    only_pass.mkdir()
    (only_pass / "repo_contract_map.json").write_text('{"gate": {"status": "PASS"}}', encoding="utf-8")
    assert R.decide_clean_cp_v1({"gates": R.collect_gates(only_pass)}, {}) == R.PILOT_ONLY, \
        "even all-PASS gates cannot verify outdoor capability without a real OOD set"


