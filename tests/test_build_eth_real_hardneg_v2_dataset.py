"""Tests for tools/build_eth_real_hardneg_v2_dataset.py (Task 2)."""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_eth_real_hardneg_v2_dataset as B  # noqa: E402

V1_TRAIN = REPO / "data" / "eth_only_v1_train_manifest.csv"
V1_VAL = REPO / "data" / "eth_only_v1_val_manifest.csv"
AUDIT = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_real_hardneg_v2_data_audit.json"
CANDIDATES = (REPO / "outputs" / "shuttle_capability" / "metrics"
              / "eth_real_hardneg_v2_hardneg_candidates.csv")


def _row(loc, sha, is_negative=False, source="eth_main"):
    r = {c: "" for c in B.V1_MANIFEST_COLUMNS}
    r.update({
        "image": "/remote/%s/%s.jpg" % (loc, sha[:6]),
        "label": "" if is_negative else "/remote/%s/%s.txt" % (loc, sha[:6]),
        "location": loc, "difficulty": "easy", "split": "train", "set_dir": loc + "_easy",
        "role": "official_positives" if not is_negative else "official_negatives",
        "in_official_training": "True" if not is_negative else "False",
        "source": source, "is_negative": is_negative, "n_boxes": 0 if is_negative else 1,
        "width": 1920, "height": 1200, "sha256": sha, "equiv_size_640": 0.0,
    })
    return r


@pytest.fixture()
def v1_train():
    rows = [_row("cab_1", "%064x" % i) for i in range(1, 7)]
    rows += [_row("coco_train", "%064x" % i, is_negative=True, source="eth_negative") for i in (7, 8)]
    return B.load_v1_manifest_from_frame(pd.DataFrame(rows)) if hasattr(B, "load_v1_manifest_from_frame") \
        else _coerce(pd.DataFrame(rows))


def _coerce(df):
    df["is_negative"] = df["is_negative"].astype(str).str.lower().isin(["true", "1"])
    df["is_positive"] = ~df["is_negative"]
    return df


@pytest.fixture()
def v1_val():
    return _coerce(pd.DataFrame([_row("glc_2", "%064x" % 100)]))


@pytest.fixture()
def hardneg():
    rows = []
    for i, (pool, name) in enumerate([("hard_negatives", "hn_001.jpg"), ("hard_negatives2", "hn2_001.jpg"),
                                      ("hard_negatives2", "hn2_002.jpg")]):
        rows.append({"source": pool, "image": "/remote/%s/raw/%s" % (pool, name), "file": name,
                     "sha256": "%064x" % (200 + i), "bytes": 1000 + i, "width": 1280, "height": 853,
                     "label_nonempty": False, "is_negative": True, "provenance_present": True,
                     "provenance_title": "t", "provenance_license": "l", "provenance_url": "u"})
    return pd.DataFrame(rows)


def test_v2_positive_rows_equal_v1_positive_rows(v1_train, hardneg):
    v2 = B.build_v2_train_manifest(v1_train, hardneg, 1)
    v1_pos = v1_train[v1_train["is_positive"]]
    v2_pos = v2[v2["is_positive"]]
    assert len(v2_pos) == len(v1_pos)
    assert set(v2_pos["sha256"]) == set(v1_pos["sha256"])
    assert B.hash_set_digest(v2_pos["sha256"]) == B.hash_set_digest(v1_pos["sha256"])


def test_repeat_creates_exactly_that_many_copies_per_unique_negative(v1_train, hardneg):
    v2 = B.build_v2_train_manifest(v1_train, hardneg, 8)
    neg = v2[~v2["is_positive"]]
    assert len(neg) == 8 * len(hardneg) + int((~v1_train["is_positive"]).sum())
    hn = neg[neg["role"] == "hard_negative"]
    assert len(hn) == 24
    assert sorted(set(hn["repeat_index"])) == list(range(8))
    assert hn["sha256"].nunique() == 3
    assert hn["image"].nunique() == 3


def test_repeat_must_be_positive(v1_train, hardneg):
    with pytest.raises(B.V2DatasetAuditError):
        B.build_v2_train_manifest(v1_train, hardneg, 0)


def test_hardneg_must_be_real_empty_label(v1_train, v1_val, hardneg):
    bad = hardneg.copy()
    bad.loc[0, "label_nonempty"] = True
    with pytest.raises(B.V2DatasetAuditError, match="non-empty labels"):
        B.audit_v2_dataset(v1_train, v1_val, bad, set(), set(), 8)


def test_missing_provenance_is_rejected(hardneg):
    bad = hardneg.copy()
    bad.loc[1, "provenance_present"] = False
    path = REPO / "_tmp_candidates_bad_prov.csv"
    bad.to_csv(path, index=False)
    try:
        with pytest.raises(B.V2DatasetAuditError, match="provenance"):
            B.inventory_hard_negatives(path, ["hard_negatives", "hard_negatives2"])
    finally:
        path.unlink()


def test_prohibited_source_is_rejected(hardneg):
    bad = hardneg.copy()
    bad.loc[0, "source"] = "synthetic_3d"
    path = REPO / "_tmp_candidates_bad_source.csv"
    bad.to_csv(path, index=False)
    try:
        with pytest.raises(B.V2DatasetAuditError):
            B.inventory_hard_negatives(path, ["hard_negatives", "hard_negatives2", "synthetic_3d"])
    finally:
        path.unlink()


def test_source_outside_the_frozen_list_is_rejected(hardneg):
    path = REPO / "_tmp_candidates_extra_source.csv"
    hardneg.to_csv(path, index=False)
    try:
        with pytest.raises(B.V2DatasetAuditError, match="outside the frozen list"):
            B.inventory_hard_negatives(path, ["hard_negatives"])
    finally:
        path.unlink()


def test_repeat_or_weight_fields_are_rejected(hardneg):
    bad = hardneg.copy()
    bad["repeat"] = 3
    path = REPO / "_tmp_candidates_weighted.csv"
    bad.to_csv(path, index=False)
    try:
        with pytest.raises(B.V2DatasetAuditError, match="weighting columns"):
            B.inventory_hard_negatives(path, ["hard_negatives", "hard_negatives2"])
    finally:
        path.unlink()


def test_eval_hash_overlap_blocks_training(v1_train, v1_val, hardneg):
    with pytest.raises(B.V2DatasetAuditError, match="SHA256 overlap"):
        B.audit_v2_dataset(v1_train, v1_val, hardneg, set(), {hardneg.loc[0, "sha256"]}, 8)


def test_eval_path_overlap_blocks_training(v1_train, v1_val, hardneg):
    with pytest.raises(B.V2DatasetAuditError, match="path overlap"):
        B.audit_v2_dataset(v1_train, v1_val, hardneg, {hardneg.loc[0, "image"]}, set(), 8)


def test_location_overlap_blocks_training(v1_train, hardneg):
    val = _coerce(pd.DataFrame([_row("cab_1", "%064x" % 300)]))
    with pytest.raises(B.V2DatasetAuditError, match="location overlap"):
        B.audit_v2_dataset(v1_train, val, hardneg, set(), set(), 8)


def test_hardneg_duplicates_collapse_to_one():
    dup = pd.DataFrame([
        {"source": "hard_negatives2", "image": "/r/a.jpg", "sha256": "a" * 64, "label_nonempty": False,
         "is_negative": True, "provenance_present": True},
        {"source": "hard_negatives2", "image": "/r/b.jpg", "sha256": "a" * 64, "label_nonempty": False,
         "is_negative": True, "provenance_present": True},
        {"source": "hard_negatives2", "image": "/r/c.jpg", "sha256": "c" * 64, "label_nonempty": False,
         "is_negative": True, "provenance_present": True},
    ])
    out = B.dedupe_hard_negatives(dup)
    assert len(out) == 2
    assert out["sha256"].nunique() == len(out) and out["image"].nunique() == len(out)


def test_audit_reports_the_required_fields(v1_train, v1_val, hardneg):
    rep = B.audit_v2_dataset(v1_train, v1_val, hardneg, set(), set(), 8)
    required = ("v1_positive_count", "v1_positive_hash_set_sha256", "v2_positive_count",
                "v2_positive_hash_set_sha256", "hardneg_raw_count", "hardneg_unique_count",
                "hardneg_by_source", "hardneg_nonempty_label_count", "hardneg_unreadable_count",
                "duplicate_path_count", "duplicate_sha256_count", "eval_path_overlap_count",
                "eval_sha256_overlap_count", "train_val_location_overlap_count",
                "prohibited_source_counts", "v1_val_manifest_sha256", "v2_val_manifest_sha256", "status")
    for key in required:
        assert key in rep, key
    assert rep["status"] == "PASS" and rep["problems"] == []
    assert rep["v1_positive_hash_set_sha256"] == rep["v2_positive_hash_set_sha256"]


def test_val_manifest_is_a_byte_copy_of_the_v1_manifest():
    if not AUDIT.is_file():
        pytest.skip("V2 artifacts not built yet")
    import json
    rep = json.loads(AUDIT.read_text(encoding="utf-8"))
    v1 = hashlib.sha256(V1_VAL.read_bytes()).hexdigest()
    v2 = hashlib.sha256((REPO / "data" / "eth_real_hardneg_v2_val_manifest.csv").read_bytes()).hexdigest()
    assert v1 == v2 == rep["v1_val_manifest_sha256"] == rep["v2_val_manifest_sha256"]


def test_real_v2_artifacts_pass_every_blocking_rule():
    if not AUDIT.is_file():
        pytest.skip("V2 artifacts not built yet")
    import json
    rep = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert rep["status"] == "PASS" and rep["problems"] == []
    assert rep["hardneg_unique_count"] == 89 and rep["hardneg_raw_count"] == 89
    assert rep["excluded_names_count"] == 7
    assert rep["hard_negative_repeat"] == 8
    assert rep["negative_exposure_rows"] == 89 * 8
    assert rep["eval_path_overlap_count"] == 0 and rep["eval_sha256_overlap_count"] == 0
    assert rep["train_val_location_overlap_count"] == 0
    assert rep["hardneg_nonempty_label_count"] == 0 and rep["hardneg_unreadable_count"] == 0
    assert rep["dup" + "licate_path_count"] == 0 and rep["duplicate_sha256_count"] == 0
    assert rep["v1_positive_hash_set_sha256"] == rep["v2_positive_hash_set_sha256"]
    train = pd.read_csv(REPO / "data" / "eth_real_hardneg_v2_train_manifest.csv", keep_default_na=False,
                        dtype=str)
    assert len(train) == rep["train_rows"]
    hn = train[train["role"] == "hard_negative"]
    assert hn["sha256"].nunique() == 89 and len(hn) == 712
    assert set(hn["source"]) == {"hard_negative"}
    assert CANDIDATES.is_file()
    cand = pd.read_csv(CANDIDATES, keep_default_na=False, dtype=str)
    assert len(cand) == 89 and cand["sha256"].nunique() == 89
