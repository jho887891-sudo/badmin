"""Tests for tools/build_tiny_recovery_v1_dataset.py (tiny-object recovery V1 data gates)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_tiny_recovery_v1_dataset as B  # noqa: E402

AUDIT = REPO / "outputs" / "shuttle_capability" / "metrics" / "tiny_recovery_v1_data_audit.json"
EXPOSURE = REPO / "outputs" / "shuttle_capability" / "metrics" / "tiny_recovery_v1_tiny_exposure.csv"
V2_TRAIN = REPO / "data" / "eth_real_hardneg_v2_train_manifest.csv"
V2_VAL = REPO / "data" / "eth_real_hardneg_v2_val_manifest.csv"


def _row(sha, eq, positive=True, source="eth_main", role=None):
    return {"image": "/r/%s.jpg" % sha[:8], "label": "" if not positive else "/r/%s.txt" % sha[:8],
            "location": "cab_1", "difficulty": "easy", "split": "train", "set_dir": "cab_1_easy",
            "role": role or ("official_positives" if positive else "hard_negative"),
            "in_official_training": "True", "source": source, "is_negative": not positive, "n_boxes": 1,
            "width": 1920, "height": 1200, "sha256": sha, "equiv_size_640": eq}


def _frame(n_per_bucket=(3, 4, 5)):
    rows = []
    i = 0
    for count, eq in zip(n_per_bucket, (3.0, 5.0, 7.0)):
        for _ in range(count):
            i += 1
            rows.append(_row("%064x" % i, eq))
    for _ in range(4):                      # non-tiny positives
        i += 1
        rows.append(_row("%064x" % i, 12.0))
    for _ in range(2):                      # negatives
        i += 1
        rows.append(_row("%064x" % i, 0.0, positive=False))
    df = pd.DataFrame(rows)
    df["is_negative"] = df["is_negative"].astype(bool)
    df["is_positive"] = ~df["is_negative"]
    df["extra_exposure"] = 0
    df["selection_seed"] = 0
    return df


def test_bucket_boundaries():
    assert [B.bucket_of_tiny(x) for x in (0.0, 3.99, 4.0, 5.99, 6.0, 7.99)] == [
        "<4", "<4", "4-6", "4-6", "6-8", "6-8"]


def test_tiny_pool_keeps_only_positive_rows_below_the_cutoff():
    pool = B.tiny_pool(_frame())
    assert len(pool) == 12
    assert set(pool["size_bucket"]) == {"<4", "4-6", "6-8"}
    assert (pool["equiv_size_640"].astype(float) < 8.0).all()


def test_allocate_stratified_is_exact_and_proportional():
    # raw shares are 1.5 / 2.0 / 2.5; floors leave one seat and the 0.5 tie goes to the earliest bucket.
    alloc = B.allocate_stratified({"<4": 3, "4-6": 4, "6-8": 5}, 6)
    assert sum(alloc.values()) == 6
    assert alloc == {"<4": 2, "4-6": 2, "6-8": 2}
    assert sum(B.allocate_stratified({"<4": 161, "4-6": 1282, "6-8": 4522}, 712).values()) == 712


def test_allocate_raises_instead_of_oversampling_when_the_pool_is_too_small():
    with pytest.raises(B.TinyRecoveryAuditError, match="forbids"):
        B.allocate_stratified({"<4": 1, "4-6": 1, "6-8": 1}, 712)


def test_selection_is_deterministic_unique_and_respects_the_allocation():
    pool = B.tiny_pool(_frame())
    alloc = {"<4": 1, "4-6": 2, "6-8": 2}
    a = B.select_tiny_exposures(pool, alloc, 42)
    b = B.select_tiny_exposures(pool, alloc, 42)
    assert list(a["sha256"]) == list(b["sha256"])
    assert len(a) == 5 and a["sha256"].nunique() == 5
    assert dict(a["size_bucket"].value_counts()) == alloc


def test_selection_raises_when_a_bucket_cannot_fund_its_share():
    pool = B.tiny_pool(_frame())
    with pytest.raises(B.TinyRecoveryAuditError, match="only 3 images are eligible|needs"):
        B.select_tiny_exposures(pool, {"<4": 99, "4-6": 0, "6-8": 0}, 42)


def test_build_manifest_adds_exactly_the_selected_rows():
    df = _frame()
    pool = B.tiny_pool(df)
    selected = B.select_tiny_exposures(pool, {"<4": 1, "4-6": 1, "6-8": 1}, 42)
    tiny = B.build_tiny_train_manifest(df, selected)
    assert len(tiny) == len(df) + 3
    assert int(tiny["extra_exposure"].sum()) == 3
    assert tiny[tiny["extra_exposure"] == 1]["sha256"].nunique() == 3


def test_exposure_table_accounts_base_and_extra_for_every_pool_image():
    df = _frame()
    selected = B.select_tiny_exposures(B.tiny_pool(df), {"<4": 2, "4-6": 0, "6-8": 0}, 42)
    table = B.exposure_table(df, selected)
    assert len(table) == 12
    assert all(r["base_exposure"] == 1 for r in table)
    assert sum(r["extra_exposure"] for r in table) == 2
    assert all(r["final_exposure"] == r["base_exposure"] + r["extra_exposure"] for r in table)
    assert all(r["selection_seed"] == 42 for r in table)
    assert set(table[0]) == {"image", "sha256", "location", "equiv_size_640", "size_bucket", "base_exposure",
                             "extra_exposure", "final_exposure", "selection_seed"}


def test_exposure_summary_has_the_four_design_rows():
    df = _frame()
    summary = B.exposure_summary(df, B.select_tiny_exposures(B.tiny_pool(df), {"<4": 1, "4-6": 1, "6-8": 1}, 42))
    assert [r["bucket"] for r in summary] == ["<4", "4-6", "6-8", ">=8"]
    assert summary[-1]["extra_exposure"] == 0 and summary[-1]["v2_exposure"] == 4
    assert sum(r["extra_exposure"] for r in summary) == 3


def _audit(df, tiny, **kw):
    args = dict(eval_paths=set(), eval_shas=set(), budget=int(tiny["extra_exposure"].sum()),
                v2_val_manifest=None, tiny_val_manifest=None)
    args.update(kw)
    return B.audit_tiny_dataset(df, None, tiny, tiny[tiny["extra_exposure"] == 1], **args)


def test_audit_passes_on_a_clean_frame():
    df = _frame()
    selected = B.select_tiny_exposures(B.tiny_pool(df), {"<4": 1, "4-6": 1, "6-8": 1}, 42)
    rep = _audit(df, B.build_tiny_train_manifest(df, selected))
    assert rep["status"] == "PASS" and rep["problems"] == []
    assert rep["unique_positive_set_identical"] is True
    assert rep["extra_non_tiny_positive"] == 0 and rep["tiny_extra_per_unique_image_max"] == 1


def test_audit_blocks_positive_set_drift():
    df = _frame()
    tiny = B.build_tiny_train_manifest(df, B.select_tiny_exposures(B.tiny_pool(df), {"<4": 1, "4-6": 0, "6-8": 0}, 42))
    tiny.loc[tiny["extra_exposure"] == 1, "sha256"] = "f" * 64
    with pytest.raises(B.TinyRecoveryAuditError, match="positive image set drifted"):
        _audit(df, tiny)


def test_audit_blocks_a_non_tiny_extra_exposure():
    df = _frame()
    tiny = B.build_tiny_train_manifest(df, B.select_tiny_exposures(B.tiny_pool(df), {"<4": 1, "4-6": 0, "6-8": 0}, 42))
    extra_idx = list(tiny.index[tiny["extra_exposure"] == 1])[:1]
    tiny.loc[extra_idx, "equiv_size_640"] = 20.0
    with pytest.raises(B.TinyRecoveryAuditError, match="non-tiny"):
        _audit(df, tiny)


def test_audit_blocks_more_than_one_extra_copy_of_an_image():
    df = _frame()
    selected = B.select_tiny_exposures(B.tiny_pool(df), {"<4": 1, "4-6": 0, "6-8": 0}, 42)
    tiny = B.build_tiny_train_manifest(df, selected)
    tiny = pd.concat([tiny, selected.assign(extra_exposure=1)], ignore_index=True, sort=False)
    with pytest.raises(B.TinyRecoveryAuditError, match="more than one extra exposure"):
        _audit(df, tiny)


def test_audit_blocks_evaluation_hash_overlap():
    df = _frame()
    selected = B.select_tiny_exposures(B.tiny_pool(df), {"<4": 1, "4-6": 0, "6-8": 0}, 42)
    tiny = B.build_tiny_train_manifest(df, selected)
    hn_sha = set(tiny[tiny["role"] == "hard_negative"]["sha256"])
    with pytest.raises(B.TinyRecoveryAuditError, match="evaluation overlap"):
        _audit(df, tiny, eval_shas={sorted(hn_sha)[0]})


def test_audit_blocks_a_wrong_budget():
    df = _frame()
    tiny = B.build_tiny_train_manifest(df, B.select_tiny_exposures(B.tiny_pool(df), {"<4": 1, "4-6": 0, "6-8": 0}, 42))
    with pytest.raises(B.TinyRecoveryAuditError, match="budget"):
        _audit(df, tiny, budget=712)


def test_real_artifacts_match_the_design():
    if not AUDIT.is_file():
        pytest.skip("tiny recovery artifacts not built yet")
    rep = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert rep["status"] == "PASS" and rep["problems"] == []
    assert rep["tiny_train_rows"] == 15967 and rep["v2_train_rows"] == 15255
    assert rep["extra_tiny_exposure"] == 712 and rep["extra_non_tiny_positive"] == 0
    assert rep["tiny_extra_per_unique_image_max"] == 1
    assert rep["allocation"] == {"<4": 19, "4-6": 153, "6-8": 540}
    assert rep["unique_positive_set_identical"] is True
    assert rep["eval_path_overlap_count"] == 0 and rep["eval_sha256_overlap_count"] == 0
    assert rep["v2_val_manifest_sha256"] == rep["tiny_val_manifest_sha256"]
    summary = {r["bucket"]: r for r in rep["exposure_summary"]}
    assert summary["<4"]["extra_exposure"] == 19 and summary["4-6"]["extra_exposure"] == 153
    assert summary["6-8"]["extra_exposure"] == 540 and summary[">=8"]["extra_exposure"] == 0
    rows = list(csv.DictReader(EXPOSURE.open(encoding="utf-8")))
    assert len(rows) == 5965 and sum(int(r["extra_exposure"]) for r in rows) == 712
    assert all(int(r["final_exposure"]) <= 2 for r in rows)
