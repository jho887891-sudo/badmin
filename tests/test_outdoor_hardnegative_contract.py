"""Task 5 tests: COCO train2017 outdoor hard negatives must be human-reviewed, empty-labelled, authorised
pictures that stay disjoint from the clean-background pool and from every historical V2 / frozen eval source.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 5, Step 1/2).
This file was written before tools/audit_outdoor_hardnegatives.py existed, so the first run was RED
(ModuleNotFoundError). Every fixture is synthetic; no human review, licence or COCO image is invented here.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import audit_outdoor_hardnegatives as H  # noqa: E402
import clean_cp_contract as C  # noqa: E402
import index_coco_train2017 as I  # noqa: E402

NEW_CONFIG = REPO / "configs" / "eth_real_clean_outdoor_copypaste_v1.yaml"
ZERO_SHA = "0" * 64


# --- synthetic COCO train2017 fixtures ------------------------------------------------------------

def _write_image(path: Path, colour) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path)
    return path


def _candidate(root: Path, coco_id: int, **overrides) -> dict:
    """One reviewed-candidate row whose image and (by default zero-byte) label exist on disk."""
    image = Path(root) / "train2017" / ("%012d.jpg" % coco_id)
    if not image.is_file():
        _write_image(image, (coco_id % 251, (coco_id * 3) % 251, (coco_id * 7) % 251))
    label = image.with_suffix(".txt")
    label.write_text(overrides.pop("label_text", ""), encoding="utf-8")
    if overrides.pop("label_missing", False):
        label.unlink()
    row = {
        "coco_image_id": coco_id,
        "file_name": image.name,
        "split": "train2017",
        "path": str(image),
        "width": 16,
        "height": 16,
        "license_name": "Attribution License",
        "license_url": "https://creativecommons.org/licenses/by/2.0/",
        "permission_state": "declared",
        "reviewer_id": "reviewer-fixture",
        "no_shuttle_reviewed": True,
        "review_uncertain_tiny_dot": False,
        "reviewed_category": "white_reflection",
        "source_cluster": "cluster-a",
        "qa_status": "APPROVED",
    }
    row.update(overrides)
    return row


def _empty_excluded() -> pd.DataFrame:
    return I.empty_prohibited_index()


def _empty_clean_backgrounds() -> pd.DataFrame:
    return pd.DataFrame(columns=["coco_image_id", "sha256", "near_duplicate_group"])


def _run_audit(rows, excluded=None, clean_backgrounds=None) -> dict:
    inventory = H.inventory_outdoor_hardnegatives(pd.DataFrame(rows))
    return H.audit_outdoor_hardnegatives(
        inventory,
        _empty_excluded() if excluded is None else excluded,
        _empty_clean_backgrounds() if clean_backgrounds is None else clean_backgrounds,
    )


def _rejection(result: dict, coco_id: int):
    frame = result["rejected"]
    if frame.empty:
        return None
    match = frame[frame["coco_image_id"].astype(str) == str(int(coco_id))]
    return None if match.empty else match.iloc[0].to_dict()


def _accepted_ids(result: dict) -> set:
    return {int(value) for value in result["accepted"]["coco_image_id"]}


# --- required tests -------------------------------------------------------------------------------

def test_only_coco_train2017_hardnegative(tmp_path):
    """val2017/test2017 may never be inventoried, and a hand-built row claiming another split is rejected."""
    val_image = _write_image(tmp_path / "val2017" / "000000000007.jpg", (9, 9, 9))
    val_label = val_image.with_suffix(".txt")
    val_label.write_text("", encoding="utf-8")
    val_row = _candidate(tmp_path, 101)
    val_row["split"] = "val2017"
    val_row["file_name"] = val_image.name
    val_row["path"] = str(val_image)

    with pytest.raises(H.OutdoorHardNegBlocked):
        H.inventory_outdoor_hardnegatives(pd.DataFrame([val_row]))

    train_row = _candidate(tmp_path, 102)
    inventory = H.inventory_outdoor_hardnegatives(pd.DataFrame([train_row]))
    assert set(inventory["split"]) == {"train2017"}
    assert set(inventory["role"]) == {C.ROLE_COCO_OUTDOOR_HARDNEG}

    result = H.audit_outdoor_hardnegatives(pd.DataFrame([val_row]), _empty_excluded(), _empty_clean_backgrounds())
    assert result["status"] == "INVALID_OR_LEAK"
    assert result["gate"] == "INVALID_OR_LEAK"
    assert _rejection(result, 101)["reason"] == "not_coco_train2017"
    assert _accepted_ids(result) == set()


def test_no_coco_shuttle_class_does_not_imply_empty_label(tmp_path):
    """COCO has no shuttlecock class: an absent annotation is not a human 'no shuttle' review."""
    annotation_only = _candidate(tmp_path, 201, no_shuttle_reviewed="", qa_status="",
                                 coco_annotation_no_shuttle=True)
    declined = _candidate(tmp_path, 202, no_shuttle_reviewed=False)
    control = _candidate(tmp_path, 203)

    result = _run_audit([annotation_only, declined, control])
    assert _rejection(result, 201)["reason"] == "no_shuttle_review_missing"
    assert _rejection(result, 202)["reason"] == "shuttle_present_or_uncertain"
    assert _accepted_ids(result) == {203}
    assert "absence" in result["note"] or "no shuttlecock" in result["note"]


def test_review_uncertain_tiny_dot_rejected(tmp_path):
    """A reviewer who flags an uncertain tiny white dot has not approved the picture: it must be excluded."""
    uncertain = _candidate(tmp_path, 301, review_uncertain_tiny_dot=True)
    unsure = _candidate(tmp_path, 302, review_uncertain_tiny_dot=True, no_shuttle_reviewed=False)
    control = _candidate(tmp_path, 303)

    result = _run_audit([uncertain, unsure, control])
    assert _rejection(result, 301)["reason"] == "uncertain_tiny_dot_review"
    assert _rejection(result, 302)["reason"] == "shuttle_present_or_uncertain"
    assert _accepted_ids(result) == {303}


def test_nonempty_yolo_label_rejected(tmp_path):
    """A hard negative is a YOLO .txt that exists and is ZERO bytes; a boxed label is a positive, not a negative."""
    nonempty = _candidate(tmp_path, 401, label_text="0 0.5 0.5 0.1 0.1\n")
    missing = _candidate(tmp_path, 402, label_missing=True)
    control = _candidate(tmp_path, 403)

    result = _run_audit([nonempty, missing, control])
    assert _rejection(result, 401)["reason"] == "nonempty_label"
    assert _rejection(result, 402)["reason"] == "label_missing"
    accepted = result["accepted"]
    assert _accepted_ids(result) == {403}
    assert int(accepted["label_bytes"].iloc[0]) == 0
    assert Path(accepted["label_path"].iloc[0]).stat().st_size == 0
    assert accepted["n_boxes"].iloc[0] == 0
    assert not bool(accepted["is_positive"].iloc[0])


def test_same_coco_id_or_nearduplicate_as_clean_bg_rejected(tmp_path):
    """The same COCO id, the same bytes or the same near-duplicate cluster may not be a background and a negative."""
    control = _candidate(tmp_path, 501)
    same_id = _candidate(tmp_path, 502)
    same_bytes = _candidate(tmp_path, 503)
    same_group = _candidate(tmp_path, 504, near_duplicate_group="nd-clean-1")
    unrelated = _candidate(tmp_path, 505)

    clean_backgrounds = pd.DataFrame([
        {"coco_image_id": 502, "sha256": H.sha256_file(unrelated["path"]), "near_duplicate_group": ""},
        {"coco_image_id": 506, "sha256": H.sha256_file(same_bytes["path"]), "near_duplicate_group": ""},
        {"coco_image_id": 507, "sha256": ZERO_SHA, "near_duplicate_group": "nd-clean-1"},
    ])
    result = _run_audit([control, same_id, same_bytes, same_group], clean_backgrounds=clean_backgrounds)
    assert _rejection(result, 502)["reason"] == "coco_image_id_in_clean_bg"
    assert _rejection(result, 503)["reason"] == "content_hash_in_clean_bg"
    assert _rejection(result, 504)["reason"] == "near_duplicate_with_clean_bg"
    assert _accepted_ids(result) == {501}


def test_previously_used_v2_coco_negative_rejected(tmp_path):
    """A COCO image the frozen V2 work already trained on (by id or by bytes) may never re-enter as NEW data."""
    reused_id = _candidate(tmp_path, 601)
    reused_bytes = _candidate(tmp_path, 602)
    control = _candidate(tmp_path, 603)
    excluded = I.build_prohibited_source_index(
        historical_coco_ids={601},
        historical_sha256={I.sha256_file(reused_bytes["path"])},
        historical_coco_id_completeness="KNOWN",
    )
    result = _run_audit([reused_id, reused_bytes, control], excluded=excluded)
    assert _rejection(result, 601)["reason"] == "previously_used_v2_coco_negative"
    assert _rejection(result, 602)["reason"] == "previously_used_v2_coco_negative"
    assert _accepted_ids(result) == {603}

    # a frozen evaluation image is not a pool shortfall: it is a leak and must stop the whole audit
    frozen_image = _candidate(tmp_path, 604)
    frozen_manifest = tmp_path / "frozen_eval_manifest.csv"
    frozen_manifest.write_text(
        "image,sha256,is_negative,role,set_dir,location\n%s,%s,False,frozen_eval,frozen_eval,\n"
        % (frozen_image["path"], I.sha256_file(frozen_image["path"])),
        encoding="utf-8",
    )
    frozen = I.build_prohibited_source_index(frozen_eval_manifests=[frozen_manifest])
    leak = _run_audit([frozen_image], excluded=frozen)
    assert leak["status"] == "INVALID_OR_LEAK"
    assert _rejection(leak, 604)["reason"] == "frozen_eval_overlap"


def test_missing_permission_rejected(tmp_path):
    """An image whose licence/permission cannot be resolved is rejected, never silently assumed reusable."""
    unknown = _candidate(tmp_path, 701, permission_state="unknown", license_name="", license_url="")
    blank = _candidate(tmp_path, 702, permission_state="", license_name="", license_url="")
    control = _candidate(tmp_path, 703)

    result = _run_audit([unknown, blank, control])
    assert _rejection(result, 701)["reason"] == "license_unverifiable"
    assert _rejection(result, 702)["reason"] == "license_unverifiable"
    assert _accepted_ids(result) == {703}
    assert result["accepted"]["license_or_permission"].iloc[0]
    assert result["accepted"]["permission_state"].iloc[0] == "declared"


def test_eval_fp_never_mined(tmp_path):
    """Optional V2 false-positive mining is read-only and can only ever be pointed at train-only COCO images."""
    checkpoint = tmp_path / "eth_real_hardneg_v2_best.pt"
    checkpoint.write_bytes(b"frozen v2 checkpoint fixture, never a real weight")
    val_dir = tmp_path / "val2017"
    _write_image(val_dir / "000000000001.jpg", (1, 2, 3))
    frozen_dir = tmp_path / "frozen_eval_496"
    _write_image(frozen_dir / "000000000002.jpg", (4, 5, 6))
    eth_dir = tmp_path / "eth_unseen_frames"
    _write_image(eth_dir / "frame_0001.jpg", (7, 8, 9))
    train_dir = tmp_path / "train2017"
    _write_image(train_dir / "000000000003.jpg", (10, 11, 12))
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    with pytest.raises(H.OutdoorHardNegBlocked):
        H.mine_v2_fp_candidates(val_dir, checkpoint)
    with pytest.raises(H.OutdoorHardNegBlocked):
        H.mine_v2_fp_candidates(frozen_dir, checkpoint)
    with pytest.raises(H.OutdoorHardNegBlocked):
        H.mine_v2_fp_candidates(eth_dir, checkpoint)

    # there is deliberately no manifest/evaluation argument to hand this function
    assert "manifest" not in inspect.signature(H.mine_v2_fp_candidates).parameters
    assert "eval" not in " ".join(inspect.signature(H.mine_v2_fp_candidates).parameters)

    with pytest.raises(H.OutdoorHardNegBlocked):
        H.mine_v2_fp_candidates(train_dir, tmp_path / "absent_v2_best.pt")

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert after == before, "mining must not write anything: it is a read-only review aid"


def test_reviewed_category_required(tmp_path):
    """Every accepted hard negative carries one reviewed distractor category from the frozen enum."""
    no_category = _candidate(tmp_path, 801, reviewed_category="")
    invented = _candidate(tmp_path, 802, reviewed_category="shiny_thing")
    control = _candidate(tmp_path, 803, reviewed_category="feather_leaf")

    result = _run_audit([no_category, invented, control])
    assert _rejection(result, 801)["reason"] == "reviewed_category_missing"
    assert _rejection(result, 802)["reason"] == "reviewed_category_invalid"
    assert _accepted_ids(result) == {803}
    assert result["accepted"]["hardness_class"].iloc[0] == "feather_leaf"

    assert H.REVIEWED_CATEGORIES == ("white_reflection", "paper_plastic", "feather_leaf",
                                     "metal_glint", "bright_signage", "other_reviewed")
    for category in H.REVIEWED_CATEGORIES:
        assert H.classify_negative_hardness(803, category) == category
    with pytest.raises(H.OutdoorHardNegBlocked):
        H.classify_negative_hardness(803, "")
    with pytest.raises(H.OutdoorHardNegBlocked):
        H.classify_negative_hardness(803, "shuttle")


# --- pool gate, manifest export and review aid ----------------------------------------------------

def test_pool_gate_shortfall_and_empty_label_export(tmp_path):
    """The pool gate is OUTDOOR_HARDNEG_INSUFFICIENT below 150 unique images, 3 categories or 3 clusters."""
    config = yaml.safe_load(NEW_CONFIG.read_text(encoding="utf-8"))
    frozen_pool = config["new_pools"]["outdoor_hard_negatives"]
    assert H.MIN_UNIQUE_IMAGES == frozen_pool["min_unique_images"] == 150
    assert H.MIN_REVIEWED_CATEGORIES == frozen_pool["min_categories"] == 3
    assert H.MIN_AUDITED_CLUSTERS == frozen_pool["min_clusters"] == 3
    assert frozen_pool["labels"] == "empty YOLO .txt (zero bytes), single class 0"

    categories = ["white_reflection", "paper_plastic", "feather_leaf"]

    def pool(count: int, category_count: int = 3, cluster_count: int = 3):
        return [_candidate(tmp_path, 5000 + index,
                           reviewed_category=categories[index % category_count],
                           source_cluster="cluster-%d" % (index % cluster_count))
                for index in range(count)]

    short = _run_audit(pool(149))
    assert short["status"] == "OUTDOOR_HARDNEG_INSUFFICIENT"
    assert short["counts"]["unique_accepted_images"] == 149
    assert short["shortfall"]["unique_images"] == 1
    assert short["thresholds"]["min_unique_images"] == 150

    full = _run_audit(pool(150))
    assert full["status"] == "PASS"
    assert full["gate"] == "PASS"
    assert full["counts"]["reviewed_categories"] == 3
    assert full["counts"]["audited_clusters"] == 3

    manifest = H.write_outdoor_hardneg_manifest(full, tmp_path / "out" / "outdoor_hardneg_manifest.csv")
    frame = pd.read_csv(manifest, keep_default_na=False, dtype=str)
    assert len(frame) == 150
    assert set(frame["role"]) == {C.ROLE_COCO_OUTDOOR_HARDNEG}
    assert set(frame["split"]) == {"train2017"}
    assert (frame["label_bytes"].astype(int) == 0).all()
    assert (frame["n_boxes"].astype(int) == 0).all()
    assert set(frame["is_negative"].str.lower()) == {"true"}
    assert set(frame["is_positive"].str.lower()) == {"false"}
    for label_path in frame["label_path"]:
        assert Path(label_path).is_file() and Path(label_path).stat().st_size == 0
    assert len(set(frame["sha256"])) == 150

    sheet = H.write_contact_sheet(full["accepted"], tmp_path / "sheets", columns=5, cell=32)
    assert sheet.is_file()
    with Image.open(sheet) as handle:
        assert handle.size == (5 * 32, 30 * 32)
    again = H.write_contact_sheet(full["accepted"], tmp_path / "sheets2", columns=5, cell=32)
    assert hashlib.sha256(sheet.read_bytes()).hexdigest() == hashlib.sha256(again.read_bytes()).hexdigest()

    two_categories = _run_audit(pool(150, category_count=2))
    assert two_categories["status"] == "OUTDOOR_HARDNEG_INSUFFICIENT"
    assert two_categories["shortfall"]["reviewed_categories"] == 1
    two_clusters = _run_audit(pool(150, cluster_count=2))
    assert two_clusters["status"] == "OUTDOOR_HARDNEG_INSUFFICIENT"
    assert two_clusters["shortfall"]["audited_clusters"] == 1

    leaky = _run_audit([_candidate(tmp_path, 5201)], clean_backgrounds=pd.DataFrame([
        {"coco_image_id": 5201, "sha256": ZERO_SHA, "near_duplicate_group": ""},
    ]))
    assert leaky["status"] == "OUTDOOR_HARDNEG_INSUFFICIENT" or leaky["status"] == "INVALID_OR_LEAK"
    with pytest.raises(H.OutdoorHardNegBlocked):
        H.write_outdoor_hardneg_manifest({"status": "INVALID_OR_LEAK", "accepted": full["accepted"]},
                                         tmp_path / "out" / "never.csv")
    assert not (tmp_path / "out" / "never.csv").exists()
