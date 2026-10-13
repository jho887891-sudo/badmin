"""Task 2b + Task 3 tests: source-role bookkeeping must make a leak or a double-used COCO image impossible to
miss, and the ETH foreground pool must be train-only, human-reviewed and mask-verified.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 2 Step 1/2, Task 3 Step 1/2).
Written before the SourceRecord/audit_source_roles and prepare_eth_real_foregrounds implementations exist,
so the first run is RED.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

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


def test_audit_accepts_a_string_typed_csv(tmp_path):
    """The CLI reads the worksheet with dtype=str; the audit must coerce the numeric columns itself.

    Without this, pandas raises "TypeError: unsupported operand type(s) for -: 'str' and 'str'" inside
    Series.quantile - which is exactly what the first real run against a 2,000-row worksheet did.
    """
    frame = pd.DataFrame([_candidate(1000 + index, clean_score=0.05 * (index + 1)) for index in range(4)])
    csv_path = tmp_path / "worksheet.csv"
    frame.to_csv(csv_path, index=False)
    as_strings = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    accepted = A.audit_clean_backgrounds(as_strings, pd.DataFrame())
    assert len(accepted) == 1, "a 4-image pool keeps its cleanest quartile even when read as text"
    assert accepted["clean_score"].dtype.kind == "f"
    assert accepted.attrs["clean_score_quartile_threshold"] == pytest.approx(0.0875)


def test_roles_must_be_from_the_frozen_enum():
    """An invented role string must be rejected: the plan fixes the role enum for the audit."""
    audit = C.audit_source_roles([_rec("x", "SOMETHING_ELSE")], [])
    assert audit["status"] == "INVALID_OR_LEAK"
    assert any(v["kind"] == "unknown_role" for v in audit["violations"])


# --- Task 3: real ETH foreground masks with mandatory review ---------------------------------------

FRAME_WIDTH = 200
FRAME_HEIGHT = 120
#: 30 x 60 px inside the 200 x 120 fixture frame: above the default 10/24 px floors.
BIG_BOX = (0.5, 0.5, 0.15, 0.5)
#: 5 x 10 px: below both floors, so it can never become a foreground.
TINY_BOX = (0.5, 0.5, 0.025, 0.0833)
FRAME_BBOX = (40, 30, 70, 90)


def _foregrounds():
    """Import the Task 3 module lazily: while it does not exist the Task 2 tests above must still run."""
    import prepare_eth_real_foregrounds as P
    return P


def _rgb_frame(width=FRAME_WIDTH, height=FRAME_HEIGHT) -> np.ndarray:
    """A structured RGB frame, never a flat fill, so no check can pass on a constant image."""
    rows = np.arange(height, dtype=np.uint16).reshape(height, 1)
    cols = np.arange(width, dtype=np.uint16).reshape(1, width)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (rows % 251).astype(np.uint8)
    frame[..., 1] = (cols % 241).astype(np.uint8)
    frame[..., 2] = ((rows + cols) % 233).astype(np.uint8)
    return frame


def _shuttle_alpha(width=FRAME_WIDTH, height=FRAME_HEIGHT, bbox=FRAME_BBOX, inset=3) -> np.ndarray:
    """An irregular shuttlecock footprint: a cone ring plus a feather spike inside the bbox, never a rectangle."""
    x1, y1, x2, y2 = bbox
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1 + inset, y1 + inset, x2 - inset - 1, y2 - inset - 1], fill=255)
    draw.ellipse([x1 + 2 * inset + 1, y1 + 2 * inset + 1, x2 - 2 * inset - 2, y2 - 2 * inset - 2], fill=0)
    spike = max(1, (x2 - x1) // 6)
    draw.polygon([(x1 + inset, y2 - inset - 1), (x2 - inset - 1, y2 - inset - 1),
                  (x2 - inset - 1 - spike, y2 - inset - 1 - spike)], fill=255)
    return np.array(mask, dtype=np.uint8)


def test_bbox_crop_not_mask():
    """A detection bbox is not a segmentation mask: filling the box is rejected, an irregular footprint is not."""
    P = _foregrounds()
    image = _rgb_frame()
    box_fill = np.zeros(image.shape[:2], dtype=np.uint8)
    box_fill[FRAME_BBOX[1]:FRAME_BBOX[3], FRAME_BBOX[0]:FRAME_BBOX[2]] = 255

    rejected = P.validate_foreground_mask(image, FRAME_BBOX, box_fill, reviewed=True, source_size=(30, 60))
    assert rejected["status"] == "REJECTED"
    assert "bbox_crop_not_mask" in rejected["reasons"]
    assert rejected["alpha_matches_bbox_rectangle"] is True

    accepted = P.validate_foreground_mask(image, FRAME_BBOX, _shuttle_alpha(), reviewed=True, source_size=(30, 60))
    assert accepted["status"] == "APPROVED"
    assert accepted["reasons"] == []
    assert accepted["bbox_fill_ratio"] < 0.98


def test_eval_image_never_selectable(tmp_path):
    """Only ETH train-only real positives are selectable; every evaluation image is refused with a stated
    reason, and a frozen evaluation frame cannot slip in through identical content either."""
    P = _foregrounds()
    rows = [
        _eth_row(tmp_path, "cab_1_00000", [BIG_BOX]),
        _eth_row(tmp_path, "cab_1_00001", [TINY_BOX]),
        _eth_row(tmp_path, "cab_1_00002", [BIG_BOX], sha256="sha-shared-with-eval"),
        _eth_row(tmp_path, "glc_2_00000", [BIG_BOX], split="val", set_dir="glc_2_easy", location="glc_2"),
        _eth_row(tmp_path, "eth_unseen_00000", [BIG_BOX], set_dir="eth_unseen", location="eth_unseen"),
        _eth_row(tmp_path, "challenge_test_00000", [BIG_BOX], set_dir="challenge_test"),
        _eth_row(tmp_path, "controlled_capability_00000", [BIG_BOX], set_dir="controlled_capability"),
        _eth_row(tmp_path, "coco_train_easy_00000", [BIG_BOX], set_dir="coco_train_easy", source="eth_negative",
                 role="official_negatives", is_negative=True, is_positive=False),
        _eth_row(tmp_path, "iphone_00000", [BIG_BOX], set_dir="iphone_set", source="iphone_capture"),
        _eth_row(tmp_path, "roboflow_00000", [BIG_BOX], set_dir="roboflow_shuttle", source="roboflow_export"),
        _eth_row(tmp_path, "isaac_00000", [BIG_BOX], set_dir="isaac_3d_render", source="3d_synthetic"),
    ]
    manifest = _write_manifest(tmp_path / "train_manifest.csv", rows)
    eval_manifest = _write_manifest(tmp_path / "frozen_eval_manifest.csv", [
        {"image": "/frozen/eval/some_eval_frame.jpg", "sha256": "sha-shared-with-eval", "split": "val"},
    ])

    audit = P.audit_eth_train_instances(manifest, frozen_eval_manifests=[eval_manifest])
    selected = P.select_eth_train_instances(manifest, frozen_eval_manifests=[eval_manifest])
    assert [i["instance_id"] for i in selected] == ["cab_1_00000#0"]
    assert audit["instances"] == selected
    assert all(i["split"] == "train" for i in selected)
    assert all(i["source"] == "eth_main" for i in selected)
    assert all(i["role"] == C.ROLE_ETH_TRAIN_FOREGROUND for i in selected)
    assert all(i["status"] == P.STATUS_PENDING_REVIEW and i["reviewed"] is False for i in selected)

    by_image = {r["image"]: r["reason"] for r in audit["rejections"]}
    assert by_image[str(tmp_path / "images" / "glc_2_00000.jpg")] == "eval_split"
    for name in ("eth_unseen_00000", "challenge_test_00000", "controlled_capability_00000"):
        assert by_image[str(tmp_path / "images" / (name + ".jpg"))] == "eval_set"
    assert by_image[str(tmp_path / "images" / "cab_1_00002.jpg")] == "eval_content"
    assert by_image[str(tmp_path / "images" / "coco_train_easy_00000.jpg")] == "not_positive"
    for name in ("iphone_00000", "roboflow_00000", "isaac_00000"):
        assert by_image[str(tmp_path / "images" / (name + ".jpg"))] == "non_eth_source"
    assert audit["rejections_by_reason"]["below_min_size"] == 1

    eval_stems = {"glc_2_00000", "eth_unseen_00000", "challenge_test_00000", "controlled_capability_00000"}
    assert not eval_stems & {Path(i["image"]).stem for i in selected}


def test_mask_dimensions_match_rgb():
    """A mask that belongs to a crop and not to the frame it is pasted into must be rejected."""
    P = _foregrounds()
    image = _rgb_frame()
    crop_mask = _shuttle_alpha(width=30, height=60, bbox=(0, 0, 30, 60), inset=2)
    result = P.validate_foreground_mask(image, FRAME_BBOX, crop_mask, reviewed=True, source_size=(30, 60))
    assert result["status"] == "REJECTED"
    assert "mask_dimensions_match_rgb" in result["reasons"]
    assert result["image_size"] == (FRAME_WIDTH, FRAME_HEIGHT)
    assert result["mask_size"] == (30, 60)


def test_opaque_rectangular_mask_rejected():
    """An opaque rectangle is a box someone drew, not a shuttlecock outline: it can never be approved."""
    P = _foregrounds()
    image = _rgb_frame()
    solid = np.zeros(image.shape[:2], dtype=np.uint8)
    solid[FRAME_BBOX[1] + 4:FRAME_BBOX[3] - 4, FRAME_BBOX[0] + 4:FRAME_BBOX[2] - 4] = 255
    result = P.validate_foreground_mask(image, FRAME_BBOX, solid, reviewed=True, source_size=(30, 60))
    assert result["status"] == "REJECTED"
    assert "opaque_rectangular_mask" in result["reasons"]
    assert result["bbox_fill_ratio"] > 0.98


def test_small_source_crop_not_upscaled_to_fake_detail():
    """A tiny source crop may not be blown up into invented feather detail: the upscale factor is checked."""
    P = _foregrounds()
    image = _rgb_frame()
    alpha = _shuttle_alpha()
    upscaled = P.validate_foreground_mask(image, FRAME_BBOX, alpha, reviewed=True, source_size=(3, 6))
    assert upscaled["status"] == "REJECTED"
    assert "upscaled_source_crop" in upscaled["reasons"]
    assert upscaled["upscale_factor"] == pytest.approx(10.0)

    unknown = P.validate_foreground_mask(image, FRAME_BBOX, alpha, reviewed=True)
    assert unknown["status"] == "REJECTED"
    assert "source_size_unknown" in unknown["reasons"]
    assert unknown["upscale_factor"] is None

    native = P.validate_foreground_mask(image, FRAME_BBOX, alpha, reviewed=True, source_size=(30, 60))
    assert native["status"] == "APPROVED"
    assert native["upscale_factor"] == pytest.approx(1.0)


def test_unreviewed_mask_not_accepted():
    """reviewed must be an explicit human True: a falsy, stringy or integer placeholder is never sign-off."""
    P = _foregrounds()
    image = _rgb_frame()
    alpha = _shuttle_alpha()
    for placeholder in (False, None, "yes", "true", 1, 0):
        result = P.validate_foreground_mask(image, FRAME_BBOX, alpha, reviewed=placeholder, source_size=(30, 60))
        assert result["status"] == "REJECTED", "reviewed=%r must not be accepted" % (placeholder,)
        assert "reviewed_required" in result["reasons"]
        assert result["reviewed"] is False

    approved = P.validate_foreground_mask(image, FRAME_BBOX, alpha, reviewed=True, source_size=(30, 60))
    assert approved["status"] == "APPROVED"
    assert approved["reviewed"] is True


def test_foreground_gate_below_100_is_insufficient(tmp_path):
    """Fewer than 100 reviewed instances (or fewer than 2 train locations) reports FOREGROUND_INSUFFICIENT."""
    P = _foregrounds()
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    def approved_row(index, location):
        mask_path = mask_dir / ("mask_%03d.png" % index)
        Image.fromarray(_shuttle_alpha()).save(mask_path)
        return {
            "instance_id": "instance_%03d#0" % index,
            "image": "/eth/%s/%03d.jpg" % (location, index),
            "label": "/eth/%s/labels/train/%03d.txt" % (location, index),
            "location": location,
            "set_dir": location + "_easy",
            "split": "train",
            "source": "eth_main",
            "sha256": "sha-%03d" % index,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "bbox_xyxy": list(FRAME_BBOX),
            "bbox_wh": [30, 60],
            "bbox_short": 30,
            "bbox_long": 60,
            "n_boxes": 1,
            "role": C.ROLE_ETH_TRAIN_FOREGROUND,
            "status": P.STATUS_APPROVED,
            "reviewed": True,
            "reviewer_id": "reviewer-1",
            "review_method": "manual_polygon",
            "mask_path": str(mask_path),
            "mask_sha256": C.sha256_file(mask_path),
        }

    short = [approved_row(i, "cab_1" if i % 2 == 0 else "cab_2") for i in range(99)]
    report = P.foreground_gate(short, mask_root=mask_dir)
    assert report["gate"] == "FOREGROUND_INSUFFICIENT"
    assert report["approved"] == 99
    assert report["approved_required"] == 100
    with pytest.raises(P.ForegroundInsufficient):
        P.assert_foreground_sufficient(short, mask_root=mask_dir)

    complete = short + [approved_row(99, "glc_1")]
    passed = P.foreground_gate(complete, mask_root=mask_dir)
    assert passed["gate"] == "PASS"
    assert passed["approved"] == 100
    assert sorted(passed["distinct_locations"]) == ["cab_1", "cab_2", "glc_1"]
    P.assert_foreground_sufficient(complete, mask_root=mask_dir)

    single_location = [approved_row(i, "cab_1") for i in range(100)]
    lopsided = P.foreground_gate(single_location, mask_root=mask_dir)
    assert lopsided["gate"] == "FOREGROUND_INSUFFICIENT"

    tampered = [dict(row) for row in complete]
    tampered[0]["mask_sha256"] = "0" * 64
    flagged = P.foreground_gate(tampered, mask_root=mask_dir)
    assert flagged["gate"] == "FOREGROUND_INSUFFICIENT"
    assert any(item["reason"] == "mask_sha_mismatch" for item in flagged["unverified"])

    dest = P.write_foreground_inventory(complete, tmp_path / "foreground_manifest.csv")
    assert dest == tmp_path / "foreground_manifest.csv"
    assert dest.is_file()
    assert P.foreground_gate(dest, mask_root=mask_dir)["gate"] == "PASS"


# --- Task 3 fixtures for the ETH manifest rows ------------------------------------------------------


def _eth_label(root: Path, name: str, boxes) -> Path:
    """A real YOLO label file (class cx cy w h, normalised) so the size filter reads real bytes."""
    label = root / "labels" / "train" / (name + ".txt")
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("".join("0 %.6f %.6f %.6f %.6f\n" % tuple(box) for box in boxes), encoding="utf-8")
    return label


def _eth_row(root: Path, name: str, boxes, **kw) -> dict:
    """One manifest row in the frozen V2 manifest schema, pointing at its real label file."""
    return {
        "image": str(root / "images" / (name + ".jpg")),
        "label": str(_eth_label(root, name, boxes)),
        "location": kw.get("location", "cab_1"),
        "difficulty": kw.get("difficulty", "easy"),
        "split": kw.get("split", "train"),
        "set_dir": kw.get("set_dir", "cab_1_easy"),
        "role": kw.get("role", "official_positives"),
        "source": kw.get("source", "eth_main"),
        "is_negative": kw.get("is_negative", False),
        "is_positive": kw.get("is_positive", True),
        "n_boxes": kw.get("n_boxes", len(boxes)),
        "width": kw.get("width", FRAME_WIDTH),
        "height": kw.get("height", FRAME_HEIGHT),
        "sha256": kw.get("sha256", "sha-" + name),
        "in_official_training": kw.get("in_official_training", True),
    }


def _write_manifest(path: Path, rows) -> Path:
    columns = ["image", "label", "location", "difficulty", "split", "set_dir", "role", "source", "is_negative",
               "is_positive", "n_boxes", "width", "height", "sha256", "in_official_training"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


# -----------------------------------------------------------------------------------------------------------
# Task 4: COCO train2017 CLEAN outdoor Copy-Paste backgrounds
# -----------------------------------------------------------------------------------------------------------

import audit_clean_outdoor_backgrounds as A  # noqa: E402


def _candidate(coco_id, clean_score=0.10, **overrides):
    row = {
        "coco_image_id": str(coco_id), "source": "coco_train2017", "permission_state": "declared",
        "outdoor_reviewed": "True", "no_shuttle_reviewed": "True", "roi_signoff": "True",
        "reviewer_id": "reviewer-1", "scene_type": "park", "source_cluster": "cluster-a",
        "clean_roi": "[0, 0, 512, 512]", "clean_score": clean_score, "near_duplicate_group": "",
        "image_sha256": "sha-%s" % coco_id, "location_id": "", "edge_density": "1.0", "entropy": "3.0",
        "luminance_mean": "120.0", "distinct_colors": "900",
    }
    row.update(overrides)
    return row


def test_coco_train2017_only():
    """Anything that is not COCO train2017 is refused outright, with a reason."""
    frame = pd.DataFrame([_candidate(1), _candidate(2, source="eth_main"), _candidate(3, source="isaac_pool")])
    accepted = A.audit_clean_backgrounds(frame, pd.DataFrame())
    assert [row["coco_image_id"] for row in accepted.to_dict("records")] == ["1"]
    assert set(accepted.attrs["rejections"]["reason"]) == {"source_not_coco_train2017"}


def test_coco_annotation_has_no_shuttle_is_not_proof_of_no_shuttle():
    """A missing COCO annotation never substitutes for the recorded human no-shuttle judgement."""
    frame = pd.DataFrame([_candidate(1, no_shuttle_reviewed="")])
    accepted = A.audit_clean_backgrounds(frame, pd.DataFrame())
    assert len(accepted) == 0
    assert accepted.attrs["rejections"].iloc[0]["reason"] == "no_shuttle_unreviewed"
    assert "not proof" in accepted.attrs["rejections"].iloc[0]["detail"]


def test_true_outdoor_photo_required():
    frame = pd.DataFrame([_candidate(1, outdoor_reviewed="False")])
    accepted = A.audit_clean_backgrounds(frame, pd.DataFrame())
    assert len(accepted) == 0
    assert accepted.attrs["rejections"].iloc[0]["reason"] == "not_outdoor"


def test_clean_score_bottom_quartile_with_fixed_metrics():
    """Only the cleanest quartile survives, and the ranking is the frozen candidate-only statistic."""
    metrics = [
        {"edge_density": 0.5, "entropy": 1.0, "luminance_distance": 0.9},
        {"edge_density": 9.0, "entropy": 6.0, "luminance_distance": 0.1},
        {"edge_density": 4.0, "entropy": 3.0, "luminance_distance": 0.5},
        {"edge_density": 1.0, "entropy": 2.0, "luminance_distance": 0.7},
    ]
    ranked = A.rank_clean_rois(pd.DataFrame(metrics))
    assert list(ranked["clean_score"]) == sorted(ranked["clean_score"])
    assert ranked["clean_score"].between(0.0, 1.0).all()
    assert ranked.iloc[0]["edge_density"] == 0.5, "lower edge density must rank cleaner"

    frame = pd.DataFrame([_candidate(1000 + index, clean_score=0.05 * (index + 1)) for index in range(8)])
    accepted = A.audit_clean_backgrounds(frame, pd.DataFrame())
    assert len(accepted) == 2, "an 8-image pool keeps the cleanest quartile"
    assert accepted.attrs["clean_score_quartile_threshold"] == pytest.approx(0.1375)
    rejected = set(accepted.attrs["rejections"]["reason"])
    assert rejected == {"clean_score_above_quartile"}


def test_roi_manual_signoff_required():
    frame = pd.DataFrame([_candidate(1, roi_signoff="False"), _candidate(2, reviewer_id="")])
    accepted = A.audit_clean_backgrounds(frame, pd.DataFrame())
    assert len(accepted) == 0
    assert set(accepted.attrs["rejections"]["reason"]) == {"roi_missing_signoff"}


def test_pure_color_and_artificial_bg_rejected():
    frame = pd.DataFrame([
        _candidate(1, distinct_colors="3", edge_density="0.1"),
        _candidate(2, distinct_colors="900", edge_density="6.0"),
    ])
    accepted = A.audit_clean_backgrounds(frame, pd.DataFrame())
    assert [row["coco_image_id"] for row in accepted.to_dict("records")] == ["2"]
    assert accepted.attrs["rejections"].iloc[0]["reason"] == "artificial_or_flat"


def test_old_eth_coco_image_excluded():
    """A picture already used by the frozen V2 work can never come back as a new background."""
    excluded = pd.DataFrame([{"coco_image_id": "1000", "sha256": "", "near_duplicate_group": ""},
                             {"coco_image_id": "", "sha256": "sha-1001", "near_duplicate_group": ""},
                             {"coco_image_id": "", "sha256": "", "near_duplicate_group": "nd-0007"}])
    frame = pd.DataFrame([_candidate(1000), _candidate(1001), _candidate(1002, near_duplicate_group="nd-0007"),
                          _candidate(1003)])
    accepted = A.audit_clean_backgrounds(frame, excluded)
    assert [row["coco_image_id"] for row in accepted.to_dict("records")] == ["1003"]
    assert set(accepted.attrs["rejections"]["reason"]) == {"excluded_by_history"}


def test_unknown_location_not_fabricated():
    """COCO cannot prove a location: blanks stay blank and are counted as unknown, never invented."""
    frame = pd.DataFrame([_candidate(1000), _candidate(1001, location_id="farm-road")])
    accepted = A.audit_clean_backgrounds(frame, pd.DataFrame())
    gate = A.evaluate_background_pool(accepted)
    assert gate["gate"] == "CLEAN_BG_INSUFFICIENT"
    assert gate["known_locations"] == ["farm-road"]
    assert gate["unknown_location_count"] == 1
    assert "" in list(accepted["location_id"])


def test_background_pool_gate_needs_images_types_and_clusters():
    """The Task 4 gate is a conjunction: 100 images, 3 scene types and 3 source clusters."""
    rows = []
    for index in range(100):
        rows.append(_candidate(2000 + index, scene_type="scene-%d" % (index % 3),
                               source_cluster="cluster-%d" % (index % 3), clean_score=0.01))
    accepted = A.audit_clean_backgrounds(pd.DataFrame(rows), pd.DataFrame())
    gate = A.evaluate_background_pool(accepted)
    assert gate["gate"] == "PASS"
    assert gate["images"] == 100 and len(gate["scene_types"]) == 3 and len(gate["source_clusters"]) == 3


