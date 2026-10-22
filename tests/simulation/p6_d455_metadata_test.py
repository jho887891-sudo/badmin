# -*- coding: utf-8 -*-
"""P6-03: D455 metadata contract tests.

Every rejection path that would otherwise produce a plausible-looking number is
exercised here, plus the one PROVEN fact (the optical/mount permutation). The
fixtures are synthetic and say so; no D455 quantity is claimed anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.sensors import d455_metadata as md  # noqa: E402

FIX = ROOT / "tests/simulation/fixtures/p6_d455_replay"
P6_CAMERA_RECORD = (
    ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/camera_metadata.json"
)

FIELD_NAMES = [spec.name for spec in md.FIELD_SPECS]


def load(rel: str, **kwargs):
    return md.load_metadata(FIX / rel, **kwargs)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_record(tmp_path: Path, mutate, *, name: str = "record.json") -> Path:
    """Copy the good synthetic record, let mutate() edit it, write it back."""
    document = json.loads((FIX / "good/metadata.json").read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- defaults --
def test_no_field_has_a_default_that_looks_like_data():
    meta = md.D455Metadata()
    for name in FIELD_NAMES:
        assert getattr(meta, name) is md.UNKNOWN, name
    assert meta.status == "INCOMPLETE"
    assert meta.is_calibrated is False
    assert meta.known_fields() == []
    assert sorted(meta.unknown_fields()) == sorted(FIELD_NAMES)


def test_unknown_is_falsy_and_not_a_number():
    assert bool(md.UNKNOWN) is False
    assert not isinstance(md.UNKNOWN, (int, float))
    assert repr(md.UNKNOWN) == "UNKNOWN"
    assert md.UNKNOWN == md.UNKNOWN
    assert md.UNKNOWN != 0.0
    assert md.is_known(md.UNKNOWN) is False


def test_dataclass_has_no_numeric_field_defaults():
    import dataclasses

    for held in dataclasses.fields(md.D455Metadata):
        if held.name in ("units", "record_path", "record_sha256", "notes"):
            continue
        assert held.default is md.UNKNOWN, held.name
        assert held.default_factory is dataclasses.MISSING, held.name


# ------------------------------------------------------- verified loading --
def test_good_fixture_becomes_calibrated_and_keeps_provenance():
    meta = load("good/metadata.json")
    assert meta.status == "CALIBRATED_FULL"
    assert meta.is_calibrated is True
    assert meta.fx_px.as_float() == pytest.approx(4.0)  # synthetic toy pinhole
    record_sha = sha256(FIX / "good/metadata.json")
    assert meta.record_sha256 == record_sha
    for name in meta.known_fields():
        datum = meta.get(name)
        assert datum.source.sha256 == record_sha, name
        assert datum.source.locator.startswith("fields.") or name == "device_model"
        assert datum.source.file == "@self"
        assert datum.unit is not None or md.FIELD_BY_NAME[name].unit_key is None


def test_k_matrix_is_built_from_the_datum_values():
    meta = load("good/metadata.json")
    K = meta.intrinsics_matrix()
    assert K[0][0] == pytest.approx(4.0)
    assert K[1][1] == pytest.approx(4.0)
    assert K[0][2] == pytest.approx(3.5)
    assert K[1][2] == pytest.approx(2.5)
    assert K[2] == [0.0, 0.0, 1.0]


def test_external_witness_provenance_is_verified_against_real_bytes():
    meta = load("external_witness/metadata.json")
    datum = meta.fx_px
    witness = FIX / "witness_lab_note.json"
    assert datum.source.file == "../witness_lab_note.json"
    assert datum.source.sha256 == sha256(witness)
    assert meta.status == "CALIBRATED_FULL"


def test_bad_witness_sha256_is_rejected():
    with pytest.raises(md.ProvenanceViolationError) as exc:
        load("bad_sha/metadata.json")
    assert "hashes to" in str(exc.value)


def test_missing_witness_file_is_rejected(tmp_path):
    path = write_record(
        tmp_path,
        lambda d: d["fields"].__setitem__(
            "fx_px",
            {
                "value": 4.0,
                "unit": "px",
                "source": {
                    "file": "nope/absent.json",
                    "sha256": "1" * 64,
                    "locator": "fx_px",
                },
            },
        ),
    )
    with pytest.raises(md.ProvenanceViolationError) as exc:
        md.load_metadata(path)
    assert "does not exist" in str(exc.value)


def test_locator_that_does_not_resolve_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["fx_px"]["source"]["locator"] = "fields.fx_px.nope"

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ProvenanceViolationError) as exc:
        md.load_metadata(path)
    assert "does not resolve" in str(exc.value)


def test_self_declared_sha256_mismatch_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["fx_px"]["source"]["sha256"] = "2" * 64

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ProvenanceViolationError) as exc:
        md.load_metadata(path)
    assert "@self" in str(exc.value)


def test_a_value_without_a_source_is_rejected(tmp_path):
    def mutate(document):
        del document["fields"]["fx_px"]["source"]

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ProvenanceViolationError):
        md.load_metadata(path)


def test_a_field_marked_unknown_cannot_also_carry_a_value(tmp_path):
    def mutate(document):
        document["fields"]["fx_px"] = {
            "status": "UNKNOWN",
            "value": 4.0,
            "missing_datum": "none",
        }

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ProvenanceViolationError) as exc:
        md.load_metadata(path)
    assert "marked" in str(exc.value)


def test_schema_mismatch_is_rejected(tmp_path):
    path = write_record(tmp_path, lambda d: d.__setitem__("schema", "p6_d455/0"))
    with pytest.raises(md.D455MetadataError) as exc:
        md.load_metadata(path)
    assert "schema mismatch" in str(exc.value)


# ------------------------------------------------------------ unit guards --
def test_missing_units_block_is_rejected(tmp_path):
    path = write_record(tmp_path, lambda d: d.pop("units"))
    with pytest.raises(md.UnitViolationError) as exc:
        md.load_metadata(path)
    assert "units" in str(exc.value)


def test_millimetre_depth_scale_unit_is_rejected(tmp_path):
    def mutate(document):
        document["units"]["depth_scale"] = "mm_per_unit"

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.UnitViolationError) as exc:
        md.load_metadata(path)
    assert "mm_per_unit" in str(exc.value)


def test_millimetre_depth_scale_on_the_field_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["depth_scale_m_per_unit"]["unit"] = "mm_per_unit"

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.UnitViolationError) as exc:
        md.load_metadata(path)
    assert "m_per_unit" in str(exc.value)


def test_depth_scale_missing_its_unit_is_rejected(tmp_path):
    def mutate(document):
        del document["fields"]["depth_scale_m_per_unit"]["unit"]

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.UnitViolationError) as exc:
        md.load_metadata(path)
    assert "explicit 'unit'" in str(exc.value)


def test_millimetre_value_in_a_metres_field_is_rejected():
    """The classic 1000x error: 1.0 mm/unit written where 0.001 m/unit belongs."""
    with pytest.raises(md.ConsistencyViolationError) as exc:
        load("mm_value_in_metres_field/metadata.json")
    assert any("millimetres-per-unit" in p for p in exc.value.problems)


def test_degree_valued_unit_is_rejected(tmp_path):
    def mutate(document):
        document["units"]["angle"] = "deg"

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.UnitViolationError) as exc:
        md.load_metadata(path)
    assert "deg" in str(exc.value)


def test_degree_valued_fov_in_a_radian_field_is_rejected():
    with pytest.raises(md.ConsistencyViolationError) as exc:
        load("deg_fov/metadata.json")
    assert any("hfov_rad" in p and "degrees" in p for p in exc.value.problems)


def test_radian_valued_fov_is_accepted():
    meta = load("good/metadata.json")
    assert meta.hfov_rad.as_float() == pytest.approx(math.pi / 2)


# ------------------------------------------------------ consistency rules --
def test_principal_point_outside_the_image_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["cx_px"]["value"] = 99.0

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ConsistencyViolationError) as exc:
        md.load_metadata(path)
    assert any("outside the image" in p for p in exc.value.problems)


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("fx_px", 0.0, "must be > 0"),
        ("fy_px", -1.0, "must be > 0"),
        ("depth_scale_m_per_unit", 0.0, "must be > 0"),
        ("fps", 0.0, "must be > 0"),
        ("width_px", 0, "must be > 0"),
    ],
)
def test_nonpositive_values_are_rejected(tmp_path, field, value, fragment):
    def mutate(document):
        document["fields"][field]["value"] = value

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ConsistencyViolationError) as exc:
        md.load_metadata(path)
    assert any(fragment in p for p in exc.value.problems)


def test_negative_focal_length_caught_even_without_consistency_mode(tmp_path):
    def mutate(document):
        document["fields"]["fx_px"]["value"] = -4.0

    path = write_record(tmp_path, mutate)
    meta = md.load_metadata(path, require_consistent_values=False)
    assert meta.fx_px.value == -4.0
    assert md.validate_consistency(meta) != []


def test_distortion_model_with_wrong_coefficient_count_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["distortion_coefficients"]["value"] = [0.0] * 4

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ConsistencyViolationError) as exc:
        md.load_metadata(path)
    assert any("needs 5 coefficients" in p for p in exc.value.problems)


def test_unknown_distortion_model_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["distortion_model"]["value"] = "magic_lens"

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ConsistencyViolationError) as exc:
        md.load_metadata(path)
    assert any("magic_lens" in p for p in exc.value.problems)


def test_distortion_model_without_coefficients_is_rejected(tmp_path):
    def mutate(document):
        del document["fields"]["distortion_coefficients"]

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ConsistencyViolationError) as exc:
        md.load_metadata(path)
    assert any("coefficient vector is UNKNOWN" in p for p in exc.value.problems)


def test_bool_field_that_is_not_a_bool_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["rgb_depth_aligned"]["value"] = 1

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.D455MetadataError) as exc:
        md.load_metadata(path)
    assert "JSON bool" in str(exc.value)


def test_unknown_field_name_is_rejected(tmp_path):
    def mutate(document):
        document["fields"]["fx_mm"] = {"value": 1.0, "unit": "px", "source": {}}

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.D455MetadataError) as exc:
        md.load_metadata(path)
    assert "unknown field" in str(exc.value)


# ------------------------------------------------------- incomplete state --
def test_unknown_intrinsics_never_become_calibrated():
    meta = load("unknown_intrinsics/metadata.json")
    assert meta.status == "INCOMPLETE"
    assert meta.is_calibrated is False
    assert "fx_px" in meta.missing_required("intrinsics")
    assert "T_mount_camera" in meta.missing_required("full")
    with pytest.raises(md.MetadataIncompleteError) as exc:
        meta.require_calibrated_intrinsics()
    assert "fx_px" in exc.value.missing
    with pytest.raises(md.MetadataIncompleteError):
        meta.intrinsics_matrix()
    with pytest.raises(md.MetadataIncompleteError):
        meta.require_calibrated_full()


def test_unknown_reasons_are_preserved_for_the_reader():
    meta = load("unknown_intrinsics/metadata.json")
    reasons = meta.notes["unknown_reasons"]
    assert "fx_px" in reasons
    assert "intrinsics record" in reasons["fx_px"]
    assert meta.notes["state"] == "INCOMPLETE"


def test_require_complete_flag_refuses_an_incomplete_record():
    with pytest.raises(md.MetadataIncompleteError) as exc:
        load("unknown_intrinsics/metadata.json", require_complete=True)
    assert exc.value.level == "CALIBRATED_FULL"
    assert "T_mount_camera" in exc.value.missing


def test_intrinsics_only_calibration_is_distinguished_from_full(tmp_path):
    def mutate(document):
        del document["fields"]["T_mount_camera"]

    path = write_record(tmp_path, mutate)
    meta = md.load_metadata(path)
    assert meta.status == "CALIBRATED_INTRINSICS"
    assert meta.is_calibrated is True
    assert meta.missing_required("intrinsics") == []
    assert meta.missing_required("full") == ["T_mount_camera"]


# ------------------------------------------------------------ frame math --
def test_permutation_has_determinant_plus_one_in_both_directions():
    assert md.det3(md.OPTICAL_TO_MOUNT) == pytest.approx(1.0)
    assert md.det3(md.MOUNT_TO_OPTICAL) == pytest.approx(1.0)
    md.require_rotation(md.OPTICAL_TO_MOUNT, name="optical->mount")
    md.require_rotation(md.MOUNT_TO_OPTICAL, name="mount->optical")


def test_permutation_directions_are_exact_and_opposite():
    # proven statement: (X,Y,Z)_mount = (Z,-X,-Y)_optical
    assert md.optical_to_mount((1.0, 2.0, 3.0)) == pytest.approx((3.0, -1.0, -2.0))
    # its inverse: (X,Y,Z)_optical = (-Y,-Z,X)_mount
    assert md.mount_to_optical((1.0, 2.0, 3.0)) == pytest.approx((-2.0, -3.0, 1.0))
    # the two are not the same map: conflating them is the 90-degree bug
    assert md.optical_to_mount((1.0, 2.0, 3.0)) != md.mount_to_optical(
        (1.0, 2.0, 3.0)
    )


@pytest.mark.parametrize(
    "point",
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
     (0.31, -2.7, 5.5), (-1.0, -1.0, -1.0), (1e-9, 1e6, -3.2)],
)
def test_permutation_round_trip_is_identity(point):
    there = md.optical_to_mount(point)
    back = md.mount_to_optical(there)
    assert back == pytest.approx(point, abs=1e-12)


def test_committed_p6_camera_record_matches_this_module():
    report = md.check_optical_mount_contract(P6_CAMERA_RECORD)
    assert report["committed_matrix_equals_MOUNT_TO_OPTICAL"] is True
    assert report["det_optical_to_mount"] == pytest.approx(1.0)
    assert report["det_mount_to_optical"] == pytest.approx(1.0)
    assert report["status"] == "CONVENTION_NOT_CALIBRATION"
    assert report["record_sha256"] == sha256(P6_CAMERA_RECORD)


def test_committed_record_distinction_rule_matches_the_inverse_label():
    """The committed record states the optical->mount arrow, not the task text."""
    record = json.loads(P6_CAMERA_RECORD.read_text(encoding="utf-8"))
    rule = record["frames"]["distinction_rule"]
    assert "(X,Y,Z)_optical is (Z, -X, -Y) in the mount frame" in rule
    assert md.optical_to_mount((1.0, 2.0, 3.0)) == pytest.approx((3.0, -1.0, -2.0))


def test_left_handed_rotation_is_refused():
    mirrored = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    assert md.det3(mirrored) == pytest.approx(-1.0)
    with pytest.raises(md.FrameConventionError) as exc:
        md.require_rotation(mirrored, name="mirror")
    assert "det" in str(exc.value)


def test_non_orthonormal_matrix_is_refused():
    scaled = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    with pytest.raises(md.FrameConventionError) as exc:
        md.require_rotation(scaled, name="scaled")
    assert "orthonormal" in str(exc.value)


def test_mount_frame_unknown_refuses_the_se3_conversion():
    meta = load("unknown_intrinsics/metadata.json")
    with pytest.raises(md.MountFrameUnknownError) as exc:
        meta.optical_point_to_mount((0.0, 0.0, 1.0))
    assert "hand-eye" in str(exc.value)
    # the convention half still works and is explicitly not a placement
    assert meta.axis_convention_only_point_to_mount((1.0, 2.0, 3.0)) == pytest.approx(
        (3.0, -1.0, -2.0)
    )


def test_known_mount_transform_applies_rotation_then_translation(tmp_path):
    def mutate(document):
        document["fields"]["T_mount_camera"]["value"] = [
            [0.0, 0.0, 1.0, 0.10],
            [-1.0, 0.0, 0.0, 0.20],
            [0.0, -1.0, 0.0, 0.30],
            [0.0, 0.0, 0.0, 1.0],
        ]

    path = write_record(tmp_path, mutate)
    meta = md.load_metadata(path)
    assert meta.status == "CALIBRATED_FULL"
    p = meta.optical_point_to_mount((1.0, 2.0, 3.0))
    assert p == pytest.approx((3.0 + 0.10, -1.0 + 0.20, -2.0 + 0.30))


def test_mirrored_mount_rotation_is_rejected_by_consistency(tmp_path):
    def mutate(document):
        document["fields"]["T_mount_camera"]["value"] = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],  # flipped sign -> det = -1
            [0.0, 0.0, 0.0, 1.0],
        ]

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ConsistencyViolationError) as exc:
        md.load_metadata(path)
    assert any("det" in p for p in exc.value.problems)


def test_bottom_row_of_the_mount_transform_is_validated(tmp_path):
    def mutate(document):
        document["fields"]["T_mount_camera"]["value"] = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.0],
        ]

    path = write_record(tmp_path, mutate)
    with pytest.raises(md.ConsistencyViolationError) as exc:
        md.load_metadata(path)
    assert any("bottom row" in p for p in exc.value.problems)


# ------------------------------------------------------- optional consumers --
ROSBAG_TEXT = """path:        capture_0001
version:     5
storage:     sqlite3
duration:    2.033s
start:       Oct 20 2026 11:00:00.000
messages:    147
topic information: Topic: /camera/color/image_raw | Type: sensor_msgs/msg/Image | Count: 61 | Serialization Format: cdr
               Topic: /camera/depth/image_rect_raw | Type: sensor_msgs/msg/Image | Count: 61 | Serialization Format: cdr
"""


def test_rosbag_info_parsing_yields_counts_only():
    info = md.parse_rosbag_info(ROSBAG_TEXT, path="capture_0001.db3")
    assert info.duration_s == pytest.approx(2.033)
    assert info.message_count == 147
    assert info.topics["/camera/color/image_raw"]["messages"] == 61
    assert info.topics["/camera/depth/image_rect_raw"]["type"] == "sensor_msgs/msg/Image"
    # a bag info can NEVER carry intrinsics
    assert not hasattr(info, "fx_px")
    assert "fx" not in info.raw


def test_frame_index_probe_counts_files_and_reports_missing():
    good = md.probe_frame_index(FIX / "good/index.json")
    assert (good.rgb_count, good.depth_count) == (6, 6)
    assert good.is_usable is True
    assert good.declared_resolution == (8, 6)
    assert good.timestamps_s[:2] == pytest.approx([0.0, 1 / 30])

    broken = md.probe_frame_index(FIX / "missing_frames/index.json")
    assert broken.is_usable is False
    assert broken.depth_count == 5
    assert len(broken.problems) == 1


def test_frame_index_probe_refuses_an_empty_index(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"schema": md.SCHEMA_ID, "frames": []}), encoding="utf-8")
    with pytest.raises(md.D455MetadataError):
        md.probe_frame_index(path)


# --------------------------------------------------------------- yaml path --
def test_yaml_record_loads_through_the_same_contract():
    pytest.importorskip("yaml")
    meta = load("good/metadata.yaml")
    assert meta.status == "CALIBRATED_FULL"
    assert meta.fx_px.as_float() == pytest.approx(4.0)
    assert meta.record_sha256 == sha256(FIX / "good/metadata.yaml")


def test_yaml_without_pyyaml_fails_loudly(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(md.D455MetadataError) as exc:
        load("good/metadata.yaml")
    assert "PyYAML" in str(exc.value)
