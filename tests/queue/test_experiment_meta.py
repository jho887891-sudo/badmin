"""A05: experiment metadata + manifest/applied validation, incl. the real gravity bug."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.queue import experiment_meta as a05


def write_experiment(base: Path, run_id="G5L_r01", gravity=(0.0, 0.0, -9.81),
                     command=None, lift=0.4, write_assets=True, extra_manifest=None):
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "stage": "P2",
        "run_id": run_id,
        "scene_variant": "REAL_CHASSIS_SIMPLIFIED",
        "physics_dt_s": 1.0 / 120.0,
        "gravity_m_s2": list(gravity),
        "lift_m": lift,
        "steps": 360,
        "command": command if command is not None
        else ("/home/dgut/runs/g/scripts/simulation/p2_wc_g6_collision_variants.py "
              "--run-id %s --out-dir /home/dgut/runs/g/%s --lift %s --read-contacts "
              "--gravity-z %s --steps 360" % (run_id, run_id, lift, gravity[2])),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (run_dir / "result.json").write_text(
        json.dumps({"run_id": run_id, "gate_result": "PASS", "gates": {"all_finite": True}}),
        encoding="utf-8")
    if write_assets:
        (run_dir / "raw_poses.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
        (run_dir / "authored_chassis.usda").write_text("#usda 1.0\n", encoding="utf-8")
    return run_dir


def codes(failures):
    return [failure["code"] for failure in failures]


# ------------------------------------------------------------------ the happy path
def test_matching_manifest_and_command_pass(tmp_path):
    run_dir = write_experiment(tmp_path)
    meta, failures = a05.validate_experiment(run_dir)
    assert failures == []
    assert meta["applied_parameters"]["gravity_z"] == -9.81
    assert meta["manifest_claimed"]["gravity_z_m_s2"] == -9.81
    assert meta["config_hash"].startswith("sha256:")
    assert {row["path"] for row in meta["assets"]} >= {
        "manifest.json", "result.json", "raw_poses.jsonl", "authored_chassis.usda"}


def test_assets_are_hashed_not_just_listed(tmp_path):
    run_dir = write_experiment(tmp_path)
    meta = a05.build_metadata(run_dir)
    by_path = {row["path"]: row for row in meta["assets"]}
    assert by_path["raw_poses.jsonl"]["sha256"] == a05.file_sha256(run_dir / "raw_poses.jsonl")
    assert by_path["raw_poses.jsonl"]["bytes"] == (run_dir / "raw_poses.jsonl").stat().st_size


# ------------------------------------------------------------------ gravity bug
def test_gravity_mismatch_fails_with_the_real_bug_wording(tmp_path):
    """The bug we actually hit: manifest [0,0,0] while the command used --gravity-z -9.81."""
    run_dir = write_experiment(tmp_path, gravity=(0.0, 0.0, 0.0),
                               command="/home/dgut/runs/g/scripts/simulation/"
                                       "p2_wc_g6_collision_variants.py --run-id G5L_r01 "
                                       "--out-dir /home/dgut/runs/g/G5L_r01 --lift 0.4 "
                                       "--read-contacts --gravity-z -9.81 --steps 360")
    _meta, failures = a05.validate_experiment(run_dir)
    assert "GRAVITY_MISMATCH" in codes(failures)
    gravity = [f for f in failures if f["code"] == "GRAVITY_MISMATCH"][0]
    assert gravity["manifest_value"] == 0.0
    assert gravity["applied_value"] == -9.81
    assert "ZERO gravity" in gravity["reason"]
    assert "not zero-gravity" in gravity["reason"].lower()


def test_gravity_mismatch_via_explicit_applied_flag(tmp_path):
    run_dir = write_experiment(tmp_path, gravity=(0.0, 0.0, 0.0))
    _meta, failures = a05.validate_experiment(run_dir, applied={"gravity_z": -9.81})
    assert "GRAVITY_MISMATCH" in codes(failures)


def test_gravity_agrees_when_applied_flag_matches_manifest(tmp_path):
    run_dir = write_experiment(tmp_path)
    _meta, failures = a05.validate_experiment(run_dir, applied={"gravity_z": -9.81})
    assert failures == []


def test_command_without_gravity_flag_is_unparsed(tmp_path):
    run_dir = write_experiment(tmp_path, command="/bin/run --run-id G5L_r01 "
                                                "--out-dir /runs/G5L_r01 --lift 0.4")
    _meta, failures = a05.validate_experiment(run_dir)
    assert "COMMAND_UNPARSED" in codes(failures)


def test_lift_mismatch_is_reported(tmp_path):
    run_dir = write_experiment(tmp_path, lift=0.4,
                               command="/bin/run --run-id G5L_r01 --out-dir /runs/G5L_r01 "
                                       "--gravity-z -9.81 --lift 0.30")
    _meta, failures = a05.validate_experiment(run_dir)
    assert "LIFT_M_MISMATCH" in codes(failures)


# ------------------------------------------------------------------ assets
def test_declared_asset_with_wrong_hash_fails(tmp_path):
    run_dir = write_experiment(tmp_path, run_id="G6A_r02",
                               extra_manifest={"asset_sha256": {"raw_poses.jsonl": "0" * 64}})
    _meta, failures = a05.validate_experiment(run_dir)
    assert "ASSET_SHA_MISMATCH" in codes(failures)


def test_declared_asset_that_is_missing_fails(tmp_path):
    run_dir = write_experiment(tmp_path, run_id="G6A_r03",
                               extra_manifest={"asset_sha256": {"raw_poses.jsonl": "0" * 64}})
    (run_dir / "raw_poses.jsonl").unlink()
    _meta, failures = a05.validate_experiment(run_dir)
    assert "ASSET_MISSING" in codes(failures)


def test_non_evidence_files_are_not_hashed(tmp_path):
    run_dir = write_experiment(tmp_path)
    (run_dir / "debug.bin").write_bytes(b"\x00\x01")
    meta = a05.build_metadata(run_dir)
    assert "debug.bin" not in {row["path"] for row in meta["assets"]}
    assert {"manifest.json", "result.json"} <= {row["path"] for row in meta["assets"]}


def test_missing_manifest_fails(tmp_path):
    run_dir = tmp_path / "G5L_r01"
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    (run_dir / "raw_poses.jsonl").write_text("{}\n", encoding="utf-8")
    _meta, failures = a05.validate_experiment(run_dir)
    assert "MISSING_MANIFEST" in codes(failures)


# ------------------------------------------------------------------ run / out dir
def test_run_id_must_match_directory(tmp_path):
    run_dir = write_experiment(tmp_path, run_id="G5_r01",
                               command="/bin/run --run-id G5_r01 --out-dir /runs/G5_r01 "
                                       "--gravity-z -9.81")
    inherited = tmp_path / "G5L_r01"
    inherited.mkdir()
    (inherited / "manifest.json").write_text(
        (run_dir / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    (inherited / "result.json").write_text(
        (run_dir / "result.json").read_text(encoding="utf-8"), encoding="utf-8")
    _meta, failures = a05.validate_experiment(inherited)
    assert "RUN_ID_MISMATCH" in codes(failures)
    assert "OUT_DIR_MISMATCH" in codes(failures)


# ------------------------------------------------------------------ hash contract
def test_hash_is_stable_across_key_order():
    left = {"gravity_z": -9.81, "lift_m": 0.4, "impl": "box_child_xform", "steps": 360}
    right = {"steps": 360, "impl": "box_child_xform", "lift_m": 0.4, "gravity_z": -9.81}
    assert a05.config_hash(left) == a05.config_hash(right)
    assert a05.config_hash({"nested": left}) == a05.config_hash({"nested": right})


def test_hash_is_stable_for_equivalent_float_spellings():
    assert a05.config_hash({"lift_m": 0.4}) == a05.config_hash({"lift_m": 0.40000000000000002})


def test_hash_changes_when_one_value_changes():
    base = {"gravity_z": -9.81, "lift_m": 0.4, "steps": 360}
    changed = dict(base, gravity_z=-9.8)
    assert a05.config_hash(base) != a05.config_hash(changed)
    assert a05.config_hash(base) != a05.config_hash(dict(base, steps=361))
    assert a05.config_hash(base) != a05.config_hash(dict(base, lift_m=0.4, extra=True))


def test_expected_config_hash_is_enforced(tmp_path):
    run_dir = write_experiment(tmp_path)
    meta = a05.build_metadata(run_dir)
    _meta, failures = a05.validate_experiment(
        run_dir, expect_config_hash=meta["config_hash"])
    assert failures == []
    _meta, failures = a05.validate_experiment(run_dir, expect_config_hash="sha256:" + "0" * 64)
    assert "CONFIG_HASH_MISMATCH" in codes(failures)


# ------------------------------------------------------------------ fingerprint
def test_fingerprint_is_python_and_platform_not_a_gpu_probe():
    fingerprint = a05.environment_fingerprint()
    assert fingerprint["python_version"].startswith("3.")
    assert fingerprint["platform_system"]
    assert fingerprint["gpu_probe"] is None
    assert "packages_importable" in fingerprint
    assert "pytest" in fingerprint["packages_importable"]
    assert set(fingerprint["packages_importable"]) == set(a05.FINGERPRINT_PACKAGES)


def test_fingerprint_marks_absent_packages_null():
    fingerprint = a05.environment_fingerprint(packages=("definitely_not_a_module_xyz",))
    assert fingerprint["packages_importable"] == {"definitely_not_a_module_xyz": None}


# ------------------------------------------------------------------ CLI
def test_cli_validate_exit_codes(tmp_path, capsys):
    good = write_experiment(tmp_path, run_id="G6A_r01")
    assert a05.main(["validate", "--experiment", str(good)]) == 0
    assert "PASS (A05_VALIDATION)" in capsys.readouterr().out

    bad = write_experiment(tmp_path, run_id="G5L_r01", gravity=(0.0, 0.0, 0.0),
                            command="/bin/run --run-id G5L_r01 --out-dir /runs/G5L_r01 "
                                    "--gravity-z -9.81 --lift 0.4")
    assert a05.main(["validate", "--experiment", str(bad)]) == 2
    captured = capsys.readouterr()
    assert "FAIL (A05_VALIDATION)" in captured.err
    assert "GRAVITY_MISMATCH" in captured.err


def test_cli_build_writes_metadata(tmp_path, capsys):
    run_dir = write_experiment(tmp_path)
    out = tmp_path / "meta.json"
    assert a05.main(["build", "--experiment", str(run_dir), "--out", str(out),
                     "--ref", "dev/p2-wheel-contact-g1g4", "--quiet"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["task"] == "A05"
    assert payload["source_ref"] == "dev/p2-wheel-contact-g1g4"
    assert payload["config_hash"].startswith("sha256:")


def test_cli_fingerprint_mode(capsys):
    assert a05.main(["fingerprint"]) == 0
    assert "python_version" in capsys.readouterr().out


def test_parse_command_understands_equals_form_and_repeats():
    parsed = a05.parse_command("/x/run --gravity-z=-9.81 --steps=360 --read-contacts --seed 7")
    assert parsed == {"gravity_z": -9.81, "steps": 360, "read_contacts": True, "seed": 7}
