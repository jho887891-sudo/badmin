"""Synthetic contract tests only; they do NOT count as PhysX evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "simulation"))
from p2_trajectory_gate import audit, main


def fixture():
    manifest = {
        "run_id": "synthetic_contract_only",
        "commit": "not-real-gpu-execution",
        "scene_variant": "TEST_SYNTHETIC",
        "physics_dt_s": 1 / 120,
        "settle_steps": 60,
        "joint_names": [],
        "authored_world": {
            "Chassis": {"position_m": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]},
            "Arm/base_link": {"position_m": [.12, -.05, .2], "quat_wxyz": [1, 0, 0, 0]},
        },
        "expected_mount": {"position_m": [.12, -.05, .2],
                           "quat_wxyz": [1, 0, 0, 0]},
    }
    frames = []
    for step in range(61):
        bodies = {}
        for name, pos in (("Chassis", [0, 0, 0]),
                          ("Arm/base_link", [.12, -.05, .2])):
            bodies[name] = {
                "position_m": pos, "raw_quaternion": [0, 0, 0, 1],
                "quat_layout": "xyzw",
                "source_api": "synthetic fixture matching observed xyzw layout",
                "linear_velocity_m_s": [0, 0, 0],
                "angular_velocity_rad_s": [0, 0, 0],
            }
        frames.append({
            "step": step, "t_s": step / 120,
            "event": "reset0" if step == 0 else "post_step",
            "bodies": bodies, "joints": {},
            "contact_status": "measured",
            "unauthorized_contacts": 0,
        })
    return manifest, frames


class TelemetryContractTest(unittest.TestCase):
    def test_synthetic_clean_evidence_passes_contract_only(self):
        m, f = fixture()
        result = audit(m, f)
        self.assertEqual(result["gate_result"], "PASS")
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["metrics"]["step_count"], 60)

    def test_c0_payload_name_in_manifest_is_accepted_without_alias(self):
        m, f = fixture()
        m["authored_world"]["Payload"] = m["authored_world"].pop("Arm/base_link")
        m["expected_mount"]["base_body_name"] = "Payload"
        for row in f:
            row["bodies"]["Payload"] = row["bodies"].pop("Arm/base_link")
            row["bodies"]["Payload"]["source_prim_path"] = "/World/C0/Payload"
        result = audit(m, f)
        self.assertEqual(result["gate_result"], "PASS")
        self.assertAlmostEqual(result["metrics"]["t0_mount_position_error_m"], 0)

    def test_c0_bad_payload_label_fails_not_silently_alias(self):
        m, f = fixture()
        m["expected_mount"]["base_body_name"] = "Payload"
        with self.assertRaises(ValueError):
            audit(m, f)

    def test_world_t0_shift_fails_while_local_mount_passes(self):
        m, f = fixture()
        for rec in f:
            for body in rec["bodies"].values():
                body["position_m"][2] += .0379
        result = audit(m, f)
        self.assertFalse(result["gates"]["t0_world_position_within_tol"])
        self.assertTrue(result["gates"]["t0_mount_position_within_tol"])
        self.assertTrue(result["gates"]["settle_drift_within_tol"])

    def test_upward_drift_fails(self):
        m, f = fixture()
        for rec in f:
            dz = rec["step"] / 60 * .369
            for body in rec["bodies"].values():
                body["position_m"][2] += dz
        result = audit(m, f)
        self.assertFalse(result["gates"]["settle_drift_within_tol"])
        self.assertAlmostEqual(result["metrics"]["max_chassis_settle_drift_m"], .369)

    def test_intermediate_drift_cannot_hide_at_final_frame(self):
        m, f = fixture()
        f[30]["bodies"]["Chassis"]["position_m"][2] = .1
        f[30]["bodies"]["Arm/base_link"]["position_m"][2] = .3
        result = audit(m, f)
        self.assertFalse(result["gates"]["settle_drift_within_tol"])
        self.assertEqual(result["metrics"]["last_chassis_settle_drift_m"], 0)

    def test_missing_contact_evidence_cannot_claim_full_pass(self):
        m, f = fixture()
        f[1]["contact_status"] = "unavailable"
        result = audit(m, f)
        self.assertFalse(result["gates"]["contact_evidence_complete"])

    def test_unauthorized_contact_fails(self):
        m, f = fixture()
        f[1]["unauthorized_contacts"] = 1
        result = audit(m, f)
        self.assertFalse(result["gates"]["no_unauthorized_contacts"])

    def test_missing_velocity_fails(self):
        m, f = fixture()
        del f[1]["bodies"]["Chassis"]["linear_velocity_m_s"]
        result = audit(m, f)
        self.assertFalse(result["gates"]["velocity_telemetry_complete"])

    def test_nan_coordinates_fail_closed(self):
        m, f = fixture()
        f[2]["bodies"]["Chassis"]["position_m"][2] = float("nan")
        with self.assertRaises(ValueError):
            audit(m, f)

    def test_missing_snapshot_fails_closed(self):
        m, f = fixture()
        del f[17]
        with self.assertRaises(ValueError):
            audit(m, f)

    def test_duplicate_primary_step_fails_closed(self):
        m, f = fixture()
        f.append(deepcopy(f[20]))
        with self.assertRaises(ValueError):
            audit(m, f)

    def test_cli_writes_fail_json_and_exit_nonzero_for_missing_step(self):
        m, f = fixture()
        del f[32]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            (root / "raw_poses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in f), encoding="utf-8")
            ret = main(["--manifest", str(root / "manifest.json"),
                        "--trajectory", str(root / "raw_poses.jsonl"),
                        "--result", str(root / "result.json")])
            report = json.loads((root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(ret, 1)
            self.assertEqual(report["gate_result"], "FAIL")
            self.assertFalse(report["gates"]["evidence_schema_valid"])

    def test_cli_rejects_existing_result(self):
        m, f = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "result.json").write_text("do not overwrite")
            with self.assertRaises(SystemExit):
                main(["--manifest", str(root / "manifest.json"),
                      "--trajectory", str(root / "raw_poses.jsonl"),
                      "--result", str(root / "result.json")])
            self.assertEqual((root / "result.json").read_text(), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
