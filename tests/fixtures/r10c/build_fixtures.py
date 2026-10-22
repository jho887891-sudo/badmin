#!/usr/bin/env python
"""Build the R10-C evidence-contract fixtures deterministically.

Run:  python tests/fixtures/r10c/build_fixtures.py
The generated trees under good/ and bad/ are committed, so the CLI exit codes
in management/morph_one_rs/R10_C_EVIDENCE_CONTRACTS.md can be reproduced
without running this script.  Every fixture is synthetic: no fixture is ever
presented as a real Isaac/Kit run.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.queue.evidence_contract import parameter_hash, sha256_file  # noqa: E402

ENV = {
    "python_version": "3.11.9",
    "platform": "Linux-5.15.0-x86_64",
    "sim_backend": "isaac_sim_4.5_kit",
    "isaac_sim_version": "4.5.0",
}
CONFIG = {
    "run_id": None,
    "physics_dt_s": 0.008333333333333333,
    "steps": 360,
    "gravity_m_s2": [0.0, 0.0, -9.81],
    "lift_m": 0.4,
    "impl": "box_child_xform",
}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def base_config(run_id: str) -> dict:
    config = dict(CONFIG)
    config["run_id"] = run_id
    return config


def finish_manifest(run_dir: Path, payload: dict, index_overrides=None) -> None:
    """Index every file except manifest.json, then write manifest.json."""
    index = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        index[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    if index_overrides is not None:
        index_overrides(index)
    payload["files"] = index
    write_json(run_dir / "manifest.json", payload)


NORMAL_RESULT = {
    "run_id": None,
    "gate_result": "PASS",
    "metrics": {
        "final_chassis_z_m": 0.333999693394,
        "tail_speed_max_m_s": 0.000513395,
        "steps": 360,
    },
    "gates": {"all_finite": True, "contact_detected": True, "settles": True},
}

POSES = (
    '{"step": 0, "t_s": 0.0, "pos_m": [0.0, 0.0, 0.4], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}\n'
    '{"step": 360, "t_s": 3.0, "pos_m": [0.0, 0.0, 0.334], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}\n'
)


def make_normal(run_dir: Path, run_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = base_config(run_id)
    result = json.loads(json.dumps(NORMAL_RESULT))
    result["run_id"] = run_id
    write_json(run_dir / "result.json", result)
    write_text(run_dir / "raw_poses.jsonl", POSES)
    # .txt, not .log: the repository .gitignore ignores *.log, and a fixture that
    # silently disappears from the commit would make the contract untestable.
    write_text(run_dir / "child_stdout.txt", "kit: scene loaded\nstep 360/360\n")
    write_text(run_dir / "child_stderr.txt", "warn: deprecated api (fixture)\n")
    write_json(run_dir / "gate_runner_summary.json", {
        "run_id": run_id,
        "command": "python scripts/simulation/fixture_gate.py --run-id %s" % run_id,
        "child_returncode": 0,
        "timed_out": False,
        "timeout_s": 1200.0,
        "json_gate": "PASS",
        "outer_exit": 0,
        "child_stdout_log": "child_stdout.txt",
        "child_stderr_log": "child_stderr.txt",
    })
    finish_manifest(run_dir, {
        "schema_version": "1.0",
        "kind": "NORMAL",
        "run_id": run_id,
        "stage": "P2-WC",
        "command": "python scripts/simulation/fixture_gate.py --run-id %s" % run_id,
        "config": config,
        "config_sha256": parameter_hash(config),
        "env_fingerprint": dict(ENV),
        "preregistration": "management/morph_one_rs/R10_C_EVIDENCE_CONTRACTS.md",
    })


def make_fault(run_dir: Path, run_id: str, *, logs: bool = True,
               fault_code: str = "KIT_PYTHON_ERROR") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config = base_config(run_id)
    write_json(run_dir / "fault_record.json", {
        "schema_version": "1.0",
        "run_id": run_id,
        "fault_code": fault_code,
        "fault_class": "UNEXPECTED",
        "outer_exit": 1,
        "child_returncode": 0,
        "message": "Python RuntimeError traceback while child_rc stayed 0",
        "observed_at": "2026-10-22T04:10:00Z",
    })
    if logs:
        write_text(run_dir / "child_stdout.txt", "kit: launching\n")
        write_text(run_dir / "child_stderr.txt",
                   "Traceback (most recent call last):\nRuntimeError: fixture typed fault\n")
    summary = {
        "run_id": run_id,
        "command": "python scripts/simulation/fixture_gate.py --inject fault",
        "child_returncode": 0,
        "timed_out": False,
        "json_gate": None,
        "outer_exit": 1,
        "child_stdout_log": "child_stdout.txt",
        "child_stderr_log": "child_stderr.txt",
    }
    write_json(run_dir / "gate_runner_summary.json", summary)
    finish_manifest(run_dir, {
        "schema_version": "1.0",
        "kind": "FAULT",
        "run_id": run_id,
        "stage": "P0-07",
        "command": summary["command"],
        "config": config,
        "config_sha256": parameter_hash(config),
        "env_fingerprint": dict(ENV),
    })


# ------------------------------------------------------------------- builders


def build_good(root: Path) -> None:
    make_normal(root / "good" / "N1_r01", "N1_r01")
    make_fault(root / "good" / "F1_r01", "F1_r01")


def build_bad(root: Path) -> None:
    bad = root / "bad"

    # B1: an ordinary run that lost result.json - must FAIL (not silently a fault)
    run = bad / "B1_r01_no_result"
    make_normal(run, run.name)
    (run / "result.json").unlink()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"].pop("result.json", None)
    write_json(run / "manifest.json", manifest)

    # B2: a declared fault run with no raw log - must FAIL
    make_fault(bad / "B2_r01_fault_no_log", "B2_r01_fault_no_log", logs=False)

    # B3: both kinds at once - must FAIL as a contradiction
    run = bad / "B3_r01_kind_conflict"
    make_normal(run, run.name)
    write_json(run / "fault_record.json", {
        "run_id": run.name, "fault_code": "CRASH", "fault_class": "UNEXPECTED",
        "outer_exit": 1, "child_returncode": 1,
    })
    finish_manifest(run, {"schema_version": "1.0", "kind": "NORMAL", "run_id": run.name,
                          "config": base_config(run.name),
                          "config_sha256": parameter_hash(base_config(run.name)),
                          "env_fingerprint": dict(ENV)})

    # B4: a fault record with a free-text code instead of the typed taxonomy
    run = bad / "B4_r01_untyped_fault"
    make_fault(run, run.name, fault_code="something_broke")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    finish_manifest(run, manifest)

    # B5: config edited after the hash was computed - must FAIL
    run = bad / "B5_r01_param_hash"
    make_normal(run, run.name)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["steps"] = 720
    write_json(run / "manifest.json", manifest)

    # B6: no environment fingerprint - must FAIL
    run = bad / "B6_r01_no_env"
    make_normal(run, run.name)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("env_fingerprint", None)
    write_json(run / "manifest.json", manifest)

    # B7: a file changed after it was indexed - must FAIL on SHA256
    run = bad / "B7_r01_sha_tamper"
    make_normal(run, run.name)
    write_text(run / "raw_poses.jsonl", POSES.replace("0.334", "0.999"))

    # B8: a stale artifact from an older run left in the directory - must FAIL
    run = bad / "B8_r01_stale"
    make_normal(run, run.name)
    write_json(run / "result_prev.json", {"run_id": run.name, "note": "stale leftover"})

    # B9: an artifact carrying another run's identity - must FAIL
    run = bad / "B9_r01_cross_run"
    make_normal(run, run.name)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["run_id"] = "N1_r01"
    write_json(run / "result.json", result)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    finish_manifest(run, manifest)

    # B10: the same RUN_ID claimed by two directories - must FAIL
    make_normal(root / "bad_dup" / "a" / "DUP_r01", "DUP_r01")
    make_normal(root / "bad_dup" / "b" / "DUP_r01", "DUP_r01")

    # B11: raw log path escapes the run directory (the real 3a366e7 shape)
    run = bad / "B11_r01_log_outside"
    make_normal(run, run.name)
    summary = json.loads((run / "gate_runner_summary.json").read_text(encoding="utf-8"))
    summary["child_stdout_log"] = "/home/dgut/runs/g/%s/gate_child.stdout.log" % run.name
    write_json(run / "gate_runner_summary.json", summary)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    finish_manifest(run, manifest)


def main() -> int:
    for name in ("good", "bad", "bad_dup"):
        target = HERE / name
        if target.exists():
            shutil.rmtree(target)
    build_good(HERE)
    build_bad(HERE)
    print("fixtures rebuilt under %s" % HERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
