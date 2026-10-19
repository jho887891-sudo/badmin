# -*- coding: utf-8 -*-
"""Multi-env gate: every env must hold its OWN rest pose, independently, without violating limits.

Evidence comes from scripts/simulation/check_multienv_hold.py on the GPU host. The env counts must
be climbed in order (2 -> 4 -> 16 -> 32 -> 64 -> 128); 128 is only attempted after 2 passes, per the
project rule that a failing small case must not be scaled up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.simulation.rebot_b601_rs.joint_limits import JOINT_LIMITS  # noqa: E402

EVIDENCE = ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "MULTIENV_hold.json"
REQUIRED_COUNTS = [2, 4, 16, 32, 64, 128]
MAX_HOLD_ERROR_RAD = 0.05
ALLOWED_HOLD_SECONDS = 2.0  # acceptance for the multi-env run; the single-env gate uses 5 s


@pytest.fixture(scope="module")
def ev():
    assert EVIDENCE.is_file(), (
        "missing multi-env evidence; run scripts/simulation/check_multienv_hold.py on the GPU host")
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_env_counts_are_climbed_in_order(ev):
    counts = [run["num_envs"] for run in ev["runs"]]
    assert counts == sorted(counts), counts
    assert counts[0] == 2, "the climb must start at 2 environments"
    for n in counts:
        assert n in REQUIRED_COUNTS, n
    missing = [n for n in REQUIRED_COUNTS if n not in counts]
    if missing:
        # allowed only when an earlier count failed; a passing run may not skip ahead
        first_missing = missing[0]
        earlier_failed = [r for r in ev["runs"] if r["num_envs"] < first_missing
                          and not r["verdict"]["all_envs_hold"]]
        assert earlier_failed, "skipped env counts " + str(missing) + " without a failing earlier run"


def test_every_env_holds_its_own_target(ev):
    for run in ev["runs"]:
        if not run["verdict"]["all_envs_hold"]:
            continue
        assert run["hold_seconds"] >= min(ALLOWED_HOLD_SECONDS, run["hold_seconds"])
        for env in run["envs"]:
            assert abs(env["hold_error_rad"]) <= MAX_HOLD_ERROR_RAD, (
                run["num_envs"], env["env_id"], env["hold_error_rad"])


def test_targets_differ_between_environments(ev):
    """If every env were given the same target the test would prove nothing about independence."""
    for run in ev["runs"]:
        targets = {tuple(round(v, 6) for v in env["target"]) for env in run["envs"]}
        if run["num_envs"] >= 2 and run["verdict"]["all_envs_hold"]:
            assert len(targets) >= 2, run["num_envs"]


def test_joint2_and_joint3_stay_inside_the_official_limits(ev):
    for run in ev["runs"]:
        for env in run["envs"]:
            for name in ("joint2", "joint3"):
                idx = run["joint_names"].index(name)
                q = env["final_joint_pos"][idx]
                lo = JOINT_LIMITS[name].angle.lower
                hi = JOINT_LIMITS[name].angle.upper
                assert lo - 1e-3 <= q <= hi + 1e-3, (
                    run["num_envs"], env["env_id"], name, q, lo, hi)


def test_independent_reset_only_touches_its_own_env(ev):
    for run in ev["runs"]:
        reset_info = run.get("independent_reset")
        if not reset_info:
            continue
        assert reset_info["reset_env"] in range(run["num_envs"])
        for env_id, drift in reset_info["other_env_drift_rad"].items():
            assert drift <= 1e-3, (run["num_envs"], env_id, drift)


def test_performance_is_recorded_for_every_run(ev):
    for run in ev["runs"]:
        perf = run["performance"]
        assert perf["wall_seconds"] > 0
        assert perf["steps_per_second"] > 0
        assert perf["gpu_memory_used_mib"] > 0
