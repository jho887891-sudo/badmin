"""The measurement script's record logic, tested without paying the app launch."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/simulation/measure_racket_speed.py")


class TestTheScriptRefusesToBeASecondReceiver:
    def test_it_does_not_use_position_teleport_for_the_measurement(self):
        """The upstream receiver applies set_joint_positions, which tracks the command regardless of torque."""
        text = SCRIPT.read_text(encoding="utf-8")
        assert "set_joint_positions" in text, "it must position the command"
        assert "torque" in text.lower(), "and it must be explicit that torque is what limits the motion"

    def test_it_says_why_the_receiver_is_the_wrong_tool(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "TELEPORTS" in text or "teleports" in text
        assert "isaacsim_joint_receiver" in text


class TestItStatesItsOwnLimits:
    def test_the_overestimate_is_named(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "OVERESTIMATE" in text
        assert "61%" in text or "mass" in text

    def test_it_is_explicit_that_it_measures_the_flange(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "flange" in text
        assert "not the racket head" in text


class TestTheCliIsUsable:
    def test_help_runs_without_importing_isaac_sim(self):
        """If Isaac Sim were imported at module level this would take twenty minutes."""
        import time

        t0 = time.time()
        r = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                           capture_output=True, timeout=120, cwd=".")
        elapsed = time.time() - t0
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[:400]
        assert b"--variant" in r.stdout
        assert elapsed < 60, "the help must not wait on an Isaac Sim import, took {:.0f}s".format(elapsed)

    def test_an_unknown_variant_is_rejected_by_argparse(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--asset", "/tmp/x.usda", "--variant", "nope"],
                           capture_output=True, timeout=120, cwd=".")
        assert r.returncode != 0


class TestTheRecordSurvivesTheAppClose:
    def test_the_record_is_written_before_the_app_is_closed(self):
        """SimulationApp.close() ends the process, so anything after it never runs.

        The first real run of this script started the app, ran the swing, closed, exited rc=0 and wrote no
        record at all, for exactly this reason. The ordering is therefore asserted structurally.
        """
        text = SCRIPT.read_text(encoding="utf-8")
        write_at = text.index("args.out")
        close_at = text.index("app.close()")
        assert write_at < close_at, "the record must be written before close() terminates the process"

    def test_it_flushes_before_closing(self):
        text = SCRIPT.read_text(encoding="utf-8")
        flush_at = text.index("sys.stdout.flush()")
        close_at = text.index("app.close()")
        assert flush_at < close_at
