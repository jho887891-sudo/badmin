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


class TestItRefusesToReportASelfDefeatingMeasurement:
    def test_it_does_not_zero_the_velocity_it_then_reads(self):
        """The broken first version set velocities to zero, stepped, and read them back: it measured 0.086
        rad/s against a 20.94 rad/s command and passed its own bound because the number was near zero."""
        text = SCRIPT.read_text(encoding="utf-8")
        assert "set_joint_velocities([0.0]" not in text, "zeroing the measured quantity is the bug"
        assert "set_joint_velocity_targets" in text, "the joint must be driven, not teleported"

    def test_it_refuses_a_near_zero_result(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "under 5 percent" in text or "0.05 *" in text
        assert "No record written" in text or "no record written" in text

    def test_the_applied_torque_is_recorded_with_the_result(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "applied_torque_nm" in text


class TestTheAppliedTorqueMatchesTheDeclaredConvention:
    """The first version applied a hardcoded 27 N m to any joint while the record claimed "rated".

    joint6 is a DM4310: rated 3 N m, peak 7. The drive therefore received 9.0x the rated figure and 3.9x the
    peak, and the provenance in the record was false. A plausible number with a false label is worse than an
    obviously broken one, because nobody re-checks it.
    """

    def test_no_standalone_torque_constant_is_assigned(self):
        """The name may appear in the comment that records why it was removed; it must not be assigned."""
        code = [
            line for line in SCRIPT.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("#")
        ]
        joined = "\n".join(code)
        assert "ABSOLUTE_TORQUE_LIMIT_NM" not in joined, "a shared constant is how the lie got in"
        assert "= 27.0" not in joined, "no hardcoded torque may reach a joint"

    def test_the_torque_comes_from_the_convention_resolver(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "resolve_torque_limit(" in text
        assert "convention=args.torque_convention" in text

    def test_the_applied_torque_is_recorded_not_just_the_convention(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "applied_torque_nm" in text

    def test_the_resolver_gives_the_figures_the_record_would_claim(self):
        from src.simulation.rebot_b601dm.torque_convention import resolve_torque_limit

        assert resolve_torque_limit("joint6", convention="rated") == pytest.approx(3.0)
        assert resolve_torque_limit("joint6", convention="peak") == pytest.approx(7.0)
        assert resolve_torque_limit("joint1", convention="rated") == pytest.approx(9.0)

    def test_the_old_constant_would_have_been_nine_times_the_rated_torque(self):
        """Kept as a regression figure: this is the size of the error that was applied."""
        from src.simulation.rebot_b601dm.torque_convention import resolve_torque_limit

        rated = resolve_torque_limit("joint6", convention="rated")
        assert 27.0 / rated == pytest.approx(9.0)
