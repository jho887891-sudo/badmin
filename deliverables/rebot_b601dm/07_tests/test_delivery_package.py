"""The local delivery package must be complete, collision-free, and regenerable."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path("deliverables/rebot_b601dm")

pytestmark = pytest.mark.skipif(not PKG.is_dir(), reason="package not assembled")


def test_it_has_a_readme_that_states_the_gate_status():
    text = (PKG / "README.md").read_text(encoding="utf-8")
    assert "UNEVALUATED" in text
    assert "not accepted" in text.lower()


def test_it_says_it_is_generated():
    text = (PKG / "README.md").read_text(encoding="utf-8")
    assert "generated" in text.lower()
    assert "Do not edit it in place" in text


def test_every_folder_is_present():
    names = {p.name for p in PKG.iterdir() if p.is_dir()}
    for expected in ("01_reports", "02_joint_limits", "03_simulation_assets",
                     "04_torque_and_mass", "05_launch_and_racket", "06_measurement",
                     "07_tests", "08_package"):
        assert expected in names, expected


def test_the_two_asset_variants_are_both_present_and_distinct():
    """A flat copy let the second variant overwrite the first, leaving 5 of 8 asset files.

    Both variants carry reBot_B601_DM.usda, PATCH_NOTES.md and physx.usda under the SAME NAME, so the variant
    has to be part of the filename.
    """
    assets = PKG / "03_simulation_assets"
    for name in ("reBot_B601_DM__real_limits.usda", "reBot_B601_DM__recommended_70.usda",
                 "physx__real_limits.usda", "physx__recommended_70.usda"):
        assert (assets / name).is_file(), name
    real = (assets / "physx__real_limits.usda").read_text(encoding="utf-8")
    reduced = (assets / "physx__recommended_70.usda").read_text(encoding="utf-8")
    assert "315.0" in real and "1200.0" in real
    assert "220.5" in reduced and "840.0" in reduced
    assert real != reduced, "the two variants must not be the same file twice"


def test_the_tests_suite_is_complete_in_the_package():
    tests = sorted(p.name for p in (PKG / "07_tests").glob("test_*.py"))
    source = sorted(p.name for p in Path("tests/simulation/rebot_b601dm").glob("test_*.py"))
    assert tests == source, "the packaged tests must be the full suite"


def test_no_file_is_empty():
    empties = [str(p) for p in PKG.rglob("*") if p.is_file() and p.stat().st_size == 0]
    assert not empties, empties


def test_the_package_can_be_rebuilt():
    r = subprocess.run([sys.executable, "tools/assemble_b601dm_delivery.py"],
                       capture_output=True, timeout=300, cwd=".")
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[:400]
    out = r.stdout.decode("utf-8", "replace")
    assert "copied" in out
    assert "MISSING" not in out, "a source went missing: " + out


def test_rebuilding_produces_the_same_file_count():
    before = sum(1 for p in PKG.rglob("*") if p.is_file())
    subprocess.run([sys.executable, "tools/assemble_b601dm_delivery.py"],
                   capture_output=True, timeout=300, cwd=".")
    after = sum(1 for p in PKG.rglob("*") if p.is_file())
    assert before == after
