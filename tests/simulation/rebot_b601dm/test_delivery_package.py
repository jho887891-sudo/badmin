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


def test_the_layout_mirrors_the_repository():
    """Numbered folders looked tidy and were useless: the tests import src.simulation.rebot_b601dm.*, so the
    paths have to be real for anything in the package to run."""
    for expected in ("src/simulation/rebot_b601dm", "tests/simulation/rebot_b601dm",
                     "configs/simulation", "scripts/simulation",
                     "outputs/simulation/rebot_b601dm", "management/rebot_b601dm"):
        assert (PKG / expected).is_dir(), expected


def test_the_package_runs_its_own_tests():
    """The proof that it is a package rather than a pile of files."""
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/simulation/rebot_b601dm", "-q"],
                       capture_output=True, timeout=600, cwd=str(PKG))
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    assert r.returncode == 0, out[-1200:]
    assert "failed" not in out.lower().split("passed")[-1], out[-400:]


def test_the_two_asset_variants_are_both_present_and_distinct():
    """A flat copy let the second variant overwrite the first, leaving 5 of 8 asset files.

    The paths are preserved now, so the variants live in their own trees - which is what makes them distinct.
    """
    root = PKG / "outputs/simulation/rebot_b601dm"
    for variant, expected in (("real_limits", ("315.0", "1200.0")),
                              ("recommended_70_limits", ("220.5", "840.0"))):
        physx = root / variant / "payloads/Physics/physx.usda"
        assert physx.is_file(), variant
        text = physx.read_text(encoding="utf-8")
        for token in expected:
            assert token in text, (variant, token)
    real = (root / "real_limits/payloads/Physics/physx.usda").read_text(encoding="utf-8")
    reduced = (root / "recommended_70_limits/payloads/Physics/physx.usda").read_text(encoding="utf-8")
    assert real != reduced, "the two variants must not be the same file twice"


def test_the_tests_suite_is_complete_modulo_the_documented_exclusions():
    """Every test of the shipped code is packaged, minus the two that test the repository layout.

    Those two assert things about the repository - that its index lists files which exist, that the package
    folder is present - and inside the package neither is true. Shipping them would make the packaged suite
    fail on arrival, which teaches a reader to ignore failures.
    """
    packaged = PKG / "tests/simulation/rebot_b601dm"
    tests = sorted(p.name for p in packaged.glob("test_*.py"))
    source = sorted(p.name for p in Path("tests/simulation/rebot_b601dm").glob("test_*.py"))
    excluded = {"test_deliverables_index.py", "test_delivery_package.py"}
    assert set(source) - set(tests) == excluded, "only the documented exclusions may be absent"
    assert len(tests) == len(source) - len(excluded)
    assert (packaged / "conftest.py").is_file(), "the packaged suite needs the conftest that sets sys.path"


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
