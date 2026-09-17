"""Task 2: generate a B601-DM USD that enforces the REAL joint speeds.

The shipped asset permits 9.09x and 9.55x the real motor speeds. This test file is written before the
implementation, and its most important case is not "did the new value land" but "did anything ELSE move":
a patch that quietly alters geometry or inertias would invalidate every later measurement in a way no
velocity assertion would notice.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from src.simulation.rebot_b601dm.joint_limits import rad_s_from_rpm

pytestmark = pytest.mark.skipif(
    not Path("_scratch_rebot/reBot-Isaacsim/usd/reBot_B601_DM").is_dir(),
    reason="reBot-Isaacsim not present locally",
)

SOURCE = Path("_scratch_rebot/reBot-Isaacsim/usd/reBot_B601_DM")
PHYSX = "payloads/Physics/physx.usda"
PHYSICS = "payloads/Physics/physics.usda"


@pytest.fixture
def patched(tmp_path):
    from src.simulation.rebot_b601dm.patch_asset import patch_asset
    out = patch_asset(SOURCE, tmp_path / "real_limits", variant="real")
    return out


class TestTheConversionIsExact:
    def test_rpm_to_degrees_per_second_is_exact_for_both_motors(self):
        """52.5 rpm is exactly 315 deg/s and 200 rpm is exactly 1200 deg/s, because rpm*6 is deg/s."""
        assert rad_s_from_rpm(52.5) * 180.0 / math.pi == pytest.approx(315.0, rel=1e-12)
        assert rad_s_from_rpm(200.0) * 180.0 / math.pi == pytest.approx(1200.0, rel=1e-12)


class TestTheEnforcedValuesBecomeReal:
    def test_ninety_percent_variant_writes_the_real_no_load_speeds(self, patched):
        text = (patched / PHYSX).read_text(encoding="utf-8")
        assert "maxJointVelocity = 315.0" in text, "DM4340 52.5 rpm must become 315 deg/s"
        assert "maxJointVelocity = 1200.0" in text, "DM4310 200 rpm must become 1200 deg/s"
        assert "2864.789" not in text, "the 9.09x overshoot must be gone"
        assert "11459.156" not in text, "the 9.55x overshoot must be gone"

    def test_seventy_percent_variant_matches_the_official_recommendation(self, tmp_path):
        from src.simulation.rebot_b601dm.patch_asset import patch_asset
        out = patch_asset(SOURCE, tmp_path / "p70", variant="recommended_70")
        text = (out / PHYSX).read_text(encoding="utf-8")
        assert "maxJointVelocity = 220.5" in text
        assert "maxJointVelocity = 840.0" in text

    def test_an_unknown_variant_is_refused_rather_than_defaulted(self, tmp_path):
        from src.simulation.rebot_b601dm.patch_asset import patch_asset
        with pytest.raises(ValueError):
            patch_asset(SOURCE, tmp_path / "nope", variant="whatever")


class TestTheGripperIsLeftAlone:
    def test_gripper_velocity_keeps_its_shipped_value(self, patched):
        """No hardware source exists for the gripper speed, so patching it would be invention."""
        text = (patched / PHYSX).read_text(encoding="utf-8")
        assert "maxJointVelocity = 859.4367" in text


class TestNothingElseMoves:
    def test_the_only_changed_lines_are_velocity_lines(self, patched):
        import difflib
        for rel in (PHYSX, PHYSICS):
            before = (SOURCE / rel).read_text(encoding="utf-8").splitlines()
            after = (patched / rel).read_text(encoding="utf-8").splitlines()
            changed = [
                l for l in difflib.unified_diff(before, after, lineterm="", n=0)
                if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
            ]
            for line in changed:
                assert "velocity" in line.lower(), f"{rel}: unexpected change -> {line!r}"

    def test_geometry_and_inertia_files_are_copied_byte_for_byte(self, patched):
        import hashlib

        def sha(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()

        for rel in ("payloads/geometries.usd", "payloads/materials.usda", "payloads/base.usda",
                    "payloads/instances.usda", "payloads/robot.usda"):
            src, dst = SOURCE / rel, patched / rel
            if src.is_file():
                assert sha(src) == sha(dst), f"{rel} was modified"

    def test_the_source_tree_is_never_touched(self, tmp_path):
        import hashlib

        def digest():
            h = hashlib.sha256()
            for p in sorted(SOURCE.rglob("*")):
                if p.is_file():
                    h.update(p.name.encode())
                    h.update(p.read_bytes())
            return h.hexdigest()

        before = digest()
        from src.simulation.rebot_b601dm.patch_asset import patch_asset
        patch_asset(SOURCE, tmp_path / "again", variant="real")
        assert digest() == before, "the upstream asset tree must not be modified"


class TestTheResultIsLoadableInPrinciple:
    def test_the_entry_point_exists_and_still_references_its_payloads(self, patched):
        entry = patched / "reBot_B601_DM.usda"
        assert entry.is_file(), "the patched tree must keep the stage entry point"
        text = entry.read_text(encoding="utf-8")
        assert "payloads/" in text, "payload references must survive the copy"

    def test_the_patch_is_recorded_next_to_the_asset(self, patched):
        note = patched / "PATCH_NOTES.md"
        assert note.is_file(), "a patched asset without a record of what changed is not traceable"
        body = note.read_text(encoding="utf-8")
        assert "315.0" in body and "1200.0" in body
        assert "2864.789" in body, "the value that was replaced must be named"