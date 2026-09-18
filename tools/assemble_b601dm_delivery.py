"""Assemble the B601-DM deliverables into one self-contained local folder.

WHY THE FOLDER MIRRORS THE REPOSITORY LAYOUT.

The first version scattered the files into numbered folders - 02_joint_limits/, 06_measurement/ and so on. It
looked tidy and it was useless: the packaged tests import src.simulation.rebot_b601dm.speed_bound, and in that
arrangement speed_bound.py sat in 06_measurement/ where no import could reach it. The package could not run a
single one of its own tests. That is the same failure this work keeps finding elsewhere: something that looks
like a deliverable while verifying nothing.

This version preserves the paths, so the README indexes the contents and pytest runs from inside the folder.

The SOURCE remains the single point of truth. This folder is generated.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BT = chr(96)
FENCE = BT * 3
OUT_REL = "deliverables/rebot_b601dm"

# Paths preserved as they are. The tests folder is GLOBBED: a hand-written list went stale within one task.
SOURCES = [
    "management/rebot_b601dm/ARM_IDENTITY_DECISION.md",
    "management/rebot_b601dm/REBOT_B601DM_REPORT.md",
    "management/rebot_b601dm/BADMINTON_CAPABILITY_RESULT.md",
    "management/rebot_b601dm/DELIVERABLES.md",
    "management/rebot_b601dm/TEMP_README.md",
    "management/rebot_b601dm/TASK7_MEASUREMENT_STATUS.md",
    "management/rebot_b601dm/FIRST_SWING_RECORD_INVALID.md",
    "docs/superpowers/plans/2026-09-17-rebot-b601dm-badminton-capability-plan.md",
    "src/simulation/__init__.py",
    "src/simulation/rebot_b601dm/__init__.py",
    "src/simulation/rebot_b601dm/joint_limits.py",
    "src/simulation/rebot_b601dm/patch_asset.py",
    "src/simulation/rebot_b601dm/torque_convention.py",
    "src/simulation/rebot_b601dm/motor_mass.py",
    "src/simulation/rebot_b601dm/launch.py",
    "src/simulation/rebot_b601dm/racket.py",
    "src/simulation/rebot_b601dm/speed_bound.py",
    "configs/simulation/rebot_b601dm_joint_limits.yaml",
    "scripts/simulation/isaacsim_receiver_driver.py",
    "scripts/simulation/run_isaacsim_receiver.sh",
    "scripts/simulation/measure_racket_speed.py",
    "scripts/simulation/wait_for_quiet_window.sh",
    "tools/generate_b601dm_limits_yaml.py",
    "tools/generate_b601dm_temp_readme.py",
    "outputs/simulation/rebot_b601dm/real_limits/reBot_B601_DM.usda",
    "outputs/simulation/rebot_b601dm/real_limits/PATCH_NOTES.md",
    "outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physx.usda",
    "outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physics.usda",
    "outputs/simulation/rebot_b601dm/recommended_70_limits/reBot_B601_DM.usda",
    "outputs/simulation/rebot_b601dm/recommended_70_limits/PATCH_NOTES.md",
    "outputs/simulation/rebot_b601dm/recommended_70_limits/payloads/Physics/physx.usda",
]
GLOBS = ["tests/simulation/rebot_b601dm/*.py"]

# NOT packaged: these two test the REPOSITORY's documentation layout rather than the shipped code. The index
# test asserts that DELIVERABLES.md lists files that exist, and inside the package that index points at
# repository paths which are not here; the package test asserts the package exists, which it does not, inside
# itself. Shipping them would produce a suite that fails on arrival and teaches a reader to ignore failures.
REPO_LEVEL_TESTS = (
    "tests/simulation/rebot_b601dm/test_deliverables_index.py",
    "tests/simulation/rebot_b601dm/test_delivery_package.py",
)

SECTIONS = [
    ("Reports and decision records", "management/rebot_b601dm/", "the decision record, the A-M inspection, the gate result, the TEMP register, and the two records of what went wrong"),
    ("The plan", "docs/superpowers/plans/", "the task plan this work follows"),
    ("The package", "src/simulation/rebot_b601dm/", "joint limits with provenance, the asset patcher, the torque convention, the mass audit, the launcher, the racket, the speed bound"),
    ("Configuration", "configs/simulation/", "the generated YAML mirror of the limit table"),
    ("Scripts", "scripts/simulation/", "the launcher, the receiver driver, the swing measurement, the window watcher"),
    ("Generators", "tools/", "the two scripts that keep the YAML and the TEMP register from drifting"),
    ("Patched assets", "outputs/simulation/rebot_b601dm/", "two USD variants: real limits, and the official 70 percent variant"),
    ("Tests", "tests/simulation/rebot_b601dm/", "the suite, runnable from this folder"),
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(REPO)).stdout.strip()


def main() -> int:
    out = REPO / OUT_REL
    if out.exists():
        shutil.rmtree(out)
    head = git("rev-parse", "HEAD")

    wanted = list(SOURCES)
    for pattern in GLOBS:
        # as_posix, not str: on Windows relative_to yields backslashes and the exclusion list below is written
        # with forward slashes, so the first version of this filter silently matched nothing and shipped both
        # files it was meant to leave out.
        wanted += sorted(
            rel for rel in (p.relative_to(REPO).as_posix() for p in REPO.glob(pattern))
            if rel not in REPO_LEVEL_TESTS
        )

    copied, missing, rows = [], [], []
    for rel in wanted:
        src = REPO / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(rel)
        rows.append((rel, dest.stat().st_size, sha(dest)[:16]))

    readme = [
        "# reBot B601-DM badminton capability: delivery package",
        "",
        "Assembled by " + BT + "tools/assemble_b601dm_delivery.py" + BT + " from commit " + BT + head + BT + ".",
        "",
        "**This folder is generated. Do not edit it in place.** The repository above it is the single point of",
        "truth; a change made here is lost the next time the assembler runs, and is not in the repository at all.",
        "",
        "**It mirrors the repository layout on purpose.** The tests import " + BT + "src.simulation.rebot_b601dm.*" + BT,
        "so the paths have to be real for anything here to run. An earlier version filed the files into numbered",
        "folders where no import could reach them and not one of its own tests would execute.",
        "",
        "---",
        "",
        "## Verify it before trusting it",
        "",
        FENCE + "bash",
        "cd deliverables/rebot_b601dm",
        "pytest tests/simulation/rebot_b601dm -q",
        FENCE,
        "",
        "That runs the packaged suite against the packaged modules, entirely inside this folder. If it does not",
        "pass, the package is not a package. Tests that skip do so because the upstream repositories are not",
        "in here, which is expected.",
        "",
        "**Two test files are deliberately not packaged**: " + BT + "test_deliverables_index.py" + BT + " and",
        BT + "test_delivery_package.py" + BT + ". They check the repository layout rather than the shipped code - the",
        "first asserts that the repository index lists files that exist, and inside this folder that index points",
        "at repository paths which are not here. Shipping them would make the suite fail on arrival and teach a",
        "reader to ignore failures. They run in the repository.",
        "",
        "---",
        "",
        "## Gate status",
        "",
        "| | |",
        "|---|---|",
        "| Gate 1, asset honesty | **PASS** |",
        "| Gate 2, provenance completeness | **PASS** |",
        "| Gate 3, physical plausibility | **UNEVALUATED - no valid measurement exists** |",
        "",
        "**The delivery is not accepted, because one gate cannot be evaluated.** The one swing record that was",
        "produced is invalid; see " + BT + "management/rebot_b601dm/FIRST_SWING_RECORD_INVALID.md" + BT + ".",
        "",
        "## The single most important number here",
        "",
        "The shipped simulation asset permitted joint speeds **9.09x and 9.55x** the real motors, because the real",
        "rpm figures had been used as rad/s. The patched trees enforce **315 and 1200 deg/s** where the upstream",
        "ones enforced 2864.789 and 11459.156. Everything else exists to make that correction trustworthy.",
        "",
        "## What is in here",
        "",
        "| Section | Folder | What it is |",
        "|---|---|---|",
    ]
    for title, folder, what in SECTIONS:
        n = sum(1 for rel, _, _ in rows if rel.startswith(folder))
        readme.append("| " + title + " | " + BT + folder + BT + " | " + what + " (" + str(n) + " files) |")

    readme += [
        "",
        "## Every file, with its hash",
        "",
        "| Path in this package | Size | sha256 (16) |",
        "|---|---|---|",
    ]
    for rel, size, digest in rows:
        readme.append("| " + BT + rel + BT + " | {:,} B | ".format(size) + BT + digest + BT + " |")

    readme += [
        "",
        "---",
        "",
        "## What is deliberately not here",
        "",
        "| Missing | Why |",
        "|---|---|",
        "| a valid racket-head speed | none has been measured; the first record measured a joint it had itself told to stop |",
        "| the upstream repositories | they are inputs, unchanged, at 1958511342dab181b94a2b7c068c77e54eb85d3a and f01a1dc189ecb74b5435a336a68b287c146c1b60 |",
        "| the binary geometry payload | 14 MB, regenerable, gitignored; not needed to load these assets |",
        "| V4 motor data | the official table documents -2EC variants while the BOM specifies V4 |",
        "| real joint angle limits | no hardware document in either repository states a range |",
        "| racket mass and inertia | the asset is a mesh with no inertial data |",
        "| a thermal model | the published testing gives a duty-cycle constraint, not resistance or capacity |",
        "",
        "## Rebuilding",
        "",
        FENCE + "bash",
        "python tools/assemble_b601dm_delivery.py",
        FENCE,
        "",
    ]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("assembled", out)
    print("  copied {} files, paths preserved".format(len(copied)))
    if missing:
        print("  MISSING sources:")
        for m in missing:
            print("    ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
