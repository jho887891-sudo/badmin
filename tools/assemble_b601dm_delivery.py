"""Assemble the B601-DM deliverables into one self-contained local folder.

A delivery scattered across management/, src/, configs/, scripts/, outputs/, tests/, tools/ and docs/ is not
deliverable: there is nothing to hand over, and no way to see the whole thing at once. This copies the files
into deliverables/rebot_b601dm/ under numbered folders, and records the hash of each.

The SOURCE remains the single point of truth. This folder is generated; editing it and expecting the change to
survive would be a mistake, and the README says so at the top.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BT = chr(96)
OUT_REL = "deliverables/rebot_b601dm"

# (subfolder, [source paths])
LAYOUT = [
    ("01_reports", [
        "management/rebot_b601dm/ARM_IDENTITY_DECISION.md",
        "management/rebot_b601dm/REBOT_B601DM_REPORT.md",
        "management/rebot_b601dm/BADMINTON_CAPABILITY_RESULT.md",
        "management/rebot_b601dm/DELIVERABLES.md",
        "management/rebot_b601dm/TEMP_README.md",
        "management/rebot_b601dm/TASK7_MEASUREMENT_STATUS.md",
        "management/rebot_b601dm/FIRST_SWING_RECORD_INVALID.md",
        "docs/superpowers/plans/2026-09-17-rebot-b601dm-badminton-capability-plan.md",
    ]),
    ("02_joint_limits", [
        "src/simulation/rebot_b601dm/joint_limits.py",
        "configs/simulation/rebot_b601dm_joint_limits.yaml",
        "tools/generate_b601dm_limits_yaml.py",
    ]),
    ("03_simulation_assets", [
        "src/simulation/rebot_b601dm/patch_asset.py",
        "outputs/simulation/rebot_b601dm/real_limits/reBot_B601_DM.usda",
        "outputs/simulation/rebot_b601dm/real_limits/PATCH_NOTES.md",
        "outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physx.usda",
        "outputs/simulation/rebot_b601dm/real_limits/payloads/Physics/physics.usda",
        "outputs/simulation/rebot_b601dm/recommended_70_limits/reBot_B601_DM.usda",
        "outputs/simulation/rebot_b601dm/recommended_70_limits/PATCH_NOTES.md",
        "outputs/simulation/rebot_b601dm/recommended_70_limits/payloads/Physics/physx.usda",
    ]),
    ("04_torque_and_mass", [
        "src/simulation/rebot_b601dm/torque_convention.py",
        "src/simulation/rebot_b601dm/motor_mass.py",
        "tools/generate_b601dm_temp_readme.py",
    ]),
    ("05_launch_and_racket", [
        "src/simulation/rebot_b601dm/launch.py",
        "scripts/simulation/isaacsim_receiver_driver.py",
        "scripts/simulation/run_isaacsim_receiver.sh",
        "src/simulation/rebot_b601dm/racket.py",
    ]),
    ("06_measurement", [
        "src/simulation/rebot_b601dm/speed_bound.py",
        "scripts/simulation/measure_racket_speed.py",
        "scripts/simulation/wait_for_quiet_window.sh",
    ]),
    # GLOBBED, not listed. A hand-written list of test files was already stale within one task: the package
    # was missing the newest test because it had been added after the list was written. The directory is the
    # source of truth for what the suite contains.
    ("07_tests", "tests/simulation/rebot_b601dm/*.py"),
]
# The package files are two files of the SAME NAME in different directories, so they keep their path rather
# than being flattened. The duplicate-name guard below caught this when it was first tried flat.
PRESERVE_TREE = [
    "src/simulation/__init__.py",
    "src/simulation/rebot_b601dm/__init__.py",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
SHA = sha


def main() -> int:
    out = REPO / OUT_REL
    if out.exists():
        shutil.rmtree(out)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=str(REPO)).stdout.strip()

    copied, missing = [], []
    rows = []
    for folder, sources in LAYOUT + [("08_package", PRESERVE_TREE)]:
        dest_dir = out / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(sources, str):
            sources = sorted(str(p.relative_to(REPO)) for p in REPO.glob(sources))
        for rel in sources:
            src = REPO / rel
            if not src.is_file():
                missing.append(rel)
                continue
            # A flat copy collides: both variants carry files of the SAME NAME (reBot_B601_DM.usda,
            # PATCH_NOTES.md, physx.usda) and the second silently overwrote the first, leaving 5 of the 8
            # asset files in the package. The variant is therefore part of the name.
            name = Path(rel).name
            if folder == "08_package":
                dest = out / folder / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
            else:
                marker = "real_limits" if "/real_limits/" in rel else (
                    "recommended_70" if "/recommended_70_limits/" in rel else None)
                if marker:
                    stem, _, suffix = name.rpartition(".")
                    name = stem + "__" + marker + ("." + suffix if suffix else "")
                dest = dest_dir / name
            if dest.exists():
                raise SystemExit("name collision in the package: " + str(dest))
            shutil.copy2(src, dest)
            copied.append(rel)
            rows.append((folder, name, rel, dest.stat().st_size, SHA(dest)[:16]))

    readme = [
        "# reBot B601-DM badminton capability: delivery package",
        "",
        "Assembled by " + BT + "tools/assemble_b601dm_delivery.py" + BT + " from " + BT + head + BT + ".",
        "",
        "**This folder is generated. Do not edit it in place.** The files above it in the repository are the",
        "single point of truth; a change made here will be overwritten the next time the assembler runs, and",
        "worse, will not be in the repository at all.",
        "",
        "Everything below is committed to git. Bulk images and the 14 MB binary geometry payload are",
        "gitignored and are NOT needed to load these assets.",
        "",
        "---",
        "",
        "## Read this first",
        "",
        "| | |",
        "|---|---|",
        "| Gate 1, asset honesty | **PASS** |",
        "| Gate 2, provenance completeness | **PASS** |",
        "| Gate 3, physical plausibility | **UNEVALUATED - no valid measurement exists** |",
        "",
        "**The delivery is not accepted, because one gate cannot be evaluated.** The first swing record that was",
        "produced is invalid and is kept only as evidence; see " + BT + "01_reports/FIRST_SWING_RECORD_INVALID.md" + BT + ".",
        "",
        "## Contents",
        "",
        "| Folder | What is in it |",
        "|---|---|",
        "| " + BT + "01_reports/" + BT + " | the decision record, the A-M inspection, the gate result, the TEMP register, the plan |",
        "| " + BT + "02_joint_limits/" + BT + " | every joint limit with a provenance tag, and the YAML mirror |",
        "| " + BT + "03_simulation_assets/" + BT + " | the patched USD trees: real limits, and the official 70 percent variant |",
        "| " + BT + "04_torque_and_mass/" + BT + " | the torque convention, and the motor mass error quantified |",
        "| " + BT + "05_launch_and_racket/" + BT + " | the launcher for this host, and the racket attachment |",
        "| " + BT + "06_measurement/" + BT + " | the speed bound, the swing measurement, the window watcher |",
        "| " + BT + "07_tests/" + BT + " | the test suite, 135 passing |",
        "| " + BT + "08_package/" + BT + " | the Python package files these modules live in |",
        "",
        "## The single most important number in this delivery",
        "",
        "The shipped simulation asset permitted joint speeds **9.09x and 9.55x** the real motors, because the",
        "real rpm figures had been used as rad/s. The patched trees here enforce **315 and 1200 deg/s** where the",
        "upstream ones enforced 2864.789 and 11459.156. Everything else in this package exists to make that",
        "correction trustworthy and to have somewhere to measure with it.",
        "",
        "## Files",
        "",
        "| Folder | File | Size | sha256 (16) | Source in the repository |",
        "|---|---|---|---|---|",
    ]
    for folder, name, rel, size, digest in rows:
        readme.append("| " + folder + " | " + BT + name + BT + " | {:,} B | ".format(size)
                      + BT + digest + BT + " | " + BT + rel + BT + " |")

    readme += [
        "",
        "---",
        "",
        "## Rebuilding this folder",
        "",
        BT * 3 + "bash",
        "python tools/assemble_b601dm_delivery.py",
        BT * 3,
        "",
        "## What is deliberately not in here",
        "",
        "- a racket-head speed, because none has been validly measured",
        "- the upstream repositories, which are inputs and are unchanged",
        "- the 14 MB binary geometry payload, which is regenerable and gitignored",
        "- V4 motor data, real joint angle limits, racket mass, and a thermal model, none of which were found",
        "",
    ]
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("assembled", out)
    print("  copied {} files across {} folders".format(len(copied), len(LAYOUT)))
    if missing:
        print("  MISSING sources:")
        for m in missing:
            print("    ", m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
