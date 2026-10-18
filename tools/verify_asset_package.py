# -*- coding: utf-8 -*-
"""Verify a portable simulation asset package against its manifest.

Two modes:
  --manifest-only   schema and internal consistency (no asset files needed; runs anywhere)
  --root DIR        full verification: every listed file exists with the recorded size and sha256

Exit code 0 only when every check passes, so it can gate a handover.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_STRUCTURE = {
    "revolute_joints": 6,     # reBot B601-RS: 6 axis arm
    "prismatic_joints": 2,    # two gripper fingers
    "fixed_joints": 1,        # j_gripper_end -> merged by --merge-joints, root_joint stays fixed
    "drive_stiffness_attrs": 6,   # angular drives only; the fingers use drive:linear:physics:*
}
ALLOWED_SUFFIXES = {".urdf", ".csv", ".stl", ".usd", ".usda", ".usdc", ".md", ".json",
                    ".txt", ".yaml"}
#: extension-less files that legitimately belong to the package
ALLOWED_NAMES = {"LICENSE_reBot-DevArm", ".asset_hash"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(problems: list, msg: str) -> None:
    problems.append(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=None, help="package directory to verify file-by-file")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    problems: list[str] = []
    checked = 0
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print("FAIL: manifest not found: " + str(manifest_path))
        return 2
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in ("package", "entry_usd", "entry_urdf", "robot_name", "source", "structure", "files"):
        if key not in doc:
            fail(problems, "manifest is missing the key " + key)
    if problems:
        for p in problems:
            print("FAIL: " + p)
        return 2

    files = doc["files"]
    if doc["entry_usd"] not in files:
        fail(problems, "entry_usd is not listed in files: " + doc["entry_usd"])
    if doc["entry_urdf"] not in files:
        fail(problems, "entry_urdf is not listed in files: " + doc["entry_urdf"])

    total = 0
    for rel, meta in files.items():
        if rel.startswith("/") or ":" in rel.split("/")[0] or "\\" in rel:
            fail(problems, "not a relative POSIX path: " + rel)
        if ".." in rel.split("/"):
            fail(problems, "path escapes the package: " + rel)
        if Path(rel).name in ALLOWED_NAMES:
            pass
        elif Path(rel).suffix.lower() not in ALLOWED_SUFFIXES:
            fail(problems, "unexpected file type in the package: " + rel)
        if len(meta.get("sha256", "")) != 64:
            fail(problems, "bad sha256 for " + rel)
        total += int(meta.get("bytes", -1))

    if doc.get("file_count") != len(files):
        fail(problems, "file_count " + str(doc.get("file_count")) + " != " + str(len(files)))
    if doc.get("total_bytes") != total:
        fail(problems, "total_bytes " + str(doc.get("total_bytes")) + " != " + str(total))

    for key, expected in EXPECTED_STRUCTURE.items():
        got = doc["structure"].get(key)
        if got != expected:
            fail(problems, "structure." + key + " is " + str(got) + ", expected " + str(expected))
    if doc["structure"].get("up_axis") != "Z" or doc["structure"].get("meters_per_unit") != 1:
        fail(problems, "expected Z-up and metersPerUnit 1")

    for key in ("repo", "commit", "license", "license_file"):
        if not doc["source"].get(key):
            fail(problems, "source." + key + " is empty")
    if doc["source"].get("license_file") not in files:
        fail(problems, "license file is not part of the package: " + str(doc["source"].get("license_file")))

    if args.root:
        root = Path(args.root)
        if not root.is_dir():
            fail(problems, "package root is not a directory: " + str(root))
        else:
            for rel, meta in files.items():
                p = root / rel
                checked += 1
                if not p.is_file():
                    fail(problems, "missing file: " + rel)
                    continue
                size = p.stat().st_size
                if size != meta["bytes"]:
                    fail(problems, "size mismatch for " + rel + ": " + str(size) + " != " + str(meta["bytes"]))
                if sha256_of(p) != meta["sha256"]:
                    fail(problems, "sha256 mismatch for " + rel)
            extra = []
            for p in root.rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(root)).replace(os.sep, "/")
                    if rel not in files and rel not in ("manifest.json", "MANIFEST.sha256"):
                        extra.append(rel)
            if extra:
                fail(problems, "unlisted files in the package: " + ", ".join(sorted(extra)[:5]))

    summary = {
        "package": doc["package"],
        "mode": "full" if args.root else "manifest-only",
        "listed_files": len(files),
        "checked_files": checked,
        "total_bytes": total,
        "problems": problems,
        "result": "PASS" if not problems else "FAIL",
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
