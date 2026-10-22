#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-01: audit the COMMITTED dataset manifests (no downloads, no image reads).

Writes outputs/p7_audit/dataset_audit.json.  Missing files are reported as
DANGLING counts, never raised.

Exit codes:
  0  report written (findings, if any, are inside the JSON)
  1  a manifest is unreadable (missing header / missing file)
  3  --fail-on-unexplained and a session/clip crossing is not in KNOWN_SESSION_CROSSINGS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perception.shuttle_detection.committed_dataset_audit import (  # noqa: E402
    ManifestAuditError,
    audit_manifests,
    dumps,
)

DEFAULT_MANIFESTS = [
    ("outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", "v1_dataset"),
    ("outputs/shuttle_capability/train_data/manifest_train.csv", "synthetic_render"),
    ("outputs/shuttle_capability/train_data/manifest_val.csv", "synthetic_render"),
]
DEFAULT_OUT = "outputs/p7_audit/dataset_audit.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None,
                        metavar="PATH:KIND", help="repeatable; KIND is v1_dataset|synthetic_render")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--fail-on-unexplained", action="store_true")
    parser.add_argument("--no-sha256", action="store_true")
    args = parser.parse_args(argv)

    pairs = []
    for spec in (args.manifest or []):
        path, _, kind = spec.rpartition(":")
        pairs.append((path, kind))
    if not pairs:
        pairs = DEFAULT_MANIFESTS
    repo_root = Path(args.repo_root).resolve()
    try:
        report = audit_manifests(pairs, repo_root=repo_root,
                                 evidence_class="COMMITTED_MANIFESTS")
    except ManifestAuditError as exc:
        print("manifest error: " + str(exc), file=sys.stderr)
        return 1
    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(report), encoding="utf-8")

    print("scope:", report["scope"], "| evidence:", report["evidence_class"])
    for block in report["manifests"]:
        splits = block["splits"]
        sizes = block["size_distribution"]
        print("-", Path(block["manifest"]).name, "rows=%d splits=%s negatives=%d dangling=%d"
              % (block["rows"], splits, block["negatives_total"], block["dangling"]["rows"]))
        print("    size buckets:", {k: v for k, v in sizes.items() if v})
        print("    cross-split paths=%d basenames=%d clip-frame-index=%d"
              % (len(block["overlaps"]["same_path_cross_split"]),
                 len(block["overlaps"]["same_basename_cross_split"]),
                 len(block["overlaps"]["same_clip_frame_index_cross_split"])))
    leakage = report["leakage_summary"]
    print("session crossings:", leakage["session_crossings"])
    print("clip crossings:", leakage["clip_crossings"])
    print("unexplained:", leakage["unexplained_session_crossings"],
          leakage["unexplained_clip_crossings"])
    print("wrote", out)
    if args.fail_on_unexplained and (leakage["unexplained_session_crossings"]
                                    or leakage["unexplained_clip_crossings"]):
        print("UNEXPLAINED GROUP LEAKAGE", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
