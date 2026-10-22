#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10-F2 runner: write the labelled verified-vs-proxy reachability comparison.

    python tools/r10f/run_r10f_backend_compare.py [--p6-root DIR] [--out PATH]

If the P6 verified FK checkout cannot be found the verified section is written as
UNAVAILABLE and NO proxy number is placed under it.  Exit code 0 means the report was
written; it does not mean both backends were available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from src.badminton_brain.reachability import backend_compare as compare  # noqa: E402
from src.badminton_brain.reachability import backends as backends_mod  # noqa: E402

DEFAULT_OUT = ROOT / "outputs" / "badminton_brain" / "r10f" / "backend_compare.json"

SCENARIOS = (
    compare.ComparisonScenario(name="proxy_convergent", intercept_point_m=(0.45, 0.0, 1.0),
                               approach_direction=(-1.0, 0.0, 0.0), time_to_target_s=0.30,
                               note="the point the P7-09 proxy tests converge to"),
    compare.ComparisonScenario(name="base_column_proxy", intercept_point_m=(0.20, 0.0, 1.0),
                               approach_direction=(-1.0, 0.0, 0.0), time_to_target_s=0.30,
                               note="inside the self-collision proxy cylinder"),
    compare.ComparisonScenario(name="outside_box", intercept_point_m=(2.40, 0.0, 1.0),
                               approach_direction=(-1.0, 0.0, 0.0), time_to_target_s=0.30,
                               note="outside the workspace box"),
    compare.ComparisonScenario(name="low_confidence", intercept_point_m=(0.45, 0.0, 1.0),
                               approach_direction=(-1.0, 0.0, 0.0), time_to_target_s=0.30,
                               position_sigma_m=0.20, note="sigma above the 0.05 m threshold"),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p6-root", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    verified_kwargs = {}
    root = args.p6_root or backends_mod.resolve_p6_root()
    if root is not None:
        verified_kwargs["root"] = str(root)
    report = compare.compare_backends(SCENARIOS, verified_kwargs=verified_kwargs)
    report["p6_root"] = str(root) if root else None
    report["status_by_backend"] = {name: section["status"]
                                   for name, section in report["backends"].items()}
    report["markdown_table"] = compare.format_table(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(compare.format_table(report))
    print("\nstatus:", report["status_by_backend"])
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
