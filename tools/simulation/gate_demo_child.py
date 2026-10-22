# -*- coding: utf-8 -*-
"""Minimal standalone demo experiment for the P0 gate runner (no Isaac Sim, no GPU).

Used by management/morph_one_rs/R10_A_RUNNER_FAULT_SEMANTICS.md to show a RUNNABLE positive
and negative example of the outer gate contract, including the R10-A fault semantics.

    PASS run            python tools/simulation/p2_gate_runner.py \
                            --script tools/simulation/gate_demo_child.py \
                            --run-id DEMO_ok --out-dir <fresh dir> -- --demo-gate PASS
    FAIL run            ... -- --demo-gate FAIL                      -> outer exit 1
    typed fault         ... -- --inject-fault joint_limit           -> EXPECTED_FAULT, exit 10
    undeclared fault    run the child directly / without the flag   -> UNEXPECTED_FAULT, exit 1
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--demo-gate", default="PASS", choices=("PASS", "FAIL"))
    parser.add_argument("--inject-fault", default="")
    parser.add_argument("--demo-traceback", action="store_true")
    args, _extra = parser.parse_known_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.demo_traceback:
        sys.stderr.write(
            "Traceback (most recent call last):\n"
            '  File "%s", line 51, in <module>\n'
            "    raise SystemExit(main())\n"
            "RuntimeError: INJECTED_PYTHON_EXCEPTION at step 60\n" % __file__)
    if args.inject_fault:
        # The P0-07 shape: a typed record, and NO result.json (the run did not finish).
        with open(os.path.join(out_dir, "fault_record.json"), "w", encoding="utf-8") as fh:
            json.dump({"run_id": args.run_id, "fault": args.inject_fault,
                       "code": "INJECTED_%s" % args.inject_fault.upper(),
                       "sim_steps_taken": 0,
                       "note": "demo typed fault: never a PASS"}, fh, indent=1)
        print("demo child: injected fault %s" % args.inject_fault)
        return 0
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump({"run_id": args.run_id, "gate_result": args.demo_gate,
                   "gates": {"demo_self_check": args.demo_gate == "PASS"},
                   "metrics": {"demo_metric": 1.0}}, fh, indent=1)
    print("demo child: gate=%s" % args.demo_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
