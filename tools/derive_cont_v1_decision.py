#!/usr/bin/env python3
"""Apply the pre-registered decision criteria of YOLO26S_V2_FULL_ETH_CONTINUATION_V1 (spec sections 19-22, 29).

The criteria and the V2 baseline live in configs/yolo26s_v2_full_eth_cont_v1.yaml, frozen before any metric was
observed. This module only evaluates them; it cannot invent a label and it emits exactly one primary decision.
"""
import argparse
import json
import sys
from pathlib import Path

CONTRACT = Path("configs/yolo26s_v2_full_eth_cont_v1.yaml")
OUT = Path("outputs/shuttle_capability/metrics/yolo26s_v2_full_eth_cont_v1_decision.json")
DELTA_KEYS = ("delta_recall_min", "delta_recall_max", "delta_recall_lt8_min", "delta_recall_lt8_max")


def _satisfies(rule: dict, d: dict) -> bool:
    checks = []
    for key, value in rule.items():
        if key not in DELTA_KEYS:
            raise ValueError("unknown decision rule key %r" % key)
        metric = "d_recall" if key.startswith("delta_recall_") and key.endswith(("min", "max")) and "lt8" not in key \
            else "d_recall_lt8"
        if key.endswith("_min"):
            checks.append(d[metric] >= float(value))
        else:
            checks.append(d[metric] <= float(value))
    return all(checks)


def classify(primary: list, d: dict) -> tuple:
    """First match in precedence order. The tiers nest (STRONG implies USEFUL), so overlaps are by design."""
    for entry in primary:
        if _satisfies_any(entry, d):
            return entry["label"], entry
    return None, None


def derive(measured: dict, contract: dict) -> dict:
    base = contract["baseline_v2"]
    crit = contract["decision_criteria"]
    ev = measured["legacy_eth_eval"]
    nt = measured["no_target"]
    d = {"d_recall": float(ev["recall"]) - float(base["recall"]),
         "d_map5095": float(ev["map5095"]) - float(base["map5095"]),
         "d_recall_lt8": float(ev["recall_lt8"]) - float(base["recall_lt8"]),
         "d_precision": float(ev["precision"]) - float(base["precision"]),
         "d_tp_lt8": int(ev["tp_lt8"]) - int(base["tp_lt8"]),
         "fp": int(nt["fp"]), "fp_images": int(nt["images"])}
    order = {lab: i for i, lab in enumerate(crit["precedence"])}
    primary = sorted(crit["primary"], key=lambda e: order.get(e["label"], len(order) + 1))
    label, entry = classify(primary, d)
    if label is None:
        raise ValueError("no primary criterion matched; the frozen criteria are not exhaustive")
    all_matching = [e["label"] for e in primary if _satisfies_any(e, d)]
    flags = crit["flags"]
    out_flags = []
    if d["fp"] <= int(flags["FP_STABLE"]["fp_max"]):
        out_flags.append("FP_STABLE")
    if d["fp"] >= int(flags["FP_REGRESSION"]["fp_min"]):
        out_flags.append("FP_REGRESSION")
    if d["d_recall"] > float(flags["RECALL_GAIN_WITH_FP_REGRESSION"]["delta_recall_min"]) \
            and d["fp"] >= int(flags["RECALL_GAIN_WITH_FP_REGRESSION"]["fp_min"]):
        out_flags.append("RECALL_GAIN_WITH_FP_REGRESSION")
    if d["d_map5095"] <= float(flags["MAP_REGRESSION"]["delta_map5095_max"]):
        out_flags.append("MAP_REGRESSION")
    return {"experiment": contract["experiment"], "primary_decision": label,
            "primary_decision_rule": entry,
            "flags": out_flags, "deltas": d, "baseline_v2": base, "measured": measured,
            "precedence": crit["precedence"], "exactly_one_primary_emitted": True,
            "all_criteria_that_hold": all_matching, "resolved_by": "precedence",
            "completion_answers": completion_answers(measured, d)}


def _satisfies_any(entry, d):
    if "all_of" in entry:
        return all(_satisfies(r, d) for r in entry["all_of"])
    return any(_satisfies(r, d) for r in entry["any_of"])


def completion_answers(measured: dict, d: dict) -> dict:
    """The ten questions the final report must answer (spec section 29)."""
    ev = measured["legacy_eth_eval"]
    sb = ev.get("source_buckets") or {}
    eth68 = (sb.get("eth_main") or {}).get("6-8") or {}
    return {"1_overall_recall_delta": d["d_recall"], "2_map5095_delta": d["d_map5095"],
            "3_recall_lt8_delta": d["d_recall_lt8"], "4_extra_tp_lt8": d["d_tp_lt8"],
            "5_real_6_8_recall": eth68.get("recall"), "5_real_6_8_recall_before": eth68.get("recall_before"),
            "6_sub_stride_evidence_improved": ev.get("sub_stride_evidence", "not measured"),
            "7_no_target_fp_regressed": d["fp"] >= 4, "7_no_target_fp": d["fp"],
            "8_precision_preserved": d["d_precision"] >= -0.02, "8_precision_delta": d["d_precision"],
            "10_proceed_to_p2": "NEXT_ARCHITECTURE_EXPERIMENT = P2_STRIDE_4"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", required=True, help="JSON written by the evaluation comparator")
    ap.add_argument("--contract", default=str(CONTRACT))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args([] if argv is None else argv)
    import yaml
    contract = yaml.safe_load(Path(a.contract).read_text(encoding="utf-8"))
    measured = json.loads(Path(a.measured).read_text(encoding="utf-8"))
    decision = derive(measured, contract)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print("PRIMARY DECISION: %s" % decision["primary_decision"])
    print("flags           : %s" % (decision["flags"] or "none"))
    print("deltas          : %s" % json.dumps(decision["deltas"]))
    print("WROTE %s (%d B)" % (out, out.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
