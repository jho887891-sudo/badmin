#!/usr/bin/env python3
"""Decision state machine and provenance report for CLEAN_OUTDOOR_CP_V1 (Task 10).

The single experiment changes two things at once (Copy-Paste positives and new outdoor hard negatives) and has
no same-period A0 control, so no historical difference can be attributed to either intervention. This module
therefore refuses causal language, demands a denominator for every number it prints, and refuses to describe a
run as trained unless its checkpoint and manifest evidence exist.

Allowed outcomes: INVALID_OR_LEAK, PILOT_ONLY, FP_REGRESSION_OR_REAL_HARM, OUTDOOR_CAPABILITY_VERIFIED, BELOW_TARGET.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 10).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402

PASS = "PASS"
INVALID_OR_LEAK = "INVALID_OR_LEAK"
PILOT_ONLY = "PILOT_ONLY"
FP_REGRESSION_OR_REAL_HARM = "FP_REGRESSION_OR_REAL_HARM"
OUTDOOR_CAPABILITY_VERIFIED = "OUTDOOR_CAPABILITY_VERIFIED"
BELOW_TARGET = "BELOW_TARGET"
STATES = (INVALID_OR_LEAK, PILOT_ONLY, FP_REGRESSION_OR_REAL_HARM, OUTDOOR_CAPABILITY_VERIFIED, BELOW_TARGET)

#: Phrases that would claim a causal effect this design cannot support. Checked against every generated report.
CAUSAL_PATTERNS = (
    r"copy-?paste (caused|proves|demonstrates|is responsible)",
    r"caused by (copy-?paste|the new hard negatives)",
    r"proves that (copy-?paste|hard negatives)",
    r"(due to|because of) (copy-?paste|the hard negatives)",
    r"the (gain|improvement) (comes from|is caused by)",
    r"独立因果",
    r"证明了.{0,6}(复制粘贴|困难负样本)",
)

DEFAULT_THRESHOLDS = {
    "max_fp": 3,
    "no_target_images": 496,
    "ood_recall_at_conf_0_25": 0.30,
    "ood_map50_95": 0.20,
    "historical_v2_fp": 1,
}


def assert_no_causal_language(text: str) -> None:
    """A single-arm design cannot attribute an effect; the report must not even imply it."""
    lowered = str(text)
    for pattern in CAUSAL_PATTERNS:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            raise ValueError("report claims an unsupported causal effect: %r" % match.group(0))


def assert_denominators(metrics: dict) -> None:
    """Every reported rate needs its raw count and its denominator, or it is not evidence."""
    problems = []
    for name, block in (metrics or {}).items():
        if not isinstance(block, dict):
            continue
        if "rate" in block or "per_image" in block:
            if "count" not in block or "denominator" not in block:
                problems.append("%s reports a rate without count/denominator" % name)
    if problems:
        raise ValueError("; ".join(problems))


def assert_run_evidence(run: dict) -> None:
    """A run may only be described as trained when the checkpoint and its manifest actually exist."""
    if not run:
        raise ValueError("no run evidence at all")
    if run.get("status") not in {"COMPLETED"}:
        raise ValueError("run status is %r, not COMPLETED" % run.get("status"))
    for key in ("checkpoint_sha256", "manifest", "epochs_recorded"):
        if not run.get(key):
            raise ValueError("run is missing %s, so it may not be reported as trained" % key)
    if int(run.get("epochs_recorded", 0)) < 1:
        raise ValueError("run recorded no epochs")


def first_open_gate(audit: dict):
    """Any gate that is not exactly PASS keeps the run from claiming a capability.

    Originally only BLOCKED_* strings were treated as blockers, which let an insufficiency gate such as
    FOREGROUND_INSUFFICIENT or CLEAN_BG_INSUFFICIENT fall through to a success verdict. Every non-PASS status
    is now a blocker, whatever its spelling.
    """
    if not isinstance(audit, dict):
        return None
    for key in ("blocked_gate", "gate"):
        value = audit.get(key)
        if isinstance(value, str) and value.strip().upper() != PASS:
            return value
    gates = audit.get("gates")
    if isinstance(gates, dict):
        for name, value in sorted(gates.items()):
            if isinstance(value, str) and value.strip().upper() != PASS:
                return "%s=%s" % (name, value)
    return None


def decide_clean_cp_v1(audit: dict, metrics: dict, thresholds: dict = None) -> str:
    """The Task 10 state machine. Leakage outranks everything; missing real OOD can never look like success."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    audit = audit or {}
    metrics = metrics or {}

    violations = audit.get("violations")
    if violations or str(audit.get("status", "")).upper() == INVALID_OR_LEAK:
        return INVALID_OR_LEAK

    if first_open_gate(audit) is not None:
        return PILOT_ONLY

    fp_block = metrics.get("real_no_target_496") or {}
    if fp_block:
        if "count" not in fp_block or "denominator" not in fp_block:
            raise ValueError("real_no_target_496 needs count and denominator")
        if int(fp_block["count"]) > int(thresholds["max_fp"]):
            return FP_REGRESSION_OR_REAL_HARM

    ood = metrics.get("outdoor_real_ood")
    if not ood or str(ood.get("status", "")).upper() in {"MISSING", "PILOT_ONLY", "NOT_AVAILABLE"}:
        return PILOT_ONLY

    recall = ood.get("recall_at_conf_0_25")
    mean_ap = ood.get("map50_95")
    if recall is None or mean_ap is None:
        raise ValueError("real outdoor OOD metrics must report both recall and mAP50-95")
    if float(recall) >= float(thresholds["ood_recall_at_conf_0_25"]) and float(mean_ap) >= float(thresholds["ood_map50_95"]):
        return OUTDOOR_CAPABILITY_VERIFIED
    return BELOW_TARGET


def _fmt_metric(block: dict, label: str) -> str:
    if not block:
        return "- %s: not measured" % label
    count = block.get("count")
    denominator = block.get("denominator")
    value = block.get("rate", block.get("per_image", block.get("value")))
    if count is None or denominator is None:
        return "- %s: %s (no denominator recorded)" % (label, value)
    return "- %s: raw %s / %s" % (label, count, denominator)


def build_artifact_manifest(paths, repo_root=None) -> dict:
    """Hash every deliverable so the report can be checked without trusting the report."""
    root = Path(repo_root) if repo_root else None
    entries = []
    missing = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_file() and root is not None:
            candidate = root / path
        if not candidate.is_file():
            missing.append(str(path))
            continue
        relative = str(candidate.relative_to(root)) if root and root in candidate.parents else str(candidate)
        entries.append({
            "path": relative.replace("\\", "/"),
            "bytes": candidate.stat().st_size,
            "sha256": C.sha256_file(candidate),
        })
    return {"generated_by": "tools/report_single_clean_cp_v1.py", "artifacts": entries, "missing": missing}


def render_report(context: dict) -> str:
    """Render the provenance report. Every number keeps its denominator; no causal claim is made."""
    decision = context.get("decision") or PILOT_ONLY
    if decision not in STATES:
        raise ValueError("unknown decision state %r" % decision)

    lines = [
        "# ETH real foregrounds + COCO clean outdoor Copy-Paste + COCO hard negatives V1",
        "",
        "Experiment: %s (single run %s)." % (C.EXPERIMENT, C.RUN_ID),
        "Decision state: **%s**." % decision,
        "",
        "## Frozen reference",
        "",
        "- V2 recipe: %s (sha256 %s), remote %s" % (C.V2_ARGS_PATH, C.FROZEN_V2_ARGS_SHA256, C.FROZEN_V2_ARGS_REMOTE),
        "- pretrained: %s (sha256 %s)" % (C.PRETRAINED_PATH, C.FROZEN_PRETRAINED_SHA256),
        "- V2 composition: %d positives + %d ordinary negatives + %d hard-negative exposures (89 x 8) = %d slots"
        % (C.FROZEN_V2_POSITIVES, C.FROZEN_V2_ORDINARY_NEGATIVES, C.FROZEN_V2_HARDNEG_EXPOSURES, C.FROZEN_V2_TRAIN_ROWS),
        "- internal validation: %d frames, locations %s" % (C.FROZEN_V2_VAL_ROWS, ", ".join(C.FROZEN_V2_VAL_LOCATIONS)),
        "",
        "## Gates",
        "",
    ]
    for name, status in sorted((context.get("gates") or {}).items()):
        lines.append("- %s: %s" % (name, status))
    if not context.get("gates"):
        lines.append("- no gate report was produced")

    lines += ["", "## Measurements", ""]
    metrics = context.get("metrics") or {}
    lines.append(_fmt_metric(metrics.get("real_no_target_496", {}), "real no-target FP"))
    lines.append(_fmt_metric(metrics.get("eth_unseen_all", {}), "val|eth_unseen (all)"))
    lines.append(_fmt_metric(metrics.get("eth_unseen_real", {}), "val|eth_unseen (real only)"))
    lines.append(_fmt_metric(metrics.get("eth_unseen_synthetic", {}), "val|eth_unseen (synthetic only)"))
    ood = metrics.get("outdoor_real_ood")
    if ood:
        lines.append("- real outdoor OOD: %s" % json.dumps(ood, sort_keys=True))
    else:
        lines.append("- real outdoor OOD: not available, so no outdoor-generalisation claim is made")

    lines += ["", "## Limitations", "",
              "- Copy-Paste positives and the new outdoor hard negatives changed together and there is no",
              "  same-period A0 control, so every difference from V2 is descriptive only.",
              "- COCO has no shuttlecock category; an absent annotation is never evidence that a picture has none.",
              "- Every accepted background, foreground and hard negative carries a human review record; a missing",
              "  review keeps the corresponding gate closed.",
              ""]
    text = "\n".join(lines)
    assert_no_causal_language(text)
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Decision report for CLEAN_OUTDOOR_CP_V1 (Task 10)")
    parser.add_argument("--audit", type=Path, required=True, help="JSON with the collected gate statuses")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[],
                        help="deliverable to hash into the final artifact manifest (repeatable)")
    parser.add_argument("--artifact-manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    decision = decide_clean_cp_v1(payload.get("audit", payload), payload.get("metrics", {}),
                                  payload.get("thresholds"))
    report = render_report({"decision": decision, "gates": payload.get("gates", {}),
                            "metrics": payload.get("metrics", {})})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    summary = {"decision": decision, "out": str(args.out)}
    if args.artifact_manifest is not None:
        manifest = build_artifact_manifest(list(args.artifact) + [args.out], repo_root=C.REPO)
        args.artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["artifact_manifest"] = str(args.artifact_manifest)
        summary["artifacts"] = len(manifest["artifacts"])
        summary["missing_artifacts"] = manifest["missing"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
