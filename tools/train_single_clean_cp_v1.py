#!/usr/bin/env python3
"""Single-run training launcher for CLEAN_OUTDOOR_CP_V1 (Task 8).

There is exactly one model, initialised from the official yolo26s.pt, trained for the frozen 50 epochs on the
frozen schedule, selected on the internal V2 validation split only. This module does not reimplement training:
it reuses tools/train_eth_only_v1.py (the canonical trainer) and refuses to start unless every upstream QA gate
is PASS and the frozen recipe, the pretrained hash and the rehearsal schedule all verify.

Because a smoke run needs the real datasets and a GPU, the launcher can build and validate the exact argv without
executing it (execute=False). That is what the tests assert; executing it is a deliberate, gated action.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 8).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_single_clean_cp_train as B  # noqa: E402
import clean_cp_contract as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

PASS = "PASS"
BLOCKED_PREFLIGHT = "BLOCKED_TRAINING_PREFLIGHT"

#: Every gate that must be PASS before a single epoch may start, in the order the plan lists them.
REQUIRED_GATES = (
    "Source contract",
    "COCO inventory",
    "Foreground",
    "CLEAN backgrounds",
    "Outdoor hard negatives",
    "Sampling",
    "Composite QA",
)

#: Paths a training-time operation may never touch: the frozen evaluation sets.
FORBIDDEN_EVAL_HINTS = (
    "challenge_test", "controlled_capability", "real_images", "real_video", "real_match_frames",
    "real_train", "val|eth_unseen", "eth_unseen", "no_target",
)


class TrainingBlocked(RuntimeError):
    """Raised when the run must not start: never downgraded to a warning."""


def _gate_status(payload: dict):
    """Accept every shape our audit files use, but never guess: unknown shape returns None."""
    gate = payload.get("gate")
    if isinstance(gate, dict) and "status" in gate:
        return gate["status"]
    if isinstance(gate, str):
        return gate
    if "status" in payload:
        return payload["status"]
    return None


def read_gate(path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "status": None, "detail": "audit file is missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "status": None, "detail": "audit file is unreadable: %s" % exc}
    return {"path": str(path), "status": _gate_status(payload), "name": payload.get("task") or payload.get("gate")}


def assert_no_eval_access(paths) -> None:
    """Training-side helper: any path that belongs to a frozen evaluation set is a hard error."""
    offenders = [str(path) for path in paths
                 if any(hint in str(path).replace("\\", "/").lower() for hint in FORBIDDEN_EVAL_HINTS)]
    if offenders:
        raise TrainingBlocked("training code may not touch evaluation data: %s" % offenders)


def assert_selection_contract(config: dict) -> None:
    """Best-epoch selection may only use the internal V2 validation split."""
    evaluation = config.get("evaluation") or {}
    scope = str(evaluation.get("selection_scope") or config.get("selection_scope") or "")
    if "internal" not in scope.lower():
        raise TrainingBlocked("checkpoint selection must be internal V2 validation only, got %r" % scope)
    if str(evaluation.get("selection_metric") or config.get("selection_metric") or "") != "mAP50-95":
        raise TrainingBlocked("selection metric must be mAP50-95")
    # Evaluation-only sets may legitimately appear under evaluation.eval_sets; what must never mention them is
    # the SELECTION configuration itself.
    selection_text = json.dumps({key: value for key, value in evaluation.items() if key.startswith("selection")})
    forbidden = [name for name in ("challenge_test", "controlled_capability", "real_no_target_496", "val|eth_unseen")
                 if name in selection_text]
    if forbidden:
        raise TrainingBlocked("evaluation-only sets must never drive selection: %s" % forbidden)


def verify_training_preflight(config_path, audits, schedule, pretrained=None, budget=None) -> dict:
    """Verify every upstream gate, the frozen recipe and the rehearsal schedule before any epoch."""
    budget = dict(budget or B.default_budget())
    config_path = Path(config_path)
    config = C.load_yaml_mapping(config_path)

    problems: list = []
    gates: dict = {}

    try:
        C.assert_single_model_config(config)
    except C.ContractBlocked as exc:
        problems.append("single-model contract: %s" % exc)
    try:
        assert_selection_contract(config)
    except TrainingBlocked as exc:
        problems.append(str(exc))

    contract = C.load_v2_contract(REPO / C.V2_ARGS_PATH, REPO / C.V2_TRAIN_MANIFEST_PATH,
                                  REPO / C.V2_VAL_MANIFEST_PATH, REPO / C.PRETRAINED_PATH)
    try:
        C.assert_frozen_recipe(config.get("resolved_kwargs") or {}, contract)
    except C.ContractBlocked as exc:
        problems.append("frozen recipe: %s" % exc)

    for path in audits:
        gate = read_gate(path)
        gates[Path(path).name] = gate["status"]
        if gate["status"] != PASS:
            problems.append("gate %s is %r (%s)" % (Path(path).name, gate["status"], gate.get("detail", "")))

    weights = Path(pretrained) if pretrained else (REPO / C.PRETRAINED_PATH)
    try:
        C.require_file(weights, "official pretrained yolo26s.pt")
        if C.sha256_file(weights) != C.FROZEN_PRETRAINED_SHA256:
            problems.append("pretrained checkpoint is not the frozen official yolo26s.pt")
    except C.ContractBlocked as exc:
        problems.append(str(exc))

    schedule_path = Path(schedule) if schedule else None
    exposure: dict = {}
    if schedule_path is None or not schedule_path.is_file():
        problems.append("frozen schedule is missing: %s" % schedule_path)
    else:
        sampler = B.CleanCPIndexSampler(schedule_path, budget)
        exposure = sampler.exposure_report()
        by_role = exposure["by_role"]
        if by_role.get(B.ROLE_V2_ORDINARY_NEGATIVE) != budget["old_ordinary_negatives"] * len(sampler.epochs()):
            problems.append("schedule does not preserve the 551 old ordinary negatives per epoch")
        if by_role.get(B.ROLE_V2_HARD_NEGATIVE) != budget["old_hard_negative_exposures"] * len(sampler.epochs()):
            problems.append("schedule does not preserve the 712 old hard-negative exposures per epoch")

    steps = {
        "schedule_plan": B.step_plan(budget),
        "canonical_nominal": _canonical_steps(budget),
        "note": "two conventions are recorded on purpose: the canonical repository arithmetic counts "
                "floor(images/batch) dataloader batches per epoch, while the accumulation-adjusted optimizer "
                "steps are ceil(batches / gradient_accumulation). The authoritative count is the one measured "
                "from the real len(dataloader) during the smoke run; nothing here guesses it.",
    }

    return {
        "ok": not problems,
        "status": PASS if not problems else BLOCKED_PREFLIGHT,
        "problems": problems,
        "gates": gates,
        "required_gates": list(REQUIRED_GATES),
        "steps": steps,
        "exposure": exposure,
        "pretrained": str(weights),
        "config": str(config_path),
    }


def _canonical_steps(budget: dict) -> dict:
    """The repository's own arithmetic, imported lazily so this module stays torch-free."""
    try:
        from train_eth_only_v1 import optimizer_steps as canonical
    except Exception as exc:  # noqa: BLE001
        return {"error": "canonical optimizer_steps unavailable: %s" % exc}
    return canonical(budget["total_slots"], budget["batch"], budget["epochs"], budget["nbs"])


def build_run_argv(config_path, out_dir, smoke: bool = False, data_yaml=None, epochs=None) -> list:
    """The exact canonical-trainer argv. Frozen knobs are pinned; nothing is left to a default."""
    config = C.load_yaml_mapping(config_path)
    frozen = dict(config.get("frozen") or {})
    argv = [sys.executable, str(REPO / "tools" / "train_eth_only_v1.py"),
            "--contract", str(config_path),
            "--out-dir", str(out_dir)]
    if data_yaml:
        argv += ["--data-yaml", str(data_yaml)]
    argv += ["--batch", str(frozen.get("batch", B.DEFAULT_BATCH)),
             "--epochs", str(int(epochs if epochs is not None else frozen.get("epochs", B.DEFAULT_EPOCHS))),
             "--imgsz", str(frozen.get("imgsz", 1024)),
             "--seed", str(frozen.get("seed", B.DEFAULT_SEED))]
    if smoke:
        argv.append("--smoke")
    return argv


def run_single_clean_cp(config_path, audits, schedule, out_dir, *, smoke: bool = False, execute: bool = False,
                        runner=None, budget=None) -> dict:
    """Preflight, then optionally hand the frozen argv to the canonical trainer (the only formal run)."""
    preflight = verify_training_preflight(config_path, audits, schedule, budget=budget)
    result = {"preflight": preflight, "executed": False, "smoke": bool(smoke)}
    if not preflight["ok"]:
        result["status"] = BLOCKED_PREFLIGHT
        result["detail"] = "no epoch may start while any gate is open"
        return result

    argv = build_run_argv(config_path, out_dir, smoke=smoke)
    result["argv"] = argv
    if not execute:
        result["status"] = "READY_NOT_EXECUTED"
        return result

    caller = runner or _default_runner
    result["returncode"] = caller(argv)
    result["executed"] = True
    result["status"] = "EXECUTED" if result["returncode"] == 0 else "TRAINING_FAILED"
    return result


def _default_runner(argv) -> int:
    import subprocess
    return subprocess.call(argv, cwd=str(REPO))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Single-run training launcher for CLEAN_OUTDOOR_CP_V1 (Task 8)")
    parser.add_argument("--config", type=Path, default=REPO / C.NEW_CONFIG_PATH)
    parser.add_argument("--audit", type=Path, action="append", default=[])
    parser.add_argument("--schedule", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/shuttle_capability/clean_outdoor_cp_v1/runs/CLEAN_OUTDOOR_CP_V1")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="verify the gates without training")
    args = parser.parse_args(argv)

    if args.preflight_only:
        report = verify_training_preflight(args.config, args.audit, args.schedule)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2

    result = run_single_clean_cp(args.config, args.audit, args.schedule, args.out_dir, smoke=args.smoke, execute=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"EXECUTED", "READY_NOT_EXECUTED"} else 2


if __name__ == "__main__":
    sys.exit(main())
