#!/usr/bin/env python3
"""One-run exposure budget and replayed sampling schedule for CLEAN_OUTDOOR_CP_V1 (Task 7).

The plan freezes a single 50-epoch run whose every epoch has exactly 15,255 sample slots, keeps the old 551
ordinary negatives and the old 89x8 = 712 hard-negative exposures untouched, holds at least 11,194 original
positives per epoch, and lets the two NEW pools (copy-paste positives and outdoor hard negatives) share at
most 2,798 replacement slots - with new hard negatives capped at 762 (5%) and used at most once per unique
image per epoch. No slot is invented: the 50-epoch schedule is materialised once and replayed by the loader.

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Task 7).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_cp_contract as C  # noqa: E402

ROLE_V2_POSITIVE = "v2_positive"
ROLE_V2_ORDINARY_NEGATIVE = "v2_ordinary_negative"
ROLE_V2_HARD_NEGATIVE = "v2_hard_negative"
ROLE_CP_POSITIVE = "cp_positive"
ROLE_NEW_HARD_NEGATIVE = "new_hard_negative"

SLOT_COLUMNS = (
    "epoch", "slot_index", "role", "sample_id", "label_path", "is_positive", "repeat_index",
    "coco_image_id", "background_id", "near_duplicate_group", "exposure_index",
)

# Frozen arithmetic copied from the plan (Task 7), restated here so the numbers exist in exactly one place.
TOTAL_SLOTS = 15255
OLD_ORDINARY_NEGATIVES = 551
OLD_HARD_NEGATIVE_EXPOSURES = 712
MIN_ORIGINAL_POSITIVES = 11194
MAX_NEW_HN_SLOTS = 762
MAX_JOINT_REPLACEMENT_SLOTS = 2798
DEFAULT_EPOCHS = 50
DEFAULT_SEED = 42
DEFAULT_BATCH = 8
DEFAULT_NBS = 32


class BudgetConflict(RuntimeError):
    """Raised when the frozen budget cannot be honoured without silently changing the experiment."""


def default_budget() -> dict:
    return {
        "total_slots": TOTAL_SLOTS,
        "old_ordinary_negatives": OLD_ORDINARY_NEGATIVES,
        "old_hard_negative_exposures": OLD_HARD_NEGATIVE_EXPOSURES,
        "min_original_positives": MIN_ORIGINAL_POSITIVES,
        "max_new_hn_slots": MAX_NEW_HN_SLOTS,
        "max_joint_replacement_slots": MAX_JOINT_REPLACEMENT_SLOTS,
        "epochs": DEFAULT_EPOCHS,
        "seed": DEFAULT_SEED,
        "batch": DEFAULT_BATCH,
        "nbs": DEFAULT_NBS,
        "cp_allocation": "fill_remaining_cap",
        "step_inflation_allowed": False,
    }


def load_v2_rows(manifest, role_column: str = "role") -> pd.DataFrame:
    """Map the frozen V2 manifest onto slot columns. The three V2 roles must be exactly as frozen."""
    frame = pd.read_csv(manifest, keep_default_na=False, dtype=str)
    records = []
    for row in frame.to_dict("records"):
        is_positive = str(row.get("is_positive", "")).strip().lower() in {"true", "1", "yes"}
        repeated = int(row.get("repeat_index") or 0)
        is_hard = repeated > 0 or row.get("role") == "hard_negative"
        if is_positive:
            role = ROLE_V2_POSITIVE
        elif is_hard:
            role = ROLE_V2_HARD_NEGATIVE
        else:
            role = ROLE_V2_ORDINARY_NEGATIVE
        records.append({
            "role": role,
            "sample_id": row.get("image") or row.get("abs_path"),
            "label_path": row.get("label") or row.get("abs_label_path") or "",
            "is_positive": is_positive,
            "repeat_index": repeated,
            "coco_image_id": None,
            "background_id": None,
            "near_duplicate_group": None,
        })
    return pd.DataFrame(records, columns=list(SLOT_COLUMNS[2:]))


def _assert_single_run(config: dict) -> None:
    """One experiment, one model, one schedule. Anything else is refused before any slot is built."""
    forbidden = [key for key in ("runs", "arms", "variants", "extra_runs") if config.get(key)]
    if forbidden:
        raise BudgetConflict("exactly one run is allowed; config declares %s" % forbidden)
    if config.get("step_inflation_allowed"):
        raise BudgetConflict("step inflation is forbidden: the 50 epochs and slot count are frozen")


def assert_pools_disjoint(cp_rows: pd.DataFrame, new_hn_rows: pd.DataFrame) -> None:
    """The two new pools must be mutually exclusive at COCO id and near-duplicate-group level."""
    cp_ids = set(cp_rows["coco_image_id"].dropna().astype(str)) - {""}
    hn_ids = set(new_hn_rows["coco_image_id"].dropna().astype(str)) - {""}
    shared_ids = cp_ids & hn_ids
    if shared_ids:
        raise BudgetConflict("COCO image ids in both new pools: %s" % sorted(shared_ids)[:5])
    cp_groups = set(cp_rows["near_duplicate_group"].dropna().astype(str)) - {""}
    hn_groups = set(new_hn_rows["near_duplicate_group"].dropna().astype(str)) - {""}
    shared_groups = cp_groups & hn_groups
    if shared_groups:
        raise BudgetConflict("near-duplicate groups in both new pools: %s" % sorted(shared_groups)[:5])


def build_epoch_slots(v2_rows: pd.DataFrame, cp_rows: pd.DataFrame, new_hn_rows: pd.DataFrame,
                      config: dict, epoch: int) -> pd.DataFrame:
    """Deterministically materialise one epoch: old negatives kept byte-for-byte, floors and caps enforced."""
    _assert_single_run(config)
    total = int(config["total_slots"])
    old_ordinary = v2_rows[v2_rows["role"] == ROLE_V2_ORDINARY_NEGATIVE]
    old_hard = v2_rows[v2_rows["role"] == ROLE_V2_HARD_NEGATIVE]
    positives = v2_rows[v2_rows["role"] == ROLE_V2_POSITIVE]

    if len(old_ordinary) != int(config["old_ordinary_negatives"]):
        raise BudgetConflict("frozen V2 ordinary negatives changed: %d" % len(old_ordinary))
    if len(old_hard) != int(config["old_hard_negative_exposures"]):
        raise BudgetConflict("frozen V2 hard-negative exposures changed: %d" % len(old_hard))

    replacement_budget = total - len(old_ordinary) - len(old_hard)
    joint_cap = min(int(config["max_joint_replacement_slots"]),
                    replacement_budget - int(config["min_original_positives"]))
    if joint_cap < 0:
        raise BudgetConflict("the frozen positives floor leaves no replacement slots")

    unique_new_hn = new_hn_rows.drop_duplicates(subset=["sample_id"])
    new_hn_slots = min(int(config["max_new_hn_slots"]), len(unique_new_hn), joint_cap)
    cp_available = joint_cap - new_hn_slots
    if str(config.get("cp_allocation", "fill_remaining_cap")) == "fill_remaining_cap":
        cp_slots = min(len(cp_rows), cp_available)
    else:
        raise BudgetConflict("unknown cp_allocation %r" % config.get("cp_allocation"))
    original_positive_slots = replacement_budget - cp_slots - new_hn_slots

    if original_positive_slots < int(config["min_original_positives"]):
        raise BudgetConflict(
            "original positives would fall to %d, below the frozen floor %d"
            % (original_positive_slots, int(config["min_original_positives"]))
        )

    rng = random.Random(int(config["seed"]) + int(epoch))
    chosen_positive_indexes = rng.sample(range(len(positives)), original_positive_slots)
    chosen_cp_indexes = rng.sample(range(len(cp_rows)), cp_slots) if cp_slots else []
    chosen_hn_indexes = rng.sample(range(len(unique_new_hn)), new_hn_slots) if new_hn_slots else []

    parts = []
    parts.append(old_ordinary.assign(epoch=epoch))
    parts.append(old_hard.assign(epoch=epoch))
    parts.append(positives.iloc[chosen_positive_indexes].assign(epoch=epoch))
    if cp_slots:
        parts.append(cp_rows.iloc[chosen_cp_indexes].assign(epoch=epoch))
    if new_hn_slots:
        parts.append(unique_new_hn.iloc[chosen_hn_indexes].assign(epoch=epoch))

    slots = pd.concat(parts, ignore_index=True)
    for column in SLOT_COLUMNS:
        if column not in slots.columns:
            slots[column] = None
    slots = slots[list(SLOT_COLUMNS)]
    slots["epoch"] = int(epoch)
    slots = slots.sample(frac=1.0, random_state=int(config["seed"]) * 1000 + int(epoch)).reset_index(drop=True)
    slots["slot_index"] = range(len(slots))
    slots["exposure_index"] = slots.groupby("sample_id").cumcount()
    return slots


def validate_epoch_budget(slots: pd.DataFrame, config: dict) -> dict:
    """Check one epoch against every frozen budget clause. Returns counts plus a PASS/BLOCKED gate."""
    violations: list = []
    total = int(config["total_slots"])
    counts = slots["role"].value_counts().to_dict()

    if len(slots) != total:
        violations.append("epoch has %d slots, expected %d" % (len(slots), total))
    if counts.get(ROLE_V2_ORDINARY_NEGATIVE, 0) != int(config["old_ordinary_negatives"]):
        violations.append("old ordinary negatives %d != %d" % (counts.get(ROLE_V2_ORDINARY_NEGATIVE, 0),
                                                               int(config["old_ordinary_negatives"])))
    if counts.get(ROLE_V2_HARD_NEGATIVE, 0) != int(config["old_hard_negative_exposures"]):
        violations.append("old hard-negative exposures %d != %d" % (counts.get(ROLE_V2_HARD_NEGATIVE, 0),
                                                                    int(config["old_hard_negative_exposures"])))
    if counts.get(ROLE_V2_POSITIVE, 0) < int(config["min_original_positives"]):
        violations.append("original positives %d below the floor %d" % (counts.get(ROLE_V2_POSITIVE, 0),
                                                                        int(config["min_original_positives"])))
    new_hn = counts.get(ROLE_NEW_HARD_NEGATIVE, 0)
    if new_hn > int(config["max_new_hn_slots"]):
        violations.append("new hard negatives %d above the cap %d" % (new_hn, int(config["max_new_hn_slots"])))
    if total and new_hn / float(total) > 0.05 + 1e-12:
        violations.append("new hard negatives are %.4f of the epoch, above 5%%" % (new_hn / float(total)))
    joint = new_hn + counts.get(ROLE_CP_POSITIVE, 0)
    if joint > int(config["max_joint_replacement_slots"]):
        violations.append("joint CP + new-hard-negative slots %d above %d" % (joint, int(config["max_joint_replacement_slots"])))

    hn_rows = slots[slots["role"] == ROLE_NEW_HARD_NEGATIVE]
    duplicated = hn_rows["sample_id"].duplicated().any()
    if duplicated:
        violations.append("a new hard negative was drawn more than once in one epoch")

    cp_rows = slots[slots["role"] == ROLE_CP_POSITIVE]
    if len(cp_rows) and not cp_rows["is_positive"].all():
        violations.append("a copy-paste slot is not marked positive")
    if len(cp_rows) and (cp_rows["label_path"].astype(str).str.strip() == "").any():
        violations.append("a copy-paste slot has an empty label path")

    overlap = set(hn_rows["coco_image_id"].dropna()) & set(cp_rows["coco_image_id"].dropna())
    if overlap:
        violations.append("COCO image ids used as both background and hard negative: %s" % sorted(overlap)[:5])

    return {
        "ok": not violations,
        "violations": violations,
        "counts": counts,
        "total_slots": len(slots),
        "new_hard_negative_fraction": (new_hn / float(total)) if total else 0.0,
        "gate": "PASS" if not violations else "BLOCKED_BUDGET_CONFLICT",
    }


def step_plan(config: dict) -> dict:
    """Frozen optimizer-step arithmetic: derived from the frozen slot count, never from a live run."""
    total = int(config["total_slots"])
    batch = int(config["batch"])
    nbs = int(config["nbs"])
    batches = math.ceil(total / batch)
    accumulate = max(1, nbs // batch)
    steps_per_epoch = math.ceil(batches / accumulate)
    return {
        "dataloader_batches": batches,
        "gradient_accumulation": accumulate,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": steps_per_epoch * int(config["epochs"]),
        "source": "frozen slot count and frozen batch/nbs",
    }


def freeze_training_slots(v2_rows, cp_rows, new_hn_rows, config, output_csv) -> dict:
    """Materialise all epochs once, validate every one, and write the replayable schedule."""
    epochs = int(config["epochs"])
    frames = []
    per_epoch: list = []
    for epoch in range(epochs):
        slots = build_epoch_slots(v2_rows, cp_rows, new_hn_rows, config, epoch)
        report = validate_epoch_budget(slots, config)
        if not report["ok"]:
            raise BudgetConflict("epoch %d violates the frozen budget: %s" % (epoch, report["violations"]))
        per_epoch.append({"epoch": epoch, "counts": report["counts"], "total": report["total_slots"]})
        frames.append(slots)
    schedule = pd.concat(frames, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output_csv, index=False)
    digest = hashlib.sha256(output_csv.read_bytes()).hexdigest()
    return {
        "epochs": epochs,
        "rows": int(len(schedule)),
        "total_slots": TOTAL_SLOTS,
        "per_epoch": per_epoch,
        "steps": step_plan(config),
        "schedule_csv": str(output_csv),
        "schedule_sha256": digest,
    }


class CleanCPIndexSampler:
    """Replays the frozen schedule with no shuffle of its own: what the CSV says is what the loader sees."""

    def __init__(self, schedule_csv, config: dict):
        self.schedule_csv = Path(schedule_csv)
        frame = pd.read_csv(self.schedule_csv, keep_default_na=False, dtype=str)
        frame["epoch"] = frame["epoch"].astype(int)
        frame["slot_index"] = frame["slot_index"].astype(int)
        self.frame = frame.sort_values(["epoch", "slot_index"]).reset_index(drop=True)
        self.config = dict(config)

    def epochs(self) -> list:
        return sorted(self.frame["epoch"].unique().tolist())

    def epoch_frame(self, epoch: int) -> pd.DataFrame:
        return self.frame[self.frame["epoch"] == int(epoch)].reset_index(drop=True)

    def epoch_sample_ids(self, epoch: int) -> list:
        return self.epoch_frame(epoch)["sample_id"].tolist()

    def dataloader_batches(self, epoch: int) -> int:
        return math.ceil(len(self.epoch_frame(epoch)) / int(self.config["batch"]))

    def replay_digest(self) -> str:
        order = "|".join(self.frame["sample_id"].astype(str).tolist())
        return hashlib.sha256(order.encode("utf-8")).hexdigest()

    def exposure_report(self) -> dict:
        grouped = self.frame.groupby("sample_id").size()
        return {
            "unique_samples": int(grouped.shape[0]),
            "max_exposure": int(grouped.max()),
            "total_slots": int(len(self.frame)),
            "by_role": self.frame.groupby("role").size().to_dict(),
        }

    def assert_replay_matches(self, expected: pd.DataFrame) -> None:
        """The loader must reproduce the frozen order exactly; any drift is a hard error."""
        expected = expected.copy()
        # slot_index may arrive as text (a CSV round-trip); sort numerically or the order silently changes
        expected["epoch"] = expected["epoch"].astype(int)
        expected["slot_index"] = expected["slot_index"].astype(int)
        expected_ids = expected.sort_values(["epoch", "slot_index"])["sample_id"].astype(str).tolist()
        actual_ids = self.frame["sample_id"].astype(str).tolist()
        if expected_ids != actual_ids:
            raise BudgetConflict("sampler replay drifted from the frozen schedule")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the CLEAN_OUTDOOR_CP_V1 exposure schedule (Task 7)")
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--cp-manifest", type=Path, required=True)
    parser.add_argument("--new-hn-manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args(argv)

    config = default_budget()
    config["epochs"] = args.epochs
    v2_rows = load_v2_rows(args.v2_manifest)
    cp_rows = pd.read_csv(args.cp_manifest, keep_default_na=False, dtype=str)
    new_hn_rows = pd.read_csv(args.new_hn_manifest, keep_default_na=False, dtype=str)
    for frame in (cp_rows, new_hn_rows):
        for column in ("sample_id", "label_path", "coco_image_id", "background_id", "near_duplicate_group"):
            if column not in frame.columns:
                frame[column] = ""
        frame["is_positive"] = frame.get("is_positive", True)
    summary = freeze_training_slots(v2_rows, cp_rows, new_hn_rows, config, args.out_csv)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": summary["rows"], "epochs": summary["epochs"], "steps": summary["steps"],
                      "sha256": summary["schedule_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
