#!/usr/bin/env python3
"""Build the tiny-object recovery V1 dataset: V2 rows plus exactly 712 tiny-positive exposures.

The only variable this experiment may change is the training exposure of EXISTING real tiny positives, so the
builder is written as a set of gates rather than as a generator: the unique positive set must stay identical to
V2, the internal val must stay byte-identical, no non-tiny row may be added, no unique image may be repeated
more than once, and the exposure budget must be hit exactly. If the eligible pool cannot fund the budget the
audit raises instead of silently raising the per-image repeat (design section 5).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_eth_real_hardneg_v2_dataset as v2b  # noqa: E402  (manifest loader + canonical eval enumeration)
import train_eth_only_v1 as T  # noqa: E402  (canonical contract loader)

BUCKETS = ("<4", "4-6", "6-8")


class TinyRecoveryAuditError(RuntimeError):
    """Raised for any violation of the frozen tiny-recovery contract."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_manifest(path) -> pd.DataFrame:
    """Read a training manifest that may legitimately repeat rows (V2 repeats each hard negative 8 times),
    unlike the candidate-pool loader in the V2 builder which refuses duplicate SHA256s by design."""
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [c for c in v2b.V1_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise TinyRecoveryAuditError("train manifest %s misses columns: %s" % (path, missing))
    df["is_negative"] = df["is_negative"].astype(str).str.lower().isin(["true", "1"])
    df["is_positive"] = ~df["is_negative"]
    return df


def bucket_of_tiny(eq: float) -> str:
    eq = float(eq)
    if eq < 4.0:
        return "<4"
    if eq < 6.0:
        return "4-6"
    return "6-8"


def tiny_pool(v2_train: pd.DataFrame, max_eq: float = 8.0) -> pd.DataFrame:
    pos = v2_train[v2_train["is_positive"]].copy()
    pool = pos[pos["equiv_size_640"].astype(float) < float(max_eq)].copy()
    pool["size_bucket"] = pool["equiv_size_640"].apply(bucket_of_tiny)
    return pool


def allocate_stratified(counts: dict, budget: int) -> dict:
    """Largest-remainder allocation of the budget over the buckets, preserving their original proportions."""
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        raise TinyRecoveryAuditError("no eligible tiny positives")
    if budget > total:
        raise TinyRecoveryAuditError("budget %d exceeds the eligible pool %d; the design forbids raising a "
                                     "per-image repeat to compensate" % (budget, total))
    raw = {b: budget * int(counts[b]) / total for b in counts}
    alloc = {b: int(raw[b]) for b in raw}
    remainder = budget - sum(alloc.values())
    for b in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), BUCKETS.index(k) if k in BUCKETS else 99)):
        if remainder <= 0:
            break
        alloc[b] += 1
        remainder -= 1
    return alloc


def select_tiny_exposures(pool: pd.DataFrame, alloc: dict, seed: int) -> pd.DataFrame:
    """Pick the unique images that get one extra exposure, stratified and deterministic."""
    picked = []
    for bucket in sorted(alloc, key=lambda b: BUCKETS.index(b) if b in BUCKETS else 99):
        want = int(alloc[bucket])
        if want == 0:
            continue
        sub = pool[pool["size_bucket"] == bucket].sort_values("sha256")
        if want > len(sub):
            raise TinyRecoveryAuditError("bucket %s needs %d extra exposures but only %d images are eligible"
                                         % (bucket, want, len(sub)))
        picked.append(sub.sample(n=want, random_state=int(seed), replace=False))
    if not picked:
        return pool.iloc[0:0].copy()
    return pd.concat(picked, ignore_index=True)


def build_tiny_train_manifest(v2_train: pd.DataFrame, selected: pd.DataFrame, max_eq: float = 8.0) -> pd.DataFrame:
    base = v2_train.copy()
    base["extra_exposure"] = 0
    base["selection_seed"] = 0
    extra = selected.copy()
    if len(extra):
        extra["extra_exposure"] = 1
        extra["selection_seed"] = 0
        base["selection_seed"] = 0
        for col in set(extra.columns) - set(base.columns):
            base[col] = ""
        extra = extra[base.columns]
    return pd.concat([base, extra], ignore_index=True, sort=False)


def exposure_table(v2_train: pd.DataFrame, selected: pd.DataFrame, max_eq: float = 8.0, seed: int = 42) -> list:
    """Per-image exposure accounting for every tiny positive that exists in V2 (design section 20)."""
    pool = tiny_pool(v2_train, max_eq).copy()
    chosen = {str(s) for s in selected["sha256"]} if len(selected) else set()
    pool["base_exposure"] = 1
    pool["extra_exposure"] = [1 if s in chosen else 0 for s in pool["sha256"]]
    pool["final_exposure"] = pool["base_exposure"] + pool["extra_exposure"]
    pool["selection_seed"] = int(seed)
    cols = ["image", "sha256", "location", "equiv_size_640", "size_bucket", "base_exposure", "extra_exposure",
            "final_exposure", "selection_seed"]
    out = pool[cols].sort_values(["size_bucket", "sha256"])
    return out.to_dict("records")


def source_counts(v2_train: pd.DataFrame, selected: pd.DataFrame, max_eq: float = 8.0) -> list:
    pool = tiny_pool(v2_train, max_eq)
    chosen = {str(s) for s in selected["sha256"]} if len(selected) else set()
    rows = []
    counts = Counter(zip(pool["source"].astype(str), pool["size_bucket"].astype(str)))
    for (source, bucket), n in sorted(counts.items()):
        extra = sum(1 for _, r in pool[(pool["source"].astype(str) == source)
                                       & (pool["size_bucket"].astype(str) == bucket)].iterrows()
                    if str(r["sha256"]) in chosen)
        rows.append({"source": source, "size_bucket": bucket, "v2_exposure": n, "extra_exposure": extra,
                     "final_exposure": n + extra})
    return rows


def exposure_summary(v2_train: pd.DataFrame, selected: pd.DataFrame, max_eq: float = 8.0) -> list:
    pool = tiny_pool(v2_train, max_eq)
    chosen = {str(s) for s in selected["sha256"]} if len(selected) else set()
    rows = []
    for bucket in BUCKETS:
        sub = pool[pool["size_bucket"] == bucket]
        extra = sum(1 for s in sub["sha256"] if str(s) in chosen)
        rows.append({"bucket": bucket, "v2_exposure": int(len(sub)), "extra_exposure": int(extra),
                     "final_exposure": int(len(sub) + extra)})
    pos = v2_train[v2_train["is_positive"]]
    non_tiny = pos[pos["equiv_size_640"].astype(float) >= float(max_eq)]
    rows.append({"bucket": ">=8", "v2_exposure": int(len(non_tiny)), "extra_exposure": 0,
                 "final_exposure": int(len(non_tiny))})
    return rows


def audit_tiny_dataset(v2_train, v2_val, tiny_train, selected, eval_paths, eval_shas,
                       v2_val_manifest=None, tiny_val_manifest=None, budget=712, max_eq=8.0) -> dict:
    problems = []
    v2_pos = v2_train[v2_train["is_positive"]]
    tiny_pos = tiny_train[tiny_train["is_positive"]]
    v2_digest = v2b.hash_set_digest(v2_pos["sha256"])
    tiny_digest = v2b.hash_set_digest(tiny_pos["sha256"])
    if set(tiny_pos["sha256"]) != set(v2_pos["sha256"]):
        problems.append("unique positive image set drifted from V2")
    extra = tiny_train[tiny_train["extra_exposure"].astype(int) == 1]
    if len(extra) != int(budget):
        problems.append("extra tiny exposure is %d, budget is %d" % (len(extra), budget))
    if len(extra) and (extra["equiv_size_640"].astype(float) >= float(max_eq)).any():
        problems.append("a non-tiny positive received extra exposure")
    per_image = Counter(extra["sha256"]) if len(extra) else Counter()
    if per_image and max(per_image.values()) > 1:
        problems.append("a unique image received more than one extra exposure")
    hn_paths = set(tiny_train[tiny_train["role"] == "hard_negative"]["image"].astype(str).str.replace(chr(92), "/"))
    hn_shas = set(tiny_train[tiny_train["role"] == "hard_negative"]["sha256"].astype(str).str.lower())
    path_overlap = len(hn_paths & set(eval_paths))
    sha_overlap = len(hn_shas & set(eval_shas))
    if path_overlap or sha_overlap:
        problems.append("evaluation overlap: %d paths, %d hashes" % (path_overlap, sha_overlap))
    prohibited = Counter()
    for src in set(tiny_train["source"].astype(str)):
        low = src.lower()
        if any(k in low for k in v2b.PROHIBITED_SOURCE_HINTS):
            prohibited[src] += 1
    if prohibited:
        problems.append("prohibited sources present: %s" % dict(prohibited))
    if v2_val_manifest and tiny_val_manifest:
        if sha256_file(v2_val_manifest) != sha256_file(tiny_val_manifest):
            problems.append("internal val manifest is not byte-identical to V2")

    report = {
        "status": "FAIL" if problems else "PASS",
        "problems": problems,
        "unique_positive_set_v2_sha256": v2_digest,
        "unique_positive_set_tiny_sha256": tiny_digest,
        "unique_positive_set_identical": v2_digest == tiny_digest,
        "v2_positive_count": int(len(v2_pos)),
        "tiny_positive_count": int(len(tiny_pos)),
        "extra_tiny_exposure": int(len(extra)),
        "extra_non_tiny_positive": int((extra["equiv_size_640"].astype(float) >= float(max_eq)).sum())
                                    if len(extra) else 0,
        "tiny_extra_per_unique_image_max": int(max(per_image.values())) if per_image else 0,
        "eligible_tiny_unique": int(len(tiny_pool(v2_train, max_eq))),
        "v2_train_rows": int(len(v2_train)),
        "tiny_train_rows": int(len(tiny_train)),
        "extra_exposure_fraction_of_train": (float(len(extra)) / float(len(tiny_train))) if len(tiny_train) else None,
        "extra_exposure_fraction_of_v2_train": (float(len(extra)) / float(len(v2_train))) if len(v2_train) else None,
        "eval_path_overlap_count": path_overlap,
        "eval_sha256_overlap_count": sha_overlap,
        "prohibited_source_counts": dict(prohibited),
        "hard_negative_rows": int(len(tiny_train[tiny_train["role"] == "hard_negative"])),
        "v2_val_manifest_sha256": sha256_file(v2_val_manifest) if v2_val_manifest else None,
        "tiny_val_manifest_sha256": sha256_file(tiny_val_manifest) if tiny_val_manifest else None,
        "exposure_summary": exposure_summary(v2_train, selected, max_eq),
    }
    if problems:
        raise TinyRecoveryAuditError("; ".join(problems))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the tiny-object recovery V1 dataset (V2 + 712 tiny exposures)")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-manifest", action="append", default=[])
    ap.add_argument("--no-frozen-sets", action="store_true")
    ap.add_argument("--train-out", default=None)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--yaml-out", default=None)
    ap.add_argument("--audit-out", default=None)
    ap.add_argument("--exposure-out", default=None)
    ap.add_argument("--source-counts-out", default=None)
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve() if a.repo else Path(__file__).resolve().parents[1]
    cfg = T.load_contract(repo / a.config)
    src = repo / str(cfg["tiny_source_manifest"])
    v2_train = load_train_manifest(src)
    max_eq = float(cfg["tiny_positive_max_equiv_size_640"])
    budget = int(cfg["tiny_positive_extra_exposure"])
    seed = int(cfg["tiny_selection_seed"])

    pool = tiny_pool(v2_train, max_eq)
    counts = Counter(pool["size_bucket"])
    if len(pool) < budget and cfg["tiny_stop_if_insufficient"]:
        raise TinyRecoveryAuditError("eligible tiny images %d < budget %d: STOP (design forbids compensating by "
                                     "raising the per-image repeat)" % (len(pool), budget))
    alloc = allocate_stratified(counts, budget)
    selected = select_tiny_exposures(pool, alloc, seed)
    tiny_train = build_tiny_train_manifest(v2_train, selected, max_eq)

    v2_val_manifest = repo / "data" / "eth_real_hardneg_v2_val_manifest.csv"
    eval_manifests = [str(v2_val_manifest)] + [str(m) for m in a.eval_manifest]
    eval_paths, eval_shas = v2b.collect_eval_images(repo, eval_manifests,
                                                    include_frozen_sets=not a.no_frozen_sets)

    train_out = repo / a.train_out
    val_out = repo / a.val_out
    yaml_out = repo / a.yaml_out
    audit_out = repo / a.audit_out
    for p in (train_out.parent, val_out.parent, yaml_out.parent, audit_out.parent):
        p.mkdir(parents=True, exist_ok=True)
    val_out.write_bytes(v2_val_manifest.read_bytes())
    report = audit_tiny_dataset(v2_train, v2_val_manifest, tiny_train, selected, eval_paths, eval_shas,
                                v2_val_manifest=v2_val_manifest, tiny_val_manifest=val_out,
                                budget=budget, max_eq=max_eq)
    report["allocation"] = {k: int(v) for k, v in alloc.items()}
    report["config_sha256"] = sha256_file(repo / a.config)
    tiny_train.to_csv(train_out, index=False)
    stem = train_out.name.replace("_manifest.csv", "")
    train_list = train_out.with_name(stem + "_train.txt")
    val_list = train_out.with_name(stem.replace("_train", "_val") + ".txt")
    train_list.write_text(chr(10).join(tiny_train["image"].astype(str)) + chr(10), encoding="utf-8")
    val_list.write_text(chr(10).join(load_train_manifest(v2_val_manifest)["image"].astype(str)) + chr(10),
                        encoding="utf-8")
    yaml_out.write_text(chr(10).join([
        "path: data/" + Path(a.train_out).name.replace("_train_manifest.csv", ""),
        "train: " + train_list.name,
        "val: " + val_list.name,
        "nc: 1",
        "names:",
        "  0: shuttlecock",
    ]) + chr(10), encoding="utf-8")

    if a.exposure_out:
        cols = list(exposure_table(v2_train, selected, max_eq, seed)[0].keys())
        with (repo / a.exposure_out).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in exposure_table(v2_train, selected, max_eq, seed):
                w.writerow(row)
    if a.source_counts_out:
        rows = source_counts(v2_train, selected, max_eq)
        with (repo / a.source_counts_out).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)
    report["train_manifest_sha256"] = sha256_file(train_out)
    report["train_list_sha256"] = sha256_file(train_list)
    report["val_list_sha256"] = sha256_file(val_list)
    report["yaml_sha256"] = sha256_file(yaml_out)
    audit_out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[tiny] status=%s rows=%d extra=%d allocation=%s" % (report["status"], report["tiny_train_rows"],
                                                               report["extra_tiny_exposure"], report["allocation"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
