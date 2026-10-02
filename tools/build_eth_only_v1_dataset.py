#!/usr/bin/env python3
"""Build the audited ETH-only V1 train/val manifests, dataset yaml and data audit (Task 2).

Leakage discipline: every image that appears in ANY evaluation manifest (ours) is removed from the training
candidate pool; the split is location-disjoint; synthetic and iPhone sources are rejected outright.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

FORBIDDEN_SOURCES = {"synthetic", "synthetic_3d", "synthetic_on_real_bg", "isaac_sim", "isaacsim",
                     "eth_iphone", "iphone", "d455", "roboflow", "project_synthetic", "near_field_synthetic"}
ALLOWED_SOURCES = {"eth_main", "eth_negative"}
VAL_BAND = (0.15, 0.20)
BUCKETS = [(0, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 1e9)]
BUCKET_LABELS = ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]


class DatasetAuditError(RuntimeError):
    """Raised when the ETH-only dataset build violates a leakage or content rule."""


def equiv_size_640(width_px: float, height_px: float, original_width: int, original_height: int) -> float:
    scale = 640.0 / max(float(original_width), float(original_height))
    return math.sqrt(max(width_px, 0.0) * max(height_px, 0.0)) * scale


def bucket_of(eq: float) -> str:
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        if lo <= eq < hi:
            return lab
    return BUCKET_LABELS[-1]


def _sha256(path: Path, limit_bytes=None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _label_lines(path: Path):
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return len([ln for ln in txt.splitlines() if ln.strip()])


def inventory_eth_dataset(data_root, recipe, only_sets=None, progress_every=2000) -> pd.DataFrame:
    data_root = Path(data_root)
    locs = list(recipe.get("official_train_locations") or [])
    diffs = list(recipe.get("official_train_difficulties") or [])
    if not locs or not diffs:
        raise DatasetAuditError("recipe has no official training locations/difficulties")
    if not data_root.is_dir():
        raise DatasetAuditError("dataset root not found: %s" % data_root)
    from PIL import Image
    rows = []
    wanted = set(only_sets) if only_sets else None
    for set_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        name = set_dir.name
        if wanted and name not in wanted:
            continue
        loc, _, diff = name.rpartition("_")
        if not loc:
            loc, diff = name, "?"
        if loc == "coco_train":
            role, trained, source = "official_negatives", True, "eth_negative"
        elif loc == "coco_val":
            role, trained, source = "benchmark_holdout", False, "eth_holdout"
        elif loc in locs and diff in diffs:
            role, trained, source = "official_positives", True, "eth_main"
        elif loc in locs:
            role, trained, source = "same_location_holdout", False, "eth_holdout"
        else:
            role, trained, source = "other", False, "eth_other"
        for split in ("train", "val", "test"):
            img_dir = set_dir / "images" / split
            if not img_dir.is_dir():
                continue
            for img in sorted(p for p in img_dir.iterdir() if p.is_file()):
                lab = set_dir / "labels" / split / (img.stem + ".txt")
                has_label = lab.exists()
                n_boxes = _label_lines(lab) if has_label else 0
                try:
                    with Image.open(img) as im:
                        W, H = im.size
                except Exception:
                    continue
                eq = equiv_size_640(1, 1, W, H)
                rows.append({
                    "image": str(img), "label": str(lab) if has_label else "",
                    "location": loc, "difficulty": diff, "split": split, "set_dir": name,
                    "role": role, "in_official_training": trained, "source": source,
                    "is_negative": bool(has_label is False or n_boxes == 0),
                    "n_boxes": n_boxes, "width": W, "height": H, "sha256": _sha256(img),
                    "equiv_size_640": eq,
                })
                if progress_every and len(rows) % progress_every == 0:
                    print("  inventory %d images (last %s)" % (len(rows), name), flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise DatasetAuditError("inventory is empty under %s" % data_root)
    return df


def _subset_fractions(counts: dict, target: float, max_size: int = 3):
    names = sorted(counts)
    total = sum(counts.values())
    best = None
    for size in range(1, max_size + 1):
        for combo in _combinations(names, size):
            n = sum(counts[c] for c in combo)
            frac = n / total
            key = (abs(frac - target), size, combo)
            cand = (key, combo, n, frac)
            if best is None or key < best[0]:
                best = cand
    return best


def _combinations(items, size):
    if size == 1:
        for it in items:
            yield (it,)
        return
    for i in range(len(items) - size + 1):
        for rest in _combinations(items[i + 1:], size - 1):
            yield (items[i],) + rest


def choose_location_split(samples: pd.DataFrame, target_val_fraction: float = 0.18):
    if samples.empty:
        raise DatasetAuditError("cannot split an empty inventory")
    counts = samples.groupby("location").size().to_dict()
    total = sum(counts.values())
    band_candidates = []
    names = sorted(counts)
    for size in range(1, 4):
        for combo in _combinations(names, size):
            n = sum(counts[c] for c in combo)
            frac = n / total
            if VAL_BAND[0] <= frac <= VAL_BAND[1]:
                band_candidates.append(((abs(frac - target_val_fraction), size, combo), combo, n, frac))
    if band_candidates:
        band_candidates.sort(key=lambda t: t[0])
        _, combo, _n, _frac = band_candidates[0]
    else:
        _key, combo, _n, _frac = _subset_fractions(counts, target_val_fraction)
    val_df = samples[samples["location"].isin(list(combo))].copy()
    train_df = samples[~samples["location"].isin(list(combo))].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _collect_eval_index(evaluation_paths):
    paths, hashes, relpaths = set(), set(), set()
    for item in evaluation_paths or []:
        if isinstance(item, pd.DataFrame):
            for col in ("image", "file", "path"):
                if col in item.columns:
                    for v in item[col].astype(str):
                        if v and v.lower() != "nan":
                            paths.add(str(Path(v)))
                    break
            if "sha256" in item.columns:
                for v in item["sha256"].astype(str):
                    if v and v.lower() != "nan":
                        hashes.add(v.lower())
            if "name" in item.columns:
                for v in item["name"].astype(str):
                    if v and v.lower() != "nan":
                        relpaths.add(v)
            continue
        p = Path(item)
        if p.suffix.lower() == ".csv" and p.exists():
            try:
                df = pd.read_csv(p)
            except Exception as exc:
                raise DatasetAuditError("cannot read evaluation manifest %s: %r" % (p, exc))
            for col in ("image", "file", "path"):
                if col in df.columns:
                    for v in df[col].astype(str):
                        if v and v.lower() != "nan":
                            paths.add(str(Path(v)))
                    break
            if "sha256" in df.columns:
                for v in df["sha256"].astype(str):
                    if v and v.lower() != "nan":
                        hashes.add(v.lower())
        else:
            paths.add(str(Path(item)))
    for v in list(paths):
        relpaths.add(v.replace("\\", "/").split("/")[-1])
    return paths, hashes, relpaths


def filter_eval_overlap(candidates: pd.DataFrame, evaluation_paths):
    """Remove every training candidate that appears in ANY of our evaluation manifests.

    Matching is by absolute path, by file name and by sha256, so that the same frame cannot enter training
    under a different root (our local ETH copy vs the remote one) or under a renamed copy.
    """
    if candidates.empty:
        return candidates.copy(), candidates.copy()
    eval_paths, eval_hashes, eval_names = _collect_eval_index(evaluation_paths)
    names = candidates["image"].astype(str).map(lambda p: p.replace("\\", "/").split("/")[-1])
    hit_name = names.isin(eval_names) if eval_names else pd.Series(False, index=candidates.index)
    hit_path = candidates["image"].astype(str).isin(eval_paths)
    hit_hash = candidates["sha256"].astype(str).isin(eval_hashes) if eval_hashes else pd.Series(False, index=candidates.index)
    drop = hit_name | hit_path | hit_hash
    return candidates[~drop].copy(), candidates[drop].copy()


def audit_split(train_df: pd.DataFrame, val_df: pd.DataFrame, evaluation_paths) -> dict:
    problems = []
    for df, tag in ((train_df, "train"), (val_df, "val")):
        if df.empty:
            continue
        bad_src = sorted(set(df["source"]) & FORBIDDEN_SOURCES) if "source" in df.columns else []
        if len(bad_src) == len(set(df["source"])) and bad_src:
            problems.append("%s contains only forbidden sources: %s" % (tag, bad_src))
        elif bad_src:
            problems.append("%s contains forbidden sources: %s" % (tag, bad_src))
        if "source" in df.columns and not set(df["source"]) <= (ALLOWED_SOURCES | FORBIDDEN_SOURCES):
            problems.append("%s has unknown sources: %s" % (tag, sorted(set(df["source"]) - ALLOWED_SOURCES - FORBIDDEN_SOURCES)))
    loc_overlap = sorted(set(train_df.get("location", [])) & set(val_df.get("location", [])))
    if loc_overlap:
        problems.append("train/val locations overlap: %s" % loc_overlap)
    hash_overlap = sorted(set(train_df.get("sha256", [])) & set(val_df.get("sha256", [])))
    if hash_overlap:
        problems.append("train/val sha256 overlap: %d images" % len(hash_overlap))
    nonempty_neg = 0
    if "is_negative" in train_df.columns:
        nonempty_neg = int(((train_df["is_negative"]) & (train_df["n_boxes"] > 0)).sum())
    if nonempty_neg:
        problems.append("negative rows with non-empty labels: %d" % nonempty_neg)
    eval_paths, eval_hashes, eval_names = _collect_eval_index(evaluation_paths)
    eval_hits = []
    if not train_df.empty:
        tpaths = set(train_df["image"].astype(str))
        tnames = {p.replace("\\", "/").split("/")[-1] for p in tpaths}
        thashes = set(train_df["sha256"].astype(str))
        eval_hits = sorted((tpaths & eval_paths) | ((tnames & eval_names) if eval_names else set()))
        hash_hits = thashes & eval_hashes
    else:
        hash_hits = set()
    if eval_hits:
        problems.append("training images also appear in an evaluation manifest: %d (e.g. %s)"
                        % (len(eval_hits), eval_hits[:3]))
    if hash_hits:
        problems.append("training images byte-identical to evaluation images: %d" % len(hash_hits))
    total = len(train_df) + len(val_df)
    frac = (len(val_df) / total) if total else 0.0
    report = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "train_images": int(len(train_df)), "val_images": int(len(val_df)),
        "train_negatives": int(train_df["is_negative"].sum()) if "is_negative" in train_df.columns else 0,
        "negative_images": int(train_df["is_negative"].sum()) if "is_negative" in train_df.columns else 0,
        "location_overlap": loc_overlap, "sha256_overlap": hash_overlap,
        "eval_overlap_images": len(eval_hits), "eval_hash_overlap_images": len(hash_hits),
        "nonempty_negative_labels": nonempty_neg,
        "project_synthetic_rows": int(sum(1 for s in set(train_df.get("source", [])) if s in FORBIDDEN_SOURCES)),
        "val_positive_fraction": frac,
        "train_locations": sorted(set(train_df.get("location", []))),
        "val_locations": sorted(set(val_df.get("location", []))),
        "train_location_counts": train_df.groupby("location").size().to_dict() if not train_df.empty else {},
        "val_location_counts": val_df.groupby("location").size().to_dict() if not val_df.empty else {},
    }
    if problems:
        raise DatasetAuditError("; ".join(problems))
    return report


def write_ultralytics_dataset_yaml(train_df: pd.DataFrame, val_df: pd.DataFrame, output, root=None) -> None:
    import yaml
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.stem
    train_list = output.parent / (stem + "_train.txt")
    val_list = output.parent / (stem + "_val.txt")
    train_list.write_text(chr(10).join(train_df["image"].astype(str)) + chr(10), encoding="utf-8")
    val_list.write_text(chr(10).join(val_df["image"].astype(str)) + chr(10), encoding="utf-8")
    doc = {"path": str(root) if root else str(output.parent),
           "train": str(train_list) if root is None else train_list.name,
           "val": str(val_list) if root is None else val_list.name,
           "nc": 1, "names": {0: "shuttlecock"}}
    output.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    boxes = df[~df["is_negative"]] if "is_negative" in df.columns else df
    for lab in BUCKET_LABELS:
        sub = boxes[boxes["equiv_size_640"].apply(bucket_of) == lab]
        rows.append({"bucket": lab, "images": int(len(sub)),
                     "median_equiv_size_640": float(sub["equiv_size_640"].median()) if len(sub) else None,
                     "mean_equiv_size_640": float(sub["equiv_size_640"].mean()) if len(sub) else None,
                     "median_width_px": float(sub["width"].median()) if len(sub) else None,
                     "median_height_px": float(sub["height"].median()) if len(sub) else None})
    return pd.DataFrame(rows)


def build(eth_data, recipe_path, exclude_eval, train_out, val_out, dataset_yaml, audit_out, size_out,
          dataset_root=None, target_val_fraction=0.18, val_locations=None, negative_fraction=None, seed=42,
          only_sets=None):
    print("[build] inventory %s" % eth_data, flush=True)
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    inv = inventory_eth_dataset(eth_data, recipe, only_sets=only_sets)
    print("[build] inventory rows=%d" % len(inv), flush=True)
    cand = inv[inv["role"].isin(["official_positives", "official_negatives"])].copy()
    if val_locations:
        wanted = [v.strip() for v in str(val_locations).split(",") if v.strip()]
        missing = [w for w in wanted if w not in set(cand["location"])]
        if missing:
            raise DatasetAuditError("pinned val locations not available after exclusions: %s" % missing)
        val_pool = cand[cand["location"].isin(wanted)]
        positives = val_pool[val_pool["role"] == "official_positives"]
        frac = len(positives) / max(1, int((cand["role"] == "official_positives").sum()))
        if not (0.14 <= frac <= 0.21):
            raise DatasetAuditError("pinned val fraction %.4f outside [0.14, 0.21]" % frac)
        val_locs = wanted
    else:
        pos_pool = cand[cand["role"] == "official_positives"]
        _tr, val_pos = choose_location_split(pos_pool, target_val_fraction)
        val_locs = sorted(set(val_pos["location"]))
    cand, dropped = filter_eval_overlap(cand, exclude_eval)
    if (dropped["role"] == "official_positives").any():
        pass
    neg = cand[cand["role"] == "official_negatives"].copy()
    if negative_fraction is not None and 0.0 < float(negative_fraction) < 1.0:
        n_keep = int(round(len(neg) * float(negative_fraction)))
        neg = neg.sort_values("image").head(n_keep)
    keep = cand[cand["location"].isin(val_locs)] | neg
    train_df = cand[~cand["location"].isin(val_locs)].copy()
    train_df = train_df[train_df["role"] == "official_positives"].copy()
    train_df = pd.concat([train_df, neg], ignore_index=True)
    val_df = cand[cand["location"].isin(val_locs)].copy()
    excluded_eval = inv[~inv.index.isin(cand.index)].copy()
    report = audit_split(train_df, val_df, exclude_eval)
    report.update({
        "eth_data": str(eth_data), "recipe": str(recipe_path),
        "inventory_images": int(len(inv)),
        "candidate_positives_before_exclusion": int((inv["role"] == "official_positives").sum()),
        "negatives_used": int(len(neg)), "negative_fraction": negative_fraction,
        "excluded_eval_images": int(len(dropped)),
        "excluded_eval_positives": int((dropped["role"] == "official_positives").sum()),
        "excluded_eval_by_location": dropped.groupby("location").size().to_dict() if not dropped.empty else {},
        "excluded_non_training_images": int(len(excluded_eval)),
        "excluded_by_role": excluded_eval.groupby("role").size().to_dict() if not excluded_eval.empty else {},
        "train_images_total": int(len(train_df)),
        "val_locations": sorted(val_locs),
        "target_val_fraction": target_val_fraction,
        "seed": seed,
    })
    rep = rec = None
    if train_out:
        Path(train_out).parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_out, index=False)
    if val_out:
        Path(val_out).parent.mkdir(parents=True, exist_ok=True)
        val_df.to_csv(val_out, index=False)
    if dataset_yaml:
        write_ultralytics_dataset_yaml(train_df, val_df, dataset_yaml, root=dataset_root)
    if audit_out:
        Path(audit_out).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    if size_out:
        Path(size_out).parent.mkdir(parents=True, exist_ok=True)
        size_distribution(train_df).to_csv(size_out, index=False)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the audited ETH-only V1 dataset (Task 2).")
    ap.add_argument("--eth-data", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--exclude-eval", nargs="*", default=[])
    ap.add_argument("--train-out", default=None)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--dataset-yaml", default=None)
    ap.add_argument("--audit-out", default=None)
    ap.add_argument("--size-out", default=None)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--target-val-fraction", type=float, default=0.18)
    ap.add_argument("--val-locations", default=None)
    ap.add_argument("--negative-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-sets", default=None, help="debug: comma list of dataset subdir names")
    a = ap.parse_args()
    try:
        only = [s.strip() for s in (a.only_sets or "").split(",") if s.strip()] or None
        rep = build(a.eth_data, a.recipe, a.exclude_eval, a.train_out, a.val_out, a.dataset_yaml,
                    a.audit_out, a.size_out, dataset_root=a.dataset_root,
                    target_val_fraction=a.target_val_fraction, val_locations=a.val_locations,
                    negative_fraction=a.negative_fraction, seed=a.seed, only_sets=only)
    except DatasetAuditError as exc:
        print("DATASET_AUDIT_FAIL: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps({k: rep[k] for k in ["status", "train_images", "val_images", "negatives_used",
                                          "val_locations", "val_positive_fraction", "eval_overlap_images",
                                          "location_overlap", "nonempty_negative_labels"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
