#!/usr/bin/env python3
"""Build the audited ETH real-hard-negative V2 manifests, dataset yaml and blocking data audit (Task 2).

V2 differs from V1 only by ADDING verified real hard negatives (repeat factor from the contract, ruling 2).
Every guard on the causal contrast is therefore an audit: the positive rows and their SHA256 set must equal V1
exactly; the val manifest is a byte copy of the V1 one; each added image must come from an explicitly frozen
pool with empty labels and complete provenance; nothing may overlap an evaluation manifest or a canonical
frozen set (enumerated through tools/eval_yolo26_v1.py, so no second listing implementation exists); duplicate
paths/hashes and any repeat/weight column in the candidate pool are blocking errors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_yolo26_v1 as ev  # noqa: E402  (canonical frozen-set enumeration)
import train_eth_only_v1 as T  # noqa: E402  (canonical contract loader)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PROHIBITED_SOURCE_HINTS = ("synthetic", "isaac", "iphone", "d455", "roboflow", "depth", "nearfield", "near_field")
PROHIBITED_COLUMNS = ("repeat", "weight", "sample_weight", "oversample", "dup", "copies")
V1_MANIFEST_COLUMNS = ("image", "label", "location", "difficulty", "split", "set_dir", "role",
                       "in_official_training", "source", "is_negative", "n_boxes", "width", "height",
                       "sha256", "equiv_size_640", "median_equiv_size_640", "min_equiv_size_640",
                       "equiv_sizes")


class V2DatasetAuditError(RuntimeError):
    """Raised for any blocking V2 dataset, provenance or leakage violation."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_set_digest(hashes) -> str:
    """Stable digest of a SHA256 set: proves membership equality without shipping a huge list."""
    joined = chr(10).join(sorted(set(hashes)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_v1_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [c for c in V1_MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise V2DatasetAuditError("V1 manifest %s misses columns: %s" % (path, missing))
    df["is_negative"] = df["is_negative"].astype(str).str.strip().str.lower().isin(["true", "1"])
    df["is_positive"] = ~df["is_negative"]
    if df["sha256"].duplicated().any():
        raise V2DatasetAuditError("V1 manifest %s contains duplicate SHA256 rows" % path)
    return df


def _image_size(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def freeze_hard_negative_candidates(cfg: dict, repo: Path, out_path) -> pd.DataFrame:
    """Freeze the verified real pools into a candidate manifest with SHA256, byte size and provenance."""
    root = repo / str(cfg["hard_negative_root"])
    excluded = set()
    excl_path = repo / str(cfg["hard_negative_exclude_list"])
    if excl_path.is_file():
        excluded = set(x.strip() for x in excl_path.read_text(encoding="utf-8").split() if x.strip())
    rows = []
    for pool in list(cfg["hard_negative_sources"]):
        pool_dir = root / pool / "raw"
        if not pool_dir.is_dir():
            raise V2DatasetAuditError("hard-negative pool directory missing: %s" % pool_dir)
        prov = {}
        meta = root / pool / "hard_negative_metadata.csv"
        if meta.is_file():
            for r in csv.DictReader(meta.open(encoding="utf-8")):
                prov[str(r.get("file", "")).strip()] = r
        for img in sorted(p for p in pool_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
            if img.name in excluded:
                continue
            label = img.with_suffix(".txt")
            label_text = label.read_text(encoding="utf-8", errors="replace") if label.is_file() else ""
            meta_row = prov.get(img.name, {})
            try:
                w, h = _image_size(img)
            except Exception:
                w = h = 0
            rows.append({
                "source": pool,
                "image": "%s/%s/raw/%s" % (str(cfg["hard_negative_remote_root"]).rstrip("/"), pool, img.name),
                "local_path": str(img),
                "file": img.name,
                "sha256": sha256_file(img),
                "bytes": img.stat().st_size,
                "width": w,
                "height": h,
                "label_nonempty": bool(label_text.strip()),
                "is_negative": True,
                "provenance_title": meta_row.get("title", ""),
                "provenance_license": meta_row.get("license", ""),
                "provenance_url": meta_row.get("url", ""),
                "provenance_present": bool(meta_row),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise V2DatasetAuditError("no candidate hard negatives found under %s" % root)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df


def inventory_hard_negatives(candidate_manifest, allowed_sources) -> pd.DataFrame:
    """Strict reader for the frozen candidate manifest: rejects prohibited sources and weighting columns."""
    df = pd.read_csv(candidate_manifest, keep_default_na=False, dtype=str)
    required = {"source", "image", "sha256", "is_negative", "label_nonempty", "provenance_present"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise V2DatasetAuditError("candidate manifest misses columns: %s" % missing)
    bad_columns = [col for col in df.columns if col.strip().lower() in PROHIBITED_COLUMNS]
    if bad_columns:
        raise V2DatasetAuditError("candidate manifest carries weighting columns: %s" % bad_columns)
    allowed = [str(s) for s in allowed_sources]
    source = df["source"].astype(str)
    not_allowed = sorted(set(source) - set(allowed))
    if not_allowed:
        raise V2DatasetAuditError("candidate sources outside the frozen list: %s" % not_allowed)
    prohibited = source.str.lower().apply(lambda s: any(p in s for p in PROHIBITED_SOURCE_HINTS))
    if prohibited.any():
        raise V2DatasetAuditError("prohibited sources present: %s" % sorted(set(source[prohibited])))
    if not df["is_negative"].astype(str).str.lower().isin(["true", "1"]).all():
        raise V2DatasetAuditError("candidate rows must all be negatives")
    if df["label_nonempty"].astype(str).str.lower().isin(["true", "1"]).any():
        raise V2DatasetAuditError("hard negatives must have empty labels")
    if not df["provenance_present"].astype(str).str.lower().isin(["true", "1"]).all():
        raise V2DatasetAuditError("hard negatives without provenance metadata must be dropped, not trained on")
    return df.reset_index(drop=True)


def dedupe_hard_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly one row per unique path and per unique SHA256; the repeat factor is applied later, once."""
    if df.empty:
        return df.reset_index(drop=True)
    out = df.sort_values("image").drop_duplicates(subset=["sha256"], keep="first")
    out = out.drop_duplicates(subset=["image"], keep="first")
    return out.reset_index(drop=True)


def build_v2_train_manifest(v1_train: pd.DataFrame, hardneg: pd.DataFrame, repeat: int) -> pd.DataFrame:
    """V1 rows untouched plus the configured number of copies of every unique hard negative (ruling 2)."""
    if int(repeat) < 1:
        raise V2DatasetAuditError("repeat must be >= 1, got %r" % repeat)
    base = v1_train.copy()
    base["repeat_index"] = 0
    frames = [base]
    for idx in range(int(repeat)):
        if hardneg.empty:
            break
        block = hardneg.copy()
        block["repeat_index"] = idx
        block["location"] = "hard_negative"
        block["difficulty"] = "n/a"
        block["split"] = "train"
        block["set_dir"] = block["source"]
        block["role"] = "hard_negative"
        block["in_official_training"] = "False"
        block["source"] = "hard_negative"
        block["is_negative"] = True
        block["is_positive"] = False
        block["n_boxes"] = 0
        block["label"] = ""
        block["equiv_size_640"] = 0.0
        block["median_equiv_size_640"] = 0.0
        block["min_equiv_size_640"] = 0.0
        block["equiv_sizes"] = ""
        frames.append(block)
    return pd.concat(frames, ignore_index=True, sort=False)


def collect_eval_images(repo: Path, eval_manifests, include_frozen_sets: bool = True):
    """Paths and SHA256s of every evaluation image: our manifests plus the canonical frozen sets."""
    paths, shas = set(), set()
    for man in eval_manifests or []:
        man = Path(man)
        if not man.is_file():
            raise V2DatasetAuditError("evaluation manifest not found: %s" % man)
        for r in csv.DictReader(man.open(encoding="utf-8")):
            p = r.get("image") or r.get("path") or ""
            if p:
                paths.add(str(p).replace(chr(92), "/"))
            if r.get("sha256"):
                shas.add(str(r["sha256"]).lower())
    if include_frozen_sets:
        for group in ev.discover_frozen_groups(repo / "outputs" / "shuttle_capability"):
            for img in group.images:
                paths.add(str(img).replace(chr(92), "/"))
                try:
                    shas.add(sha256_file(img))
                except OSError:
                    pass
    return paths, shas


def audit_v2_dataset(v1_train, v1_val, hardneg, eval_paths, eval_shas, repeat,
                     v1_val_manifest=None, v2_val_manifest=None) -> dict:
    """Blocking audit: raises V2DatasetAuditError and still returns the full field set otherwise."""
    v1_pos = v1_train[v1_train["is_positive"]]
    raw_count = int(len(hardneg))
    unique = dedupe_hard_negatives(hardneg)
    v2 = build_v2_train_manifest(v1_train, unique, repeat)
    v2_pos = v2[v2["is_positive"]]
    hn_rows = v2[v2["role"] == "hard_negative"]          # the only negatives V2 adds
    neg_effective = v2[(~v2["is_positive"]) & (v2["role"] != "hard_negative")]   # V1's own negatives

    problems = []
    if set(v2_pos["sha256"]) != set(v1_pos["sha256"]):
        problems.append("positive SHA256 set drifted from V1")
    if int(len(v2_pos)) != int(len(v1_pos)):
        problems.append("positive row count drifted from V1")
    if raw_count != len(unique):
        problems.append("candidate pool contains duplicate paths or SHA256s")
    hn_paths = set(unique["image"].astype(str).str.replace(chr(92), "/")) if len(unique) else set()
    hn_shas = set(unique["sha256"].astype(str).str.lower()) if len(unique) else set()
    path_overlap = len(hn_paths & set(eval_paths))
    sha_overlap = len(hn_shas & set(eval_shas))
    if path_overlap:
        problems.append("hard-negative path overlap with evaluation: %d" % path_overlap)
    if sha_overlap:
        problems.append("hard-negative SHA256 overlap with evaluation: %d" % sha_overlap)
    train_locs = set(v2[v2["is_positive"]]["location"])
    val_locs = set(v1_val["location"])
    loc_overlap = len(train_locs & val_locs)
    if loc_overlap:
        problems.append("train/val location overlap: %d" % loc_overlap)
    if len(unique) and unique["label_nonempty"].astype(str).str.lower().isin(["true", "1"]).any():
        problems.append("hard negatives with non-empty labels")
    if len(hn_rows) != len(unique) * int(repeat):
        problems.append("unexpected hard-negative exposure: %d rows for %d unique x %s"
                        % (len(hn_rows), len(unique), repeat))

    report = {
        "status": "FAIL" if problems else "PASS",
        "problems": problems,
        "v1_positive_count": int(len(v1_pos)),
        "v1_positive_hash_set_sha256": hash_set_digest(v1_pos["sha256"]),
        "v2_positive_count": int(len(v2_pos)),
        "v2_positive_hash_set_sha256": hash_set_digest(v2_pos["sha256"]),
        "hardneg_raw_count": raw_count,
        "hardneg_unique_count": int(len(unique)),
        "hardneg_by_source": ({str(k): int(v) for k, v in unique["source"].value_counts().items()}
                              if len(unique) else {}),
        "hardneg_nonempty_label_count": int(unique["label_nonempty"].astype(str).str.lower()
                                            .isin(["true", "1"]).sum()) if len(unique) else 0,
        "hardneg_unreadable_count": int((unique[["width", "height"]].astype(int) <= 0).any(axis=1).sum())
                                    if len(unique) else 0,
        "hardneg_provenance_missing_count": int((~unique["provenance_present"].astype(str).str.lower()
                                                 .isin(["true", "1"])).sum()) if len(unique) else 0,
        "duplicate_path_count": int(raw_count - hardneg["image"].nunique()),
        "duplicate_sha256_count": int(raw_count - hardneg["sha256"].str.lower().nunique()),
        "eval_path_overlap_count": path_overlap,
        "eval_sha256_overlap_count": sha_overlap,
        "train_val_location_overlap_count": loc_overlap,
        "prohibited_source_counts": {},
        "hard_negative_repeat": int(repeat),
        "train_rows": int(len(v2)),
        "negative_exposure_rows": int(len(hn_rows)),
        "negative_exposure_fraction": float(len(hn_rows)) / float(len(v2)),
        "v1_negative_rows_retained": int(len(neg_effective)),
        "val_rows": int(len(v1_val)),
        "v1_val_manifest_sha256": (sha256_file(v1_val_manifest) if v1_val_manifest else None),
        "v2_val_manifest_sha256": (sha256_file(v2_val_manifest) if v2_val_manifest else None),
        "eval_reference_images": len(set(eval_paths)),
        "eval_reference_sha256": len(set(eval_shas)),
    }
    if problems:
        raise V2DatasetAuditError("; ".join(problems))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the leakage-free ETH real-hard-negative V2 dataset")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--v1-train", default=None)
    ap.add_argument("--v1-val", default=None)
    ap.add_argument("--candidates-out", default=None)
    ap.add_argument("--candidate-manifest", default=None)
    ap.add_argument("--eval-manifest", action="append", default=[])
    ap.add_argument("--no-frozen-sets", action="store_true")
    ap.add_argument("--train-out", default=None)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--yaml-out", default=None)
    ap.add_argument("--audit-out", default=None)
    a = ap.parse_args(argv)
    repo = Path(a.repo).resolve() if a.repo else Path(__file__).resolve().parents[1]
    cfg = T.load_contract(repo / a.config)
    v1_train_path = repo / (a.v1_train or cfg["v1_train_manifest"])
    v1_val_path = repo / (a.v1_val or cfg["v1_val_manifest"])
    v1_train = load_v1_manifest(v1_train_path)
    v1_val = load_v1_manifest(v1_val_path)

    repeat = int(cfg["hard_negative_repeat"])
    if a.candidate_manifest:
        hardneg = inventory_hard_negatives(repo / a.candidate_manifest, cfg["hard_negative_sources"])
        candidates_path = repo / a.candidate_manifest
    else:
        candidates_path = repo / (a.candidates_out or "outputs/shuttle_capability/metrics/"
                                  "eth_real_hardneg_v2_hardneg_candidates.csv")
        freeze_hard_negative_candidates(cfg, repo, candidates_path)
        hardneg = inventory_hard_negatives(candidates_path, cfg["hard_negative_sources"])
    raw = hardneg.copy()
    hardneg = dedupe_hard_negatives(hardneg)

    eval_manifests = [str(v1_val_path)] + [str(m) for m in a.eval_manifest]
    eval_paths, eval_shas = collect_eval_images(repo, eval_manifests, include_frozen_sets=not a.no_frozen_sets)

    train_out = repo / a.train_out
    val_out = repo / a.val_out
    yaml_out = repo / a.yaml_out
    audit_out = repo / a.audit_out
    for p in (train_out.parent, val_out.parent, yaml_out.parent, audit_out.parent):
        p.mkdir(parents=True, exist_ok=True)
    val_out.write_bytes(v1_val_path.read_bytes())
    report = audit_v2_dataset(v1_train, v1_val, raw, eval_paths, eval_shas, repeat,
                              v1_val_manifest=v1_val_path, v2_val_manifest=val_out)
    v2 = build_v2_train_manifest(v1_train, hardneg, repeat)
    v2.to_csv(train_out, index=False)

    stem = train_out.name.replace("_manifest.csv", "")
    train_list = train_out.with_name(stem + "_train.txt")
    val_list = train_out.with_name(stem.replace("_train", "_val") + ".txt")
    train_list.write_text(chr(10).join(v2["image"].astype(str)) + chr(10), encoding="utf-8")
    val_list.write_text(chr(10).join(v1_val["image"].astype(str)) + chr(10), encoding="utf-8")
    yaml_out.write_text(chr(10).join([
        "path: data/" + Path(a.train_out).name.replace("_train_manifest.csv", ""),
        "train: " + train_list.name,
        "val: " + val_list.name,
        "nc: 1",
        "names:",
        "  0: shuttlecock",
    ]) + chr(10), encoding="utf-8")

    report["candidate_manifest"] = str(candidates_path.relative_to(repo)).replace(chr(92), "/")
    report["candidate_manifest_sha256"] = sha256_file(candidates_path)
    report["train_manifest_sha256"] = sha256_file(train_out)
    report["train_list_sha256"] = sha256_file(train_list)
    report["val_list_sha256"] = sha256_file(val_list)
    report["yaml_sha256"] = sha256_file(yaml_out)
    report["excluded_names_count"] = len([x for x in
                                          (repo / str(cfg["hard_negative_exclude_list"]))
                                          .read_text(encoding="utf-8").split() if x.strip()])
    audit_out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[v2] status=%s train=%d val=%d hardneg=%d x%d exposure=%.4f"
          % (report["status"], report["train_rows"], report["val_rows"],
             report["hardneg_unique_count"], repeat, report["negative_exposure_fraction"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




