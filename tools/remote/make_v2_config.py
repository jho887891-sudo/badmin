#!/usr/bin/env python3
"""Generate configs/eth_real_hardneg_yolo26s_v2.yaml from the V1 contract (line-based copy, no retyping).

Only three edits touch the V1 text: the experiment id, the spec path, and an appended V2 section. Every frozen
training/evaluation value therefore stays character-identical to V1, which the contract tests verify again on
the resolved kwargs.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "configs" / "eth_only_yolo26s_1024_v1.yaml"
V2 = ROOT / "configs" / "eth_real_hardneg_yolo26s_v2.yaml"

EXTRA = '''
# --- V2 data policy (rulings 2026-10-05) ---
# 计划里的 hardneg_train / hardneg2_train 无法核验（hardneg2_train 不存在；hardneg_train 86 行里 53 张
# 本地与远端都缺，见 management/shuttle_detection/DATASET_RESOLUTION.md 第 4/7 节）→ 按裁决 1 改用
# 两个已核验池（本地与远端 sha256 96/96 一致），并剔除 excluded.txt 里的 7 个名字 = 89 张唯一负样本。
include_synthetic: false
include_iphone: false
include_d455_positive: false
include_roboflow: false
adds_positives: false
hard_negative_sources: [hard_negatives, hard_negatives2]
hard_negative_root: outputs/shuttle_capability       # local copy (resource SSOT: /home/T7/dgut/robot_sim/...)
hard_negative_remote_root: /home/T7/dgut/robot_sim/outputs/shuttle_capability
hard_negative_exclude_list: outputs/shuttle_capability/hard_negatives2/excluded.txt
hard_negative_expected_unique: 89
hard_negative_repeat: 8            # ruling 2: repeat=1 gives only +0.61% exposure, endpoint would be underpowered
hard_negative_oversample: false    # no other oversampling / source weighting is allowed
hard_negative_source_weights: null
v1_train_manifest: data/eth_only_v1_train_manifest.csv
v1_val_manifest: data/eth_only_v1_val_manifest.csv
training_launcher: tools/train_eth_only_v1.py       # ruling 4: reuse, do not duplicate
evaluator: tools/eval_yolo26_v1.py

# --- primary endpoint (ruling 3) ---
primary_endpoint: pooled_real_no_target_fp_per_image
primary_endpoint_images: 496        # real_images/backgrounds 30 + real_images/raw 59 + real_video/frames 150
                                    # + real_match_frames/images 235 + real_train/raw 22
primary_endpoint_sets: [real_images/backgrounds, real_images/raw, real_video/frames, real_match_frames/images, real_train/raw]
primary_endpoint_requires_ci: true  # raw FP counts + 95% CI on the pooled rate, per-set breakdown kept
secondary_endpoint: val|eth_unseen recall / mAP50-95 / Recall_<8

# --- rulings (frozen 2026-10-05) ---
rulings:
  - id: 1
    text: "replace the nonexistent/incomplete historical source names with the two verified real pools hard_negatives/raw and hard_negatives2/raw; exclude the 7 entries in excluded.txt; freeze 89 unique SHA256s"
  - id: 2
    text: "hard_negative_repeat 1 -> 8 because repeat=1 yields only 0.61% training exposure and makes the planned endpoint underpowered; no other oversampling or source weighting"
  - id: 3
    text: "evaluate the pooled 496-image real no-target FP/image with raw FP counts and 95% confidence intervals while retaining per-set breakdowns"
  - id: 4
    text: "reuse the repository train_eth_only_v1.py CLI and checkpoint-selection implementation instead of creating duplicate launchers"
'''


def main() -> int:
    text = V1.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        if line.startswith("experiment:"):
            line = "experiment: eth_real_hardneg_yolo26s_v2"
        elif line.startswith("spec:"):
            line = "spec: docs/superpowers/specs/2026-10-05-eth-real-hardneg-yolo26s-v2-design.md"
        out.append(line)
    body = chr(10).join(out).rstrip() + chr(10) + EXTRA
    V2.write_text(body, encoding="utf-8")
    print("wrote %s (%d lines)" % (V2, len(body.splitlines())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
