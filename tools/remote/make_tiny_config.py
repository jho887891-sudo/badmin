#!/usr/bin/env python3
"""Generate configs/tiny_recovery_yolo26s_v1.yaml from the frozen V2 contract (line-based copy).

Only the experiment id, the spec path and an appended tiny-recovery block differ, so every frozen training and
evaluation value stays character-identical to V2 (the contract tests verify that on the resolved kwargs).
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "configs" / "eth_real_hardneg_yolo26s_v2.yaml"
OUT = ROOT / "configs" / "tiny_recovery_yolo26s_v1.yaml"

EXTRA = """
# --- tiny-object recovery V1: the only allowed change is the exposure of existing real tiny positives ---
# Gate verified before writing this file: the V2 training positives contain 5,965 unique images with
# equiv_size_640 < 8 (161 <4 / 1,282 4-6 / 4,522 6-8), all from real ETH frames and free of synthetic /
# iPhone / D455 / Roboflow sources, so the 712-exposure budget is reachable with at most one extra copy per
# unique image (the design forbids raising the per-image repeat if it were not).
tiny_positive_max_equiv_size_640: 8.0
tiny_positive_extra_exposure: 712
tiny_extra_per_unique_image_max: 1
tiny_selection_seed: 42
tiny_stratify_buckets: ["<4", "4-6", "6-8"]
tiny_require_exact_budget: true
tiny_stop_if_insufficient: true
tiny_source_manifest: data/eth_real_hardneg_v2_train_manifest.csv
tiny_hard_negative_contract_frozen: true      # 89 unique x repeat 8, pool and exclusion list unchanged

# --- endpoints and guards (design 2026-10-05, pre-registered) ---
primary_endpoint: val|eth_unseen_recall_lt8
primary_min_tp_lt8: 8
strong_tp_lt8: 12
guard_recall_min: 0.1840
guard_no_target_fp_max: 3
guard_map5095_delta_min: -0.01
guard_recall_8_16_delta_min: -0.02
no_target_pool_images: 496
v2_reference: {recall: 0.2040, map5095: 0.189727, tp_lt8: 4, fp_no_target: 1, recall_lt8: 0.0381}
v1_reference: {tp_lt8: 8, recall_lt8: 0.0762, fp_no_target: 6}
decisions: [USEFUL, STRONG_SUCCESS, NO_TINY_GAIN, FP_REGRESSION, SIZE_TRADEOFF, HARMFUL]
"""


def main() -> int:
    text = V2.read_text(encoding="utf-8")
    out = []
    renames = {
        # V2 endpoint keys stay as reference only: this experiment has a different primary endpoint.
        "primary_endpoint:": "v2_primary_endpoint:",
        "primary_endpoint_sets:": "no_target_pool_sets:",
        "primary_endpoint_requires_ci:": "no_target_pool_requires_ci:",
        "secondary_endpoint:": "v2_secondary_endpoint:",
        "primary_endpoint_images:": "v2_primary_endpoint_images:",
    }
    for line in text.splitlines():
        if line.startswith("experiment:"):
            line = "experiment: eth_hardneg_tiny_recovery_v1"
        elif line.startswith("spec:"):
            line = "spec: docs/superpowers/specs/2026-10-05-tiny-object-recovery-v1-design.md"
        else:
            for old, new in renames.items():
                if line.startswith(old):
                    line = new + line[len(old):]
                    break
        out.append(line)
    body = chr(10).join(out).rstrip() + chr(10) + EXTRA
    OUT.write_text(body, encoding="utf-8")
    print("wrote %s (%d lines)" % (OUT, len(body.splitlines())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
