# Tiny-object Recovery V1 - repository spec

Authoritative source: the design supplied 2026-10-05 (`2026-10-05-tiny-object-recovery-v1-design.md`, 840 lines,
READ-ONLY attachment). This file records the same contract in repository form plus the gate checks that were run
before any artifact was produced. Experiment id: `ETH_HARDNEG_TINY_RECOVERY_V1`.

## Question this experiment may answer
With the V2 hard-negative policy frozen, does a **budgeted increase of the training exposure of existing real
`equiv_size_640 < 8 px` positives** recover `Recall_<8` while keeping V2's low false-positive rate?

It may not answer anything about synthetic / iPhone / D455 / new locations / releasing ml_3 or ml_6 / P2 / larger
imgsz / YOLOv8s / loss reweighting (design section 24).

## Frozen base: hard-negative V2
```text
best_epoch 21, best_checkpoint sha256 3c8339c6...9ff7
val|eth_unseen: Recall 0.2040, mAP50-95 0.189727, Precision 0.8783, FP 14, TP_<8 4/105, Recall_<8 0.0381
frozen real no-target pool: 496 images, V2 = 1 FP (V1 = 6)
```

## The only allowed change
```text
tiny_positive_extra_exposure = 712          # symmetric counterweight to the 89 x 8 = 712 hard-negative exposure
selection: existing V2 real ETH training positives only, equiv_size_640 < 8, seed 42,
           stratified by the original <4 / 4-6 / 6-8 proportions, at most ONE extra copy per unique image,
           exact budget, no new image / location / label / domain
hard negatives: pool, exclusion list and repeat=8 completely frozen
model + recipe: plain YOLO26s, imgsz 1024, 50 epochs, AdamW 1e-4, freeze 0, batch 8, nbs 32, seed 42,
                identical loss / augmentation / scheduler / memory cap / internal val / selection rule / evaluator
```

## Gate verification before building (design section 5)
| gate | measured | verdict |
|---|---|---|
| eligible unique tiny positives in the V2 training set | **5,965** (161 <4, 1,282 4-6, 4,522 6-8) | >= 712, so the budget is reachable |
| one extra copy per unique image possible | 5,965 unique SHA256s | possible |
| prohibited sources inside the tiny pool | none | clean |
| stratified allocation of 712 | `{<4: 19, 4-6: 153, 6-8: 540}` | sums to exactly 712 (largest remainder) |

## Hard gates enforced by the builder (design section 10)
Unique positive set identical to V2 (digest equality); internal val byte-identical; evaluation path and SHA256
overlap zero (val|eth_unseen, frozen real no-target, controlled, challenge via the canonical frozen-set
enumeration); `extra_tiny_exposure == 712`; `extra_non_tiny_positive == 0`;
`tiny_extra_per_unique_image <= 1`; prohibited sources zero. Any failure is STOP (the design forbids funding a
short budget by raising a per-image repeat).

## Endpoints and guards (design sections 13-19, pre-registered)
Primary: `TP_<8 >= 8` on `val|eth_unseen` (recover V1's 0.0762); `STRONG_SUCCESS` at `TP_<8 >= 12`.
Guards: overall Recall `>= 0.1840`; frozen no-target `FP <= 3/496`; `delta mAP50-95 > -0.01`;
`delta Recall_8_16 >= -0.02` (else `SIZE_TRADEOFF`).
Decision labels: `USEFUL`, `STRONG_SUCCESS`, `NO_TINY_GAIN`, `FP_REGRESSION`, `SIZE_TRADEOFF`, `HARMFUL`.

## Known confounder (must appear in the final report)
The manifest grows from 15,255 to 15,967 rows (+4.67% relative to V2, +4.46% of the new total), so sample exposure
and optimiser updates also grow slightly. The conclusion must be phrased as the effect of a budgeted
tiny-positive exposure intervention, not as the isolated effect of size resampling.

## Artifacts
Pre-training: `tiny_recovery_v1_data_audit.json`, `tiny_recovery_v1_source_counts.csv`,
`tiny_recovery_v1_tiny_exposure.csv` (per-image: path, sha256, location, equiv_size_640, size_bucket,
base_exposure, extra_exposure, final_exposure, selection_seed) and the train/val manifests + yaml + lists.
Training: `tiny_recovery_v1_smoke.json` (diagnostic_only), `tiny_recovery_v1_full_e50_manifest.json`,
`tiny_recovery_v1_full_e50_results.csv`.
Evaluation: `tiny_recovery_v1_vs_v2.csv`, `tiny_recovery_v1_size_buckets.csv`,
`tiny_recovery_v1_no_target_fp.csv`, `tiny_recovery_v1_decision.json`, `tiny_recovery_v1_report.md`.

## Execution order
1. contract + gate verification (done: 8 contract tests, 16 builder tests)
2. dataset build + audit PASS (done: 15,967 rows, 712 extra, allocation as above)
3. smoke (3 epochs) then full 50 epochs with the frozen recipe
4. canonical evaluation (val|eth_unseen primary, 496 no-target guard, controlled/challenge diagnostic)
5. comparison and decision, then the report with raw counts and CIs (never percentages alone)
