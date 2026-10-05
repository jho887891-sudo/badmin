# ETH Real Hard-Negative YOLO26s V2 - decision report

> **STATUS: SKELETON (2026-10-05).** Every numeric cell below is either measured already or marked
> `FILL(id)`. The frozen fillers are listed in the appendix and each one names its source artifact, so no number
> can be hand-copied without a source. The full 50-epoch V2 run started 2026-10-05T06:23:05Z.
>
> Interpretation scope is enforced by the plan: this report may conclude **only** whether verified real
> hard negatives helped FP behaviour without materially harming Recall under the frozen YOLO26s V1 recipe. It
> must not claim anything about synthetic vs real generalisation, iPhone, D455, YOLO26s vs YOLOv8s, or P2.

## 1. Contract and hashes

| item | value |
|---|---|
| experiment | `eth_real_hardneg_yolo26s_v2` (spec `docs/superpowers/specs/2026-10-05-eth-real-hardneg-yolo26s-v2-design.md`) |
| V2 contract | `configs/eth_real_hardneg_yolo26s_v2.yaml`, sha256 `FILL(contract_sha256)` |
| V1 contract (reference) | `configs/eth_only_yolo26s_1024_v1.yaml` |
| pretrained init | official `yolo26s.pt`, 20,422,725 B, sha256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` (verified at launch) |
| V2 train manifest | `data/eth_real_hardneg_v2_train_manifest.csv`, sha256 `d2bd887357555c9262f769ba81d3bfd49350e1670e80a94ecdc1d166898a8d4d`, 15,255 rows |
| V2 val manifest | `data/eth_real_hardneg_v2_val_manifest.csv`, sha256 `0ed405719af1d62763fb41fdad79b2581e0e7787da9a9c237c203a62ef735918` (byte-identical to V1) |
| V2 run | `full_v2_e50` on the remote A6000, started 2026-10-05T06:23:05Z, finished `FILL(run_finished)` |
| V2 checkpoint | `best.pt`, bytes `FILL(best_bytes)`, sha256 `FILL(best_sha256)`, epoch `FILL(best_epoch)` |
| V1 checkpoint (reference) | `best.pt`, 20,344,069 B, sha256 `7a836a2621686affbdd3f1da7c3a0a57c4b86f14432835be2bc562c2899fc6f2`, epoch 21 |
| evaluator | `tools/eval_yolo26_v1.py` (SSOT) through `tools/eval_eth_official_baseline.py`, parity pinned by `tests/test_eth_only_v1_eval_parity.py` |
| decision rule | `tools/compare_eth_hardneg_v2.py` (`compute_v2_decision`), thresholds FP/image -20%, Recall > -0.02, harm at <=-0.05, low value <10% |

## 2. Data composition

| block | rows | notes |
|---|---|---|
| V1 positives | 13,992 | exact V1 membership, SHA256 set digest `6f2b3453175c31b4d3bad996c68d35278482aa0ed2ced7f0b2e51dac33fe48bd` on both sides |
| V1 negatives (retained) | 551 | coco_train 550 + 1 empty-label positive-location frame |
| added hard negatives | 89 unique x 8 = 712 | ruling 1 pools `hard_negatives` 36 + `hard_negatives2` 53 (7 names excluded), ruling 2 repeat=8 |
| V2 train total | 15,255 | hard-negative exposure 4.67% |
| V2 val | 2,920 | byte-identical to the V1 internal val; selection only |

Hard-negative provenance: both pools carry `hard_negative_metadata.csv` (title/license/page/url), every image
has an empty label, and the local and remote copies agree on all 96 files by SHA256 (the 7 names in
`hard_negatives2/excluded.txt` are dropped, which is ISSUE-027).

## 3. Leakage audit

`outputs/shuttle_capability/metrics/eth_real_hardneg_v2_data_audit.json` -> **status `FILL(audit_status)`** with
`eval_path_overlap 0`, `eval_sha_overlap 0` (5,957 evaluation reference images), `train_val_location_overlap 0`,
`duplicate_path 0`, `duplicate_sha256 0`, `nonempty_label 0`, `unreadable 0`, `provenance_missing 0`.
V2 positives and the V2 val manifest are provably unchanged from V1 (hash-set digest equality, byte-identical val).

## 4. Pre-training V1 FP/FN diagnostic (read-only)

`outputs/shuttle_capability/metrics/eth_only_v1_error_diagnostic.{json,csv}` and `eth_only_v1_fn_analysis.{json,csv}`.

| quantity | V1 value |
|---|---|
| pooled real no-target FP (496 images @ conf 0.25) | **6** -> FP/image 0.012097, 95% CI [0.004439, 0.026330] |
| per set | backgrounds 2/30, raw 3/59, video 0/150, match_frames 0/235, real_train 1/22 |
| `val|eth_unseen` GT hit / missed | 81 / **414** of 495 (recall 0.1636) |
| FN rate by size bucket (gt/fn/rate) | <4 24/24/1.0000, 4-6 37/37/1.0000, 6-8 44/36/0.8182, 8-12 180/140/0.7778, 12-16 110/95/0.8636, 16-24 66/53/0.8030, 24-32 11/9/0.8182, 32-64 12/9/0.7500, >64 11/11/1.0000; **<8px aggregate 105/97/0.9238** |
| worst locations by FN rate | synthetic 120/117/0.9750, iphone_20251015 112/96/0.8571, ml_3 144/115/0.7986, ml_6 115/84/0.7304, uetlibergstrasse_1 4/2/0.5000 |
| FN by frame domain (source) | eth_main 263/201/0.7643, synthetic 120/117/0.9750, eth_iphone 112/96/0.8571 |
| FNs the ETH official model hits (recoverable) | **194 of 414** (46.9% of the V1 gap; 72.1% of ETH's 269 hits); only **6** GT go the other way; concentrated in ml_3 108/137 and ml_6 79/110 and in 8-16px boxes (8-12 101/141, 12-16 57/72); zero in <4 and >64 |
| FN-rate confidence | 0.83636, Wilson 95% CI [0.80121, 0.86634]; Poisson FN count [375.08, 455.87] |

Reading: V1's dominant real-domain failure is **missed detections**, not false alarms - it fires only 6 false
positives on 496 real no-target images. The planned primary endpoint therefore has a single-digit event count,
which is why ruling 3 requires the raw count, the interval and the per-set breakdown.

Two structural caveats that this analysis makes explicit and that later sections must respect:
1. `val|eth_unseen` is **not purely real ETH data**: by frame domain its 495 GT split into eth_main 263
   (same-location ETH frames), synthetic 120 and iPhone 112. Nearly half of the secondary endpoint therefore
   measures our own synthetic/iPhone domains, not the ETH real domain.
2. 187 of the 194 recoverable FNs sit in ml_3 and ml_6 - precisely the two ETH locations that the leakage-free
   protocol had to remove from training entirely because our evaluation frames live there (ISSUE-031). A large
   part of the recall gap is the **price of the protocol**, not a data-mixture or negative-sampling defect. The
   97 <8px FNs are essentially unrecoverable (the ETH official model misses them too), so they need data or
   scale, not a different head or more negatives.

## 5. Training and checkpoint selection

| item | value |
|---|---|
| recipe | imgsz 1024, AdamW, lr0 1e-4, nbs 32, batch 8 (fallback 8/6/4 unused), freeze 0, seed 42, V1 loss/augmentation |
| epochs | 50 (`FILL(epochs_recorded)` recorded) |
| selection | internal location-disjoint val mAP50-95 only: epoch `FILL(best_epoch)`, mAP50-95 `FILL(best_map5095)`, recall `FILL(best_recall)` |
| fitness-argmax equals mAP50-95 argmax | `FILL(fitness_check)` |
| memory | 24 GiB per-process cap (fraction 0.5063), peak VRAM `FILL(peak_vram)` |
| smoke evidence | `eth_real_hardneg_v2_smoke.json` status PASS, peak 6.78 GiB, batch 8 first try |

## 6. `val|eth_unseen` V1 vs V2 (secondary endpoint)

| model | images | GT | P | R | AP50 | mAP50-95 | Recall_<8 | FP/image |
|---|---|---|---|---|---|---|---|---|
| V1 `eth_only_v1_best` | 1,648 | 495 | 0.6864 | 0.1636 | 0.3510 | 0.1909 | 0.0762 | 0.0225 |
| V2 `eth_real_hardneg_v2_best` | `FILL(v2_unseen_images)` | 495 | `FILL(v2_unseen_p)` | `FILL(v2_unseen_r)` | `FILL(v2_unseen_ap50)` | `FILL(v2_unseen_map)` | `FILL(v2_unseen_lt8)` | `FILL(v2_unseen_fpimg)` |
| delta (V2 - V1) | | | | `FILL(d_recall)` | `FILL(d_ap50)` | `FILL(d_map5095)` | `FILL(d_lt8)` | `FILL(d_fpimg)` |

## 7. Real no-target FP/image V1 vs V2 (primary endpoint)

| model | images | FP raw | FP/image | 95% CI (Poisson, raw count) | image-level rate (Wilson) |
|---|---|---|---|---|---|
| V1 | 496 | **6** | 0.012097 | [0.004439, 0.026330] | 0.010081 [0.004507, 0.022482] |
| V2 | `FILL(v2_pooled_images)` | `FILL(v2_pooled_fp)` | `FILL(v2_pooled_rate)` | `FILL(v2_pooled_ci)` | `FILL(v2_pooled_imgrate)` |

Per-set breakdown (raw FP, conf 0.25): `FILL(v2_pooled_per_set)`.
Decision arithmetic: a 20% reduction of 6 requires landing at **<= 4 FPs**; relative reduction `FILL(fp_reduction)`.

## 8. Small-object effect

| bucket | V1 GT | V1 FN | V1 FN rate | V2 GT | V2 FN | V2 FN rate |
|---|---|---|---|---|---|---|
| <4 | 24 | 24 | 1.0000 | | | |
| 4-6 | 37 | 37 | 1.0000 | | | |
| 6-8 | 44 | 36 | 0.8182 | | | |
| 8-12 | 180 | 140 | 0.7778 | | | |
| 12-16 | 110 | 95 | 0.8636 | | | |
| 16-24 | 66 | 53 | 0.8030 | | | |
| **<8px aggregate** | **105** | **97** | **0.9238** | | | |
Sources: `eth_only_v1_fn_analysis.csv` (V1) and the V2 run of the same tool.

## 9. Synthetic diagnostic sets (clearly non-real)

Reported for continuity only; they are **not** part of the causal claim and must not be read as real-domain
generalisation evidence.

| set | model | images | GT | R | AP50 | mAP50-95 | Recall_<8 |
|---|---|---|---|---|---|---|---|
| controlled_capability/images | V1 | 2,070 | 2,070 | 0.0053 | 0.0669 | 0.0455 | 0.0000 |
| controlled_capability/images | V2 | 2,070 | 2,070 | `FILL(v2_ctrl_r)` | `FILL(v2_ctrl_ap50)` | `FILL(v2_ctrl_map)` | `FILL(v2_ctrl_lt8)` |
| challenge_test/images | V1 | 227 | 194 | 0.0052 | 0.0934 | 0.0535 | 0.0000 |
| challenge_test/images | V2 | 227 | 194 | `FILL(v2_chal_r)` | `FILL(v2_chal_ap50)` | `FILL(v2_chal_map)` | `FILL(v2_chal_lt8)` |

V1 false positives on the synthetic unlabelled sets (diagnostic, conf 0.25): synthetic_3d 13/84,
synthetic_on_real_bg 17/160. V2: `FILL(v2_synth_fp)`.

## 10. Decision

**`FILL(decision)`** - `FILL(decision_why)`.

| criterion | threshold | measured |
|---|---|---|
| primary: pooled real no-target FP/image | down >= 20% | `FILL(fp_reduction)` |
| secondary: `val|eth_unseen` recall | change > -0.02 | `FILL(d_recall)` |
| harm guard | recall <= -0.05 is HARMFUL_TRADEOFF | `FILL(harm_check)` |
| low-value guard | FP change < 10% with accuracy stable | `FILL(low_value_check)` |

Source of truth: `eth_real_hardneg_v2_decision.json` (produced by `tools/compare_eth_hardneg_v2.py`).

## 11. Limits

1. The primary endpoint has 6 baseline events: any verdict rests on removing 1-3 false positives, and the 95%
   CI on the pooled rate spans a factor of ~6. The raw counts and the interval are therefore reported together.
2. `val|eth_unseen` carries 495 GT (105 below 8 px), so differences under ~0.02 there are inconclusive by the
   plan's own thresholds.
3. V2 differs from V1 **only** by the 89 hard negatives at repeat 8 (4.67% exposure); the positive set, the
   negatives, the recipe, the val set and the evaluator are unchanged, which is what makes the contrast causal.
4. Both models are plain YOLO26s at batch 8 / nbs 32; nothing here supports claims about other architectures,
   batch sizes or data mixtures (see the scope block at the top).
5. `val|eth_unseen` mixes frame domains (eth_main 263 GT, synthetic 120, iPhone 112), so the secondary endpoint
   is a mixed-domain number; `eth_only_v1_fn_analysis.csv` keeps the per-domain split.
6. Because the leakage-free protocol removed ml_3 and ml_6 from training entirely, 187 of the 194 recoverable
   FNs are located there: the secondary-endpoint ceiling is set by protocol scope, not by the hard negatives. A
   V2 gain on that endpoint could only come from generalising to unseen locations, which 89 negatives cannot
   deliver - so the primary (false-positive) endpoint is the one this experiment can actually move.

## 12. Next experiment recommendation

Filled from the decision: `FILL(next_experiment)`. The predetermined branches are:

- `USEFUL`: keep real hard negatives in the training mixture and grow the verified pool before adding budget
  elsewhere; re-measure with the pooled endpoint and its interval.
- `HARMFUL_TRADEOFF` / `LOW_VALUE`: stop spending training budget on negative-only data and target
  **positive-domain coverage**, because V1's dominant real-domain failure is missed detections (414 of 495 GT),
  not false alarms (6 of 496 images).
- `INCONCLUSIVE`: raise the endpoint's power first (larger verified real no-target pool, or a pre-registered
  repeat/aggregation change) before another negative-only run.

---

## Appendix: frozen fillers (each names its source artifact)

| filler | source artifact |
|---|---|
| `contract_sha256` | the V2 run manifest `contract.sha256` |
| `run_finished`, `best_bytes`, `best_sha256`, `best_epoch`, `epochs_recorded`, `best_map5095`, `best_recall`, `fitness_check`, `peak_vram` | `eth_real_hardneg_v2_full_e50_manifest.json` (+ its `results.csv`) |
| `audit_status` | `eth_real_hardneg_v2_data_audit.json` |
| `fn_by_bucket`, `fn_by_location`, `recoverable_fn`, `b_*` | `eth_only_v1_fn_analysis.{json,csv}` |
| `v2_unseen_*`, `d_*`, `v2_ctrl_*`, `v2_chal_*`, `v2_synth_fp` | the V2 run of `tools/eval_yolo26_v1.py` + `tools/eval_eth_official_baseline.py`, summarised into `eth_real_hardneg_v2_vs_v1.csv` |
| `v2_pooled_*` | V2 rows of the canonical frozen-set negatives table, pooled by `tools/analyze_eth_only_v1_errors.py` |
| `decision`, `decision_why`, `fp_reduction`, `d_recall`, `harm_check`, `low_value_check`, `next_experiment` | `eth_real_hardneg_v2_decision.json` |
