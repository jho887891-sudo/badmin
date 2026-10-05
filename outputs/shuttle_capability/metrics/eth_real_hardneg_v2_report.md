# ETH Real Hard-Negative YOLO26s V2 - decision report

> **STATUS: FINAL (2026-10-05).** Every numeric cell is measured from the frozen artifacts named in the appendix.
> The 50-epoch V2 run finished 2026-10-05T13:53:51Z; the ETH-official reference values come from the earlier
> committed run of the same tool and protocol (`eth_vs_ours.csv`).
>
> Interpretation scope is enforced by the plan: this report may conclude **only** whether verified real
> hard negatives helped FP behaviour without materially harming Recall under the frozen YOLO26s V1 recipe. It
> must not claim anything about synthetic vs real generalisation, iPhone, D455, YOLO26s vs YOLOv8s, or P2.

## 1. Contract and hashes

| item | value |
|---|---|
| experiment | `eth_real_hardneg_yolo26s_v2` (spec `docs/superpowers/specs/2026-10-05-eth-real-hardneg-yolo26s-v2-design.md`) |
| V2 contract | `configs/eth_real_hardneg_yolo26s_v2.yaml`, sha256 `01ba35b2e0fbdc4ce90fdd5c2dff5e6ebedf0ac43fef683b82fdf671d63b150b` |
| V1 contract (reference) | `configs/eth_only_yolo26s_1024_v1.yaml` |
| pretrained init | official `yolo26s.pt`, 20,422,725 B, sha256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` (verified at launch) |
| V2 train manifest | `data/eth_real_hardneg_v2_train_manifest.csv`, sha256 `d2bd887357555c9262f769ba81d3bfd49350e1670e80a94ecdc1d166898a8d4d`, 15,255 rows |
| V2 val manifest | `data/eth_real_hardneg_v2_val_manifest.csv`, sha256 `0ed405719af1d62763fb41fdad79b2581e0e7787da9a9c237c203a62ef735918` (byte-identical to V1) |
| V2 run | `full_v2_e50` on the remote A6000, started 2026-10-05T06:23:08Z, **finished 2026-10-05T13:53:51Z (7.51 h)** |
| V2 checkpoint | `best.pt`, **20,344,133 B**, sha256 `3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7`, **epoch 21** |
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

`outputs/shuttle_capability/metrics/eth_real_hardneg_v2_data_audit.json` -> **status `PASS`** with
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

### 4b. Supplementary: what the leakage-free protocol cost (read-only)

`eth_only_v1_protocol_cost.{json,csv}` turns the exclusion into per-location numbers (tool:
`tools/analyze_protocol_cost.py`, 9 tests).

| fact | value |
|---|---|
| ETH locations consumed 100% by our evaluation sets | ml_6 (1,123 frames), ml_3 (1,004), uetlibergstrasse_1 (638) |
| frames unusable for training / trainable frames left | **2,765** / 16,913 |
| `val|eth_unseen` GT from those dropped locations | **263**, FN rate 0.7643 (ml_3 115/144, ml_6 84/115, uetli 2/4) |
| `val|eth_unseen` GT from our own domains | **232**: synthetic 120 (FN 0.9750), iPhone 112 (FN 0.8571) |
| `val|eth_unseen` GT from locations this model trained on | **0** |
| recoverable FNs inside the dropped locations | **187 of 194** (96.4%) |

The budget closes exactly: 263 + 232 = 495 GT, the total of the canonical table.

Consequence for interpretation: the secondary endpoint has **no in-distribution component at all** - every GT
in it is either from a location the model never saw or from a domain it never saw. That is why its recall is
0.1636, and it is also why 89 hard negatives cannot move it. The primary false-positive endpoint is the one this
experiment can actually move. Closing the recall gap requires a protocol change (release part of those
evaluation frames, or collect frames from those locations that are in no evaluation manifest), not a data-mixture
tweak.

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
| epochs | 50 (50 recorded) |
| selection | internal location-disjoint val mAP50-95 only: epoch **21**, mAP50-95 **0.66923**, mAP50 0.95781, recall **0.91634** |
| fitness-argmax equals mAP50-95 argmax | **True** (both epoch 21), so `best.pt` is the contract-selected checkpoint |
| memory | 24 GiB per-process cap (fraction 0.5063), peak VRAM **6.71 GiB** |
| smoke evidence | `eth_real_hardneg_v2_smoke.json` status PASS, peak 6.78 GiB, batch 8 first try |

## 6. `val|eth_unseen` V1 vs V2 (secondary endpoint)

| model | images | GT | P | R | AP50 | mAP50-95 | Recall_<8 | FP/image |
|---|---|---|---|---|---|---|---|---|
| V1 `eth_only_v1_best` | 1,648 | 495 | 0.6864 | 0.1636 | 0.3510 | 0.1909 | 0.0762 | 0.0225 |
| V2 `eth_real_hardneg_v2_best` | 1,648 | 495 | **0.8783** | **0.2040** | 0.3710 | 0.1897 | **0.0381** | 0.0085 |
| delta (V2 - V1) | | | +0.1918 | **+0.0404** | +0.0200 | **-0.0012** | **-0.0381** | -0.0140 |

## 7. Real no-target FP/image V1 vs V2 (primary endpoint)

| model | images | FP raw | FP/image | 95% CI (Poisson, raw count) | image-level rate (Wilson) |
|---|---|---|---|---|---|
| V1 | 496 | **6** | 0.012097 | [0.004439, 0.026330] | 0.010081 [0.004507, 0.022482] |
| V2 | 496 | **1** | **0.002016** | [0.000051, 0.011233] (raw count [0.025, 5.572]) | 0.002016 [0.000356, 0.011331] |

Per-set breakdown (raw FP, conf 0.25): backgrounds **2 -> 0** of 30, raw **3 -> 0** of 59, video 0 -> 0 of 150, match_frames 0 -> 0 of 235, real_train 1 -> 1 of 22. The single remaining false positive has confidence 0.4757.
Decision arithmetic: a 20% reduction of 6 requires landing at **<= 4 FPs**; measured relative reduction **83.33%** (6 -> 1). Under the V1 rate, observing <= 1 event has one-sided Poisson probability **0.0174**.

## 8. Small-object effect

| bucket | V1 GT | V1 FN | V1 FN rate | V2 GT | V2 FN | V2 FN rate |
|---|---|---|---|---|---|---|
| <4 | 24 | 24 | 1.0000 | 24 | 24 | 1.0000 |
| 4-6 | 37 | 37 | 1.0000 | 37 | 37 | 1.0000 |
| 6-8 | 44 | 36 | 0.8182 | 44 | 40 | 0.9090 |
| 8-12 | 180 | 140 | 0.7778 | 180 | 124 | 0.6888 |
| 12-16 | 110 | 95 | 0.8636 | 110 | 84 | 0.7636 |
| 16-24 | 66 | 53 | 0.8030 | 66 | 52 | 0.7878 |
| **<8px aggregate** | **105** | **97** | **0.9238** | **105** | **101** | **0.9619** |
Sources: `eth_only_v1_fn_analysis.csv` and `eth_hardneg_v2_fn_analysis.csv` (same tool, two dumps).
Overall the FNs fall from 414 to 394; the gain is concentrated in 8-16 px (235 -> 208 FNs) while **<8 px gets
worse (97 -> 101)**, as does 24-32 (9 -> 11). Small objects are the one bucket the hard negatives did not help.

## 9. Synthetic diagnostic sets (clearly non-real)

Reported for continuity only; they are **not** part of the causal claim and must not be read as real-domain
generalisation evidence.

| set | model | images | GT | R | AP50 | mAP50-95 | Recall_<8 |
|---|---|---|---|---|---|---|---|
| controlled_capability/images | V1 | 2,070 | 2,070 | 0.0053 | 0.0669 | 0.0455 | 0.0000 |
| controlled_capability/images | V2 | 2,070 | 2,070 | 0.0034 | 0.0345 | 0.0241 | 0.0000 |
| challenge_test/images | V1 | 227 | 194 | 0.0052 | 0.0934 | 0.0535 | 0.0000 |
| challenge_test/images | V2 | 227 | 194 | 0.0000 | 0.0438 | 0.0245 | 0.0000 |

V1 false positives on the synthetic unlabelled sets (diagnostic, conf 0.25): synthetic_3d 13/84,
synthetic_on_real_bg 17/160. V2: **synthetic_3d 2/84, synthetic_on_real_bg 2/160**.

## 10. Decision

**`USEFUL`** - FP/image down 83.3% (>= 20%) with recall change +0.0404 > -0.02.

| criterion | threshold | measured |
|---|---|---|
| primary: pooled real no-target FP/image | down >= 20% | **down 83.33%** (0.012097 -> 0.002016) |
| secondary: `val|eth_unseen` recall | change > -0.02 | **+0.0404** (0.1636 -> 0.2040) |
| harm guard | recall <= -0.05 is HARMFUL_TRADEOFF | **not triggered** (+0.0404); Recall_<8 does fall by 0.0381 |
| low-value guard | FP change < 10% with accuracy stable | **not triggered** (83.33% change) |

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
   FNs are located there: the secondary-endpoint ceiling is set by protocol scope, not by the hard negatives.
   V2 did still gain +0.0404 recall on that endpoint, but the gain is a precision/threshold effect (FP 37 -> 14),
   not new location competence.
7. **The hard negatives cost small-object recall**: the <8 px bucket goes from 97/105 missed (0.9238) to
   101/105 (0.9619), and 24-32 from 9/11 to 11/11, while 8-16 px improves (235 -> 208 FNs). The frozen decision
   rule does not include a small-object term, so the verdict is unaffected, but any deployment that relies on
   sub-8 px detections must re-measure this.
8. **Endpoint power**: the primary endpoint moved from 6 to 1 event, which under the V1 rate has one-sided
   Poisson probability 0.0174 - suggestive rather than overwhelming, and the interval on the new rate
   [0.000051, 0.011233] still overlaps low values. Repeated seeds or a larger real no-target pool would settle it.

## 12. Next experiment recommendation

Decision-driven recommendation: **keep the verified real hard negatives in the mixture** and grow the verified pool before spending budget elsewhere; in parallel attack the two things this run did not fix - the small-object regression (section 8) and the protocol-scope recall ceiling (section 4b). The predetermined branches are:

- `USEFUL`: keep real hard negatives in the training mixture and grow the verified pool before adding budget
  elsewhere; re-measure with the pooled endpoint and its interval.
- `HARMFUL_TRADEOFF` / `LOW_VALUE`: stop spending training budget on negative-only data and target
  **positive-domain coverage**, because V1's dominant real-domain failure is missed detections (414 of 495 GT),
  not false alarms (6 of 496 images). Section 4b already localises that gap: 187 of the 194 recoverable misses sit
  in the two ETH locations our evaluation sets consume entirely, and 232 of the 495 GT come from our own
  synthetic/iPhone domains - so the first lever is protocol scope (release or re-collect those frames), the second
  is domain coverage, and only then model capacity.
- `INCONCLUSIVE`: raise the endpoint's power first (larger verified real no-target pool, or a pre-registered
  repeat/aggregation change) before another negative-only run.

---

## Appendix: frozen fillers (each names its source artifact)

| filler | source artifact |
|---|---|
| `contract_sha256` | the V2 run manifest `contract.sha256` (filled) |
| `run_finished`, `best_bytes`, `best_sha256`, `best_epoch`, `epochs_recorded`, `best_map5095`, `best_recall`, `fitness_check` | `eth_real_hardneg_v2_full_e50_manifest.json` (+ its `results.csv`) - all filled |
| `peak_vram` | the remote run log (`full_v2_e50_run.log`) - filled from it |
| `audit_status` | `eth_real_hardneg_v2_data_audit.json` = **PASS** (filled) |
| `fn_by_bucket`, `fn_by_location`, `recoverable_fn`, `b_*` | `eth_only_v1_fn_analysis.{json,csv}` |
| `v2_unseen_*`, `d_*`, `v2_ctrl_*`, `v2_chal_*`, `v2_synth_fp` | `eth_vs_ours_eth_hardneg_v2.csv`, `eth_hardneg_v2_frozen_test_negatives.csv`, `eth_vs_ours_eth_hardneg_v2_size_buckets.csv` - all filled |
| `v2_pooled_*` | V2 rows of `eth_hardneg_v2_frozen_test_negatives.csv`, pooled by `tools/analyze_eth_only_v1_errors.py` - filled |
| `decision`, `decision_why`, `fp_reduction`, `d_recall`, `harm_check`, `low_value_check` | `eth_real_hardneg_v2_decision.json` - filled |
| `fn_by_bucket`, `fn_by_location`, `recoverable_fn` | `eth_only_v1_fn_analysis.{json,csv}` (V1) and `eth_hardneg_v2_fn_analysis.{json,csv}` (V2) - filled |
