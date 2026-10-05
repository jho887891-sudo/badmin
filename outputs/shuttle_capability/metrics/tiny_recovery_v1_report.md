# Tiny-object Recovery V1 - decision report

> **STATUS: SKELETON (2026-10-05 16:0x UTC).** The smoke gate and the 50-epoch run are in flight, so every
> `FILL(id)` cell is pending. Sections 2, 3, 4, 11, 12 and 13 are already measured, and section 4 was measured
> **before** the run finished (pre-registered, not post-hoc). The appendix maps every filler to its source artifact.
>
> Experiment `ETH_HARDNEG_TINY_RECOVERY_V1`; authoritative design:
> `2026-10-05-tiny-object-recovery-v1-design.md` (840 lines, supplied read-only); repository spec:
> `docs/superpowers/specs/2026-10-05-tiny-object-recovery-v1-design.md`.
>
> The report may conclude only whether a budgeted increase of existing real tiny-positive exposure recovers
> `<8 px` recall while keeping V2's low false-positive rate. It must not conclude anything about synthetic,
> iPhone, D455, new locations, releasing ml_3/ml_6, P2, larger imgsz, YOLOv8s or loss reweighting.

## 1. Contract and hashes

| item | value |
|---|---|
| V3 contract | `configs/tiny_recovery_yolo26s_v1.yaml`, sha256 `473ab4513c2137e6ec1148ef9f1c455986c3cb0924062a638acc45440035b9b5` |
| frozen base | V2 contract `configs/eth_real_hardneg_yolo26s_v2.yaml` (line-copied; recipe/val/selection/evaluator unchanged) |
| pretrained init | official `yolo26s.pt`, 20,422,725 B, sha256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| V3 train manifest | `data/tiny_recovery_v1_train_manifest.csv`, 15,967 rows, sha256 `9e2bf91152cf393c3a4b5d16c7d2f7a901824a56c7835c4530b80aeb3c925352` |
| V3 val manifest | `data/tiny_recovery_v1_val_manifest.csv`, sha256 `0ed405719af1d62763fb41fdad79b2581e0e7787da9a9c237c203a62ef735918` (byte-identical to V2/V1) |
| train/val lists | `00de1471b171e220d6b60c18910eaada0c53e25208979fcea837bc92f108e22c` / `578da7435615cbf8911e9a7610900270eb888f8d4941e594e4f868e3f9020d77` |
| run | `tiny_full_e50`, started `FILL(run_started)`, finished `FILL(run_finished)` |
| selected checkpoint | epoch `FILL(best_epoch)`, bytes `FILL(best_bytes)`, sha256 `FILL(best_sha256)` |
| selection rule | internal val mAP50-95 argmax only; fitness argmax must equal it: `FILL(fitness_check)` |
| memory | 24 GiB per-process cap, fraction 0.5063; peak VRAM `FILL(peak_vram)` |
| smoke | `tiny_recovery_v1_smoke.json` -> `FILL(smoke_status)` (diagnostic only, never a result) |

## 2. Data composition (measured)

| block | rows | note |
|---|---|---|
| V2 rows retained | 15,255 | positives 13,992 + V2 negatives 551 + 712 hard-negative repeats |
| plus tiny-positive exposures | **+712** | the only change; 89 unique hard negatives x 8 = 712 made this the symmetric budget |
| V3 train total | **15,967** | extra exposure is 4.67% of the V2 set (4.46% of the new total) |
| V3 val | 2,920 | byte-identical to the V2 internal val |

Stratified selection of the 712 extra exposures, from existing V2 real tiny positives only (seed 42, at most one
extra copy per unique image):

| bucket | V2 exposure | extra | final | eligible unique |
|---|---|---|---|---|
| <4 | 161 | **19** | 180 | 161 |
| 4-6 | 1,282 | **153** | 1,435 | 1,282 |
| 6-8 | 4,522 | **540** | 5,062 | 4,522 |
| >=8 | 8,027 | 0 | 8,027 | (untouched by design) |
| total tiny | 5,965 | **712** | 6,677 | 5,965 |

Per-image accounting: `tiny_recovery_v1_tiny_exposure.csv` (5,965 rows with base/extra/final exposure and the
selection seed); source breakdown: `tiny_recovery_v1_source_counts.csv`.

## 3. Gates (audit PASS, measured)

`tiny_recovery_v1_data_audit.json` -> **status PASS**, with: unique positive SHA256 set identical to V2
(`6f2b3453175c31b4d3bad996c68d35278482aa0ed2ced7f0b2e51dac33fe48bd` on both sides), internal val byte-identical,
evaluation path overlap 0 and SHA256 overlap 0 against the canonical frozen sets, `extra_tiny_exposure == 712`,
`extra_non_tiny_positive == 0`, `tiny_extra_per_unique_image_max == 1`, prohibited sources 0.
The eligible pool (5,965) exceeds the budget (712), so the design's stop condition was never reached and no
per-image repeat was raised.

## 4. Pre-training diagnostics (measured before the run)

### 4a. V2 tiny false negatives
On `val|eth_unseen` V2 misses **101 of 105** `<8 px` GT (recall 0.0381); the 4 hits all come from the 6-8 bucket.
Full profile: `eth_hardneg_v2_fn_analysis.{json,csv}`.

### 4b. Representability and weak evidence (new, read-only, `tiny_representability_v1.{json,csv}`)

A tiny box is defined in 640-equivalent pixels; in the frozen 1024 network frame it is 1.6x larger, and a plain
YOLO26s predicts on an 8 px P3 stride.

| bucket | GT | TP | FN | recall | median size in the 1024 frame | below one 8 px stride |
|---|---|---|---|---|---|---|
| <4 | 24 | 0 | 24 | **0.0000** | 5.33 px | **24 / 24** |
| 4-6 | 37 | 0 | 37 | **0.0000** | 7.98 px | **19 / 37** |
| 6-8 | 44 | 4 | 40 | 0.0909 | 11.46 px | 0 / 44 |
| 8-12 | 180 | 56 | 124 | 0.3111 | 16.02 px | 0 / 180 |
| 12-16 | 110 | 26 | 84 | 0.2364 | 21.40 px | 0 / 110 |
| 16-24 | 66 | 14 | 52 | 0.2121 | 27.87 px | 0 / 66 |

Weak-evidence probe: matching every GT against detections at **conf >= 0.01** and any overlap (IoU >= 0.1, no
greedy constraint), only **5 of the 101 tiny false negatives** (4.95%) have any candidate box on them at all
(median confidence of those 5: 0.127).

Reading, stated before any V3 number existed:
1. The design's hypothesis (tiny evidence is *present but pushed below the boundary* by hard-negative learning)
   is **mostly not supported for the <4 and 4-6 buckets**: 61 of the 105 tiny GT (and 61 of the 101 tiny FN) sit
   inside a single P3 stride cell and recall there is exactly 0.0000, and for 96 of 101 misses the model emits no
   detection anywhere near the object even at a 0.01 floor.
2. Therefore exposure alone can only plausibly move the **6-8 bucket** (40 FN at 9.6-12.8 px, where all 4 current
   hits live). The design's minimum line (TP_<8 >= 8, i.e. +4) requires more than doubling that bucket's hits.
3. If V3 returns `NO_TINY_GAIN`, this diagnostic already points at grid resolution (P2 / higher effective input
   scale) for the sub-stride buckets rather than at more exposure - which is consistent with the design's own
   ladder in section 25, but the reason is now measured rather than assumed.

## 5. Training and checkpoint selection

| item | value |
|---|---|
| epochs | `FILL(epochs_recorded)` of 50 |
| selection | epoch `FILL(best_epoch)`, internal val mAP50-95 `FILL(best_map5095)`, mAP50 `FILL(best_map50)`, recall `FILL(best_recall)` |
| curve | `FILL(curve)` from `tiny_recovery_v1_full_e50_results.csv` |
| V2 reference | epoch 21, mAP50-95 0.66923 / R 0.91634 |

## 6. Primary endpoint: `<8 px` recovery on `val|eth_unseen`

| bucket | V1 TP/FN | V2 TP/FN | V3 TP/FN | V3 recall | delta vs V2 |
|---|---|---|---|---|---|
| <4 | 0 / 24 | 0 / 24 | `FILL(v3_lt4)` | `FILL(v3_lt4_r)` | `FILL(v3_lt4_d)` |
| 4-6 | 1 / 36 | 0 / 37 | `FILL(v3_4_6)` | `FILL(v3_4_6_r)` | `FILL(v3_4_6_d)` |
| 6-8 | 7 / 37 | 4 / 40 | `FILL(v3_6_8)` | `FILL(v3_6_8_r)` | `FILL(v3_6_8_d)` |
| **<8 total** | **8 / 97** | **4 / 101** | `FILL(v3_lt8)` | `FILL(v3_lt8_r)` | `FILL(v3_lt8_d)` |
| 8-12 | `FILL(v1_8_12)` | 56 / 124 | `FILL(v3_8_12)` | `FILL(v3_8_12_r)` | `FILL(v3_8_12_d)` |
| 12-16 | `FILL(v1_12_16)` | 26 / 84 | `FILL(v3_12_16)` | `FILL(v3_12_16_r)` | `FILL(v3_12_16_d)` |
| 8-16 combined | `FILL(v1_8_16_r)` | 0.282759 | `FILL(v3_8_16_r)` | | `FILL(v3_8_16_d)` |

Source: `tiny_recovery_v1_vs_v2.csv` and `tiny_recovery_v1_size_buckets.csv` (V1/V2 columns already committed).
Every cell is reported with raw TP/FN counts as well as the rate, per design section 23.

## 7. Guards

| guard | threshold | V2 | V3 | verdict |
|---|---|---|---|---|
| overall Recall | >= 0.1840 (drop <= 0.02) | 0.2040 | `FILL(v3_recall)` | `FILL(v3_recall_verdict)` |
| frozen real no-target FP | <= 3 of 496 | 1 | `FILL(v3_fp)` | `FILL(v3_fp_verdict)` |
| mAP50-95 | delta > -0.01 | 0.189727 | `FILL(v3_map5095)` | `FILL(v3_map_verdict)` |
| 8-16 px recall | delta >= -0.02 (else SIZE_TRADEOFF) | 0.282759 | `FILL(v3_8_16_r)` | `FILL(v3_8_16_verdict)` |

## 8. Frozen real no-target pool (496 images, primary guard)

| cohort | images | FP raw | FP/image | 95% CI (raw count) | image-level rate (Wilson) |
|---|---|---|---|---|---|
| V1 | 496 | 6 | 0.012097 | [0.004439, 0.026330] | 0.010081 [0.004507, 0.022482] |
| V2 | 496 | 1 | 0.002016 | [0.000051, 0.011233] | 0.002016 [0.000356, 0.011331] |
| V3 | 496 | `FILL(v3_fp)` | `FILL(v3_fp_image)` | `FILL(v3_fp_ci)` | `FILL(v3_fp_imgrate)` |

Per-set breakdown and per-set intervals: `tiny_recovery_v1_no_target_fp.csv`.

## 9. Synthetic diagnostic sets (clearly non-real, never used for selection)

| set | V2 R / mAP50-95 | V3 R / mAP50-95 |
|---|---|---|
| controlled_capability/images | 0.0034 / 0.0241 | `FILL(v3_ctrl)` |
| challenge_test/images | 0.0000 / 0.0245 | `FILL(v3_chal)` |

## 10. Decision

**`FILL(decision)`** - `FILL(decision_why)`.

Matrix applied in the frozen precedence `HARMFUL -> FP_REGRESSION -> SIZE_TRADEOFF -> STRONG_SUCCESS -> USEFUL ->
NO_TINY_GAIN`, from `tiny_recovery_v1_decision.json` (which also records every threshold and the measured values).

| label | condition | measured |
|---|---|---|
| STRONG_SUCCESS | TP_<8 >= 12 with all guards | `FILL(m_strong)` |
| USEFUL | TP_<8 >= 8, recall >= 0.1840, FP <= 3, mAP delta > -0.01 | `FILL(m_useful)` |
| NO_TINY_GAIN | TP_<8 < 8 with other metrics stable | `FILL(m_nogain)` |
| FP_REGRESSION | frozen no-target FP >= 4 | `FILL(m_fpreg)` |
| SIZE_TRADEOFF | 8-16 px recall delta < -0.02 | `FILL(m_size)` |
| HARMFUL | recall delta <= -0.02 or mAP delta <= -0.01 | `FILL(m_harm)` |

## 11. Known confounder (required by the design)

The manifest grows from 15,255 to 15,967 rows: +712 exposures is **+4.67% relative to V2** (+4.46% of the new
total), so for the same 50 epochs the total sample exposure and the number of optimiser updates also grow
slightly. The conclusion must therefore be phrased as the effect of a **budgeted tiny-positive exposure
intervention under the frozen hard-negative policy**, not as the isolated effect of size resampling.

## 12. Limits and what this report cannot conclude

1. `<8 px` has only 105 GT on `val|eth_unseen` and the sub-stride buckets have 24 and 37; single-digit changes are
   reported with raw counts and intervals for that reason.
2. The 496-image no-target pool yields 1-3 events, so its interval spans an order of magnitude.
3. Section 4b shows most tiny misses have no candidate box at all; this limits how much any exposure-only
   intervention can recover and is measured before the run, not inferred from it.
4. Scope, per design section 24: nothing here is evidence about synthetic, iPhone, D455, new locations, releasing
   ml_3/ml_6, P2, larger imgsz, YOLOv8s, or loss reweighting.

## 13. Next steps per label (frozen in the design section 25)

| label | next step |
|---|---|
| `STRONG_SUCCESS` | keep `hard_negative_repeat=8` and `tiny_positive_extra_exposure=712`, move to real deployment validation |
| `USEFUL` | keep the policy, validate next on real D455 onboard data |
| `NO_TINY_GAIN` | stop increasing oversampling; test in order (1) tiny-object loss weighting, (2) P2, (3) higher effective input scale - one variable at a time. Section 4b already argues that (2)/(3) is the binding constraint for the sub-stride buckets |
| `FP_REGRESSION` | there is a sensitivity/specificity trade-off; next run should ablate the hard-negative : tiny-positive exposure ratio instead of pushing tiny exposure further |
| `SIZE_TRADEOFF` / `HARMFUL` | do not deploy; re-examine the intervention before spending more training budget |

---

## Appendix: filler -> source artifact

| filler | source |
|---|---|
| `run_started`, `run_finished`, `epochs_recorded`, `best_epoch`, `best_bytes`, `best_sha256`, `fitness_check`, `best_map5095`, `best_map50`, `best_recall` | `tiny_recovery_v1_full_e50_manifest.json` (+ `_results.csv`) |
| `peak_vram` | the remote chain log `tiny_chain.log` / `tiny_full_e50_run.log` |
| `smoke_status` | `tiny_recovery_v1_smoke.json` |
| `v3_*`, `m_*`, `decision` | `tiny_recovery_v1_vs_v2.csv`, `tiny_recovery_v1_size_buckets.csv`, `tiny_recovery_v1_no_target_fp.csv`, `tiny_recovery_v1_decision.json` |
| `v1_8_12`, `v1_12_16`, `v1_8_16_r` | `eth_vs_ours_eth_only_v1.csv` (already committed) |
| V2 columns throughout | `eth_vs_ours_eth_hardneg_v2.csv`, `eth_hardneg_v2_frozen_test_negatives.csv` (already committed) |
