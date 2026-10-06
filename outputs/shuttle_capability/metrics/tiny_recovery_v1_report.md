# Tiny-object Recovery V1 - decision report

> **STATUS: FINAL (2026-10-06).** The 50-epoch run finished 2026-10-05T23:02:26Z and every cell below is measured. Sections 2, 3, 4, 11, 12 and 13 are already measured, and section 4 was measured
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
| run | `tiny_full_e50`, started **2026-10-05T16:20:34Z**, finished **2026-10-05T23:02:26Z (6.70 h)** |
| selected checkpoint | **epoch 48**, **20,344,133 B**, sha256 `7af5a563b28e5f642e2209d7eca6e94281cad858d0758d47fc4e932dab8fbd57` |
| selection rule | internal val mAP50-95 argmax only; fitness argmax equals it: **True** (both epoch 48) |
| memory | 24 GiB per-process cap, fraction 0.5063; peak VRAM **6.78 GiB** |
| smoke | `tiny_recovery_v1_smoke.json` -> **PASS** (batch 8 first try, 3 epochs, finite losses; diagnostic only) |

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
| epochs | **50 of 50** |
| selection | epoch **48**, internal val mAP50-95 **0.67581**, mAP50 0.95516, recall 0.9015 |
| curve | e5 0.63742, e10 0.65889, e15 0.67265, e20 0.66848, e25 0.66610, e30 0.67002, e35 0.67185, e40 0.67273, e45 0.67429, e50 0.67563 (best 0.67581 at e48) |
| V2 reference | epoch 21, mAP50-95 0.66923 / R 0.91634 |

## 6. Primary endpoint: `<8 px` recovery on `val|eth_unseen`

| bucket | V1 TP/FN | V2 TP/FN | V3 TP/FN | V3 recall | delta vs V2 |
|---|---|---|---|---|---|
| <4 | 0 / 24 | 0 / 24 | 0 / 24 | 0.0000 | 0 (recall 0.0000 -> 0.0000) |
| 4-6 | 1 / 36 | 0 / 37 | 0 / 37 | 0.0000 | 0 (recall 0.0000 -> 0.0000) |
| 6-8 | 7 / 37 | 4 / 40 | **6 / 38** | 0.1364 | **+2 hits (recall 0.0909 -> 0.1364)** |
| **<8 total** | **8 / 97** | **4 / 101** | **6 / 99** | **0.057143** | **+2 hits (recall 0.038095 -> 0.057143)**; target >= 8 hits NOT met |
| 8-12 | 0.222222 | 56 / 124 (0.3111) | 50 / 130 | 0.2778 | **-0.0333** |
| 12-16 | 0.136364 | 26 / 84 (0.2364) | 19 / 91 | 0.1727 | **-0.0636** |
| 8-16 combined | 0.189655 | 0.282759 | 0.237931 | | **-0.0448** |

Source: `tiny_recovery_v1_vs_v2.csv` and `tiny_recovery_v1_size_buckets.csv` (V1/V2 columns already committed).
Every cell is reported with raw TP/FN counts as well as the rate, per design section 23.

## 7. Guards

| guard | threshold | V2 | V3 | verdict |
|---|---|---|---|---|
| overall Recall | >= 0.1840 (drop <= 0.02) | 0.2040 | **0.167677** | **FAIL (change -0.0364)** |
| frozen real no-target FP | <= 3 of 496 | 1 | **0** | **PASS (0 <= 3, better than V2)** |
| mAP50-95 | delta > -0.01 | 0.189727 | **0.138830** | **FAIL (change -0.0509)** |
| 8-16 px recall | delta >= -0.02 (else SIZE_TRADEOFF) | 0.282759 | **0.237931** | **FAIL (change -0.0448) -> the SIZE_TRADEOFF condition is also met, but HARMFUL has precedence** |

## 8. Frozen real no-target pool (496 images, primary guard)

| cohort | images | FP raw | FP/image | 95% CI (raw count) | image-level rate (Wilson) |
|---|---|---|---|---|---|
| V1 | 496 | 6 | 0.012097 | [0.004439, 0.026330] | 0.010081 [0.004507, 0.022482] |
| V2 | 496 | 1 | 0.002016 | [0.000051, 0.011233] | 0.002016 [0.000356, 0.011331] |
| V3 | 496 | **0** | **0.000000** | [0.000000, 0.007438] (raw count [0, 3.689]) | 0.000000 [0.000000, 0.007685] |

Per-set breakdown and per-set intervals: `tiny_recovery_v1_no_target_fp.csv`.

## 9. Synthetic diagnostic sets (clearly non-real, never used for selection)

| set | V2 R / mAP50-95 | V3 R / mAP50-95 |
|---|---|---|
| controlled_capability/images | 0.0034 / 0.0241 | 0.0024 / 0.0324 |
| challenge_test/images | 0.0000 / 0.0245 | 0.0000 / 0.0227 |

## 10. Decision

**`HARMFUL`** - recall change **-0.0364** (<= -0.02) and mAP50-95 change **-0.0509** (<= -0.01) breached the harm guard.

Matrix applied in the frozen precedence `HARMFUL -> FP_REGRESSION -> SIZE_TRADEOFF -> STRONG_SUCCESS -> USEFUL ->
NO_TINY_GAIN`, from `tiny_recovery_v1_decision.json` (which also records every threshold and the measured values).

| label | condition | measured |
|---|---|---|
| STRONG_SUCCESS | TP_<8 >= 12 with all guards | not met (TP_<8 = 6) |
| USEFUL | TP_<8 >= 8, recall >= 0.1840, FP <= 3, mAP delta > -0.01 | **not met**: TP_<8 = 6 (target 8), recall 0.1677 (needs >= 0.1840), mAP delta -0.0509; only the FP guard passed |
| NO_TINY_GAIN | TP_<8 < 8 with other metrics stable | TP_<8 = 6 < 8 **but** the other metrics are not stable (recall -0.0364, mAP -0.0509), so this label does not apply |
| FP_REGRESSION | frozen no-target FP >= 4 | not met (0 of 496 - the opposite of a regression) |
| SIZE_TRADEOFF | 8-16 px recall delta < -0.02 | **condition met** (-0.0448), but HARMFUL has higher precedence |
| HARMFUL | recall delta <= -0.02 or mAP delta <= -0.01 | **both met** (-0.0364 and -0.0509) -> the reported decision |

### What the intervention actually did

The target moved in the intended direction but far too little, and everything else moved backwards:

- `<8 px` hits **4 -> 6** (all of them in the 6-8 bucket, recall 0.038095 -> 0.057143) against a minimum success line
  of 8 hits; the `<4` and `4-6` buckets stayed at exactly 0.0000.
- Overall `val|eth_unseen` recall **0.2040 -> 0.1677** and mAP50-95 **0.1897 -> 0.1388**: most of V2's hard-negative
  gain was given back, which is why the harm guard fires before any tiny-specific label.
- The 8-16 px range that V2 had improved also regressed (Recall_8_12 0.3111 -> 0.2778, Recall_12_16 0.2364 ->
  0.1727), so the SIZE_TRADEOFF condition is met as well.
- The one metric that improved is the frozen no-target false-positive rate: **1 -> 0 of 496**.

Mechanism check, using the same probe as section 4b but on the V3 predictions
(`tiny_representability_v1_v3.{json,csv}`): the share of tiny false negatives that have *any* detection on them at
conf >= 0.01 is **5 of 99 (0.0505)** versus **5 of 101 (0.0495)** before. The exposure change did not create new
localisation evidence - it nudged two 6-8 bucket objects over the operating threshold while perturbing the shared
features enough to lose mid-size recall. That is exactly the outcome the pre-registered representability reading
anticipated for a sub-stride object population.

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
