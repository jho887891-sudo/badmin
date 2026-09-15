# DETECTOR_RELEASE_ID: `shuttle-detector-v1-2026-09-15`

Per `docs/superpowers/specs/09_FREEZE_HANDOFF_SPEC.md`. **Status: FROZEN_CANDIDATE — not FROZEN.**
Section 12 allows `FROZEN` only after final acceptance, and final acceptance cannot be granted in this
repository for reasons listed in section 10 below. Everything the spec asks for is recorded here so that
acceptance, when the missing inputs arrive, is a comparison rather than a repeat.

## 1. Frozen model

| Item | Value |
|---|---|
| architecture | YOLO26s, single-stage detector |
| nc | **1** |
| class 0 | `shuttlecock` |
| weights | `weights/best.pt`, 20,317,957 bytes |
| **weight sha256** | `2530af3515983e27dd088fcabd2c80ae575eeb28adc6ee5ce3e7dc97b171b9b4` |
| weights/last.pt sha256 | `bba886cd1b618d120493c03a20db2e5d179555fc321c1d3287721bbd4248619b` |
| resolved_config.json sha256 | `9ca41993f7774f1346a04b8b17ac1188c51216fcf64f512045c6510b3f8ed54d` |
| args.yaml sha256 | `8f8b4be8b987df558d017d4f7e6d67e42951f05df1864a562196e8d29ead52c8` |
| run_metadata.json sha256 | `b979af6f270526e009b0f6068b8a34707c35550d1e960b4f044484d1f4c548b8` |
| code commit | `9904d3e2e030aa64483b12b2d1440eed253b0aee` |

## 2. Frozen inference configuration

| Setting | Value |
|---|---|
| input resolution | 640 |
| confidence threshold | **0.05** (see the operating-point table in `THRESHOLD_OPERATING_POINT.md`) |
| IoU / NMS | Ultralytics default 0.7 NMS IoU; evaluation match IoU 0.5 |
| Top-K | 5 |
| preprocessing | letterbox to 640x640, RGB, /255, ImageNet normalisation (Ultralytics default) |
| postprocessing | NMS then rank by confidence, keep Top-K, class-agnostic (single class) |
| device | CUDA, `device=0` |

The confidence threshold is the one setting a consumer is most likely to want to change. The scan exists
because that choice is a trade: at 0.05 the detector produces 46 boxes on 30 shuttle-free scenes with a
worst confidence of 0.931 for the older model, and raising it to 0.40 cut the count by 76% while costing
S8 recall 0.466 -> 0.344. **0.05 is frozen because it is the setting every measurement in this project
was taken at**, not because it is optimal for deployment.

## 3. Frozen data version

| Split | Rows | Manifest sha256 (first 16) | Notes |
|---|---|---|---|
| TRAIN | 1108 | `f01ef6281ed5bb18` | 922 positives (382 Isaac-rendered, 41%) + 186 negatives (17%) |
| VAL | 144 | `71b0e5f340dc3c9f` | unchanged since the original baseline |
| FIXED_CORE_TEST (controlled matrix) | 2070 | `4eaf056f9d8d4375` | numpy-rendered, see the caveat in section 9 |
| FIXED_CORE_TEST (frozen P3) | 160 | `2b7728ffe1c811f6` | numpy-rendered on 30 real backgrounds |
| FIXED_CORE_TEST (real photographs) | 16 | `99a268e7a1c08d2f` | **10 rows are 6 distinct images; 4 are byte-identical duplicates** |
| CHALLENGE_TEST | 227 | `996ebd5a8dd96e2c` | built after the frozen core and proven disjoint from it |

TRAIN positive size distribution, in px: <4 32 | 4-8 214 | 8-16 250 | 16-32 82 | 32-64 66 | 64-128 82 |
128-256 86 | 256-512 68 | >512 42. Synthetic/real ratio: **all training images are synthetic composites**;
the synthetic shuttle is rendered either by the project numpy rasteriser or by Isaac Sim, and composited
onto 25 real training backgrounds. **The project holds no real training photographs at all** - every real
image it owns is frozen test data.

## 4. Frozen capability report

| Question the spec asks | Answer |
|---|---|
| stable detection range | **>= 24 px**, where recall is 0.775 to 1.000 across the controlled size curve |
| degradation band | **8-24 px**, recall falling from about 0.475 at 12-16 px to 0.775 at 24-32 px |
| below which unreliable | **< 8 px**: 0.400 at 8-12 px, 0.250 at 6-8 px, and 1-3 px measures 0.000 in every model this project trained |
| hardest pose | **side view**, recall 0.150 against 0.600 for flight rotation |
| hardest video condition | **NOT MEASURED** - no per-frame ground truth exists (section 10) |
| Left camera | **NOT APPLICABLE** - the project is monocular (section 10) |
| Right camera | **NOT APPLICABLE** - same |
| longest miss streak | **NOT MEASURED** - requires video ground truth |
| end-to-end detection latency | median **51.2 ms** at imgsz 640 on the A6000, measured across preprocessing, inference, NMS and Top-K |
| peak VRAM | **108 MB** for inference; 5.8 GB during training at batch 16 |

Latency honesty: the P95 and P99 in the same run were 13.8 s and 58.0 s. Those are the host reading
images through a contended FUSE mount, not the detector - the median reproduces the 51.7 ms measured on
an idle host earlier. A deployment latency figure must be taken on the deployment machine.

Supporting evidence, all on frozen sets that never entered training:

| Protocol | Result for this release |
|---|---|
| real photographs (6 distinct images) | **recall 0.600, precision 0.500, mAP50 0.505** |
| false positives, 30 shuttle-free scenes | **43 boxes, 8/30 scenes clean, 5/30 carry a box above 0.5, worst 0.669** |
| frozen P3, 160 images | precision 0.196, recall 0.438, mAP50 0.253 |
| CHALLENGE C5, historical 1100-1524 px | **recall 1.000** |
| CHALLENGE C1 vs C1n, paired | recall 0.333 on scenes that fire 0.97 boxes each when empty - **no discrimination** |
| controlled matrix | recall 0.426, mAP50 0.351 (see the renderer caveat) |

## 4b. The deployment-aligned capability curve, and it supersedes the numpy-rendered figures

Section 4 quotes the controlled matrix, which is numpy-rendered. That set measures an appearance this
release was NOT trained for, so its numbers understate the model - measurably so: on an appearance-aligned
ladder built from Isaac Sim renders, `baseline_round3` (trained on numpy renders) scores 0.05-0.20 across
24-1024 px while this release scores 0.50-1.00 over the same range.

**The figure to plan against is therefore this one**, measured on 237 rows at 12 sizes x 20 repeats, built
from 260 renders that training never saw (validated by content hash: 1 of 250 frames identical):

| target size | recall | | target size | recall |
|---|---|---|---|---|
| 4 px | 0.050 | | 32 px | 0.600 |
| 6 px | 0.150 | | 64 px | **0.850** |
| 8 px | 0.300 | | 128 px | **0.900** |
| 12 px | 0.500 | | 256 px | 0.850 |
| 16 px | 0.500 | | 512 px | **1.000** |
| 24 px | 0.400 | | 1024 px | **1.000** |

Overall: precision 0.272, recall **0.586**, mAP50 **0.510**.

So the honest statements are: **stable detection (>= 0.85) from 64 px upward**, roughly 0.40-0.60 across
12-32 px, and nothing reliable below 6 px. The section 4 figure of "stable >= 24 px" came from the
numpy-rendered controlled curve and is **not** the deployment figure; this is.

## 5. Frozen known limitations

1. **No discrimination on cluttered scenes with small targets.** On 33 hard scenes a 10 px target is
   found 33% of the time while the same scenes with nothing present yield 0.97 boxes each. A detection
   there carries almost no information. Downstream must not trust a positive on cluttered input.
2. **Nothing below about 8 px.** 1-3 px measures 0.000 across every model trained. Below 4 px the frozen
   sets do not even contain a detectable target.
3. **The controlled matrix and the challenge set are numpy-rendered.** This release was trained toward
   photographic appearance, so its scores on those sets UNDERSTATE its capability on photographic input
   and must not be used as the deployment figure. This is a property of the test sets, not an excuse:
   they should be re-rendered with Isaac Sim for v2.
4. **The real-photograph denominator is 6 distinct images behind 10 rows**, one of which is a plastic
   shuttle rather than a feathered one. The 0.600 recall is therefore a small-sample result.
5. **No real training photographs exist.** Every gain in the real domain came from improving a renderer,
   which worked, but the model has never seen a real photograph during training.
6. **Rendering fidelity is uncalibrated.** `AMBIENT`, `KEY` and the Isaac Sim lighting were chosen for
   plausibility, never fitted to a real camera. There is no guarantee the appearance match generalises to
   the deployment camera.
7. **`bg_001` is anomalously hard** for a reason no measurement in this project identified.
- **A Level-1 adjustment was tried against the cluttered-scene defect and REVERTED.** Feeding whole scenes
  instead of 960x960 centre crops as hard negatives - a clean single-variable change - halved real-photo
  recall (0.600 -> 0.300) and collapsed its precision (0.500 -> 0.158) while making false positives
  slightly worse (43 -> 47). `baseline_whole` is kept on disk as a recorded negative result.
- **The challenge set partially overlaps the training pool**: 8 of 66 C1 rows and 4 of 33 C1n rows use a
  scene that also appears among the training negatives, because C1/C1n were built from those scenes before
  they were added to training. The discrimination finding was recomputed on the 29 non-overlapping scenes
  and is unchanged (recall 0.345 against 0.93 boxes per empty scene).

## 6. Frozen interface

Per detection result, the contract offered to the downstream fine-localisation module:

```
timestamp        float   seconds, monotonic source clock
camera_id        str     one value today; the rig is monocular
candidate_rank   int     0 = highest confidence, ascending
bbox             [x1,y1,x2,y2] float pixels, absolute, in the ORIGINAL image frame
confidence       float   0..1, the detector score, not a calibrated probability
valid            bool    False means the frame could not be evaluated (see below)
```

Conventions, stated because the spec requires them explicit:

- **pixel origin**: top-left, x right, y down. `x2 > x1`, `y2 > y1`; boxes are half-open in the sense
  that `x2 - x1` is the width in pixels.
- **coordinates are in the original received image**, not in the 640-letterboxed network input. The
  detector undoes its own letterbox before returning.
- **ranking** is by descending confidence, and Top-K is 5.
- **empty detection**: `valid=True` with a ZERO-LENGTH candidate list. **An empty result is NOT an
  all-zero bbox.** A consumer that receives no candidates must distinguish "nothing was found" from "a
  box at the origin", and this detector never emits the latter as a stand-in.
- **invalid frame**: `valid=False` with an empty list. Use it when the frame could not be read or the
  capture is corrupted. `valid=False` must not be treated as "no shuttle present".

## 7. Contract with the downstream fine-localisation module

The detector provides coarse candidate regions, confidences and a Top-K ranking. It does NOT provide an
accurate centre, a 3D position, or a trajectory. **The bbox centre is not the shuttle centre**: the
measured centre error and the GT coverage rate are both in the controlled-matrix reports, and on small
targets a whole-pixel bbox quantisation places the centre several percent of the object width away.
Downstream must refine rather than assume.

## 8. Regression test freeze

```bash
# re-run after ANY code change, retrain, or deployment-environment change
bash experiments/shuttle_detection/05_retrain_and_retest.sh \
  <new-run-name> configs/shuttle_detection/baseline_cached.yaml manifest_train_isaac2.csv
# or, to re-measure a FROZEN weight without retraining:
python scripts/shuttle_detection/evaluate_controlled.py --weights <frozen.pt> \
  --manifest outputs/shuttle_capability/controlled_capability/manifest.csv \
  --images   outputs/shuttle_capability/controlled_capability \
  --out <out-dir> --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0
```

Expected metrics for this release, with the tolerance that matters: the frozen sets are deterministic, so
**a re-measurement of the same weights on the same manifests must reproduce these numbers EXACTLY**, not
approximately. Any drift means the code, the data or the environment changed.

| Metric | Expected |
|---|---|
| real photographs (16 rows) | recall 0.600, precision 0.500, mAP50 0.505 |
| 30 shuttle-free scenes | 43 boxes, 8 clean |
| frozen P3 (160) | precision 0.196, recall 0.438, mAP50 0.253 |
| controlled matrix (2070) | recall 0.426, mAP50 0.351 |
| CHALLENGE C5 | recall 1.000 |
| appearance-aligned ladder (237 rows, holdout renders) | precision 0.272, recall 0.586, mAP50 0.510 |
| appearance-aligned ladder, per rung | 4 px 0.050, 8 px 0.300, 16 px 0.500, 32 px 0.600, 64 px 0.850, 128 px 0.900, 512 px 1.000, 1024 px 1.000 |

The appearance-aligned ladder was validated against a first version built from frames the training pool had
also used: the two agreed (recall 0.586 on fresh renders against 0.564 on reused ones), so the numbers above
are the holdout ones and the reuse did not inflate them. The holdout frames are in
`outputs/shuttle_capability/isaac_pool_holdout` and the ladder manifest in `isaac_ladder_holdout`.

## 8b. Regression verification, actually run rather than asserted

The expectations in section 8 were verified by re-measuring the frozen weights on the frozen manifests,
not by restating the numbers they were taken from:

| Quantity | Expected | Re-measured |
|---|---|---|
| weight sha256 | `2530af3515983e27dd088fcabd2c80ae575eeb28adc6ee5ce3e7dc97b171b9b4` | identical |
| frozen P3 precision / recall / mAP50 | 0.196 / 0.438 / 0.253 | **0.195531 / 0.4375 / 0.252826** |
| real photographs precision / recall / mAP50 | 0.500 / 0.600 / 0.505 | **0.500 / 0.600 / 0.50495** |
| challenge overall precision / recall / mAP50 | 0.278 / 0.335 / 0.274 | **0.277778 / 0.335052 / 0.273676** |
| CHALLENGE C5 (>32 px) recall | 1.000 | **1.000** |
| CHALLENGE C1 (8-12 px) recall | 0.333 | **0.333333** |

Every value reproduces exactly, which is the property the frozen sets are supposed to have and the
reason section 8 sets an exact-match tolerance rather than a band. **A future run that drifts on these
numbers has changed the code, the data or the environment, and that is the signal this regresses for.**

## 9. What is deliberately NOT frozen, and why

The tests themselves must not change under a frozen release, but three of them carry known defects that
v2 should fix rather than inherit silently: the controlled matrix and the challenge set are numpy-rendered
and therefore measure the wrong appearance; the real-photograph set contains four byte-identical
duplicate rows; and the FIXED_CORE_TEST near-field ladder still stops at 1023 px while the real positives
run to 1578 px. Freezing them as-is keeps v1 reproducible; recording the defects keeps v2 from repeating
them.

## 10. Why the status is FROZEN_CANDIDATE and not FROZEN

Spec 08 section 12 gates acceptance on thirteen items. Of them, seven cannot be decided in this
repository:

| Gate item | Blocker |
|---|---|
| minimum target size requirement | needs camera intrinsics and the required working distance |
| temporal continuity requirement | needs per-frame video ground truth |
| real-video requirement | same |
| left / right camera requirements | the rig is monocular; no stereo split has ever been defined |
| Top-K, latency, VRAM requirements | measured, but no thresholds exist to compare them against because Spec 05 is `REQUIREMENT_PARTIAL` |

Two gate items are decidable and **now PASS**: the real-image requirement (recall 0.600 against 0.000 for
every earlier model) and the challenge item (run, with its results recorded above). One is decidable in
part and fails: precision, where 5 of 30 shuttle-free scenes still carry a box above 0.5 confidence.

The tests earlier expected of a monocular 2D detector - left and right camera acceptance - cannot be
satisfied by this project at all, which is a scope gap against the spec rather than a measurement gap.