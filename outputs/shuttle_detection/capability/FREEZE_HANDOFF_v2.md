# DETECTOR_RELEASE_ID: `shuttle-detector-v2-2026-09-15`

Supersedes `shuttle-detector-v1-2026-09-15` (`FREEZE_HANDOFF_v1.md`), which is kept unmodified because spec
09 section 11 forbids quietly overwriting a frozen version. **Status: FROZEN_CANDIDATE**, for the reasons in
section 10.

## 1. What changed from v1, and it is not data

**One change: the 86 hard negatives declare `repeat=3` in the training manifest.** They carry 28% of the
training entries instead of 7.8%. No images were added, removed or altered.

| | v1 (`baseline_isaac2`) | **v2 (`baseline_os`)** |
|---|---|---|
| **real photographs recall / precision / mAP50** | 0.600 / 0.500 / 0.505 | **0.800 / 0.800 / 0.769** |
| controlled matrix precision / recall / mAP50 | 0.240 / 0.426 / 0.351 | **0.350** / 0.429 / 0.343 |
| frozen P3 precision / mAP50 | 0.196 / 0.253 | **0.226 / 0.284** |
| appearance ladder precision / recall | 0.272 / **0.586** | **0.389** / 0.553 |
| CHALLENGE precision / recall / mAP50 | 0.278 / **0.335** / 0.274 | **0.401** / 0.325 / **0.287** |
| false positives, 30 shuttle-free scenes | 43, 8/30 clean | **36, 10/30 clean** |
| worst false-positive confidence | **0.669** | 0.724 |
| validation mAP50 | **0.570** | 0.529 |

Precision improves on every set, real-photograph recall rises by 0.200, false positives fall by 16%, and the
cost is about 0.03 of recall on the synthetic sets. **v2 is the frozen candidate on those grounds.**

The reason this is worth its own release rather than a footnote: the three previous real-domain improvements
all came from changing the DATA. This one changes only how the existing images are weighted, which means the
remainder - false positives on clutter and weak discrimination - was at least partly a training-balance
problem rather than the data problem it had been called.

## 2. Frozen model

| Item | Value |
|---|---|
| architecture | YOLO26s, single-stage |
| nc / class 0 | **1** / `shuttlecock` |
| weights | `weights/best.pt`, 20,317,957 bytes |
| **weight sha256** | `61c491206904bba66ee866a8d5b7a45b4b10668c87c07804e2b58e147d74c963` |
| weights/last.pt | `d3fde13e37d1c7a4173b06dbad1a427d42fdc92449590550eadf54355f1315cf` |
| resolved_config.json | `f66f3bebbfa9b440df1eb57597726f1987a66ee9853952c6a9041037b806cdac` |
| args.yaml | `12869988da000ac8236fcacc78c8bee3e298135e5ca8f117c162807f5fe3f009` |
| run_metadata.json | `9d399f098cc674a2d809e681c3eb261b72e7a7657d0ea8e6205ad473b979614a` |
| code commit | `a4e1a5d95c4ce19038dea55a7f21cf78c1855911` |

## 3. Frozen data version

| Split | Rows | sha256 |
|---|---|---|
| TRAIN (`manifest_train_os.csv`) | 1108 rows, **1280 expanded entries** | `8e35c116a05f7198a1d3d50a143b0e58d36c9052a66de49a9e6167c6932dd153` |
| VAL | 144 | `71b0e5f340dc3c9fb3e8cc7a2de2ed44` (unchanged since the original baseline) |

The training manifest is verified rather than assumed: its `repeat` column holds `{1: 1022 rows, 3: 86 rows}`,
which expands to exactly the 1280 entries the dataset builder wrote to `train.txt`, and the local copy hashes
identically to the one on the training host. The test-set manifests are unchanged from v1 - the controlled
matrix `4eaf056f9d8d4375`, frozen P3 `2b7728ffe1c811f6`, real photographs `99a268e7a1c08d2f`, challenge
`996ebd5a8dd96e2c`, appearance ladder `isaac_ladder_holdout`.

## 4. Frozen inference configuration

Unchanged from v1: imgsz 640, confidence 0.05, NMS IoU 0.7 (Ultralytics default), match IoU 0.5, Top-K 5,
letterbox to 640 RGB with ImageNet normalisation, CUDA device 0. **Confidence 0.05 remains frozen for
reproducibility and is still not the recommended deployment value** - see v1 section 6b, whose operating
point analysis applies to v2 as well and should be re-run before choosing one.

## 5. Capability report

| Question | v2 answer |
|---|---|
| real photographs | **recall 0.800, precision 0.800, mAP50 0.769** - the best of any model trained here, on 6 distinct images behind 10 rows |
| stable detection range | **>= 64 px** at 0.85-1.00 on the appearance-aligned ladder; 0.55 at 32 px, 0.40 at 12-16 px |
| below which unreliable | **< 6 px**, where every model trained here measures 0.00-0.15 |
| historical failure band, 1100-1524 px | recall 1.000 |
| hardest pose | **still unknown in the trained appearance** - the Isaac pool has no true side view |
| cluttered-scene discrimination | improved: CHALLENGE precision 0.278 -> 0.401. Not resolved, and threshold-dependent |
| end-to-end latency | median **51.2 ms** at imgsz 640 on the A6000, covering preprocessing through Top-K |
| peak inference VRAM | **108 MB** |
| longest miss streak, left/right camera | **NOT MEASURABLE** - see section 10 |

## 6. Known limitations

Carried from v1 unless stated: no discrimination below about conf 0.40 on cluttered scenes (improved but
unresolved); nothing below 6 px; the controlled matrix, challenge set and pose sweeps are numpy-rendered and
understate the model, so the appearance-aligned ladder is the deployment figure; the real-photograph set is 6
distinct images behind 10 rows, one a plastic shuttle; **no real training photographs exist**; rendering
appearance is plausible but never calibrated against a real camera; `bg_001` is anomalously hard for an
unidentified reason; the Isaac training pool contains **no true side view**, so pose behaviour in the trained
appearance is unmeasured; and worst false-positive confidence rose slightly from 0.669 to 0.724.

## 7. Regression expectations, verified

| Quantity | Expected |
|---|---|
| weight sha256 | `61c491206904bba66ee866a8d5b7a45b4b10668c87c07804e2b58e147d74c963` |
| real photographs precision / recall / mAP50 | 0.800 / 0.800 / 0.768977 |
| frozen P3 precision / recall / mAP50 | 0.226027 / 0.4125 / 0.284195 |
| appearance ladder precision / recall / mAP50 | 0.388724 / 0.552743 / 0.502934 |
| challenge precision / recall / mAP50 | 0.401274 / 0.324742 / 0.286934 |
| controlled matrix precision / recall / mAP50 | 0.349901 / 0.428502 / 0.342986 |
| false positives, 30 shuttle-free scenes | 36 boxes, 10/30 clean, worst confidence 0.724 |

All four re-measured here reproduced their earlier values exactly, and the frozen sets are deterministic, so
the tolerance is exact match rather than a band. The same regression command as v1 applies, with
`manifest_train_os.csv` as the training manifest.

## 8. Interface

Unchanged from v1 section 6, including the two semantics the spec insists on: an empty result is
`valid=True` with a zero-length candidate list and is **not** an all-zero bbox, and `valid=False` means the
frame could not be evaluated and must not be read as "no shuttle present". Section 6b of v1 also still
applies and is the more important reading for a consumer: confidence 0.05 is not a deployment value.

## 9. Why FROZEN_CANDIDATE and not FROZEN

Unchanged from v1 section 10: seven of the thirteen gate items in spec 08 section 12 cannot be decided in
this repository because the camera intrinsics, image resolution, stereo baseline, required working distance,
whole-chain latency budget and deployment GPU do not exist here.

What changed since v1 is that the real-image gate item now passes with more margin (recall 0.800 against the
0.600 of v1), and the challenge item has been run and improved.
