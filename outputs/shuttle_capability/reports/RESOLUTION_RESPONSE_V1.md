# Resolution-response diagnostic V1 (zero training) - report

> **STATUS 2026-10-06: DONE - the pre-registered rule returns `P2_STRIDE_4`.** Two runs, same frozen weights, no
> training. Step 2a inferred the 20 real `eth_main` tiny GT on the remote A6000 (the checkpoint was already there
> with a byte-identical sha256). Step 2b completed the full 105-GT coverage on the local RTX 4060 after the checkpoint
> arrived as 20 verified chunks (20,344,133 B, sha256
> `3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7`, identical to the remote file). The frozen
> `compute_branch` then chose the branch from the measured weak-candidate counts - see "Step 2b" for the decision and
> its two coverage caveats: the `4-6` bucket is 100 % synthetic renders, and the real-object sub-stride evidence rests
> on a single GT. The protocol and the pre-registered reading were fixed before the numbers and are kept verbatim
> below.

## Question

With the V2 weights completely untouched, does raising the inference resolution make a strictly sub-stride tiny
object (equiv_size_640 < 8, i.e. < 12.8 px in the network frame) produce any spatial response at all?

## Protocol (frozen before the run)

| item | value |
|---|---|
| checkpoint | `eth_real_hardneg_v2_best` (epoch 21, sha256 `3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7`), weights frozen |
| resolutions | 1024, 1280, 1536 |
| images | the `val|eth_unseen` images that contain at least one <8 px GT |
| operating point | conf 0.25, greedy IoU 0.5 (the canonical evaluator convention) |
| weak-response probe | best IoU of any detection at conf >= 0.01, marks at IoU 0.1 / 0.3 / 0.5 |
| preprocessing | ultralytics letterbox to the square resolution, NMS iou 0.7, max_det 300 (evaluator settings) |
| latency | median of repeated single-image forward passes, plus the measured pass time per image |
| tool (full 105-GT coverage) | `tools/analyze_resolution_response.py` (8 tests + 1 artifact test), needs the checkpoint locally |
| tool (real-ETH part, ran on the remote) | `tools/remote/remote_tiny_resolution.py` + `tools/analyze_remote_tiny_resolution.py` (5 tests) |

## Coverage caveat that must be reported with the result

On this evaluation set the tiny population is dominated by our own synthetic renders, so the diagnostic is mostly
a statement about rendered tiny objects and only partly about real ETH ones:

| source | <4 | 4-6 | 6-8 | total | images on the remote? |
|---|---|---|---|---|---|
| eth_main (real ETH frames) | 1 | 0 | 19 | 20 | yes, already there |
| synthetic (our renders) | 23 | 37 | 25 | 85 | **no, local only (24.9 MB)** |
| strict sub-stride (net < 8 px) | 42 synthetic + 1 eth_main | | | 43 | mixed |

## Step 2a - real-ETH tiny part on the A6000 (zero training, weights untouched)

Run 2026-10-06T17:04:38Z-17:05:0xZ. The 20 real `eth_main` tiny GT of `val|eth_unseen` (all of the real tiny
population of this evaluation set: 19 in 6-8, 1 in `<4`, none in 4-6) were re-inferred at three input sizes with the
frozen V2 weights.

### Provenance (every number below is reproducible from these)

| item | value |
|---|---|
| host / runtime | RTX A6000 49,140 MiB; `/home/T7/public/miniconda3/bin/python` 3.13.2, torch 2.10.0+cu128, ultralytics 8.4.150, PIL 12.3.0, CUDA 12.8 |
| checkpoint | `/home/dgut/.dsh-bench/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt`, 20,344,133 B, sha256 `3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7` (re-verified on the remote before the run) |
| images / GT | 20/20 located under `/home/T7/dgut/robot_sim/eth_shuttle_detection/<loc>_hard/images/train/`, labels `labels/train/*.txt` (38 B, one box each); 20 tiny GT decoded, 0 missing |
| part list | `tools/remote/eth_tiny_part.csv` 2,715 B sha256 `26b2cf57954bbfa0c52f98d7bdeb5bd61ada1b5d0b7bc081d965f5c1b3314e2b` |
| probe | `tools/remote/remote_tiny_resolution.py` sha256 `83ae2a5cc3602681cd9dcee3e808c975a16153518a3a5d227c4c3ed651c485d5` |
| GT decoder | `tools/eval_yolo26_v1.py` sha256 `3dbf00264ba16c9c411fde6fb4fd92bfd85495936b6c2eeebd4f0370d77ad955` (uploaded on purpose: the remote revision is a 2025-09-29 older copy) |
| raw result | remote `/home/dgut/.dsh-bench/resdiag/tiny_resolution.json` 63,332 B sha256 `6197fb1c2efadbc10782677194679068e877f08f329d4aa99990a598fc787fef`, committed verbatim as `outputs/shuttle_capability/metrics/resolution_response_remote_v1_probe.json` (same sha256, the file the analyzer reads) |
| settings | letterbox square `imgsz`, NMS iou 0.7, max_det 300, candidate floor conf 0.01, operating point conf 0.25 with greedy IoU 0.5 |
| post-processing | `tools/analyze_remote_tiny_resolution.py` (5 tests) -> `outputs/shuttle_capability/metrics/resolution_response_remote_v1_{gt,summary}.csv` |

### Table 1 - response by resolution (20 real tiny GT, operation IoU 0.5)

| imgsz | candidates >=0.01 | candidates >=0.25 | hits @conf .25 | @.10 | @.01 | GT bestIoU >=0.5 | >=0.3 | <0.01 | images with 0 candidates |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 23 | 6 | 6 | 7 | 7 | 7 | 8 | 12 | 8 |
| 1280 | 22 | 13 | 8 | 8 | 9 | 9 | 9 | 11 | 7 |
| 1536 | 15 | 8 | 6 | 7 | 7 | 7 | 7 | 13 | 8 |

By bucket:

| bucket | n | 1024 | 1280 | 1536 |
|---|---|---|---|---|
| `<4` | 1 | 0/1, no candidate anywhere (bestIoU 0.000) | 1/1, IoU 0.665 @conf 0.325 | 1/1, IoU 0.606 @conf 0.600 |
| `4-6` | 0 | - | - | - (empty bucket: absent, not negative) |
| `6-8` | 19 | 6/19 (weak 7/19) | 7/19 (weak 8/19) | 5/19 (weak 6/19) |

### Table 2 - latency (median of 30 single-image passes, batch 1)

| imgsz | end-to-end ms | preprocess ms | inference ms | postprocess ms | sequential img/s | peak VRAM |
|---|---|---|---|---|---|---|
| 1024 | 39.69 | 4.68 | 10.59 | 1.57 | 23.67 | 203 MiB |
| 1280 | 46.26 (+16.6 %) | 6.65 | 11.19 (+5.7 %) | 1.65 | 22.72 | 266 MiB |
| 1536 | 51.77 (+30.4 %) | 9.73 | 12.38 (+16.9 %) | 0.86 | 21.85 | 364 MiB |

The network itself is almost free here: at batch 1 the forward pass grows only 5.7 % / 16.9 % while the input pixels
grow 56 % / 125 %, so the A6000 is not compute-bound at these sizes. Most of the end-to-end time is host side
(~23 ms unaccounted = ~1 MB JPEG decode plus per-call overhead) and pre-processing grows fastest (+108 % at 1536).
Peak VRAM at batch 1 grows 203 -> 364 MiB, i.e. the 8 GB local card is not the constraint for this diagnostic.
The earlier published throughput (126.9 img/s for `eth_only_v1_best`) used a different procedure (streamed list) and
must not be compared against the sequential 21.9-23.7 img/s column above.

### Table 3 - operating-point accounting (conf >= 0.25)

| imgsz | confident candidates | TP | unmatched | GT with bestIoU < 0.3 | images with 0 confident candidate |
|---|---|---|---|---|---|
| 1024 | 6 | 6 | 0 | 12 | 14 |
| 1280 | 13 | 8 | 5 | 11 | 8 |
| 1536 | 8 | 6 | 2 | 13 | 13 |

The surplus at 1280 is **not** duplicate boxes on one object: only 1 of 20 images produced 2 boxes at conf >= 0.25
and only one of those two overlaps its GT, so the extra confident output is a new (unmatched under IoU 0.5) box.

### Triage - "not looked at" versus "looked at, no response"

| imgsz | GT with bestIoU < 0.01 | its frame had other candidates | its frame had none at all |
|---|---|---|---|
| 1024 | 12 | 4 | 8 |
| 1280 | 11 | 4 | 7 |
| 1536 | 13 | 5 | 8 |

For 7-8 of the 20 real frames the frozen model emits **no detection anywhere in the image** (conf >= 0.01, max_det
300): that is silence, not a mislocalized box. The remaining 4-5 dead GT sit in frames where the model did respond
elsewhere, i.e. the object is not attended to rather than not representable.

### Cross-machine consistency with the local V2@1024 measurement

Best IoU agrees with the locally stored references for 20/20 GT within 0.02 (max deviation 0.0137), so the boxes are
the same object. Confidence, however, is runtime-sensitive: 7 of 20 shifted up and 1 down, and two GT
(`ml_3_00211` 0.243 -> 0.320, `ml_3_00229` 0.127 -> 0.280) cross the 0.25 operating point, so the single-threshold
count is local 4/20 versus remote 6/20 on identical weights. Consequence adopted here: report the response **curve**
over confidence thresholds, never a lone threshold count.

### Branch read - partial, and explicitly not yet the pre-registered decision

- The pre-registered `HIGHER_RESOLUTION_TRAINING` trigger (weak-candidate count rises by >= 3 and IoU moves off 0 in
  `<4`/`4-6`) is **not met** on this part: total candidates 23 -> 22 -> 15 (it falls) and hits 6 -> 8 -> 6, i.e. +2 at
  1280 and back to 6 at 1536 - not monotone, with 11-13 of 20 GT still at zero response at every input size.
- But the single real `<4` GT in the set (`uetlibergstrasse_1_00041`, equiv_size_640 3.30, ~5.3 px in the 1024
  network frame) went from **no candidate at all** at 1024 to a well-localized, confident hit at 1280 (IoU 0.665,
  conf 0.325) and at 1536 (IoU 0.606, conf 0.600). At 1536 its object is still below one P3 stride (~7.9 px) and it
  is nonetheless found with conf 0.60. n = 1, so this is a hint about the bucket, not a measurement of it.
- The `4-6` bucket is empty in the real tiny part and the synthetic GT that fills `4-6` (37) and `<4` (23) is
  local-only, so the branch decision stays open; naming a branch from this run would be deciding on an unmeasured
  bucket.
- What this run does support: resolution alone does not rescue the bulk of real tiny GT (0.55-0.65 never respond at
  any tested size, and 1536 is worse than 1280), which points at the P2 / stride-4 direction; while for the one truly
  sub-stride object available, a larger input made it detectable **without any weight change**, which is the
  strongest single piece of evidence so far for the resolution/representability reading.

### Resolved

Both gaps listed here (the 85 synthetic tiny images and the local checkpoint) were closed before Step 2b: the render
pool is present in `outputs/shuttle_capability/train_data/val/images` and the frozen V2 checkpoint is local and
sha256-verified. Nothing on this list is outstanding.

## Step 2b - full coverage (105 tiny GT) on the local RTX 4060, and the frozen decision

The synthetic half of the tiny population turned out to be fully present locally, so the diagnostic was completed on
all 105 tiny GT with the same frozen weights (the checkpoint that had been blocking this was finally fetched as 20
verified chunks and assembled to the exact SSOT sha256).

### Provenance

| item | value |
|---|---|
| host / runtime | RTX 4060 Laptop 8 GB; python 3.11.9, torch 2.14.0+cu126, ultralytics 8.4.150, CUDA 12.6 |
| checkpoint | local `_scratch_resdiag/v2_best.pt`, 20,344,133 B, sha256 `3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7` (20/20 verified chunks; scratch copy, not a deliverable) |
| part list | `tools/remote/local_tiny_part.csv` 27,132 B: 105 rows, 20 `eth_main` + 85 synthetic, every image and label verified present before the run |
| probe | `tools/remote/remote_tiny_resolution.py` (extended with absolute-path support so one part list can span the ETH pool on D: and the render pool under `outputs/`) |
| raw result | `outputs/shuttle_capability/metrics/resolution_response_local_v1_probe.json` 303,125 B sha256 `823fae225a2fb4a81cfcd50684f2d5f19f2799cd2b0731cbb9c14368631ba32a` |
| derived | `resolution_response_local_v1_{gt,summary}.csv` by `tools/analyze_remote_tiny_resolution.py --tag local_v1` |
| decision | `resolution_response_v1_decision.json` sha256 `5ab594f4a6029f63216e1403ea9984199f1b9f353e7afcf9c5532473300fd5d8`, written by `tools/derive_resolution_branch.py`, which calls the **frozen** `summarise`/`compute_branch` in `tools/analyze_resolution_response.py` |

### Table 4 - response by resolution (105 tiny GT, operation IoU 0.5)

| imgsz | candidates >=0.01 | candidates >=0.25 | hits @conf .25 | @.10 | @.01 | bestIoU >=0.5 | >=0.3 | <0.01 | images with no candidate at all |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 42 | 7 | 6 | 7 | 8 | 8 | 9 | 96 | 78 |
| 1280 | 40 | 13 | 8 | 8 | 12 | 12 | 12 | 93 | 78 |
| 1536 | 31 | 8 | 6 | 8 | 10 | 10 | 10 | 95 | 78 |

By bucket (hits at IoU 0.5, conf 0.25; "weak" = best IoU >= 0.1):

| bucket | n | 1024 | 1280 | 1536 |
|---|---|---|---|---|
| `<4` | 24 | 0/24 (weak 0) | 1/24 (weak 1) | 1/24 (weak 1) |
| `4-6` | 37 | 0/37 (weak 0) | 0/37 (weak 0) | 0/37 (weak 0) |
| `6-8` | 44 | 6/44 (weak 9) | 7/44 (weak 11) | 5/44 (weak 9) |

By source x bucket:

| source | bucket | n | 1024 | 1280 | 1536 |
|---|---|---|---|---|---|
| eth_main | `<4` | 1 | 0/1 | 1/1, IoU 0.664 | 1/1, IoU 0.606 |
| eth_main | `6-8` | 19 | 6/19 (weak 7) | 7/19 (weak 8) | 5/19 (weak 6) |
| synthetic | `<4` | 23 | 0/23, no candidate anywhere | 0/23, no candidate anywhere | 0/23, no candidate anywhere |
| synthetic | `4-6` | 37 | 0/37, no candidate anywhere | 0/37, no candidate anywhere | 0/37, no candidate anywhere |
| synthetic | `6-8` | 25 | 0/25 (weak 1) | 0/25 (weak 3) | 0/25 (weak 3) |

Two facts that Table 4 does not show on its own:

- **78 of the 105 images emit no detection anywhere in the frame at any of the three input sizes** - the same 78 at
  1024, 1280 and 1536. For those images input scale is not the question; the object is simply not represented. All 23
  synthetic `<4` and all 37 synthetic `4-6` GT sit inside that group.
- **The synthetic tiny GT produce zero true hits in every bucket** (0/85 at IoU 0.5, conf 0.25) with only 1-3 weak
  responses confined to `6-8`. That is a domain-gap statement about our renders, not about real shuttles, and it is
  why the branch below is reported with its coverage attached.

Strict sub-stride subset (net_px < 8 px at the given input, i.e. smaller than one P3 stride): 43 GT at 1024, 24 at
1280, 13 at 1536 - the subset itself shrinks as the input grows. Weak responses (best IoU >= 0.1): 0 / 1 / 1. At 1536,
12 of the 13 GT that are still strictly sub-stride remain completely silent; the one that responds is the real ETH GT
already singled out in Step 2a.

### Table 5 - latency (local RTX 4060, batch 1, median of 30 passes)

| imgsz | end-to-end ms | preprocess ms | inference ms | postprocess ms | sequential img/s | peak VRAM |
|---|---|---|---|---|---|---|
| 1024 | 50.58 | 10.61 | 26.71 | 1.41 | 15.44 | 177 MiB |
| 1280 | 53.61 | 16.33 | 22.23 | 1.45 | 17.26 | 238 MiB |
| 1536 | 64.77 | 23.18 | 26.11 | 1.56 | 13.77 | 364 MiB |

The local inference medians are non-monotone (26.7 -> 22.2 -> 26.1 ms) because this is a laptop dGPU sharing its power
budget; **use the A6000 numbers in Step 2a Table 2 for any cost statement**. Peak VRAM again stays under 400 MiB, so
the 8 GB card was never the limit.

### The pre-registered decision, applied by the frozen code

`tools/derive_resolution_branch.py` reshapes the probe JSON into the row format `summarise()` expects and calls the
`compute_branch()` frozen before any number existed. It compares 1024 with the largest tested input:

    sub-stride (<4 + 4-6) weak candidates, IoU >= 0.1 : 0 -> 1   (trigger needs a gain of >= 3)
    mid bucket (6-8)      weak candidates, IoU >= 0.1 : 9 -> 9   (P2_PLUS_MID needs a gain of >= 3)
    DECISION: P2_STRIDE_4
      "sub-stride objects still produce essentially no response at 1536 (weak candidates 0 -> 1),
       so the binding constraint is the sampling grid itself"

The same rule applied to the real-ETH 20 GT alone returns `P2_STRIDE_4` as well (sub-stride 0 -> 1, mid 8 -> 6), so
the decision does not hinge on the synthetic majority. Both invocations are reproducible; the full-coverage one is the
committed `resolution_response_v1_decision.json` (tag `local_v1`).

### What this decision does and does not say

- It says: for the measured tiny population, enlarging the input by 1.25x/1.5x does not make the sub-stride population
  appear (gain +1 against a pre-registered threshold of +3), while `6-8` responds and is already partly detectable.
  The sampling grid, not the input scale, is the binding constraint - so P2 / stride 4 is the next experiment.
- It does **not** say higher resolution is useless: the single real sub-stride GT in the set goes from no candidate at
  1024 to IoU 0.665 @conf 0.325 at 1280 and IoU 0.606 @conf 0.600 at 1536, still below one stride at 1536. n = 1, so
  the rule's answer stands, but the counter-example is recorded rather than averaged away.
- It does **not** measure real 4-6 px shuttles: the `4-6` bucket is 100 % our synthetic renders and this evaluation
  set contains no real `4-6` tiny GT at all. "4-6 stays silent" is therefore a statement about renders.
- The renders' total silence (0/85 hits) is itself the strongest confound here: most of the sub-stride evidence is
  about synthetic objects that this model never detects in any bucket.

## Pre-registered decision (three branches, frozen)

| observation at 1280/1536 | branch | next experiment |
|---|---|---|
| the <4 and 4-6 buckets start producing candidates (weak-candidate count rises by >= 3, IoU moves off 0) | `HIGHER_RESOLUTION_TRAINING` | higher-resolution training |
| only the 6-8 bucket responds while the sub-stride buckets stay at ~0 | `P2_PLUS_MID_BUCKET_WEIGHTING` | P2 plus a separate weighting strategy for 6-8 |
| ref: sub-stride buckets still produce essentially no response | `P2_STRIDE_4` | P2 / stride 4 |

Implementation: `compute_branch` in the tool applies exactly this table to the measured weak-candidate counts and
writes `resolution_response_v1_decision.json`; the report cannot choose the branch after seeing the numbers.

**Applied outcome (Step 2b):** the code returned `P2_STRIDE_4` - sub-stride (`<4` + `4-6`) weak candidates at
IoU >= 0.1 moved 0 -> 1 between 1024 and 1536 (the trigger required >= 3) and the `6-8` bucket did not gain either
(9 -> 9, so the `P2_PLUS_MID_BUCKET_WEIGHTING` condition also failed). The real-ETH 20-GT subset alone returns the
same branch.

## Expected tables to fill

1. per resolution: `<4 / 4-6 / 6-8` GT, TP, recall, weak-candidate counts at IoU 0.1/0.3/0.5, median and max best IoU
   - **filled**: Step 2a Tables 1-3 for the 20 real GT, Step 2b Table 4 for all 105 GT, both with the full
   confidence-threshold curve and the source x bucket split;
2. strict sub-stride subset at each resolution - the decisive table - **filled**: 43 / 24 / 13 GT at 1024 / 1280 / 1536
   (the subset shrinks as the input grows), weak responses 0 / 1 / 1, in Step 2b;
3. latency per resolution (ms/image) and the relative cost versus 1024 - **filled twice**: Step 2a Table 2 (A6000,
   the numbers to quote) and Step 2b Table 5 (local 4060), both with the preprocess/inference/postprocess split;
4. the branch and its evidence - **filled**: `P2_STRIDE_4` by the frozen `compute_branch`, with the two coverage
   caveats stated in Step 2b.

## What would make this diagnostic inconclusive

- if the weak-candidate counts move by 1-2 only, the 43-GT sub-stride sample is too small to separate the branches;
- if 1536 cannot run on the available card (the local 4060 has 8 GB and the A6000 48 GB, so this applies only if the
  A6000 route is used with other tenants present);
- if the synthetic dominance above is not stated alongside the conclusion.
