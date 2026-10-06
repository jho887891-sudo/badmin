# Resolution-response diagnostic V1 (zero training) - report

> **STATUS: BLOCKED on the network as of 2026-10-06.** The frozen V2 checkpoint is only partially local (9.5 MB of
> 20.3 MB; two transfers died mid-stream) and the only reachable path is a Tailscale DERP relay at ~1 KB/s with
> 340-820 ms RTT and repeated disconnects (the public and campus addresses both time out). A per-chunk verified,
> resumable fetch is running in the background. Nothing in this file is a result yet - it fixes the protocol and
> the pre-registered reading so that the numbers cannot be reinterpreted later.

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
| tool | `tools/analyze_resolution_response.py` (8 tests + 1 artifact test) |

## Coverage caveat that must be reported with the result

On this evaluation set the tiny population is dominated by our own synthetic renders, so the diagnostic is mostly
a statement about rendered tiny objects and only partly about real ETH ones:

| source | <4 | 4-6 | 6-8 | total | images on the remote? |
|---|---|---|---|---|---|
| eth_main (real ETH frames) | 1 | 0 | 19 | 20 | yes, already there |
| synthetic (our renders) | 23 | 37 | 25 | 85 | **no, local only (24.9 MB)** |
| strict sub-stride (net < 8 px) | 42 synthetic + 1 eth_main | | | 43 | mixed |

## Pre-registered decision (three branches, frozen)

| observation at 1280/1536 | branch | next experiment |
|---|---|---|
| the <4 and 4-6 buckets start producing candidates (weak-candidate count rises by >= 3, IoU moves off 0) | `HIGHER_RESOLUTION_TRAINING` | higher-resolution training |
| only the 6-8 bucket responds while the sub-stride buckets stay at ~0 | `P2_PLUS_MID_BUCKET_WEIGHTING` | P2 plus a separate weighting strategy for 6-8 |
| ref: sub-stride buckets still produce essentially no response | `P2_STRIDE_4` | P2 / stride 4 |

Implementation: `compute_branch` in the tool applies exactly this table to the measured weak-candidate counts and
writes `resolution_response_v1_decision.json`; the report cannot choose the branch after seeing the numbers.

## Expected tables to fill

1. per resolution: `<4 / 4-6 / 6-8` GT, TP, recall, weak-candidate counts at IoU 0.1/0.3/0.5, median and max best IoU;
2. strict sub-stride subset (43 GT) at each resolution - the decisive table;
3. latency per resolution (ms/image) and the relative cost versus 1024;
4. the branch and its evidence.

## What would make this diagnostic inconclusive

- if the weak-candidate counts move by 1-2 only, the 43-GT sub-stride sample is too small to separate the branches;
- if 1536 cannot run on the available card (the local 4060 has 8 GB and the A6000 48 GB, so this applies only if the
  A6000 route is used with other tenants present);
- if the synthetic dominance above is not stated alongside the conclusion.
