# Frozen P3 capability: before vs after the nc=1 baseline

Date: 2026-09-15  |  Model under test: `baseline_synthetic_only` (commit 0358200)
Protocol identical for both runs: frozen 160-image `SYNTHETIC_ON_REAL_BG` set, `imgsz=640`,
`conf>=0.05`, match `IoU>=0.5`, Top-K with K=5, GPU inference on jxxy.

## 1. The headline

| Bucket | n | Recall BEFORE | Recall AFTER | Delta | TopK before | TopK after |
|---|---|---|---|---|---|---|
| `<4 px` | 16 | **0.000** | **0.000** | 0.000 | 0.000 | 0.000 |
| `4-6 px` | 24 | 0.042 | **0.250** | +0.208 | 0.042 | 0.250 |
| `6-8 px` | 28 | 0.071 | **0.429** | +0.357 | 0.036 | 0.393 |
| `8-12 px` | 50 | 0.140 | **0.540** | +0.400 | 0.140 | 0.500 |
| `12-16 px` | 19 | 0.316 | **0.579** | +0.263 | 0.211 | 0.579 |
| `16-24 px` | 21 | 0.095 | **0.667** | +0.571 | 0.000 | 0.667 |
| `24-32 px` | 2 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 |

| Overall | BEFORE | AFTER |
|---|---|---|
| Recall | 0.1188 | **0.4437** (x3.7) |
| Total predictions | 2471 | 681 |
| False positives | 2452 | **610** (x4.0 fewer) |

**Protocol cross-check:** the BEFORE overall recall of 0.1188 reproduces the `0.119` recorded in
`P3_REPORT.md` exactly, so this evaluator reproduces the pre-existing baseline measurement rather
than inventing a new protocol.

## 2. Answers to the questions this measurement exists to answer

1. **Which size range is detected reliably?** None of the buckets on this set reaches a level that
   deserves the word reliable. The best is `16-24 px` at 0.667, then `12-16 px` at 0.579.
2. **Where does it start degrading?** Below 8 px: 0.540 at `8-12 px` falls to 0.429 at `6-8 px`,
   0.250 at `4-6 px`, and 0.000 below 4 px.
3. **Below which size is it unreliable?** **`<4 px` is a hard floor: recall 0.000 in BOTH models,
   0 of 16 detected.** This is not a training deficiency that more epochs would fix; the baseline
   never produced a single matched candidate there.
4. **`24-32 px`: n=2. Not evidence of anything.** It is reported for completeness only.
5. Precision moved the same way: at `8-12 px` the untrained COCO model produced 647 false positives
   for 7 true positives; the trained baseline produces 198 for 27. Both are still far too many.

## 3. What this does NOT establish

- **Not real-image, not real-video.** Every number is on synthetic shuttles composited on real
  backgrounds. The REAL_IMAGE and REAL_VIDEO domains are separate acceptance criteria and remain
  unmeasured with this model.
- **No pose / position / blur / occlusion conditioning.** The P3 manifest carries yaw/pitch/roll but
  no `pose_bucket`, and no blur, occlusion or position-bucket columns at all. The evaluator reports
  those files with zero rows and marks them not-computed rather than writing a zero. Producing those
  curves is exactly why a purpose-built controlled set is being generated.
- **Precision is still poor.** 610 false positives across 160 images is the next thing to attack, not
  a solved problem.

## 4. Reproduction

```bash
IMG=outputs/shuttle_capability/synthetic_on_real_bg/images
MAN=outputs/shuttle_capability/metrics/p3_real_bg_manifest.csv
# before
python scripts/shuttle_detection/evaluate_controlled.py --weights yolo26s.pt \
  --manifest $MAN --images $IMG --out <out> --imgsz 640 --conf 0.05 --expected-class any --top-k 5 --device 0
# after
python scripts/shuttle_detection/evaluate_controlled.py \
  --weights outputs/shuttle_detection/training/baseline_synthetic_only/weights/best.pt \
  --manifest $MAN --images $IMG --out <out> --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0
```

Note: `--expected-class any` for the untrained model is deliberately generous to it - COCO has no
shuttlecock class, so any of its 80 classes that happens to cover the shuttle counts. It still only
reached 0.119.
