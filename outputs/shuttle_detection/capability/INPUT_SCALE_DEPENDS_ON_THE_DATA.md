# Input scale is a lever whose SIGN depends on the data, and the synthetic gain does not transfer

Date: 2026-09-15  |  Corrections to `INPUT_SCALE_IS_THE_LARGEST_LEVER.md`, which was one round too
enthusiastic. **The lever is real on the synthetic sets and it does not transfer to real photographs.**

## 1. What the wider test shows

The previous round established, on the appearance-aligned ladders, that a 640 centre crop beats feeding the
full frame. That raised the obvious question - are the published numbers limited by input scale? - which has
a clean test: keep the set, keep the model, and change only `imgsz`.

| Set | imgsz 640 | imgsz 1280 | Winner |
|---|---|---|---|
| appearance ladder recall / mAP50 | 0.553 / 0.503 | **0.658 / 0.563** | 1280 |
| small-target ladder recall / mAP50 | 0.519 / 0.442 | **0.684 / 0.605** | 1280 |
| frozen P3 recall / mAP50 | 0.413 / 0.284 | **0.650 / 0.556** | 1280 |
| **real photographs recall / precision** | **0.800 / 0.800** | 0.600 / 0.273 | **640** |
| **false positives, 30 clean scenes** | **36 boxes, 10 clean** | 85 boxes, 4 clean | **640** |
| appearance ladder precision | **0.389** | 0.186 | 640 |

Per rung on the small ladder, 1280 is better at **every** size, and by the most at the small end: 12 px
0.250 -> 0.550, 8 px 0.275 -> 0.525, 16 px 0.475 -> 0.650, 64 px 0.850 -> 0.875.

## 2. The correction

The previous round concluded that the published capability figures were limited by input scale and that
input scale is the largest Level-2 lever found. **Half of that survives and half does not.**

- **Survives:** at a fixed scale, cropping beats letterboxing, and the adaptive-zoom ceiling on the
  synthetic ladders is +0.169. Downsampling a large frame really does cost.
- **Does not survive:** "raise the input resolution" is NOT a general improvement. On the two real-domain
  measurements it is a large regression - real-photograph recall 0.800 -> 0.600 with precision 0.800 ->
  0.273, and false positives 36 -> 85 with clean scenes 10 -> 4.

**So the honest statement is that the optimal scale is data-dependent and there is no single scale that
wins everywhere.** The synthetic ladders were generated at 1280 and are therefore 1:1 at `imgsz 1280` and
downsampled 2x at 640, which is a plausible reason they favour it; the real photographs are natively 1920 and
get 0.67x at 1280 against 0.33x at 640, so the "upscaling" explanation does not apply to them and the cause
of their regression is not established here.

## 3. Which scale the frozen release should use

**640, and that is now a better-supported choice than before.** It is what every published number was taken
at, it is reproducible, and on the two measurements that reflect deployment - real photographs and false
positives on real scenes - it is clearly the better scale. Raising it is a change that would need to be
justified on real data, and the real data says the opposite.

What this round does change is the STATUS of the frozen numbers: they are not scale-limited in the sense of
being improvable for free. They are what this model achieves at the scale where it works best on real input.

## 4. The recurring lesson, now with a third instance

Three times now, a change that improved the synthetic sets regressed the real ones: whole-scene negatives,
side views in the trained appearance, and input resolution. **Every one of them looked like a win on the
controlled matrix or the appearance ladder.** The pattern is consistent enough to state as a rule this
project should follow: **a synthetic-set gain is a hypothesis, not a result, until it is checked on real
scenes.**

That rule is the reason this round exists - the previous round checked the synthetic gain and stopped.
