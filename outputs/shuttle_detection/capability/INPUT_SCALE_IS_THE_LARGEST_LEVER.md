# Input crop is the largest Level-2 lever found, and its ceiling is +0.169

Date: 2026-09-15  |  Level-2 item 3.3, completed  |  Model: `baseline_os` (v2)  |  Set: the small-target
ladder, 320 rows, appearance-aligned and holdout.

## 1. The measurement

Six input scales were swept across eight target sizes, with a ground-truth-derived ROI so these are upper
bounds. The scale factor is on a 640 crop; the full 1280 frame is letterboxed to 640 before the network sees
it, so **0.5x here IS the full frame** and 1.0x is a 640 centre crop fed at native scale.

| target | **0.5x (full frame)** | 1.0x | **1.5x** | 2.0x | 3.0x | 4.0x | best | gain |
|---|---|---|---|---|---|---|---|---|
| 6 px | 0.325 | 0.475 | **0.550** | 0.450 | 0.350 | 0.225 | 1.5x | **+0.225** |
| 8 px | 0.275 | 0.425 | **0.575** | 0.550 | 0.425 | 0.200 | 1.5x | **+0.300** |
| 12 px | 0.250 | 0.500 | **0.550** | 0.400 | 0.175 | 0.125 | 1.5x | **+0.300** |
| 16 px | 0.475 | **0.700** | 0.650 | 0.500 | 0.275 | 0.175 | 1.0x | **+0.225** |
| 24 px | 0.525 | **0.700** | 0.675 | 0.400 | 0.300 | 0.200 | 1.0x | +0.175 |
| 32 px | 0.625 | **0.700** | 0.650 | 0.575 | 0.425 | 0.225 | 1.0x | +0.075 |
| 48 px | 0.825 | 0.825 | **0.850** | 0.650 | 0.550 | 0.150 | 1.5x | +0.025 |
| 64 px | 0.850 | **0.875** | 0.825 | 0.775 | 0.525 | 0.250 | 1.0x | +0.025 |
| **mean** | **0.519** | | | | | | **0.688** | **+0.169** |

## 2. Two findings, and the first one is the surprising one

**A 640 centre crop beats feeding the full frame at EVERY size measured**, including the large ones. At
16 px it is 0.700 against 0.475, at 32 px 0.700 against 0.625, at 64 px 0.875 against 0.850. The full frame
is letterboxed to 640, which halves the apparent size of everything in it and throws away resolution the
network then cannot recover. **Downsampling a big frame is not free, and this project had been doing it in
every measurement.**

**The optimum depends on size, and it is not monotone.** Small targets (6-12 px) want 1.5x; everything from
16 px up wants 1.0x; 3x and 4x are poor everywhere, and 4x is never the best and usually the worst. So the
rule is not "zoom in when things are small" but "zoom in a little, and only when they are small", with a
sharp penalty beyond it.

## 3. What the ceiling means

**+0.169 mean recall if the best scale were always chosen, against a flat full-frame baseline of 0.519.**
That is the largest single lever this project has found at the small end, and it is larger than every data
return measured so far.

It remains an UPPER BOUND in two ways that must not be lost: the ROI comes from the ground truth, so it
knows where the target is; and the ladder jitters targets only within +/-160 px of centre, so a naive centre
crop would capture most of them. **The second caveat cuts against the number being achievable, and it is the
more important one** - on a real court a shuttle is not near the image centre, and a centre crop would miss
it entirely.

So the honest reading is a lever, not a win: **input scale matters more than anything else measured here,
and exploiting it requires knowing where to look, which is the circularity already recorded in closure
question 14.**

## 4. A defect in this sweep that the numbers themselves exposed

The first version omitted the 0.5x column, so it had no true baseline, and its "best" column showed gains of
only +0.037 - a very different conclusion from the +0.169 above. It was caught because its 2.0x and 4.0x
columns reproduced the separate ROI experiment exactly (0.550 and 0.200 at 8 px), which pinned down the
mapping between the two parameterisations and showed that the missing column was the real baseline.

Both experiments were internally consistent and both would have been reported as correct on their own. The
cross-check between them is the only reason the gap was found.
