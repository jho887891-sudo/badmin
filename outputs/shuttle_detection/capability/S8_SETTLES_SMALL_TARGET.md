# The 6-12 px regression was a single-scene artefact — settled by the S8 block design

Date: 2026-09-15  |  Set: 2070 rows, S8 = 8 size buckets x 5 scenes x 8 repeats (commit bf7ca75)
Models: `baseline_synthetic_only` (orig) and `baseline_nearfield` (nf), same protocol throughout.

## 1. The dispute this settles

After the first data return two measurements of the same question disagreed:

| 6-8 px | 8-12 px |
|---|---|
| S1 size curve, ONE background (bg_001), n=40 | 0.225 -> 0.025 (-0.200) | 0.300 -> 0.125 (-0.175) |
| frozen P3 set, 30 backgrounds, n=28/50 | 0.429 -> 0.393 (-0.036) | 0.540 -> 0.560 (+0.020) |

`CONTROLLED_MATRIX_1750.md` published the first column as a regression. The second column then came
from the S8 block design, which runs the same size ladder over 5 scenes x 8 repeats.

## 2. The answer: marginalised over scenes, there is no regression

**Version note.** The first run of this analysis used a controlled set synced BEFORE the S8 implementer
landed its bucketing fix (commit bf7ca75), so its 6-8 bucket held 27 measured rows and 8-12 held 53.
The numbers below are from a re-run against the corrected manifest, verified at 40 rows in every S8
bucket by measured size. The earlier figures are kept in the right-hand column so the correction is
visible rather than quietly overwritten; the conclusion is unchanged.

| bucket | n | orig | nf | delta | (stale-version run) |
|---|---|---|---|---|---|
| `<4 px` | 40 | 0.025 | 0.000 | -0.025 | same |
| `4-6 px` | 40 | 0.275 | 0.250 | -0.025 | same |
| `6-8 px` | **40** | 0.350 | **0.400** | **+0.050** | n=27, +0.074 |
| `8-12 px` | **40** | 0.450 | **0.425** | **-0.025** | n=53, -0.019 |
| `12-16 px` | 40 | 0.575 | 0.625 | +0.050 | same |
| `16-24 px` | 40 | 0.575 | **0.825** | +0.250 | same |
| `24-32 px` | 40 | 0.500 | **0.900** | +0.400 | same |
| `>32 px` | 40 | 0.225 | **0.900** | +0.675 | same |

At 6-8 px the marginalised effect is **positive**, and at 8-12 px it is **-0.025**, inside the noise.
So the small-target cost of the data return was an artefact, and the frozen P3 set - measured across 30
backgrounds - had it right all along.

## 3. Why, and the mechanism is the interesting part

| scene | 6-8 px | 8-12 px | 12-16 px |
|---|---|---|---|
| **bg_001** | **0.00 -> 0.00** | **0.12 -> 0.12** | 0.12 -> 0.38 |
| bg_006 | 0.38 -> 0.12 | 0.38 -> 0.25 | 0.50 -> 0.38 |
| bg_011 | 0.62 -> 0.62 | 0.62 -> 0.62 | 0.75 -> 0.75 |
| bg_019 | 0.50 -> 0.75 | 0.75 -> 0.75 | 1.00 -> 1.00 |
| bg_029 | 0.25 -> 0.50 | 0.38 -> 0.38 | 0.50 -> 0.62 |

**bg_001 is the hardest of the five scenes by a wide margin: at 6-12 px its recall is 0.00-0.12 for
BOTH models.** The S1 size curve was measured entirely on bg_001. Where both models sit near zero,
tiny absolute changes become large relative ones, so S1 reported a dramatic -0.20 where the
marginalised truth is +0.05.

The block design also settles the mechanism question the implementer framed in advance: it predicted
that a regression appearing mainly on **bg_019**, the low-contrast scene, would be a visibility effect.
The drop is in fact on **bg_006**, while bg_019 IMPROVES (0.50 -> 0.75), and the implementer's
stimulus-side check found no scene loses contrast from 6-8 to 8-12 px. So this is not target
visibility. **bg_001 is hard for a reason none of these measurements identifies, and that remains
unexplained rather than explained away.**

## 4. What survives, and it is stronger than before

Every gain is confirmed on the marginalised curve with n=40 per bucket: 16-24 px +0.250, 24-32 px
+0.400 and >32 px +0.675. Overall on the 2070-row set: recall 0.206 -> 0.480, mAP50 0.061 -> 0.251.

The data return is therefore an unambiguous improvement on the axes measured here, with no
demonstrated small-target cost. The false-positive confidence worsening (5 -> 9 scenes carrying a
conf >= 0.5 box on shuttle-free scenes) is NOT addressed by this and still stands.

## 5. The methodological lesson, which cost a published error

A conditioned curve pinned to ONE scene is that scene curve. Reading it as the marginal effect is an
error I made and published, and the fix required a block design - deliberately varying the nuisance
that was pinned - rather than more samples of the same pinned configuration. The S8 implementer said
it plainly when asked to add repeats: "I would rather build it as its own sweep than quietly turn S1
into two variables." It was right, and the cost of not having done so from the start was one wrong
published number and a disputed conclusion that stood for three rounds.

## 6. Reproduction

```bash
python scripts/shuttle_detection/evaluate_controlled.py \
  --weights outputs/shuttle_detection/training/<run>/weights/best.pt \
  --manifest outputs/shuttle_capability/controlled_capability/manifest.csv \
  --images   outputs/shuttle_capability/controlled_capability \
  --out outputs/shuttle_detection/capability/cc2070_<tag> \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0
# then filter predictions to sweep == "S8" (join on file) and group by size bucket, and again by
# background, which is what separates the marginal effect from the single-scene one.
```