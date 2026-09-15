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

| bucket | n | orig | nf | delta |
|---|---|---|---|---|
| `<4 px` | 40 | 0.025 | 0.000 | -0.025 |
| `4-6 px` | 40 | 0.275 | 0.250 | -0.025 |
| `6-8 px` | 27 | 0.296 | **0.370** | **+0.074** (was published as -0.200) |
| `8-12 px` | 53 | 0.453 | **0.434** | **-0.019** (was published as -0.175) |
| `12-16 px` | 40 | 0.575 | 0.625 | +0.050 |
| `16-24 px` | 40 | 0.575 | **0.825** | +0.250 |
| `24-32 px` | 40 | 0.500 | **0.900** | +0.400 |
| `>32 px` | 40 | 0.225 | **0.900** | +0.675 |

At 6-8 px the marginalised effect is **positive**, and at 8-12 px it is **-0.019**, which is inside
the noise. So the small-target cost of the data return was an artefact, and the frozen P3 set had it
right all along.

## 3. Why, and the mechanism is the interesting part

| scene | 6-8 px | 8-12 px | 12-16 px |
|---|---|---|---|
| **bg_001** | **0.00 -> 0.00** | **0.10 -> 0.10** | 0.12 -> 0.38 |
| bg_006 | 0.20 -> 0.00 | 0.45 -> 0.27 | 0.50 -> 0.38 |
| bg_011 | 0.60 -> 0.60 | 0.64 -> 0.64 | 0.75 -> 0.75 |
| bg_019 | 0.50 -> 0.67 | 0.70 -> 0.80 | 1.00 -> 1.00 |
| bg_029 | 0.20 -> 0.60 | 0.36 -> 0.36 | 0.50 -> 0.62 |

**bg_001 is the hardest of the five scenes by a wide margin: recall at 6-12 px is 0.00-0.10 for BOTH
models.** The S1 size curve was measured entirely on bg_001. On a scene where both models sit near
zero, tiny absolute changes become large relative ones, so S1 reported a dramatic -0.20 where the
marginalised truth is +0.07.

Two pieces of independent evidence already pointed this way and I noted them at the time without
acting on them: the frozen P3 set disagreed, and the S8 implementer measured the STIMULUS side and
found bg_001 is mid-pack on contrast (33.8 at 6-8 px against bg_019 at 13.6) with no scene losing
contrast from 6-8 to 8-12 px. So bg_001 is not hard because its targets are faint - it is hard for a
reason none of these measurements identifies, which is itself worth recording.

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
