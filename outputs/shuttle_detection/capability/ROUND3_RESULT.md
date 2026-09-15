# Round-3 data return: the synthetic ceiling moved past 1023 px, false positives fell 29%,
# and the real photographs are STILL at zero — which is the important result

Model: `baseline_round3`, 100 epochs, trained on 673 samples (540 positives, 133 negatives) with the
training ceiling extended from 283.94 px to **807.43 px**, plus 33 hard negatives selected for the
measured false-positive character. Config: the declared baseline with `cache: true` and nothing else
(verified programmatically: the two YAMLs differ in exactly one field).

## 1. The size rule is CONFIRMED, and it moved the ceiling exactly as predicted

The S7 near-field ladder, rung by rung:

| rung | baseline_nearfield | **baseline_round3** |
|---|---|---|
| 64 px | 0.975 | 0.900 |
| 96 px | 1.000 | 0.900 |
| 127 px | 0.975 | 0.925 |
| 192 px | 1.000 | 1.000 |
| 256 px | 1.000 | 1.000 |
| 384 px | 1.000 | 1.000 |
| 512 px | 1.000 | 1.000 |
| **768 px** | **0.000** | **1.000** |
| **1023 px** | **0.000** | **1.000** |

The rule derived two rounds ago - the detection ceiling sits at roughly twice the training ceiling -
predicted that a 807 px training maximum would reach about 1600 px. The ladder now detects 1023 px
perfectly where it previously detected nothing. The rule held.

## 2. And the real photographs are STILL at 0.000

This is the decisive result of the round. **16 verified real positives: recall 0.000, exactly as
before.** Meanwhile a 1023 px SYNTHETIC target is detected 100% of the time.

So the real-domain failure is **not** a size-coverage problem, and the size hypothesis that drove the
last two data returns is now excluded. Something else separates the real photographs from the renders,
and the two candidates that this measurement CANNOT separate are:

1. **real-domain appearance** - the renders are a plausible but uncalibrated approximation, and
   `AMBIENT`/`KEY` have never been fitted to a real camera;
2. **perspective** - the S7 ladder is rendered with a long lens (fx=11200, distances 0.78-12.4 m) so
   its perspective is MILD, whereas a real 838-1578 px positive is a genuine close-up with strong
   perspective. The implementer flagged this asymmetry when it built the ladder, and it is now the
   live explanation rather than a footnote.

Both are DATA problems, not architecture problems. The binding constraint is therefore **real-domain
training data**, which this project has never had - every real image it owns is frozen test data.

## 3. What the round bought and what it cost

| axis | orig | nearfield | round3 |
|---|---|---|---|
| 768 / 1023 px ladder | 0.000 | 0.000 | **1.000** |
| controlled matrix recall | 0.206 | 0.480 | **0.504** |
| controlled matrix mAP50 | 0.061 | 0.251 | **0.350** |
| frozen P3 (160 small targets) recall | 0.4437 | 0.4375 | 0.3938 |
| S8 marginalised 16-24 px | - | 0.825 | **0.525** |
| real photographs (16) recall | 0.000 | 0.000 | **0.000** |
| shuttle-free scenes: total FP | 63 | 65 | **46** |
| shuttle-free scenes: worst FP confidence | 0.833 | 0.931 | **0.874** |

**The hard-negative pool worked: total false positives fell 29% (65 -> 46) and the worst false-positive
confidence fell from 0.931 to 0.874.** That is the first measured improvement on the precision problem,
and it came from 33 images selected by measured model response rather than by inspection.

**The cost is small-target recall.** The S8 marginalised curve shows drops at every small bucket, most
severely 16-24 px (0.825 -> 0.525, n=40, well outside the 15.5% half-width). Extending the training
range pulled the model away from the smallest targets. Whether that is acceptable depends on the
deployment working distance, which is exactly the number Plan 5 is waiting on.

## 4. What this means for the level decision

Spec 06 section 4 requires the data explanations to be excluded before any structural conclusion. The
chain so far:

- "the model cannot see large targets" -> true, and FIXED by data (round 1);
- "the model cannot see the real size range" -> the size rule held, and FIXING IT did not help (round 3);
- therefore the remaining gap is real-domain data, not size and not architecture.

Level 1/2 (data) remains the indicated adjustment. The specific data is now well defined: real
photographs of shuttlecocks, including genuine close-ups, in the size range already measured, and the
project has none. Plan 5 remains blocked on camera intrinsics and working distance; this round adds a
third item to that list, because without a real close-up nobody can separate appearance from
perspective.
