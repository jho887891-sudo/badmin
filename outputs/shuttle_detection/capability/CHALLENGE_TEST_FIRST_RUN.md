# CHALLENGE_TEST, first run: the paired design shows there is no discrimination on the hardest scenes

Date: 2026-09-15  |  Models: `baseline_isaac2` (candidate) and `baseline_round3`  |  Set: 227 rows,
built by the challenge-split task and never run until now.

## 1. The measured result

| Challenge category | n | isaac2 | round3 |
|---|---|---|---|
| C1 extreme background, 10 px target | 66 | recall 0.333 | 0.455 |
| C2 very small targets, 1-3 px | 40 | recall **0.000** | **0.000** |
| C3+C4 fast blur and extreme pose, ~18 px | 64 | recall 0.297 | 0.359 |
| **C5 historical failure, 1100-1524 px** | 24 | **recall 1.000** | **recall 1.000** |
| overall, 194 evaluable rows | 194 | P 0.278 R 0.335 mAP50 0.274 | P 0.299 R 0.407 mAP50 0.310 |

**C5 is the good news and it is unambiguous: the size range that failed for the whole project - the
1100 to 1524 px band that the real photographs live in - is now detected at 1.000 by BOTH models.**
The historical failure case is closed.

**C2 is the bad news and it is equally unambiguous: 1 to 3 px scores 0.000**, which is the same floor
seen everywhere else in this project.

## 2. The paired design, and why it changes how C1 must be read

The challenge set was built with a deliberate pair: C1 renders a 10 px shuttle on 33 hard scenes, and C1n
is the SAME 33 scenes with no shuttle at all. Comparing them scene by scene separates "fires on the
clutter" from "found the shuttle":

| | isaac2 | round3 |
|---|---|---|
| recall on the 10 px target over 33 scenes | 11/33 = **0.333** | 15/33 = **0.455** |
| boxes produced by the SAME scenes when EMPTY | **32 (0.97 per scene)** | **33 (1.00 per scene)** |

**On these scenes a detection is no more likely to be the shuttle than to be noise.** The model fires
roughly once per scene whether or not anything is there, and additionally matches the target in a third
to a half of them.

So "C1 recall 0.333" is NOT a capability figure and must never be quoted as one. It is the product of a
model that has not learned to discriminate on cluttered scenes, plus a 10 px target that is near its
floor anyway. The honest headline from C1 is the pair, not the recall.

## 3. What this adds to the picture

| Axis | Status |
|---|---|
| real photographs (the deployment blocker) | resolved: 0.000 -> 0.600 recall, 0.500 precision |
| historical large-target failure, 1100-1524 px | resolved: 1.000 |
| false positives on shuttle-free scenes | largely resolved: 82 -> 43, worst confidence 0.669 |
| **discrimination on cluttered scenes** | **NOT resolved, and now measured precisely: recall equals the empty-scene firing rate** |
| very small targets, 1-3 px | not resolved, and the project has measured no model above 0.000 there |

The third row is new and it is the most useful thing this split has produced: it converts "the model has
a precision problem" into "on these scenes the model does not discriminate at all", which is a stronger
and more actionable statement. It also lines up with the false-positive work: hard negatives reduced the
COUNT of spurious boxes substantially, but they have not taught discrimination where the clutter is dense
and the target is at the small end.

## 4. Caveats

- The challenge set, like the controlled matrix, is numpy-rendered. `baseline_round3` was trained on that
  appearance and scores better on it than `baseline_isaac2`, which was trained toward photographic
  appearance. The C1 comparison is therefore also a renderer-alignment comparison and not purely a
  capability one.
- The "scenes where both fire and hit" line printed by the analysis script is an artefact - every scene
  has rows in both C1 and C1n by construction - and is not reported as a finding.
- The real positives remain 6 distinct photographs behind 10 rows.
