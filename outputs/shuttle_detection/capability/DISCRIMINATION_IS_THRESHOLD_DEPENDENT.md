# The cluttered-scene defect is threshold-dependent, and my earlier "no discrimination" claim was too strong

Date: 2026-09-15  |  Model: `baseline_isaac2`  |  Set: the paired C1/C1n design, 66 rows with a 10 px
shuttle and 33 rows of the SAME scenes with nothing in them.

## 1. The correction

`CHALLENGE_TEST_FIRST_RUN.md` concluded that on cluttered scenes "a detection is no more likely to be the
shuttle than to be noise". That was measured at the frozen confidence threshold of 0.05 and stated without
that qualification. Sweeping the threshold over the same pair:

| confidence | C1 recall | C1n boxes per empty scene | **discrimination ratio** |
|---|---|---|---|
| **0.05** (frozen) | 22/66 = 0.333 | **0.97** | **0.34** |
| 0.10 | 22/66 = 0.333 | 0.73 | 0.46 |
| 0.20 | 20/66 = 0.303 | 0.39 | 0.77 |
| 0.30 | 17/66 = 0.258 | 0.30 | 0.85 |
| **0.40** | 16/66 = 0.242 | 0.24 | **1.00** |
| 0.50 | 11/66 = 0.167 | 0.15 | 1.10 |
| **0.60** | 7/66 = 0.106 | **0.03** | **3.50** |
| 0.70 | 0/66 = 0.000 | 0.00 | no detections at all |

The ratio is true positives per empty-scene box, so 1.0 is break-even and above it a detection carries
information. **It crosses 1.0 at about confidence 0.40 and reaches 3.5 at 0.60.**

So the accurate statement is: **at the frozen threshold of 0.05 the model does not discriminate on these
scenes - a detection there is three times more likely to be noise than to be the target - and above about
0.40 it does.** The defect is real but it is an operating-point defect, not a capability wall, and the
earlier phrasing overstated it.

## 2. Why the earlier sweep missed this

The Level-1 threshold scan measured false positives on 30 shuttle-free scenes and recall on the size curve,
but it never measured the two TOGETHER on matched scenes. A threshold that lowers false positives and
lowers recall looks like a pure trade in that framing; paired against the same scenes it is not a trade at
all in the range 0.40-0.60, because the false positives fall faster than the true positives.

That is the practical value of the C1/C1n pair, and it is the argument for building paired designs rather
than separate positive and negative sets.

## 3. What this changes in the release

The frozen inference configuration keeps confidence 0.05 **because every measurement in this project was
taken at it and the freeze must be reproducible**. That has always been stated as the reason rather than
optimality, and this sweep is the strongest evidence yet that 0.05 is a poor deployment choice:

| use case | recommended threshold | rationale |
|---|---|---|
| cluttered scenes, a detection must be trustworthy | **0.50-0.60** | ratio 1.1-3.5, at 0.11-0.17 recall |
| open scenes, recall matters more than precision | 0.05-0.20 | the size curve holds up; clutter is the risk |

A consumer that cannot tolerate false positives on cluttered input should run at 0.50 or above. The
detector returns Top-K candidates with confidences precisely so that this decision can be made downstream,
and the interface in `FREEZE_HANDOFF_v1.md` section 6 supports it.

## 4. What it does not change

- The cluttered-scene defect is **not fixed**, only bounded: at 0.60 the model finds 0.106 of 10 px targets
  on those scenes. Discrimination improved by discarding most detections, not by finding more.
- Nothing below 6 px is affected; the C1 target is 10 px and the floor stands.
- The 0.600 real-photograph recall was measured at 0.05 and is unaffected by this, but it was also measured
  on only 6 distinct photographs, so its own threshold behaviour is not established.
