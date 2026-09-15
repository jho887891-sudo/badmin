# Final acceptance readiness: what the Plan 8 gate can and cannot decide today

Spec: `docs/superpowers/specs/08_FINAL_EVALUATION_SPEC.md`. Written before the round-2 retrain so the
gaps are known in advance rather than discovered at the gate.

## 1. The entry conditions are not met, and one of them is structural

Spec 08 section 2 requires Specs 01-05 to have passed, with Spec 05 in state `SYSTEM_REQUIREMENTS_MAPPED`.
The actual state is `REQUIREMENT_PARTIAL`: the mapping exists (see `PLAN5_REQUIREMENT_MAPPING.md`) but
the single number does not, because no camera intrinsics and no required working distance exist in this
repository or from the human partner yet.

That matters more than it first appears. **The acceptance gate compares measurements against
requirements. Without Spec 05 there are no thresholds to compare against** - so most gate items cannot
be marked PASS or FAIL at all, only "not decidable".

## 2. Gate item by item

| Gate item (spec 08 section 12) | Decidable now? | Current reading |
|---|---|---|
| `minimum target size requirement` | **No** | needs fx and the required working distance |
| `single-frame requirement` | Partly | the curve exists (0.900-1.000 across 24-512 px) but the required size is unknown |
| `temporal continuity requirement` | **No** | needs per-frame video ground truth, which does not exist |
| `real-image requirement` | **Yes - and it FAILS** | recall **0.000** on 10 verified real positives |
| `real-video requirement` | **No** | no ground truth; diagnostic only |
| `left-camera requirement` | **No** | the project has ONE camera; no stereo split has ever been defined |
| `right-camera requirement` | **No** | same |
| `precision requirement` | **Yes - and it FAILS** | 0.186 overall, and 65 boxes on 30 shuttle-free scenes |
| `Top-K requirement` | Partly | Top-K hit 0.478 measured; the required value is unknown |
| `latency requirement` | Partly | 51.7 ms/image measured on an A6000; the DEPLOYMENT GPU is unspecified |
| `VRAM requirement` | Partly | 4595 MB peak TRAINING; inference VRAM not measured on a deployment target |
| `fixed core test` | **Yes** | the frozen sets exist, are audited and never enter training |
| `challenge test` | **Yes now** | built: 227 rows, audited 0 errors / 0 warnings, no shared material with the core |

Three items fail unambiguously no matter what thresholds are chosen, because no threshold accepts them:
**real-image recall 0.000**, **precision 0.186 with 65 boxes on empty scenes**, and the absence of a
challenge set. Those are the honest headline.

## 3. Gaps that need building, not just measuring

**CHALLENGE_TEST now exists** (commit 973a4be, `outputs/shuttle_capability/challenge_test/manifest.csv`,
227 rows, audit 0 errors / 0 warnings, its own directory and split value so the frozen core is
byte-identical). Composition by spec 08 section 4.2 source, measured:

| source | n | measured spread |
|---|---|---|
| extreme_background | C1 66 + C1n 33 | 10.0 px on 33 scenes MEASURED to fire the baseline; C1n is the same 33 scenes with NO shuttle, so a paired false-positive-vs-recall comparison is possible scene by scene |
| very_small_targets | 40 | 1.0-3.0 px, at and below the core set 2 px floor |
| fast_blur | 24 | motion 10/14/20 px against the core sweep 7 px ceiling |
| extreme_pose | 40 | the side family, measured hardest at 0.150, 8 orientations |
| historical_failure | 24 | 1100-1524 px, above the core 1024 px ceiling and inside the measured 838-1578 px real range |
| **real_machine_footage** | **0** | **NOT COVERED** - no real-machine footage exists here. Recorded as a gap in the source of truth, not substituted |

Isolation is proven by measurement rather than by naming: 227 distinct image content hashes with ZERO
shared with the core, empty scene-name intersection, and `find_leaked_backgrounds` clean against both the
core and the training pool. That last check earned its keep - three of the hard-negative candidates were
core scenes under other filenames (hn_001/bg_026, hn_007/bg_025, hn_008/bg_028, r 0.99-1.00), so they are
excluded and the exclusion prints with its correlation each run. A name-based check would have shipped
them, and the challenge signal would have been partly the core set measured twice.

The training guard refuses the split under four spellings and writes nothing when it refuses.

**Still not run:** the paired C1/C1n comparison. It is now runnable and is the direct false-positive
measurement this split was built to make.

**Left/right camera acceptance cannot be attempted.** The spec asks for both cameras to be reported
separately. The project has a single monocular pipeline; `camera_id` exists as a manifest field and the
controlled set records one value. Producing a left/right split needs stereo calibration and stereo data,
neither of which exists. This is a genuine scope gap against the spec, not a measurement gap.

**Video temporal metrics cannot be computed.** Longest miss streak, reacquisition frames/ms and stable
detection duration all need per-frame ground truth. The available 150 frames have none, and
verification on the actual frames found the footage does not reliably contain an identifiable
shuttlecock at all.

## 4. What IS ready and will be run after the retrain

- the 2070-row controlled matrix, every conditioned group at n>=20 with published error bars;
- the frozen 160-image real-background set, before and after every change;
- the 16 verified real photographs, the one set with human-verified labels;
- the 30 shuttle-free scenes, which is the only clean false-positive protocol;
- the fairness comparison spec 08 section 11 requires, including the "what got worse" half, which this
  project has been recording all along (the false-positive confidence regression is already on record).

## 5. The honest summary

Of thirteen gate items, **two can be decided today and both FAIL**; four are partly decidable but have no
threshold to compare against; and seven cannot be decided at all, of which three need material that does
not exist anywhere yet (a stereo rig, real-machine video, a challenge split built from real-machine
footage).

The state machine therefore cannot reach `FINAL_ACCEPTED` from here. What it CAN reach, and what the
remaining work targets, is a fully documented `FINAL_EVALUATION` on the sets that do exist, with every
undecidable item named and its unblocking condition stated - which is what this file is for.