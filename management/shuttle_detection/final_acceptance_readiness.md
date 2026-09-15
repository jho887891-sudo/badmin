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
| `challenge test` | **No** | **no CHALLENGE_TEST set has ever been built** |

Three items fail unambiguously no matter what thresholds are chosen, because no threshold accepts them:
**real-image recall 0.000**, **precision 0.186 with 65 boxes on empty scenes**, and the absence of a
challenge set. Those are the honest headline.

## 3. Gaps that need building, not just measuring

**CHALLENGE_TEST does not exist.** Spec 01 lists the two splits and the P0 asset report already recorded
that neither had been created; Plan 3 then built and froze the FIXED_CORE_TEST sets, and the challenge
split was never added. Spec 08 section 4.2 expects it to grow from real-machine footage, real video,
extreme backgrounds, very small targets, fast blur, extreme poses and historical failure cases. Some of
that material now EXISTS in this repository and simply has not been assembled into a labelled split: the
30 shuttle-free scenes, the historical false positives, the near-field ladder rungs, and the
`bg_001` anomaly. Assembling it is buildable work with the data in hand; labelling real-machine footage
is not, because there is no real-machine footage.

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
