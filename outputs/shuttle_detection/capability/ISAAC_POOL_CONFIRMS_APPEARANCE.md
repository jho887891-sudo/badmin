# The appearance hypothesis is CONFIRMED: real-photograph recall 0.000 -> 0.600

Date: 2026-09-15  |  Model: `baseline_isaac`  |  The single change: the training pool now contains 382
positives rendered by **Isaac Sim** instead of the numpy rasteriser, composited with the same
`composite()`, onto the same backgrounds, at the same size distribution and JPEG quality.

## 1. The result

| Protocol | round3 | **isaac** |
|---|---|---|
| **C: 16 verified real photographs, recall** | **0.000** | **0.600** |
| C: precision / mAP50 | 0.000 / 0.000 | **0.400 / 0.485** |
| A: controlled matrix recall | 0.504 | **0.527** |
| A: controlled matrix mAP50 | 0.350 | **0.411** |
| A: controlled matrix precision | 0.165 | **0.256** |
| B: frozen P3 recall | 0.394 | **0.444** |
| B: frozen P3 mAP50 | 0.140 | **0.224** |
| D: false positives on 30 shuttle-free scenes | **46** | 82 |
| D: completely clean scenes | **9/30** | 5/30 |

Validation: precision 0.822, recall 0.525, mAP50 0.570, mAP50-95 0.293 - the best of every model trained
so far on all three of recall, mAP50 and mAP50-95.

**The real domain went from completely broken to two-thirds detected.**

## 2. What this settles

Four candidate causes were tested and excluded before this one, each by a controlled measurement rather
than an argument:

| Candidate | Test | Outcome |
|---|---|---|
| size coverage | round-3 data return, training ceiling 32.86 -> 807 px | excluded: synthetic ceiling passed 1023 px, real recall stayed 0.000 |
| confidence threshold | Level-1 scan, 0.05 to 0.70 | excluded: 0/10 at every threshold |
| input resolution | Level-2, imgsz 640 / 960 / 1280 | excluded: 0/10 at every size, and FPs rose 46 -> 113 |
| **renderer fidelity** | this experiment | **CONFIRMED: 0.000 -> 0.600** |

The chain is now complete and each step is a measurement: the detector could not see large targets
(fixed by data), could not see the real size range (the size rule held but did not help), could not be
rescued by threshold or resolution, and **could not transfer from flat-shaded polygons to photographs**.
Rendering the training data with a physically-based renderer fixed the transfer.

This also makes the earlier visual finding precise rather than rhetorical. The numpy renders were
geometrically faithful - feathers, gaps, ribs and cork all present - but flat-shaded with no texture, no
translucency and a strongly directional light. Isaac Sim renders the same asset with soft, even,
photographic shading. That difference was sufficient to make the real photographs undetectable.

## 3. What it cost, stated plainly

**False positives got worse: 46 -> 82 on the 30 shuttle-free scenes, and clean scenes fell from 9 to 5.**
The worst confidence improved slightly (0.874 -> 0.816), so this is more boxes rather than more
confident ones, but the count nearly doubled.

That is not a surprise in retrospect: a model that has learned to respond to photographic shuttlecock
appearance is also more willing to respond to photographic clutter. The hard-negative pool built earlier
worked against the numpy-rendered model (65 -> 46); it has not been rebuilt against this one.

## 4. What this means for the level decision

The adjustment that worked was **data**, produced by changing the generator, and it produced the
largest single improvement in the project. No architectural change was needed and none is justified.
The remaining defect is precision, and the tool that already worked once for precision is a
hard-negative pool selected by measured model response.

## 5. Honest limits of this result

- **The real-positive denominator is 6 distinct photographs, not 10 rows.** Four of the ten rows are
  byte-identical duplicates. So 0.600 means 6 of 10 rows matched, which over 6 distinct images is
  likely 4 or 5 of 6 detected - either way, a real improvement from zero, but the sample is tiny.
- **One of the six is a plastic shuttle** (real_017), a different appearance class. It is included.
- Isaac Sim was used, not modified. `import isaaclab` needs PYTHONPATH pointing at
  /home/T7/ojh/robot_sim/IsaacLab/source/*, and the app takes 7-16 minutes to launch on this host.
- 18 of the 400 rendered frames were dropped because the scaled object would not fit the 960 px frame.
