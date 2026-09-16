# On real match footage the strongest detections are a STATIC wall fixture, confirmed geometrically

Date: 2026-09-15  |  Model: `baseline_os` (v2)  |  Material: 235 frames sampled from 30 rally clips of six
real matches, provided by the user and unpacked from the Badminton-hit-detection dataset.

## 1. Why this material matters

The project has never had real match footage. The earlier video work used 150 low-quality frames whose
suitability was itself in doubt, and a later re-check found no identifiable shuttlecock in them at all. This
is six real matches - professional, amateur singles, amateur doubles - 30 clips, 202 MB, with hit-event
markers. **It is the first material capable of showing what the detector does on the real task.**

## 2. What it does

| Measure | Value |
|---|---|
| frames producing any box | 96 / 235 (40.9%) |
| frames producing a box at or above 0.5 | **16 / 235 (6.8%)** |
| total boxes | 116 |
| **highest confidence anywhere** | **0.610** |
| distribution of the >= 0.5 frames | **14 of 16 in a single match**, match25 |

Compare the same model on the frozen sets: 0.800 on real photographs, 0.850-1.000 from 64 px on the
appearance-aligned ladder. **On real match footage its ceiling is 0.610.**

## 3. What the detections actually are

Magnifying the 24 strongest detections put all of them on **one object: a pale wall fixture on the green
wall of the match_china venue**, boxed at 0.61 down to 0.48. They are not shuttlecocks.

That was then confirmed without relying on my reading of a contact sheet, by testing the property that
distinguishes a shuttle from a fixture: **a shuttle in flight moves, a wall fitting does not.** Measuring the
spatial spread of box centres within each clip:

| Clip | Boxes | Centre spread (px) | Confidence | |
|---|---|---|---|---|
| 1_02_00, 1_04_00, 1_05_01, 1_05_02 | 8-9 each | **0.0, 0.1** | 0.27-0.54 | **STATIC** |
| 1_01_00, 1_03_00, 1_05_00 | 8-9 each | 22-131 | 0.05-0.60 | moving |
| doubles0-3 | 10-15 each | 147-275 | 0.05-0.61 | moving |

**Four clips have boxes that do not move at all across eight frames spanning the whole clip.** Those four
are the source of the 14 high-confidence frames in match25, which is exactly the distribution the summary
table showed and could not explain on its own.

## 4. What this establishes

1. **The detector has a systematic false positive on a real venue fixture**, at confidences up to 0.61 -
   high enough to survive the 0.40 operating point recommended in the freeze package.
2. **Its confident detections on real footage are not shuttles.** On this material, everything above 0.48
   is the same wall object.
3. **The synthetic and real-photograph protocols could never have caught this.** They contain no real venue
   interiors with fixtures, because the background pool was assembled by keyword and is 37% non-venue.

Point 3 is the uncomfortable one: this is a defect that only appears on material the project did not have,
and the protocols were passing without it.

## 5. The constructive use of this material

These 235 frames - and the 30 clips behind them - are **hard negatives of a character the synthetic pools
cannot produce**: real badminton venues, real lighting, real wall fittings, real people. The project's own
hard-negative protocol is precisely the tool for this, and it has the strongest evidence behind it of any
hard-negative batch collected so far, because the false positive is now identified rather than inferred.

What it is NOT: a source of shuttle boxes. This dataset annotates player boxes and hit events, not the
shuttle; its pickles need the original classes to open. Any training use must therefore be as negatives, or
as backgrounds for composites, until real shuttle labels arrive from a source that has them.
