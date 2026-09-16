# The background pool has a real quality defect, and I tested whether it drives the results. It does not.

Date: 2026-09-15  |  Found while looking at WHY the model misses small targets - the contact sheet of
misses showed shuttles composited onto Greek inscriptions, a laboratory microscope, a construction site and
a trophy cabinet, which prompted an audit of the 30 core backgrounds.

## 1. The defect

| Group | Count | What they are |
|---|---|---|
| **Construction commencement ceremony** | **8** (bg_001-008) | ONE event, near-duplicate frames: a ground-breaking site plus crowds in white shirts |
| Other non-venue | 3 | bg_013 a trophy cabinet, bg_016 a palace film hall, bg_025 a portrait of a duke standing OUTSIDE Badminton House - which is a stately home, not a court |
| Venue-like | 19 | sports halls and gymnasiums, which is what the pool was meant to be |

**11 of 30 (37%) are not badminton venues, and 8 of those are the same event photographed repeatedly.**
The titles are in `bg_metadata.csv` and were never screened for relevance; the pool was assembled by keyword
search and the ceremony photographs match the keyword "badminton court" while showing a building site.

## 2. What is NOT wrong, and this matters

**The frozen-core discipline held.** The 8 ceremony photographs are in the TEST pool only:

```
core TEST backgrounds : 30
TRAIN backgrounds     : 25
overlap between them : 0
```

No test background has ever entered training, which is the property the whole protocol depends on. The
defect is one of REPRESENTATIVENESS, not of contamination.

## 3. The test I ran on the defect, and it failed to confirm my expectation

The obvious worry was that 8 near-duplicate non-venue scenes would carry the false-positive count - 27% of
the scenes being one building site. Measured per scene on `baseline_os`:

| Group | Scenes | Boxes | Boxes per scene | Completely clean |
|---|---|---|---|---|
| ceremony, non-venue | 8 | 11 (31%) | **1.38** | 1/8 |
| other non-venue | 3 | 4 | 1.33 | 1/3 |
| **venue-like** | 19 | **21** | **1.11** | **8/19** |
| total | 30 | 36 | 1.20 | 10/30 |

**The venue-like scenes produce 21 of the 36 boxes at 1.11 per scene.** The false positives are spread
across the pool rather than concentrated in the defective part of it, so the published figure of 36 boxes
with 10 clean scenes is a real property of the model and not an artefact of pool composition.

The ceremony scenes ARE harder - only 1 of 8 is clean against 8 of 19 for the venue-like group - which is
consistent with the false-positive diagnosis that named white-clad crowds, and it is why they were useful
as a stress case even while being poor examples of a badminton venue.

## 4. What this does and does not change

**Does not change:** the false-positive numbers, the real-photograph results, the appearance-aligned curves,
the v2 candidate, or any regression expectation. All remain as published.

**Does change:** the composition of the reported false-positive protocol should be stated. "30 shuttle-free
scenes" implies 30 independent venues and it is really 19 venue-like scenes plus 11 that are not venues, 8 of
those being one event. A reader weighing 36 boxes should know that.

**For v2 of the data:** the background pool should be screened for relevance rather than assembled by
keyword, and near-duplicates from a single event should be culled. That is a curation task, not a
measurement one, and it is the same class of defect as the duplicated real-photograph rows found earlier -
the pool was built for coverage and never audited for what it actually depicts.
