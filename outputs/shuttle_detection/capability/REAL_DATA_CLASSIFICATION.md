# Classification of the acquired real data

Date: 2026-09-16  |  Sources supplied by the user. Classified before any of it is used, because the first job
with third-party data is to find out what it actually is rather than what its page says.

## 1. Summary by source

| Source | Acquired | Label type | Source diversity | Label quality | Verdict |
|---|---|---|---|---|---|
| ① Roboflow Shuttlecock (`mathieu-cartron/shuttlecock-cqzy3`) | **8,053 images** | box, class 0 `Shuttlecock` | **3 videos** | **fixed-size markers, not measured extents** | usable only as a narrow-scale supplement |
| ④ Badminton-hit-detection | 30 rally clips, 202 MB, 6 matches | player boxes + hit events | 6 matches, 3 venues | good for what it labels | **no shuttle boxes at all** |
| ② One-Shot Badminton Shuttle Detection | not acquired | - | - | download link is a placeholder in the repo README | blocked |
| ③ TrackNetV2 | not acquired | centre x/y + visibility | many matches | coordinates only | needs conversion, not attempted |

## 2. The Roboflow set, classified

### By source video - and this is the finding that matters

| Group | Images | Empty labels | Boxes/image | Box (w,h) normalised | **Equivalent px at 640** |
|---|---|---|---|---|---|
| `video_label_2` | 5,391 | **69** | 0.99 | 0.00703 x 0.01172 | **4.5 - 5.8** |
| `video_label_3` | 1,447 | 0 | 1.00 | 0.01016 x 0.01797 | **7.2 - 8.6** |
| `video_label_1` | 1,215 | 0 | 1.00 | 0.01016 x 0.01797 | **6.2 - 8.6** |

**8,053 "images" come from three source videos.** They are dense frame samples of three matches, not 8,053
scenes. Any claim about this dataset generalising must be read against that.

### By annotation geometry

Across 5,569 boxes in the training split there are **8 distinct (w,h) pairs, and two of them account for
99.8% of the total**: 0.00703 x 0.01172 (66%) and 0.01016 x 0.01797 (34%). **The box size is constant within
a source video**, which means these are point annotations rendered as fixed rectangles rather than measured
object extents.

That is not a defect in the data so much as a fact about it: **the annotations carry POSITION reliably and
SIZE not at all.**

### By image property

All images are **640 x 640**, already cropped by the dataset author. Mean brightness 89-111, so no exposure
extremes. Content is professional broadcast footage - HSBC BWF World Tour, VICTOR and YONEX branding,
`clidep.com` watermarks - which makes it the same family as the `match22` clips in source ④.

### Negatives

**69 images have empty label files and are genuine real negatives**, all in `video_label_2` (48 train, 16
valid, 5 test). That is a small but real contribution: the project has no other real negatives except the 96
Commons-acquired hard negatives and the 235 match frames.

## 3. Overlap with this project, checked rather than assumed

The user warned that public shuttlecock datasets are likely to be copies or re-workings of one another and
must be hash-deduped and near-duplicate-checked first. Checked against **all ten frozen sets, 3,090 images**:

| Check | Result |
|---|---|
| byte-identical to a frozen image | **0** |
| near-duplicate at r >= 0.90 | **39** |
| overlap with the real-photograph test set | **0** |
| overlap with the controlled matrix, challenge set, core backgrounds, hard negatives | **0** |

The 39 near-duplicates all point at `real_match_frames/match22_*`, which I created thirty minutes earlier
from source ④. **They confirm that the Roboflow set and ④ share broadcast footage** - exactly the
same-family risk the user named - and they touch none of the frozen sets.

## 4. What the classification implies for use

**The dataset is one scale band, not a general real-domain solution.** Its 5,569 labelled targets are all
4.5-8.6 px in a 640 frame, from three videos. It can contribute exactly one thing: **real appearance of a
real shuttlecock at the size where this project is weakest and has no real data whatsoever** - the 4-12 px
band, where the frozen sets measure 0.05 to 0.30 recall.

What it demonstrably cannot contribute:

- **scale diversity** - the boxes are constant per video, so training on them as-is teaches a fixed size
  prior. The project's entire size ladder depends on scale invariance, and this would push against it.
- **scene diversity** - three videos.
- **real boxes at other distances** - there are none in the set.

## 5. The classification of source ④, for completeness

| Property | Value |
|---|---|
| Contents | 30 rally clips, 202 MB, 6 matches (`match22`, `match25`, `match_china`, `match_china2`, `match_clementi`, `match_yewtee`) |
| Levels | professional singles, amateur singles, amateur doubles |
| Labels | player bounding boxes (normalised, with visibility) and hit events; **no shuttle labels** |
| Usable as | real venue frames for hard negatives and backgrounds |
| Evidence it is worth using | the model's strongest detections on it are a static wall fixture at up to 0.61 confidence, confirmed by a 0.0 px centre spread across eight frames (`REAL_MATCH_FOOTAGE_STATIC_FP.md`) |

## 6. What is still missing

**Real bounding boxes at real, varied sizes.** Neither acquired source has them: ① has reliable positions at
a single scale, ④ has no shuttle labels. That is what ② or ③ would have to supply, and ② is unreachable
because its repository README carries a placeholder download link.
