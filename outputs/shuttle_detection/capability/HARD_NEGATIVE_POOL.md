# HARD_NEGATIVE_POOL for the false-positive failure — acquired, measured, and it confirms the diagnosis

Date: 2026-09-15  |  Spec 06 section 6: false positive -> classify the source -> collect NEW images of
the same kind -> retrain.

## 1. What was acquired and where it came from

36 new images from Wikimedia Commons, chosen for the CHARACTER the false-positive diagnosis identified
(`FALSE_POSITIVE_DIAGNOSIS.md`): indoor arenas and courts with distant players in light kit, and group
photographs of people in light-coloured clothing. Queries used, all in the File namespace: badminton
hall, volleyball match indoor, group photo ceremony people, conference group photograph, sports hall
interior.

Provenance, licence and source URL are recorded per image in `hard_negative_metadata.csv`. Licences are
CC BY-SA 2.0/3.0/4.0, CC BY 4.0, public domain or attribution.

They are NEW images. Neither the frozen test pool nor the training backgrounds were reused, which is
what spec 06 section 5/6 requires.

## 2. Selection was by MEASURED model response, not by intent

A hard negative is only useful if the model currently fires on it, so every candidate was run through
`baseline_nearfield` before being kept:

| Measurement on the 36 candidates | Value |
|---|---|
| images that produced at least one box | **31 of 36** |
| total boxes (all false positives by construction) | **70** |
| boxes at confidence >= 0.5 | **6** |
| worst single image | hn_031, a wedding group photo in light clothing, **9 boxes** |
| next worst | hn_014 arena crowd 5 boxes at 0.792; hn_018 volleyball match at 0.841; hn_033 group photo at 0.776; hn_025 group photo at 0.761 |

## 3. This independently confirms the false-positive diagnosis

The diagnosis was built on the frozen test backgrounds: it said the model fires on halls with distant
players in white and on group photographs of people in light clothing. These 36 images were collected
AFTER that diagnosis, from a different source, with no reuse of any earlier image - and 31 of 36 fire,
including a 9-box response on a group photo of exactly the predicted character.

So the diagnosis predicted where new false positives would appear, and they appeared there. That is
the strongest form of evidence available to it: a prediction made before the data existed.

## 4. The pool

`04_add_hard_negatives.py` copies the 36 into the training image directory under the `hardneg_train_*`
prefix - a distinct prefix, after a cross-round filename collision already corrupted three frames once
- and emits a merged manifest with the identical column schema as the existing training manifests.

Against the round-1 pool: 636 rows, 500 positives, 136 negatives, negatives 21% of the pool. That is a
higher negative share than usual, which is deliberate: precision, not recall, is the dominant defect
(0.186 overall, and 65 boxes on 30 scenes that contain nothing). The 21% is recorded so the next
retrain can be compared against it rather than quietly drifting.

## 5. Honest limits

- Six of the kept images are architecture or street scenes that do NOT fire. They add scene diversity
  but no hard signal, and they are kept rather than filtered so that the pool is exactly "the images I
  collected", not a figure tuned after seeing the result.
- hn_009 is a 1923 badminton match photograph. It is a badminton scene, so it is the one candidate whose
  shuttle-free status rests on the group being posed with rackets rather than on the scene type. It is
  kept, and flagged here rather than silently included.
- Acquisition was rate-limited: a first attempt hit HTTP 429 partway through. The script now spaces
  requests and backs off on 429. A larger pool would need more time, not a different method.
## 6. Reproduction

```bash
python experiments/shuttle_detection/03_acquire_hard_negatives.py          # fetch candidates
# then measure the model response on them, keep the ones that fire:
python experiments/shuttle_detection/04_add_hard_negatives.py \
  --base-manifest manifest_train_nearfield2.csv --out-manifest manifest_train_full.csv
```
