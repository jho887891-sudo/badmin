# A second adjustment tried and NOT adopted: closing the side-view coverage gap

Spec 07 requires each adjustment to be re-measured on the original frozen protocols and kept only if
effective. This one is not clearly effective, so it is not adopted.

## 1. The gap it was meant to close, which is real

The previous round found that the Isaac render pool samples pitch from only +/-60 degrees, so **no true side
view exists** in the 382 Isaac training samples - which are 41% of the training positives. Meanwhile the
numpy pool covers side views abundantly, 62.4% of its positives at |pitch| >= 65 degrees. And the
numpy-rendered controlled matrix calls side the hardest pose. So the release had seen side views only in
appearance it was not trained for.

209 new images were rendered with pitch up to +/-90 degrees, 69 of which (31%) are true side views,
composited the same way as the rest of the pool and added to training. Manifest: 1317 rows.

## 2. The measurements, in full, because the verdict is genuinely mixed

| Axis | `baseline_isaac2` | `baseline_side` | Better |
|---|---|---|---|
| appearance ladder recall / precision / mAP50 | 0.586 / 0.272 / 0.510 | **0.603 / 0.314 / 0.569** | side |
| controlled matrix recall / mAP50 | 0.426 / 0.351 | **0.480 / 0.399** | side |
| frozen P3 mAP50 | 0.253 | **0.297** | side |
| **validation mAP50** | **0.570** | 0.512 | isaac2 |
| **real photographs recall / precision** | **0.600 / 0.500** | 0.300 / 0.231 | isaac2 |
| **false positives, 30 shuttle-free scenes** | **43, worst confidence 0.669** | 60, worst 0.876 | isaac2 |

Three axes improve, three get worse, and **which three matters**: the two most deployment-relevant
measurements - real photographs and false positives on real scenes - both regressed, while the synthetic
ones improved. That is the same pattern the whole-scene negative experiment showed, and it is the reason
neither was adopted.

## 3. The real-photograph regression is not statistically significant, and I am not hiding behind that

The real-photograph set has 10 rows, so every hit is worth 0.1 and 0.600 -> 0.300 is a change of three
rows. A two-sided test on 6/10 against 3/10 gives p of roughly 0.37: **not significant either way.**

The same caution cuts the other way, and this is the part I must not skip: the appearance-ladder gain
(0.586 -> 0.603 recall on 237 rows) is also within its own noise. **Neither side of this comparison is
strongly evidenced**, and the decision rests on the false-positive result, which is the one measurement
here that is both deployment-relevant and clearly outside noise: 43 boxes against 60, and a worst
false-positive confidence rising from 0.669 to 0.876.

## 4. Decision

**NOT ADOPTED.** `baseline_isaac2` remains the frozen candidate.

```
ATTEMPTED, NOT ADOPTED: add side views in the deployment appearance
  measured on: controlled matrix, frozen P3, 16 real photographs, 30 shuttle-free scenes,
               appearance-aligned ladder
  outcome: synthetic axes improved (ladder mAP50 0.510 -> 0.569), real-scene axes regressed
           (real-photo recall 0.600 -> 0.300 on n=10, false positives 43 -> 60, worst confidence
           0.669 -> 0.876)
  decision: keep baseline_isaac2; retain baseline_side and manifest_train_side.csv on disk
```

## 5. What remains open, stated precisely

**The side-view coverage gap is still open.** The release still has no true side view in the appearance it
was trained for, and whether side views are easy or hard there is still unknown. This experiment closed the
gap in the data but not in the delivered model, because closing it cost more than it bought on the axes
that matter.

Two remedies against cluttered-scene and pose weaknesses have now been tried and not adopted, and one
(more hard negatives of the same character) succeeded. The pattern across all three is consistent:
**improvements on synthetic tests do not predict improvements on real scenes**, which is itself the most
useful thing this project has learned about how to judge its own experiments.
