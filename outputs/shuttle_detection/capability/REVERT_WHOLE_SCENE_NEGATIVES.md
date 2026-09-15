# A Level-1 adjustment that FAILED, measured and reverted: whole-scene hard negatives

Spec 07 section 1 requires every adjustment to be recorded, re-measured on the original frozen protocols,
kept if effective and REVERTED if not. This one was reverted.

## 1. The hypothesis, and where it came from

The previous round found a concrete mechanism rather than a vague "the model has a precision problem".
Four of the 33 challenge C1n scenes overlap the training negatives, yet the model still places about one
box on each of them. The training negatives were **960x960 CENTRE CROPS**, so the model learned the centre
of each scene and not its edges.

The proposed fix followed directly: feed **whole scenes** instead, since Ultralytics letterboxes every
image to the training size anyway and nothing needs to be cropped away. A clean replacement was built -
86 whole-scene negatives, de-duplicated against the frozen core, the training backgrounds and each other,
replacing the 86 crop-style ones. Same count, same negative share of 16.8%, so the only variable was the
framing.

## 2. The measurement

| Protocol | `baseline_isaac2` (centre crops) | `baseline_whole` (whole scenes) | verdict |
|---|---|---|---|
| **real photographs recall** | **0.600** | **0.300** | halved |
| **real photographs precision** | **0.500** | **0.158** | collapsed |
| real photographs mAP50 | **0.505** | 0.307 | worse |
| false positives, 30 shuttle-free scenes | **43** | 47 | slightly worse |
| scenes carrying a FP above 0.5 | **5/30** | 10/30 | worse |
| completely clean scenes | 8/30 | **10/30** | marginally better |
| worst FP confidence | **0.669** | 0.683 | marginally worse |
| controlled matrix precision | 0.240 | **0.276** | better |
| controlled matrix recall | **0.426** | 0.416 | about equal |
| frozen P3 mAP50 | 0.253 | **0.285** | better |
| validation mAP50 | **0.570** | 0.554 | worse |

## 3. The verdict, and why the hypothesis was wrong

The adjustment **halved the recall on the axis that matters most** and did not reduce false positives at
all: 47 against 43, and it doubled the number of scenes carrying a confident false positive from 5 to 10.
The small gains on the synthetic sets do not come close to paying for that.

The most likely reason the reasoning failed: a whole scene resized to fit 960 px, then letterboxed to 640,
shows each hard object at a **much smaller apparent size** than a 960x960 centre crop did. So the negatives
became weaker as hard negatives - the small white blobs the model fires on were shrunk - while
simultaneously teaching it to suppress photographic content more broadly, which is what cost the real
photograph recall.

The lesson is the one the project keeps relearning from a different angle: **the mechanism was measured
correctly and the remedy still did not follow from it.** "The crops missed the edges" was true; "so use
whole scenes" was an inference, and the inference was wrong.

## 4. REVERT

`baseline_isaac2` remains the frozen candidate. `baseline_whole` and `manifest_train_whole.csv` are kept on
disk as a recorded negative result rather than deleted, because spec 07 wants the attempt in the record:

```
REVERTED: whole-scene hard negatives
  measured on: controlled matrix, frozen P3, 16 real photographs, 30 shuttle-free scenes
  outcome: real-photo recall 0.600 -> 0.300, precision 0.500 -> 0.158, FP 43 -> 47
  decision: revert to baseline_isaac2
```

## 5. What this closes

The "cluttered-scene discrimination" defect stands **unfixed**. Two remedies have now been tried against
it - more hard negatives of the same character (which did help the shuttle-free protocol, 82 -> 43) and
whole-scene framing (which did not) - and neither taught discrimination where the clutter is dense and the
target is small. That is now the best-characterised open defect in the project, and it should be attacked
with real footage rather than with more synthetic negatives.
