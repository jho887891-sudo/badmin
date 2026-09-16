# Level-2 section 3.4 oversampling: the experiment was designed, run, and INVALIDATED by a deliberate guard

Date: 2026-09-15  |  Attempt: difficult-example oversampling, the training-organisation comparison spec 07
section 3.4 requires and the only one of them that does not need real data.

## 1. What was done

The remaining defect is false positives on cluttered scenes, so the hard negatives - the images selected
precisely because the model fires on them - were repeated in the training list by a factor of three. The
mechanism chosen was duplication of manifest rows rather than a loss reweighting, because Ultralytics takes
an image list and duplicating a row is what its sampler actually honours.

Result: `manifest_train_os.csv` with 1281 rows against the reference 1109, and a full retrain plus the four
frozen protocols.

## 2. The tell, and why it was chased

Three of the four protocols came back **identical to the reference run to the last digit**:

| Protocol | `baseline_isaac2` | `baseline_os` |
|---|---|---|
| A controlled matrix precision / recall / mAP50 | 0.239989 / 0.425604 / 0.350676 | **identical** |
| B frozen P3 precision / recall / mAP50 | 0.195531 / 0.4375 / 0.252826 | **identical** |
| C real photographs precision / recall / mAP50 | 0.500 / 0.600 / 0.50495 | **identical** |
| validation mAP50 | 0.570 | 0.521 | *different* |

Identical evaluation results across two DIFFERENT weight files - the hashes differ, `e5ebf2c8...` against
`2530af35...` - means the two models saw effectively the same data. That is a contradiction, and it is the
reason this was investigated rather than written up as "oversampling made no difference".

## 3. The cause, verified rather than guessed

```
oversampled manifest : 1281 rows
train.txt            : 1108 lines, 1108 unique
reference manifest   : 1109 rows
reference train.txt  : 1108 lines, 1108 unique
```

`ultralytics_dataset._pool_lines` ends with `return _dedupe(lines)`, and `_dedupe` is deliberate:
"Keep the file order while dropping repeated images from one pool." **The image list is deduplicated before
the trainer ever sees it, so duplicating manifest rows cannot oversample anything.** The two runs trained on
the same 1108 images; only run-to-run training nondeterminism separates them, which is what moved the
validation number.

## 4. Verdict

```
NOT A VALID EXPERIMENT: difficult-example oversampling by manifest duplication
  why: ultralytics_dataset._dedupe removes the repeats before training, by design
  consequence: baseline_os is a re-run of baseline_isaac2 on the same data, not a Level-2 comparison
  spec 07 section 3.4 item "difficult-example oversampling": STILL UNTESTED
```

The guard itself should NOT simply be removed: it prevents an image being silently trained on twice, which
is a real hazard. What the item needs is a mechanism that expresses importance without repeating list
entries - a sampler, or a loss weight - and that is a code change to the training path rather than a data
change, so it belongs to a later revision.

## 5. What this round is worth

Not a measurement: a prevented false finding. Had the identical numbers been taken at face value, the
result would have been reported as "oversampling the hard negatives by three made no measurable difference",
which would have been wrong, would have closed a Level-2 item that is in fact open, and would have been
cited later as evidence that oversampling does not help.

The general lesson is narrower than the ones before it: **when two different models produce bit-identical
evaluation output, suspect the pipeline, not the training.** Deterministic evaluation makes that a reliable
signal, and it is the only reason this was caught.
