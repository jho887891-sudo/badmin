# YOLO26S_V2_FULL_ETH_CONTINUATION_V1

**Status:** PRE-REGISTERED DESIGN SPEC  
**Experiment type:** Continued training / positive-domain expansion  
**Primary model:** Existing trained plain YOLO26s V2 best checkpoint  
**Primary purpose:** Measure how much additional ETH real-positive coverage can recover recall, especially tiny-shuttle recall, before changing architecture to P2 / stride 4.  
**Training resolution:** 1024  
**Architecture change:** None  
**P2:** Forbidden in this experiment

---

## 0. Executive summary

This experiment continues training from the current best plain YOLO26s detector rather than restarting from official `yolo26s.pt`.

Starting checkpoint:

```text
eth_real_hardneg_v2_best
```

The experiment adds broader ETH official-style real training coverage while keeping the model architecture unchanged.

The primary question is:

> After the current trained YOLO26s has learned the clean ETH subset and the real hard-negative intervention, how much additional detection capability can be recovered simply by exposing it to the full eligible ETH real training distribution?

This experiment is intentionally placed **before** the P2 / stride-4 experiment.

The zero-training resolution-response diagnostic has already selected `P2_STRIDE_4` as the next architecture branch, but that diagnostic does **not** prove that P2 is necessary. It only says that higher inference resolution alone did not sufficiently revive the measured sub-stride population. Therefore this experiment first tests whether missing positive-domain coverage is still the dominant limitation.

The experiment must not be interpreted as an unseen-generalization benchmark because part of the legacy ETH evaluation population may overlap the newly expanded training pool.

---

# 1. Research question

Current V2 uses a plain YOLO26s and has already undergone:

```text
official pretrained YOLO26s
    ↓
ETH clean real-data training
    ↓
real hard-negative intervention
    ↓
eth_real_hardneg_v2_best
```

The current model has strong false-positive control but weak tiny recall.

Current reference values on the legacy `val|eth_unseen` diagnostic:

```text
Precision        ≈ 0.8783
Recall           ≈ 0.2040
mAP50-95         ≈ 0.1897
Recall_<8        ≈ 0.0381
TP_<8            = 4 / 105
```

The core question is:

> If we keep the current trained plain YOLO26s unchanged architecturally and expand its ETH real-positive training exposure to the full official-style pool, does tiny recall materially recover?

The experiment is therefore a:

```text
POSITIVE_DOMAIN_EXPANSION_CONTINUATION
```

not:

```text
architecture ablation
P2 experiment
synthetic-data experiment
from-scratch baseline
```

---

# 2. Hypotheses

## H1 — Positive coverage is still a major bottleneck

The current V2 may be too conservative because it has seen limited positive-domain coverage relative to the official ETH training distribution.

Prediction:

```text
broader ETH positive exposure
    ↓
more stable shuttle representation
    ↓
higher overall recall
    ↓
higher 6-8 px recall
```

while keeping the existing hard-negative knowledge.

## H2 — Tiny recovery may be concentrated in 6-8 px

The prior zero-training resolution diagnostic showed that the real 6-8 px bucket already produces some response, while strictly sub-stride evidence is sparse and heavily confounded by synthetic renders.

Prediction:

```text
6-8 px recall may improve more than <4 / 4-6
```

under continued real-data training.

## H3 — If positive expansion does little, architecture becomes the stronger suspect

If overall and tiny recall remain nearly unchanged after full ETH positive exposure, then the case for the next P2 / stride-4 experiment becomes stronger.

This still does **not** prove plain YOLO26s is intrinsically incapable of tiny detection.

---

# 3. Frozen starting checkpoint

Use exactly:

```text
eth_real_hardneg_v2_best
```

Expected checkpoint identity:

```text
bytes: 20,344,133
sha256:
3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7
```

Before training:

1. locate the checkpoint;
2. verify byte size;
3. verify SHA256;
4. load successfully with the current Ultralytics runtime;
5. record architecture summary;
6. verify this is plain YOLO26s;
7. verify there is no P2 detection layer.

If checkpoint hash does not match:

```text
STOP
```

Do not silently substitute another V2 checkpoint.

---

# 4. Frozen architecture

The architecture must remain:

```text
plain YOLO26s
```

No architecture modification is allowed.

Forbidden:

```text
P2
stride-4 detection head
custom neck
custom backbone
new detection head
loss architecture change
new attention module
new feature fusion module
```

The experiment must reuse the V2 model structure exactly.

Record:

```text
parameter count
stride list
model YAML / resolved architecture
Ultralytics version
PyTorch version
CUDA version
```

---

# 5. Training-data intervention

## 5.1 Positive data

Use the full eligible ETH main real-positive training distribution, following the official ETH training recipe as closely as repository evidence supports.

Expected eligible difficulties:

```text
easy
medium
```

Current audited counts:

```text
easy   = 15,085
medium = 4,593
```

Expected positive pool:

```text
~19,678 labeled real images
```

Use all eligible ETH locations.

Do not exclude an ETH frame merely because it appeared in our historical `val|eth_unseen` diagnostic.

This is deliberate.

The experiment is measuring:

```text
continued training capacity under broad ETH exposure
```

not:

```text
unseen generalization
```

## 5.2 Official ETH background negatives

Include the official ETH background pool:

```text
coco_train_easy
```

Expected:

```text
5,500 negative images
```

Every included background image must be checked to have an empty shuttle label.

## 5.3 Existing V2 hard-negative knowledge

Do **not** add a new hard-negative intervention in this experiment.

The starting checkpoint already contains the effect of the previous 89-image real hard-negative training intervention.

The new training manifest must not intentionally oversample those hard negatives again unless they are naturally part of the selected official ETH pool.

This experiment changes the positive-domain exposure, not the hard-negative policy.

---

# 6. Explicitly prohibited data

Do not add:

```text
our synthetic positives
our synthetic negatives
Isaac Sim renders
numpy-rendered shuttle images
synthetic-on-real-background
Roboflow pseudo boxes
D455 positive data
new D455 negatives
new mined hard negatives
ETH iPhone extras
near-field synthetic
controlled_capability images
challenge_test images
legacy external evaluation images that are not official ETH train-pool members
```

The only new training intervention is:

```text
expanded official ETH real training coverage
```

---

# 7. Data audit

Before any training run, generate:

```text
data/yolo26s_v2_full_eth_cont_v1_train_manifest.csv
metrics/yolo26s_v2_full_eth_cont_v1_data_audit.json
```

The manifest must record at least:

```text
path
source
location
difficulty
label_path
has_label
bbox_count
equiv_size_640
sha256
is_positive
is_negative
```

The audit must report:

```text
total images
positive images
negative images
box count
location counts
difficulty counts
size-bucket counts
duplicate hashes
missing images
missing labels
non-empty negative labels
prohibited-source hits
```

Training must not start unless:

```text
missing images = 0
unreadable images = 0
non-empty official negative labels = 0
prohibited-source hits = 0
```

---

# 8. Legacy evaluation overlap policy

This experiment intentionally allows overlap between expanded ETH training data and the old ETH diagnostic set.

Therefore:

```text
TRAIN_EVAL_OVERLAP_EXPECTED = true
```

The old set formerly called:

```text
val|eth_unseen
```

must be renamed in this experiment report to:

```text
legacy_eth_eval
```

or:

```text
official_overlap_diagnostic
```

The report must not describe it as:

```text
unseen
held-out
generalization
test set
```

Its purpose is only:

```text
capacity / recognition comparison against historical checkpoints
```

---

# 9. Training schedule

Because the model is already trained, this is continuation training, not from-scratch fine-tuning.

Primary continuation schedule:

```text
epochs = 20
imgsz = 1024
freeze = 0
optimizer = AdamW
lr0 = 3e-5
```

Rationale:

- the model is already converged enough to be useful;
- `1e-4` is appropriate for the original ETH-style training recipe, but is unnecessarily aggressive for a mature checkpoint;
- the goal is to expand the learned positive manifold without erasing V2 hard-negative calibration;
- 20 epochs is enough to expose the model repeatedly to the full ETH pool without blindly committing another 50-epoch retraining cycle.

The following are frozen:

```text
epochs = 20
lr0 = 3e-5
imgsz = 1024
optimizer = AdamW
freeze = 0
```

Do not tune them after seeing evaluation results.

---

# 10. Batch configuration

Target hardware:

```text
RTX A6000 48 GB
```

Preferred:

```text
physical batch = 32
nbs = 64
```

If batch 32 OOMs:

```text
32 -> 24 -> 16
```

Only the physical batch may change.

Keep:

```text
nbs = 64
lr0 = 3e-5
imgsz = 1024
epochs = 20
optimizer = AdamW
```

Record:

```text
physical batch
nominal batch
gradient accumulation
optimizer-step count
peak VRAM
```

Never use:

```text
optimizer=auto
```

---

# 11. Augmentation and loss

Reuse the resolved official-style ETH augmentation/loss settings already audited for the earlier ETH-only run unless the official repository contains a more authoritative explicit setting.

Freeze before training:

```text
mosaic
mixup
scale
translate
degrees
fliplr
flipud
HSV settings
close_mosaic
weight_decay
warmup
box loss gain
cls loss gain
DFL loss gain
```

Priority:

```text
official ETH explicit setting
    >
same-version resolved Ultralytics default
    >
UNKNOWN with explicit reason
```

Do not change augmentation after observing results.

Generate:

```text
metrics/yolo26s_v2_full_eth_cont_v1_resolved_recipe.json
```

before full training.

---

# 12. Smoke test

Before the 20-epoch full run:

```text
1-2 epoch smoke
same checkpoint
same dataset
same imgsz
same optimizer
same LR
same intended batch
```

PASS conditions:

```text
checkpoint hash verified
plain YOLO26s confirmed
AdamW confirmed
lr0 = 3e-5 confirmed
no NaN / Inf
loss decreases normally
gradients update
checkpoint save/reload works
all intended ETH locations loaded
official negatives loaded
no prohibited data source
GPU memory stable
```

Smoke checkpoint:

```text
DIAGNOSTIC_ONLY
```

It must never be used for the final comparison.

---

# 13. Checkpoint policy

Primary endpoint:

```text
epoch 20 / last.pt
```

This is fixed before training.

Save at least:

```text
epoch_5.pt
epoch_10.pt
epoch_15.pt
epoch_20.pt
last.pt
```

The primary scientific comparison uses:

```text
epoch 20
```

not the best post-hoc epoch.

Secondary curve analysis may report all saved checkpoints.

If an automatically generated:

```text
best.pt
```

exists, preserve it for diagnostics only.

Do not use `legacy_eth_eval` to choose a checkpoint.

---

# 14. Evaluation protocol

Use the same canonical evaluator family used for V1/V2/V3.

Frozen evaluator settings:

```text
imgsz = 1024
operating confidence = 0.25
AP confidence floor = 0.001
NMS IoU = 0.7
max_det = 300
```

Report the primary endpoint at epoch 20.

---

# 15. Primary comparison

The primary causal-style engineering comparison is:

```text
V2 best
    vs
V2 best + full ETH continuation
```

Both are:

```text
plain YOLO26s
imgsz 1024
```

The main new intervention is:

```text
expanded ETH real-positive coverage
```

Do not use clean V1 as the primary causal baseline.

Clean V1 may be shown as historical context only.

---

# 16. Evaluation sets

## 16.1 legacy_eth_eval

Use the same historical ETH diagnostic population for continuity.

Report:

```text
Precision
Recall
F1
AP50
mAP50-95
Recall_<8
TP_<8
```

But explicitly label it:

```text
training-overlap diagnostic
```

## 16.2 Frozen real no-target pool

Use the frozen 496-image real no-target pool.

Report:

```text
FP raw
FP/image
image-level FP rate
confidence distribution
```

This is a critical guard because V2's main strength is low false-positive rate.

## 16.3 controlled_capability

Run as synthetic diagnostic only.

## 16.4 challenge_test

Run as synthetic diagnostic only.

Neither synthetic set may be used to claim real-world generalization.

---

# 17. Tiny-object reporting

Continue using:

```text
equiv_size_640 = sqrt(w*h)
```

Required buckets:

```text
<4
4-6
6-8
<8
8-12
12-16
16-24
24-32
32-64
>64
```

For each bucket:

```text
GT
TP
FN
Recall
AP50
mAP50-95
```

Primary tiny endpoints:

```text
TP_<8
Recall_<8
Recall_6-8
```

---

# 18. Source-aware tiny analysis

The previous resolution-response diagnostic established an important coverage limitation:

```text
real ETH tiny:
    <4 : n=1
    4-6: n=0
    6-8: n=19
```

and the synthetic tiny population behaved very differently.

Therefore every tiny table must separate:

```text
eth_main real
synthetic
```

Do not merge the two and interpret the result as real tiny performance.

Required source × bucket table:

```text
eth_main / <4
eth_main / 4-6
eth_main / 6-8
synthetic / <4
synthetic / 4-6
synthetic / 6-8
```

If a bucket has no real GT:

```text
EMPTY_BUCKET
```

not:

```text
Recall = 0
```

---

# 19. Pre-registered decision criteria

Baseline V2 reference:

```text
Recall           ≈ 0.2040
mAP50-95         ≈ 0.1897
Recall_<8        ≈ 0.0381
TP_<8            = 4 / 105
real no-target FP = 1 / 496
```

## 19.1 POSITIVE_EXPANSION_STRONG

Return:

```text
POSITIVE_EXPANSION_STRONG
```

if:

```text
Delta Recall >= +0.10
AND
Delta Recall_<8 >= +0.05
```

Interpretation:

> broader ETH positive coverage materially restores recognition capacity, including tiny-shuttle recall.

## 19.2 POSITIVE_EXPANSION_USEFUL

Return:

```text
POSITIVE_EXPANSION_USEFUL
```

if STRONG is not met, but either:

```text
Delta Recall >= +0.05
```

or:

```text
Delta Recall_<8 >= +0.03
```

Interpretation:

> positive-domain coverage contributes meaningfully but does not fully explain the remaining gap.

## 19.3 POSITIVE_EXPANSION_LOW_EFFECT

Return:

```text
POSITIVE_EXPANSION_LOW_EFFECT
```

if:

```text
Delta Recall < +0.05
AND
Delta Recall_<8 < +0.03
```

Interpretation:

> the extra ETH exposure produces little recovery; architecture / sampling-grid limitations become a stronger next hypothesis.

This does not prove P2 will work.

---

# 20. False-positive guard

The continuation must preserve V2's hard-negative benefit as much as possible.

Baseline:

```text
1 FP / 496 real no-target images
```

Define:

```text
FP_STABLE:
    FP <= 3 / 496
```

Define:

```text
FP_REGRESSION:
    FP >= 4 / 496
```

If recall improves but:

```text
FP >= 4 / 496
```

add flag:

```text
RECALL_GAIN_WITH_FP_REGRESSION
```

Do not hide a recall/precision trade-off.

---

# 21. mAP guard

Baseline:

```text
mAP50-95 ≈ 0.1897
```

Define material regression:

```text
Delta mAP50-95 <= -0.02
```

If triggered, add:

```text
MAP_REGRESSION
```

This flag does not override the primary positive-expansion classification, but it must be highlighted.

---

# 22. Decision precedence

Primary decision:

```text
POSITIVE_EXPANSION_STRONG
    >
POSITIVE_EXPANSION_USEFUL
    >
POSITIVE_EXPANSION_LOW_EFFECT
```

Independent flags:

```text
RECALL_GAIN_WITH_FP_REGRESSION
MAP_REGRESSION
FP_STABLE
```

The final report must emit exactly one primary decision.

---

# 23. What this experiment may conclude

Allowed:

> Continuing V2 with broader ETH real-positive coverage materially improves / partially improves / barely changes recognition performance.

Allowed:

> Most recovered tiny performance is concentrated in the 6-8 px real bucket.

Allowed:

> Positive-domain expansion does or does not preserve the V2 hard-negative false-positive advantage.

Allowed:

> The result strengthens or weakens the motivation for the next P2 / stride-4 experiment.

---

# 24. What this experiment may NOT conclude

Forbidden:

```text
YOLO26 is intrinsically worse than YOLOv8
P2 is proven necessary
P2 is proven effective
higher resolution is useless
synthetic data is harmful
the new model generalizes better
4-6 px real shuttle detection is solved
```

This experiment contains deliberate training/evaluation overlap on the legacy ETH diagnostic.

---

# 25. Relationship to the P2 experiment

The prior zero-training resolution-response diagnostic selected:

```text
P2_STRIDE_4
```

as the next architecture branch.

This continuation experiment must be completed first.

Then the next experiment should be:

```text
YOLO26S_P2_V2_FULL_ETH_CONTINUATION_V1
```

The P2 experiment should reuse, as closely as architecture compatibility allows:

```text
same V2 starting knowledge
same ETH expanded training pool
same imgsz
same continuation epochs
same optimizer
same LR
same augmentations
same evaluator
same decision endpoints
```

The intended architectural intervention is then:

```text
plain YOLO26s
    ->
YOLO26s-P2 / stride 4
```

with minimal additional changes.

---

# 26. Required outputs

Create:

```text
configs/yolo26s_v2_full_eth_cont_v1.yaml

data/
    yolo26s_v2_full_eth_cont_v1_train_manifest.csv

metrics/
    yolo26s_v2_full_eth_cont_v1_data_audit.json
    yolo26s_v2_full_eth_cont_v1_resolved_recipe.json
    yolo26s_v2_full_eth_cont_v1_environment.json
    yolo26s_v2_full_eth_cont_v1_smoke.json
    yolo26s_v2_full_eth_cont_v1_train_metrics.csv
    yolo26s_v2_full_eth_cont_v1_overall.csv
    yolo26s_v2_full_eth_cont_v1_size_buckets.csv
    yolo26s_v2_full_eth_cont_v1_source_size_buckets.csv
    yolo26s_v2_full_eth_cont_v1_no_target_fp.csv
    yolo26s_v2_full_eth_cont_v1_vs_v2.csv
    yolo26s_v2_full_eth_cont_v1_decision.json

reports/
    YOLO26S_V2_FULL_ETH_CONTINUATION_V1_REPORT.md
```

Do not overwrite any V1 / V2 / V3 / resolution-response artifacts.

---

# 27. Training provenance

Record:

```text
experiment name
git commit
start timestamp
end timestamp
wall time
host
GPU model
driver
CUDA
PyTorch
Ultralytics
checkpoint source path
checkpoint bytes
checkpoint SHA256
physical batch
nominal batch
gradient accumulation
optimizer
learning rate
epoch count
optimizer-step count
peak VRAM
training image count
positive count
negative count
location count
difficulty counts
manifest SHA256
final checkpoint SHA256
```

---

# 28. Mandatory pre-run gates

Training may start only if all are true:

```text
[ ] V2 checkpoint SHA256 matches
[ ] model is plain YOLO26s
[ ] no P2
[ ] imgsz = 1024
[ ] optimizer = AdamW
[ ] lr0 = 3e-5
[ ] epochs = 20
[ ] freeze = 0
[ ] full intended ETH easy/medium pool built
[ ] official background pool built
[ ] prohibited data sources = 0
[ ] all files readable
[ ] negative labels empty
[ ] smoke test PASS
[ ] evaluator parity PASS
[ ] primary endpoint fixed to epoch 20 before full run
```

Any failed gate blocks the full run.

---

# 29. Completion questions

The final report must answer all of the following:

1. How much did overall Recall change from V2?
2. How much did mAP50-95 change?
3. How much did Recall_<8 change?
4. How many additional `<8` true positives were recovered?
5. How much did real 6-8 px Recall change?
6. Did any real sub-stride evidence improve?
7. Did the 496-image real no-target FP rate regress?
8. Did broader positive coverage preserve V2's precision advantage?
9. Is the primary result:
   - `POSITIVE_EXPANSION_STRONG`
   - `POSITIVE_EXPANSION_USEFUL`
   - `POSITIVE_EXPANSION_LOW_EFFECT`
10. Does the result support proceeding to the frozen P2 / stride-4 experiment?

---

# 30. Completion rule

The experiment is complete only when:

```text
data audit PASS
smoke PASS
20 epochs completed
epoch-20 checkpoint identified and hashed
legacy ETH diagnostic evaluated
496 real no-target guard evaluated
size-bucket evaluation completed
source-aware tiny evaluation completed
decision JSON written
final report written
all experiment-specific tests PASS
```

The final line of the report must state:

```text
NEXT_ARCHITECTURE_EXPERIMENT = P2_STRIDE_4
```

unless the run is invalid because of a protocol violation.

That line means:

> the data-coverage continuation has been measured, so the next architecture intervention may now be tested.

It does not mean P2 has already been shown to work.
