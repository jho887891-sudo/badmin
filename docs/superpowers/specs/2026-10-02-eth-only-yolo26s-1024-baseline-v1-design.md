# ETH-Only YOLO26s 1024 Baseline V1 - design spec

Derived from the approved implementation plan (ETH_ONLY_YOLO26S_1024_BASELINE_V1_IMPLEMENTATION_PLAN.md);
this file records the same contract in repository form. Task numbering follows the plan.

## Goal
Train and fairly evaluate a plain YOLO26s detector at imgsz=1024 on ETH real positives + ETH official
negatives only, then compare it with the ETH official YOLOv8s baseline and our YOLO26-P2 A/B runs, using the
same evaluator family, the same frozen evaluation sets and the same cost protocol.

## Non-negotiables
- One 24 GB GPU (the remote RTX A6000 is used with a 24 GB per-process cap).
- Plain YOLO26s, standard strides only, no P2, no custom head.
- Init from the verified official pretrained asset yolo26s.pt (20,422,725 B, sha256 646f8bc3...).
- Single class shuttlecock (class id 0).
- Training data: ETH real positives (official train locations, difficulties easy+medium) plus ETH official
  negatives (coco_train, official fraction 0.1). iPhone, Isaac Sim, near-field synthetic,
  synthetic-on-real-background, D455 hard negatives, Roboflow pseudo boxes and every evaluation image are
  excluded.
- The train/val split is location-disjoint and frozen before any metric is observed; internal validation is
  15-20% of eligible positives.
- Recipe: imgsz 1024, freeze 0, AdamW, lr0 1e-4, epochs 50, nbs 32, physical batch 8 with fallback 8 -> 6 -> 4
  on OOM; optimizer auto is forbidden; nothing is tuned from intermediate metrics.
- Checkpoint selection uses ONLY the internal location-disjoint validation mAP50-95 (Recall reported
  alongside). val|eth_unseen, external_real_only, controlled_capability and challenge_test are evaluation-only.
- New-model metrics must reproduce the existing evaluator conventions (conf 0.25 operating point, AP floor
  0.001, NMS iou 0.7 / max_det 300 / rect False, equiv_size_640 buckets).
- Existing Stage A/B/ETH-official artifacts are never overwritten.
- Interpretation: |difference| < 0.02 inconclusive; 0.02-0.05 needs confirmation; > 0.05 may guide the next
  experiment subject to sample size and leakage checks. Poor metrics do not invalidate correct execution.

## Resolved official recipe (evidence: runs/final-model/config.json in the ETH repo)
yolov8s, imgsz 1024, epochs 50, batch 32, nbs 64, adamw lr 1e-4, momentum 0.9, weight_decay 5e-4,
confidence 0.5, dist_threshold 25 px, fraction_coco_train 0.1, mosaic 1.0, mixup 0.7, scale 0.5, fliplr 0.5,
hsv_h 0.015, shear 0, perspective 0, degrees 0. Training locations: cab_1, cab_2, glc_1, glc_2, ml_3, ml_4,
ml_6, ticino_1, ticino_2, uetlibergstrasse_1, uetlibergstrasse_2, coco_train. Difficulties: easy + medium.
Consequence for us: batch 8 / nbs 32 are the only deliberate deviations (24 GB budget).

## Resolved dataset facts (remote /home/T7/dgut/robot_sim/eth_shuttle_detection)
- positives (easy+medium, 11 locations): 19,678 frames
- official negatives (coco_train_easy): 5,500 images, zero label files (pure background)
- non-training holdouts: coco_val_easy 1,000 + 11 hard dirs 831 = 1,831
- leakage to remove: every evaluation image of ours that lives in the ETH dataset (2,765 val frames from
  ml_6 / ml_3 / uetlibergstrasse_1), which empties those three locations for training.

## Tasks
1. (done) Pin the experiment contract and resolve the official recipe -> config, resolver, tests, recipe json.
2. Build audited ETH-only manifests + frozen location-disjoint split (train/val manifests, dataset yaml,
   data audit, size distribution).
3. 24 GB-safe training launcher + 3-epoch smoke test (diagnostic only, never reported as final).
4. Full 50-epoch run with internal-only checkpoint selection.
5. Evaluate unseen-real performance, size buckets and localization with evaluator parity.
6. Benchmark deployment efficiency under the same protocol.
7. Unified comparison report and the decision for the next experiment.

## Execution order (four days)
Day 1: tasks 1-2 (+ smoke of task 3). Day 2-3: task 4 training. Day 4: tasks 5-7.

## Completion gate
Every task's steps executed with its verification, evaluator parity test green, leakage audit PASS, no
evaluation set used for selection, and the comparison report answering the plan's eight questions.
