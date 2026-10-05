# ETH Real Hard-Negative YOLO26s V2 - design spec

Derived from the approved implementation plan `2026-10-05-eth-real-hardneg-yolo26s-v2.md` plus four user
rulings issued 2026-10-05. Task numbering follows the plan.

## Goal
Train and fairly evaluate a leakage-free YOLO26s V2 that differs from ETH-only V1 **only** by adding verified
real hard negatives, to test whether false positives fall without materially harming recall.

## Non-negotiables (from the plan)
- Plain YOLO26s, no P2 and no architecture change, initialised from the verified official `yolo26s.pt`.
- Identical V1 recipe: `imgsz=1024`, `epochs=50`, AdamW `lr0=1e-4`, `freeze=0`, `batch=8`, `nbs=32`,
  `seed=42`, identical loss/augmentation/deterministic values, identical internal validation set and
  checkpoint-selection rule, identical evaluator and 24 GiB per-process memory policy.
- Positives are the **exact** V1 leakage-free ETH positive train rows; no new positive domain may enter.
- Only verified real hard negatives with empty labels; no repeat factor beyond ruling 2, no oversampling, no
  source weighting, every unique image exposed equally.
- Excluded: synthetic, iPhone, D455 positive, Roboflow pseudo boxes, `controlled_capability`,
  `challenge_test`, `val|eth_unseen`, frozen real no-target evaluation images, and any path/SHA256 overlap
  with evaluation.
- The pre-training FP/FN diagnostic is explanatory only and cannot change membership or hyperparameters.
- `best.pt` is selected only by the internal location-disjoint validation mAP50-95 rule (fitness argmax must
  equal the mAP50-95 argmax, or the run is rejected).
- Decision thresholds: `FP/image` down >= 20% AND `val|eth_unseen` Recall change > -0.02 -> `USEFUL`;
  FP improves but Recall <= -0.05 -> `HARMFUL_TRADEOFF`; FP change < 10% with stable accuracy -> `LOW_VALUE`;
  otherwise `INCONCLUSIVE`.
- Commit experiment code and metadata only; never commit checkpoints or `_scratch_*` artifacts.

## Rulings (2026-10-05, frozen)
1. **Source names replaced.** The plan's `hardneg_train` / `hardneg2_train` cannot be verified:
   `hardneg2_train` exists nowhere, and `hardneg_train` has 86 manifest rows of which 53 files are missing
   both locally and remotely (`management/shuttle_detection/DATASET_RESOLUTION.md` sections 4 and 7). V2 uses
   the two verified pools `hard_negatives/raw` (36) and `hard_negatives2/raw` (60) minus the 7 names in
   `excluded.txt` -> **89 unique SHA256s**, 0 duplicates, 0 overlap with V1 train/val, and the local and
   remote copies agree on all 96 files by SHA256.
2. **`hard_negative_repeat = 8`.** repeat=1 exposes only +0.61% of the training set and leaves the planned
   endpoint underpowered; repeat=8 exposes +712 rows (+4.9%). No other oversampling or weighting is allowed.
3. **Primary endpoint = pooled real no-target FP/image.** 496 frozen real no-target images
   (`real_images/backgrounds` 30, `real_images/raw` 59, `real_video/frames` 150, `real_match_frames/images`
   235, `real_train/raw` 22) reported with raw FP counts and a 95% confidence interval, keeping the per-set
   breakdown. `val|eth_unseen` accuracy remains the secondary endpoint.
4. **Reuse the existing machinery.** `tools/train_eth_only_v1.py` (selection built in) is the launcher;
   `tools/select_eth_only_v1_checkpoint.py` from the plan does not exist and will not be created.

## Deviations from the plan text (explicit, never silent)
| Plan says | Repository reality | V2 does |
|---|---|---|
| `tools/train_eth_only_yolo26s_v1.py` with `--config/--data/--output/--smoke` | canonical launcher is `tools/train_eth_only_v1.py` with `--contract/--data-yaml/--project/--name/--out-manifest/--smoke/--mem-cap-gib` | reuse the canonical launcher (ruling 4); the mapping is recorded here |
| `tools/select_eth_only_v1_checkpoint.py` | selection lives inside the launcher (`select_best_epoch`, alias-aware) | no new tool |
| `test_v2_training_recipe_matches_v1(v1_cfg, v2_cfg)` compares keys such as `box/cls/dfl/mosaic` | those values live in the resolved official recipe JSON, not in the V1 yaml | parity asserted on the **resolved** training kwargs via `resolve_recipe`, which covers every frozen key and cannot drift |
| `--candidate-manifest <HISTORICAL_V1_MANIFEST>` | the pools carry `hard_negative_metadata.csv` (provenance) but no SHA256/path columns | V2 first freezes a candidate manifest `outputs/shuttle_capability/metrics/eth_real_hardneg_v2_hardneg_candidates.csv` (source, path, sha256, bytes, width, height, provenance) and audits that |
| hard negatives referenced by local paths | training runs on the remote A6000 | the training manifest uses the remote pool paths (`/home/T7/dgut/robot_sim/outputs/shuttle_capability/...`), verified byte-identical to the local copies (96/96 SHA256) |

## Tasks
1. Freeze the V2 contract (`configs/eth_real_hardneg_yolo26s_v2.yaml`) and prove V1 parity.
2. Build the leakage-free V2 dataset, hard-negative candidate manifest and blocking audit.
3. Produce the pre-training V1 FP/FN diagnostic (read-only).
4. Smoke-check and run V2 with the unchanged V1 training machinery (3-epoch smoke, then 50 epochs).
5. Evaluate V2 with canonical parity and measure the hard-negative trade-off (pooled FP/image + CI).
6. Write the decision report (`USEFUL` / `HARMFUL_TRADEOFF` / `LOW_VALUE` / `INCONCLUSIVE`).

## Completion gate
Task 1-3 complete with the Task 2 audit `PASS` before any training starts; evaluator parity tests green;
V1 positive membership preserved exactly (blocking); no evaluation overlap; the report claims nothing outside
the frozen interpretation scope.
