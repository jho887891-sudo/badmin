# CLEAN_OUTDOOR_CP_V1 plan compliance audit (independent review gate)

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md, Task 10 Step 5
("Final independent review gate: reopen the authoritative Spec and audit every clause for a matching
artifact/test"). The authoritative spec file named by the plan
(docs/superpowers/specs/2026-10-13-eth-coco-clean-outdoor-copypaste-hardneg-v1-design.md) was never supplied, so
the plan itself is the only normative document available. That gap is recorded as BLOCKED_MISSING_SPEC in
outputs/shuttle_capability/clean_outdoor_cp_v1/audits/repo_contract_map.json.

Every "enforced by" entry names a test that fails if the behaviour regresses.

## Global constraints

| # | Clause | State | Enforced by / artifact |
|---|---|---|---|
| 1 | Exactly one model (CLEAN_OUTDOOR_CP_V1); no A0, no retrain, no curriculum, no HPO; formal run starts from the official yolo26s.pt | ENFORCED | tests/test_clean_cp_training_contract.py::test_single_run_only, tests/test_clean_cp_contract.py::test_no_a0_or_p2, test_one_official_pretrained_initialization |
| 2 | Frozen model/recipe: imgsz 1024, epochs 50, AdamW, lr0 1e-4, freeze 0, batch 8, nbs 32, seed 42; every other field copied from the V2 resolved config | ENFORCED | tools/clean_cp_contract.py::assert_frozen_recipe + tests/test_clean_cp_contract.py::test_recipe_parity_rejects_optimizer_auto, test_new_config_passes_the_frozen_recipe |
| 3 | V2 baseline 15,255 slots (13,992 / 551 / 712), internal val 2,920 unchanged, steps verified from the real loader (not assumed) | ENFORCED (arithmetic), MEASUREMENT PENDING | tests/test_clean_cp_contract.py::test_v2_counts_13992_551_712, test_val_2920_unchanged; steps: tools/train_single_clean_cp_v1.py records both conventions and defers to a real dataloader trace |
| 4 | New CP foregrounds only from V2 train-only ETH real instances with reviewed masks; bbox is not a mask; no eval/iPhone/D455/Roboflow/3D assets | ENFORCED | tools/prepare_eth_real_foregrounds.py + tests/test_clean_cp_sources.py (6 Task 3 tests incl. test_bbox_crop_not_mask, test_eval_image_never_selectable) |
| 5 | New CP backgrounds only from COCO train2017, real outdoor, no shuttlecock, CLEAN low-texture ROI, historical V2/eval images excluded, human scene review, bottom-quartile clean_score | ENFORCED in code, GATE OPEN | tools/audit_clean_outdoor_backgrounds.py + tests/test_clean_cp_sources.py (Task 4 tests incl. test_old_eth_coco_image_excluded, test_clean_score_bottom_quartile_with_fixed_metrics) |
| 6 | New hard negatives only from COCO train2017 with white reflections/paper/plastic/feathers/foliage, per-image human no-shuttle confirmation, strictly empty labels, mutually exclusive with the CLEAN pool | ENFORCED in code, GATE OPEN | tools/audit_outdoor_hardnegatives.py + tests/test_outdoor_hardnegative_contract.py (9 tests + gate test) |
| 7 | Negative budget: keep 551 + 712 old exposures; new hard negatives at most once per unique image per epoch and at most 5% of slots; >= 11,194 original positives per epoch; no step inflation | ENFORCED | tools/build_single_clean_cp_train.py + tests/test_clean_cp_training_contract.py (11 tests incl. test_new_outdoor_hardneg_at_most_5pct, test_no_double_shuffle_or_loader_step_inflation) |
| 8 | Pool feasibility: >= 100 reviewed foregrounds, >= 100 clean backgrounds (>= 3 scene types, >= 3 clusters), >= 150 hard negatives (>= 3 categories, >= 3 clusters), >= 200 reviewed CP contact sheets; stop if short | PARTIAL: foreground candidates exist (10,627), reviews outstanding; contact sheets for the other pools require the COCO inventory | foreground: FOREGROUND_INSUFFICIENT (0/100); the other pools are gated in code |
| 9 | Leakage firewall across paths, SHA256, near-duplicates, locations, sessions, video neighbours; eval pixels never enter training | ENFORCED | tools/index_coco_train2017.py::filter_coco_candidates (+ LSH near-duplicate grouping), data/clean_outdoor_cp_v1/frozen_eval_index.csv (3,989 hashes), tests/test_coco_train2017_index.py (8 tests), tests/test_clean_cp_sources.py::test_eval_source_role_never_allowed |
| 10 | Selection only on the internal V2 val, fixed eval settings (imgsz 1024, conf 0.25, AP floor 0.001, NMS 0.7, max_det 300, equiv_size_640), 1,648-frame val|eth_unseen reported whole, 496-image FP <= 3, outdoor claims only with a real OOD set | ENFORCED in code; MEASUREMENT NOT_EXECUTED (no checkpoint) | tools/evaluate_clean_cp_generalization.py + tests/test_clean_cp_eval_integrity.py (8 Task 9 tests), tools/report_single_clean_cp_v1.py (Task 10 tests) |
| 11 | Single-arm science boundary: no causal claim for either intervention; historical differences are descriptive | ENFORCED | tools/report_single_clean_cp_v1.py::assert_no_causal_language + tests/test_clean_cp_eval_integrity.py::test_report_never_claims_copy_paste_caused_gain |
| 12 | Resources: A6000 24 GiB cap preferred; only scripts, configs, tests, review metadata and small reports are committed; weights/bulk images stay on the data store; never fabricate a pass | HONOURED | every gate above reports BLOCKED_*/INSUFFICIENT rather than PASS; full COCO inventories stay under /home/T7/dgut/robot_sim/data/clean_outdoor_cp_v1 |

## Review focus items

| # | Risk | Enforced by |
|---|---|---|
| 1 | COCO has no shuttlecock class, so an absent annotation is not proof of absence | tests/test_clean_cp_sources.py::test_coco_annotation_has_no_shuttle_is_not_proof_of_no_shuttle, tests/test_outdoor_hardnegative_contract.py::test_no_coco_shuttle_class_does_not_imply_empty_label |
| 2 | Historical COCO negatives re-entering / evaluation leakage (same content, new filename, crop) | tests/test_coco_train2017_index.py::test_existing_v2_coco_train_easy_id_rejected, test_existing_v2_negative_hash_rejected, test_same_content_different_filename_rejected, test_crop_or_near_duplicate_flagged_for_review |
| 3 | Copy-paste target vanishing while the label survives | tests/test_clean_cp_mask_and_labels.py::test_invisible_ball_has_no_positive_label, test_bbox_follows_post_composite_alpha, test_alpha_rgb_transform_same_frame |
| 4 | New negatives crowding out positives | tests/test_clean_cp_training_contract.py::test_original_positive_exposures_at_least_11194, test_new_outdoor_hardneg_at_most_5pct, test_old_ordinary_negative_551_and_old_hardneg_712_retained |
| 5 | One COCO image used in two new pools | tests/test_clean_cp_sources.py::test_same_coco_id_cannot_have_two_new_roles, tests/test_clean_cp_training_contract.py::test_no_coco_image_id_or_neardup_across_pools |

## Deliverables checklist

- [x] V2 contract/hash audit and model/evaluator parity tests
- [ ] Authoritative spec checked in - **BLOCKED_MISSING_SPEC** (never supplied with the plan)
- [x] ETH train-only foreground inventory (10,627 candidates) and a 120-instance contact sheet
- [ ] All 100 foreground masks human-approved - **FOREGROUND_INSUFFICIENT**
- [ ] COCO train2017 index/annotation hash + exclusion audit - index running, then needs review
- [ ] CLEAN outdoor ROI inventory with image_id, rights and review decisions - tool ready, review outstanding
- [ ] Outdoor hard-negative inventory with empty labels - tool ready, review outstanding
- [x] Source/leakage/near-duplicate audit machinery (frozen eval index + firewall + role audit)
- [ ] Copy-paste images/labels/manifests + replay QA - generator ready, no approved inputs
- [ ] One frozen 50-epoch schedule and single run - schedule builder ready; run **NOT_EXECUTED**
- [ ] Independent real outdoor OOD set - **does not exist**; Task 9 therefore can only report PILOT_ONLY
- [ ] Frozen eval P/R/mAP/FP and size-bucket diagnostics - adapter ready; **NOT_EXECUTED** (no checkpoint)
- [ ] Final report with one-run claims only - generator ready; the decision at handoff is PILOT_ONLY

## Verdict at handoff

The experiment is **not executed and cannot honestly be executed yet**: three gates (foreground masks, COCO clean
backgrounds, COCO hard negatives) need a human, and the COCO inventory needs the download plus the index to finish.
Everything that could be built, tested and verified without fabricating a pass has been; 93 tests across the plan's
seven test files pass.
