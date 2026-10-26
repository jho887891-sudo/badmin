# ETH Four-Stage Shuttlecock Copy-Paste Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing ETH foreground/alpha Copy-Paste code into a reproducible four-stage YOLO curriculum: (1) non-white solid backgrounds; (2) motion/exposure degradation; (4) multi-environment Copy-Paste pool; (3) original-background final fine-tuning, with real-video evidence and frozen eval isolation.

**Architecture:** Reuse the audited ETH inventory/mask validator and RGBA compositor, exposing additive *opt-in* four-stage modes rather than replacing frozen CLEAN_OUTDOOR_CP_V1 behavior. Add source-policy, masking, stage dataset assembly, gated training and evaluation adapters; keep every existing frozen experiment untouched. Retain the user's Stage 3 wording and scope; Stage 4 runs *before* the Stage 3 final fine-tune as a dataset dependency, though its identifier remains Stage 4.

**Tech Stack:** Python 3, pytest, numpy, PIL/Pillow, pandas, PyYAML, existing Ultralytics runner/evaluator; OpenCV only if an explicit dependency preflight establishes that it is available on the execution host. Do not presume GPU or raw ETH files are mounted in the current session.

**Spec:** `docs/superpowers/specs/2026-10-25-eth-four-stage-shuttle-copypaste-design.md` (approved in chat 2026-10-26).

## Global Constraints

- Use `feature/eth-four-stage-copypaste` based on `feature/shuttle-detection`, not `main`; never modify existing checkpoint binaries, manifests, outputs or frozen `CLEAN_OUTDOOR_CP_V1` defaults.
- All eligible ETH **TRAIN** positive bboxes appear in inventory, including tiny/blurred ones; never silently exclude via existing `min_short=10`, `min_long=24` defaults. Evaluation-origin content and derivative images must never enter new training pools.
- Bbox is not an instance mask. No hard-rectangle crop gets auto-approved; each mask requires a traceable human review or remains PENDING/REJECTED. For unrecoverable tiny targets preserve the original ETH training image for Stage 3.
- Reuse `tools/prepare_eth_real_foregrounds.py` and `tools/generate_clean_outdoor_copypaste.py` where compatible. Existing functions and old default CLI behavior must remain backwards-compatible.
- Real-photo foregrounds must not be magnified to invent feather detail; existing mask validator default `max_upscale=1.0` remains intact.
- Stage 1 uses multiple explicitly enumerated non-white/near-white-excluding solid colors. Stage 2 keeps motion blur and multi-exposure/processing ghosting distinct. Stage 4 uses audited real train-safe ROI backgrounds with provenance and label integrity.
- Keep the user-defined **Stage 3: 回到真实背景，做最终训练 / 微调** intact: ETH original backgrounds, real badminton venues, lamps/reflections/lines/stands/walls; transfer Stage 1/2 subject representation and fine-tune on real data. Validate real-video recall, false positives, lamp FP reduction and blurry-shuttle improvement. Stage 4 can provide an *additional* pool for a controlled Stage 3 training comparison.
- Primary input size `imgsz=1024`; use whichever model/checkpoint is actually selected via a validated checkpoint SHA, never silently swap YOLOv8s for frozen YOLO26s. Freeze all train/eval knobs *within each comparison*; avoid inventing checkpoint addresses.
- Use the existing frozen eval semantics when appropriate: IoU 0.5 primary, center-distance secondary; preserve known NMS and confidence settings once selected and audit parity. Without framewise human GT: report real-video Recall = `UNKNOWN`.
- `AGENTS.md`: bulk images/weights stay on verified remote/other storage, not Git; only code, config, audit reports, evidence thumbnails and manifest/hash references enter repository. Never delete source until destination access + counts + hashes verified.
- TDD for code changes, failing test first, then minimal implementation, run precise tests and legacy regression suite, commit after each task. Document `PASS`, `FAIL`, `BLOCKED` or `UNKNOWN` with evidence; absence of GPU is not a passing GPU evaluation. No robot deployment.

## File Map

| Path | Role |
|---|---|
| `configs/eth_four_stage_copypaste_v1.yaml` (new) | Versioned stage config, background palette, data sources, train/eval settings, limits, outputs |
| `tools/eth_four_stage_contract.py` (new) | Four-stage source roles, provenance, dataset/version/checkpoint preflight, SHA, split firewalls |
| `tools/prepare_eth_real_foregrounds.py` (modify additive) | Exhaustive ETH train bbox enumeration and per-instance dispositions |
| `tools/propose_eth_instance_masks.py` (new) | Box-guided mask proposals only; never mark human reviewed |
| `tools/generate_clean_outdoor_copypaste.py` (modify additive) | Reuse RGBA/mask-aligned compositor; add optional external background/retained ROI policies without changing legacy default |
| `tools/eth_four_stage_solid.py` (new) | Stage 1 palette sampling, solid canvas, visible bbox/label, manifest |
| `tools/eth_four_stage_degrade.py` (new) | Stage 2 motion blur, ghosting, photo noise/compression and mask/box propagation |
| `tools/eth_four_stage_environments.py` (new) | Stage 4 audited-background scheduler, composites and provenance QA |
| `tools/build_eth_four_stage_train.py` (new) | Stage 3 real-first set assembly, E3/E4 balanced comparison manifests |
| `tools/train_eth_four_stage.py` (new) | Separate gated stage1/stage2/stage3 trainer, dry-run and explicit execute |
| `tools/eval_eth_four_stage.py` (new) | Reuse canonical bbox matching/evaluation and real-video FP/motion/lamp analyses |
| `tests/test_eth_four_stage_*.py` (new) | Isolated pytest fixtures/tests for every change |
| `docs/superpowers/specs/2026-10-25-eth-four-stage-shuttle-copypaste-design.md` (read-only) | Approved design; do not rewrite |
| `docs/superpowers/plans/2026-10-26-eth-four-stage-copypaste-implementation.md` | This plan |

## Review Focus

1. Tiny boxes quantize to 0 px / masks cannot be recovered: Task 2 must *inventory* these with explicit reason, not silently discard or pretend foreground exists.
2. A proposed mask is all-opaque or copied from bbox: Task 3 rejects it as usable data; no unreviewed row marked APPROVED.
3. Synthetic target becomes invisible / bbox stretches outside image after blur or ghosting: Tasks 4 and 5 reject positives with empty support and log why; no label leak.
4. Background is actually from frozen eval or has unlabeled shuttlecock: Tasks 1 and 6 block use even if photo otherwise looks clean.
5. Stage 3 E4 receives different real data, training steps, or test-set-driven checkpoint selection than E3: Tasks 7-9 block/report mismatch and never claim E4 improvement from a biased comparison.

---

### Task 1: Four-Stage Experiment Contract and Source Firewall

**Files:**
- Create: `tools/eth_four_stage_contract.py`
- Create: `configs/eth_four_stage_copypaste_v1.yaml`
- Test: `tests/test_eth_four_stage_contract.py`

**Interfaces:**
- Consumes: source paths and role metadata from existing `tools/clean_cp_contract.py`; hash records from existing inventory.
- Produces: `load_stage_contract(path: str | Path) -> dict`, `audit_training_sources(rows: list[dict], frozen_rows: list[dict]) -> dict`, `assert_stage_preflight(config: dict, source_report: dict, checkpoint: dict | None = None) -> dict`.

- [ ] **Step 1: Write failing tests** `test_rejects_eval_derived_source_even_with_new_path`, `test_stage_ids_and_palette_required`, `test_missing_checkpoint_blocks_training_not_inventory`, and `test_legacy_frozen_config_unchanged`. Include same SHA in eval+train paths, empty palette, missing stage IDs and valid stage-1 dry run fixtures.
- [ ] **Step 2: Run tests and observe RED.** Run: `pytest -q tests/test_eth_four_stage_contract.py`. Expected: missing new module/interface failures.
- [ ] **Step 3: Add independent contract** with `stage1`, `stage2`, `stage4`, `stage3`, explicit ETH train-only roles, forbidden eval role/SHA lists, palette, output roots, stage-specific seed and version; implement functions above. The existing `clean_cp_contract.py` frozen experiment continues to load without changes.
- [ ] **Step 4: Verify GREEN and legacy source contract.** Run: `pytest -q tests/test_eth_four_stage_contract.py tests/test_clean_cp_contract.py tests/test_clean_cp_sources.py`. Expected: PASS for all; no legacy contract changes.
- [ ] **Step 5: Commit.** `git add configs/eth_four_stage_copypaste_v1.yaml tools/eth_four_stage_contract.py tests/test_eth_four_stage_contract.py && git commit -m "feat: add isolated four-stage ETH source contract"`

### Task 2: Exhaustive ETH Train Bbox Inventory

**Files:**
- Modify: `tools/prepare_eth_real_foregrounds.py` (add a new exhaustive interface; leave old defaults)
- Test: `tests/test_eth_four_stage_inventory.py`

**Interfaces:**
- Consumes: Task 1 `audit_training_sources`, original `read_manifest`, `parse_yolo_label`, `_pixel_box`, frozen-eval exclusion logic.
- Produces: `enumerate_eth_train_instances(v2_manifest: str | Path, *, label_root: str | Path | None, frozen_eval_manifests: tuple[str | Path, ...]) -> dict` with `instances`, `rejections`, `counts`, each train-box `instance_id`, `source_sha256`, bbox, and `status/reason`.

- [ ] **Step 1: Write failing tests** `test_two_boxes_same_frame_have_distinct_ids`, `test_tiny_box_included_not_filtered_by_old_thresholds`, `test_zero_area_roundoff_has_explicit_reason`, `test_eval_only_frame_excluded_with_reason`, and `test_inventory_reconciles_all_train_label_boxes`. Synthetic 200x120 fixture must include both 5x10 and 30x60 positive boxes.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_inventory.py`. Expected: `enumerate_eth_train_instances` unavailable.
- [ ] **Step 3: Implement new enumeration** by reusing existing parser and split/provenance checks but without old 10/24 size floors. Stable unique ID must incorporate source hash plus bbox index, not just stem. Return every authorized train instance with exact disposition. Invalid/missing/round-to-zero boxes become auditable invalid entries, not positive composites. Missing ETH raw files/labels must block rather than generate an empty apparent success.
- [ ] **Step 4: GREEN and old behavior parity.** `pytest -q tests/test_eth_four_stage_inventory.py tests/test_clean_cp_sources.py`. Existing `audit_eth_train_instances` with defaults must behave unchanged.
- [ ] **Step 5: Commit.** `git add tools/prepare_eth_real_foregrounds.py tests/test_eth_four_stage_inventory.py && git commit -m "feat: inventory all ETH training boxes without silent size filtering"`

### Task 3: Mask Proposal and Quality-Control Handoff

**Files:**
- Create: `tools/propose_eth_instance_masks.py`
- Test: `tests/test_eth_four_stage_masks.py`

**Interfaces:**
- Consumes: Task 2 instance record and original `validate_foreground_mask(image, bbox, alpha, reviewed, source_size=...)`.
- Produces: `propose_instance_mask(image: np.ndarray, bbox_xyxy: tuple[int, int, int, int], *, method: str = "grabcut") -> dict` with `candidate_alpha` or `None`, `status=PENDING_REVIEW|UNRECOVERABLE`, `reason`, algorithm version; `validate_reviewed_candidate(record: dict, image: np.ndarray, mask: np.ndarray) -> dict`.

- [ ] **Step 1: Write failing tests** `test_candidate_is_never_auto_approved`, `test_hard_rectangle_not_a_valid_mask`, `test_tiny_unsegmentable_instance_remains_recorded`, `test_approved_mask_requires_human_boolean_and_id`, `test_mask_geometry_aligns_to_original_image`. Use synthetic ellipse and a 1px nonrecoverable target.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_masks.py`. Expected: missing module/interfaces.
- [ ] **Step 3: Implement candidate generator** via optional OpenCV GrabCut box-guided proposal; dependency must be checked in dry-run and reported `BLOCKED_MASK_BACKEND` if absent, not silently fabricate an alpha. Output full-frame mask coordinates. Leverage existing validator and mandatory manual `reviewer_id`, `review_method`; provide compact contact sheet index for review. Unrecoverable objects return no mask, but their raw train images remain eligible for Stage 3.
- [ ] **Step 4: GREEN + legacy QA regression.** `pytest -q tests/test_eth_four_stage_masks.py tests/test_clean_cp_sources.py tests/test_clean_cp_mask_and_labels.py`. Expected: PASS.
- [ ] **Step 5: Commit.** `git add tools/propose_eth_instance_masks.py tests/test_eth_four_stage_masks.py && git commit -m "feat: propose mask candidates with explicit human QA gate"`

### Task 4: Stage 1 Non-White Solid-Background Copy-Paste

**Files:**
- Create: `tools/eth_four_stage_solid.py`
- Modify: `tools/generate_clean_outdoor_copypaste.py` (ONLY additive reusable primitives / optional background mode; default `coco_train2017` unchanged)
- Test: `tests/test_eth_four_stage_solid.py`

**Interfaces:**
- Consumes: Task 3 approved RGBA foregrounds, Task 1 palette/seed; existing `composite_real_shuttle`, `write_yolo_label`, `bbox_from_visible_alpha`.
- Produces: `sample_solid_rgb(palette: list[tuple[int,int,int]], seed: int) -> tuple[int,int,int]`; `create_stage1_sample(fg_rgb: np.ndarray, alpha: np.ndarray, *, canvas_size: tuple[int,int], color: tuple[int,int,int], center: tuple[float,float], seed: int) -> dict` with `image`, `label`, `visible_alpha`, `metadata`.

- [ ] **Step 1: Write failing tests** `test_palette_excludes_white_and_near_white`, `test_multiple_non_white_colors_appear_over_seed_set`, `test_visible_bbox_matches_yolo_label`, `test_transparent_pixels_keep_exact_background`, `test_invisible_positive_rejected`, `test_legacy_coco_mode_still_works`.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_solid.py`. Expected: new stage API unavailable.
- [ ] **Step 3: Implement Stage 1** using solid canvas plus the existing same-RGBA-transform compositor. Define palette explicitly in config; reject (255,255,255) and near-white by validated palette policy, not pixel-based background recoloring. No categorical target-size hardcode; log actual visible bbox size and alpha support; never enlarge raw ETH cutouts beyond their verified resolution. Stage manifest includes parent foreground, background RGB, seed, alpha/label/image SHA, status/rejection reason.
- [ ] **Step 4: GREEN and legacy tests.** `pytest -q tests/test_eth_four_stage_solid.py tests/test_clean_cp_mask_and_labels.py`. Expected: PASS; prior COCO copy-paste behavior unmodified.
- [ ] **Step 5: Commit.** `git add tools/eth_four_stage_solid.py tools/generate_clean_outdoor_copypaste.py tests/test_eth_four_stage_solid.py && git commit -m "feat: add non-white solid-background copy paste mode"`

### Task 5: Stage 2 Realistic Degradation With Mask and Label Tracking

**Files:**
- Create: `tools/eth_four_stage_degrade.py`
- Test: `tests/test_eth_four_stage_degrade.py`

**Interfaces:**
- Consumes: Stage 1 sample (`image`, `visible_alpha`, `metadata`), Task 1 degradation settings.
- Produces: `apply_motion_blur_rgba(rgb: np.ndarray, alpha: np.ndarray, *, length_px: float, angle_deg: float) -> tuple[np.ndarray,np.ndarray]`; `apply_ghosting_rgba(rgb: np.ndarray, alpha: np.ndarray, *, displacement: tuple[float,float], weights: tuple[float,float]) -> tuple[np.ndarray,np.ndarray]`; `degrade_stage2_sample(sample: dict, params: dict, seed: int) -> dict`.

- [ ] **Step 1: Write failing tests** `test_zero_blur_is_identity`, `test_direction_changes_tail_axis`, `test_ghosting_differs_from_single_exposure_blur`, `test_bbox_recomputed_from_final_visible_alpha`, `test_overblurred_invisible_target_has_no_positive_label`, `test_seed_replays_identically`.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_degrade.py`. Expected: missing module.
- [ ] **Step 3: Implement separate degradation operators** using physically motivated directional exposure integration and separately parameterized double-image ghosting. Keep RGB and alpha geometrically aligned; compute visible bbox after degradation and reject empty/invalid support. Apply global exposure, gamma, noise, and compression to the completed image, preserving a config log. Bound operator parameters using real ETH video observations once measured; do not assert numeric realism without supporting observation.
- [ ] **Step 4: GREEN.** `pytest -q tests/test_eth_four_stage_degrade.py tests/test_eth_four_stage_solid.py`. Expected: PASS.
- [ ] **Step 5: Commit.** `git add tools/eth_four_stage_degrade.py tests/test_eth_four_stage_degrade.py && git commit -m "feat: add stage2 motion and imaging degradation"`

### Task 6: Stage 4 Multi-Environment Copy-Paste Pool

**Files:**
- Create: `tools/eth_four_stage_environments.py`
- Modify: `tools/generate_clean_outdoor_copypaste.py` (additional audited train-safe real-background path; old explicit legacy policy preserved)
- Test: `tests/test_eth_four_stage_environments.py`

**Interfaces:**
- Consumes: train-safe ETH/real venue background manifest, approved target foregrounds and Stage 2 options, `audit_training_sources`, existing RGBA compositor.
- Produces: `validate_real_background(record: dict, frozen: list[dict]) -> dict`; `make_stage4_sample(foreground: dict, background: dict, placement: dict, seed: int) -> dict` with composite image, alpha-derived label, environment type, source hashes, ROI, seed, transform ledger.

- [ ] **Step 1: Write failing tests** `test_eval_background_blocked_by_sha`, `test_unreviewed_roi_blocked`, `test_unlabeled_shuttle_in_background_blocked`, `test_lamp_background_retained_with_valid_roi`, `test_source_provenance_and_final_bbox_saved`, `test_legacy_only_coco_policy_unchanged`.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_environments.py`. Expected: module/interface not found.
- [ ] **Step 3: Implement source-aware background pool**, explicitly distinguish ETH train original background, real badminton venues and approved difficult background types (lamps/reflections/white lines/stands/walls). Use human-reviewed empty ROI and leakage firewall; reject ambiguous/unlabeled original target. Composite with same-RGBA transforms; for global noise/compression apply after image composition, and record all target/background parameters. Reject hard halos/invalid labels; keep per-environment QA contact sheets.
- [ ] **Step 4: GREEN and historical modes.** `pytest -q tests/test_eth_four_stage_environments.py tests/test_clean_cp_mask_and_labels.py tests/test_clean_cp_sources.py`. Expected: PASS.
- [ ] **Step 5: Commit.** `git add tools/eth_four_stage_environments.py tools/generate_clean_outdoor_copypaste.py tests/test_eth_four_stage_environments.py && git commit -m "feat: add audited multi-environment composites"`

### Task 7: Stage 3 Real-Background Fine-Tune Data and Fair E3/E4 Arms

**Files:**
- Create: `tools/build_eth_four_stage_train.py`
- Test: `tests/test_eth_four_stage_train_sets.py`

**Interfaces:**
- Consumes: Stage 1/2/4 manifests, unchanged original ETH train images, independently approved real venue train data, frozen evaluation blacklist, fixed comparison schedule.
- Produces: `build_stage_training_manifests(config: dict, pool_manifests: dict[str,str]) -> dict`; `assert_comparison_parity(e3: dict, e4: dict) -> dict`.

- [ ] **Step 1: Write failing tests** `test_stage3_retains_all_allowed_original_train_positives`, `test_unrecoverable_mask_sample_keeps_original_real_image`, `test_e3_e4_use_identical_real_image_ids`, `test_e3_e4_equal_exposure_and_steps`, `test_e4_only_adds_stage4_pool`, `test_eval_image_derivative_rejected`.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_train_sets.py`. Expected: missing builder.
- [ ] **Step 3: Assemble explicit stage manifests** without modifying the approved Stage 3 definition. Preserve all eligible raw ETH training images even when segmentation failed. Record per-epoch exposure counts of originals vs synthetic; E3 and E4 must use same real positives, seed, model initialization and optimizer step budget, with E4 only substituting a predeclared number of synthetic examples. Ratio is a config value frozen after pool inspection, not a guessed constant. Store ID-level parity/violation report.
- [ ] **Step 4: GREEN.** `pytest -q tests/test_eth_four_stage_train_sets.py tests/test_eth_four_stage_contract.py`. Expected: PASS.
- [ ] **Step 5: Commit.** `git add tools/build_eth_four_stage_train.py tests/test_eth_four_stage_train_sets.py && git commit -m "feat: assemble controlled stage3 real finetune comparisons"`

### Task 8: Gated Stage Trainer (Dry-Run By Default)

**Files:**
- Create: `tools/train_eth_four_stage.py`
- Test: `tests/test_eth_four_stage_training.py`

**Interfaces:**
- Consumes: Tasks 1 and 7 `assert_stage_preflight`, stage manifests and parity checks, selected checkpoint path+SHA, current Ultralytics training environment.
- Produces: `build_train_command(config: dict, stage: str, manifest_path: str, checkpoint_path: str, output_dir: str) -> list[str]`; `training_preflight(config: dict, inputs: dict) -> dict`; CLI `--stage {stage1,stage2,stage3_e3,stage3_e4} --preflight --dry-run` (default), and explicit `--execute`.

- [ ] **Step 1: Write failing tests** `test_default_dry_run_never_imports_gpu_stack_or_trains`, `test_missing_checkpoint_hash_blocks_execute`, `test_failed_mask_or_source_gate_blocks_execute`, `test_stage2_warm_starts_from_stage1`, `test_stage3_warm_starts_from_stage2`, `test_eval_sets_cannot_select_checkpoint`, `test_distinct_output_dirs_no_overwrite`.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_training.py`. Expected: missing entrypoints.
- [ ] **Step 3: Implement separate launcher** with explicit stage/checkpoint/seed/imgsz=1024/recipe/output/manifest validation, no inherited frozen `CLEAN_OUTDOOR_CP_V1` recipe assumptions, no automatic launch. Resolve physical batch only after remote GPU preflight; require approved config and verified real data path. Save resolved YAML, run metadata, optimizer-step and sample exposure logs, GPU memory, checkpoint SHA, and failure status.
- [ ] **Step 4: GREEN plus legacy trainer check.** `pytest -q tests/test_eth_four_stage_training.py tests/test_train_eth_only_v1.py tests/test_clean_cp_training_contract.py`. Expected: PASS. `python tools/train_eth_four_stage.py --help` must work without GPU; dry-run returns no optimizer steps.
- [ ] **Step 5: Commit.** `git add tools/train_eth_four_stage.py tests/test_eth_four_stage_training.py && git commit -m "feat: add safe four-stage YOLO training orchestrator"`

### Task 9: Source-Backed Evaluation and Stage-4 Ablation

**Files:**
- Create: `tools/eval_eth_four_stage.py`
- Test: `tests/test_eth_four_stage_evaluation.py`

**Interfaces:**
- Consumes: E0/E1/E2/E3/E4 checkpoint hashes, GT-based frozen real image/video manifests, existing `tools/eval_yolo26_v1.py` `match_greedy`, `equiv_size_640` and official eval settings.
- Produces: `evaluate_from_predictions(samples: list[dict], config: dict) -> dict`, `compare_e3_e4(e3_report: dict, e4_report: dict) -> dict`, and immutable per-run evaluation report/JSONL.

- [ ] **Step 1: Write failing tests** `test_no_gt_yields_recall_unknown`, `test_detection_count_not_confused_with_recall`, `test_identical_frozen_iou_center_logic`, `test_lamp_fp_count_and_blur_recall_reported_separately`, `test_e3_e4_mismatched_eval_manifest_refused`, `test_e4_improvement_requires_real_data_and_fp_guardrail`.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_evaluation.py`. Expected: missing adapter.
- [ ] **Step 3: Implement report adapter** reusing existing matcher (no separate conflicting bbox matcher), pinning model-version-appropriate NMS/confidence and fixed real GT split. Report TP/FP/FN, Recall/Precision, lamp FP, blurry-shuttle Recall, size bucket Recall, Top-K and optional video miss-streak. No GT -> `UNKNOWN`, not 0 or asserted success. Check evaluator parity on frozen reference before new checkpoint scoring, and record human label provenance.
- [ ] **Step 4: GREEN and evaluator regression.** `pytest -q tests/test_eth_four_stage_evaluation.py tests/test_eval_yolo26_v1.py tests/test_clean_cp_eval_integrity.py`. Expected: PASS; historical matcher behavior not altered.
- [ ] **Step 5: Commit.** `git add tools/eval_eth_four_stage.py tests/test_eth_four_stage_evaluation.py && git commit -m "feat: evaluate four-stage gains on frozen real GT"`

### Task 10: Integrated Pilot, QA and Reproducibility Handoff

**Files:**
- Create: `tests/test_eth_four_stage_integration.py`
- Create: `docs/shuttle_detection/ETH_FOUR_STAGE_RUNBOOK.md` (or `docs/shuttle_detection/` directory if missing)
- Test: `tests/test_eth_four_stage_integration.py`

**Interfaces:**
- Consumes: Tasks 1-9 dry-run interfaces and tiny synthetic fixture samples.
- Produces: end-to-end **CPU fixture** manifest, rejection/audit logs and dry-run commands; no actual ETH GPU results.

- [ ] **Step 1: Write failing integration tests** `test_tiny_fixture_pilot_stage1_stage2_stage4_stage3_manifest`, `test_replay_same_seed_same_hash`, `test_all_gates_block_without_remote_real_data`, `test_stage3_wording_preserved_in_runbook`, `test_old_frozen_configs_checksums_unmodified`.
- [ ] **Step 2: RED.** `pytest -q tests/test_eth_four_stage_integration.py`. Expected: missing fixtures/runbook or missing integration interface.
- [ ] **Step 3: Implement fixture pilot and runbook** with exact stage prerequisites, input manifests, paths, CLI flags, output schemas, PASS/FAIL/BLOCKED/UNKNOWN gates, human mask approval instructions, real-background review, 1024 inference settings, and E0-E4 reproducibility manifests. Reference source hashes and remote resource locations; do not commit bulk images or model binaries.
- [ ] **Step 4: GREEN and regression** `pytest -q tests/test_eth_four_stage_*.py tests/test_clean_cp_*.py tests/test_train_eth_only_v1.py tests/test_eval_yolo26_v1.py`. Expected: PASS. Run `git diff --check` and verify committed files do not include frozen configs/weights or root `main`.
- [ ] **Step 5: Commit.** `git add docs/shuttle_detection/ETH_FOUR_STAGE_RUNBOOK.md tests/test_eth_four_stage_integration.py && git commit -m "test: validate end-to-end four-stage dry run and handoff"`

## Remote Data/GPU Execution Gates (Not Executed by This Plan)

These are post-implementation operator phases; they must not be reported PASS unless actual remote evidence is returned.

| Phase | Prerequisites | Outputs | Gate |
|---|---|---|---|
| R0 Source inventory | Actual ETH raw train images, labels, frozen eval index mounted, SHA verified | `all_train_instances.csv`, excluded split report, source manifest | All eligible train bboxes accounted for, no eval derivative |
| R1 Mask pilot | Mask proposal backend available, human mask QA | approved/pending/rejected mask counts, contact sheet | No unreviewed approval; insufficient pool -> BLOCKED |
| R2 Stage 1 | Approved foregrounds + palette + storage | images, YOLO labels, sample provenance, Stage 1 checkpoint | Mask/label QA PASS, training complete and logged |
| R3 Stage 2 | R2 + observed blur/exposure distribution | degradation JSONL, examples, Stage 2 checkpoint | Motion/ghost operators independently inspected, label QA PASS |
| R4 Stage 4 | Approved no-target train backgrounds and ROI | multi-environment dataset, SHA, QA/rejections | Provenance/ROI/alpha labels pass; no eval leakage |
| R5 Stage 3 E3/E4 | Original real train set + R3/R4 + frozen recipes and matching exposures | E3/E4 best/last weights, step logs, normalized manifests | Training reaches target without drift or overwrite |
| R6 Final real eval | Fixed real GT (including annotated 336-frame video), frozen evaluator/parity and timestamps | JSONL/overlay, TP/FP/FN, lamp FP, blurred recall, per-environment report | Evidence-based real-world gains; otherwise FAIL/UNKNOWN |

Real GT for the 336-frame video remains `UNKNOWN` until humans actually annotate and QA it. Ground-truth split/frames cannot be selected after seeing E4's results. No model is deployed to hardware by this project.

## Dependency / Critical Path

`Task 1 → Task 2 → Task 3 → Task 4 → Task 5`

`Task 1 + Task 4 + Task 5 → Task 6` (Stage 4)

`Task 1 + Task 2 + Task 5 + Task 6 → Task 7 → Task 8 → Task 9 → Task 10`

The mask proposal pilot and real GT annotation can proceed on independent resources once data provenance is frozen; do not mistake parallel work for waived quality gates.

## Self-Review / Coverage

- Spec stages covered: Stage 1 Task 4, Stage 2 Task 5, Stage 4 Task 6, Stage 3 Tasks 7-8; final verification Tasks 9-10.
- Review Focus tests: tiny box Task 2; fake mask Task 3; invisible target Task 4/5; eval/unlabeled backgrounds Task 1/6; unfair E3/E4 comparison Task 7/9.
- Key types consistent: `audit_training_sources(rows, frozen_rows)`, `enumerate_eth_train_instances(...)`, `propose_instance_mask(...)`, `create_stage1_sample(...)`, `degrade_stage2_sample(...)`, `make_stage4_sample(...)`, `build_stage_training_manifests(...)`.
- New run's stage contract deliberately not using frozen `CLEAN_OUTDOOR_CP_V1` trainer gate. Old mode regression tests run before every merge.
- No GPU, real ETH extraction, model training or actual evaluation has been executed in this design/plan authoring step.

## Execution Handoff

Recommended: **subagent-driven** implementation, with fresh code review at each task boundary, because incorrect mask/source isolation can invalidate every training result. If subagents are unavailable, use `superpowers:executing-plans` and apply the same task-level test and review gates.

**Wait for explicit user review of this implementation plan before editing the existing code or launching data/training commands.**
