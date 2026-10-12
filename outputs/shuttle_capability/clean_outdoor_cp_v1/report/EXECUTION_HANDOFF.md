# CLEAN_OUTDOOR_CP_V1 execution handoff (agent, night run)

Plan: ETH_COCO_CLEAN_OUTDOOR_COPYPASTE_HARDNEG_V1_IMPLEMENTATION_PLAN.md (Tasks 1-10).
Requested by the user as "finish it, do not wait for me". This document is the morning handoff: what is done,
what is proven, what is blocked, and the exact next action for a human.

## 1. Bottom line

All ten tasks now have implementation + tests (93 tests green across the plan's seven test files). Nothing was
fabricated: no training run, no metric, no human review and no licence was invented. The experiment cannot start
because three gates need a human or a completed download:

| Gate | Status at handoff | What unblocks it |
|---|---|---|
| Source contract (V2 recipe/weights/manifests/evaluator) | **PASS** | nothing, verified by hash |
| COCO train2017 inventory | run in progress (download ~10.7/18 GB, then the index) | let the download + index finish |
| ETH foreground masks | **FOREGROUND_INSUFFICIENT** (0 approved / 100) | human review of the 120-instance contact sheet |
| COCO CLEAN backgrounds | not runnable yet (needs the COCO inventory) | index + human review |
| COCO outdoor hard negatives | **OUTDOOR_HARDNEG_INSUFFICIENT** (0 approved / 150) | index + human review |
| Sampling (frozen schedule) | **BLOCKED_BUDGET_CONFLICT** by design: no approved pools yet | the two reviews above |
| Training run | **NOT_EXECUTED** (no legitimate schedule, no approved pools) | the gates above |
| Canonical evaluation | **NOT_EXECUTED / PILOT_ONLY** (no checkpoint; no independent real outdoor test set exists) | a trained checkpoint + a real OOD set |

## 2. What each task now contains

| Task | Deliverable | State |
|---|---|---|
| 1 | tools/clean_cp_contract.py, configs/eth_real_clean_outdoor_copypaste_v1.yaml, audits/repo_contract_map.json | PASS, committed |
| 2 | tools/index_coco_train2017.py + leakage firewall, frozen_eval_index.csv (3,989 hashes) | code PASS, real run in progress |
| 3 | tools/prepare_eth_real_foregrounds.py, foreground_manifest.csv (10,627 candidates) | gate honestly closed |
| 4 | tools/audit_clean_outdoor_backgrounds.py | code PASS, needs COCO candidates + review |
| 5 | tools/audit_outdoor_hardnegatives.py | code PASS, needs COCO candidates + review |
| 6 | tools/generate_clean_outdoor_copypaste.py | code PASS (10 tests), no approved inputs yet |
| 7 | tools/build_single_clean_cp_train.py (15,255 slots/epoch, 50 epochs, replayable) | code PASS (11 tests) |
| 8 | tools/train_single_clean_cp_v1.py (preflight + frozen argv) | code PASS (9 tests), run NOT_EXECUTED |
| 9 | tools/evaluate_clean_cp_generalization.py | code PASS (8 tests), scoring NOT_EXECUTED |
| 10 | tools/report_single_clean_cp_v1.py | code PASS (8 tests) |

## 3. Frozen facts this work is pinned to (independently verified)

- V2 resolved recipe: configs/shuttle_detection/eth_real_hardneg_v2_resolved_args.yaml,
  sha256 41e96b029f250c12256743522af65e30dd4ac2cf5c62187f7513b4017c751c3b, copied verbatim from
  /home/dgut/.dsh-bench/runs/detect/runs_eth_hardneg_v2/full_v2_e50/args.yaml.
- V2 composition, derived from the manifests (never typed in): 13,992 positives + **551** ordinary negatives
  (550 coco_train_easy + 1 mislabelled ticino negative whose role column says official_positives) + 712 hard-negative
  exposures (89 x 8) = 15,255 rows. Internal val 2,920 rows, locations glc_2 / uetlibergstrasse_2.
- Official pretrained yolo26s.pt: 20,422,725 bytes, sha256 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b.
- 496-image real no-target pool reconstructed from the local sets: 89 + 150 + 235 + 22 = 496 (matches the plan).

## 4. The three things only a human can do next

1. **Review the ETH foreground contact sheet** (120 real shuttlecock crops, 20 per location across 6 locations):
   outputs/shuttle_capability/clean_outdoor_cp_v1/contact_sheets/foreground_masks/foreground_review_01.png
   (sidecar foreground_review_index.json, sha256 8fd1c160...04a7). Approving >=100 of them with a reviewer id and a
   mask per instance turns FOREGROUND_INSUFFICIENT into PASS.
2. **Review the COCO train2017 candidates** once section 1's download+index finish: pick >=100 CLEAN outdoor
   backgrounds (>=3 scene types, >=3 source clusters, bottom clean_score quartile) and >=150 hard negatives
   (>=3 distractor categories) at full resolution, recording reviewer_id and no_shuttle_reviewed for each.
   COCO has no shuttlecock category: the review is the only evidence that a picture has no shuttlecock.
3. **Confirm the two open policy questions**: (a) the CP allocation policy is "fill the remaining joint cap"
   (up to 2,798 replacement slots per epoch, positives floor 11,194 untouched); (b) the authoritative spec
   docs/superpowers/specs/2026-10-13-eth-coco-clean-outdoor-copypaste-hardneg-v1-design.md was never supplied, so
   the plan itself is the only normative document available (recorded as BLOCKED_MISSING_SPEC).

## 5. Two defects the first production run exposed (found and fixed, not hidden)

1. **Unbounded near-duplicate dedupe set.** The first COCO index pass grew to **27 GB RSS** during the LSH
   grouping phase and had to be killed after 58 minutes with no output (the machine has 58 GB and was down to
   1 GB free). The set existed only to avoid recomputing a pair that shares several bands; the union-find is
   idempotent, so it was removed. A benchmark on 118,287 synthetic hashes with injected duplicates now completes
   in **354 s** at flat memory and still recovers every injected group.
2. **Two decodes per image.** The inventory opened every image once for the size check and again for the average
   hash, so the first pass read **39 GB for a 19 GB split**. Both values now come from a single decode.

Neither defect was visible to the unit tests, which is exactly why the real run mattered; both are now fixed and
the index was re-run. The first pass produced no artifact, so nothing downstream was built on it.

## 6. Honest deviations and notes

- **Task 6 TDD order**: the module and its tests were written together, so a captured RED run for that task does
  not exist (Tasks 1, 2, 7, 8, 9, 10 do have captured RED). The tests were the specification and all pass.
- **Step arithmetic**: the canonical repository arithmetic counts floor(images/batch) dataloader batches per epoch
  (1,906) while real optimizer steps are ceil(batches / gradient_accumulation) = 477; both are recorded by the
  launcher on purpose, and the authoritative number must come from a real len(dataloader) trace during the smoke run.
- **Data placement**: full COCO inventories stay on the resource host (/home/T7/dgut/robot_sim/data/clean_outdoor_cp_v1);
  only small audits, manifests and evidence thumbnails are committed locally.
- **Concurrent work**: another session was editing scripts/simulation/render_shuttle_3d_probe.py and outputs/three_d_render_probe/
  during this run; those files were never staged by this work.
