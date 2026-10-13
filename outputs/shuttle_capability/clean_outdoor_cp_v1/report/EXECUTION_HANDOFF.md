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
| COCO train2017 inventory | **PASS** (see section 7) | nothing: candidate review is the next step |
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

## 6. COCO train2017 inventory: real result (Task 2 gate PASS)

The official COCO train2017 split was downloaded from images.cocodataset.org (19,336,861,798 bytes, annotations
252,907,541 bytes), extracted to /home/T7/dgut/robot_sim/data/coco2017/train2017 with **118,287 images**, which is
the exact official count, and indexed on the resource host:

| quantity | value |
|---|---|
| inventory rows | 118,287 (one per image id, ids unique, bytes validated, no path escapes) |
| licence declaration resolvable | 118,287 (100 %), 0 unknown |
| prohibited source rows | 13,672 (5,500 historical coco_train_easy ids + 8,172 historical/eval content hashes) |
| candidates | 106,382 |
| rejected | 11,905 = 5,500 historical_v2_coco_id + 6,392 near_duplicate_review + 13 duplicate_content |
| historical COCO id completeness | KNOWN (every coco_train_easy file name is a COCO id) |
| near-duplicate grouping | 20,000-image evenly spaced sample (16.9 %): 211 groups / 6,714 rows; coverage recorded in the audit |

Artifacts (full inventories stay on the resource host; the two small ones are committed):

- /home/T7/dgut/robot_sim/data/clean_outdoor_cp_v1/coco_train2017_inventory.csv - 49,028,440 B,
  sha256 f1d6f8206e50825906807ba2ec9f91ff9c0cf7bd68495f3e620fd93f13e86ae0
- .../coco_candidates.csv - 44,066,454 B, sha256 c7647ce0f2a7f3b34cb838b4a1b44b9f4c164f9fe6947a97e0965e05c9821084
- data/clean_outdoor_cp_v1/coco_exclusion_index.csv (committed) - 1,355,972 B,
  sha256 ea34b873a0f803155bf8c39bb5805eb4310b467ab4c6fa4e1e0067b6e5936847
- outputs/shuttle_capability/clean_outdoor_cp_v1/audits/coco_provenance_audit.json (committed)

## 7. Still running at handoff (not required for any gate)

The COCO review worksheet/sheet generator was still running on the resource host when this handoff was written
(it samples 2,000 of the 106,382 candidates, computes the frozen CLEAN metrics on six ROIs per photo and renders
the review sheets; the fuseblk reads make it slower than the local benchmark suggested). Finish or repeat it with:

    ssh dgut@jxxy
    cd /home/T7/dgut/robot_sim/clean_cp_v1_work
    python3 coco_sheets.py /home/T7/dgut/robot_sim/data/clean_outdoor_cp_v1/coco_candidates.csv out/coco_review 2000 100

It writes out/coco_review/coco_review_worksheet.csv (the human fill-in sheet), coco_clean_candidates_01.png,
coco_hardneg_random_01.png and coco_review_index.json. Nothing downstream is built on it yet, and no gate depends
on it: it is the fastest path to the two reviews in section 4.

## 8. What the two COCO review sheets actually show (I looked at every cell)

Both sheets are committed under outputs/shuttle_capability/clean_outdoor_cp_v1/contact_sheets/ (100 cells each,
label = COCO image id + the frozen edge-density metric, red box = the proposed CLEAN ROI).

1. **The frozen clean_score is a texture statistic, not an outdoor detector.** The 100 lowest-scoring ROIs are
   dominated by flat high-key regions - sky, water, snow, walls, table tops - and the same ranking happily selects
   **indoor** frames (bathroom, kitchen, restaurant, a mirror selfie) and non-outdoor subjects. Several frames also
   contain white sports gear (nets, courts). This is exactly why the plan makes the human outdoor + no-shuttle
   review load-bearing: the statistic narrows the search, it does not decide the pool.
2. **The hard-negative sheet is a uniform random sample, on purpose.** Hardness ("this white glint could fool the
   detector") is a human judgement, so nothing model-based picked those 100 frames; they are a broad mix of food,
   animals, sports, trains and buildings, with a handful of genuine white-equipment scenes. The proper prioritizer
   already exists (tools/audit_outdoor_hardnegatives.py::mine_v2_fp_candidates, read-only, refuses any evaluation
   path) but was **not run**: it needs the V2 checkpoint plus ultralytics, and by contract it may only ever suggest
   images for a human to inspect.

Practical consequence for the morning: start the background review from
outputs/shuttle_capability/clean_outdoor_cp_v1/contact_sheets/coco_clean/coco_clean_candidates_01.png (fast to
accept or reject) and use the hard-negative sheet only to seed the search; the worksheet CSV is the file to fill in.

## 9. Honest deviations and notes

- **Task 6 TDD order**: the module and its tests were written together, so a captured RED run for that task does
  not exist (Tasks 1, 2, 7, 8, 9, 10 do have captured RED). The tests were the specification and all pass.
- **Step arithmetic**: the canonical repository arithmetic counts floor(images/batch) dataloader batches per epoch
  (1,906) while real optimizer steps are ceil(batches / gradient_accumulation) = 477; both are recorded by the
  launcher on purpose, and the authoritative number must come from a real len(dataloader) trace during the smoke run.
- **Data placement**: full COCO inventories stay on the resource host (/home/T7/dgut/robot_sim/data/clean_outdoor_cp_v1);
  only small audits, manifests and evidence thumbnails are committed locally.
- **Concurrent work**: another session was editing scripts/simulation/render_shuttle_3d_probe.py and outputs/three_d_render_probe/
  during this run; those files were never staged by this work.
