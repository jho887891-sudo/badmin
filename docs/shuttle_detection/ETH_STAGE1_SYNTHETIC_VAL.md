# ETH Stage1 YOLO26s | Same-domain synthetic validation

GPT-6 designs and patches code. DeepSeek/Trae executes real-server CPU/GPU work and returns evidence.

## Contract and caveats

Previously Stage1 trained on solid-colour BBox composites and validated on real photos (9 Epochs with zero reported validation P/R/mAP). This opt-in *new* mode makes train and validation both use unscaled BBox crops pasted on the same 1024x1024 solid-colour PALETTE. This helps measure learning of the synthetic task only; real-scene accuracy remains UNKNOWN.

Source splitting occurs before copy/paste. For each (location,set_dir) image sequence, use the first ~80% of frame indices for training and last ~20% for validation, with a 10-frame-number gap on each side of the boundary. Excluded boundary frames are embargoed, not trained or validated. Same original source image SHA and instance ID may never cross train/val/embargo. This is **temporal blocked holdout within a recording, not independent-video holdout**. It does not establish statistical independence from correlated frames.

Train mode is balanced-full across all 6 scenes x 10 relative-time bins, with repeated exposures inside train only. Validation mode is full, one composite per held-out source instance. Both use the existing pixel-exact BBox crop method with no masks, no GrabCut, no scaling and no extra Ultralytics augmentation. Because sources are held out, the 13,992 original instances are no longer all used as training sources: report train, val and embargo counts separately.

The previous real-val train entry tools/train_eth_stage1_bbox.py is unchanged; use it for comparison only.

## CPU generation

In a fresh isolated git worktree at the committed branch revision:

    python -m pytest -q tests/test_eth_stage1_synthetic_holdout.py tests/test_eth_stage1_bbox_train.py tests/test_eth_stage1_bbox_preview.py

    python tools/build_eth_stage1_synthetic_holdout.py \
      --inventory /home/dgut/eth_audit/out/inventory_20261026T074352Z/instances.csv \
      --image-root /home/T7/dgut/robot_sim/eth_shuttle_detection \
      --output-root /home/dgut/eth_audit/out/stage1_yolo26s_synthetic_80_20_NEW \
      --expected-instances 13992 --train-fraction 0.8 \
      --embargo-frames 10 --time-bins 10 --seed 20261027 \
      --max-train-outputs 100000

Stop BLOCKED if any scene/time bucket becomes empty, if max output budget is exceeded, if source SHA overlap exists, or if any source image/label SHA mismatches. Never silently alter the split or reuse a partly generated directory. Generation is not training.

Inspect split_audit.json, train_inventory.csv, val_inventory.csv, embargo_inventory.csv, train_synthetic/manifest.csv, val_synthetic/manifest.csv, both audit.json and contact_sheet.png and SHA256SUMS. Full visual quality review needs human evidence; otherwise UNKNOWN.

## Dry-run and GPU

With the actual official YOLO26s Detect model file and a real on-disk SHA256:

    MODEL=/verified/path/yolo26s.pt
    MODEL_SHA=$(sha256sum "$MODEL" | awk '{print $1}')

    python tools/train_eth_stage1_synthetic_holdout.py \
      --split-root /home/dgut/eth_audit/out/stage1_yolo26s_synthetic_80_20_NEW \
      --model "$MODEL" --model-sha256 "$MODEL_SHA" \
      --run-root /home/dgut/eth_audit/out/yolo26s_stage1_synthetic_smoke_NEW \
      --expected-original 13992 --time-bins 10 --epochs 1 --batch 4

Dry-run status must be DRY_RUN_PASS_NO_TRAINING and must not create training output or CUDA context. On success and ONLY if GPU execution is authorized, append --execute for 1 Epoch GPU smoke. Check emitted dataset.yaml points train to train_synthetic/images and val to val_synthetic/images. Inspect Ultralytics actual model architecture, results.csv, args.yaml, epoch losses, P/R/mAP, weight checksums and log exit codes. GPU smoke PASS may authorize a new 30 Epoch job with --batch 8 in ANOTHER new run-root. Keep original YOLO26s weights and historical checkpoints intact.

The launcher inherits the existing explicit AdamW, lr0=0.0001, warmup_bias_lr=0 configuration. Compared with previous 9-Epoch logs, this experiment changes both the data partition/validation domain and already-patched warmup setting; it is NOT a strict one-variable comparison. Independent real-video testing still required. No automatic deployment. Never modify main.
