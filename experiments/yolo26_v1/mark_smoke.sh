#!/usr/bin/env bash
S=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_smoke_musgd_lr0001_20260927-211053
cat > "$S/DIAGNOSTIC_ONLY_STAGE_B_WARMUP.txt" <<EOF
DIAGNOSTIC_ONLY_STAGE_B_WARMUP
run_id: gate6B_smoke_musgd_lr0001_20260927-211053
purpose: verify that explicit MuSGD + lr0=0.001 + warmup_bias_lr=0 does not blow up on the
         Stage B (full-unfreeze) path when starting from Stage A best.pt (epoch 6).
NOTE: epochs=3 shortens the scheduler lifetime, so this checkpoint is NOT epoch 1-3 of the
      formal 70-epoch Stage B. Do NOT resume the formal run from this best.pt / last.pt.
keep: this directory must not be deleted or overwritten.
EOF
echo "marker written:"; ls -l "$S/DIAGNOSTIC_ONLY_STAGE_B_WARMUP.txt"