#!/usr/bin/env bash
cd /home/T7/dgut/robot_sim/outputs/shuttle_capability
for pool in hard_negatives hard_negatives2; do
  for f in "$pool"/raw/*.jpg; do
    [ -f "$f" ] || continue
    printf "%s  %s\n" "$(sha256sum "$f" | cut -d" " -f1)" "$f"
  done
done
