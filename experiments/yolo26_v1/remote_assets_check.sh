#!/usr/bin/env bash
sha256sum /home/T7/ojh/robot_sim/outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv
wc -l /home/T7/ojh/robot_sim/outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv
ls -1 /home/T7/ojh/robot_sim/outputs/shuttle_capability/hard_negatives/raw 2>/dev/null | wc -l
ls -1 /home/T7/ojh/robot_sim/outputs/shuttle_capability/hard_negatives2/raw 2>/dev/null | wc -l
ls -1 /home/T7/ojh/robot_sim/outputs/shuttle_capability/real_images/backgrounds 2>/dev/null | wc -l