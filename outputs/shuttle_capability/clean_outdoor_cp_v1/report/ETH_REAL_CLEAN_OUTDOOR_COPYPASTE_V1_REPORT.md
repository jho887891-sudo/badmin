# ETH real foregrounds + COCO clean outdoor Copy-Paste + COCO hard negatives V1

Experiment: eth_real_clean_outdoor_copypaste_v1 (single run CLEAN_OUTDOOR_CP_V1).
Decision state: **PILOT_ONLY**.

## Frozen reference

- V2 recipe: configs/shuttle_detection/eth_real_hardneg_v2_resolved_args.yaml (sha256 41e96b029f250c12256743522af65e30dd4ac2cf5c62187f7513b4017c751c3b), remote /home/dgut/.dsh-bench/runs/detect/runs_eth_hardneg_v2/full_v2_e50/args.yaml
- pretrained: assets/external/_staging/F_yolo/weights/yolo26s.pt (sha256 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b)
- V2 composition: 13992 positives + 551 ordinary negatives + 712 hard-negative exposures (89 x 8) = 15255 slots
- internal validation: 2920 frames, locations glc_2, uetlibergstrasse_2

## Gates

- coco_provenance_audit.json: PASS
- foreground_audit.json: FOREGROUND_INSUFFICIENT
- repo_contract_map.json: PASS

## Measurements

- real no-target FP: not measured
- val|eth_unseen (all): not measured
- val|eth_unseen (real only): not measured
- val|eth_unseen (synthetic only): not measured
- real outdoor OOD: not available, so no outdoor-generalisation claim is made

## Limitations

- Copy-Paste positives and the new outdoor hard negatives changed together and there is no
  same-period A0 control, so every difference from V2 is descriptive only.
- COCO has no shuttlecock category; an absent annotation is never evidence that a picture has none.
- Every accepted background, foreground and hard negative carries a human review record; a missing
  review keeps the corresponding gate closed.
