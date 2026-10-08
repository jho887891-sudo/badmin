# Three-way Stage B final evaluation: Stage A best vs Stage B best(e10) vs Stage B last(e70)
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
Set-Location "E:\具身智能\badmin_project"
$py = "D:\_yolo26v1\venv\Scripts\python.exe"
$ck = @(
  "stageA_best=D:\_eth_dl\ckpts\stageA_best.pt",
  "stageB_best_e10=D:\_eth_dl\ckpts\stageB_best_e10_stripped.pt",
  "stageB_last_e70=D:\_eth_dl\ckpts\stageB_last_e70.pt"
)
$ckArgs = @(); foreach ($c in $ck) { $ckArgs += @("--ckpt", $c) }
Write-Output "=== [1/4] val size-bucket eval (3 checkpoints) $(Get-Date -Format o) ==="
& $py tools\eval_yolo26_v1.py --repo . --split val --imgsz 1024 --conf 0.25 --ap-conf 0.001 --iou 0.5 --center-dist 25 --device cuda:0 @ckArgs
Write-Output "=== [2/4] hard-negative FP eval (3 checkpoints) $(Get-Date -Format o) ==="
& $py tools\eval_hard_negative.py --repo . @ckArgs `
  --images manifest:hard_negative:train --set-name hard_negative `
  --images manifest:coco_bg:val --set-name normal_neg_coco_val `
  --images manifest:bg_negative:val --set-name normal_neg_bg_val `
  --images dir:outputs/shuttle_capability/real_images/backgrounds --set-name frozen_real_backgrounds `
  --thresholds 0.25,0.50,0.75 --conf-floor 0.01 --imgsz 1024 --device cuda:0 --topk 20
Write-Output "=== [3/4] compare + histogram $(Get-Date -Format o) ==="
& $py tools\compare_hard_negative_eval.py --repo .
& $py tools\hard_negative_confidence_hist.py --repo .
Write-Output "=== [4/4] re-render top FP images $(Get-Date -Format o) ==="
Get-ChildItem "outputs\shuttle_capability\hard_negative_eval" -Recurse -File -Filter *.jpg -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
& $py tools\render_top_fp.py --repo . --min-confidence 0.25 --topk 20
Write-Output "=== DONE $(Get-Date -Format o) ==="