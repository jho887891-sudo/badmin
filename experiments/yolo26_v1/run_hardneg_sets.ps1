# Hard-negative FP eval for all 4 negative sets x 3 checkpoints (one invocation per set: the tool takes a single --images spec).
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
Set-Location "E:\具身智能\badmin_project"
$py = "D:\_eval26\venv\Scripts\python.exe"
$ck = @("stageA_best=D:\_eth_dl\ckpts\stageA_best.pt", "stageB_best_e10=D:\_eth_dl\ckpts\stageB_best_e10_stripped.pt", "stageB_last_e70=D:\_eth_dl\ckpts\stageB_last_e70.pt")
$ckArgs = @(); foreach ($c in $ck) { $ckArgs += @("--ckpt", $c) }
$defaults = @("--repo", ".", "--manifest", "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv",
              "--thresholds", "0.25,0.50,0.75", "--conf-floor", "0.01", "--imgsz", "1024", "--device", "cuda:0", "--topk", "20")
$sets = @(
  @{ n = "hard_negative";           spec = "manifest:hard_negative:train" },
  @{ n = "normal_neg_coco_val";     spec = "manifest:coco_bg:val" },
  @{ n = "normal_neg_bg_val";       spec = "manifest:bg_negative:val" },
  @{ n = "frozen_real_backgrounds"; spec = "dir:outputs/shuttle_capability/real_images/backgrounds" }
)
foreach ($s in $sets) {
  Write-Output "=== set $($s.n) $(Get-Date -Format o) ==="
  & $py tools\eval_hard_negative.py @defaults --images $s.spec --set-name $s.n @ckArgs
}
Write-Output "=== compare + hist + render $(Get-Date -Format o) ==="
& $py tools\compare_hard_negative_eval.py --repo .
& $py tools\hard_negative_confidence_hist.py --repo .
Get-ChildItem "outputs\shuttle_capability\hard_negative_eval" -Recurse -File -Filter *.jpg -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
& $py tools\render_top_fp.py --repo . --min-confidence 0.25 --topk 20
Write-Output "=== DONE $(Get-Date -Format o) ==="