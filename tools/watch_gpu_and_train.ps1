# tools/watch_gpu_and_train.ps1
#
# Space-driven launcher for YOLO26 V1 Gate 6 (imgsz=1024) on the SHARED A6000 box.
#
# Policy (measured on this project, conservative):
#   peak_GB(batch, imgsz) ~= 0.90 + 0.45 * batch * (imgsz/640)^2
#   measured anchors: 1024/batch2 -> 2.44 GB, 640/batch8 -> 3.53 GB (Ultralytics GPU_mem)
#   start condition: free_GB >= peak_GB * SafetyFactor + SafetyAddGB  AND  util <= MaxUtil
#   candidate batches are tried from largest to smallest; if none fits the script keeps waiting.
#
# It never touches other users processes: nvidia-smi is used read-only.
param(
  [string]$Remote = "dgut@172.31.68.251",
  [string]$RemoteRepo = "/home/T7/ojh/robot_sim",
  [string]$RemoteEth = "/home/T7/dgut/robot_sim/eth_shuttle_detection",
  [int]$Imgsz = 1024,
  [int[]]$Batches = @(16, 8, 4, 2, 1),
  [int]$IntervalSec = 300,
  [double]$SafetyFactor = 1.4,
  [double]$SafetyAddGB = 1.0,
  [int]$MaxUtil = 70,
  [int]$ExpectedEthImages = 29377,
  [string]$Log = "D:\_eth_dl\gpu_watch.log",
  [switch]$DryRun
)
$ErrorActionPreference = "Continue"
function Log($m) { Add-Content $Log ((Get-Date).ToString("s") + " " + $m); Write-Output $m }

function Get-Gpu($Remote) {
  $raw = (ssh.exe -o BatchMode=yes -o ConnectTimeout=20 $Remote "nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits") 2>$null
  if (-not $raw) { return $null }
  $parts = ($raw -join "").Trim().Split(",")
  if ($parts.Count -lt 2) { return $null }
  return [pscustomobject]@{ free_gb = [double]$parts[0].Trim() / 1024.0; util = [int]$parts[1].Trim() }
}

function Get-PeakGB($batch, $imgsz) { return 0.90 + 0.45 * $batch * [math]::Pow($imgsz / 640.0, 2) }

function Get-EthImageCount($Remote, $RemoteEth) {
  $cmd = "find " + $RemoteEth + " -type f \( -name *.jpg -o -name *.png \) | wc -l"
  $raw = (ssh.exe -o BatchMode=yes -o ConnectTimeout=20 $Remote $cmd) 2>$null
  if (-not $raw) { return 0 }
  return [int](($raw -join "").Trim())
}

Log ("WATCH_START imgsz=" + $Imgsz + " candidates=" + ($Batches -join "/") + " safety=" + $SafetyFactor + "x+" + $SafetyAddGB + "GB maxUtil=" + $MaxUtil + "%")

while ($true) {
  $gpu = Get-Gpu $Remote
  if ($null -eq $gpu) { Log "gpu_query_failed (ssh?) - retry after $IntervalSec s"; Start-Sleep -Seconds $IntervalSec; continue }
  $pick = $null
  foreach ($b in $Batches) {
    $need = (Get-PeakGB $b $Imgsz) * $SafetyFactor + $SafetyAddGB
    if ($gpu.free_gb -ge $need -and $gpu.util -le $MaxUtil) { $pick = [pscustomobject]@{ batch = $b; need_gb = [math]::Round($need, 2) }; break }
  }
  if ($null -eq $pick) {
    Log ("wait: free=" + [math]::Round($gpu.free_gb,2) + "GB util=" + $gpu.util + "% ; need >= " + [math]::Round((Get-PeakGB $Batches[-1] $Imgsz) * $SafetyFactor + $SafetyAddGB,2) + "GB for batch=" + $Batches[-1])
    Start-Sleep -Seconds $IntervalSec
    continue
  }
  Log ("ready: free=" + [math]::Round($gpu.free_gb,2) + "GB util=" + $gpu.util + "% -> imgsz=" + $Imgsz + " batch=" + $pick.batch + " (needs ~" + $pick.need_gb + "GB)")

  $have = Get-EthImageCount $Remote $RemoteEth
  Log ("eth images on remote = " + $have + " / expected " + $ExpectedEthImages)
  if ($have -lt $ExpectedEthImages) {
    if ($DryRun) { Log "DRYRUN: would sync dataset"; break }
    Log "syncing dataset + repo data + tools to the remote ..."
    & (Join-Path $PSScriptRoot "sync_v1_to_remote.ps1") -Remote $Remote -RemoteEth $RemoteEth -RemoteRepo $RemoteRepo -Log $Log | Out-Null
    $have2 = Get-EthImageCount $Remote $RemoteEth
    Log ("after sync eth images = " + $have2)
    $gpu = Get-Gpu $Remote
    $need = (Get-PeakGB $pick.batch $Imgsz) * $SafetyFactor + $SafetyAddGB
    if ($gpu.free_gb -lt $need) { Log ("gpu shrank during sync (free=" + [math]::Round($gpu.free_gb,2) + "GB) - back to waiting"); Start-Sleep -Seconds $IntervalSec; continue }
  }

  $runid = "gate6A_imgsz" + $Imgsz + "_b" + $pick.batch + "_" + (Get-Date).ToString("yyyyMMdd-HHmmss")
  $cmd = "cd " + $RemoteRepo + " && mkdir -p runs/shuttle_yolo26_v1 && PYTHONPATH=$PWD/experiments/yolo26_p2_ab/_wheel_extract:$PWD/experiments/yolo26_p2_ab/_deps YOLO_CONFIG_DIR=/home/T7/ojh/robot_sim/.yolo_cfg nohup setsid env_isaaclab/bin/python tools/train_yolo26_v1.py --stage A --imgsz " + $Imgsz + " --batch " + $pick.batch + " --device 0 --run-id " + $runid + " > runs/shuttle_yolo26_v1/" + $runid + ".log 2>&1 < /dev/null & echo STARTED=$!"
  if ($DryRun) { Log ("DRYRUN would launch: " + $cmd); break }
  $out = (ssh.exe -o BatchMode=yes -o ConnectTimeout=20 $Remote $cmd) 2>&1
  Log ("LAUNCH " + ($out -join " ") + " run_id=" + $runid)
  break
}
Log "WATCH_DONE"