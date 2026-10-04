# tools/sync_v1_to_remote.ps1
# Copies everything Gate 6 needs to the shared GPU box: ETH dataset (per group, resumable),
# the repo-local training outputs the manifest points at, plus tools/configs/lists.
param(
  [string]$Remote = "dgut@172.31.68.251",
  [string]$RemoteEth = "/home/T7/dgut/robot_sim/eth_shuttle_detection",
  [string]$RemoteRepo = "/home/T7/ojh/robot_sim",
  [string]$EthLocal = "D:\_eth_data\eth_shuttle_detection",
  [string]$Log = "D:\_eth_dl\sync_v1.log",
  [switch]$SkipEth
)
$ErrorActionPreference = "Continue"
function Log($m) { Add-Content $Log ((Get-Date).ToString("s") + " " + $m) }

Log "SYNC_START"
ssh.exe -o BatchMode=yes $Remote ("mkdir -p " + $RemoteEth) | Out-Null

if (-not $SkipEth) {
  $dirs = Get-ChildItem -Directory $EthLocal | Select-Object -ExpandProperty Name
  foreach ($d in $dirs) {
    $t0 = Get-Date
    scp.exe -q -r -o BatchMode=yes ("$EthLocal\$d") ($Remote + ":" + $RemoteEth + "/") 2>>$Log
    Log ("eth group=" + $d + " rc=" + $LASTEXITCODE + " secs=" + [math]::Round(((Get-Date) - $t0).TotalSeconds,1))
  }
  Get-ChildItem -File $EthLocal -Filter *.yaml | ForEach-Object { scp.exe -q -o BatchMode=yes $_.FullName ($Remote + ":" + $RemoteEth + "/") 2>>$Log }
  scp.exe -q -o BatchMode=yes ("$EthLocal\sorting_progress.csv") ($Remote + ":" + $RemoteEth + "/") 2>>$Log
}

# repo-local data the manifest points at (local versions are the SSOT)
foreach ($sub in @("train_data", "hard_negatives", "hard_negatives2", "v1_dataset")) {
  $p = "outputs\shuttle_capability\$sub"
  if (Test-Path $p) {
    ssh.exe -o BatchMode=yes $Remote ("mkdir -p " + $RemoteRepo + "/outputs/shuttle_capability") | Out-Null
    scp.exe -q -r -o BatchMode=yes $p ($Remote + ":" + $RemoteRepo + "/outputs/shuttle_capability/") 2>>$Log
    Log ("repo data=" + $sub + " rc=" + $LASTEXITCODE)
  }
}

# tools + configs
ssh.exe -o BatchMode=yes $Remote ("mkdir -p " + $RemoteRepo + "/tools " + $RemoteRepo + "/configs/shuttle_detection " + $RemoteRepo + "/experiments/yolo26_v1") | Out-Null
scp.exe -q -o BatchMode=yes tools\train_yolo26_v1.py tools\build_yolo26_v1_dataset.py ($Remote + ":" + $RemoteRepo + "/tools/") 2>>$Log
scp.exe -q -o BatchMode=yes configs\shuttle_detection\yolo26_p2_v1.yaml configs\shuttle_detection\eth_location_split_v1.yaml ($Remote + ":" + $RemoteRepo + "/configs/shuttle_detection/") 2>>$Log
Log ("tools+configs rc=" + $LASTEXITCODE)

# verification: image counts on both sides
$localEth = (Get-ChildItem -Recurse -File -Include *.jpg,*.jpeg,*.png $EthLocal -EA SilentlyContinue | Measure-Object).Count
$remoteEth = (ssh.exe -o BatchMode=yes $Remote ("find " + $RemoteEth + " -type f -name *.jpg -o -name *.png | wc -l")) -join ""
Log ("VERIFY eth_local_images=" + $localEth + " eth_remote_images=" + $remoteEth.Trim())
Log "SYNC_DONE"