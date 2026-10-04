# Stage B read-only event monitor (v2, durable). Logs ONLY on the 11 event types.
# Never writes to the remote host; only runs the read-only probe script over ssh.
$ErrorActionPreference = "Continue"
$Log    = "D:\_eth_dl\stageb_monitor.log"
$Stop   = "D:\_eth_dl\stageb_monitor.stop"
$PidFile = "D:\_eth_dl\stageb_monitor2.pid"
$Probe  = "bash /home/T7/ojh/robot_sim/experiments/yolo26_v1/stageb_probe.sh"
$Ssh    = "ssh -o BatchMode=yes -o ConnectTimeout=20 dgut@172.31.68.251"
$Baseline = 0.76012   # Stage A epoch-6 recall
$HeadLrMax = 0.0033   # MuSGD head group limit (0.001 * 3 * 1.10)
$MemMin = 5.0
$SwapIoMax = 20480.0  # KB/s
$CgroupMax = 0.90
$Checkpoints = @(40, 50, 60, 70)

function Log($msg) { Add-Content -Path $Log -Value ((Get-Date -Format "yyyy-MM-ddTHH:mm:ss") + " " + $msg) -Encoding ASCII }

$state = @{ bestEpoch = -1; fitnessHigh = $false; prevEpoch = -1; prevMap50 = $null; prevBox = $null;
            nan = $false; lrAnom = $false; memLow = $false; swapHigh = $false; cgroupHigh = $false;
            exitLogged = $false; ckpt = @{} }
Log "MONITOR2_START (read-only; training untouched; pid=$PID)"

while ($true) {
  if (Test-Path $Stop) { Log "MONITOR2_STOP_FILE"; break }
  $raw = & cmd /c "$Ssh `"$Probe`" 2>&1" | Select-Object -Last 1
  if (-not $raw -or $raw -notmatch "^{") { Start-Sleep -Seconds 60; continue }
  $d = $null
  try { $d = $raw | ConvertFrom-Json } catch { Log ("PROBE_PARSE_ERROR " + $raw.Substring(0, [Math]::Min(200, $raw.Length))); Start-Sleep -Seconds 60; continue }
  if ($d.alive -eq $false -and $state.exitLogged -eq $false) { Log "PROCESS_EXIT epochs=$($d.epochs)"; $state.exitLogged = $true }
  if ($d.best -and [int]$d.best.epoch -ne $state.bestEpoch) {
    $state.bestEpoch = [int]$d.best.epoch
    Log ("BEST_REFRESHED epoch=" + $d.best.epoch + " fitness=" + $d.best.fitness + " mAP50=" + $d.best.map50 + " mAP50-95=" + $d.best.map5095 + " recall=" + $d.best.recall)
  }
  if ($d.best -and [double]$d.best.fitness -ge 0.60 -and -not $state.fitnessHigh) { $state.fitnessHigh = $true; Log ("FITNESS_HIGH epoch=" + $d.best.epoch + " fitness=" + $d.best.fitness) }
  if ($d.last) {
    $e = [int]$d.last.epoch; $m = [double]$d.last.map50; $b = [double]$d.last.box
    if ($e -ne $state.prevEpoch) {
      if ($state.prevMap50 -ne $null -and [Math]::Abs($m - $state.prevMap50) -ge 0.05) { Log ("METRIC_ANOMALY epoch=$e mAP50=" + $m + " prev=" + $state.prevMap50) }
      if ($state.prevBox -ne $null -and ($b - $state.prevBox) -ge 0.05) { Log ("METRIC_ANOMALY epoch=$e box_loss=" + $b + " prev=" + $state.prevBox) }
      $state.prevEpoch = $e; $state.prevMap50 = $m; $state.prevBox = $b
      foreach ($c in $Checkpoints) { if ($e -ge $c -and -not $state.ckpt.ContainsKey($c)) { $state.ckpt[$c] = $true; Log ("CHECKPOINT epoch=$c reached | current_epoch=$e best_epoch=" + $state.bestEpoch + " best_fitness=" + $d.best.fitness + " last_mAP50=" + $m + " last_mAP50-95=" + $d.last.map5095 + " last_recall=" + $d.last.recall + " box=" + $b + " MemAvailable=" + $d.mem.MemAvailable_GB + " RSS=" + $d.mem.trainer_rss_GB) } }
    }
    if ([double]$d.last.recall -lt (0.5 * $Baseline)) { Log ("RECALL_COLLAPSE epoch=$e recall=" + $d.last.recall + " baseline=$Baseline") }
  }
  $lrAll = @($d.lr_groups | Where-Object { $_ -ne $null })
  if ($lrAll.Count -gt 0) { $mx = ($lrAll | Measure-Object -Maximum).Maximum; if ($mx -gt $HeadLrMax -and -not $state.lrAnom) { $state.lrAnom = $true; Log ("LR_ANOMALY max_group_lr=$mx limit=$HeadLrMax") } }
  if ($d.mem) {
    if ($d.mem.MemAvailable_GB -ne $null -and [double]$d.mem.MemAvailable_GB -lt $MemMin -and -not $state.memLow) { $state.memLow = $true; Log ("MEM_LOW MemAvailable_GB=" + $d.mem.MemAvailable_GB) }
    $io = 0.0; if ($d.mem.swap_in_KBps -ne $null) { $io += [double]$d.mem.swap_in_KBps }; if ($d.mem.swap_out_KBps -ne $null) { $io += [double]$d.mem.swap_out_KBps }
    if ($io -gt $SwapIoMax -and -not $state.swapHigh) { $state.swapHigh = $true; Log ("SWAP_IO_HIGH KBps=" + $io) }
    if ($d.mem.cgroup_frac -ne $null -and [double]$d.mem.cgroup_frac -gt $CgroupMax -and -not $state.cgroupHigh) { $state.cgroupHigh = $true; Log ("CGROUP_HIGH frac=" + $d.mem.cgroup_frac) }
  }
  if ($state.exitLogged -and $state.ckpt.ContainsKey(70)) { Log "MONITOR2_END (epoch 70 reached and process gone)"; break }
  Start-Sleep -Seconds 60
}