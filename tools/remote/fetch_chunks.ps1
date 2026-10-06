$ErrorActionPreference = "Continue"
cd E:\具身智能\badmin_project
$remote = "dgut@jxxy.taildd42cc.ts.net"
$dir = "/home/dgut/.dsh-bench"
$src = "$dir/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt"
New-Item -ItemType Directory -Force -Path _scratch_resdiag\chunks | Out-Null
# 1 MiB chunks + per-chunk sha256 on the remote (idempotent)
ssh $remote "cd $dir && rm -f v2c.* && split -b 1048576 -d -a 2 $src v2c. && sha256sum v2c.* > v2c.sha256 && ls v2c.* | wc -l"
$manifest = (ssh $remote "cat $dir/v2c.sha256") -split "?
" | Where-Object { $_ }
"chunks_total=" + $manifest.Count
$ok = 0; $fail = 0; $t0 = Get-Date
foreach ($line in $manifest) {
  $parts = $line -split "\s+"
  $sha = $parts[0]; $name = $parts[1]
  $local = "_scratch_resdiag\chunks\$name"
  $attempt = 0
  while ($attempt -lt 3) {
    $attempt++
    if (Test-Path $local) { if ((Get-FileHash $local -Algorithm SHA256).Hash.ToLower() -eq $sha) { break } }
    scp -o BatchMode=yes -o ConnectTimeout=20 "${remote}:${dir}/$name" $local 2>$null | Out-Null
    if ((Test-Path $local) -and (Get-FileHash $local -Algorithm SHA256).Hash.ToLower() -eq $sha) { break }
    Start-Sleep -Seconds 3
  }
  if ((Test-Path $local) -and (Get-FileHash $local -Algorithm SHA256).Hash.ToLower() -eq $sha) { $ok++ } else { $fail++; "MISSING $name" }
  if (((Get-Date) - $t0).TotalSeconds -gt 540) { "time_budget_reached after $ok ok / $fail bad"; break }
}
"chunks_ok=$ok chunks_failed=$fail elapsed_s=" + [math]::Round(((Get-Date) - $t0).TotalSeconds,1)
Get-ChildItem _scratch_resdiag\chunks -File | Measure-Object Length -Sum | ForEach-Object { "bytes=" + $_.Sum }