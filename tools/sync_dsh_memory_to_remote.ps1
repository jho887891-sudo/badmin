# Sync the local project "memory" (text artifacts + evidence images) to the remote DSH workspace.
# Mirrors the working tree at the remote tree used by the remote Harness (no deletion of remote-only files).
param(
  [string]$Remote = "dgut@172.31.68.251",
  [string]$Dest   = "/home/T7/ojh/badmin_project",
  [string]$Repo   = "E:\具身智能\badmin_project",
  [string]$Tgz    = "D:\_eth_dl\dsh_memory.tgz"
)
$ErrorActionPreference = "Stop"
$textExts = @(".md",".csv",".json",".txt",".yaml",".yml",".py",".sh",".ps1",".toml",".cfg",".ini")
$files = Get-ChildItem $Repo -Recurse -File -Force -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch "\\_scratch_|\\\.git\\|\\env_isaaclab\\|\\IsaacLab\\|\\node_modules\\|\\assets\\|__pycache__" } |
  Where-Object { ($textExts -contains $_.Extension) -or (($_.Extension -in @(".jpg",".png")) -and ($_.FullName -match "\\outputs\\shuttle_capability\\hard_negative_eval\\")) } |
  Where-Object { $_.Name -ne "relay.log" }
$rel = $files | ForEach-Object { $_.FullName.Substring($Repo.Length + 1) }
$list = Join-Path $env:TEMP "dsh_memory_files.txt"
[IO.File]::WriteAllText($list, ($rel -join "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))
Write-Output ("local files=" + $rel.Count + " bytes=" + (($files | Measure-Object -Sum Length).Sum))
tar -czf $Tgz -C $Repo -T $list
Write-Output ("tgz=" + [math]::Round((Get-Item $Tgz).Length/1MB, 2) + " MB")
scp -o BatchMode=yes $Tgz ($Remote + ":/tmp/dsh_memory.tgz")
$remoteCmd = "mkdir -p " + $Dest + " && tar -xzf /tmp/dsh_memory.tgz -C " + $Dest + " && echo files=$(find " + $Dest + " -type f | wc -l) && du -sh " + $Dest
ssh -o BatchMode=yes $Remote $remoteCmd