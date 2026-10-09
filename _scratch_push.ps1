$ErrorActionPreference='Continue'
Set-Location 'E:\具身智能\badmin_project'
Remove-Item -Force '_scratch_git_status.ps1' -ErrorAction SilentlyContinue
Write-Output '=== sanity: python syntax ==='
foreach ($py in @('tools/remote/speed_chain_analyze.py','tools/remote/speed_chain_helpers.py','tools/build_cont_v1_measured.py','tools/remote/strip_cont_v1_ckpt.py')) {
  $r = & python -m py_compile $py 2>&1
  Write-Output ('py_compile {0} exit={1} {2}' -f $py, $LASTEXITCODE, ($r -join ' '))
}
Write-Output '=== sanity: bash -n ==='
foreach ($sh in @('tools/remote/run_window_queue.sh','tools/remote/postrun_cont_v1.sh')) {
  $o = & 'D:\Git\bin\bash.exe' -n $sh 2>&1
  Write-Output ('bash_n {0} exit={1} {2}' -f $sh, $LASTEXITCODE, ($o -join ' '))
}
Write-Output '=== sanity: json parse ==='
foreach ($line in (git status --porcelain)) {
  if ($line -match '\.json$') {
    $j = $line.Substring(3).Trim('"')
    try { Get-Content -LiteralPath $j -Raw | ConvertFrom-Json | Out-Null; Write-Output ('json_ok  ' + $j) } catch { Write-Output ('json_BAD ' + $j + ' :: ' + $_.Exception.Message) }
  }
}
Write-Output '=== commit 1: tooling record ==='
git add management/tooling
git commit -m 'docs(tooling): record superpowers-dsh plugin status and the 0.2.0 Windows blocker' 2>&1 | Select-Object -Last 3
Write-Output '=== commit 2: remaining shuttle-detection artifacts ==='
git add -A
git commit -m 'shuttle-detection: continuation-v1 metrics, checkpoint config and remote queue tooling' 2>&1 | Select-Object -Last 3
Write-Output '=== push ==='
git push origin feature/shuttle-detection 2>&1 | Select-Object -Last 8
Write-Output '=== final ==='
git log --oneline -4
git status --short
git log --oneline '@{u}..HEAD'