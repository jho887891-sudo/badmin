#!/usr/bin/env bash
set -u
DEST=/home/T7/ojh/badmin_project
mkdir -p "$DEST"
tar -xzf /tmp/dsh_memory.tgz -C "$DEST"
echo "== extracted =="
echo "files=$(find "$DEST" -type f | wc -l)"
du -sh "$DEST"
echo "== key memory files =="
for f in AGENTS.md management/DAILY_LOG.md management/ISSUES.md outputs/shuttle_capability/reports/STAGE_B_FINAL_REPORT.md docs/REMOTE_DSH_DEPLOY.md; do
  if [ -f "$DEST/$f" ]; then echo "OK   $f  $(stat -c%s "$DEST/$f") B  $(stat -c%y "$DEST/$f" | cut -d. -f1)"; else echo "MISS $f"; fi
done
echo "== top-level =="; ls "$DEST" | head -20
echo "== server still up? =="; pgrep -af "[b]in.js web" | head -2; ss -ltn | grep :3080