#!/usr/bin/env bash
DEST=/home/T7/ojh/badmin_project
echo "files=$(find "$DEST" -type f 2>/dev/null | wc -l)"
echo "extract_running=$(pgrep -cf "[t]ar -xzf" 2>/dev/null || echo 0)"
du -sh "$DEST" 2>/dev/null
ls "$DEST" 2>/dev/null | head -10
echo "--- key ---"
for f in AGENTS.md management/DAILY_LOG.md management/ISSUES.md outputs/shuttle_capability/reports/STAGE_B_FINAL_REPORT.md; do [ -f "$DEST/$f" ] && echo "OK $f $(stat -c%s "$DEST/$f")" || echo "MISS $f"; done
echo "--- server ---"; pgrep -af "[b]in.js web" | head -2; ss -ltn | grep :3080