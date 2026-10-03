#!/usr/bin/env bash
echo "=== my footprint on the root partition ==="
df -h / | tail -1
du -sh "$HOME/.dsh-bench" 2>/dev/null
du -sh "$HOME/.dsh-bench/runs" 2>/dev/null
du -sh "$HOME/.dsh-bench/runs/detect/runs_eth_only_v1" 2>/dev/null
du -sh "$HOME/.dsh-bench/runs/detect/runs_eth_only_v1"/full_e50/weights 2>/dev/null
echo "=== what else is big under /home (top 6, may take a moment) ==="
du -sh "$HOME"/* 2>/dev/null | sort -rh | head -6
