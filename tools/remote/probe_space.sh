#!/usr/bin/env bash
echo "=== remote date ==="; date -Is; uptime | tr -s " "
echo "=== root fs ==="; df -h / /home /tmp 2>/dev/null | sed -n "1,6p"
echo "=== reclaim candidates ==="
for d in "$HOME/.trae-server" "$HOME/.trae-cn-server"; do [ -d "$d" ] && du -sh "$d" 2>/dev/null; done
ls -d "$HOME"/.trae-server/* "$HOME"/.trae-cn-server/* 2>/dev/null | wc -l
ls -1dt "$HOME"/.trae-server/* "$HOME"/.trae-cn-server/* 2>/dev/null | head -4
ls -la /tmp/dsh_sessions.tgz 2>/dev/null; du -sh /tmp/dshmem-* 2>/dev/null | head -3
echo "=== training ==="
tail -c 200 "$HOME/.dsh-bench/full_e50_run.log" | tr "\r" "\n" | tail -1
pgrep -c -f train_eth_only_v1.py
echo "=== gpu ==="; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
