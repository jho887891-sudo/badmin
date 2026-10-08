#!/usr/bin/env bash
echo "== 远端 -> 你的公网 VPS 223.109.239.11 =="
ping -c 5 -W 2 223.109.239.11 2>&1 | tail -3
echo "-- TCP 20736 是否可达（用于估算自建 DERP 收益） --"
timeout 6 bash -c "cat < /dev/null > /dev/tcp/223.109.239.11/20736" 2>/dev/null && echo "tcp_20736_open" || echo "tcp_20736_blocked_or_filtered"