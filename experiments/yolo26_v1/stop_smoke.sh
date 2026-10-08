#!/usr/bin/env bash
P=$(pgrep -f "[t]rain_yolo26_v1" | tr "\n" " ")
echo "stopping smoke: $P"
[ -n "$P" ] && kill -INT $P && sleep 15
P2=$(pgrep -f "[t]rain_yolo26_v1" | tr "\n" " ")
[ -n "$P2" ] && kill -TERM $P2 && sleep 8
P3=$(pgrep -f "[t]rain_yolo26_v1" | tr "\n" " ")
[ -n "$P3" ] && kill -KILL $P3
echo "remaining=$(pgrep -f "[t]rain_yolo26_v1" | wc -l)"
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader