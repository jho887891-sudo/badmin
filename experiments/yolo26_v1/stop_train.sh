#!/usr/bin/env bash
set -u
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
PIDS=$(pgrep -f "[t]rain_yolo26_v1.py" | tr "\n" " ")
echo "TRAIN_PIDS=$PIDS"
if [ -z "$PIDS" ]; then echo "NO_TRAINING_PROCESS_RUNNING"; else
  kill -INT $PIDS 2>/dev/null || true
  sleep 20
  LEFT=$(pgrep -f "[t]rain_yolo26_v1.py" | tr "\n" " ")
  if [ -n "$LEFT" ]; then echo "after SIGINT still: $LEFT -> SIGTERM"; kill -TERM $LEFT 2>/dev/null || true; sleep 15; fi
  LEFT2=$(pgrep -f "[t]rain_yolo26_v1.py" | tr "\n" " ")
  if [ -n "$LEFT2" ]; then echo "after SIGTERM still: $LEFT2 -> SIGKILL"; kill -KILL $LEFT2 2>/dev/null || true; sleep 5; fi
fi
echo "REMAINING=$(pgrep -f "[t]rain_yolo26_v1.py" | wc -l)"
echo "=== gpu after stop ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader
echo "=== run artifacts ==="
echo "RUN=$RUN"
ls -l "$RUN/weights" 2>/dev/null || echo "(no weights dir)"
tail -c 300 "$RUN.log" 2>/dev/null | tr -d "\r" | tail -2