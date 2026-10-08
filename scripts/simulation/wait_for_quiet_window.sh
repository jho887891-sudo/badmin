#!/bin/bash
# Wait for a window, judged primarily on the GPU - the scarce shared resource - with CPU load as a sanity
# check rather than the gate.
#
# Revised reasoning: the first version required load < 2.0 on a 32-core machine and spent ten minutes not
# firing while the GPU sat at 0%. A load of 6 on 32 cores is under 20% CPU utilisation and is not a conflict
# with anybody; hogging the GPU would be. So the GPU utilisation is the gate, the GPU memory is a second
# gate, and the load threshold is widened to catch only genuine saturation.
set -u
cd /home/T7/dgut/robot_sim

LOG=/tmp/window_watcher.log
RECORD=outputs/simulation/rebot_b601dm/swing_record.json

MAX_LOAD=${MAX_LOAD:-12.0}       # 32 cores; below this is under 40% utilisation
MAX_GPU=${MAX_GPU:-15}           # the real gate: do not take the GPU while someone is training
MAX_GPU_MEM=${MAX_GPU_MEM:-8000} # MiB; a training job holds far more than this
NEED=${NEED:-3}
POLL=${POLL:-90}
DEADLINE=$(( ${DEADLINE_H:-8} * 3600 ))

echo "[watcher2] start $(date +%H:%M:%S) cores=$(nproc) thresholds load<$MAX_LOAD gpu<${MAX_GPU}%% gpumem<${MAX_GPU_MEM}MiB x$NEED" > $LOG

START=$(date +%s); QUIET=0
while [ $(( $(date +%s) - START )) -lt $DEADLINE ]; do
  LOAD=$(cut -d" " -f1 /proc/loadavg)
  read -r GPU MEM < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr "," " ")
  GPU=${GPU:-999}; MEM=${MEM:-999999}
  LOK=$(awk -v l="$LOAD" -v m="$MAX_LOAD" "BEGIN{print (l<m)?1:0}")
  GOK=$([ "$GPU" -lt "$MAX_GPU" ] && echo 1 || echo 0)
  MOK=$([ "$MEM" -lt "$MAX_GPU_MEM" ] && echo 1 || echo 0)
  if [ "$LOK" = 1 ] && [ "$GOK" = 1 ] && [ "$MOK" = 1 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  echo "[watcher2] $(date +%H:%M:%S) load=$LOAD gpu=${GPU}%% mem=${MEM}MiB quiet=$QUIET/$NEED" >> $LOG

  if [ "$QUIET" -ge "$NEED" ]; then
    echo "=== WINDOW at $(date +%H:%M:%S) ==="
    uptime; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; echo
    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* 2>/dev/null | tr "\n" ":")
    /home/T7/ojh/robot_sim/env_isaaclab/bin/python -u scripts/simulation/measure_racket_speed.py \
      --asset outputs/simulation/rebot_b601dm/real_limits/reBot_B601_DM.usda \
      --variant real --torque-convention rated --target-joint joint6 --seconds 6 \
      --out "$RECORD" >> $LOG 2>&1
    RC=$?
    echo "[watcher2] measurement exited rc=$RC" >> $LOG
    if [ -f "$RECORD" ]; then echo "=== RECORD WRITTEN ==="; head -45 "$RECORD"; else echo "=== no record; tail ==="; tail -25 $LOG; fi
    exit $RC
  fi
  sleep $POLL
done
echo "[watcher2] deadline without a window"
