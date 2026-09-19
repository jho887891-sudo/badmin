#!/bin/bash
# Print the REAL API of SingleArticulation, so the velocity-target fix is evidence-based rather than recalled.
# isaacsim.core cannot be imported before SimulationApp exists, so this starts one and asks it there.
set -u
export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* 2>/dev/null | tr "\n" ":")
cd /home/T7/dgut/robot_sim

echo "=== 1. from the source file, which needs no app ==="
SP=/home/T7/ojh/robot_sim/env_isaaclab/lib/python3.12/site-packages
F=$(find "$SP/isaacsim/core/prims" -name "single_articulation.py" 2>/dev/null | head -1)
echo "  source: $F"
if [ -n "$F" ]; then
  grep -nE "^    def [a-z]" "$F" | grep -iE "velocit|effort|target|action|position" | sed "s/^/    /"
fi

echo
echo "=== 2. the multi-articulation class, which is where velocity targets usually live ==="
G=$(find "$SP/isaacsim/core/prims" -name "articulation.py" 2>/dev/null | head -1)
echo "  source: $G"
if [ -n "$G" ]; then
  grep -nE "^    def [a-z]" "$G" | grep -iE "velocit|effort|target|action" | sed "s/^/    /" | head -20
fi

echo
echo "=== 3. the action API, the documented route for physics-driven control ==="
grep -rn "class ArticulationAction" "$SP/isaacsim/core/" 2>/dev/null | head -3 | sed "s/^/  /"
grep -rn "def apply_action" "$SP/isaacsim/core/prims/" 2>/dev/null | head -5 | sed "s/^/  /"

echo
echo "=== 4. what the upstream receiver uses, which is known to work ==="
grep -nE "self\.articulation\.[a-z_]+\(" reBot-Isaacsim/reBotArm_Isaacsim/isaacsim_joint_receiver.py 2>/dev/null | head -20 | sed "s/^/  /"
