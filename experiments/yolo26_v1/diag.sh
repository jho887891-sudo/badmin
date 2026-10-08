#!/usr/bin/env bash
PID=$(pgrep -f "[g]ates_1_3_audit.py" | head -1)
echo "PID=$PID"
if [ -z "$PID" ]; then echo NO_PROCESS; exit 0; fi
echo -n "wchan: "; cat /proc/$PID/wchan; echo
echo "--- io ---"; cat /proc/$PID/io
echo "--- fd tail ---"; ls -l /proc/$PID/fd 2>/dev/null | tail -6
echo "--- status ---"; grep -E "^(State|Threads)" /proc/$PID/status
echo "--- stack ---"; cat /proc/$PID/task/$PID/stack 2>/dev/null | head -10
echo "--- children ---"; ps --ppid $PID -o pid,etime,time,args 2>/dev/null | head -5