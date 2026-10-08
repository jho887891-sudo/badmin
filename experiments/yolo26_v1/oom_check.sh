#!/usr/bin/env bash
set -u
echo "=== free ==="
free -g
echo "=== oom evidence ==="
(journalctl -k --since "06:00" 2>/dev/null | grep -i -E "oom|out of memory|killed process" | tail -8 || echo "(journalctl not readable)")
echo "=== syslog oom ==="
(grep -i -E "oom|killed process" /var/log/syslog 2>/dev/null | tail -8 || echo "(syslog not readable)")
echo "=== auditd/other ==="
(grep -i "python" /var/log/syslog 2>/dev/null | tail -3 || true)
echo "=== dmesg via sudo-less ==="
(cat /var/log/kern.log 2>/dev/null | grep -i oom | tail -5 || echo "(kern.log not readable)")