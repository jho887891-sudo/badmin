#!/usr/bin/env bash
echo "== who / w =="; who; echo "---"; w -h 2>/dev/null | head -5
echo "== sshd sessions (established :22) =="; ss -tn state established "( sport = :22 )" 2>/dev/null | head -8
echo "== vscode-server per user =="
for u in dgut T7 root; do h=$(getent passwd $u | cut -d: -f6); [ -n "$h" ] && printf "  %-6s home=%-22s vscode-server=%s\n" "$u" "$h" "$([ -d "$h/.vscode-server" ] && echo yes || echo no)"; done
echo "== vscode processes =="; ps -eo user,pid,etime,cmd 2>/dev/null | grep -i "[v]scode-server\|[c]ode-server" | head -6
echo "== installer file =="; ls -l /tmp/ts_install.sh; sha256sum /tmp/ts_install.sh | cut -c1-16
echo "== sudo policy for dgut =="; sudo -n -l 2>&1 | head -3
echo "== any pending sudo/apt =="; pgrep -af "[s]udo" | head -3 || echo none; pgrep -af "[a]pt-get" | head -3 || echo no-apt