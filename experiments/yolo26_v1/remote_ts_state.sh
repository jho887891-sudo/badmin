#!/usr/bin/env bash
echo "== service =="; systemctl is-active tailscaled; systemctl is-enabled tailscaled; systemctl status tailscaled --no-pager | head -6
echo "== version =="; tailscale version 2>&1 | head -1 || sudo -n tailscale version 2>&1 | head -1
echo "== up.log (login URL) =="; cat /tmp/ts_up.log 2>/dev/null || echo "no /tmp/ts_up.log"
echo "== status =="; tailscale status 2>&1 | head -8
echo "== ip =="; tailscale ip -4 2>&1 | head -2
echo "== operator setting =="; grep -i operator /var/lib/tailscale/tailscaled.state 2>/dev/null | head -1 || echo "(state not readable / n/a)"
echo "== backend state =="; tailscale debug prefs 2>/dev/null | head -12 || tailscale status --json 2>/dev/null | head -5