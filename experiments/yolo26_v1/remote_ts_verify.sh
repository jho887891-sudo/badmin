#!/usr/bin/env bash
echo "=== REMOTE: identity ==="; hostname; echo "tailscale_ip=$(tailscale ip -4)"; tailscale version | head -1
echo "=== REMOTE: status ==="; tailscale status
echo "=== REMOTE: service ==="; systemctl is-active tailscaled; systemctl is-enabled tailscaled; systemctl show tailscaled -p MainPID --value
echo "=== REMOTE -> LOCAL: tailscale ping ==="; tailscale ping -c 3 abc123456 2>&1 | head -5
echo "=== REMOTE: netcheck (first lines) ==="; tailscale netcheck 2>&1 | head -10
echo "=== REMOTE: no exit node / no routes advertised? ==="; tailscale status --json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('BackendState=',d.get('BackendState')); print('ExitNodeStatus=',d.get('ExitNodeStatus')); s=d.get('Self',{}); print('Self.HostName=',s.get('HostName'),'IPs=',s.get('TailscaleIPs'),'Online=',s.get('Online'),'OS=',s.get('OS')); print('AllowedIPs(advertised)=',s.get('AllowedIPs'))" 2>/dev/null || echo "(json parse skipped)"