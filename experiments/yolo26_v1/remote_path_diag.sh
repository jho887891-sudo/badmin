#!/usr/bin/env bash
echo "=== REMOTE -> peer abc123456 ==="
tailscale status --json 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
for k,v in d['Peer'].items():
    if v.get('HostName')=='abc123456':
        print('CurAddr=', v.get('CurAddr'))
        print('Relay=', v.get('Relay'))
        print('Online=', v.get('Online'))
        print('Endpoints=', v.get('Endpoints'))
s=d['Self']
print('Self.CurAddr=', s.get('CurAddr'), 'Self.Relay=', s.get('Relay'))
print('Self.Endpoints=', s.get('Endpoints'))
print('BackendState=', d.get('BackendState'))
"
echo "=== REMOTE netcheck (full) ==="
tailscale netcheck 2>&1 | tail -25
echo "=== REMOTE: can it reach DERP servers? (HTTPS to derp list) ==="
for h in derp1.tailscale.com derp9.tailscale.com derp17c.tailscale.com; do printf "  %-28s " "$h"; curl -sS -o /dev/null -w "%{http_code} %{time_total}s\n" --max-time 12 "https://$h/" 2>&1 | head -1; done