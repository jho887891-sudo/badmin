#!/usr/bin/env bash
echo "=== 远端(校园网) -> VPS 端口探测 ==="
for p in 22 80 443 3478 20736; do
  timeout 4 bash -c "cat < /dev/null > /dev/tcp/223.109.239.11/$p" 2>/dev/null && echo "  tcp/$p : OPEN" || echo "  tcp/$p : closed/filtered"
done
echo "=== tailnet 是否已启用 HTTPS 证书（决定证书怎么来） ==="
cd /tmp && timeout 30 tailscale cert jxxy.taildd42cc.ts.net 2>&1 | head -6; ls -l /tmp/*.crt /tmp/*.key 2>/dev/null | head -4
echo "=== 当前 tailnet DERP 映射（是否已有自建 DERP） ==="
tailscale debug netmap 2>/dev/null | head -c 300 || echo "(netmap unavailable)"