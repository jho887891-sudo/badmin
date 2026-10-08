#!/usr/bin/env bash
echo "== leftover userspace attempt? =="
ls -ld "$HOME/.local/tailscale" 2>/dev/null && du -sh "$HOME/.local/tailscale" 2>/dev/null || echo "no ~/.local/tailscale dir"
pgrep -af "[t]ailscaled" | head -3
echo "== downloaded artifacts kept =="
du -sh "$HOME/.local/tailscale-dl" 2>/dev/null; ls -1 "$HOME/.local/tailscale-dl" 2>/dev/null
echo "== dsh web still loopback-only (untouched by this step) =="
ss -ltn | grep ":3080" || echo "not listening"
echo "== tailscale settings actually applied =="
tailscale debug prefs 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); [print(k,'=',d.get(k)) for k in ['Hostname','CorpDNS','RouteAll','ExitNodeID','AdvertiseRoutes','WantRunning','NoSNAT','ShieldsUp']]" 2>/dev/null || tailscale debug prefs 2>/dev/null | head -20