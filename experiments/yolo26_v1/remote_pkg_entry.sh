#!/usr/bin/env bash
set -u
echo "== full official Packages entry for 1.102.4 =="
curl -sS --max-time 40 -o /tmp/Packages "https://pkgs.tailscale.com/stable/ubuntu/dists/jammy/main/binary-amd64/Packages"
awk "BEGIN{RS=\"\"} /Version: 1.102.4/ && /Package: tailscale/ {print}" /tmp/Packages | sed -n "1,20p"
echo "== is the index signed? =="
curl -sS -o /dev/null -w "InRelease=%{http_code}\n" --max-time 25 https://pkgs.tailscale.com/stable/ubuntu/dists/jammy/InRelease
echo "== downloaded file =="; ls -l "$HOME/.local/tailscale-dl/tailscale_1.102.4_amd64.deb"; sha256sum "$HOME/.local/tailscale-dl/tailscale_1.102.4_amd64.deb"