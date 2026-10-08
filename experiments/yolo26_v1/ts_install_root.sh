#!/usr/bin/env bash
# Option A: system install of Tailscale from the sha256-verified OFFICIAL .deb. Run as root (sudo -S).
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo "ERR: need root"; exit 1; fi
DL=/home/dgut/.local/tailscale-dl
DEB="$DL/tailscale_1.102.4_amd64.deb"
SHA=758cd0b2536d35dc1f02b785784dd18d08945ff644fcd92b1b0d597b6fa56f8b
. /etc/os-release
echo "=== target: $PRETTY_NAME $(uname -m) ==="
echo "=== [1/5] verify official .deb sha256 ==="
echo "$SHA  $DEB" | sha256sum -c -
echo "=== [2/5] install with apt (resolves Depends: iptables) ==="
apt-get update -qq || true
DEBIAN_FRONTEND=noninteractive apt-get install -y "$DEB"
echo "=== [3/5] enable + start tailscaled (starts at boot) ==="
systemctl enable --now tailscaled
echo "active=$(systemctl is-active tailscaled) enabled=$(systemctl is-enabled tailscaled) version=$(tailscale version | head -1)"
echo "=== [4/5] allow user dgut to manage tailscale without sudo ==="
tailscale set --operator=dgut || true
echo "=== [5/5] tailscale up (detached; NO dns override, NO routes, NO exit node) ==="
setsid nohup tailscale up --hostname=jxxy --accept-dns=false --accept-routes=false > /tmp/ts_up.log 2>&1 < /dev/null &
sleep 12
echo "--- /tmp/ts_up.log ---"; cat /tmp/ts_up.log
echo "--- status ---"; tailscale status 2>&1 | head -8
echo "--- ip ---"; tailscale ip -4 2>&1 | head -2