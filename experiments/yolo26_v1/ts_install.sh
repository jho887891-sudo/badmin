#!/usr/bin/env bash
# Tailscale install helper (option A). Uses the OFFICIAL .deb already downloaded & sha256-verified,
# or falls back to the official apt repo. Scope: install + enable service + operator + bring node up.
# NO DNS override, NO routes, NO exit node, NO subnet router, NO public ports.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo "ERROR: run with sudo:  sudo bash /tmp/ts_install.sh"; exit 1; fi
. /etc/os-release
DL=/home/dgut/.local/tailscale-dl
DEB="$DL/tailscale_1.102.4_amd64.deb"
SHA=758cd0b2536d35dc1f02b785784dd18d08945ff644fcd92b1b0d597b6fa56f8b
echo "=== target: $PRETTY_NAME $(uname -m) kernel=$(uname -r) ==="
if command -v tailscale >/dev/null 2>&1; then
  echo "already installed: $(tailscale version | head -1)"
else
  if [ -f "$DEB" ]; then
    echo "=== [1/4] verify the local official .deb (sha256 vs signed Packages index) ==="
    echo "$SHA  $DEB" | sha256sum -c -
    echo "=== [2/4] install it with apt (resolves Depends: iptables from Ubuntu repos) ==="
    apt-get update -qq || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y "$DEB"
  else
    echo "=== [1/4] local .deb missing -> adding official repo ==="
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list -o /etc/apt/sources.list.d/tailscale.list
    chmod 0644 /usr/share/keyrings/tailscale-archive-keyring.gpg /etc/apt/sources.list.d/tailscale.list
    echo "=== [2/4] apt update + install tailscale ==="
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y tailscale
  fi
fi
echo "=== [3/4] enable + start tailscaled (starts at boot) ==="
systemctl enable --now tailscaled
echo "active=$(systemctl is-active tailscaled) enabled=$(systemctl is-enabled tailscaled)"
tailscale version | head -1
echo "=== allow user dgut to manage tailscale without sudo ==="
tailscale set --operator=dgut || true
echo "=== [4/4] tailscale up (no DNS override, no routes, no exit node) ==="
echo "    >>> if a login URL appears below, open it in your browser and authorise this machine <<<"
tailscale up --hostname=jxxy --accept-dns=false --accept-routes=false
echo "=== result ==="
tailscale ip -4
tailscale status