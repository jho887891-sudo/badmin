#!/usr/bin/env bash
# Download the OFFICIAL Tailscale stable artifacts (no root needed) and verify checksums.
set -u
DL="$HOME/.local/tailscale-dl"
VER="1.102.4"
mkdir -p "$DL"; cd "$DL" || exit 1
echo "== official listing (jammy) =="
curl -sS --max-time 30 https://pkgs.tailscale.com/stable/ubuntu/jammy/ | grep -o -E "tailscale_[0-9.]+_amd64\.deb" | sort -u | tail -3
echo "== download .deb (apt-installable) =="
curl -fsSL -o "tailscale_${VER}_amd64.deb" "https://pkgs.tailscale.com/stable/ubuntu/jammy/tailscale_${VER}_amd64.deb" && echo "deb downloaded"
curl -fsSL -o "tailscale_${VER}_amd64.deb.sha256" "https://pkgs.tailscale.com/stable/ubuntu/jammy/tailscale_${VER}_amd64.deb.sha256" && echo "deb sha256 downloaded"
echo "== download static tarball (no-root userspace fallback) =="
curl -fsSL -o "tailscale_${VER}_amd64.tgz" "https://pkgs.tailscale.com/stable/tailscale_${VER}_amd64.tgz" && echo "tgz downloaded"
curl -fsSL -o "tailscale_${VER}_amd64.tgz.sha256" "https://pkgs.tailscale.com/stable/tailscale_${VER}_amd64.tgz.sha256" && echo "tgz sha256 downloaded"
echo "== published checksums =="; cat *.sha256
echo "== verify =="
for f in "tailscale_${VER}_amd64.deb" "tailscale_${VER}_amd64.tgz"; do printf "  %-32s " "$f"; sha256sum -c "$f.sha256" 2>&1 | tail -1; done
echo "== files =="; ls -l
du -sh "$DL"