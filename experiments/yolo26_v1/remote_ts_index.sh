#!/usr/bin/env bash
set -u
DL="$HOME/.local/tailscale-dl"
echo "== 1) official apt repo line (authoritative) =="
curl -sS --max-time 25 https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list | head -5
echo "== 2) official Packages index -> real .deb path =="
for u in "https://pkgs.tailscale.com/stable/ubuntu/dists/jammy/main/binary-amd64/Packages" "https://pkgs.tailscale.com/stable/ubuntu/dists/jammy/main/binary-amd64/Packages.gz"; do
  printf "  %s -> " "$u"; curl -sS -o /tmp/pkgidx -w "%{http_code} %{size_download}B\n" --max-time 30 "$u"
  if [ -s /tmp/pkgidx ]; then
    case "$u" in *.gz) zcat /tmp/pkgidx 2>/dev/null | grep -E "^(Package|Version|Filename):" | head -12 ;; *) grep -E "^(Package|Version|Filename):" /tmp/pkgidx | head -12 ;; esac
    break
  fi
done
echo "== 3) verify static tarball manually against published hash =="
PUB=$(tr -d "[:space:]" < "$DL/tailscale_1.102.4_amd64.tgz.sha256")
ACT=$(sha256sum "$DL/tailscale_1.102.4_amd64.tgz" | cut -d" " -f1)
echo "  published=$PUB"
echo "  actual   =$ACT"
[ "$PUB" = "$ACT" ] && echo "  TARBALL_SHA256_OK" || echo "  TARBALL_SHA256_MISMATCH"