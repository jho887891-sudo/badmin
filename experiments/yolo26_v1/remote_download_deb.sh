#!/usr/bin/env bash
set -u
DL="$HOME/.local/tailscale-dl"; cd "$DL" || exit 1
VER=1.102.4
echo "== download .deb from the authoritative pool path =="
curl -fsSL -o "tailscale_${VER}_amd64.deb" "https://pkgs.tailscale.com/stable/ubuntu/pool/tailscale_${VER}_amd64.deb" && echo "downloaded"
ls -l "tailscale_${VER}_amd64.deb"
echo "== official checksum for this exact file (from the signed Packages index) =="
curl -sS --max-time 40 -o /tmp/Packages "https://pkgs.tailscale.com/stable/ubuntu/dists/jammy/main/binary-amd64/Packages"
python3 - <<PY
import re
txt = open("/tmp/Packages", encoding="utf-8", errors="replace").read()
for b in txt.split("\n\n"):
    if "\nPackage: tailscale\n" in "\n" + b + "\n" and "\nVersion: ${VER}\n" in "\n" + b + "\n":
        d = dict(re.findall(r"^([A-Za-z-]+): (.*)$", b, re.M))
        print("Version :", d.get("Version"))
        print("Filename:", d.get("Filename"))
        print("Size    :", d.get("Size"))
        print("SHA256  :", d.get("SHA256"))
        print("Depends :", d.get("Depends"))
        print("Installed-Size:", d.get("Installed-Size"))
PY
echo "== local hash =="; sha256sum "tailscale_${VER}_amd64.deb"
echo "== dpkg metadata (no root needed) =="; dpkg-deb -I "tailscale_${VER}_amd64.deb" 2>/dev/null | sed -n "1,14p"
echo "== files in package (top-level) =="; dpkg-deb -c "tailscale_${VER}_amd64.deb" 2>/dev/null | awk "{print \$6}" | grep -E "^/(usr/)?(s)?bin/|^/lib/systemd|^/usr/lib/systemd" | head -8