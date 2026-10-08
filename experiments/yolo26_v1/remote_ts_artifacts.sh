#!/usr/bin/env bash
echo "== official tarball availability (for the no-root option) =="
for u in https://pkgs.tailscale.com/stable/tailscale_1.102.4_amd64.tgz https://pkgs.tailscale.com/stable/tailscale_1.102.4_amd64.tgz.sha256; do
  printf "  %-70s " "$u"; curl -sS -o /dev/null -w "%{http_code} %{size_download}B\n" --max-time 25 -r 0-0 "$u"
done
echo "== published sha256 (official) =="
curl -sS --max-time 25 https://pkgs.tailscale.com/stable/tailscale_1.102.4_amd64.tgz.sha256 2>/dev/null | head -2
echo "== apt repo key/list reachable (for the root option) =="
for u in https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list; do
  printf "  %-70s " "$u"; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 20 "$u"
done