#!/usr/bin/env bash
echo "== os/arch =="; . /etc/os-release; echo "$PRETTY_NAME $(uname -m) kernel=$(uname -r)"; ldd --version | head -1
echo "== tailscale present? =="; command -v tailscale tailscaled || echo "tailscale MISSING (both binaries)"
echo "== apt candidate =="; apt-cache policy tailscale 2>/dev/null | head -6 || echo "no apt metadata"
echo "== sudo non-interactive? =="; sudo -n -v 2>&1 | head -2; echo "sudo_n_rc=$?"
echo "== systemd =="; systemctl --version 2>/dev/null | head -1; echo "user_systemd=$(systemctl --user is-system-running 2>&1 | head -1)"
echo "== tun device =="; ls -l /dev/net/tun 2>&1; echo "kernel_tun=$(lsmod | grep -c "^tun" 2>/dev/null)"
echo "== network egress to Tailscale =="
for u in https://pkgs.tailscale.com/stable/ https://controlplane.tailscale.com/ https://login.tailscale.com/ https://tailscale.com/; do printf "  %-42s " "$u"; curl -sS -o /dev/null -w "%{http_code} %{time_total}s\n" --max-time 20 "$u" 2>&1 | head -1; done
echo "== latest stable version (official repo listing) =="
curl -sS --max-time 25 https://pkgs.tailscale.com/stable/ 2>/dev/null | grep -o -E "tailscale_[0-9.]+" | head -3
curl -sS --max-time 25 https://pkgs.tailscale.com/stable/ubuntu/jammy/tailscale_latest_amd64.deb 2>/dev/null | head -c 0; echo "jammy_latest_head_rc=$?"
echo "== disk =="; df -h / | tail -1
echo "== /tmp space for download =="; df -h /tmp | tail -1