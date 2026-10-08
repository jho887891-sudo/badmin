#!/usr/bin/env bash
echo "== os =="; (cat /etc/os-release | head -3); uname -srm
echo "== identity =="; whoami; echo "HOME=$HOME"; id
echo "== tools =="
for c in node npm npx git python3 pip3 curl wget tar xz jq; do
  if command -v $c >/dev/null 2>&1; then printf "%-8s %s | " "$c" "$(command -v $c)"; $c --version 2>&1 | head -1; else echo "$c MISSING"; fi
done
echo "== node installs present =="; ls -d $HOME/.nvm $HOME/.local/bin /usr/local/bin/node /opt/node* /usr/lib/node_modules 2>/dev/null
echo "== disk =="; df -h / /home/dgut /tmp 2>/dev/null | tail -4
echo "== sudo non-interactive =="; sudo -n true 2>&1 | head -2; echo "sudo_rc=$?"
echo "== tailscale / net =="; command -v tailscale || echo "tailscale MISSING"; ip -brief addr 2>/dev/null | head -6
echo "== listening ports =="; ss -ltn 2>/dev/null | head -12
echo "== existing dsh =="; command -v dsh || echo "dsh MISSING"; ls -d $HOME/.dsh 2>/dev/null || echo "no ~/.dsh"
echo "== npm prefix/cache =="; (npm config get prefix 2>/dev/null); (npm config get cache 2>/dev/null)
echo "== resources =="; nproc; free -g | head -2
echo "== glibc =="; ldd --version | head -1