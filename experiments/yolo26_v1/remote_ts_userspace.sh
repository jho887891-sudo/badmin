#!/usr/bin/env bash
# Unprivileged (userspace-networking) Tailscale for the remote host. No sudo, no system config changes.
set -u
DL="$HOME/.local/tailscale-dl"
TS="$HOME/.local/tailscale"
mkdir -p "$TS/bin" "$TS/state" "$TS/run"
echo "== extract official tarball =="
tar xzf "$DL/tailscale_1.102.4_amd64.tgz" -C "$TS/bin" --strip-components=1
ls -l "$TS/bin"
"$TS/bin/tailscale" --version | head -1
echo "== stop any previous userspace instance =="
pkill -f "[t]ailscaled --tun=userspace-networking" 2>/dev/null || true
sleep 2
echo "== start userspace daemon (no TUN, no root) =="
setsid nohup "$TS/bin/tailscaled" --tun=userspace-networking \
  --state="$TS/state/tailscaled.state" \
  --socket="$TS/run/tailscaled.sock" \
  --socks5-server=localhost:1055 \
  --port=0 > "$TS/tailscaled.log" 2>&1 < /dev/null &
sleep 6
echo "daemon_pid=$(pgrep -f "[t]ailscaled --tun=userspace-networking" | head -1)"
echo "== daemon log =="; tail -5 "$TS/tailscaled.log"
echo "== start login (background; prints URL and waits for your authorisation) =="
setsid nohup "$TS/bin/tailscale" --socket="$TS/run/tailscaled.sock" up \
  --hostname=jxxy --accept-dns=false --accept-routes=false > "$TS/up.log" 2>&1 < /dev/null &
sleep 15
echo "== up.log =="; cat "$TS/up.log"
echo "== status =="; "$TS/bin/tailscale" --socket="$TS/run/tailscaled.sock" status 2>&1 | head -10