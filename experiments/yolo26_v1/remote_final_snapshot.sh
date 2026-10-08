#!/usr/bin/env bash
echo "== unit =="; echo "active=$(systemctl --user is-active dsh-web) enabled=$(systemctl --user is-enabled dsh-web) linger=$(loginctl show-user dgut -p Linger --value) pid=$(systemctl --user show dsh-web -p MainPID --value)"
echo "== serve =="; tailscale serve status 2>&1 | head -4
echo "== url =="; "$HOME/.dsh-web/url.sh"
echo "== listeners =="; ss -ltn | grep 3080