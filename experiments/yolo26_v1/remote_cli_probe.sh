#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
echo "== dsh --help =="; "$HOME/.local/bin/dsh" --help 2>&1 | head -45
echo
echo "== try session-ish subcommands =="
for c in sessions session list ls status profile profiles; do printf "--- %s ---\n" "$c"; "$HOME/.local/bin/dsh" $c --help 2>&1 | head -6; done