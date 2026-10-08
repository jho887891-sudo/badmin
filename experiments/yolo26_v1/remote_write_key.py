#!/usr/bin/env python3
"""Set refs.DEEPSEEK_API_KEY in ~/.dsh/.credentials.yaml. The key is read from stdin (never argv)."""
import os, re, shutil, sys, time

key = sys.stdin.readline().strip()
if not key.startswith("sk-") or len(key) < 20:
    sys.exit("refusing: unexpected key shape")
p = os.path.expanduser("~/.dsh/.credentials.yaml")
txt = open(p, encoding="utf-8").read() if os.path.exists(p) else "version: 1\n"
bak = None
if os.path.exists(p):
    bak = p + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(p, bak)
lines = txt.splitlines()
removed = False
new = []
for line in lines:
    if re.match(r"^\s+DEEPSEEK_API_KEY:\s*", line):
        removed = True
        continue
    new.append(line)
if not any(re.match(r"^refs:\s*$", l) for l in new):
    idx = 0
    for i, l in enumerate(new):
        if re.match(r"^version:\s*", l):
            idx = i + 1
            break
    new.insert(idx, "refs:")
out, inserted = [], False
for l in new:
    out.append(l)
    if not inserted and re.match(r"^refs:\s*$", l):
        out.append("  DEEPSEEK_API_KEY: " + key)
        inserted = True
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
os.chmod(p, 0o600)
print("backup:", bak)
print("replaced_existing:", removed, "inserted:", inserted, "lines:", len(out))