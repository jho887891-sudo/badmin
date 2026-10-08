#!/usr/bin/env python3
import hashlib, os, re
p = os.path.expanduser("~/.dsh/.credentials.yaml")
print("exists:", os.path.exists(p), "mode:", oct(os.stat(p).st_mode & 0o777) if os.path.exists(p) else "-", "bytes:", os.path.getsize(p) if os.path.exists(p) else 0)
for i, line in enumerate(open(p, encoding="utf-8").read().splitlines(), 1):
    m = re.match(r"^(\s*)([A-Za-z0-9_./-]+):\s*(.*)$", line)
    if m and m.group(3) != "":
        v = m.group(3)
        print("%3d: %s%s: <masked len=%d sha256=%s>" % (i, m.group(1), m.group(2), len(v), hashlib.sha256(v.encode()).hexdigest()[:8].upper()))
    else:
        print("%3d: %s" % (i, line))