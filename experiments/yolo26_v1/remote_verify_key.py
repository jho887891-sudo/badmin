#!/usr/bin/env python3
import hashlib, os, re, urllib.request
p = os.path.expanduser("~/.dsh/.credentials.yaml")
txt = open(p, encoding="utf-8").read()
m = re.search(r"^\s+DEEPSEEK_API_KEY:\s*(\S+)\s*$", txt, re.M)
val = m.group(1) if m else None
print("file_mode:", oct(os.stat(p).st_mode & 0o777), "bytes:", os.path.getsize(p), "has_ref:", bool(val))
if not val:
    raise SystemExit("NO KEY FOUND")
print("key_len: %d  key_sha256_prefix: %s" % (len(val), hashlib.sha256(val.encode()).hexdigest()[:8].upper()))
req = urllib.request.Request("https://api.deepseek.com/models", headers={"Authorization": "Bearer " + val})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read(160).decode("utf-8", "replace").replace("\n", " ")
        print("api_status:", r.status, "body_head:", body[:70])
except Exception as e:
    print("api_status:", getattr(e, "code", None), "err:", type(e).__name__, str(e)[:140])
# redacted structural view
for i, line in enumerate(txt.splitlines(), 1):
    mm = re.match(r"^(\s*)([A-Za-z0-9_./-]+):\s*(.*)$", line)
    if mm and mm.group(3) != "":
        v = mm.group(3)
        print("  %d: %s%s: <masked len=%d sha256=%s>" % (i, mm.group(1), mm.group(2), len(v), hashlib.sha256(v.encode()).hexdigest()[:8].upper()))
    else:
        print("  %d: %s" % (i, line))