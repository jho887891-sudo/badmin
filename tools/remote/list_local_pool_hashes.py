import hashlib
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""):
            h.update(c)
    return h.hexdigest()
root=Path("outputs/shuttle_capability")
lines=[]
for pool in ("hard_negatives","hard_negatives2"):
    d=root/pool/"raw"
    for p in sorted(d.glob("*.jpg")):
        lines.append("%s  %s/raw/%s" % (sha(p), pool, p.name))
print(chr(10).join(lines))