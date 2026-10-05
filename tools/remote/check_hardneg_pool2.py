import hashlib, csv
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""):
            h.update(c)
    return h.hexdigest()
root=Path("outputs/shuttle_capability")
all_files=[]
for pool in ("hard_negatives","hard_negatives2"):
    d=root/pool
    print("==", pool)
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() not in (".jpg",".jpeg",".png"):
            print("   doc:", p.relative_to(d), p.stat().st_size)
    imgs=sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"))
    print("   images=%d dirs=%s" % (len(imgs), sorted(set(str(p.parent.relative_to(d)) for p in imgs))))
    all_files += imgs
out={}
for p in all_files:
    out.setdefault(sha(p), []).append(str(p))
print("combined images=%d unique_sha=%d duplicates=%d" % (len(all_files), len(out), len(all_files)-len(out)))
excl=set((root/"hard_negatives2"/"excluded.txt").read_text(encoding="utf-8").split())
kept=[p for p in all_files if p.name not in excl]
print("after excluding the %d names listed in excluded.txt: images=%d" % (len(excl), len(kept)))
tr=set(r["sha256"] for r in csv.DictReader(open("data/eth_only_v1_train_manifest.csv",encoding="utf-8")))
va=set(r["sha256"] for r in csv.DictReader(open("data/eth_only_v1_val_manifest.csv",encoding="utf-8")))
ks=set(sha(p) for p in kept)
print("overlap kept-vs-V1train=%d kept-vs-V1val=%d" % (len(ks&tr), len(ks&va)))
print("exposure: +%d rows on top of 14543 = +%.2f%%" % (len(ks), 100.0*len(ks)/14543))