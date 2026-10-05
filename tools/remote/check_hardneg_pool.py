import hashlib, csv
from pathlib import Path
from collections import Counter
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda: f.read(1<<20), b""):
            h.update(c)
    return h.hexdigest()
out={}
for pool in ("hard_negatives/raw","hard_negatives2"):
    d=Path("outputs/shuttle_capability")/pool
    files=sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"))
    shas=[sha(p) for p in files]
    out[pool]=dict(files=files, shas=shas)
    print("%-20s files=%-4d unique_sha=%-4d" % (pool, len(files), len(set(shas))))
    lab=list(d.glob("*.txt"))
    print("   txt/csv:", [x.name for x in d.iterdir() if x.suffix.lower() in (".txt",".csv")])
    for x in d.glob("*.txt"):
        body=x.read_text(encoding="utf-8",errors="replace").splitlines()
        print("   %s lines=%d sample=%s" % (x.name, len(body), body[:4]))
    for x in d.glob("*.csv"):
        rows=list(csv.DictReader(x.open(encoding="utf-8")))
        print("   %s rows=%d cols=%s" % (x.name, len(rows), list(rows[0].keys())[:8] if rows else []))
allsha = out["hard_negatives/raw"]["shas"] + out["hard_negatives2"]["shas"]
print("combined files=%d unique_sha=%d" % (len(allsha), len(set(allsha))))
tr=set(r["sha256"] for r in csv.DictReader(open("data/eth_only_v1_train_manifest.csv",encoding="utf-8")))
va=set(r["sha256"] for r in csv.DictReader(open("data/eth_only_v1_val_manifest.csv",encoding="utf-8")))
print("overlap with V1 train=%d, V1 val=%d" % (len(set(allsha)&tr), len(set(allsha)&va)))