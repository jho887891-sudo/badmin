"""Leak guard: are any Roboflow images byte-identical to, or near-duplicates of, the frozen sets?

The user warned that public shuttlecock datasets are likely to be copies or re-workings of each other and
should be hash-deduped and near-duplicate-checked before entering training. This checks them against every
frozen set in this repository, which is the rule the whole project runs on.
"""
import hashlib, sys, io
from pathlib import Path
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\具身智能\badmin_project")
RF = REPO / "_scratch_rf" / "extracted"

def read(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def sig(p, n=48):
    im = read(p)
    if im is None:
        return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)
    g -= g.mean()
    s = g.std()
    return g / s if s > 1e-9 else None

FROZEN = {
    "core test backgrounds": REPO / "outputs/shuttle_capability/real_images/backgrounds",
    "real photographs": REPO / "outputs/shuttle_capability/real_images/raw",
    "synthetic on real bg": REPO / "outputs/shuttle_capability/synthetic_on_real_bg/images",
    "synthetic 3d": REPO / "outputs/shuttle_capability/synthetic_3d/images",
    "controlled matrix": REPO / "outputs/shuttle_capability/controlled_capability/images",
    "challenge test": REPO / "outputs/shuttle_capability/challenge_test/images",
    "real video frames": REPO / "outputs/shuttle_capability/real_video/frames",
    "hard negatives": REPO / "outputs/shuttle_capability/hard_negatives/raw",
    "hard negatives 2": REPO / "outputs/shuttle_capability/hard_negatives2/raw",
    "real match frames": REPO / "outputs/shuttle_capability/real_match_frames/images",
}

frozen_hash = {}
frozen_sig = []
for label, d in FROZEN.items():
    if not d.is_dir():
        print("  (absent)", label); continue
    n = 0
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        frozen_hash[sha(p)] = "{}/{}".format(label, p.name)
        s = sig(p)
        if s is not None:
            frozen_sig.append((label, p.name, s))
        n += 1
    print("  {:>5} images  {}".format(n, label))
print()
print("frozen images hashed:", len(frozen_hash), "| signatures:", len(frozen_sig))

rf_imgs = sorted((RF / "train" / "images").glob("*.jpg")) + sorted((RF / "valid" / "images").glob("*.jpg"))
print("roboflow images to check:", len(rf_imgs))
print()
exact = 0
near = []
checked = 0
for p in rf_imgs:
    checked += 1
    if sha(p) in frozen_hash:
        exact += 1
        print("  EXACT:", p.name, "==", frozen_hash[sha(p)])
        continue
    s = sig(p)
    if s is None:
        continue
    best = (0.0, None, None)
    for label, name, fs in frozen_sig:
        c = float((s * fs).mean())
        if c > best[0]:
            best = (c, label, name)
    if best[0] >= 0.90:
        near.append((best[0], p.name, best[1], best[2]))
print()
print("=== result over {} images ===".format(checked))
print("  byte-identical to a frozen image :", exact)
print("  near-duplicate (r >= 0.90)       :", len(near))
for r, a, b, c in sorted(near, reverse=True)[:12]:
    print("     r={:.4f}  {}  ~  {}/{}".format(r, a[:40], b, c[:30]))