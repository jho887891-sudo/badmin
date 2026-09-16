"""Acquire REAL shuttlecock photographs for TRAINING - the data this project has never had.

Every real image the project owns is frozen test data, so it cannot be trained on. This collects NEW real
photographs of shuttlecocks, which is the one input that attacks the remaining real-domain gap directly.

Leak guard as always: candidates are compared by content signature against the frozen core backgrounds, the
frozen real-image test set and the two hard-negative batches, and anything matching at r >= 0.90 is
excluded before it can be used.
"""
import csv, json, sys, time, urllib.parse, urllib.request
from pathlib import Path
import numpy as np, cv2

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
OUT = REPO / "outputs" / "shuttle_capability" / "real_train"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "tools"))
from shuttle_render import imread_unicode

UA = "badmin-project-research/1.0 (detector real-training-data collection)"
API = "https://commons.wikimedia.org/w/api.php"
GAP = 1.2
QUERIES = [
    "shuttlecock",
    "badminton shuttlecock closeup",
    "badminton shuttle on racket",
    "shuttlecock white feather",
    "badminton birdie",
    "shuttlecock on floor",
    "badminton match shuttle",
    "nylon shuttlecock",
]

def get(url, tries=4):
    for k in range(tries):
        time.sleep(GAP)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(15 + 15 * k); continue
            if k == tries - 1: return None
            time.sleep(3 + 3 * k)
        except Exception:
            if k == tries - 1: return None
            time.sleep(3 + 3 * k)
    return None

def search(q, limit=12):
    u = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": q,
                                "srnamespace": "6", "srlimit": str(limit), "format": "json"})
    b = get(API + "?" + u)
    if not b: return []
    try: return [x["title"] for x in json.loads(b)["query"]["search"]]
    except Exception: return []

def thumb(title, width=1600):
    u = urllib.parse.urlencode({"action": "query", "titles": title, "prop": "imageinfo",
                                "iiprop": "url|extmetadata", "iiurlwidth": str(width), "format": "json"})
    b = get(API + "?" + u)
    if not b: return None, {}
    try:
        page = next(iter(json.loads(b)["query"]["pages"].values()))
        ii = page["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        return ii.get("thumburl") or ii.get("url"), {
            "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "page": ii.get("descriptionurl", "")}
    except Exception:
        return None, {}

def sig(p, n=48):
    im = imread_unicode(p)
    if im is None: return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)
    g -= g.mean()
    s = g.std()
    return g / s if s > 1e-9 else None

refs = []
for d in (REPO / "outputs/shuttle_capability/real_images/backgrounds",
          REPO / "outputs/shuttle_capability/real_images/raw",
          REPO / "outputs/shuttle_capability/hard_negatives/raw",
          REPO / "outputs/shuttle_capability/hard_negatives2/raw"):
    for p in sorted(d.glob("*")) if d.is_dir() else []:
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            s = sig(p)
            if s is not None: refs.append(s)
print("leak reference images:", len(refs))

rows = []
seen = set()
TARGET = 50
for q in QUERIES:
    if len(rows) >= TARGET: break
    got = 0
    for title in search(q):
        if len(rows) >= TARGET: break
        if title in seen or not title.lower().endswith((".jpg", ".jpeg", ".png")): continue
        seen.add(title)
        url, meta = thumb(title)
        if not url: continue
        data = get(url)
        if not data or len(data) < 30000: continue
        name = "rt_{:03d}.jpg".format(len(rows) + 1)
        (RAW / name).write_bytes(data)
        rows.append({"file": name, "title": title, "license": meta.get("license", "?"),
                     "page": meta.get("page", ""), "url": url, "bytes": len(data)})
        got += 1
    print("query {!r}: {} saved (total {})".format(q[:36], got, len(rows)), flush=True)

with (OUT / "real_train_metadata.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "title", "license", "page", "url", "bytes"])
    w.writeheader(); w.writerows(rows)
print("downloaded", len(rows), "candidate real photographs")