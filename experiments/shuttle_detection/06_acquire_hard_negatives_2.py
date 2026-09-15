"""Acquire a SECOND hard-negative batch, targeted at what baseline_isaac fires on.

The first batch (33 images) is already in training, and the new model fires on 28 of the 36 candidates
while firing MORE on the 30 frozen shuttle-free scenes (65 -> 82). So the character has not changed - it
needs new instances of it, which is what spec 06 section 6 asks for.

Rate limiting: a previous run hit HTTP 429, so requests are spaced and 429 is retried with a long
backoff. Output goes to a separate directory so the first batch is not overwritten.
"""
import csv, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

# Resolve the repo from this file so the script runs where the network is. The training host has no
# outbound internet (an api.wikimedia.org request dies with an SSL handshake timeout), so acquisition
# happens on the workstation and the images are synced afterwards.
_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
OUT = REPO / "outputs" / "shuttle_capability" / "hard_negatives2"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = "badmin-project-research/1.0 (detector hard-negative collection)"
API = "https://commons.wikimedia.org/w/api.php"
GAP = 1.2

QUERIES = [
    "badminton tournament players court",
    "badminton doubles match",
    "sports hall badminton",
    "volleyball women match",
    "basketball game indoor players",
    "team photo group sport",
    "graduation ceremony group photo",
    "conference delegates group",
    "audience auditorium seated",
    "spectators indoor sport",
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
                time.sleep(15 + 15 * k)
                continue
            if k == tries - 1:
                return None
            time.sleep(3 + 3 * k)
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(3 + 3 * k)
    return None

def search(query, limit=10):
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                "srnamespace": "6", "srlimit": str(limit), "format": "json"})
    body = get(API + "?" + q)
    if not body:
        return []
    try:
        return [x["title"] for x in json.loads(body)["query"]["search"]]
    except Exception:
        return []

def thumb(title, width=1280):
    q = urllib.parse.urlencode({"action": "query", "titles": title, "prop": "imageinfo",
                                "iiprop": "url|extmetadata", "iiurlwidth": str(width), "format": "json"})
    body = get(API + "?" + q)
    if not body:
        return None, {}
    try:
        page = next(iter(json.loads(body)["query"]["pages"].values()))
        ii = page["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        return ii.get("thumburl") or ii.get("url"), {
            "license": meta.get("LicenseShortName", {}).get("value", "unknown"),
            "page": ii.get("descriptionurl", "")}
    except Exception:
        return None, {}

rows = []
seen = set()
TARGET = 60
for query in QUERIES:
    if len(rows) >= TARGET:
        break
    got = 0
    for title in search(query):
        if len(rows) >= TARGET:
            break
        if title in seen or not title.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        seen.add(title)
        url, meta = thumb(title)
        if not url:
            continue
        data = get(url)
        if not data or len(data) < 20000:
            continue
        name = "hn2_{:03d}.jpg".format(len(rows) + 1)
        (RAW / name).write_bytes(data)
        rows.append({"file": name, "title": title, "license": meta.get("license", "?"),
                     "page": meta.get("page", ""), "url": url, "bytes": len(data)})
        got += 1
    print("query {!r}: {} saved (total {})".format(query[:40], got, len(rows)), flush=True)

with (OUT / "hard_negative_metadata.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "title", "license", "page", "url", "bytes"])
    w.writeheader()
    w.writerows(rows)
print("downloaded", len(rows), "new candidates to", RAW)