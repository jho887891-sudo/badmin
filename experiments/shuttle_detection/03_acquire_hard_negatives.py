"""Acquire HARD_NEGATIVE_POOL images from Wikimedia Commons (spec 06 section 6).

The false-positive diagnosis found the model firing on two scene types: badminton halls with distant
players in white, and group photographs of people in light-coloured shirts. This collects NEW images
with that character. It never touches the frozen test pool or the training backgrounds.
"""
import json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "shuttle_capability" / "hard_negatives"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = "badmin-project-research/1.0 (detector hard-negative collection)"
# Wikimedia rate-limits aggressively: a 36-image run hit HTTP 429 partway through on upload.wikimedia.org.
# Requests are therefore spaced, 429 is retried with a long backoff, and the console output is forced to
# ASCII-safe text because this host's console codec is GBK and titles are frequently non-ASCII.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
REQUEST_GAP_S = 1.2
API = "https://commons.wikimedia.org/w/api.php"

QUERIES = [
    "badminton hall",
    "badminton court indoor players",
    "volleyball match indoor",
    "group photo ceremony people",
    "conference group photograph",
    "sports hall interior",
]

def get(url, tries=4):
    for k in range(tries):
        time.sleep(REQUEST_GAP_S)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(15 + 15 * k)
                continue
            if k == tries - 1:
                print("   FAIL", type(exc).__name__, str(exc)[:70])
                return None
            time.sleep(3 + 3 * k)
        except Exception as exc:
            if k == tries - 1:
                print("   FAIL", type(exc).__name__, str(exc)[:70])
                return None
            time.sleep(3 + 3 * k)
    return None

def search(query, limit=12):
    q = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": query, "srnamespace": "6",
        "srlimit": str(limit), "format": "json"})
    body = get(API + "?" + q)
    if not body:
        return []
    try:
        return [x["title"] for x in json.loads(body)["query"]["search"]]
    except Exception:
        return []

def thumb_url(title, width=1280):
    q = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "iiurlwidth": str(width), "format": "json"})
    body = get(API + "?" + q)
    if not body:
        return None, {}
    try:
        pages = json.loads(body)["query"]["pages"]
        page = next(iter(pages.values()))
        ii = page["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "unknown")
        return ii.get("thumburl") or ii.get("url"), {"license": lic, "page": ii.get("descriptionurl", "")}
    except Exception:
        return None, {}

rows = []
seen = set()
for query in QUERIES:
    print("== query:", query)
    for title in search(query):
        if title in seen:
            continue
        seen.add(title)
        if not title.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        url, meta = thumb_url(title)
        if not url:
            continue
        data = get(url)
        if not data or len(data) < 20000:
            continue
        name = "hn_{:03d}.jpg".format(len(rows) + 1)
        (RAW / name).write_bytes(data)
        rows.append({"file": name, "title": title, "license": meta.get("license", "?"),
                     "page": meta.get("page", ""), "url": url, "bytes": len(data)})
        safe = title[:48].encode("ascii", "replace").decode("ascii")
        print("   saved {}  {:<48} {}".format(name, safe, meta.get("license", "?")))
        if len(rows) >= 36:
            break
    if len(rows) >= 36:
        break

import csv
with (OUT / "hard_negative_metadata.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file", "title", "license", "page", "url", "bytes"])
    w.writeheader()
    w.writerows(rows)
print()
print("downloaded", len(rows), "candidate hard negatives to", RAW)