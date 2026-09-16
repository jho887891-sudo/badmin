"""Download the Shuttlecock dataset in YOLO format. Keeps the key out of every file on disk."""
import json, os, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path
KEY = os.environ.get("RF_KEY", "")
UA = "research/1.0"
WS, PROJ, VER, FMT = "mathieu-cartron", "shuttlecock-cqzy3", "1", "yolo26"
OUT = Path(r"E:\具身智能\badmin_project\_scratch_rf")
OUT.mkdir(parents=True, exist_ok=True)

url = "https://api.roboflow.com/{}/{}/{}/{}?api_key={}".format(WS, PROJ, VER, FMT, urllib.parse.quote(KEY))
req = urllib.request.Request(url, headers={"User-Agent": UA})
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.loads(r.read().decode("utf-8", "replace"))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read(400).decode("utf-8", "replace")); sys.exit(1)
print("response keys:", list(body)[:10])
link = body.get("export", {}).get("link") or body.get("link")
if not link:
    print("no link in response:", json.dumps(body)[:500]); sys.exit(1)
print("download link host:", urllib.parse.urlparse(link).netloc)
print("link length:", len(link))

dest = OUT / "shuttlecock_yolo26_v1.zip"
print("downloading to", dest)
t0 = time.time()
req2 = urllib.request.Request(link, headers={"User-Agent": UA})
with urllib.request.urlopen(req2, timeout=180) as r, dest.open("wb") as fh:
    total = int(r.headers.get("Content-Length") or 0)
    print("content-length: {:,} bytes ({:.1f} MB)".format(total, total / 1e6) if total else "content-length: unknown")
    done = 0
    last = 0
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        fh.write(chunk)
        done += len(chunk)
        if done - last > (50 << 20):
            last = done
            print("   {:.0f} MB  ({:.0f}s)".format(done / 1e6, time.time() - t0), flush=True)
print("done: {:,} bytes in {:.0f}s".format(dest.stat().st_size, time.time() - t0))