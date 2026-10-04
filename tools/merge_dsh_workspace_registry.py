#!/usr/bin/env python3
"""Merge the LOCAL DSH workspace registry into the REMOTE one, rewriting project paths.

Keeps the remote entries untouched, adds the local workspaces (same ids, so their sessionIds still link),
and writes a timestamped backup of the remote file first.
"""
import json, os, shutil, sys, time

MAP = {
    r"E:\具身智能\badmin_project": "/home/T7/ojh/badmin_project",
    r"E:\具身智能": "/home/T7/ojh",
}


def main() -> int:
    local_p = sys.argv[1]
    remote_p = os.path.expanduser("~/.dsh/storages/workspace.json")
    rem = json.load(open(remote_p, encoding="utf-8"))
    loc = json.load(open(local_p, encoding="utf-8"))
    bak = remote_p + ".bak-" + time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(remote_p, bak)
    ids = list(rem.get("global", {}).get("workspaceIds", []))
    tw = rem.setdefault("tables", {}).setdefault("workspaces", {})
    added, skipped = [], []
    for wid, w in loc.get("tables", {}).get("workspaces", {}).items():
        newpath = MAP.get(w.get("path", ""))
        if newpath is None:
            skipped.append((w.get("path"), "no mapping")); continue
        if wid in tw:
            skipped.append((w.get("path"), "id already present")); continue
        w2 = dict(w)
        w2["path"] = newpath
        tw[wid] = w2
        if wid not in ids:
            ids.append(wid)
        added.append({"id": wid, "path": newpath, "sessions": len(w.get("sessionIds", []))})
    rem.setdefault("global", {})["workspaceIds"] = ids
    with open(remote_p, "w", encoding="utf-8") as fh:
        json.dump(rem, fh, indent=2, ensure_ascii=False)
    print("backup:", bak)
    print("added:", json.dumps(added, ensure_ascii=False))
    print("skipped:", json.dumps(skipped, ensure_ascii=False))
    print("workspaces now:", json.dumps([(k, v.get("path"), len(v.get("sessionIds", []))) for k, v in tw.items()], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())