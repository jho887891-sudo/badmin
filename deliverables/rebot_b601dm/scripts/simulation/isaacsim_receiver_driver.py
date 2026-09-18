#!/usr/bin/env python3
"""Run the upstream reBot Isaac Sim receiver with this host's adaptations applied in memory.

Run it through scripts/simulation/run_isaacsim_receiver.sh, which sets PYTHONPATH and the project python.

What it adapts, and why each is necessary here:
  headless   the upstream forces {"headless": False} and DISPLAY is unset on this host
  bind host  the upstream binds 192.168.1.66, which this machine does not own
  asset      optionally the patched tree carrying real joint velocity limits instead of the shipped 9x ones

Nothing is written into either upstream repository; the receiver file on disk still says headless False.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.rebot_b601dm.launch import (  # noqa: E402
    choose_asset_path,
    choose_bind_host,
    force_headless,
    upstream_receiver_path,
)

UPSTREAM = Path(os.environ.get("REBOT_UPSTREAM", REPO_ROOT / "reBot-Isaacsim"))
PATCHED = Path(os.environ.get("REBOT_ASSET_DIR", REPO_ROOT / "outputs/simulation/rebot_b601dm/real_limits"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=os.environ.get("REBOT_BIND_HOST", "192.168.1.66"),
                   help="requested bind address; falls back to loopback if this host does not own it")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--asset", default=None, help="explicit USD path; otherwise the patched tree is preferred")
    p.add_argument("--upstream", default=str(UPSTREAM))
    p.add_argument("--patched-dir", default=str(PATCHED))
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved configuration and exit without starting Isaac Sim")
    return p.parse_args(argv)


def load_upstream(path: Path):
    spec = importlib.util.spec_from_file_location("rebot_upstream_receiver", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    args = parse_args(argv)
    receiver = upstream_receiver_path(args.upstream)
    if not receiver.is_file():
        print("[error] upstream receiver not found: " + str(receiver), file=sys.stderr)
        return 2

    host = choose_bind_host(args.host)
    asset = Path(args.asset) if args.asset else choose_asset_path(args.patched_dir, args.upstream)

    print("=" * 72)
    print("  reBot B601-DM Isaac Sim receiver (candidate arm)", flush=True)
    print("  upstream    : " + str(receiver))
    print("  bind        : udp://" + host + ":" + str(args.port)
          + ("   (requested " + args.host + ", not owned by this host)" if host != args.host else ""))
    print("  asset       : " + str(asset))
    print("  headless    : forced at runtime")
    print("=" * 72)

    if args.dry_run:
        return 0

    if not asset.is_file():
        print("[error] asset not found: " + str(asset), file=sys.stderr)
        return 2

    print("[step 1/4] loading the upstream receiver module", flush=True)
    module = load_upstream(receiver)
    force_headless(module)
    print("[step 2/4] constructing the mirror (binds " + host + ":" + str(args.port) + ")", flush=True)

    mirror = module.IsaacJointMirror(host=host, port=args.port)
    mirror.asset_path = asset
    try:
        print("[step 3/4] starting Isaac Sim - this host has taken 100-200 s before", flush=True)
        mirror.setup_isaac_sim()
        print("[step 4/4] Isaac Sim started; ground and robot asset loaded", flush=True)
        mirror.run()
    finally:
        mirror.shutdown()
        print("[done] exited safely")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
