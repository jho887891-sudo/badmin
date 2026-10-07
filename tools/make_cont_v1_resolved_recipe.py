#!/usr/bin/env python3
"""Turn the pre-training dry-run manifest into the resolved-recipe artifact (spec sections 11 and 26).

Nothing here is hand-written: the resolved ultralytics kwargs come from the trainer's own resolve_recipe() output that
the dry-run recorded, so the artifact cannot drift from what training will actually use.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun-manifest", required=True)
    ap.add_argument("--out", default="outputs/shuttle_capability/metrics/"
                                    "yolo26s_v2_full_eth_cont_v1_resolved_recipe.json")
    a = ap.parse_args([] if argv is None else argv)
    src = Path(a.dryrun_manifest)
    run = json.loads(src.read_text(encoding="utf-8"))
    out = {"experiment": run["experiment"],
           "generated_from": {"dryrun_manifest": str(src), "sha256": sha256(src)},
           "contract": run.get("contract"), "weights": run.get("weights"), "data": run.get("data"),
           "resolved_kwargs": run.get("resolved_kwargs"), "batch_policy": {
               "contract_batch": run.get("batch_attempts", [None])[0],
               "batch_attempts": run.get("batch_attempts"),
               "note": "only the physical batch may fall back, and only on CUDA OOM"},
           "endpoint": run.get("endpoint"), "mem_cap_gib": run.get("mem_cap_gib"),
           "optimizer_steps_nominal": run.get("steps"),
           "selection_metric": run.get("selection_metric"),
           "frozen_knobs": ["imgsz", "nbs", "optimizer", "lr0", "epochs"],
           "augmentation_and_loss_source": "outputs/shuttle_capability/metrics/eth_only_v1_official_recipe.json",
           "spec_sections": ["11 resolved recipe", "9 schedule", "10 batch configuration"]}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    kw = out["resolved_kwargs"]
    print("resolved: imgsz=%s epochs=%s lr0=%s nbs=%s optimizer=%s freeze=%s save_period=%s seed=%s"
          % (kw.get("imgsz"), kw.get("epochs"), kw.get("lr0"), kw.get("nbs"), kw.get("optimizer"),
             kw.get("freeze"), kw.get("save_period"), kw.get("seed")))
    print("batch_policy: %s" % out["batch_policy"]["batch_attempts"])
    print("WROTE %s (%d B)" % (p, p.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
