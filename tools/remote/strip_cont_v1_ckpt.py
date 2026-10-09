#!/usr/bin/env python3
"""Strip an ultralytics training checkpoint to inference-only weights (spec sections 13 / 26).

last.pt carries optimizer state and EMA (~80 MB); evaluation only needs the model (~20 MB), which also makes the
transfer over the slow link 4x cheaper. The full checkpoint is left untouched.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--provenance", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args([] if argv is None else argv)
    import torch
    src = Path(a.src)
    dst = Path(a.dst)
    if not src.is_file():
        print("STOP: source checkpoint missing: %s" % src)
        return 2
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        print("STOP: unexpected checkpoint structure")
        return 3
    # ultralytics checkpoints keep the weights in 'ema' and leave 'model' as None; strip_optimizer does exactly
    # this promotion. Without it the stripped file is a few KB and silently useless (caught by the dry run).
    model = ck.get("model") or ck.get("ema")
    if model is None:
        print("STOP: both 'model' and 'ema' are empty in %s" % src)
        return 4
    out = {"model": model, "ema": None, "optimizer": None,
           "epoch": ck.get("epoch"), "best_fitness": ck.get("best_fitness"),
           "date": ck.get("date"), "version": ck.get("version"), "train_args": ck.get("train_args", {}),
           "stripped_from": str(src),
           "stripped_keys_kept": ["model", "epoch", "best_fitness", "date", "version", "train_args"]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(dst))
    tensors = len(model.state_dict()) if hasattr(model, "state_dict") else 0
    if tensors == 0 or dst.stat().st_size < (5 << 20):
        print("STOP: stripped checkpoint looks empty (tensors=%d bytes=%d)" % (tensors, dst.stat().st_size))
        return 5
    rec = {"label": a.label, "source": {"path": str(src), "bytes": src.stat().st_size, "sha256": sha256(src)},
           "stripped": {"path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256(dst),
                        "tensors": tensors, "epoch": ck.get("epoch"), "best_fitness": ck.get("best_fitness")},
           "note": "inference-only weights stripped from the training checkpoint; optimizer/EMA removed"}
    if a.provenance:
        Path(a.provenance).write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(rec, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
