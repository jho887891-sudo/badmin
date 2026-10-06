#!/usr/bin/env python3
"""Assemble the V2 smoke evidence artifact from the launcher manifest plus the run log (task 4, step 3)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def peak_vram_gib(log_text: str):
    values = [float(x) for x in re.findall(r"([0-9]+\.[0-9]+)G", log_text)]
    return max(values) if values else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--expect-train-rows", type=int, default=15255)
    ap.add_argument("--expect-val-rows", type=int, default=2920)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    kwargs = m["resolved_kwargs"]
    frozen_expect = {"imgsz": 1024, "optimizer": "AdamW", "lr0": 1e-4, "nbs": 32, "freeze": 0, "seed": 42,
                     "batch": 8}
    checks = {
        "batch_8_first_try": m.get("batch") == 8 and m.get("batch_attempts_tried", [{}])[0].get("outcome") == "ok",
        "frozen_recipe_unchanged": all(kwargs.get(k) == v for k, v in frozen_expect.items()),
        "weights_verified": m["weights"]["sha256"] ==
                            "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
        "dataset_lists_frozen": m["data"]["lists"]["train"]["images"] == int(a.expect_train_rows)
                                and m["data"]["lists"]["val"]["images"] == int(a.expect_val_rows),
        "validation_completed": int(m.get("selection", {}).get("epochs_recorded") or 0) == 3,
        "finite_losses": True,          # asserted below from results.csv when provided
        "single_class": m["data"]["names"] == ["shuttlecock"] and m["data"]["nc"] == 1,
        "diagnostic_only": m.get("diagnostic_only") is True and m.get("eligible_for_final_report") is False,
    }
    epochs = []
    if a.results and Path(a.results).is_file():
        import csv
        rows = list(csv.DictReader(Path(a.results).open(encoding="utf-8")))
        epochs = [int(float(r["epoch"])) for r in rows]
        for key in ("train/box_loss", "train/cls_loss", "metrics/mAP50-95(B)"):
            for r in rows:
                try:
                    v = float(str(r.get(key, "")).strip())
                except ValueError:
                    checks["finite_losses"] = False
                    continue
                if v != v or v in (float("inf"), float("-inf")):
                    checks["finite_losses"] = False
    out = {
        "experiment": m["experiment"],
        "purpose": "3-epoch diagnostic smoke of the frozen V2 dataset; never a final result",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "epochs_recorded": epochs,
        "contract": m["contract"],
        "weights": m["weights"],
        "data": m["data"],
        "resolved_kwargs": kwargs,
        "batch": m.get("batch"),
        "batch_attempts_tried": m.get("batch_attempts_tried"),
        "gpu_cap": m.get("gpu_cap"),
        "peak_vram_gib": peak_vram_gib(Path(a.log).read_text(encoding="utf-8", errors="replace"))
                         if a.log and Path(a.log).is_file() else None,
        "selection": m.get("selection"),
        "best_checkpoint": m.get("best_checkpoint"),
        "diagnostic_reasons": m.get("diagnostic_reasons"),
        "third_party_loggers": m.get("third_party_loggers"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[smoke] status=%s peak_vram_gib=%s -> %s" % (out["status"], out["peak_vram_gib"], a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
