#!/usr/bin/env python3
"""Analyse the frozen-checkpoint resolution-response probe JSON (remote A6000 run).

Input : outputs/shuttle_capability/metrics/resolution_response_remote_v1_probe.json
        (the raw probe output, copied verbatim from the A6000 run; produced by tools/remote/remote_tiny_resolution.py)
        tools/remote/eth_tiny_part.csv                (the 20-GT part list + local V2@1024 references)
Output: outputs/shuttle_capability/metrics/resolution_response_remote_v1_gt.csv
        outputs/shuttle_capability/metrics/resolution_response_remote_v1_summary.csv
Nothing here trains or modifies weights; it is a pure post-processing of measured detections.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

JSON_PATH = Path("outputs/shuttle_capability/metrics/resolution_response_remote_v1_probe.json")
PART_PATH = Path("tools/remote/eth_tiny_part.csv")
OUT_DIR = Path("outputs/shuttle_capability/metrics")
CONF_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
IOU_OPS = [0.3, 0.5]


def matched_at(pairs, conf, iou_op):
    """pairs = [[conf, iou], ...]; True if any candidate reaches conf and iou_op."""
    return any(c >= conf and v >= iou_op for c, v in pairs)


def op_breakdown(rs):
    """Operating-point accounting: confident candidates vs the ones that actually hit their GT."""
    hits = sum(1 for r in rs if r["hit_iou0.5_conf0.25"])
    op_total = sum(r["n_candidates_op"] for r in rs)
    return {"op_confident_candidates": op_total, "op_tp": hits, "op_unmatched": op_total - hits,
            "gt_locmiss_iou_0.3_0.5": sum(1 for r in rs if not r["hit_iou0.5_conf0.25"]
                                          and 0.3 <= r["best_iou"] < 0.5),
            "gt_best_iou_lt_0.3": sum(1 for r in rs if r["best_iou"] < 0.3),
            "images_with_no_confident_candidate": sum(1 for r in rs if r["n_candidates_op"] == 0)}


def main() -> int:
    rep = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    part = {r["basename"]: r for r in csv.DictReader(open(PART_PATH, encoding="utf-8"))}
    res_list = sorted(int(k) for k in rep["resolutions"])
    print("weights sha256 : %s" % rep["weights_sha256"])
    print("env            : %s" % json.dumps(rep["env"]))
    print("GT             : %d tiny over %d images (%d part rows, missing=%s)"
          % (sum(1 for r in rep["resolutions"][str(res_list[0])]["gt"] if r["tiny"]),
             rep["n_images"], rep["n_part_rows"], rep["missing"]))

    gt_rows = []
    for R in res_list:
        block = rep["resolutions"][str(R)]
        pic = block["per_image_iou_conf"]
        for g in block["gt"]:
            pairs = None
            for e in pic.get(g["image"], []):
                if e["gt_index"] == g["gt_index"]:
                    pairs = e["pairs"]
                    break
            assert pairs is not None, (g["image"], g["gt_index"])
            meta = part.get(g["image"], {})
            row = {"imgsz": R, "image": g["image"], "gt_index": g["gt_index"],
                   "rel_dir": meta.get("rel_path", "").split("/images/")[0],
                   "bucket": g["bucket"], "eq640": round(g["eq640"], 6),
                   "net_px": round(1.6 * g["eq640"] * R / 1024.0, 3),
                   "n_candidates": g["n_dets"], "n_candidates_op": g["n_dets_op"],
                   "n_dets_iou30": g["n_dets_iou30"],
                   "best_iou": round(g["best_iou_weak"], 6),
                   "best_conf_at_best_iou": (round(g["best_conf_at_best_iou"], 6)
                                             if g["best_conf_at_best_iou"] else ""),
                   "local_v2_matched_op_1024": meta.get("v2_matched_op_1024", ""),
                   "local_v2_best_iou_1024": meta.get("v2_best_iou_weak_1024", ""),
                   "local_v2_best_conf_1024": meta.get("v2_best_conf_weak_1024", "")}
            for op in IOU_OPS:
                for t in CONF_GRID:
                    row["hit_iou%s_conf%s" % (op, t)] = matched_at(pairs, t, op)
            gt_rows.append(row)

    # sanity: per-(image,gt) candidate count must equal the number of stored pairs
    for R in res_list:
        block = rep["resolutions"][str(R)]
        for g in block["gt"]:
            pairs = [e["pairs"] for e in block["per_image_iou_conf"][g["image"]]
                     if e["gt_index"] == g["gt_index"]][0]
            assert len(pairs) == g["n_dets"], (R, g["image"], len(pairs), g["n_dets"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_csv = OUT_DIR / "resolution_response_remote_v1_gt.csv"
    with gt_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gt_rows[0].keys()))
        w.writeheader()
        for r in gt_rows:
            w.writerow(r)

    # ---------------- summary ----------------
    print("\n== response by resolution (operation IoU 0.5, 20 GT) ==")
    hdr = "imgsz  cand  cand>=.25 | " + " ".join("%5s" % ("c>=%g" % t) for t in CONF_GRID) + " |  IoU0.3: " + \
          " ".join("%5s" % ("c>=%g" % t) for t in CONF_GRID)
    print(hdr)
    summary = []
    for R in res_list:
        rs = [r for r in gt_rows if r["imgsz"] == R]
        block = rep["resolutions"][str(R)]
        hits = [sum(1 for r in rs if r["hit_iou0.5_conf%s" % t]) for t in CONF_GRID]
        hits3 = [sum(1 for r in rs if r["hit_iou0.3_conf%s" % t]) for t in CONF_GRID]
        print("%5d %5d %9d | " % (R, block["n_candidates"], block["n_candidates_op"]) +
              " ".join("%5d" % h for h in hits) + " |        " + " ".join("%5d" % h for h in hits3))
        lat = block["latency_ms"]
        inf = block["latency_speed_median_ms"]["inference"]
        pre = block["latency_speed_median_ms"]["preprocess"]
        post = block["latency_speed_median_ms"]["postprocess"]
        best_ious = [r["best_iou"] for r in rs]
        loc = [r for r in rs if r["best_iou"] >= 0.5]
        near = [r for r in rs if r["best_iou"] >= 0.3]
        zero = [r for r in rs if r["best_iou"] < 0.01]
        summary.append({
            "imgsz": R, "n_gt": len(rs), "n_candidates": block["n_candidates"],
            "n_candidates_op": block["n_candidates_op"],
            **{"hit_iou0.5_conf%s" % t: h for t, h in zip(CONF_GRID, hits)},
            **{"hit_iou0.3_conf%s" % t: h for t, h in zip(CONF_GRID, hits3)},
            "n_best_iou_ge_0.5": len(loc), "n_best_iou_ge_0.3": len(near),
            "n_best_iou_lt_0.01": len(zero),
            "median_best_iou": round(statistics.median(best_ious), 6),
            "max_best_iou": round(max(best_ious), 6),
            "latency_e2e_median_ms": round(lat["median"], 3),
            "latency_e2e_mean_ms": round(lat["mean"], 3),
            "speed_inference_median_ms": round(inf, 3),
            "speed_preprocess_median_ms": round(pre, 3),
            "speed_postprocess_median_ms": round(post, 3),
            "img_per_s": round(block["img_per_s"], 3),
            "peak_vram_mib": round(block["peak_vram_mib"], 1),
            "net_input_px": block["net_input_px"][0],
            **op_breakdown(rs),
        })
        print("      best_iou: median=%.4f max=%.4f  >=0.5: %d  >=0.3: %d  <0.01: %d"
              % (statistics.median(best_ious), max(best_ious), len(loc), len(near), len(zero)))
        print("      latency : e2e median=%.2f ms | preprocess=%.2f inference=%.2f postprocess=%.2f ms | %.2f img/s | peak %.0f MiB"
              % (lat["median"], pre, inf, post, block["img_per_s"], block["peak_vram_mib"]))

    sum_csv = OUT_DIR / "resolution_response_remote_v1_summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in summary:
            w.writerow(r)

    # ---------------- bucket detail ----------------
    print("\n== by bucket (operation IoU 0.5) ==")
    for b in ("<4", "4-6", "6-8"):
        rs0 = [r for r in gt_rows if r["bucket"] == b and r["imgsz"] == res_list[0]]
        if not rs0:
            print("  %-4s n=0" % b)
            continue
        line = []
        for R in res_list:
            rs = [r for r in gt_rows if r["bucket"] == b and r["imgsz"] == R]
            line.append("R=%d hit@.25=%d/%d hit@.01=%d/%d medBestIoU=%.3f" % (
                R, sum(1 for r in rs if r["hit_iou0.5_conf0.25"]), len(rs),
                sum(1 for r in rs if r["hit_iou0.5_conf0.01"]), len(rs),
                statistics.median([r["best_iou"] for r in rs])))
        print("  %-4s n=%d: %s" % (b, len(rs0), " | ".join(line)))

    # ---------------- operating-point breakdown (conf >= 0.25) ----------------
    print("\n== operating-point breakdown (conf>=0.25, IoU 0.5) ==")
    for s in summary:
        print("  R=%4d  confident candidates(>=.25)=%3d  TP=%2d  unmatched(>=.25)=%3d  |  "
              "GT with no hit: localization .3-.5=%d, best_iou<.3=%d  |  images with 0 confident cand=%d"
              % (s["imgsz"], s["op_confident_candidates"], s["op_tp"], s["op_unmatched"],
                 s["gt_locmiss_iou_0.3_0.5"], s["gt_best_iou_lt_0.3"],
                 s["images_with_no_confident_candidate"]))

    # ---------------- triage: is the object "not looked at" or "looked at, no response" ----------------
    print("\n== unresponsive GT (best_iou < 0.01): did its image contain any candidate elsewhere? ==")
    for R in res_list:
        rs = [r for r in gt_rows if r["imgsz"] == R]
        dead = [r for r in rs if r["best_iou"] < 0.01]
        withcand = [r for r in dead if r["n_candidates"] > 0]
        print("  R=%4d dead=%2d/%d  image_had_other_candidates=%2d  image_had_none=%2d  "
              "median n_cand(dead)=%.1f max=%d"
              % (R, len(dead), len(rs), len(withcand), len(dead) - len(withcand),
                 statistics.median([r["n_candidates"] for r in dead]) if dead else 0.0,
                 max([r["n_candidates"] for r in dead]) if dead else 0))
    print("\n== candidate counts per resolution (conf>=0.01) ==")
    for R in res_list:
        rs = [r for r in gt_rows if r["imgsz"] == R]
        n = [r["n_candidates"] for r in rs]
        print("  R=%4d total=%3d mean=%.2f max=%2d images_with_no_candidate=%d  (op>=.25 total=%d)"
              % (R, sum(n), statistics.fmean(n), max(n), sum(1 for x in n if x == 0),
                 sum(r["n_candidates_op"] for r in rs)))

    # ---------------- per-GT table ----------------
    print("\n== per-GT best IoU / best conf (local V2@1024 -> remote 1024/1280/1536) ==")
    print("%-32s %-4s %6s %7s | %7s %6s | %s" % ("image", "bkt", "eq640", "netpx1024", "locIoU", "locCnf",
                                             "  ".join("R%-4d iou/conf" % R for R in res_list)))
    for b in sorted({r["basename"] for r in part.values()}):
        pass
    by_img = {}
    for r in gt_rows:
        by_img.setdefault(r["image"], {})[r["imgsz"]] = r
    for img in sorted(by_img, key=lambda p: (by_img[p][res_list[0]]["bucket"], p)):
        row0 = by_img[img][res_list[0]]
        cells = []
        for R in res_list:
            r = by_img[img][R]
            cells.append("%6.3f/%-6s" % (r["best_iou"], ("%.3f" % r["best_conf_at_best_iou"])
                                         if r["best_conf_at_best_iou"] != "" else "-"))
        print("%-32s %-4s %6.2f %7.1f | %7s %6s | %s" % (
            img, row0["bucket"], row0["eq640"], 1.6 * row0["eq640"], row0["local_v2_best_iou_1024"],
            row0["local_v2_best_conf_1024"] or "-", "  ".join(cells)))

    # ---------------- cross-machine consistency ----------------
    print("\n== local V2@1024 vs remote V2@1024 (per-GT) ==")
    n_iou_ok = n_conf_low = n_conf_hi = 0
    devs = []
    for img, d in by_img.items():
        r = d[1024]
        if r["local_v2_best_iou_1024"] == "":
            continue
        dl = abs(r["best_iou"] - float(r["local_v2_best_iou_1024"]))
        devs.append(dl)
        n_iou_ok += dl <= 0.02
        if r["local_v2_best_conf_1024"]:
            dc = r["best_conf_at_best_iou"] - float(r["local_v2_best_conf_1024"])
            if dc < 0:
                n_conf_low += 1
            else:
                n_conf_hi += 1
    print("best-IoU agreement within 0.02: %d/%d (max dev %.4f)"
          % (n_iou_ok, len(devs), max(devs) if devs else 0.0))
    print("best-conf shift remote-local: lower=%d higher=%d" % (n_conf_low, n_conf_hi))
    print("matched@0.25/0.5  local=%d  remote=%d"
          % (sum(1 for d in by_img.values() if d[1024]["local_v2_matched_op_1024"] == "True"),
             sum(1 for d in by_img.values() if d[1024]["hit_iou0.5_conf0.25"])))

    print("\nWROTE %s (%d B)" % (gt_csv, gt_csv.stat().st_size))
    print("WROTE %s (%d B)" % (sum_csv, sum_csv.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
