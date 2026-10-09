"""Build the decision input for YOLO26S_V2_FULL_ETH_CONTINUATION_V1 from the frozen-evaluator outputs."""
import csv
import json
from pathlib import Path

EVAL = Path("_scratch_cont_v1/eval")
NEG = Path("_scratch_cont_v1/neg")   # the 496-image real no-target guard comes from eval_yolo26_v1 --split frozen_test
OUT = Path("outputs/shuttle_capability/metrics/yolo26s_v2_full_eth_cont_v1_measured.json")
CONT = "cont_v1_e20"
BASE = "eth_real_hardneg_v2_best"
POOL_PREFIX = ("real_images/backgrounds", "real_images/raw", "real_video/frames",
               "real_match_frames/images", "real_train/raw")
CAVEAT = ("legacy_eth_eval deliberately overlaps the continuation training pool "
          "(TRAIN_EVAL_OVERLAP_EXPECTED, spec section 8): it is an official-overlap diagnostic, not an "
          "unseen/held-out/generalization measurement")


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def one(tab, model, set_name):
    for r in tab:
        if r["model"] == model and r["set"] == set_name:
            return r
    raise SystemExit("missing row %s / %s" % (model, set_name))


def eval_block(tab, model):
    r = one(tab, model, "val|eth_unseen")
    return {"set": "legacy_eth_eval (val|eth_unseen, official overlap diagnostic)",
            "images": int(r["images"]), "gt": int(r["GT"]), "tp": int(r["TP"]), "fp": int(r["FP"]),
            "fn": int(r["FN"]), "precision": float(r["Precision"]), "recall": float(r["Recall"]),
            "f1": float(r["F1"]), "ap50": float(r["AP50"]), "map5095": float(r["mAP50-95"]),
            "recall_lt8": float(r["Recall_<8"]), "tp_lt8": int(r["TP_<8"]), "gt_lt8": int(r["GT_<8"]),
            "recall_6_8": float(r["Recall_6_8"]), "tp_6_8": int(r["TP_6_8"]), "gt_6_8": int(r["GT_6_8"]),
            "recall_lt4": float(r["Recall_<4"]), "gt_lt4": int(r["GT_<4"]), "tp_lt4": int(r["TP_<4"]),
            "recall_4_6": float(r["Recall_4_6"]), "gt_4_6": int(r["GT_4_6"]), "tp_4_6": int(r["TP_4_6"]),
            "caveat": CAVEAT}


def fp_block(neg, model):
    sel = [r for r in neg if r["checkpoint"] == model and r["threshold"] == "0.25"
           and r["set"].startswith(POOL_PREFIX)]
    fp = sum(int(r["total_FP"]) for r in sel)
    images = sum(int(r["images"]) for r in sel)
    return {"images": images, "fp": fp,
            "fp_per_image": (fp / images if images else None),
            "sets": {r["set"]: {"images": int(r["images"]), "fp": int(r["total_FP"]),
                                "images_with_fp": int(r["images_with_FP"])} for r in sel}}


def main():
    tab = rows(EVAL / "eth_vs_ours.csv")
    neg = rows(NEG / "frozen_test_negatives.csv")
    cont, base = eval_block(tab, CONT), eval_block(tab, BASE)
    cont_fp, base_fp = fp_block(neg, CONT), fp_block(neg, BASE)
    print("pool sets matched for the FP guard: %s" % sorted(cont_fp["sets"]))
    for name, blk, fp in (("V2", base, base_fp), ("cont_v1_e20", cont, cont_fp)):
        print("%-12s legacy_eth_eval R=%.4f mAP50-95=%.4f R<8=%.4f TP<8=%d/%d | no-target %d FP / %d imgs"
              % (name, blk["recall"], blk["map5095"], blk["recall_lt8"], blk["tp_lt8"], blk["gt_lt8"],
                 fp["fp"], fp["images"]))
    measured = {"model": {"name": CONT, "checkpoint": "_scratch_cont_v1/ckpt/e20_infer.pt",
                          "sha256": "f3935047062faf1c8990d01920463df0712bf5ad277b47cd08711122e257384b",
                          "epoch": 20, "note": "pre-registered primary endpoint, not best.pt"},
                "baseline_model": {"name": BASE,
                                   "sha256": "3c8339c6d16fc6e9808bd68c1f274ef48f3f5cb1d14e222b093e9871d69e9ff7"},
                "legacy_eth_eval": cont, "no_target": cont_fp,
                "baseline": {"legacy_eth_eval": base, "no_target": base_fp},
                "overlap": {"train_eval_overlap_expected": True, "caveat": CAVEAT}}
    OUT.write_text(json.dumps(measured, indent=2, sort_keys=True), encoding="utf-8")
    print("WROTE %s (%d B)" % (OUT, OUT.stat().st_size))


if __name__ == "__main__":
    main()
