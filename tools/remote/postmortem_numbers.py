import csv, json
from collections import Counter
from pathlib import Path
def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))
pre = rows("outputs/shuttle_capability/metrics/tiny_representability_v1.csv")
post = rows("outputs/shuttle_capability/metrics/tiny_representability_v1_v3.csv")
key = lambda r: (r["image"], r["gt_index"])
pre_t = {key(r): r for r in pre if float(r["equiv_size_640"]) < 8.0}
post_t = {key(r): r for r in post if float(r["equiv_size_640"]) < 8.0}
assert set(pre_t) == set(post_t), "GT sets must match"
pre_hit = {k for k, r in pre_t.items() if r["matched_op"] == "True"}
post_hit = {k for k, r in post_t.items() if r["matched_op"] == "True"}
new = post_hit - pre_hit
lost = pre_hit - post_hit
weak_pre = {k for k, r in pre_t.items() if float(r["best_iou_weak"]) >= 0.1}
weak_post = {k for k, r in post_t.items() if float(r["best_iou_weak"]) >= 0.1}
print("tiny GT", len(pre_t), "| pre hits", len(pre_hit), "| post hits", len(post_hit), "| new", len(new), "| lost", len(lost))
print("new hits with a pre weak candidate:", sum(1 for k in new if k in weak_pre), "of", len(new))
print("weak candidates pre", len(weak_pre), "| post", len(weak_post), "| kept", len(weak_pre & weak_post),
      "| only-pre", len(weak_pre - weak_post), "| only-post", len(weak_post - weak_pre))
def stats(sel, field):
    xs = sorted(float(post_t[k][field]) for k in sel)
    return (xs[len(xs)//2] if xs else None)
sub = [k for k in pre_t if float(pre_t[k]["net_px_1024"]) < 8.0]
print("sub-stride GT (net<8px):", len(sub), "| pre hits", len(set(sub) & pre_hit), "| post hits", len(set(sub) & post_hit))
print("sub-stride median best_iou_weak pre %.4f -> post %.4f | max pre %.4f -> post %.4f"
      % (stats(sub, "best_iou_weak") or 0, 0, 0, 0))
for label, t in (("pre", pre_t), ("post", post_t)):
    xs = sorted(float(t[k]["best_iou_weak"]) for k in sub)
    med = xs[len(xs)//2] if xs else 0.0
    print("  %s sub-stride best_iou_weak: median %.4f, p90 %.4f, max %.4f, >=0.1 count %d"
          % (label, med, xs[int(0.9*(len(xs)-1))], xs[-1], sum(1 for x in xs if x >= 0.1)))
for bucket in ("<4", "4-6", "6-8"):
    sel = [k for k in pre_t if pre_t[k]["bucket"] == bucket]
    print("  bucket %-4s GT %-3d pre hits %-2d post hits %-2d | weak>=0.1 pre %-2d post %-2d"
          % (bucket, len(sel), len(set(sel) & pre_hit), len(set(sel) & post_hit),
             sum(1 for k in sel if k in weak_pre), sum(1 for k in sel if k in weak_post)))