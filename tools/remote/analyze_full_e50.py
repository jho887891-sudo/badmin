import csv
rows = list(csv.DictReader(open("_scratch_eth_only_v1/results.csv", encoding="utf-8")))
def g(r, k):
    for f in r:
        if f.strip() == k: return float(r[f])
    raise KeyError(k)
data = []
for r in rows:
    e = int(float(r["epoch"]))
    m50 = g(r, "metrics/mAP50(B)"); m = g(r, "metrics/mAP50-95(B)"); rec = g(r, "metrics/recall(B)")
    data.append((e, m50, m, rec, 0.1*m50 + 0.9*m))
best_map = max(data, key=lambda t: t[2])
best_fit = max(data, key=lambda t: t[4])
print("epochs", len(data), "range", data[0][0], "-", data[-1][0])
print("argmax mAP50-95      -> epoch %d  mAP50-95 %.5f  mAP50 %.5f  R %.5f  fitness %.5f" % best_map)
print("argmax fitness       -> epoch %d  mAP50-95 %.5f  mAP50 %.5f  R %.5f  fitness %.5f" % best_fit)
print("same checkpoint?", best_map[0] == best_fit[0])
top = sorted(data, key=lambda t: -t[2])[:6]
print("top6 by mAP50-95:", [(e, round(m,5)) for e,_,m,_,_ in top])
print("epoch 50 row      :", [round(x,5) for x in data[-1][1:]])
print("first 5 epochs    :", [(e, round(m,5)) for e,_,m,_,_ in data[:5]])
print("val curve every 5 :", [(e, round(m,5)) for e,_,m,_,_ in data if e % 5 == 0])