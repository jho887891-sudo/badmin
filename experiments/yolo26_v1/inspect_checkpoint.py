import json, torch, os
A = "/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_imgsz1024_b16_w8_20260926-083331"
p = os.path.join(A, "weights", "best.pt")
ck = torch.load(p, map_location="cpu", weights_only=False)
out = {
  "best_pt": p,
  "bytes": os.path.getsize(p),
  "epoch": int(ck.get("epoch", -1)),
  "best_fitness": float(ck.get("best_fitness")) if ck.get("best_fitness") is not None else None,
  "train_metrics": ck.get("train_metrics"),
  "date": ck.get("date"),
  "version": ck.get("version"),
  "train_args_epochs": (ck.get("train_args") or {}).get("epochs"),
  "train_args_lr0": (ck.get("train_args") or {}).get("lr0"),
  "train_args_freeze": (ck.get("train_args") or {}).get("freeze"),
  "train_args_batch": (ck.get("train_args") or {}).get("batch"),
  "train_args_imgsz": (ck.get("train_args") or {}).get("imgsz"),
}
print(json.dumps(out, indent=1, default=str))
last = torch.load(os.path.join(A, "weights", "last.pt"), map_location="cpu", weights_only=False)
print("last_epoch:", int(last.get("epoch", -1)))