#!/usr/bin/env bash
# Chain: smoke -> gate -> full run. The full run starts only if the smoke manifest says PASS.
set -euo pipefail
cd "$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
bash run_smoke_tiny.sh tiny_smoke_e3
"$PY" - <<'PYEOF'
import json, sys
m = json.load(open("/home/dgut/.dsh-bench/tiny_smoke_e3_manifest.json"))
k = m["resolved_kwargs"]
ok = ((m.get("batch") == 8) and k["imgsz"] == 1024 and k["optimizer"] == "AdamW" and float(k["lr0"]) == 1e-4
      and int(k["nbs"]) == 32 and int(k["freeze"]) == 0 and int(k["seed"]) == 42
      and m.get("diagnostic_only") is True and (m.get("selection") or {}).get("epochs_recorded") == 3)
print("[chain] smoke gate:", "PASS" if ok else "FAIL", {kk: k[kk] for kk in ("imgsz","epochs","optimizer","lr0","nbs","freeze","seed")},
      "batch", m.get("batch"), "epochs_recorded", (m.get("selection") or {}).get("epochs_recorded"))
sys.exit(0 if ok else 9)
PYEOF
echo "[chain] smoke gate passed, starting the full run"
bash run_full_tiny.sh tiny_full_e50
