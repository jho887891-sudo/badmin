import os, sys, json
out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
except Exception as e:
    out["torch_error"] = repr(e)
try:
    import ultralytics
    out["ultralytics"] = ultralytics.__version__
    cfgdir = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "models", "26")
    out["cfg_dir"] = cfgdir
    out["cfg_files"] = sorted(os.listdir(cfgdir)) if os.path.isdir(cfgdir) else None
except Exception as e:
    out["ultralytics_error"] = repr(e)
for m in ("numpy", "cv2", "PIL", "yaml", "scipy"):
    try:
        mod = __import__(m)
        out[m] = getattr(mod, "__version__", "present")
    except Exception as e:
        out[m] = "NOT_INSTALLED"
print(json.dumps(out, indent=1))