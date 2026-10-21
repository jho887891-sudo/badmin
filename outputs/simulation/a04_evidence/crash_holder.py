import os, sys
sys.path.insert(0, r'E:\具身智能\badmin_wt_a04\tools\simulation')
from gpu_lock import GpuLock
lock = GpuLock(r'E:\具身智能\badmin_wt_a04\outputs\simulation\a04_evidence\isaac_gpu0.lock', run_id='A04-CRASH', marker='isaac')
lock.acquire()
print(lock.pid, flush=True)
os._exit(9)
