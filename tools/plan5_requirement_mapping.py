"""Plan 5, the derivable half: requirement mapping as a FUNCTION of the unknown camera parameters.

Spec 05 needs minimum_required_target_px, which is a pinhole projection once fx and the working
distance are known. Neither is calibrated in this repository (CameraIntrinsics is REQUIRES_CALIBRATION
and docs/hardware_interface.md is a placeholder), so the honest form of the answer is a mapping rather
than a number: for each (fx, working distance) pair, what apparent size results, and what the measured
recall curve gives at that size.
"""
import math

S8 = [("<4", 2.0, 0.400), ("4-6", 5.0, 0.400), ("6-8", 7.0, 0.400),
      ("8-12", 10.0, 0.425), ("12-16", 14.0, 0.625), ("16-24", 20.0, 0.825),
      ("24-32", 28.0, 0.900), (">32", 40.0, 0.900)]
S7 = [(64.0, 1.000), (96.0, 1.000), (128.0, 1.000), (192.0, 1.000), (256.0, 1.000),
      (384.0, 1.000), (512.0, 1.000), (768.0, 0.000), (1023.0, 0.000)]

def recall_at(px):
    if px < 4.0:
        return 0.0, "below the measured floor"
    if px > 512.0:
        return 0.0, "beyond the trained ceiling"
    pts = [(4.0, 0.400), (5.0, 0.400), (7.0, 0.400), (10.0, 0.425), (14.0, 0.625),
           (20.0, 0.825), (28.0, 0.900), (40.0, 0.900), (64.0, 1.000), (96.0, 1.000),
           (128.0, 1.000), (192.0, 1.000), (256.0, 1.000), (384.0, 1.000), (512.0, 1.000)]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= px <= x1:
            return y0 + (y1 - y0) * (px - x0) / (x1 - x0), "measured"
    return 0.0, "outside the measured range"

L_SKIRT = 0.0618
L_LEN = 0.0779
DISTANCES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
FX = [500, 800, 1000, 1500, 2000, 3000]

print("Physical extents: skirt diameter {:.4f} m, overall length {:.4f} m".format(L_SKIRT, L_LEN))
print("Pinhole projection: px = fx * extent / distance")
print()
print("=== apparent equivalent size in px (skirt diameter) ===")
print("{:>8}".format("fx") + "".join("{:>10}".format("{:g}m".format(d)) for d in DISTANCES))
for fx in FX:
    print("{:>8}".format(fx) + "".join("{:>10.1f}".format(fx * L_SKIRT / d) for d in DISTANCES))
print()
print("=== measured recall at that size (baseline_nearfield, marginalised curve) ===")
print("{:>8}".format("fx") + "".join("{:>11}".format("{:g}m".format(d)) for d in DISTANCES))
for fx in FX:
    row = ""
    for d in DISTANCES:
        r, _why = recall_at(fx * L_SKIRT / d)
        row += "{:>11.2f}".format(r)
    print("{:>8}".format(fx) + row)
print()
print("=== the constraint is a BAND, not a far limit: the useful working distance range ===")
print("Too close and the target exceeds the 512 px trained ceiling; too far and it falls under 40 px.")
print()
print("{:>8} {:>22} {:>22} {:>22}".format("fx", "D at recall>=0.9", "D at recall>=0.5", "band width (>=0.5)"))
for fx in FX:
    near_hi = fx * L_SKIRT / 512.0   # closer than this, the target exceeds the trained ceiling
    near_lo = fx * L_SKIRT / 40.0    # closer than this, the target is too large to be well covered
    far90 = None
    far50 = None
    dd = near_lo
    while dd < 60.0:
        r, _why = recall_at(fx * L_SKIRT / dd)
        if r < 0.9 and far90 is None:
            far90 = dd
        if r < 0.5 and far50 is None:
            far50 = dd
        dd += 0.01
    a = "{:.2f} - {:.2f} m".format(near_hi, far90) if far90 else "n/a"
    b = "{:.2f} - {:.2f} m".format(near_hi, far50) if far50 else "n/a"
    w = "{:.2f} m".format((far50 - near_hi)) if far50 else "n/a"
    print("{:>8} {:>22} {:>22} {:>22}".format(fx, a, b, w))
print()
print("D at recall>=0.9 is the band where measured recall stays at or above 0.9;")
print("D at recall>=0.5 is the wider band where it stays at or above 0.5.")
print("Both bands are bounded BELOW by the 512 px trained ceiling, which is a property of the")
print("training data, not of the camera - more large-target training would move it outwards.")