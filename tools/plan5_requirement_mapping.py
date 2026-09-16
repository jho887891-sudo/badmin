"""Plan 5, the derivable half: requirement mapping as a FUNCTION of the unknown camera parameters.

Spec 05 needs minimum_required_target_px, a pinhole projection once fx and the working distance are known.
Neither is calibrated here (CameraIntrinsics is REQUIRES_CALIBRATION and docs/hardware_interface.md is a
placeholder), so the honest form of the answer is a mapping rather than a number.

THE CURVE CHANGED IN v2, and this matters more than the code. The first version of this file used the
numpy-rendered controlled matrix. That set measures renderer alignment rather than capability: the release
is trained toward photographic (Isaac Sim) appearance, and on an appearance-aligned ladder built from
renders the model has never seen it scores 0.85-1.00 from 64 px upward while a numpy-trained model scores
0.05-0.20 over the same range. The table below is therefore built from the APPEARANCE-ALIGNED curve, which
is the deployment figure.
"""

# v2 (baseline_os) on the appearance-aligned holdout ladder, 12 rungs, n=20 each (17 at 1024 px).
# The 95% half-width is +/-0.219 at n=20, so the SHAPE is solid and individual rungs are not tight.
APPEARANCE_CURVE = [
    (4.0, 0.050), (6.0, 0.250), (8.0, 0.250), (12.0, 0.450), (16.0, 0.300), (24.0, 0.400),
    (32.0, 0.450), (64.0, 0.850), (128.0, 0.900), (256.0, 0.900), (512.0, 0.900), (1024.0, 1.000),
]


def recall_at(px):
    """Interpolate the measured appearance-aligned curve.

    Below 4 px the frozen sets contain no detectable target in any appearance, and above 1024 px the
    ladder stops while real positives run to 1578 - so the top is bracketed by the last measured rung
    rather than invented.
    """
    if px < 4.0:
        return 0.0, "below the measured floor"
    if px >= APPEARANCE_CURVE[-1][0]:
        return APPEARANCE_CURVE[-1][1], "at or beyond the top measured rung"
    for i in range(len(APPEARANCE_CURVE) - 1):
        x0, y0 = APPEARANCE_CURVE[i]
        x1, y1 = APPEARANCE_CURVE[i + 1]
        if x0 <= px <= x1:
            return y0 + (y1 - y0) * (px - x0) / (x1 - x0), "interpolated"
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
print("=== measured recall at that size (v2, appearance-aligned curve) ===")
print("{:>8}".format("fx") + "".join("{:>11}".format("{:g}m".format(d)) for d in DISTANCES))
for fx in FX:
    row = ""
    for d in DISTANCES:
        r, _ = recall_at(fx * L_SKIRT / d)
        row += "{:>11.2f}".format(r)
    print("{:>8}".format(fx) + row)
print()
print("=== useful working distance band ===")
print("{:>8} {:>24} {:>24}".format("fx", "recall >= 0.85", "recall >= 0.50"))
for fx in FX:
    bands = []
    for floor in (0.85, 0.50):
        near = None
        far = None
        dd = 0.02
        while dd < 60.0:
            r, _ = recall_at(fx * L_SKIRT / dd)
            if r >= floor:
                if near is None:
                    near = dd
                far = dd
            dd += 0.01
        bands.append("{:.2f} - {:.2f} m".format(near, far) if near else "never")
    print("{:>8} {:>24} {:>24}".format(fx, bands[0], bands[1]))
print()
print("READ THE FAR BOUND, NOT THE NEAR ONE. The near bound prints as 0.02 m because the loop starts")
print("there and the curve is flat-topped at 1.000 above the last measured rung, so it is an artefact with")
print("no physical meaning for a racket robot. What the mapping actually constrains is how FAR the camera")
print("may be: the appearance-aligned ladder reaches only 0.45 at 32 px and 0.25 at 8 px, and that is what")
print("sets the far end of each band.")
print()
print("The bands are also much NARROWER than the first version of this file reported, and that is the")
print("point rather than a regression. The first version used the numpy-rendered controlled curve, which")
print("measures renderer alignment: it said 0.90 at 32 px where the deployment-aligned appearance gives")
print("0.45. For fx=1000 the earlier mapping promised 0.12-5.37 m at recall >= 0.5; this one says 1.71 m.")
print("The narrower number is the one measured in the appearance the release was trained for.")
print()
print("More SMALL-target training data is what would move the far end outward now, exactly as the")
print("near-field data return moved it before.")

print("CAVEAT, and it is the reason this file exists rather than a number: n=20 per rung gives a 95%")
print("half-width of +/-0.219, so read the SHAPE and not the individual rungs. And the curve is measured")
print("on synthetic renders in the trained appearance, not on real photographs, of which the release has")
print("6 distinct images.")