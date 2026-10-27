# ETH Stage 1 | BBox-only pixel-exact Copy-Paste

Independent entry point: tools/generate_eth_stage1_bbox_preview.py

### Contract

* Use the existing verified ETH TRAIN inventory, expected 13,992 boxes by default.
* Crop source RGB pixels inside the existing bbox; paste onto non-white solid 1024 x 1024 background.
* **No Mask, no GrabCut, no alpha matting, no resize, no blending.** The small amount of original background inside the bbox remains. The entire crop including those pixels is bit-for-bit preserved.
* Images are created outside the source dataset. No training or model weights are touched.
* This is a separate preview and data generation path; it does not bypass the existing formal-training gates.

### Example: 100 images, then human QA

~~~bash
python tools/generate_eth_stage1_bbox_preview.py \
  --inventory /home/dgut/eth_audit/out/inventory_20261026T074352Z/instances.csv \
  --output-dir /home/dgut/eth_audit/out/stage1_bbox_preview_$(date +%Y%m%dT%H%M%S) \
  --mode preview --count 100 --seed 42 \
  --time-bins 10 --width 1024 --height 1024
~~~

Outputs: images/000001.png ... 000100.png, labels/, contact_sheet.png,
manifest.csv, audit.json, SHA256SUMS. Overview uses nearest-neighbor
**review-only** zooms; the 100 exported PNGs contain no target scaling.

### Full 13,992 unique source coverage

~~~bash
python tools/generate_eth_stage1_bbox_preview.py \
  --inventory /path/to/verified/instances.csv \
  --output-dir /NEW/full-output-dir \
  --mode full --seed 42
~~~

The full mode uses each instance once, which is NOT equal scene/time exposure.

To maintain all unique instances and equalize exposure across observed
scene x time cells via recorded repetitions:

~~~bash
python tools/generate_eth_stage1_bbox_preview.py \
  --inventory /path/to/verified/instances.csv \
  --output-dir /NEW/balanced-output-dir \
  --mode balanced-full --seed 42 --time-bins 10 \
  --max-outputs 100000
~~~

Balanced-full fails if any expected scene/time cell is empty or if requested
image count exceeds the explicit max-outputs budget; it does not silently omit
cells. This mode may generate more than 13,992 images. Review space before run.

### Source validation, time semantics, and safety

SHA256 checks verify both image and label bytes. Any mismatch aborts before
output creation; existing output folders are rejected. The input inventory
must contain only train / eth_main / eth_train instances and the six declared
locations. Per-image original and output bbox and hashes are recorded.

Within each location + set_dir, final numeric filename token is treated as a
frame index and binned into equal frame-index ranges. This is a proxy for
capture progress, **NOT a verified physical-time or uniform FPS claim**.
The report includes actual exposure counts for every observed scene/time bin.
An empty time bin blocks balanced-full. Previews balance as far as discrete
unique samples allow. Appearance quality is UNKNOWN before human inspection.

Run CPU tests with:

~~~bash
python -m unittest -v tests/test_eth_stage1_bbox_preview.py
~~~

Do not claim tests or dry-run samples establish real-data success. Check
contact_sheet.png, suspicious bbox rectangular artifacts, output label bounds,
and audit.json after executing on the remote ETH mount.

Never commit generated PNGs, datasets, weights or server-local outputs to Git.
Do not change main or the four-stage official Mask-reviewed training path.
