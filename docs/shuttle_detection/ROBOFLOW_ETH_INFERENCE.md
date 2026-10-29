# ETH Roboflow Inference Utilities

These scripts run the Roboflow-hosted model `badminton-serve-tracker-dataset/22` against ETH shuttlecock images. They use the official Serverless Inference API V2 endpoint.

## Security and request behavior

- Set `ROBOFLOW_API_KEY` in the process environment before running a script. The scripts only check/read the variable at runtime; do not place the key in command arguments, source files, or output logs.
- Requests are sent to Roboflow and may incur usage charges.
- Each image is submitted once. The runners do not retry failed API calls; they stop on HTTP/API errors.
- Prediction JSON and visualizations are outputs. ETH images, API keys, response dumps, checkpoints, and other large data must not be committed.

## Scripts

- `probe.py IMAGE OUTPUT_DIR`: one-image connectivity check; writes a sanitized evidence JSON.
- `run_10.py IMAGE_DIR`: runs the connectivity extension over `sample_*.jpg`, recording class, confidence, bounding box, and whether segmentation points were returned.
- `prepare_batch100.py OUTPUT_ROOT`: deterministic location/time-stratified selector for the existing audited ETH inventory. It expects `instances.csv` in `OUTPUT_ROOT` and the inventory's source images at `/home/T7/dgut/robot_sim/eth_shuttle_detection`.
- `run_100.py BATCH_DIR`: checks every selected source image SHA256 against `manifest.csv`, makes one API request per image, and writes per-image predictions plus bbox visualizations and a comparison sheet. `BATCH_DIR` contains `manifest.csv` and an `images/` tree matching the selected source filenames.

## Example

Run the scripts on a machine that can access the ETH image source and has Pillow installed:

```bash
export ROBOFLOW_API_KEY='(set securely in your shell environment; do not paste it into this file)'
python tools/roboflow_eth_inference/probe.py /path/to/eth.jpg /path/to/probe-output
python tools/roboflow_eth_inference/run_10.py /path/to/ten-image-dir
python tools/roboflow_eth_inference/prepare_batch100.py /path/to/audit-output
python tools/roboflow_eth_inference/run_100.py /path/to/audit-output/batch100
```

The example placeholder is explanatory only; never replace it in this document with a real key. The 100-image runner writes `inference_100.json`, `inference_100.jsonl`, full-resolution bbox visualizations, and `comparison_sheet_100.png`. Segmentation is summarized by presence and point count in JSON; the visualization draws bounding boxes.

The October 2026 run evidence was kept outside GitHub. It recorded 111 successful HTTP 200 requests across the one-, ten-, and hundred-image stages. The 100-image batch used 100 unique ETH sources with matching SHA256 values. These historical results are evidence for that run, not a promise that a future API call will return identical predictions.
