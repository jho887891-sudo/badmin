# speed chain summary

historical reference: 15.90 ms | thresholds: useful<=12.72 strong<=10.00 excellent<=8.00
A3/A4 drift: {"A3_median_ms": 15.658396994695067, "A4_median_ms": 15.411651111207902, "drift_abs_ms": 0.2467, "drift_rel": 0.01576, "threshold_rel": 0.05, "warning": null}
overall_window_valid: True

## main table (VALID tracks only)

| Track | Backend | Precision | Median ms | p95 | p99 | FPS | d vs A3 | Validity |
|---|---|---|---:|---:|---:|---:|---:|---|
| A3 | pytorch | fp32 | 15.658 | 17.727 | 28.286 | 63.86 | +0.000 | VALID |
| B | pytorch | fp16 | 19.011 | 21.428 | 22.831 | 52.6 | +3.353 | VALID |
| C | pytorch | fp16 | 14.855 | 16.568 | 17.538 | 67.32 | -0.803 | VALID |
| D | tensorrt | fp16 | 2.126 | 2.225 | 2.417 | 470.47 | -13.533 | VALID |
| E | tensorrt | fp16 | 2.319 | 2.377 | 2.522 | 431.28 | -13.340 | VALID |
| A4 | pytorch | fp32 | 15.412 | 16.633 | 17.182 | 64.89 | -0.247 | VALID |

## audit table (contaminated / rejected - NOT part of the main table)

| Track | Backend | Precision | Median ms | Validity | Reason |
|---|---|---|---:|---|---|
| (none) | | | | | |
