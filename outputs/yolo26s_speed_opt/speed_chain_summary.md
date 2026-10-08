# speed chain summary

historical reference: 15.90 ms | thresholds: useful<=12.72 strong<=10.00 excellent<=8.00
A3/A4 drift: {"drift_abs_ms": null, "drift_rel": null, "warning": "A3_A4_INCOMPLETE"}
overall_window_valid: False

## main table (VALID tracks only)

| Track | Backend | Precision | Median ms | p95 | p99 | FPS | d vs A3 | Validity |
|---|---|---|---:|---:|---:|---:|---:|---|
| A3 | pytorch | fp32 | 15.658 | 17.727 | 28.286 | 63.86 | +0.000 | VALID |
| B | pytorch | fp16 | 19.011 | 21.428 | 22.831 | 52.6 | +3.353 | VALID |
| C | pytorch | fp16 | 14.855 | 16.568 | 17.538 | 67.32 | -0.803 | VALID |

## audit table (contaminated / rejected - NOT part of the main table)

| Track | Backend | Precision | Median ms | Validity | Reason |
|---|---|---|---:|---|---|
| (none) | | | | | |
