"""CPU-only audit regression: old 4x4 velocity block cannot inflate xy sigma."""
from pathlib import Path
import importlib.util
import json
import pytest

PATH = Path(__file__).resolve().parents[2] / "tools" / "vision" / "audit_r11_covariance.py"
spec = importlib.util.spec_from_file_location("audit_r11_covariance", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_xy_sigma_ignores_velocity_scale():
    cov = [[9, 0, 0, 0], [0, 4, 0, 0],
           [0, 0, 1000000, 0], [0, 0, 0, 1000000]]
    assert module.xy_sigma_from_state_covariance(cov) == pytest.approx(3)


def test_read_only_audit_counts_records_and_time_gap(tmp_path):
    cov = [[9, 0, 0, 0], [0, 4, 0, 0],
           [0, 0, 1000000, 0], [0, 0, 0, 1000000]]
    path = tmp_path / "tracks.jsonl"
    records = [
        {"timestamp_s": 0, "horizon": []},
        {"timestamp_s": 3.8, "horizon": [{
            "horizon_s": 0.3, "covariance_px2": cov, "sigma_major_px": 999.0,
        }]},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf8")
    result = module.audit(path)
    assert result["frame_count"] == 2
    assert result["max_frame_gap_s"] == 3.8
    assert result["horizons"]["0.3"]["n"] == 1
    assert result["horizons"]["0.3"]["position_sigma_median_px"] == pytest.approx(3)
    assert result["horizons"]["0.3"]["old_reported_sigma_median_px"] == 999.0


def test_non_psd_xy_rejected():
    with pytest.raises(ValueError):
        module.xy_sigma_from_state_covariance(
            [[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
