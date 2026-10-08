#!/usr/bin/env python3
"""Structural tests for tools/remote/run_speed_opt_chain.sh.

The chain cannot be executed here (it needs the A6000), so these tests pin its contract by reading
the script: track order, one gate per track, output names, and the forbidden operations.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "remote" / "run_speed_opt_chain.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")
LINES = [ln.strip() for ln in TEXT.splitlines()]


def body_lines():
    return [ln for ln in LINES if ln and not ln.startswith("#")]


def test_script_exists_and_is_executable_style():
    assert SCRIPT.exists()
    assert TEXT.startswith("#!/usr/bin/env bash")
    assert "set -u" in TEXT


def test_track_order_is_exactly_as_specified():
    body = "\n".join(body_lines())
    order = [m.group(1) for m in re.finditer(r"^gate (A3|B|C|D|E|A4);", body, re.M)]
    assert order == ["A3", "B", "C", "D", "E", "A4"]


def test_every_track_has_its_own_gate_on_the_same_line():
    for track in ("A3", "B", "C", "D", "E", "A4"):
        pat = re.compile(r"^gate " + track + r";\s+run_(pytorch|tensorrt) " + track + r"\b", re.M)
        assert pat.search("\n".join(body_lines())), ("no gate immediately before " + track)


def test_gate_is_not_called_only_once_at_the_top():
    assert TEXT.count("gate A3;") == 1
    assert TEXT.count("gate A4;") == 1
    assert TEXT.count("    gate() {") + TEXT.count("gate() {") == 1


def test_output_filenames_match_the_spec():
    for name in ("speed_pytorch_fp32_A3.json", "speed_pytorch_fp16_clean.json",
                 "speed_torch_compile_fp16_clean.json", "speed_tensorrt_fp16_static.json",
                 "speed_tensorrt_fp16_dynamic.json", "speed_pytorch_fp32_A4.json"):
        assert name in TEXT, name


def test_monitor_csv_per_track():
    assert "speed_${tag}.monitor.csv" in TEXT
    body = "\n".join(body_lines())
    for track in ("A3", "B", "C", "D", "E", "A4"):
        assert re.search(r"run_(pytorch|tensorrt) " + track + r"\b", body), track


def test_analyzer_is_invoked_at_the_end():
    assert "speed_chain_analyze.py" in TEXT
    assert TEXT.index("gate A4;") < TEXT.index("speed_chain_analyze.py")


def test_track_f_is_not_run():
    assert "run_tensorrt F" not in TEXT
    assert "nms=False" not in TEXT
    assert "Track F" in TEXT  # documented as deliberately skipped


def test_forbidden_operations_absent():
    body = "\n".join(body_lines())
    for bad in ("pkill", "kill -9", "killall", "nvidia-smi -lgc", "nvidia-smi --lock-gpu-clocks",
                "conf=", "iou=", "max_det", "--imgsz 640", "imgsz=640"):
        assert bad not in body, ("forbidden token present in executable line: " + bad)


def test_uniform_timing_protocol():
    assert "WARMUP=${WARMUP:-50}" in TEXT
    assert "ITERS=${ITERS:-500}" in TEXT
    # run_pytorch and run_tensorrt each pass the same protocol exactly once
    assert TEXT.count("--warmup") == 2
    assert TEXT.count("--iters") == 2


def test_failures_stop_the_chain_and_keep_artifacts():
    assert "chain stops" in TEXT
    # gate(A3) 20, gate(21), pytorch 30, export 31, tensorrt 32
    assert TEXT.count("exit 2") + TEXT.count("exit 3") >= 5


def test_checkpoint_default_is_absolute_and_guarded():
    # must be absolute: the chain cd-s into RUN_DIR, so a bare filename does not resolve
    assert "CKPT=${CKPT:-$HOME/.dsh-bench/eth_only_v1_best.pt}" in TEXT
    assert "FATAL checkpoint not found" in TEXT

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
