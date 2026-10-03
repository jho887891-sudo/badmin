"""Tests for tools/train_eth_only_v1.py (Task 3, no GPU / no torch needed)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import train_eth_only_v1 as T  # noqa: E402

REAL_CONTRACT = REPO / "configs" / "eth_only_yolo26s_1024_v1.yaml"
REAL_RECIPE = REPO / "outputs" / "shuttle_capability" / "metrics" / "eth_only_v1_official_recipe.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def workspace(tmp_path):
    """A miniature but structurally complete run workspace: contract + weights + dataset yaml + list files."""
    weights = tmp_path / "yolo26s.pt"
    weights.write_bytes(b"fake-weights-" + b"0" * 64)
    contract = {
        "experiment": "eth_only_yolo26s_1024_v1",
        "spec": "spec.md",
        "model": "yolo26s.pt",
        "model_sha256": _sha(weights.read_bytes()),
        "model_bytes": weights.stat().st_size,
        "imgsz": 1024, "freeze": 0, "optimizer": "AdamW", "lr0": 1.0e-4, "epochs": 50,
        "nbs": 32, "batch": 8, "batch_fallback": [8, 6, 4],
        "never_touch": ["imgsz", "nbs", "optimizer", "lr0"], "seed": 42,
        "selection_metric": "mAP50-95",
        "evaluation_only_sets": ["val|eth_unseen", "external_real_only", "controlled_capability", "challenge_test"],
        "smoke_epochs": 3,
    }
    cpath = tmp_path / "contract.yaml"
    cpath.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    listdir = tmp_path / "lists"
    listdir.mkdir()
    (listdir / "train.txt").write_text("\n".join("a%d.jpg" % i for i in range(5)) + "\n", encoding="utf-8")
    (listdir / "val.txt").write_text("\n".join("v%d.jpg" % i for i in range(2)) + "\n", encoding="utf-8")
    dyaml = tmp_path / "ds.yaml"
    dyaml.write_text(yaml.safe_dump({"path": ".", "train": "lists/train.txt", "val": "lists/val.txt",
                                     "nc": 1, "names": {0: "shuttlecock"}}, sort_keys=False), encoding="utf-8")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "official_imgsz": 1024, "official_epochs": 50, "official_optimizer": "adamw", "official_lr": 1e-4,
        "official_momentum": 0.9, "official_weight_decay": 5e-4, "official_seed": 42,
        "official_loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
        "official_augmentations": {"mosaic": 1.0, "mixup": 0.7, "scale": 0.5, "fliplr": 0.5, "hsv_h": 0.015,
                                   "degrees": 0.0, "shear": 0.0, "perspective": 0.0, "copy_paste": 0.0,
                                   "cutmix": 0.0, "flipud": 0.0, "deterministic": True},
    }), encoding="utf-8")
    return {"contract": cpath, "weights": weights, "data_yaml": dyaml, "recipe": recipe, "dir": tmp_path,
            "contract_data": contract}


def _argv(ws, out_manifest, *extra):
    return ["--contract", str(ws["contract"]), "--recipe-json", str(ws["recipe"]),
            "--data-yaml", str(ws["data_yaml"]), "--weights", str(ws["weights"]),
            "--project", str(ws["dir"] / "runs"), "--name", "run1",
            "--out-manifest", str(out_manifest), *extra]


# --- contract ---

def test_real_contract_is_well_formed_and_pins_the_frozen_recipe():
    data = T.load_contract(REAL_CONTRACT)
    assert data["experiment"] == "eth_only_yolo26s_1024_v1"
    assert data["imgsz"] == 1024 and data["nbs"] == 32 and data["optimizer"] == "AdamW"
    assert float(data["lr0"]) == 1e-4 and data["freeze"] == 0 and data["epochs"] == 50
    assert data["batch"] == 8 and list(data["batch_fallback"]) == [8, 6, 4]
    assert list(data["never_touch"]) == ["imgsz", "nbs", "optimizer", "lr0"]
    assert data["model_bytes"] == 20422725
    assert data["model_sha256"] == "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"


def test_real_recipe_json_exists_and_has_the_plan_overrides():
    recipe = json.loads(REAL_RECIPE.read_text(encoding="utf-8"))
    assert recipe["official_lr"] == 1e-4 and recipe["official_optimizer"] == "adamw"
    assert recipe["plan_overrides_vs_official"]["batch"] == {"official": 32, "plan": 8}


def test_load_contract_rejects_missing_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"experiment": "x"}), encoding="utf-8")
    with pytest.raises(T.ContractError, match="misses required keys"):
        T.load_contract(p)


def test_load_contract_rejects_altered_never_touch(tmp_path, workspace):
    data = dict(workspace["contract_data"])
    data["never_touch"] = ["imgsz", "nbs", "optimizer"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(T.ContractError, match="never_touch"):
        T.load_contract(p)


# --- recipe resolution ---

def test_resolve_recipe_uses_contract_for_frozen_knobs(workspace):
    contract = T.load_contract(workspace["contract"])
    recipe = json.loads(workspace["recipe"].read_text(encoding="utf-8"))
    kw = T.resolve_recipe(contract, recipe)
    assert kw["imgsz"] == 1024 and kw["nbs"] == 32 and kw["freeze"] == 0
    assert kw["optimizer"] == "AdamW" and float(kw["lr0"]) == 1e-4
    assert kw["epochs"] == 50 and kw["seed"] == 42
    assert kw["momentum"] == 0.9 and kw["weight_decay"] == 5e-4
    assert kw["box"] == 7.5 and kw["cls"] == 0.5 and kw["dfl"] == 1.5
    assert kw["mixup"] == 0.7 and kw["mosaic"] == 1.0 and kw["deterministic"] is True


@pytest.mark.parametrize("knob,bad", [("imgsz", 640), ("nbs", 64), ("optimizer", "SGD"), ("lr0", 1e-3)])
def test_resolve_recipe_refuses_frozen_knob_changes(workspace, knob, bad):
    contract = T.load_contract(workspace["contract"])
    with pytest.raises(T.RecipeViolationError, match=knob):
        T.resolve_recipe(contract, None, deviating={knob: bad})


def test_resolve_recipe_allows_epochs_and_batch_changes(workspace):
    contract = T.load_contract(workspace["contract"])
    kw = T.resolve_recipe(contract, None, epochs=3, deviating={"batch": 4})
    assert kw["epochs"] == 3 and kw["batch"] == 4 and kw["imgsz"] == 1024


def test_contract_wins_when_the_recipe_json_drifts_on_a_frozen_knob(workspace):
    """Precedence: the frozen contract is authoritative; a drifting recipe json cannot move lr0/optimizer."""
    contract = T.load_contract(workspace["contract"])
    kw = T.resolve_recipe(contract, {"official_lr": 5e-4, "official_optimizer": "sgd"})
    assert float(kw["lr0"]) == 1e-4 and kw["optimizer"] == "AdamW"


# --- batch fallback ---

def test_plan_batch_attempts_default_is_8_6_4():
    assert T.plan_batch_attempts(8, [8, 6, 4]) == [8, 6, 4]


def test_plan_batch_attempts_never_grows_the_batch():
    assert T.plan_batch_attempts(4, [8, 6, 4]) == [4]
    assert T.plan_batch_attempts(6, [8, 6, 4]) == [6, 4]


def test_plan_batch_attempts_rejects_non_positive():
    with pytest.raises(T.RecipeViolationError):
        T.plan_batch_attempts(0, [8, 6, 4])


@pytest.mark.parametrize("exc,expected", [
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
    (RuntimeError("torch.cuda.OutOfMemoryError: ..."), True),
    (MemoryError("out of memory"), True),
    (RuntimeError("CUDA error: device-side assert triggered"), False),
    (ValueError("dataset yaml not found"), False),
])
def test_is_oom(exc, expected):
    assert T.is_oom(exc) is expected


def test_run_with_fallback_advances_only_on_oom():
    calls = []

    def train_fn(batch):
        calls.append(batch)
        if batch == 8:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        return {"ok": batch}

    batch, result, tried = T.run_with_fallback([8, 6, 4], train_fn)
    assert (batch, result) == (6, {"ok": 6}) and calls == [8, 6]
    assert [t["outcome"] for t in tried] == ["oom", "ok"]


def test_run_with_fallback_propagates_non_oom_errors():
    def train_fn(batch):
        raise ValueError("label cache corrupt")

    with pytest.raises(ValueError, match="label cache corrupt"):
        T.run_with_fallback([8, 6, 4], train_fn)


def test_run_with_fallback_reports_total_oom():
    def train_fn(batch):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(T.TrainingFailed, match=r"\[8, 6, 4\]"):
        T.run_with_fallback([8, 6, 4], train_fn)


# --- weights + dataset ---

def test_verify_weights_accepts_matching_file(workspace):
    contract = T.load_contract(workspace["contract"])
    info = T.verify_weights(workspace["weights"], contract["model_sha256"], contract["model_bytes"])
    assert info["bytes"] == workspace["weights"].stat().st_size
    assert info["sha256"] == contract["model_sha256"]


def test_verify_weights_rejects_wrong_sha(workspace):
    with pytest.raises(T.WeightsVerificationError, match="sha256 mismatch"):
        T.verify_weights(workspace["weights"], "0" * 64, workspace["weights"].stat().st_size)


def test_verify_weights_rejects_wrong_size(workspace):
    with pytest.raises(T.WeightsVerificationError, match="size mismatch"):
        T.verify_weights(workspace["weights"], _sha(workspace["weights"].read_bytes()), 1)


def test_verify_weights_rejects_missing_file(tmp_path):
    with pytest.raises(T.WeightsVerificationError, match="not found"):
        T.verify_weights(tmp_path / "nope.pt", "0" * 64, 1)


def test_check_dataset_yaml_accepts_single_class_and_counts_images(workspace):
    info = T.check_dataset_yaml(workspace["data_yaml"])
    assert info["nc"] == 1 and info["names"] == ["shuttlecock"]
    assert info["lists"]["train"]["images"] == 5 and info["lists"]["val"]["images"] == 2
    assert len(info["sha256"]) == 64


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update({"nc": 2}), "nc: 1"),
    (lambda d: d.update({"names": {0: "bird"}}), "class 0"),
    (lambda d: d.pop("val"), "no 'val' list"),
])
def test_check_dataset_yaml_rejects_bad_contract(tmp_path, mutate, match):
    (tmp_path / "train.txt").write_text("a.jpg\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("v.jpg\n", encoding="utf-8")
    base = {"path": ".", "train": "train.txt", "val": "val.txt", "nc": 1, "names": {0: "shuttlecock"}}
    mutate(base)
    p = tmp_path / "ds.yaml"
    p.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    with pytest.raises(T.ContractError, match=match):
        T.check_dataset_yaml(p)


def test_check_dataset_yaml_rejects_empty_list(tmp_path):
    (tmp_path / "train.txt").write_text("", encoding="utf-8")
    (tmp_path / "val.txt").write_text("v.jpg\n", encoding="utf-8")
    p = tmp_path / "ds.yaml"
    p.write_text(yaml.safe_dump({"path": ".", "train": "train.txt", "val": "val.txt", "nc": 1,
                                 "names": {0: "shuttlecock"}}, sort_keys=False), encoding="utf-8")
    with pytest.raises(T.ContractError, match="empty"):
        T.check_dataset_yaml(p)


# --- selection ---

def _results_csv(path, rows):
    header = ["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
              "metrics/mAP50-95(B)"]
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(v) for v in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_select_best_epoch_uses_internal_val_map50_95(tmp_path):
    csv_path = _results_csv(tmp_path / "results.csv", [
        (1, 0.5, 0.40, 0.30, 0.11), (2, 0.6, 0.55, 0.44, 0.19), (3, 0.4, 0.20, 0.25, 0.07)])
    best = T.select_best_epoch(csv_path)
    assert best["epoch"] == 2 and best["value"] == 0.19 and best["recall"] == 0.55
    assert best["selection_scope"] == "internal_validation_only"
    assert best["metric"] == T.SELECTION_METRIC


def test_select_best_epoch_requires_the_metric_column(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("epoch,metrics/mAP50(B)\n1,0.4\n", encoding="utf-8")
    with pytest.raises(T.ContractError, match="mAP50-95"):
        T.select_best_epoch(p)


def test_select_best_epoch_rejects_empty_results(tmp_path):
    p = tmp_path / "results.csv"
    p.write_text("epoch,metrics/mAP50-95(B)\n", encoding="utf-8")
    with pytest.raises(T.ContractError, match="no epoch rows"):
        T.select_best_epoch(p)


def test_resolve_save_dir_prefers_the_trainer_directory(tmp_path):
    from types import SimpleNamespace
    model = SimpleNamespace(trainer=SimpleNamespace(save_dir=tmp_path / "runs" / "detect" / "p" / "n"))
    assert T.resolve_save_dir("p", "n", model) == tmp_path / "runs" / "detect" / "p" / "n"


def test_resolve_save_dir_falls_back_to_project_name(tmp_path):
    assert T.resolve_save_dir(tmp_path / "p", "n", None) == tmp_path / "p" / "n"
    from types import SimpleNamespace
    assert T.resolve_save_dir(tmp_path / "p", "n", SimpleNamespace(trainer=None)) == tmp_path / "p" / "n"


def test_build_train_args_pins_wandb_off_and_the_frozen_recipe(workspace, tmp_path):
    from types import SimpleNamespace
    a = SimpleNamespace(data_yaml="d.yaml", project="p", name="n", device="0", workers=8, cache=False)
    d = T.build_train_args(a, 8, {"imgsz": 1024, "nbs": 32, "epochs": 3}, weights="y.pt")
    assert d["wandb"] is False, "wandb must stay off: it would upload artifacts of an offline experiment"
    assert d["batch"] == 8 and d["imgsz"] == 1024 and d["nbs"] == 32 and d["epochs"] == 3
    assert d["exist_ok"] is False and d["pretrained"] is True and d["val"] is True
    assert d["weight"] == "y.pt" and d["data"] == "d.yaml"


# --- manifest ---

def test_shortened_run_is_never_final_eligible():
    m = T.build_manifest({"experiment": "x", "smoke": False, "dry_run": False,
                          "resolved_kwargs": {"epochs": 1}, "contract_epochs": 50})
    assert m["diagnostic_only"] is True and m["eligible_for_final_report"] is False
    assert m["diagnostic_reasons"] == ["epochs 1 != contract 50"]


def test_full_length_run_is_final_eligible():
    m = T.build_manifest({"experiment": "x", "smoke": False, "dry_run": False,
                          "resolved_kwargs": {"epochs": 50}, "contract_epochs": 50})
    assert m["diagnostic_only"] is False and m["diagnostic_reasons"] == []


def test_build_manifest_marks_smoke_as_diagnostic():
    m = T.build_manifest({"experiment": "x", "smoke": True})
    assert m["diagnostic_only"] is True and m["eligible_for_final_report"] is False
    assert m["never_used_for_selection"] == list(T.EVALUATION_ONLY_SPLITS)


def test_build_manifest_marks_full_run_as_final_eligible():
    m = T.build_manifest({"experiment": "x", "smoke": False, "dry_run": False})
    assert m["diagnostic_only"] is False and m["eligible_for_final_report"] is True


def test_write_manifest_refuses_to_overwrite(tmp_path):
    p = tmp_path / "m.json"
    T.write_manifest(p, {"a": 1})
    with pytest.raises(T.ContractError, match="refusing to overwrite"):
        T.write_manifest(p, {"a": 2})


# --- CLI ---

def test_dry_run_writes_a_frozen_recipe_manifest(workspace, capsys):
    out = workspace["dir"] / "manifests" / "dry.json"
    code = T.run_cli(_argv(workspace, out, "--dry-run"))
    assert code == 0
    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["resolved_kwargs"]["imgsz"] == 1024 and man["resolved_kwargs"]["nbs"] == 32
    assert man["resolved_kwargs"]["optimizer"] == "AdamW" and man["resolved_kwargs"]["epochs"] == 50
    assert man["resolved_kwargs"]["mixup"] == 0.7
    assert man["batch_attempts"] == [8, 6, 4] and man["data"]["lists"]["train"]["images"] == 5
    assert man["dry_run"] is True and man["diagnostic_only"] is True
    assert "dry-run manifest" in capsys.readouterr().out


def test_dry_run_smoke_forces_three_epochs_and_stays_diagnostic(workspace):
    out = workspace["dir"] / "manifests" / "smoke.json"
    assert T.run_cli(_argv(workspace, out, "--dry-run", "--smoke")) == 0
    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["resolved_kwargs"]["epochs"] == 3 and man["smoke"] is True
    assert man["eligible_for_final_report"] is False


def test_contract_is_the_only_authority_for_imgsz(workspace):
    """resolve_recipe has no imgsz override path: the value always comes from the contract (freeze)."""
    out = workspace["dir"] / "manifests" / "authority.json"
    base = _argv(workspace, out, "--dry-run")
    broke = dict(workspace["contract_data"])
    broke["imgsz"] = 640
    p = workspace["dir"] / "contract640.yaml"
    p.write_text(yaml.safe_dump(broke, sort_keys=False), encoding="utf-8")
    argv = [x if x != str(workspace["contract"]) else str(p) for x in base]
    assert T.run_cli(argv) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["resolved_kwargs"]["imgsz"] == 640


# --- gpu cap ---

class _FakeProps:
    def __init__(self, total):
        self.total_memory = total


class _FakeCuda:
    def __init__(self, total):
        self.total = total
        self.calls = []

    def get_device_properties(self, index):
        self.calls.append(("props", index))
        return _FakeProps(self.total)

    def set_per_process_memory_fraction(self, fraction, index):
        self.calls.append(("set", fraction, index))


class _FakeTorch:
    def __init__(self, total):
        self.cuda = _FakeCuda(total)


def test_gpu_cap_fraction_matches_the_24gb_contract():
    total = 49_140 * 1024 ** 2  # the A6000's reported total
    frac = T.gpu_cap_fraction(24, total)
    assert abs(frac - 24 * 1024 / 49_140) < 1e-9 and 0.49 < frac < 0.51
    assert T.gpu_cap_fraction(None, total) == 1.0
    assert T.gpu_cap_fraction(100, total) == 1.0


def test_cap_gpu_memory_sets_the_fraction_for_the_device():
    fake = _FakeTorch(49_140 * 1024 ** 2)
    info = T.cap_gpu_memory(24, 0, fake)
    assert info["applied"] is True and info["device_total_gib"] == round(49_140 / 1024, 2)
    assert ("set", info["fraction"], 0) in fake.cuda.calls


def test_cap_gpu_memory_can_be_disabled():
    fake = _FakeTorch(49_140 * 1024 ** 2)
    info = T.cap_gpu_memory(None, 0, fake)
    assert info["applied"] is False and fake.cuda.calls == []


def test_run_cli_reports_contract_errors_as_exit_code_2(workspace, capsys):
    base = _argv(workspace, workspace["dir"] / "m.json", "--dry-run")
    base[base.index("--weights") + 1] = str(workspace["dir"] / "missing.pt")
    old = sys.argv
    try:
        sys.argv = ["train_eth_only_v1.py"] + base
        assert T.main() == 2
    finally:
        sys.argv = old
    assert "WeightsVerificationError" in capsys.readouterr().err
