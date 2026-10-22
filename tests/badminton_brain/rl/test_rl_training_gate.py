# -*- coding: utf-8 -*-
"""P8-05 tests: the training gate refuses, counts and persists -- and it never trains.

Every refusal path has its own test; the budget tests use an INJECTED clock and therefore
never sleep; the checkpoint tests exercise atomicity and hash/config mismatch refusal.
No optimiser, no rollout, no torch, no GPU: this file only asserts refusals and bookkeeping.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.seed_ledger import SeedLedger  # noqa: E402
from badminton_brain.rl.training_gate import (  # noqa: E402
    AUTHORIZATION_FLAG, REQUIRED_STOP_CONDITIONS, REQUIRED_UPSTREAM_GATES, STOP_REASONS,
    AuthorizationMissingError, BudgetAccountant, BudgetExhaustedError,
    BudgetExceedsRegisteredError, BudgetRequest, CheckpointConfigMismatchError,
    CheckpointIntegrityError, CheckpointMetadata, PreRegistration,
    PreRegistrationIncompleteError, PreRegistrationIntegrityError,
    PreRegistrationMissingError, SeedLedgerMismatchError, StopCriteria, StopReasonError,
    StopRecord, TrainingGate, UpstreamGateNotSatisfiedError, authorize_training_run,
    read_checkpoint, resume_checkpoint, write_checkpoint,
)

LEDGER = SeedLedger.derive(task_family="racket_contact_targeting", master_seed=20261022,
                           counts={"train": 8, "val": 4, "test": 4})
LEDGER_HASH = LEDGER.ledger_hash()
PROTOCOL_HASH = "a" * 64
CONFIG_HASH = "b" * 64
GRANT = {"granted_by": "user-written-authorization", AUTHORIZATION_FLAG: True}
ALL_PASS = {"P5": "DONE", "P6": "PASS", "P7": "DONE"}


class FakeClock:
    """A clock the test advances by hand: no sleeping, no wall-clock flakiness."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def registration(**overrides) -> PreRegistration:
    """A COMPLETE, PINNED pre-registration (rounded through JSON, so the pin is real)."""
    document = {
        "version": "p8-training-gate-v0.1",
        "algorithm": "PPO",
        "hyperparameters": {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
                            "clip_range": 0.2, "entropy_coef": 0.01, "n_steps": 2048,
                            "batch_size": 64, "n_epochs": 10},
        "seed_ledger_hash": LEDGER_HASH,
        "episode_budget": 100,
        "wall_clock_budget_s": 60.0,
        "gpu_budget_s": 30.0,
        "success_metric": "success_rate",
        "acceptance": {"threshold": 0.6, "comparator": ">="},
        "stop_conditions": {"max_steps": 10, "patience_evaluations": 3,
                            "plateau_window": 4, "min_delta": 0.01},
        "evaluation_protocol_hash": PROTOCOL_HASH,
    }
    document.update(overrides)
    base = PreRegistration(
        version=document["version"], algorithm=document["algorithm"],
        hyperparameters=document["hyperparameters"],
        seed_ledger_hash=document["seed_ledger_hash"],
        episode_budget=document["episode_budget"],
        wall_clock_budget_s=document["wall_clock_budget_s"],
        gpu_budget_s=document["gpu_budget_s"],
        success_metric=document["success_metric"],
        acceptance=document["acceptance"],
        stop_conditions=document["stop_conditions"],
        evaluation_protocol_hash=document["evaluation_protocol_hash"])
    return PreRegistration.from_dict(base.to_dict())


def gate_with(reg: PreRegistration) -> TrainingGate:
    return TrainingGate(registration=reg)


def check(gate: TrainingGate, *, request: BudgetRequest = None, ledger_hash: str = LEDGER_HASH,
          upstream=None, authorization=None):
    return gate.check(request=request if request is not None else BudgetRequest(),
                      seed_ledger_hash=ledger_hash,
                      upstream_gates=ALL_PASS if upstream is None else upstream,
                      authorization=GRANT if authorization is None else authorization)


# --------------------------------------------------------------------------------------
# 1/2/3: the pre-registration must exist, be complete and be pinned
# --------------------------------------------------------------------------------------
def test_a_run_without_a_pre_registration_is_refused():
    with pytest.raises(PreRegistrationMissingError):
        check(TrainingGate(registration=None))


def test_a_missing_pre_registration_file_is_refused_not_created(tmp_path):
    gate = TrainingGate.from_file(tmp_path / "does_not_exist.json")
    with pytest.raises(PreRegistrationMissingError):
        check(gate)
    assert not (tmp_path / "does_not_exist.json").exists()


@pytest.mark.parametrize("field", sorted(REQUIRED_STOP_CONDITIONS))
def test_a_missing_stop_condition_is_incomplete(field):
    conditions = {"max_steps": 10, "patience_evaluations": 3, "plateau_window": 4,
                  "min_delta": 0.01}
    conditions.pop(field)
    record = PreRegistration.from_dict(registration().to_dict())
    document = record.to_dict()
    document["stop_conditions"] = conditions
    document.pop("registration_hash")
    with pytest.raises(PreRegistrationIncompleteError):
        PreRegistration.from_dict({**document, "registration_hash": "0" * 64})


def test_a_missing_hyperparameter_is_incomplete():
    with pytest.raises(PreRegistrationIncompleteError) as excinfo:
        registration(hyperparameters={"learning_rate": 3e-4}).validate()
    assert "hyperparameters missing" in str(excinfo.value)


@pytest.mark.parametrize("overrides", [
    {"episode_budget": 0},
    {"episode_budget": 1.5},
    {"wall_clock_budget_s": 0.0},
    {"gpu_budget_s": -1.0},
    {"algorithm": "A2C"},
    {"success_metric": ""},
    {"seed_ledger_hash": "not-a-sha256"},
    {"evaluation_protocol_hash": "A" * 64},
    {"acceptance": {"threshold": 1.5, "comparator": ">="}},
    {"acceptance": {"threshold": 0.5, "comparator": "~"}},
])
def test_invalid_pre_registration_values_are_refused(overrides):
    with pytest.raises(PreRegistrationIncompleteError):
        registration(**overrides).validate()


def test_an_edited_pre_registration_fails_its_hash_pin():
    record = PreRegistration.from_dict(registration().to_dict())
    tampered = PreRegistration(
        algorithm=record.algorithm, hyperparameters=record.hyperparameters,
        seed_ledger_hash=record.seed_ledger_hash, episode_budget=999,
        wall_clock_budget_s=record.wall_clock_budget_s, gpu_budget_s=record.gpu_budget_s,
        success_metric=record.success_metric, acceptance=record.acceptance,
        stop_conditions=record.stop_conditions,
        evaluation_protocol_hash=record.evaluation_protocol_hash,
        pinned_hash=record.pinned_hash)
    with pytest.raises(PreRegistrationIntegrityError):
        tampered.verify_pin()
    with pytest.raises(PreRegistrationIntegrityError):
        check(gate_with(tampered))


def test_an_unpinned_pre_registration_object_is_refused():
    unpinned = PreRegistration(
        algorithm="PPO", hyperparameters=registration().hyperparameters,
        seed_ledger_hash=LEDGER_HASH, episode_budget=100, wall_clock_budget_s=60.0,
        gpu_budget_s=30.0, success_metric="success_rate",
        acceptance={"threshold": 0.6, "comparator": ">="},
        stop_conditions={"max_steps": 10, "patience_evaluations": 3, "plateau_window": 4,
                         "min_delta": 0.01},
        evaluation_protocol_hash=PROTOCOL_HASH)
    with pytest.raises(PreRegistrationIntegrityError):
        check(gate_with(unpinned))


def test_a_hand_edited_registration_file_is_refused_on_load(tmp_path):
    path = registration().write(tmp_path / "pre_registration.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["episode_budget"] = 100000
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PreRegistrationIntegrityError):
        TrainingGate.from_file(path)


def test_a_literal_we_will_tune_it_later_document_is_incomplete(tmp_path):
    path = tmp_path / "pre_registration.json"
    path.write_text(json.dumps({"algorithm": "PPO", "note": "TBD"}), encoding="utf-8")
    with pytest.raises(PreRegistrationIncompleteError) as excinfo:
        TrainingGate.from_file(path)
    assert "missing field" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# 4: the authorization flag must be explicit
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("authorization", [
    {},
    {"granted_by": "user"},
    {AUTHORIZATION_FLAG: "true", "granted_by": "user"},
    {AUTHORIZATION_FLAG: 1, "granted_by": "user"},
    {AUTHORIZATION_FLAG: False, "granted_by": "user"},
    {AUTHORIZATION_FLAG: True},
    {AUTHORIZATION_FLAG: True, "granted_by": "   "},
    "yes",
])
def test_missing_or_implicit_authorization_is_refused(authorization):
    with pytest.raises(AuthorizationMissingError):
        check(gate_with(registration()), authorization=authorization)


# --------------------------------------------------------------------------------------
# 5: upstream P5/P6/P7 must be DONE/PASS
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("upstream,expected", [
    ({"P6": "DONE", "P7": "DONE"}, "P5"),
    ({"P5": "IN_PROGRESS", "P6": "DONE", "P7": "DONE"}, "P5"),
    ({"P5": "DONE", "P6": "NOT_RUN", "P7": "DONE"}, "P6"),
    ({"P5": "DONE", "P6": "PASS", "P7": None}, "P7"),
    ({}, "P5"),
])
def test_upstream_gate_not_passing_is_refused_and_named(upstream, expected):
    with pytest.raises(UpstreamGateNotSatisfiedError) as excinfo:
        check(gate_with(registration()), upstream=upstream)
    assert expected in excinfo.value.unsatisfied
    assert all(gate_id in REQUIRED_UPSTREAM_GATES for gate_id in excinfo.value.unsatisfied)


def test_lower_case_pass_status_is_accepted_after_normalisation():
    decision = check(gate_with(registration()),
                     upstream={"P5": "done", "P6": " pass ", "P7": "DONE"})
    assert decision.upstream_gate_status["P6"] == " pass "


# --------------------------------------------------------------------------------------
# 6/7: budget and seed ledger
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("request_budget,dimension", [
    (BudgetRequest(episodes=101), "episode"),
    (BudgetRequest(wall_clock_s=60.1), "wall_clock"),
    (BudgetRequest(gpu_s=30.5), "gpu"),
    (BudgetRequest(steps=11), "steps"),
])
def test_requesting_more_than_the_registered_budget_is_refused(request_budget, dimension):
    with pytest.raises(BudgetExceedsRegisteredError) as excinfo:
        check(gate_with(registration()), request=request_budget)
    assert excinfo.value.dimension == dimension


def test_the_registered_budget_itself_is_allowed():
    decision = check(gate_with(registration()),
                     request=BudgetRequest(episodes=100, wall_clock_s=60.0, gpu_s=30.0,
                                           steps=10))
    assert decision.episode_budget == 100


def test_a_different_seed_ledger_is_refused():
    other = SeedLedger.derive(task_family="racket_contact_targeting", master_seed=7,
                              counts={"train": 8, "val": 4, "test": 4}).ledger_hash()
    with pytest.raises(SeedLedgerMismatchError):
        check(gate_with(registration()), ledger_hash=other)


# --------------------------------------------------------------------------------------
# the happy path (only reachable because this test writes its own registration)
# --------------------------------------------------------------------------------------
def test_a_complete_registration_with_authorization_passes(tmp_path):
    path = registration().write(tmp_path / "pre_registration.json")
    gate = TrainingGate.from_file(path)
    decision = check(gate, request=BudgetRequest(episodes=10, steps=2, gpu_s=1.0,
                                                 wall_clock_s=5.0))
    assert decision.algorithm == "PPO"
    assert decision.registration_hash == gate.registration.registration_hash()
    assert decision.seed_ledger_hash == LEDGER_HASH
    assert decision.evaluation_protocol_hash == PROTOCOL_HASH
    assert decision.authorization_flag is True
    assert set(decision.to_dict()["upstream_gate_status"]) == set(REQUIRED_UPSTREAM_GATES)


# --------------------------------------------------------------------------------------
# budget accountant (injected clock, never sleeps)
# --------------------------------------------------------------------------------------
def test_wall_clock_budget_refuses_the_next_step_without_sleeping():
    clock = FakeClock()
    accountant = BudgetAccountant(registration(), clock=clock)
    assert accountant.admit_step(episodes=1).step_index == 0
    clock.advance(30.0)
    assert accountant.admit_step(episodes=1).step_index == 1
    clock.advance(30.0)
    with pytest.raises(BudgetExhaustedError) as excinfo:
        accountant.admit_step(episodes=1)
    assert excinfo.value.dimension == "wall_clock"
    assert excinfo.value.stop_reason == "BUDGET_EXHAUSTED"
    assert accountant.stop_reason == "BUDGET_EXHAUSTED"


def test_the_look_ahead_step_cost_is_refused_before_the_step_runs():
    clock = FakeClock()
    accountant = BudgetAccountant(registration(), clock=clock)
    clock.advance(59.0)
    with pytest.raises(BudgetExhaustedError) as excinfo:
        accountant.admit_step(episodes=1, step_wall_s=2.0)
    assert excinfo.value.dimension == "wall_clock"
    assert accountant.snapshot().steps_used == 0


def test_gpu_budget_is_enforced():
    accountant = BudgetAccountant(registration(), clock=FakeClock())
    accountant.admit_step(episodes=1, gpu_s=20.0)
    with pytest.raises(BudgetExhaustedError) as excinfo:
        accountant.admit_step(episodes=1, gpu_s=20.0)
    assert excinfo.value.dimension == "gpu"
    assert accountant.snapshot().gpu_remaining_s == pytest.approx(10.0)


def test_episode_budget_is_enforced():
    accountant = BudgetAccountant(registration(), clock=FakeClock())
    accountant.admit_step(episodes=99)
    with pytest.raises(BudgetExhaustedError) as excinfo:
        accountant.admit_step(episodes=2)
    assert excinfo.value.dimension == "episode"


def test_step_budget_is_enforced():
    accountant = BudgetAccountant(registration(), clock=FakeClock())
    for _ in range(10):
        accountant.admit_step(episodes=1)
    with pytest.raises(BudgetExhaustedError) as excinfo:
        accountant.admit_step(episodes=1)
    assert excinfo.value.dimension == "max_steps"


def test_a_refusal_latches_and_keeps_refusing():
    clock = FakeClock()
    accountant = BudgetAccountant(registration(), clock=clock)
    clock.advance(120.0)
    with pytest.raises(BudgetExhaustedError):
        accountant.admit_step(episodes=1)
    clock.now = 0.0
    with pytest.raises(BudgetExhaustedError) as excinfo:
        accountant.admit_step(episodes=1)
    assert excinfo.value.dimension == "latched"
    assert accountant.snapshot().exhausted is True


def test_snapshot_reports_remaining_budget():
    clock = FakeClock()
    accountant = BudgetAccountant(registration(), clock=clock)
    accountant.admit_step(episodes=5, gpu_s=5.0)
    clock.advance(10.0)
    snapshot = accountant.snapshot()
    assert snapshot.steps_remaining == 9
    assert snapshot.episodes_remaining == 95
    assert snapshot.gpu_remaining_s == pytest.approx(25.0)
    assert snapshot.wall_remaining_s == pytest.approx(50.0)
    assert snapshot.exhausted is False
    assert snapshot.to_dict()["wall_elapsed_s"] == pytest.approx(10.0)


def test_the_accountant_refuses_an_invalid_registration():
    with pytest.raises(PreRegistrationIncompleteError):
        BudgetAccountant(registration(episode_budget=0), clock=FakeClock())


# --------------------------------------------------------------------------------------
# stop reasons / stop criteria
# --------------------------------------------------------------------------------------
def test_every_stop_reason_is_accepted_and_unknown_ones_are_refused():
    accountant = BudgetAccountant(registration(), clock=FakeClock())
    for reason in STOP_REASONS:
        record = accountant.stop(reason, "test")
        assert record.reason == reason
        assert StopRecord(reason=reason, detail="", steps_used=0, episodes_used=0,
                          step_index=0).to_dict()["reason"] == reason
    with pytest.raises(StopReasonError):
        accountant.stop("KILLED_BY_VIBES")


def test_a_vanished_run_is_recorded_as_crash_not_success():
    accountant = BudgetAccountant(registration(), clock=FakeClock())
    record = accountant.lapse_as_crash()
    assert record.reason == "CRASH"
    assert accountant.stop_reason == "CRASH"


def test_stop_criteria_from_the_registration_detect_plateau():
    criteria = StopCriteria.from_registration(registration())
    assert criteria.evaluate([0.1, 0.2, 0.3], steps_used=3) is None
    plateau = criteria.evaluate([0.5, 0.5, 0.5, 0.5], steps_used=6)
    assert plateau is not None and plateau.reason == "PLATEAU"


def test_stop_criteria_detect_divergence_and_max_steps():
    criteria = StopCriteria.from_registration(registration())
    diverged = criteria.evaluate([0.1, float("nan")], steps_used=2)
    assert diverged is not None and diverged.reason == "DIVERGED"
    capped = criteria.evaluate([0.1], steps_used=10)
    assert capped is not None and capped.reason == "BUDGET_EXHAUSTED"


def test_stop_criteria_patience_counts_consecutive_non_improvements():
    criteria = StopCriteria.from_registration(registration())
    patience_hit = criteria.evaluate([0.2, 0.4, 0.3, 0.35, 0.32], steps_used=5)
    assert patience_hit is not None and patience_hit.reason == "PLATEAU"


def test_acceptance_threshold_is_evaluated():
    criteria = StopCriteria.from_registration(registration())
    assert criteria.meets_acceptance(0.6) is True
    assert criteria.meets_acceptance(0.59) is False
    assert criteria.meets_acceptance(float("nan")) is False


# --------------------------------------------------------------------------------------
# checkpoint: atomic write, integrity, resume refusal
# --------------------------------------------------------------------------------------
def metadata(**overrides) -> CheckpointMetadata:
    values = {"step": 3, "episode": 2, "registration_hash": registration().registration_hash(),
              "seed_ledger_hash": LEDGER_HASH, "evaluation_protocol_hash": PROTOCOL_HASH,
              "config_hash": CONFIG_HASH, "git_sha": "b74994a"}
    values.update(overrides)
    return CheckpointMetadata(**values)


def test_checkpoint_roundtrip_preserves_payload_and_hash(tmp_path):
    path = tmp_path / "ckpt.json"
    payload = {"weights": b"\x00\x01not-a-real-model", "bias": b"\x02"}
    record = write_checkpoint(path, metadata=metadata(), payload=payload)
    assert path.exists()
    loaded_meta, loaded_payload, loaded_record = read_checkpoint(path)
    assert loaded_payload == payload
    assert loaded_meta == metadata()
    assert loaded_record.checkpoint_hash == record.checkpoint_hash
    assert loaded_record.payload_bytes == 19


def test_checkpoint_write_is_atomic_and_leaves_no_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "ckpt.json"
    write_checkpoint(path, metadata=metadata(), payload={"weights": b"old"})
    before = path.read_bytes()

    def explode(*args, **kwargs):
        raise OSError("simulated crash between temp write and rename")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError):
        write_checkpoint(path, metadata=metadata(step=9), payload={"weights": b"new"})
    monkeypatch.undo()

    assert path.read_bytes() == before, "the old checkpoint must survive a failed write"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.json"], "temp file leaked"


def test_a_single_flipped_byte_is_refused(tmp_path):
    path = tmp_path / "ckpt.json"
    write_checkpoint(path, metadata=metadata(), payload={"weights": b"payload"})
    document = json.loads(path.read_text(encoding="utf-8"))
    document["metadata"]["step"] = 4
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        read_checkpoint(path)


def test_a_modified_payload_is_refused(tmp_path):
    path = tmp_path / "ckpt.json"
    write_checkpoint(path, metadata=metadata(), payload={"weights": b"payload"})
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["weights"] = "cGF5bG9hZHg="  # base64("payloadx"), NOT the same bytes
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError):
        read_checkpoint(path)


def test_a_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises(CheckpointIntegrityError):
        read_checkpoint(tmp_path / "nope.json")


def test_checkpoint_metadata_refuses_an_unknown_stop_reason():
    with pytest.raises(StopReasonError):
        metadata(stop_reason="WHATEVER").validate()


def test_resume_accepts_the_same_experiment(tmp_path):
    path = tmp_path / "ckpt.json"
    write_checkpoint(path, metadata=metadata(), payload={"weights": b"w"})
    record = resume_checkpoint(path, registration=registration(),
                               seed_ledger_hash=LEDGER_HASH, config_hash=CONFIG_HASH)
    assert record.metadata.step == 3


@pytest.mark.parametrize("overrides,mismatch", [
    ({"config_hash": "c" * 64}, "config_hash"),
    ({"seed_ledger_hash": "d" * 64}, "seed_ledger_hash"),
    ({"registration_hash": "e" * 64}, "registration_hash"),
])
def test_resume_refuses_a_checkpoint_from_another_experiment(tmp_path, overrides, mismatch):
    path = tmp_path / "ckpt.json"
    write_checkpoint(path, metadata=metadata(**overrides), payload={"weights": b"w"})
    with pytest.raises(CheckpointConfigMismatchError) as excinfo:
        resume_checkpoint(path, registration=registration(), seed_ledger_hash=LEDGER_HASH,
                          config_hash=CONFIG_HASH)
    assert mismatch in excinfo.value.mismatches


def test_resume_refuses_a_different_pre_registration(tmp_path):
    path = tmp_path / "ckpt.json"
    write_checkpoint(path, metadata=metadata(), payload={"weights": b"w"})
    other = registration(episode_budget=123)
    with pytest.raises(CheckpointConfigMismatchError):
        resume_checkpoint(path, registration=other, seed_ledger_hash=LEDGER_HASH,
                          config_hash=CONFIG_HASH)


@pytest.mark.parametrize("reason", ["SAFETY", "DIVERGED"])
def test_resume_after_a_safety_or_divergence_stop_is_refused(tmp_path, reason):
    path = tmp_path / "ckpt.json"
    write_checkpoint(path, metadata=metadata(stop_reason=reason), payload={"weights": b"w"})
    with pytest.raises(CheckpointConfigMismatchError):
        resume_checkpoint(path, registration=registration(), seed_ledger_hash=LEDGER_HASH,
                          config_hash=CONFIG_HASH)


# --------------------------------------------------------------------------------------
# the flag alone cannot authorize a run: the dependency gate is mandatory
# --------------------------------------------------------------------------------------
class StubDependencyGate:
    """A gate that passes, used only to prove the flag is not the thing being bypassed."""

    requirements = ("P5-05", "P7-10")

    def require_satisfied(self) -> None:
        return None


def test_authorize_training_run_requires_the_dependency_gate():
    with pytest.raises(TypeError):
        authorize_training_run(  # dependency_gate omitted on purpose
            gate=gate_with(registration()), request=BudgetRequest(),
            seed_ledger_hash=LEDGER_HASH, upstream_gates=ALL_PASS, authorization=GRANT)


def test_authorize_training_run_passes_only_with_a_passing_dependency_gate():
    authorization = authorize_training_run(
        gate=gate_with(registration()), request=BudgetRequest(), seed_ledger_hash=LEDGER_HASH,
        upstream_gates=ALL_PASS, authorization=GRANT,
        dependency_gate=StubDependencyGate())
    assert authorization.decision.algorithm == "PPO"
    assert authorization.dependency_requirements == ("P5-05", "P7-10")
    assert authorization.to_dict()["authorized"] is True


def test_the_authorization_flag_alone_does_not_bypass_the_real_dependency_gate():
    from badminton_brain.rl.dependency_gate import RequirementMissingError, default_gate
    from badminton_brain.rl.dependency_gate import RequirementMissingError as MissingError

    with pytest.raises(MissingError) as excinfo:
        authorize_training_run(gate=gate_with(registration()), request=BudgetRequest(),
                               seed_ledger_hash=LEDGER_HASH, upstream_gates=ALL_PASS,
                               authorization=GRANT, dependency_gate=default_gate())
    assert "P5-05" in excinfo.value.missing
    assert isinstance(excinfo.value, RequirementMissingError)


# --------------------------------------------------------------------------------------
# the package still does not train
# --------------------------------------------------------------------------------------
def test_the_training_gate_module_imports_no_training_framework():
    source = (ROOT / "src" / "badminton_brain" / "rl" / "training_gate.py").read_text(
        encoding="utf-8")
    for forbidden in ("import torch", "import gym", "stable_baselines3", "isaacsim"):
        assert forbidden not in source
    assert "torch" not in sys.modules
