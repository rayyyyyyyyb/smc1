from __future__ import annotations

import copy
import dataclasses
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from smc_repro.agents import (
    CHECKPOINT_SCHEMA_VERSION,
    DualLayerValueAgent,
    Transition,
    load_checkpoint,
    save_checkpoint,
)
from smc_repro.config import ReproductionProfile, load_profile, profile_sha256
from smc_repro.experiment_contract import FAILURE_STREAM_VERSION, RunContract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _profile(name: str = "paper_repro") -> ReproductionProfile:
    base = load_profile(CONFIG_ROOT / f"{name}.yaml")
    return replace(
        base,
        training=replace(
            base.training,
            replay_capacity=32,
            batch_size=2,
            target_update_steps=5,
        ),
    )


def _contract(profile: ReproductionProfile, *, method: str = "ddqn") -> RunContract:
    return RunContract(
        schema_version=1,
        git_commit="1" * 40,
        profile_name=profile.profile.value,
        profile_sha256=profile_sha256(profile),
        bank_manifest_sha256="2" * 64,
        method=method,
        train_seed=101,
        policy_seed=202,
        failure_stream_version=FAILURE_STREAM_VERSION,
        environment_metadata_path="artifacts/environment_5090_resolved.json",
    )


def _transition(index: int, *, done: bool = False) -> Transition:
    state = np.asarray(
        [index + 0.1, index + 0.2, index + 0.3, index + 0.4, index + 0.5, index + 0.6],
        dtype=np.float32,
    )
    return Transition(
        state=state,
        rule_action=(index * 2) % 9,
        reward=float(index - 1),
        next_state=state + np.float32(0.25),
        reward_id=index % 2,
        done=done,
    )


def _agent(
    profile: ReproductionProfile | None = None,
    *,
    seed: int = 17,
    double_dqn: bool = True,
) -> DualLayerValueAgent:
    return DualLayerValueAgent(
        profile or _profile(),
        seed=seed,
        device=torch.device("cpu"),
        double_dqn=double_dqn,
    )


def _populate(agent: DualLayerValueAgent) -> None:
    for index in range(6):
        agent.remember(_transition(index, done=index == 5))
    agent.decide(np.zeros(6, dtype=np.float32), training=True)


def _snapshot(agent: DualLayerValueAgent) -> dict[str, Any]:
    return {
        "profile": agent.profile.to_dict(),
        "seed": agent.seed,
        "device": str(agent.device),
        "double_dqn": agent.double_dqn,
        "upper_online": copy.deepcopy(agent.upper_online.state_dict()),
        "upper_target": copy.deepcopy(agent.upper_target.state_dict()),
        "lower_online": copy.deepcopy(agent.lower_online.state_dict()),
        "lower_target": copy.deepcopy(agent.lower_target.state_dict()),
        "upper_optimizer": copy.deepcopy(agent.upper_optimizer.state_dict()),
        "lower_optimizer": copy.deepcopy(agent.lower_optimizer.state_dict()),
        "global_update_step": agent.global_update_step,
        "decision_count": agent.decision_count,
        "epsilon": agent.epsilon,
        "agent_rng_state": copy.deepcopy(agent.rng_state()),
        "replay_state": copy.deepcopy(agent.replay.state_dict()),
        "training_flags": (
            agent.upper_online.training,
            agent.upper_target.training,
            agent.lower_online.training,
            agent.lower_target.training,
        ),
    }


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def _rewrite_checkpoint(path: Path, mutate: Any) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, path)


def test_missing_checkpoint_raises_file_not_found_without_agent_mutation(
    tmp_path: Path,
) -> None:
    agent = _agent()
    _populate(agent)
    before = _snapshot(agent)

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            tmp_path / "missing.pt",
            agent,
            _contract(agent.profile),
            for_training=True,
        )

    _assert_nested_equal(_snapshot(agent), before)


def test_evaluation_load_restores_networks_only_clears_replay_and_forces_zero_epsilon(
    tmp_path: Path,
) -> None:
    source = _agent(seed=11)
    _populate(source)
    checkpoint = tmp_path / "eval.pt"
    save_checkpoint(checkpoint, source, _contract(source.profile))
    destination = _agent(seed=999)
    destination.remember(_transition(9))

    load_checkpoint(
        checkpoint,
        destination,
        _contract(destination.profile),
        for_training=False,
    )

    assert destination.epsilon == 0.0
    assert len(destination.replay) == 0
    assert not destination.upper_online.training
    assert not destination.upper_target.training
    assert not destination.lower_online.training
    assert not destination.lower_target.training
    for source_network, destination_network in (
        (source.upper_online, destination.upper_online),
        (source.upper_target, destination.upper_target),
        (source.lower_online, destination.lower_online),
        (source.lower_target, destination.lower_target),
    ):
        for source_parameter, destination_parameter in zip(
            source_network.parameters(), destination_network.parameters(), strict=True
        ):
            assert torch.equal(source_parameter, destination_parameter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA RNG state")
def test_cpu_evaluation_load_preserves_cpu_and_all_cuda_global_rng_states(
    tmp_path: Path,
) -> None:
    source = _agent(seed=31)
    _populate(source)
    checkpoint = tmp_path / "eval-rng.pt"
    contract = _contract(source.profile)
    save_checkpoint(checkpoint, source, contract)
    destination = _agent(seed=32)
    torch.manual_seed(9201)
    torch.cuda.manual_seed_all(9202)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    load_checkpoint(checkpoint, destination, contract, for_training=False)

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    cuda_after = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    assert len(cuda_after) == len(cuda_before)
    assert all(
        torch.equal(before, after)
        for before, after in zip(cuda_before, cuda_after, strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA RNG state")
def test_rejected_training_load_preserves_global_rng_and_destination_state(
    tmp_path: Path,
) -> None:
    source = _agent(seed=41)
    _populate(source)
    checkpoint = tmp_path / "rejected-rng.pt"
    contract = _contract(source.profile)
    save_checkpoint(checkpoint, source, contract)
    _rewrite_checkpoint(
        checkpoint,
        lambda payload: payload.__setitem__("torch_cuda_rng_states", ()),
    )
    destination = _agent(seed=42)
    _populate(destination)
    destination_before = _snapshot(destination)
    torch.manual_seed(9301)
    torch.cuda.manual_seed_all(9302)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    with pytest.raises(ValueError, match="CUDA|RNG|checkpoint"):
        load_checkpoint(checkpoint, destination, contract, for_training=True)

    _assert_nested_equal(_snapshot(destination), destination_before)
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    cuda_after = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    assert len(cuda_after) == len(cuda_before)
    assert all(
        torch.equal(before, after)
        for before, after in zip(cuda_before, cuda_after, strict=True)
    )


def test_training_resume_restores_next_exploration_and_replay_sample_exactly(
    tmp_path: Path,
) -> None:
    profile = replace(
        _profile(),
        training=replace(
            _profile().training,
            replay_capacity=32,
            batch_size=2,
            epsilon_start=1.0,
            epsilon_end=1.0,
            epsilon_decrement=0.0,
        ),
    )
    source = _agent(profile, seed=73)
    _populate(source)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, source, _contract(profile))
    expected_decision = source.decide(np.ones(6, dtype=np.float32), training=True)
    expected_sample = source.replay.sample(3)
    destination = _agent(profile, seed=900)

    load_checkpoint(
        checkpoint,
        destination,
        _contract(profile),
        for_training=True,
    )
    observed_decision = destination.decide(
        np.ones(6, dtype=np.float32), training=True
    )
    observed_sample = destination.replay.sample(3)

    assert observed_decision == expected_decision
    assert [item.rule_action for item in observed_sample] == [
        item.rule_action for item in expected_sample
    ]
    assert all(
        np.array_equal(left.state, right.state)
        for left, right in zip(observed_sample, expected_sample, strict=True)
    )


def test_uninterrupted_cpu_update_equals_save_load_resume_with_exact_tensors(
    tmp_path: Path,
) -> None:
    source = _agent(seed=51)
    _populate(source)
    first_report = source.update()
    assert first_report is not None
    checkpoint = tmp_path / "continuity.pt"
    save_checkpoint(checkpoint, source, _contract(source.profile))

    expected_report = source.update()
    expected = _snapshot(source)
    destination = _agent(seed=source.seed)
    load_checkpoint(
        checkpoint,
        destination,
        _contract(destination.profile),
        for_training=True,
    )
    observed_report = destination.update()

    assert observed_report == expected_report
    _assert_nested_equal(_snapshot(destination), expected)


@pytest.mark.parametrize("mismatch", ["contract", "profile", "double_dqn", "dimension"])
def test_checkpoint_compatibility_failure_prevalidates_without_any_agent_mutation(
    tmp_path: Path,
    mismatch: str,
) -> None:
    source = _agent(seed=88)
    _populate(source)
    checkpoint = tmp_path / f"mismatch-{mismatch}.pt"
    save_checkpoint(checkpoint, source, _contract(source.profile))

    if mismatch == "profile":
        destination = _agent(_profile("corrected_smc"), seed=333)
        expected_contract = _contract(destination.profile)
    elif mismatch == "double_dqn":
        destination = _agent(seed=333, double_dqn=False)
        expected_contract = _contract(destination.profile)
    else:
        destination = _agent(seed=333)
        expected_contract = _contract(destination.profile)
    if mismatch == "contract":
        expected_contract = dataclasses.replace(expected_contract, method="vanilla_dqn")
    if mismatch == "dimension":
        def change_dimension(payload: dict[str, Any]) -> None:
            state = payload["upper_online"]
            first_key = next(iter(state))
            state[first_key] = state[first_key][:-1]

        _rewrite_checkpoint(checkpoint, change_dimension)
    _populate(destination)
    before = _snapshot(destination)

    with pytest.raises(ValueError, match="checkpoint|contract|profile|double|dimension"):
        load_checkpoint(
            checkpoint,
            destination,
            expected_contract,
            for_training=True,
        )

    _assert_nested_equal(_snapshot(destination), before)


@pytest.mark.parametrize("failure", ["corrupted", "schema"])
def test_corrupted_bytes_and_wrong_schema_fail_clearly_without_mutation(
    tmp_path: Path,
    failure: str,
) -> None:
    agent = _agent(seed=121)
    _populate(agent)
    checkpoint = tmp_path / f"{failure}.pt"
    save_checkpoint(checkpoint, agent, _contract(agent.profile))
    if failure == "corrupted":
        checkpoint.write_bytes(b"not a torch checkpoint")
        expected = "load"
    else:
        _rewrite_checkpoint(
            checkpoint,
            lambda payload: payload.__setitem__(
                "schema_version", CHECKPOINT_SCHEMA_VERSION + 1
            ),
        )
        expected = "schema"
    destination = _agent(seed=122)
    _populate(destination)
    before = _snapshot(destination)

    with pytest.raises(ValueError, match=expected):
        load_checkpoint(
            checkpoint,
            destination,
            _contract(destination.profile),
            for_training=True,
        )

    _assert_nested_equal(_snapshot(destination), before)


def test_checkpoint_sha_matches_independent_hashlib_computation(tmp_path: Path) -> None:
    agent = _agent(seed=211)
    _populate(agent)
    checkpoint = tmp_path / "sha.pt"

    reported_save_sha = save_checkpoint(checkpoint, agent, _contract(agent.profile))
    independent_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    destination = _agent(seed=212)
    reported_load_sha = load_checkpoint(
        checkpoint,
        destination,
        _contract(destination.profile),
        for_training=True,
    )

    assert reported_save_sha == independent_sha
    assert reported_load_sha == independent_sha


def test_final_checkpoint_payload_loads_with_weights_only_true(tmp_path: Path) -> None:
    agent = _agent(seed=311)
    _populate(agent)
    checkpoint = tmp_path / "weights-only.pt"
    save_checkpoint(checkpoint, agent, _contract(agent.profile))

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["profile_sha256"] == profile_sha256(agent.profile)
    assert isinstance(payload["replay_state"]["items"], tuple)


@pytest.mark.parametrize("simulated_cuda_available", [False, True])
def test_evaluation_load_is_portable_across_opposite_cuda_rng_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulated_cuda_available: bool,
) -> None:
    source = _agent(seed=401)
    _populate(source)
    checkpoint = tmp_path / f"eval-topology-{simulated_cuda_available}.pt"
    contract = _contract(source.profile)
    save_checkpoint(checkpoint, source, contract)

    if simulated_cuda_available:
        replacement_states: tuple[torch.Tensor, ...] = ()
        simulated_device_count = 1
    else:
        replacement_states = (torch.random.get_rng_state().clone(),)
        simulated_device_count = 0
    _rewrite_checkpoint(
        checkpoint,
        lambda payload: payload.__setitem__(
            "torch_cuda_rng_states", replacement_states
        ),
    )
    training_destination = _agent(seed=402)
    _populate(training_destination)
    training_before = _snapshot(training_destination)
    evaluation_destination = _agent(seed=403)
    cpu_rng_before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: simulated_cuda_available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: simulated_device_count)

    with pytest.raises(ValueError, match="CUDA|RNG|checkpoint"):
        load_checkpoint(
            checkpoint,
            training_destination,
            contract,
            for_training=True,
        )
    _assert_nested_equal(_snapshot(training_destination), training_before)

    load_checkpoint(
        checkpoint,
        evaluation_destination,
        contract,
        for_training=False,
    )

    assert evaluation_destination.epsilon == 0.0
    assert len(evaluation_destination.replay) == 0
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    for source_network, destination_network in (
        (source.upper_online, evaluation_destination.upper_online),
        (source.upper_target, evaluation_destination.upper_target),
        (source.lower_online, evaluation_destination.lower_online),
        (source.lower_target, evaluation_destination.lower_target),
    ):
        for source_parameter, destination_parameter in zip(
            source_network.parameters(), destination_network.parameters(), strict=True
        ):
            assert torch.equal(source_parameter, destination_parameter)
