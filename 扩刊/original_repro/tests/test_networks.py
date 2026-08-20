from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from smc_repro.agents import (
    DualLayerValueAgent,
    MLPQNetwork,
    ReplayBuffer,
    Transition,
    build_lower_context,
    lower_input_dim,
)
from smc_repro.config import LowerContextMode, ReproductionProfile, load_profile
from smc_repro.rewards import RewardMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _profile(name: str = "paper_repro") -> ReproductionProfile:
    return load_profile(CONFIG_ROOT / f"{name}.yaml")


def _training_profile(
    *,
    epsilon_start: float = 0.6,
    epsilon_end: float = 0.01,
    epsilon_decrement: float = 0.0001,
    batch_size: int = 2,
    target_update_steps: int = 200,
) -> ReproductionProfile:
    base = _profile()
    return replace(
        base,
        training=replace(
            base.training,
            replay_capacity=32,
            batch_size=batch_size,
            target_update_steps=target_update_steps,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decrement=epsilon_decrement,
        ),
    )


def _linear_dimensions(network: MLPQNetwork) -> tuple[tuple[int, int], ...]:
    return tuple(
        (layer.in_features, layer.out_features)
        for layer in network.model
        if isinstance(layer, nn.Linear)
    )


def _transition(index: int, *, done: bool = False) -> Transition:
    state = np.asarray(
        [index, index + 1, index + 2, index + 3, index + 4, index + 5],
        dtype=np.float32,
    ) / 10.0
    return Transition(
        state=state,
        rule_action=index % 9,
        reward=float((index % 3) - 1),
        next_state=state + np.float32(0.05),
        reward_id=index % 2,
        done=done,
    )


@pytest.mark.parametrize(
    ("profile_name", "upper_dimensions", "lower_dimensions"),
    [
        (
            "legacy_snapshot",
            ((6, 10), (10, 10), (10, 10), (10, 2)),
            (
                (7, 50),
                (50, 50),
                (50, 50),
                (50, 50),
                (50, 50),
                (50, 50),
                (50, 50),
                (50, 9),
            ),
        ),
        (
            "paper_repro",
            ((6, 10), (10, 10), (10, 2)),
            ((7, 50), (50, 50), (50, 9)),
        ),
        (
            "corrected_smc",
            ((6, 10), (10, 10), (10, 2)),
            ((7, 50), (50, 50), (50, 9)),
        ),
    ],
)
def test_profile_architecture_creates_exact_layer_counts_and_dimensions(
    profile_name: str,
    upper_dimensions: tuple[tuple[int, int], ...],
    lower_dimensions: tuple[tuple[int, int], ...],
) -> None:
    agent = DualLayerValueAgent(
        _profile(profile_name),
        seed=101,
        device=torch.device("cpu"),
        double_dqn=True,
    )

    assert _linear_dimensions(agent.upper_online) == upper_dimensions
    assert _linear_dimensions(agent.upper_target) == upper_dimensions
    assert _linear_dimensions(agent.lower_online) == lower_dimensions
    assert _linear_dimensions(agent.lower_target) == lower_dimensions


def test_agent_initialization_is_seed_local_and_preserves_global_torch_rng() -> None:
    profile = _profile()
    torch.manual_seed(9901)
    before = torch.random.get_rng_state().clone()
    first = DualLayerValueAgent(
        profile,
        seed=77,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    after = torch.random.get_rng_state().clone()
    torch.rand(97)
    second = DualLayerValueAgent(
        profile,
        seed=77,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    different = DualLayerValueAgent(
        profile,
        seed=78,
        device=torch.device("cpu"),
        double_dqn=True,
    )

    assert torch.equal(before, after)
    for first_parameter, second_parameter in zip(
        first.upper_online.parameters(), second.upper_online.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)
    for first_parameter, second_parameter in zip(
        first.lower_online.parameters(), second.lower_online.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)
    assert any(
        not torch.equal(first_parameter, different_parameter)
        for first_parameter, different_parameter in zip(
            first.upper_online.parameters(),
            different.upper_online.parameters(),
            strict=True,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA RNG state")
def test_cpu_agent_initialization_preserves_cpu_and_all_cuda_global_rng_states() -> None:
    profile = _profile()
    torch.manual_seed(9101)
    torch.cuda.manual_seed_all(9102)
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    DualLayerValueAgent(
        profile,
        seed=77,
        device=torch.device("cpu"),
        double_dqn=True,
    )

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    cuda_after = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    assert len(cuda_after) == len(cuda_before)
    assert all(
        torch.equal(before, after)
        for before, after in zip(cuda_before, cuda_after, strict=True)
    )


def test_replay_transition_defensively_copies_and_exposes_read_only_vectors() -> None:
    state = np.arange(6, dtype=np.float32)
    next_state = state + np.float32(1.0)
    transition = Transition(state, 3, 1.0, next_state, 1, False)
    state[:] = -99.0
    next_state[:] = -88.0

    assert np.array_equal(transition.state, np.arange(6, dtype=np.float32))
    assert np.array_equal(transition.next_state, np.arange(1, 7, dtype=np.float32))
    assert not transition.state.flags.writeable
    assert not transition.next_state.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        transition.state[0] = 5.0


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            LowerContextMode.MAX_Q_SCALAR,
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
             [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 8.0]],
        ),
        (
            LowerContextMode.REWARD_ID_SCALAR,
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
             [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 0.0]],
        ),
        (
            LowerContextMode.REWARD_ID_ONE_HOT,
            [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0],
             [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 1.0, 0.0]],
        ),
    ],
)
def test_all_lower_context_modes_have_exact_values_and_dimensions(
    mode: LowerContextMode,
    expected: list[list[float]],
) -> None:
    states = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    upper_q = torch.tensor([[1.0, 9.0], [8.0, 2.0]])
    reward_ids = torch.tensor([1, 0])

    observed = build_lower_context(states, upper_q, reward_ids, mode)

    assert lower_input_dim(mode) == len(expected[0])
    assert torch.equal(observed, torch.tensor(expected))


@pytest.mark.parametrize("mode", tuple(LowerContextMode))
@pytest.mark.parametrize(
    "invalid_reward_ids",
    [
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        torch.tensor([0.5, 1.0], dtype=torch.float64),
        torch.tensor([False, True], dtype=torch.bool),
        torch.tensor([1.0e100, 0.0], dtype=torch.float64),
        cast(torch.Tensor, 10**1000),
    ],
)
def test_lower_context_rejects_noninteger_bool_fractional_and_huge_reward_ids(
    mode: LowerContextMode,
    invalid_reward_ids: torch.Tensor,
) -> None:
    states = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    upper_q = torch.tensor([[1.0, 9.0], [8.0, 2.0]])

    with pytest.raises((TypeError, ValueError)):
        build_lower_context(states, upper_q, invalid_reward_ids, mode)


@pytest.mark.parametrize("mode", tuple(LowerContextMode))
@pytest.mark.parametrize("integer_dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_lower_context_accepts_nonbool_integer_reward_ids_consistently(
    mode: LowerContextMode,
    integer_dtype: torch.dtype,
) -> None:
    states = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    upper_q = torch.tensor([[1.0, 9.0], [8.0, 2.0]])
    reward_ids = torch.tensor([1, 0], dtype=integer_dtype)

    observed = build_lower_context(states, upper_q, reward_ids, mode)

    assert observed.shape == (2, lower_input_dim(mode))


def test_agent_smoke_ignore_rule_is_clone_safe_and_exact(
    tmp_path: Path,
) -> None:
    source_ignore = PROJECT_ROOT / ".gitignore"
    assert source_ignore.is_file()
    clean_repository = tmp_path / "clean-original-repro"
    clean_repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=clean_repository, check=True)
    shutil.copyfile(source_ignore, clean_repository / ".gitignore")
    smoke_output = (
        clean_repository
        / "artifacts"
        / "preflight"
        / "agent_smoke"
        / "run.json"
    )
    adjacent_output = (
        clean_repository / "artifacts" / "preflight" / "not_agent_smoke.json"
    )
    smoke_output.parent.mkdir(parents=True)
    smoke_output.write_text("{}\n", encoding="utf-8")
    adjacent_output.write_text("{}\n", encoding="utf-8")

    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", smoke_output.relative_to(clean_repository)],
        cwd=clean_repository,
        check=False,
    )
    adjacent = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--",
            adjacent_output.relative_to(clean_repository),
        ],
        cwd=clean_repository,
        check=False,
    )

    assert ignored.returncode == 0
    assert adjacent.returncode == 1


def test_joint_epsilon_exploration_reaches_both_modes_and_all_nine_actions() -> None:
    agent = DualLayerValueAgent(
        _training_profile(epsilon_start=1.0, epsilon_end=1.0, epsilon_decrement=0.0),
        seed=1234,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    state = np.zeros(6, dtype=np.float32)

    decisions = [agent.decide(state, training=True) for _ in range(256)]

    assert {decision.reward_mode for decision in decisions} == set(RewardMode)
    assert {decision.rule_action for decision in decisions} == set(range(9))
    assert all(decision.exploratory for decision in decisions)
    assert all(decision.epsilon == 1.0 for decision in decisions)


def test_zero_epsilon_decision_matches_both_direct_network_argmax_values() -> None:
    agent = DualLayerValueAgent(
        _training_profile(epsilon_start=0.0, epsilon_end=0.0, epsilon_decrement=0.0),
        seed=31,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    state = np.asarray([0.2, 0.4, 0.1, 0.6, 0.3, 0.5], dtype=np.float32)
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        upper_q = agent.upper_online(state_tensor)
        reward_id = int(torch.argmax(upper_q, dim=1).item())
        context = build_lower_context(
            state_tensor,
            upper_q,
            torch.tensor([reward_id]),
            agent.profile.architecture.lower_context,
        )
        rule_action = int(torch.argmax(agent.lower_online(context), dim=1).item())

    decision = agent.decide(state, training=True)

    assert decision.reward_mode is RewardMode(reward_id)
    assert decision.rule_action == rule_action
    assert decision.epsilon == 0.0
    assert not decision.exploratory


def test_epsilon_decrement_never_crosses_configured_floor_and_eval_is_inert() -> None:
    agent = DualLayerValueAgent(
        _training_profile(
            epsilon_start=0.05,
            epsilon_end=0.01,
            epsilon_decrement=0.02,
        ),
        seed=19,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    state = np.zeros(6, dtype=np.float32)

    observed = [agent.decide(state, training=True).epsilon for _ in range(5)]
    decision_count = agent.decision_count
    evaluation = agent.decide(state, training=False)

    assert observed == pytest.approx([0.05, 0.03, 0.01, 0.01, 0.01])
    assert agent.epsilon == 0.01
    assert evaluation.epsilon == 0.0
    assert not evaluation.exploratory
    assert agent.decision_count == decision_count


def test_replay_sampling_continues_exactly_after_primitive_state_restore() -> None:
    first = ReplayBuffer(capacity=8, seed=42)
    for index in range(8):
        first.append(_transition(index))
    saved = first.state_dict()
    expected = first.sample(4)
    restored = ReplayBuffer(capacity=8, seed=999)

    restored.load_state_dict(saved)
    observed = restored.sample(4)

    assert [item.rule_action for item in observed] == [
        item.rule_action for item in expected
    ]
    assert all(
        np.array_equal(left.state, right.state)
        for left, right in zip(observed, expected, strict=True)
    )


def test_optimizer_step_changes_online_parameters_but_not_targets_between_syncs() -> None:
    agent = DualLayerValueAgent(
        _training_profile(batch_size=2, target_update_steps=10),
        seed=91,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    for index in range(4):
        agent.remember(_transition(index, done=index == 3))
    agent.global_update_step = 1
    online_before = [
        parameter.detach().clone()
        for network in (agent.upper_online, agent.lower_online)
        for parameter in network.parameters()
    ]
    target_before = [
        parameter.detach().clone()
        for network in (agent.upper_target, agent.lower_target)
        for parameter in network.parameters()
    ]

    report = agent.update()

    online_after = [
        parameter.detach()
        for network in (agent.upper_online, agent.lower_online)
        for parameter in network.parameters()
    ]
    target_after = [
        parameter.detach()
        for network in (agent.upper_target, agent.lower_target)
        for parameter in network.parameters()
    ]
    assert report is not None
    assert report.global_update_step == 2
    assert any(
        not torch.equal(before, after)
        for before, after in zip(online_before, online_after, strict=True)
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(target_before, target_after, strict=True)
    )


def test_explicit_target_sync_copies_both_networks_exactly() -> None:
    agent = DualLayerValueAgent(
        _training_profile(),
        seed=301,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    with torch.no_grad():
        for parameter in agent.upper_online.parameters():
            parameter.add_(1.0)
        for parameter in agent.lower_online.parameters():
            parameter.sub_(1.0)

    agent.sync_targets()

    for online, target in (
        (agent.upper_online, agent.upper_target),
        (agent.lower_online, agent.lower_target),
    ):
        for online_parameter, target_parameter in zip(
            online.parameters(), target.parameters(), strict=True
        ):
            assert torch.equal(online_parameter, target_parameter)
