from __future__ import annotations

from typing import Any

import pytest
import torch

from smc_repro.agents import MLPQNetwork, value_target


@pytest.mark.parametrize(
    ("double_dqn", "expected"),
    [
        (True, [2.0 + 0.95 * 3.0, -1.0]),
        (False, [2.0 + 0.95 * 7.0, -1.0]),
    ],
)
def test_double_and_vanilla_targets_match_hand_calculation_without_gradients(
    double_dqn: bool,
    expected: list[float],
) -> None:
    online = torch.tensor(
        [[1.0, 9.0], [8.0, 2.0]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[7.0, 3.0], [4.0, 6.0]],
        requires_grad=True,
    )
    rewards = torch.tensor([2.0, -1.0], requires_grad=True)
    dones = torch.tensor([False, True])

    observed = value_target(
        rewards,
        dones,
        online,
        target,
        0.95,
        double_dqn=double_dqn,
    )

    assert torch.equal(observed, torch.tensor(expected))
    assert not observed.requires_grad
    assert observed.grad_fn is None


@pytest.mark.parametrize("double_dqn", [True, False])
def test_terminal_transition_never_bootstraps_for_any_target_mode(
    double_dqn: bool,
) -> None:
    observed = value_target(
        torch.tensor([4.25]),
        torch.tensor([True]),
        torch.tensor([[1000.0, 2000.0]]),
        torch.tensor([[3000.0, 4000.0]]),
        0.95,
        double_dqn=double_dqn,
    )

    assert torch.equal(observed, torch.tensor([4.25]))


def test_lower_network_has_nine_actions_and_nine_action_targets_are_detached() -> None:
    lower_online = MLPQNetwork(7, (5, 5), 9)
    lower_target = MLPQNetwork(7, (5, 5), 9)
    next_context = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    online_q = lower_online(next_context)
    target_q = lower_target(next_context)

    targets = value_target(
        torch.tensor([1.0, 0.0, -1.0]),
        torch.tensor([False, False, True]),
        online_q,
        target_q,
        0.5,
        double_dqn=True,
    )

    assert online_q.shape == (3, 9)
    assert target_q.shape == (3, 9)
    assert targets.shape == (3,)
    assert not targets.requires_grad


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("rewards", torch.tensor([[2.0], [-1.0]])),
        ("rewards", torch.tensor([2, -1], dtype=torch.int64)),
        ("rewards", torch.tensor([2.0, float("inf")])),
        ("rewards", 10**1000),
        ("dones", torch.tensor([[False], [True]])),
        ("dones", torch.tensor([0.0, 1.0])),
        ("online_next_q", torch.tensor([1.0, 2.0])),
        ("online_next_q", torch.tensor([[1.0], [2.0], [3.0]])),
        ("online_next_q", torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)),
        ("online_next_q", torch.tensor([[1.0, float("nan")], [3.0, 4.0]])),
        ("target_next_q", torch.tensor([1.0, 2.0])),
        ("target_next_q", torch.tensor([[1.0], [2.0], [3.0]])),
        ("target_next_q", torch.tensor([[1.0, 2.0], [3.0, float("inf")]])),
        ("target_next_q", torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)),
    ],
)
@pytest.mark.parametrize("double_dqn", [True, False])
def test_value_target_rejects_invalid_shapes_dtypes_nonfinite_and_huge_inputs(
    field: str,
    invalid_value: Any,
    double_dqn: bool,
) -> None:
    arguments: dict[str, Any] = {
        "rewards": torch.tensor([2.0, -1.0]),
        "dones": torch.tensor([False, True]),
        "online_next_q": torch.tensor([[1.0, 9.0], [8.0, 2.0]]),
        "target_next_q": torch.tensor([[7.0, 3.0], [4.0, 6.0]]),
        "gamma": 0.95,
    }
    arguments[field] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        value_target(**arguments, double_dqn=double_dqn)


@pytest.mark.parametrize(
    "invalid_gamma",
    [True, -0.1, 1.1, float("nan"), float("inf"), 10**1000],
)
def test_value_target_rejects_invalid_gamma_without_leaking_overflow(
    invalid_gamma: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        value_target(
            torch.tensor([2.0, -1.0]),
            torch.tensor([False, True]),
            torch.tensor([[1.0, 9.0], [8.0, 2.0]]),
            torch.tensor([[7.0, 3.0], [4.0, 6.0]]),
            invalid_gamma,  # type: ignore[arg-type]
            double_dqn=True,
        )


def test_value_target_rejects_nonboolean_double_flag() -> None:
    with pytest.raises((TypeError, ValueError)):
        value_target(
            torch.tensor([2.0, -1.0]),
            torch.tensor([False, True]),
            torch.tensor([[1.0, 9.0], [8.0, 2.0]]),
            torch.tensor([[7.0, 3.0], [4.0, 6.0]]),
            0.95,
            double_dqn=1,  # type: ignore[arg-type]
        )
