from __future__ import annotations

import torch
from torch import nn

from smc_repro.config import LowerContextMode

_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or not hidden_dims:
            raise ValueError(
                "network dimensions must be positive and hidden_dims non-empty"
            )
        if any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden dimensions must be positive")
        dimensions = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(
            zip(dimensions[:-1], dimensions[1:], strict=True)
        ):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("Q-network input must be rank two [batch, features]")
        result = self.model(values)
        if not isinstance(result, torch.Tensor):
            raise TypeError("Q-network layers must return a tensor")
        return result


def lower_input_dim(mode: LowerContextMode) -> int:
    return 8 if mode is LowerContextMode.REWARD_ID_ONE_HOT else 7


def build_lower_context(
    states: torch.Tensor,
    upper_q_values: torch.Tensor,
    reward_ids: torch.Tensor,
    mode: LowerContextMode,
) -> torch.Tensor:
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("states must have shape [batch, 6]")
    if upper_q_values.shape != (states.shape[0], 2):
        raise ValueError("upper_q_values must have shape [batch, 2]")
    if not isinstance(reward_ids, torch.Tensor):
        raise TypeError("reward_ids must be a tensor")
    if reward_ids.shape != (states.shape[0],):
        raise ValueError("reward_ids must have shape [batch]")
    if reward_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("reward_ids must use a non-boolean integer dtype")
    if reward_ids.device != states.device:
        raise ValueError("reward_ids and states must be on the same device")
    if torch.any((reward_ids < 0) | (reward_ids > 1)):
        raise ValueError("reward ids must be 0 or 1")
    if mode is LowerContextMode.MAX_Q_SCALAR:
        context = torch.max(upper_q_values, dim=1, keepdim=True).values
    elif mode is LowerContextMode.REWARD_ID_SCALAR:
        context = reward_ids.to(dtype=states.dtype).unsqueeze(1)
    elif mode is LowerContextMode.REWARD_ID_ONE_HOT:
        context = torch.nn.functional.one_hot(
            reward_ids.to(dtype=torch.int64), num_classes=2
        ).to(states.dtype)
    else:
        raise AssertionError(f"unsupported lower context: {mode}")
    return torch.cat((states, context), dim=1)
