from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from smc_repro.agents.networks import (
    MLPQNetwork,
    build_lower_context,
    lower_input_dim,
)
from smc_repro.agents.replay import ReplayBuffer, Transition
from smc_repro.config import ReproductionProfile
from smc_repro.rewards import RewardMode


@dataclass(frozen=True)
class AgentDecision:
    rule_action: int
    reward_mode: RewardMode
    epsilon: float
    exploratory: bool


@dataclass(frozen=True)
class UpdateReport:
    upper_loss: float
    lower_loss: float
    epsilon: float
    global_update_step: int


def value_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    gamma: float,
    *,
    double_dqn: bool,
) -> torch.Tensor:
    for name, value in (
        ("rewards", rewards),
        ("dones", dones),
        ("online_next_q", online_next_q),
        ("target_next_q", target_next_q),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
    if rewards.ndim != 1 or rewards.shape[0] == 0:
        raise ValueError("rewards must have non-empty shape [batch]")
    if not torch.is_floating_point(rewards):
        raise TypeError("rewards must use a floating-point dtype")
    if not bool(torch.all(torch.isfinite(rewards)).item()):
        raise ValueError("rewards must be finite")
    if dones.shape != rewards.shape:
        raise ValueError("dones must have shape [batch]")
    if dones.dtype is not torch.bool:
        raise TypeError("dones must use the boolean dtype")
    if dones.device != rewards.device:
        raise ValueError("dones and rewards must be on the same device")
    for name, values in (
        ("online_next_q", online_next_q),
        ("target_next_q", target_next_q),
    ):
        if values.ndim != 2 or values.shape[0] != rewards.shape[0]:
            raise ValueError(f"{name} must have shape [batch, actions]")
        if values.shape[1] == 0:
            raise ValueError(f"{name} must have at least one action")
        if values.dtype != rewards.dtype:
            raise TypeError(f"{name} and rewards must use the same dtype")
        if values.device != rewards.device:
            raise ValueError(f"{name} and rewards must be on the same device")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError(f"{name} must be finite")
    if online_next_q.shape != target_next_q.shape:
        raise ValueError("online_next_q and target_next_q shapes must match")
    if type(gamma) not in (int, float):
        raise TypeError("gamma must be a real number")
    try:
        gamma_value = float(gamma)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gamma must be finite and in [0, 1]") from exc
    if not np.isfinite(gamma_value) or not 0.0 <= gamma_value <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if type(double_dqn) is not bool:
        raise TypeError("double_dqn must be a boolean")
    with torch.no_grad():
        if double_dqn:
            selected = torch.argmax(online_next_q, dim=1)
            bootstrap = target_next_q.gather(1, selected.unsqueeze(1)).squeeze(1)
        else:
            bootstrap = torch.max(target_next_q, dim=1).values
        return rewards + gamma_value * bootstrap * (~dones).to(rewards.dtype)


class DualLayerValueAgent:
    def __init__(
        self,
        profile: ReproductionProfile,
        *,
        seed: int,
        device: torch.device,
        double_dqn: bool,
    ) -> None:
        if type(profile) is not ReproductionProfile:
            raise TypeError("profile must be a ReproductionProfile")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(double_dqn) is not bool:
            raise ValueError("double_dqn must be a boolean")
        selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device selected but CUDA is unavailable")
        self.profile = profile
        self.seed = seed
        self.device = selected_device
        self.double_dqn = double_dqn
        fork_devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            self.upper_online = MLPQNetwork(
                6,
                profile.architecture.upper_hidden,
                2,
            ).to(self.device)
            self.lower_online = MLPQNetwork(
                lower_input_dim(profile.architecture.lower_context),
                profile.architecture.lower_hidden,
                9,
            ).to(self.device)
            self.upper_target = copy.deepcopy(self.upper_online)
            self.lower_target = copy.deepcopy(self.lower_online)
        self.upper_optimizer = torch.optim.Adam(
            self.upper_online.parameters(), lr=profile.training.learning_rate
        )
        self.lower_optimizer = torch.optim.Adam(
            self.lower_online.parameters(), lr=profile.training.learning_rate
        )
        self._loss = nn.MSELoss()
        self.replay = ReplayBuffer(profile.training.replay_capacity, seed)
        self._rng = random.Random(seed)
        self.global_update_step = 0
        self.decision_count = 0
        self.epsilon = profile.training.epsilon_start

    def rng_state(self) -> tuple[object, ...]:
        return self._rng.getstate()

    def set_rng_state(self, state: tuple[object, ...]) -> None:
        if not isinstance(state, tuple):
            raise ValueError("invalid agent RNG state")
        probe = random.Random()
        try:
            probe.setstate(state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid agent RNG state") from exc
        self._rng.setstate(state)

    def remember(self, transition: Transition) -> None:
        self.replay.append(transition)

    def decide(self, state: np.ndarray, *, training: bool) -> AgentDecision:
        values = np.asarray(state, dtype=np.float32)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("state must be a finite float32 vector with shape (6,)")
        if type(training) is not bool:
            raise ValueError("training must be a boolean")
        decision_epsilon = self.epsilon if training else 0.0
        exploratory = training and self._rng.random() < decision_epsilon
        if exploratory:
            reward_id = self._rng.randrange(2)
            rule_action = self._rng.randrange(9)
        else:
            state_tensor = torch.as_tensor(
                values, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                upper_q = self.upper_online(state_tensor)
                reward_id = int(torch.argmax(upper_q, dim=1).item())
                context = build_lower_context(
                    state_tensor,
                    upper_q,
                    torch.tensor([reward_id], dtype=torch.int64, device=self.device),
                    self.profile.architecture.lower_context,
                )
                rule_action = int(
                    torch.argmax(self.lower_online(context), dim=1).item()
                )
        if training:
            self.decision_count += 1
            self.epsilon = max(
                self.profile.training.epsilon_end,
                self.epsilon - self.profile.training.epsilon_decrement,
            )
        return AgentDecision(
            rule_action=rule_action,
            reward_mode=RewardMode(reward_id),
            epsilon=decision_epsilon,
            exploratory=exploratory,
        )

    def sync_targets(self) -> None:
        self.upper_target.load_state_dict(self.upper_online.state_dict())
        self.lower_target.load_state_dict(self.lower_online.state_dict())

    def update(self) -> UpdateReport | None:
        batch_size = self.profile.training.batch_size
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        rule_actions = torch.tensor(
            [item.rule_action for item in batch],
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        reward_ids = torch.tensor(
            [item.reward_id for item in batch],
            dtype=torch.int64,
            device=self.device,
        )
        dones = torch.tensor(
            [item.done for item in batch],
            dtype=torch.bool,
            device=self.device,
        )
        if self.global_update_step % self.profile.training.target_update_steps == 0:
            self.sync_targets()

        upper_predictions = self.upper_online(states).gather(
            1, reward_ids.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            upper_targets = value_target(
                rewards,
                dones,
                self.upper_online(next_states),
                self.upper_target(next_states),
                self.profile.training.gamma,
                double_dqn=self.double_dqn,
            )
        upper_loss = self._loss(upper_predictions, upper_targets)
        self.upper_optimizer.zero_grad()
        upper_loss.backward()
        self.upper_optimizer.step()

        with torch.no_grad():
            current_upper_q = self.upper_online(states)
            next_upper_q = self.upper_online(next_states)
            next_reward_ids = torch.argmax(next_upper_q, dim=1)
            current_context = build_lower_context(
                states,
                current_upper_q,
                reward_ids,
                self.profile.architecture.lower_context,
            )
            next_context = build_lower_context(
                next_states,
                next_upper_q,
                next_reward_ids,
                self.profile.architecture.lower_context,
            )
        lower_predictions = self.lower_online(current_context).gather(
            1, rule_actions.unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            lower_targets = value_target(
                rewards,
                dones,
                self.lower_online(next_context),
                self.lower_target(next_context),
                self.profile.training.gamma,
                double_dqn=self.double_dqn,
            )
        lower_loss = self._loss(lower_predictions, lower_targets)
        self.lower_optimizer.zero_grad()
        lower_loss.backward()
        self.lower_optimizer.step()

        self.global_update_step += 1
        return UpdateReport(
            upper_loss=float(upper_loss.detach().cpu().item()),
            lower_loss=float(lower_loss.detach().cpu().item()),
            epsilon=self.epsilon,
            global_update_step=self.global_update_step,
        )
