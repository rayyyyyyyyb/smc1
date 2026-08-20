from __future__ import annotations

import math
import random
from enum import StrEnum
from typing import cast

import numpy as np


class TabularAlgorithm(StrEnum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"


class TabularRewardProtocol(StrEnum):
    LEGACY_JOINT = "legacy_joint"
    FIXED_TARDINESS = "fixed_tardiness"
    FIXED_UTILIZATION = "fixed_utilization"


class TieBreakMode(StrEnum):
    CORRECTED_TIE_BREAK = "corrected_tie_break"
    FIRST_ARGMAX = "first_argmax"


def _finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite")
    try:
        converted = float(cast(int | float, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def discretize_six_feature_state(state: np.ndarray) -> int:
    try:
        values = np.asarray(state, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "tabular state must be a finite vector with shape (6,)"
        ) from exc
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("tabular state must be a finite vector with shape (6,)")
    mean_value = float(np.mean(np.clip(values, 0.0, 1.0)))
    return min(9, int(mean_value * 10.0))


class TabularAgent:
    def __init__(
        self,
        algorithm: TabularAlgorithm,
        *,
        seed: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        minimum_epsilon: float = 0.01,
        tie_break: TieBreakMode = TieBreakMode.CORRECTED_TIE_BREAK,
    ) -> None:
        if type(algorithm) is not TabularAlgorithm:
            raise ValueError("algorithm must be a TabularAlgorithm")
        if type(tie_break) is not TieBreakMode:
            raise ValueError("tie_break must be a TieBreakMode")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        learning_rate_value = _finite_float(learning_rate, "learning_rate")
        gamma_value = _finite_float(gamma, "gamma")
        epsilon_value = _finite_float(epsilon, "epsilon")
        epsilon_decay_value = _finite_float(epsilon_decay, "epsilon_decay")
        minimum_epsilon_value = _finite_float(
            minimum_epsilon, "minimum_epsilon"
        )
        if not 0.0 < learning_rate_value <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= gamma_value <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= epsilon_value <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0.0 < epsilon_decay_value <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if not 0.0 <= minimum_epsilon_value <= 1.0:
            raise ValueError("minimum_epsilon must be in [0, 1]")
        self.algorithm = algorithm
        self.tie_break = tie_break
        self.learning_rate = learning_rate_value
        self.gamma = gamma_value
        self.epsilon = epsilon_value
        self.epsilon_decay = epsilon_decay_value
        self.minimum_epsilon = minimum_epsilon_value
        self.q_table = np.zeros((10, 9), dtype=np.float64)
        self._rng = random.Random(seed)

    @staticmethod
    def _validate_action(action: int, name: str = "action") -> None:
        if type(action) is not int or not 0 <= action < 9:
            raise ValueError(f"{name} must be an integer in [0, 8]")

    def select_action(self, state: np.ndarray, *, training: bool) -> int:
        if type(training) is not bool:
            raise ValueError("training must be a boolean")
        state_index = discretize_six_feature_state(np.asarray(state))
        if training and self._rng.random() < self.epsilon:
            return self._rng.randrange(9)
        row = self.q_table[state_index]
        if self.tie_break is TieBreakMode.FIRST_ARGMAX:
            return int(np.argmax(row))
        maximal_actions = np.flatnonzero(row == np.max(row))
        return int(self._rng.choice(maximal_actions.tolist()))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        next_action: int | None = None,
    ) -> float:
        self._validate_action(action)
        if type(done) is not bool:
            raise ValueError("done must be a boolean")
        reward_value = _finite_float(reward, "reward")
        state_index = discretize_six_feature_state(np.asarray(state))
        next_state_index = discretize_six_feature_state(np.asarray(next_state))
        if done:
            bootstrap = 0.0
        elif self.algorithm is TabularAlgorithm.Q_LEARNING:
            bootstrap = float(np.max(self.q_table[next_state_index]))
        elif self.algorithm is TabularAlgorithm.SARSA:
            if next_action is None:
                raise ValueError("SARSA requires next_action for a nonterminal update")
            self._validate_action(next_action, "next_action")
            bootstrap = float(self.q_table[next_state_index, next_action])
        else:
            raise AssertionError(f"unsupported tabular algorithm: {self.algorithm}")
        old_value = float(self.q_table[state_index, action])
        target = reward_value + self.gamma * bootstrap
        updated = old_value + self.learning_rate * (target - old_value)
        self.q_table[state_index, action] = updated
        return updated

    def decay_epsilon(self) -> float:
        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon * self.epsilon_decay,
        )
        return self.epsilon

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "algorithm": self.algorithm.value,
            "tie_break": self.tie_break.value,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "minimum_epsilon": self.minimum_epsilon,
            "q_table": self.q_table.tolist(),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected_fields = {
            "schema_version",
            "algorithm",
            "tie_break",
            "learning_rate",
            "gamma",
            "epsilon",
            "epsilon_decay",
            "minimum_epsilon",
            "q_table",
            "rng_state",
        }
        if not isinstance(state, dict) or set(state) != expected_fields:
            raise ValueError("invalid tabular state dictionary")
        if state["schema_version"] != 1:
            raise ValueError("incompatible tabular schema")
        if state["algorithm"] != self.algorithm.value:
            raise ValueError("incompatible tabular algorithm")
        if state["tie_break"] != self.tie_break.value:
            raise ValueError("incompatible tabular tie mode")
        for field, expected in (
            ("learning_rate", self.learning_rate),
            ("gamma", self.gamma),
            ("epsilon_decay", self.epsilon_decay),
            ("minimum_epsilon", self.minimum_epsilon),
        ):
            try:
                observed = _finite_float(state[field], f"tabular {field}")
            except ValueError as exc:
                raise ValueError(f"incompatible tabular {field}") from exc
            if observed != expected:
                raise ValueError(f"incompatible tabular {field}")
        try:
            epsilon = _finite_float(state["epsilon"], "tabular epsilon")
        except ValueError as exc:
            raise ValueError("invalid tabular epsilon") from exc
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("invalid tabular epsilon")
        try:
            q_table = np.asarray(state["q_table"], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid tabular Q table") from exc
        if q_table.shape != (10, 9) or not np.all(np.isfinite(q_table)):
            raise ValueError("invalid tabular Q table")
        rng_state = state["rng_state"]
        if not isinstance(rng_state, tuple):
            raise ValueError("invalid tabular RNG state")
        probe = random.Random()
        try:
            probe.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid tabular RNG state") from exc
        self.q_table = np.array(q_table, copy=True)
        self.epsilon = epsilon
        self._rng.setstate(rng_state)
