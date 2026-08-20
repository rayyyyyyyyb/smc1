from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    rule_action: int
    reward: float
    next_state: np.ndarray
    reward_id: int
    done: bool

    def __post_init__(self) -> None:
        for name, value in (("state", self.state), ("next_state", self.next_state)):
            try:
                copied = np.array(value, dtype=np.float32, copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be a finite float32 vector with shape (6,)"
                ) from exc
            if copied.shape != (6,) or not np.all(np.isfinite(copied)):
                raise ValueError(
                    f"{name} must be a finite float32 vector with shape (6,)"
                )
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if type(self.rule_action) is not int or not 0 <= self.rule_action < 9:
            raise ValueError("rule_action must be in [0, 8]")
        if type(self.reward_id) is not int or self.reward_id not in (0, 1):
            raise ValueError("reward_id must be 0 or 1")
        if type(self.done) is not bool:
            raise ValueError("done must be a boolean")
        if type(self.reward) not in (int, float):
            raise ValueError("reward must be finite")
        try:
            reward = float(self.reward)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reward must be finite") from exc
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "reward", reward)


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be positive and seed non-negative")
        if type(seed) is not int or seed < 0:
            raise ValueError("capacity must be positive and seed non-negative")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)
        self.capacity = capacity

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        if type(transition) is not Transition:
            raise TypeError("replay items must be Transition records")
        self._items.append(transition)

    def clear(self) -> None:
        self._items.clear()

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or batch_size > len(self._items)
        ):
            raise ValueError("invalid replay sample size")
        return tuple(self._rng.sample(tuple(self._items), batch_size))

    def state_dict(self) -> dict[str, object]:
        serialized_items = tuple(
            {
                "state": item.state.tolist(),
                "rule_action": item.rule_action,
                "reward": item.reward,
                "next_state": item.next_state.tolist(),
                "reward_id": item.reward_id,
                "done": item.done,
            }
            for item in self._items
        )
        return {
            "schema_version": 1,
            "capacity": self.capacity,
            "rng_state": self._rng.getstate(),
            "items": serialized_items,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("invalid replay-buffer checkpoint payload")
        if set(state) != {"schema_version", "capacity", "rng_state", "items"}:
            raise ValueError("invalid replay-buffer checkpoint payload")
        if state.get("schema_version") != 1 or state.get("capacity") != self.capacity:
            raise ValueError("incompatible replay-buffer checkpoint")
        raw_items = state.get("items")
        rng_state = state.get("rng_state")
        if not isinstance(raw_items, tuple) or not isinstance(rng_state, tuple):
            raise ValueError("invalid replay-buffer checkpoint payload")
        if len(raw_items) > self.capacity:
            raise ValueError("replay-buffer checkpoint exceeds capacity")
        restored: list[Transition] = []
        expected_item_keys = {
            "state",
            "rule_action",
            "reward",
            "next_state",
            "reward_id",
            "done",
        }
        for raw_item in raw_items:
            if not isinstance(raw_item, dict) or set(raw_item) != expected_item_keys:
                raise ValueError("invalid replay-buffer item")
            try:
                restored.append(
                    Transition(
                        state=np.asarray(raw_item["state"], dtype=np.float32),
                        rule_action=cast(int, raw_item["rule_action"]),
                        reward=cast(float, raw_item["reward"]),
                        next_state=np.asarray(
                            raw_item["next_state"], dtype=np.float32
                        ),
                        reward_id=cast(int, raw_item["reward_id"]),
                        done=cast(bool, raw_item["done"]),
                    )
                )
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError("invalid replay-buffer item") from exc
        probe_rng = random.Random()
        try:
            probe_rng.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid replay-buffer RNG state") from exc
        self._items.clear()
        self._items.extend(restored)
        self._rng.setstate(rng_state)
