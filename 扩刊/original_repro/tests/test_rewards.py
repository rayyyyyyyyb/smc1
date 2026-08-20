from collections.abc import Callable
from pathlib import Path

import pytest

from smc_repro.config import load_profile
from smc_repro.observations import ScheduleObservation
from smc_repro.rewards import (
    RewardMode,
    legacy_tardiness_reward,
    legacy_utilization_reward,
    paper_tardiness_reward,
    paper_utilization_reward,
    transition_reward,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


@pytest.mark.parametrize(
    ("reward", "previous", "current", "expected"),
    [
        (legacy_tardiness_reward, 10.0, 9.0, 1),
        (legacy_tardiness_reward, 10.0, 10.0, 0),
        (legacy_tardiness_reward, 10.0, 10.999, 0),
        (legacy_tardiness_reward, 10.0, 11.0, -1),
        (legacy_tardiness_reward, 0.0, 0.0, -1),
        (paper_tardiness_reward, 10.0, 9.0, 1),
        (paper_tardiness_reward, 10.0, 10.0, -1),
        (paper_tardiness_reward, 10.0, 11.0, -1),
        (paper_tardiness_reward, 0.0, 0.0, -1),
        (legacy_utilization_reward, 10.0, 11.0, 1),
        (legacy_utilization_reward, 10.0, 10.0, 0),
        (legacy_utilization_reward, 10.0, 9.001, 0),
        (legacy_utilization_reward, 10.0, 9.0, -1),
        (legacy_utilization_reward, 0.0, 0.0, -1),
        (paper_utilization_reward, 10.0, 11.0, 1),
        (paper_utilization_reward, 10.0, 10.0, 0),
        (paper_utilization_reward, 10.0, 9.0, 0),
        (paper_utilization_reward, 10.0, 8.999, -1),
        (paper_utilization_reward, 0.0, 0.0, 0),
    ],
)
def test_reward_zero_equality_and_threshold_matrix(
    reward: Callable[[float, float], int],
    previous: float,
    current: float,
    expected: int,
) -> None:
    assert reward(previous, current) == expected


def test_dispatcher_uses_profile_reward_semantics_for_both_modes() -> None:
    previous = ScheduleObservation(0.0, 0.0, 0.0, 0.0, 2.0, 0.0)
    current = ScheduleObservation(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    zero = ScheduleObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    legacy = load_profile(CONFIG_ROOT / "legacy_snapshot.yaml")
    paper = load_profile(CONFIG_ROOT / "paper_repro.yaml")

    assert transition_reward(legacy, RewardMode.TARDINESS, previous, current) == 1
    assert transition_reward(paper, RewardMode.TARDINESS, previous, current) == 1
    assert transition_reward(legacy, RewardMode.UTILIZATION, zero, zero) == -1
    assert transition_reward(paper, RewardMode.UTILIZATION, zero, zero) == 0


def test_dispatcher_reads_named_attributes_without_vector_magic_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = ScheduleObservation(10.0, 0.0, 0.0, 0.0, 2.0, 0.0)
    current = ScheduleObservation(0.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    paper = load_profile(CONFIG_ROOT / "paper_repro.yaml")

    def fail_vector(
        self: ScheduleObservation, order: tuple[str, ...]
    ) -> None:
        raise AssertionError(f"reward dispatcher vectorized {self!r} with {order!r}")

    monkeypatch.setattr(ScheduleObservation, "vector", fail_vector)

    assert transition_reward(paper, RewardMode.TARDINESS, previous, current) == 1
    assert transition_reward(paper, RewardMode.UTILIZATION, previous, current) == 1
