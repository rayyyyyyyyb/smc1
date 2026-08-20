from __future__ import annotations

from enum import IntEnum

from smc_repro.config import ReproductionProfile
from smc_repro.observations import ScheduleObservation


class RewardMode(IntEnum):
    TARDINESS = 0
    UTILIZATION = 1


def legacy_tardiness_reward(previous: float, current: float) -> int:
    if current < previous:
        return 1
    if current < previous * 1.1:
        return 0
    return -1


def paper_tardiness_reward(previous: float, current: float) -> int:
    return 1 if current < previous else -1


def legacy_utilization_reward(previous: float, current: float) -> int:
    if current > previous:
        return 1
    if current > 0.9 * previous:
        return 0
    return -1


def paper_utilization_reward(previous: float, current: float) -> int:
    if current > previous:
        return 1
    if 0.9 * previous <= current <= previous:
        return 0
    return -1


def transition_reward(
    profile: ReproductionProfile,
    mode: RewardMode,
    previous: ScheduleObservation,
    current: ScheduleObservation,
) -> int:
    if mode is RewardMode.TARDINESS:
        previous_value = previous.tr_ave
        current_value = current.tr_ave
        semantics = profile.reward.tardiness_mode
        if semantics == "legacy":
            return legacy_tardiness_reward(previous_value, current_value)
        if semantics == "paper":
            return paper_tardiness_reward(previous_value, current_value)
    elif mode is RewardMode.UTILIZATION:
        previous_value = previous.u_ave
        current_value = current.u_ave
        semantics = profile.reward.utilization_mode
        if semantics == "legacy":
            return legacy_utilization_reward(previous_value, current_value)
        if semantics == "paper":
            return paper_utilization_reward(previous_value, current_value)
    raise ValueError(f"unsupported reward mode or profile semantics: {mode!r}")
