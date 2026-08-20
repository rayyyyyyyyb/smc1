from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from smc_repro.seeding import keyed_uniform


class JobSelector(StrEnum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    FIFO = "FIFO"
    EDD = "EDD"
    MRT = "MRT"
    SPT = "SPT"
    LPT = "LPT"


class MachineSelector(StrEnum):
    EARLIEST_START = "earliest_start"
    EARLIEST_COMPLETION = "earliest_completion"
    RANDOM = "random"


def _require_non_negative_id(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class JobRuleView:
    job_id: int
    op_id: int
    arrival_time: float
    due_date: float
    urgency: int
    decision_time: float
    latest_process_end: float
    operation_count: int
    completed_operation_count: int
    completion_ratio_by_count: float
    completion_ratio_by_work: float
    processed_work: float
    remaining_nominal_work: float
    next_operation_mean_processing_time: float

    def __post_init__(self) -> None:
        _require_non_negative_id("job_id", self.job_id)
        _require_non_negative_id("op_id", self.op_id)
        if not isinstance(self.urgency, int) or isinstance(self.urgency, bool):
            raise ValueError("urgency must be an integer in 1..3")
        if self.urgency not in {1, 2, 3}:
            raise ValueError("urgency must be an integer in 1..3")
        if (
            not isinstance(self.operation_count, int)
            or isinstance(self.operation_count, bool)
            or self.operation_count <= 0
        ):
            raise ValueError("operation_count must be a positive integer")
        if (
            not isinstance(self.completed_operation_count, int)
            or isinstance(self.completed_operation_count, bool)
            or not 0 <= self.completed_operation_count <= self.operation_count
        ):
            raise ValueError(
                "completed_operation_count must be an integer between zero and operation_count"
            )
        for name in (
            "arrival_time",
            "due_date",
            "decision_time",
            "latest_process_end",
            "completion_ratio_by_count",
            "completion_ratio_by_work",
            "processed_work",
            "remaining_nominal_work",
            "next_operation_mean_processing_time",
        ):
            _require_finite(name, getattr(self, name))
        for name in ("completion_ratio_by_count", "completion_ratio_by_work"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "processed_work",
            "remaining_nominal_work",
            "next_operation_mean_processing_time",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PairRuleView:
    job_id: int
    op_id: int
    machine_id: int
    earliest_start: float
    estimated_completion: float

    def __post_init__(self) -> None:
        _require_non_negative_id("job_id", self.job_id)
        _require_non_negative_id("op_id", self.op_id)
        _require_non_negative_id("machine_id", self.machine_id)
        _require_finite("earliest_start", self.earliest_start)
        _require_finite("estimated_completion", self.estimated_completion)


@dataclass(frozen=True)
class RuleContext:
    instance_id: str
    decision_index: int
    policy_seed: int
    jobs: tuple[JobRuleView, ...]
    pairs: tuple[PairRuleView, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be a non-empty string")
        if type(self.decision_index) is not int or self.decision_index < 0:
            raise ValueError("decision_index must be a non-negative integer")
        if type(self.policy_seed) is not int or self.policy_seed < 0:
            raise ValueError("policy_seed must be a non-negative integer")
        try:
            jobs = tuple(self.jobs)
            pairs = tuple(self.pairs)
        except TypeError:
            raise TypeError("jobs and pairs must be finite sequences") from None
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "pairs", pairs)
        if not jobs or not pairs:
            raise ValueError("rule context must contain ready jobs and legal pairs")
        if any(type(job) is not JobRuleView for job in jobs):
            raise TypeError("jobs must contain only JobRuleView instances")
        if any(type(pair) is not PairRuleView for pair in pairs):
            raise TypeError("pairs must contain only PairRuleView instances")
        job_keys = {(job.job_id, job.op_id) for job in jobs}
        if len(job_keys) != len(jobs):
            raise ValueError("ready job views must be unique")
        if any((pair.job_id, pair.op_id) not in job_keys for pair in pairs):
            raise ValueError("every pair must refer to a ready job view")
        pair_keys = {(pair.job_id, pair.op_id, pair.machine_id) for pair in pairs}
        if len(pair_keys) != len(pairs):
            raise ValueError("legal machine pairs must be unique")


@dataclass(frozen=True)
class DispatchDecision:
    job_id: int
    op_id: int
    machine_id: int
    rule_name: str


def _finite_job_score(job: JobRuleView, score: Callable[[JobRuleView], float]) -> float:
    value = float(score(job))
    if not math.isfinite(value):
        raise ValueError("job selector score must be finite")
    return value


def argmin_job(
    jobs: tuple[JobRuleView, ...],
    score: Callable[[JobRuleView], float],
) -> JobRuleView:
    return min(jobs, key=lambda job: (_finite_job_score(job, score), job.job_id))


def argmax_job(
    jobs: tuple[JobRuleView, ...],
    score: Callable[[JobRuleView], float],
) -> JobRuleView:
    return min(jobs, key=lambda job: (-_finite_job_score(job, score), job.job_id))


_ItemT = TypeVar("_ItemT")


def keyed_choice(items: tuple[_ItemT, ...], base_seed: int, *keys: object) -> _ItemT:
    if not items:
        raise ValueError("cannot select from an empty sequence")
    value = keyed_uniform(base_seed, *keys)
    index = min(int(value * len(items)), len(items) - 1)
    return items[index]


def pairs_for_job(context: RuleContext, job: JobRuleView) -> tuple[PairRuleView, ...]:
    pairs = tuple(
        pair
        for pair in context.pairs
        if pair.job_id == job.job_id and pair.op_id == job.op_id
    )
    if not pairs:
        raise ValueError(f"no legal machine pair for job {job.job_id} operation {job.op_id}")
    return tuple(sorted(pairs, key=lambda pair: pair.machine_id))


def select_machine(
    context: RuleContext,
    job: JobRuleView,
    selector: MachineSelector,
    *,
    namespace: str,
) -> PairRuleView:
    pairs = pairs_for_job(context, job)
    if selector is MachineSelector.EARLIEST_START:
        return min(pairs, key=lambda pair: (pair.earliest_start, pair.machine_id))
    if selector is MachineSelector.EARLIEST_COMPLETION:
        return min(pairs, key=lambda pair: (pair.estimated_completion, pair.machine_id))
    if selector is MachineSelector.RANDOM:
        selected = keyed_choice(
            pairs,
            context.policy_seed,
            "rule_machine",
            namespace,
            context.instance_id,
            context.decision_index,
            job.job_id,
            job.op_id,
        )
        if not isinstance(selected, PairRuleView):
            raise TypeError("keyed machine selection returned an invalid object")
        return selected
    raise AssertionError(f"unsupported machine selector: {selector}")
