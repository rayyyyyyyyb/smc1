from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class IntervalType(StrEnum):
    PROCESS = "PROCESS"
    SETUP = "SETUP"
    PM = "PM"
    CM = "CM"


@dataclass(frozen=True)
class OperationSpec:
    job_id: int
    op_id: int
    proc_times: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if self.job_id < 0 or self.op_id < 0:
            raise ValueError("job_id and op_id must be non-negative")
        eligible = [value for value in self.proc_times if value is not None]
        if not eligible:
            raise ValueError("operation must have at least one eligible machine")
        if any(not math.isfinite(value) for value in eligible):
            raise ValueError("eligible processing times must be finite")
        if any(value is not None and value <= 0 for value in self.proc_times):
            raise ValueError("eligible processing times must be positive")

    @property
    def eligible_machines(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.proc_times) if value is not None)

    def processing_time(self, machine_id: int) -> float:
        if machine_id < 0 or machine_id >= len(self.proc_times):
            raise KeyError(f"unknown machine {machine_id}")
        value = self.proc_times[machine_id]
        if value is None:
            raise KeyError(
                f"machine {machine_id} is not eligible for operation {(self.job_id, self.op_id)}"
            )
        return float(value)


@dataclass(frozen=True)
class JobSpec:
    job_id: int
    arrival_time: float
    due_date: float
    urgency: int
    operations: tuple[OperationSpec, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("job_id must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (self.arrival_time, self.due_date, self.weight)
        ):
            raise ValueError("arrival_time, due_date, and weight must be finite")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.due_date < self.arrival_time:
            raise ValueError("due_date must not precede arrival_time")
        if self.urgency not in (1, 2, 3):
            raise ValueError("urgency must be 1, 2, or 3")
        if not self.operations:
            raise ValueError("job must contain at least one operation")
        for expected_op_id, operation in enumerate(self.operations):
            if operation.job_id != self.job_id or operation.op_id != expected_op_id:
                raise ValueError("operations must use matching sequential job/op ids")
        if self.weight <= 0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True)
class MachineSpec:
    machine_id: int
    setup_time: float
    cm_duration: float
    eta: float = 500.0
    beta: float = 2.0
    pm_duration_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if not all(
            math.isfinite(value)
            for value in (
                self.setup_time,
                self.cm_duration,
                self.eta,
                self.beta,
                self.pm_duration_ratio,
            )
        ):
            raise ValueError("machine duration and Weibull fields must be finite")
        if self.setup_time < 0 or self.cm_duration <= 0:
            raise ValueError("invalid setup or corrective-maintenance duration")
        if self.eta <= 0 or self.beta <= 0:
            raise ValueError("Weibull eta and beta must be positive")
        if not 0 < self.pm_duration_ratio <= 1:
            raise ValueError("pm_duration_ratio must be in (0, 1]")

    @property
    def pm_duration(self) -> float:
        return self.cm_duration * self.pm_duration_ratio


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    instance_seed: int
    failure_seed: int
    jobs: tuple[JobSpec, ...]
    machines: tuple[MachineSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if self.instance_seed < 0 or self.failure_seed < 0:
            raise ValueError("seeds must be non-negative")
        if not self.jobs or not self.machines:
            raise ValueError("instance must contain jobs and machines")
        if tuple(job.job_id for job in self.jobs) != tuple(range(len(self.jobs))):
            raise ValueError("job ids must be contiguous from zero")
        if tuple(machine.machine_id for machine in self.machines) != tuple(
            range(len(self.machines))
        ):
            raise ValueError("machine ids must be contiguous from zero")
        machine_count = len(self.machines)
        for job in self.jobs:
            for operation in job.operations:
                if len(operation.proc_times) != machine_count:
                    raise ValueError("processing-time vectors must match machine count")


@dataclass(frozen=True)
class ScheduleInterval:
    machine_id: int
    start: float
    end: float
    interval_type: IntervalType
    job_id: int | None = None
    op_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("interval start and end must be finite")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        if self.interval_type is IntervalType.PROCESS:
            if self.job_id is None or self.op_id is None:
                raise ValueError("PROCESS interval requires job_id and op_id")
            if self.end <= self.start:
                raise ValueError("PROCESS interval duration must be positive")
        elif self.job_id is not None or self.op_id is not None:
            raise ValueError("non-PROCESS intervals must not carry job_id/op_id")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: ScheduleInterval, eps: float = 1e-9) -> bool:
        if self.machine_id != other.machine_id:
            return False
        return self.start < other.end - eps and other.start < self.end - eps
