from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from smc_repro.schemas import InstanceSpec, IntervalType, ScheduleInterval


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...]


def validate_schedule(
    instance: InstanceSpec,
    intervals: Iterable[ScheduleInterval],
    *,
    require_complete: bool,
) -> ValidationReport:
    schedule = tuple(intervals)
    errors: list[str] = []
    machine_count = len(instance.machines)
    by_machine: dict[int, list[ScheduleInterval]] = defaultdict(list)
    process_by_operation: dict[tuple[int, int], list[ScheduleInterval]] = defaultdict(list)

    for interval in schedule:
        if not 0 <= interval.machine_id < machine_count:
            errors.append(f"invalid machine id {interval.machine_id}")
            continue
        if interval.end < interval.start:
            errors.append(f"negative interval duration: {interval}")
        by_machine[interval.machine_id].append(interval)

        if interval.interval_type is not IntervalType.PROCESS:
            continue
        assert interval.job_id is not None and interval.op_id is not None
        key = (interval.job_id, interval.op_id)
        process_by_operation[key].append(interval)
        if not 0 <= interval.job_id < len(instance.jobs):
            errors.append(f"unknown job for PROCESS interval: {key}")
            continue
        job = instance.jobs[interval.job_id]
        if not 0 <= interval.op_id < len(job.operations):
            errors.append(f"unknown operation for PROCESS interval: {key}")
            continue
        operation = job.operations[interval.op_id]
        if interval.machine_id not in operation.eligible_machines:
            errors.append(f"ineligible machine {interval.machine_id} for operation {key}")
        else:
            nominal_duration = operation.processing_time(interval.machine_id)
            if interval.duration + 1e-9 < nominal_duration:
                errors.append(
                    f"processing duration for operation {key} on machine "
                    f"{interval.machine_id} is shorter than nominal: "
                    f"{interval.duration} < {nominal_duration}"
                )
        if interval.start < job.arrival_time - 1e-9:
            errors.append(
                f"arrival violation for job {job.job_id}: {interval.start} < {job.arrival_time}"
            )

    for machine_id, machine_intervals in by_machine.items():
        ordered = sorted(
            machine_intervals,
            key=lambda item: (item.start, item.end, item.interval_type.value),
        )
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.overlaps(current):
                errors.append(
                    f"machine {machine_id} overlap: {previous} vs {current}"
                )

    expected = {
        (job.job_id, operation.op_id)
        for job in instance.jobs
        for operation in job.operations
    }
    for key in sorted(expected):
        count = len(process_by_operation.get(key, ()))
        if count > 1:
            errors.append(f"operation {key} processed {count} times")
        if require_complete and count == 0:
            errors.append(f"missing operation {key}")

    for key in sorted(set(process_by_operation) - expected):
        errors.append(f"unexpected operation {key}")

    unique_intervals = {
        key: values[0]
        for key, values in process_by_operation.items()
        if len(values) == 1 and key in expected
    }
    for job in instance.jobs:
        for op_id in range(1, len(job.operations)):
            previous_key = (job.job_id, op_id - 1)
            current_key = (job.job_id, op_id)
            if previous_key in unique_intervals and current_key in unique_intervals:
                previous = unique_intervals[previous_key]
                current = unique_intervals[current_key]
                if current.start < previous.end - 1e-9:
                    errors.append(
                        f"precedence violation {previous_key}->{current_key}: "
                        f"{current.start} < {previous.end}"
                    )

    return ValidationReport(ok=not errors, errors=tuple(errors))
