from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from smc_repro.schemas import InstanceSpec, IntervalType, ScheduleInterval


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...]


def _finite_metadata_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


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

        machine = instance.machines[interval.machine_id]
        if interval.interval_type is IntervalType.SETUP:
            expected_duration = machine.setup_time
            label = "setup duration"
        elif interval.interval_type is IntervalType.PM:
            expected_duration = machine.pm_duration
            label = "PM duration"
        elif interval.interval_type is IntervalType.CM:
            expected_duration = machine.cm_duration
            label = "CM duration"
        else:
            expected_duration = None
            label = ""

        if (
            expected_duration is not None
            and abs(interval.duration - expected_duration) > 1e-9
        ):
            errors.append(
                f"{label} mismatch on machine {interval.machine_id}: "
                f"{interval.duration} != {expected_duration}"
            )

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
            has_nominal_metadata = "nominal_processing_time" in interval.metadata
            has_degradation_metadata = "degradation_factor" in interval.metadata
            if has_nominal_metadata != has_degradation_metadata:
                errors.append(
                    f"PROCESS duration metadata for operation {key} must contain both "
                    "nominal_processing_time and degradation_factor"
                )
            elif has_nominal_metadata:
                metadata_nominal = interval.metadata["nominal_processing_time"]
                metadata_degradation = interval.metadata["degradation_factor"]
                nominal_number = _finite_metadata_float(metadata_nominal)
                degradation_number = _finite_metadata_float(metadata_degradation)
                if nominal_number is None or degradation_number is None:
                    errors.append(
                        f"PROCESS duration metadata for operation {key} must use "
                        "finite non-boolean numbers"
                    )
                else:
                    if nominal_number <= 0.0 or degradation_number <= 0.0:
                        errors.append(
                            f"PROCESS duration metadata for operation {key} must use "
                            "finite non-boolean numbers greater than zero"
                        )
                    else:
                        expected_process_duration = (
                            nominal_number * degradation_number
                        )
                        if not math.isclose(
                            nominal_number,
                            nominal_duration,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        ):
                            errors.append(
                                f"nominal processing metadata mismatch for operation {key}: "
                                f"{metadata_nominal} != {nominal_duration}"
                            )
                        if not math.isfinite(expected_process_duration):
                            errors.append(
                                f"PROCESS duration metadata product for operation {key} "
                                "must be finite"
                            )
                        elif not math.isclose(
                            interval.duration,
                            expected_process_duration,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        ):
                            errors.append(
                                f"PROCESS duration metadata mismatch for operation {key}: "
                                f"{interval.duration} != {expected_process_duration}"
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

    if require_complete:
        process_horizon = max(
            (
                interval.end
                for interval in schedule
                if interval.interval_type is IntervalType.PROCESS
            ),
            default=0.0,
        )
        for interval in schedule:
            if (
                interval.interval_type is not IntervalType.PROCESS
                and interval.end > process_horizon + 1e-9
            ):
                errors.append(
                    f"{interval.interval_type.value} interval on machine "
                    f"{interval.machine_id} ends after final process horizon: "
                    f"{interval.end} > {process_horizon}"
                )

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
