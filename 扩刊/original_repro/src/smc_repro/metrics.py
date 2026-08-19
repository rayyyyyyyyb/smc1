from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from smc_repro.schemas import InstanceSpec, IntervalType, ScheduleInterval
from smc_repro.validator import validate_schedule


@dataclass(frozen=True)
class ScheduleMetrics:
    makespan: float
    paper_trave: float
    total_tardiness: float
    mean_tardiness: float
    weighted_tardiness: float
    tardy_rate: float
    paper_uave: float
    standard_utilization: float
    availability_adjusted_utilization: float
    total_process_time: float
    total_setup_time: float
    total_pm_time: float
    total_cm_time: float
    pm_count: int
    cm_count: int


def compute_schedule_metrics(
    instance: InstanceSpec,
    intervals: Iterable[ScheduleInterval],
) -> ScheduleMetrics:
    schedule = tuple(intervals)
    report = validate_schedule(instance, schedule, require_complete=True)
    if not report.ok:
        raise ValueError("invalid or incomplete schedule:\n" + "\n".join(report.errors))

    process_by_operation = {
        (interval.job_id, interval.op_id): interval
        for interval in schedule
        if interval.interval_type is IntervalType.PROCESS
    }
    job_completions: dict[int, float] = {}
    process_by_job: dict[int, float] = {}
    tardiness_by_job: dict[int, float] = {}

    for job in instance.jobs:
        final_key = (job.job_id, len(job.operations) - 1)
        final_interval = process_by_operation[final_key]
        completion = final_interval.end
        process_work = sum(
            process_by_operation[(job.job_id, operation.op_id)].duration
            for operation in job.operations
        )
        job_completions[job.job_id] = completion
        process_by_job[job.job_id] = process_work
        tardiness_by_job[job.job_id] = max(0.0, completion - job.due_date)

    makespan = max(job_completions.values())
    total_tardiness = sum(tardiness_by_job.values())
    mean_tardiness = total_tardiness / len(instance.jobs)
    weighted_tardiness = sum(
        job.weight * tardiness_by_job[job.job_id] for job in instance.jobs
    )
    tardy_rate = sum(value > 1e-9 for value in tardiness_by_job.values()) / len(instance.jobs)
    paper_trave = sum(
        tardiness_by_job[job.job_id] / max(process_by_job[job.job_id], 1e-12)
        for job in instance.jobs
    ) / len(instance.jobs)

    process_time = sum(
        interval.duration
        for interval in schedule
        if interval.interval_type is IntervalType.PROCESS
    )
    setup_time = sum(
        interval.duration
        for interval in schedule
        if interval.interval_type is IntervalType.SETUP
    )
    pm_intervals = tuple(
        interval for interval in schedule if interval.interval_type is IntervalType.PM
    )
    cm_intervals = tuple(
        interval for interval in schedule if interval.interval_type is IntervalType.CM
    )
    pm_time = sum(interval.duration for interval in pm_intervals)
    cm_time = sum(interval.duration for interval in cm_intervals)

    per_machine_paper_utilization: list[float] = []
    for machine in instance.machines:
        machine_process = tuple(
            interval
            for interval in schedule
            if interval.machine_id == machine.machine_id
            and interval.interval_type is IntervalType.PROCESS
        )
        busy = sum(interval.duration for interval in machine_process)
        final_process_end = max((interval.end for interval in machine_process), default=0.0)
        per_machine_paper_utilization.append(
            0.0 if final_process_end <= 0.0 else busy / final_process_end
        )
    paper_uave = sum(per_machine_paper_utilization) / len(instance.machines)

    total_capacity = len(instance.machines) * makespan
    standard_utilization = 0.0 if total_capacity <= 0 else process_time / total_capacity
    available_capacity = total_capacity - setup_time - pm_time - cm_time
    availability_adjusted_utilization = (
        0.0 if available_capacity <= 0 else process_time / available_capacity
    )

    return ScheduleMetrics(
        makespan=makespan,
        paper_trave=paper_trave,
        total_tardiness=total_tardiness,
        mean_tardiness=mean_tardiness,
        weighted_tardiness=weighted_tardiness,
        tardy_rate=tardy_rate,
        paper_uave=paper_uave,
        standard_utilization=standard_utilization,
        availability_adjusted_utilization=availability_adjusted_utilization,
        total_process_time=process_time,
        total_setup_time=setup_time,
        total_pm_time=pm_time,
        total_cm_time=cm_time,
        pm_count=len(pm_intervals),
        cm_count=len(cm_intervals),
    )
