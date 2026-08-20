from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smc_repro.config import ReproductionProfile, TRFeatureMode
from smc_repro.runtime import ScheduleRuntime
from smc_repro.schemas import IntervalType

FEATURE_NAMES = (
    "crj_ave",
    "crj_std",
    "u_ave",
    "u_std",
    "tr_ave",
    "tr_std",
)


@dataclass(frozen=True)
class ScheduleObservation:
    crj_ave: float
    crj_std: float
    u_ave: float
    u_std: float
    tr_ave: float
    tr_std: float

    def vector(self, order: tuple[str, ...]) -> np.ndarray:
        if len(order) != 6 or set(order) != set(FEATURE_NAMES):
            raise ValueError("state feature order must contain each known feature exactly once")
        values = np.asarray([getattr(self, name) for name in order], dtype=np.float32)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("state vector must contain six finite values")
        return values


def _mean_and_population_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(standard_deviation):
        raise ValueError("observation statistics must be finite")
    return mean, standard_deviation


def compute_observation(
    runtime: ScheduleRuntime, profile: ReproductionProfile
) -> ScheduleObservation:
    runtime.validate()
    instance = runtime.instance
    job_count = len(instance.jobs)
    assigned_work = [0.0] * job_count
    latest_assigned_end: list[float | None] = [None] * job_count
    process_duration_by_machine = [0.0] * len(instance.machines)
    final_process_end_by_machine = [0.0] * len(instance.machines)
    current_horizon = 0.0

    for timeline in runtime.timelines:
        for interval in timeline.intervals:
            current_horizon = max(current_horizon, interval.end)
            if interval.interval_type is not IntervalType.PROCESS:
                continue
            assert interval.job_id is not None
            if not 0 <= interval.job_id < job_count:
                raise ValueError("PROCESS interval references an unknown job")
            job_id = interval.job_id
            assigned_work[job_id] += interval.duration
            previous_end = latest_assigned_end[job_id]
            latest_assigned_end[job_id] = (
                interval.end if previous_end is None else max(previous_end, interval.end)
            )
            process_duration_by_machine[timeline.machine_id] += interval.duration
            final_process_end_by_machine[timeline.machine_id] = max(
                final_process_end_by_machine[timeline.machine_id], interval.end
            )

    completion_ratios: list[float] = []
    workload_pressures: list[float] = []
    projected_tardiness_ratios: list[float] = []
    for job, next_op_index, assigned, latest_end in zip(
        instance.jobs,
        runtime.next_op_index,
        assigned_work,
        latest_assigned_end,
        strict=True,
    ):
        remaining_nominal_work = 0.0
        for operation in job.operations[next_op_index:]:
            eligible = [
                duration
                for duration in operation.proc_times
                if duration is not None and duration > 0.0
            ]
            remaining_nominal_work += sum(eligible) / len(eligible)

        total_work = assigned + remaining_nominal_work
        completion_ratios.append(assigned / total_work if total_work > 0.0 else 0.0)
        workload_pressures.append(
            max(0.0, total_work - (job.due_date - job.arrival_time)) / total_work
            if total_work > 0.0
            else 0.0
        )
        projected_completion = (
            job.arrival_time + remaining_nominal_work
            if latest_end is None
            else latest_end + remaining_nominal_work
        )
        projected_tardiness_ratios.append(
            max(0.0, projected_completion - job.due_date) / max(total_work, 1e-12)
        )

    crj_ave, crj_std = _mean_and_population_std(completion_ratios)
    tr_values = (
        workload_pressures
        if profile.state.tr_feature is TRFeatureMode.LEGACY_WORKLOAD_PRESSURE
        else projected_tardiness_ratios
    )
    tr_ave, tr_std = _mean_and_population_std(tr_values)

    if profile.state.utilization_feature == "paper_uave":
        utilization_values = [
            duration / final_end if final_end > 0.0 else 0.0
            for duration, final_end in zip(
                process_duration_by_machine,
                final_process_end_by_machine,
                strict=True,
            )
        ]
        u_ave, u_std = _mean_and_population_std(utilization_values)
    else:
        utilization_values = [
            duration / current_horizon if current_horizon > 0.0 else 0.0
            for duration in process_duration_by_machine
        ]
        _, u_std = _mean_and_population_std(utilization_values)
        u_ave = (
            sum(process_duration_by_machine)
            / (len(instance.machines) * current_horizon)
            if current_horizon > 0.0
            else 0.0
        )

    return ScheduleObservation(
        crj_ave=crj_ave,
        crj_std=crj_std,
        u_ave=u_ave,
        u_std=u_std,
        tr_ave=tr_ave,
        tr_std=tr_std,
    )
