from dataclasses import fields

import pytest

import smc_repro.metrics as metrics_module
from smc_repro.metrics import ScheduleMetrics, compute_schedule_metrics
from smc_repro.schemas import (
    InstanceSpec,
    IntervalType,
    JobSpec,
    MachineSpec,
    OperationSpec,
    ScheduleInterval,
)


def _instance() -> InstanceSpec:
    jobs = (
        JobSpec(0, 0.0, 6.0, 1, (OperationSpec(0, 0, (3.0, None)),), 3.0),
        JobSpec(1, 0.0, 5.0, 3, (OperationSpec(1, 0, (None, 4.0)),), 1.0),
    )
    machines = (MachineSpec(0, 1.0, 4.0), MachineSpec(1, 1.0, 4.0))
    return InstanceSpec("hand", 1, 2, jobs, machines)


def test_duration_within_horizon_clips_both_ends() -> None:
    duration_within_horizon = getattr(
        metrics_module,
        "duration_within_horizon",
        None,
    )
    assert duration_within_horizon is not None
    interval = ScheduleInterval(0, 8.0, 14.0, IntervalType.PM)
    assert duration_within_horizon(interval, 10.0) == pytest.approx(2.0)


def test_duration_within_horizon_returns_zero_after_horizon() -> None:
    duration_within_horizon = getattr(
        metrics_module,
        "duration_within_horizon",
        None,
    )
    assert duration_within_horizon is not None
    interval = ScheduleInterval(0, 11.0, 12.0, IntervalType.PM)
    assert duration_within_horizon(interval, 10.0) == 0.0


def test_duration_within_horizon_rejects_negative_horizon() -> None:
    duration_within_horizon = getattr(
        metrics_module,
        "duration_within_horizon",
        None,
    )
    assert duration_within_horizon is not None
    interval = ScheduleInterval(0, 0.0, 1.0, IntervalType.PM)
    with pytest.raises(ValueError, match="horizon must be non-negative"):
        duration_within_horizon(interval, -1.0)


def test_schedule_metrics_has_frozen_public_field_order() -> None:
    assert tuple(field.name for field in fields(ScheduleMetrics)) == (
        "makespan",
        "paper_trave",
        "total_tardiness",
        "mean_tardiness",
        "weighted_tardiness",
        "tardy_rate",
        "on_time_rate",
        "mean_flow_time",
        "paper_uave",
        "standard_utilization",
        "availability_adjusted_utilization",
        "total_process_time",
        "total_setup_time",
        "total_pm_time",
        "total_cm_time",
        "total_downtime",
        "pm_count",
        "cm_count",
        "failure_count",
    )


def test_metrics_match_hand_calculation() -> None:
    intervals = [
        ScheduleInterval(0, 0.0, 3.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(1, 0.0, 2.0, IntervalType.PM),
        ScheduleInterval(1, 2.0, 6.0, IntervalType.PROCESS, 1, 0),
    ]
    metrics = compute_schedule_metrics(_instance(), intervals)
    assert metrics.makespan == 6.0
    assert metrics.total_tardiness == 1.0
    assert metrics.mean_tardiness == 0.5
    assert metrics.weighted_tardiness == 1.0
    assert metrics.tardy_rate == 0.5
    assert metrics.on_time_rate == 0.5
    assert metrics.mean_flow_time == 4.5
    assert metrics.paper_trave == pytest.approx((0.0 / 3.0 + 1.0 / 4.0) / 2.0)
    assert metrics.standard_utilization == pytest.approx(7.0 / 12.0)
    assert metrics.total_pm_time == 2.0
    assert metrics.total_cm_time == 0.0
    assert metrics.total_downtime == 2.0
    assert metrics.total_process_time == 7.0
    assert metrics.paper_uave == pytest.approx((1.0 + 4.0 / 6.0) / 2.0)
    assert metrics.pm_count == 1
    assert metrics.cm_count == 0
    assert metrics.failure_count == 0


def test_metrics_bound_maintenance_time_to_makespan() -> None:
    instance = InstanceSpec(
        "bounded",
        1,
        2,
        (
            JobSpec(0, 0.0, 20.0, 1, (OperationSpec(0, 0, (10.0, None)),), 1.0),
            JobSpec(1, 0.0, 20.0, 1, (OperationSpec(1, 0, (None, 8.0)),), 1.0),
        ),
        (
            MachineSpec(0, 0.0, 4.0),
            MachineSpec(1, 0.0, 4.0 + 1e-9),
        ),
    )
    intervals = [
        ScheduleInterval(0, 0.0, 10.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(1, 0.0, 8.0, IntervalType.PROCESS, 1, 0),
        ScheduleInterval(1, 8.0, 10.0 + 5e-10, IntervalType.PM),
    ]

    metrics = compute_schedule_metrics(instance, intervals)

    assert metrics.total_process_time == 18.0
    assert metrics.total_pm_time == 2.0
    assert metrics.total_downtime == 2.0
    assert metrics.standard_utilization == 0.9
    assert metrics.availability_adjusted_utilization == 1.0


def test_metrics_reject_nonpositive_available_capacity() -> None:
    instance = InstanceSpec(
        "nonpositive-capacity",
        1,
        2,
        (
            JobSpec(
                0,
                0.0,
                1.0,
                1,
                (OperationSpec(0, 0, (4e-10,)),),
                1.0,
            ),
        ),
        (MachineSpec(0, 6e-10, 1.0),),
    )
    intervals = [
        ScheduleInterval(0, 0.0, 4e-10, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(0, 0.0, 6e-10, IntervalType.SETUP),
    ]

    with pytest.raises(ValueError, match="available capacity must be positive"):
        compute_schedule_metrics(instance, intervals)
