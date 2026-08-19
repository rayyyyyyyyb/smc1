import pytest

from smc_repro.metrics import compute_schedule_metrics
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
    machines = (MachineSpec(0, 1.0, 10.0), MachineSpec(1, 1.0, 10.0))
    return InstanceSpec("hand", 1, 2, jobs, machines)


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
    assert metrics.paper_trave == pytest.approx((0.0 / 3.0 + 1.0 / 4.0) / 2.0)
    assert metrics.standard_utilization == pytest.approx(7.0 / 12.0)
    assert metrics.total_pm_time == 2.0
    assert metrics.total_process_time == 7.0
    assert metrics.paper_uave == pytest.approx((1.0 + 4.0 / 6.0) / 2.0)
