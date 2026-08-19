import pytest

from smc_repro.schemas import (
    InstanceSpec,
    IntervalType,
    JobSpec,
    MachineSpec,
    OperationSpec,
    ScheduleInterval,
)


def test_operation_rejects_empty_eligibility() -> None:
    with pytest.raises(ValueError, match="eligible machine"):
        OperationSpec(job_id=0, op_id=0, proc_times=(None, None))


def test_process_interval_requires_job_and_operation() -> None:
    with pytest.raises(ValueError, match="PROCESS"):
        ScheduleInterval(0, 0.0, 1.0, IntervalType.PROCESS)


def test_interval_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="end"):
        ScheduleInterval(0, 2.0, 1.0, IntervalType.PM)


def test_interval_rejects_negative_machine_id() -> None:
    with pytest.raises(ValueError, match="machine_id"):
        ScheduleInterval(-1, 0.0, 1.0, IntervalType.PM)


def test_instance_rejects_processing_time_vector_with_wrong_machine_count() -> None:
    operation = OperationSpec(job_id=0, op_id=0, proc_times=(1.0,))
    job = JobSpec(0, 0.0, 1.0, 1, (operation,))
    machines = (MachineSpec(0, 0.0, 1.0), MachineSpec(1, 0.0, 1.0))

    with pytest.raises(ValueError, match="machine count"):
        InstanceSpec("i-1", 1, 2, (job,), machines)
