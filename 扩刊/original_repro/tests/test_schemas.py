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


def test_process_interval_rejects_zero_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        ScheduleInterval(0, 1.0, 1.0, IntervalType.PROCESS, 0, 0)


@pytest.mark.parametrize(
    "interval_type", [IntervalType.SETUP, IntervalType.PM, IntervalType.CM]
)
def test_nonprocess_recorded_intervals_require_positive_duration(
    interval_type: IntervalType,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        ScheduleInterval(0, 2.0, 2.0, interval_type)


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


def test_instance_metadata_is_defensively_copied_and_read_only() -> None:
    source = {"generator": "x", "machine_count": 1}
    instance = InstanceSpec(
        "immutable",
        1,
        2,
        (JobSpec(0, 0.0, 10.0, 1, (OperationSpec(0, 0, (1.0,)),), 1.0),),
        (MachineSpec(0, 0.0, 4.0),),
        source,
    )
    source["machine_count"] = 999
    assert instance.metadata["machine_count"] == 1
    with pytest.raises(TypeError):
        instance.metadata["machine_count"] = 2  # type: ignore[index]


def test_metadata_rejects_nested_mutable_values() -> None:
    with pytest.raises(TypeError, match="metadata values"):
        InstanceSpec(
            "nested",
            1,
            2,
            (JobSpec(0, 0.0, 10.0, 1, (OperationSpec(0, 0, (1.0,)),), 1.0),),
            (MachineSpec(0, 0.0, 4.0),),
            {"bad": [1, 2]},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_operation_rejects_non_finite_processing_times(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        OperationSpec(job_id=0, op_id=0, proc_times=(value, None))


@pytest.mark.parametrize("field", ["arrival_time", "due_date", "weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_job_rejects_non_finite_numeric_fields(field: str, value: float) -> None:
    values = {"arrival_time": 0.0, "due_date": 10.0, "weight": 1.0}
    values[field] = value
    operation = OperationSpec(job_id=0, op_id=0, proc_times=(1.0,))

    with pytest.raises(ValueError, match="finite"):
        JobSpec(
            job_id=0,
            arrival_time=values["arrival_time"],
            due_date=values["due_date"],
            urgency=1,
            operations=(operation,),
            weight=values["weight"],
        )


@pytest.mark.parametrize(
    "field", ["setup_time", "cm_duration", "eta", "beta", "pm_duration_ratio"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_machine_rejects_non_finite_numeric_fields(field: str, value: float) -> None:
    values = {
        "setup_time": 0.0,
        "cm_duration": 1.0,
        "eta": 500.0,
        "beta": 2.0,
        "pm_duration_ratio": 0.5,
    }
    values[field] = value

    with pytest.raises(ValueError, match="finite"):
        MachineSpec(
            machine_id=0,
            setup_time=values["setup_time"],
            cm_duration=values["cm_duration"],
            eta=values["eta"],
            beta=values["beta"],
            pm_duration_ratio=values["pm_duration_ratio"],
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (float("nan"), 1.0),
        (float("inf"), float("inf")),
        (-float("inf"), 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, -float("inf")),
    ],
)
def test_interval_rejects_non_finite_times(start: float, end: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ScheduleInterval(0, start, end, IntervalType.PM)
