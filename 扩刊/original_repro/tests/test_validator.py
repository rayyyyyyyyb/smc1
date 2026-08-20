import pytest

from smc_repro.schemas import (
    InstanceSpec,
    IntervalType,
    JobSpec,
    MachineSpec,
    OperationSpec,
    ScheduleInterval,
)
from smc_repro.validator import validate_schedule


def _instance() -> InstanceSpec:
    job = JobSpec(
        0,
        2.0,
        20.0,
        1,
        (
            OperationSpec(0, 0, (3.0, None)),
            OperationSpec(0, 1, (None, 4.0)),
        ),
        3.0,
    )
    machines = (MachineSpec(0, 1.0, 10.0), MachineSpec(1, 1.0, 10.0))
    return InstanceSpec("validator", 1, 2, (job,), machines)


def test_validator_reports_arrival_and_precedence_errors() -> None:
    intervals = [
        ScheduleInterval(0, 0.0, 3.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(1, 2.5, 6.5, IntervalType.PROCESS, 0, 1),
    ]
    report = validate_schedule(_instance(), intervals, require_complete=True)
    assert not report.ok
    assert any("arrival" in error for error in report.errors)
    assert any("precedence" in error for error in report.errors)


def test_validator_reports_duplicate_and_missing() -> None:
    intervals = [
        ScheduleInterval(0, 2.0, 5.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(0, 5.0, 8.0, IntervalType.PROCESS, 0, 0),
    ]
    report = validate_schedule(_instance(), intervals, require_complete=True)
    assert any("processed 2 times" in error for error in report.errors)
    assert any("missing operation (0, 1)" in error for error in report.errors)


def test_validator_reports_ineligible_machine() -> None:
    intervals = [
        ScheduleInterval(1, 2.0, 5.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(1, 5.0, 9.0, IntervalType.PROCESS, 0, 1),
    ]
    report = validate_schedule(_instance(), intervals, require_complete=True)
    assert any("ineligible" in error for error in report.errors)


def test_validator_rejects_processing_shorter_than_nominal_duration() -> None:
    interval = ScheduleInterval(0, 2.0, 4.5, IntervalType.PROCESS, 0, 0)

    report = validate_schedule(_instance(), [interval], require_complete=False)

    assert any("processing duration" in error for error in report.errors)


def test_validator_allows_processing_longer_than_nominal_duration() -> None:
    interval = ScheduleInterval(0, 2.0, 6.0, IntervalType.PROCESS, 0, 0)

    report = validate_schedule(_instance(), [interval], require_complete=False)

    assert report.ok


def test_validator_allows_processing_duration_within_tolerance() -> None:
    interval = ScheduleInterval(
        0,
        2.0,
        2.0 + 3.0 - 5e-10,
        IntervalType.PROCESS,
        0,
        0,
    )

    report = validate_schedule(_instance(), [interval], require_complete=False)

    assert report.ok


def test_validator_reports_process_process_machine_overlap() -> None:
    intervals = [
        ScheduleInterval(0, 2.0, 5.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(0, 4.0, 7.0, IntervalType.PROCESS, 0, 0),
    ]

    report = validate_schedule(_instance(), intervals, require_complete=False)

    assert any("machine 0 overlap" in error for error in report.errors)


def test_validator_reports_process_maintenance_machine_overlap() -> None:
    intervals = [
        ScheduleInterval(0, 2.0, 5.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(0, 4.0, 9.0, IntervalType.PM),
    ]

    report = validate_schedule(_instance(), intervals, require_complete=False)

    assert any("machine 0 overlap" in error for error in report.errors)


@pytest.mark.parametrize(
    ("interval_type", "end", "error_substring"),
    [
        (IntervalType.SETUP, 2.0, "setup duration"),
        (IntervalType.PM, 4.0, "PM duration"),
        (IntervalType.CM, 9.0, "CM duration"),
    ],
)
def test_validator_rejects_semantic_duration_mismatch(
    interval_type: IntervalType,
    end: float,
    error_substring: str,
) -> None:
    interval = ScheduleInterval(0, 0.0, end, interval_type)

    report = validate_schedule(_instance(), [interval], require_complete=False)

    assert any(error_substring in error for error in report.errors)


@pytest.mark.parametrize(
    ("interval_type", "expected_duration"),
    [
        (IntervalType.SETUP, 1.0),
        (IntervalType.PM, 5.0),
        (IntervalType.CM, 10.0),
    ],
)
def test_validator_allows_semantic_duration_within_tolerance(
    interval_type: IntervalType,
    expected_duration: float,
) -> None:
    interval = ScheduleInterval(
        0,
        0.0,
        expected_duration + 5e-10,
        interval_type,
    )

    report = validate_schedule(_instance(), [interval], require_complete=False)

    assert report.ok


@pytest.mark.parametrize(
    ("interval_type", "end"),
    [
        (IntervalType.SETUP, 10.0),
        (IntervalType.PM, 14.0),
        (IntervalType.CM, 19.0),
    ],
)
def test_complete_schedule_rejects_nonprocess_after_final_process_horizon(
    interval_type: IntervalType,
    end: float,
) -> None:
    intervals = [
        ScheduleInterval(0, 2.0, 5.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(1, 5.0, 9.0, IntervalType.PROCESS, 0, 1),
        ScheduleInterval(0, 9.0, end, interval_type),
    ]

    report = validate_schedule(_instance(), intervals, require_complete=True)

    assert any("after final process horizon" in error for error in report.errors)


def test_complete_schedule_with_zero_setup_time_needs_no_setup_interval() -> None:
    base = _instance()
    instance = InstanceSpec(
        base.instance_id,
        base.instance_seed,
        base.failure_seed,
        base.jobs,
        (
            MachineSpec(0, 0.0, 10.0),
            MachineSpec(1, 0.0, 10.0),
        ),
    )
    intervals = [
        ScheduleInterval(0, 2.0, 5.0, IntervalType.PROCESS, 0, 0),
        ScheduleInterval(1, 5.0, 9.0, IntervalType.PROCESS, 0, 1),
    ]

    report = validate_schedule(instance, intervals, require_complete=True)

    assert report.ok
