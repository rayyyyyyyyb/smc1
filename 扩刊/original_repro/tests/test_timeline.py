import pytest

from smc_repro.schemas import IntervalType, ScheduleInterval
from smc_repro.timeline import MachineTimeline


def test_timeline_rejects_overlap() -> None:
    timeline = MachineTimeline(0)
    timeline.add(ScheduleInterval(0, 0.0, 5.0, IntervalType.PROCESS, 0, 0))
    with pytest.raises(ValueError, match="overlap"):
        timeline.add(ScheduleInterval(0, 4.0, 7.0, IntervalType.PM))


def test_touching_intervals_do_not_overlap() -> None:
    first = ScheduleInterval(0, 0.0, 5.0, IntervalType.PROCESS, 0, 0)
    second = ScheduleInterval(0, 5.0, 7.0, IntervalType.PM)
    assert not first.overlaps(second)


def test_identical_intervals_overlap() -> None:
    first = ScheduleInterval(0, 0.0, 5.0, IntervalType.PM)
    second = ScheduleInterval(0, 0.0, 5.0, IntervalType.CM)
    assert first.overlaps(second)


def test_earliest_start_uses_internal_gap() -> None:
    timeline = MachineTimeline(0)
    timeline.add(ScheduleInterval(0, 0.0, 5.0, IntervalType.PROCESS, 0, 0))
    timeline.add(ScheduleInterval(0, 10.0, 15.0, IntervalType.PROCESS, 1, 0))
    assert timeline.earliest_feasible_start(4.0, 5.0) == 5.0
    assert timeline.earliest_feasible_start(6.0, 5.0) == 15.0


def test_initial_intervals_are_validated_for_overlap() -> None:
    intervals = [
        ScheduleInterval(0, 0.0, 5.0, IntervalType.PM),
        ScheduleInterval(0, 4.0, 6.0, IntervalType.CM),
    ]

    with pytest.raises(ValueError, match="overlap"):
        MachineTimeline(0, intervals)


def test_initial_intervals_are_validated_for_machine_identity() -> None:
    with pytest.raises(ValueError, match="does not match"):
        MachineTimeline(0, [ScheduleInterval(1, 0.0, 5.0, IntervalType.PM)])


def test_intervals_view_cannot_mutate_timeline() -> None:
    interval = ScheduleInterval(0, 0.0, 5.0, IntervalType.PM)
    timeline = MachineTimeline(0, [interval])

    assert timeline.intervals == (interval,)
    with pytest.raises(AttributeError):
        timeline.intervals.append(ScheduleInterval(0, 5.0, 6.0, IntervalType.CM))


@pytest.mark.parametrize(
    ("duration", "earliest"),
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (-float("inf"), 0.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (1.0, -float("inf")),
    ],
)
def test_earliest_start_rejects_non_finite_inputs(
    duration: float, earliest: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        MachineTimeline(0).earliest_feasible_start(duration, earliest)
