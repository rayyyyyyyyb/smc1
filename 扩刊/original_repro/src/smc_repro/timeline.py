from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from smc_repro.schemas import ScheduleInterval


@dataclass(init=False)
class MachineTimeline:
    machine_id: int
    _intervals: list[ScheduleInterval] = field(default_factory=list, repr=False)

    def __init__(
        self,
        machine_id: int,
        intervals: Iterable[ScheduleInterval] = (),
    ) -> None:
        self.machine_id = machine_id
        self._intervals = []
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        for interval in intervals:
            self.add(interval)

    @property
    def intervals(self) -> tuple[ScheduleInterval, ...]:
        return tuple(self._intervals)

    def ordered(self) -> list[ScheduleInterval]:
        return sorted(
            self._intervals,
            key=lambda item: (item.start, item.end, item.interval_type.value),
        )

    def add(self, interval: ScheduleInterval) -> None:
        if interval.machine_id != self.machine_id:
            raise ValueError("interval machine does not match timeline")
        for existing in self._intervals:
            if existing.overlaps(interval):
                raise ValueError(f"timeline overlap: {existing} vs {interval}")
        self._intervals.append(interval)
        self._intervals.sort(
            key=lambda item: (item.start, item.end, item.interval_type.value)
        )

    @property
    def available_time(self) -> float:
        return max((interval.end for interval in self._intervals), default=0.0)

    def earliest_feasible_start(self, duration: float, earliest: float) -> float:
        if not math.isfinite(duration) or not math.isfinite(earliest):
            raise ValueError("duration and earliest must be finite")
        if duration <= 0 or earliest < 0:
            raise ValueError("duration must be positive and earliest must be non-negative")
        cursor = earliest
        for interval in self.ordered():
            if interval.end <= cursor:
                continue
            if cursor + duration <= interval.start + 1e-9:
                return cursor
            cursor = max(cursor, interval.end)
        return cursor
