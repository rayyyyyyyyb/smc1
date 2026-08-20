from __future__ import annotations

from dataclasses import dataclass

from smc_repro.schemas import InstanceSpec
from smc_repro.timeline import MachineTimeline


@dataclass
class MachineRuntime:
    machine_id: int
    health: float = 100.0
    effective_age: float = 0.0
    usage_time: float = 0.0
    degradation_factor: float = 1.0
    last_job_id: int | None = None
    pm_count: int = 0
    cm_count: int = 0
    process_count: int = 0


@dataclass
class ScheduleRuntime:
    instance: InstanceSpec
    next_op_index: list[int]
    timelines: list[MachineTimeline]
    machines: list[MachineRuntime]
    last_machine_by_job: list[int | None]
    decision_time: float = 0.0
    decision_index: int = 0

    def __setattr__(self, name: str, value: object) -> None:
        if name == "instance" and hasattr(self, "instance") and value is not self.instance:
            raise AttributeError("runtime instance cannot be replaced mid-episode")
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        job_count = len(self.instance.jobs)
        machine_count = len(self.instance.machines)
        for name in (
            "next_op_index",
            "timelines",
            "machines",
            "last_machine_by_job",
        ):
            if type(getattr(self, name)) is not list:
                raise ValueError(f"{name} must be a list")
        if len(self.next_op_index) != job_count:
            raise ValueError("next_op_index must contain one entry per job")
        for job, next_op_index in zip(self.instance.jobs, self.next_op_index, strict=True):
            if (
                isinstance(next_op_index, bool)
                or not isinstance(next_op_index, int)
                or not 0 <= next_op_index <= len(job.operations)
            ):
                raise ValueError("next_op_index entries must be within per-job operation bounds")
        timeline_machine_ids = tuple(timeline.machine_id for timeline in self.timelines)
        if any(
            isinstance(machine_id, bool) or not isinstance(machine_id, int)
            for machine_id in timeline_machine_ids
        ):
            raise ValueError(
                "timeline machine ids must be non-boolean integers contiguous from zero"
            )
        if len(self.timelines) != machine_count or timeline_machine_ids != tuple(
            range(machine_count)
        ):
            raise ValueError("timeline machine ids must be contiguous from zero")
        runtime_machine_ids = tuple(machine.machine_id for machine in self.machines)
        if any(
            isinstance(machine_id, bool) or not isinstance(machine_id, int)
            for machine_id in runtime_machine_ids
        ):
            raise ValueError(
                "runtime machine ids must be non-boolean integers contiguous from zero"
            )
        if len(self.machines) != machine_count or runtime_machine_ids != tuple(
            range(machine_count)
        ):
            raise ValueError("runtime machine ids must be contiguous from zero")
        if len(self.last_machine_by_job) != job_count:
            raise ValueError("last_machine_by_job must contain one entry per job")
        if any(
            machine_id is not None
            and (isinstance(machine_id, bool) or not isinstance(machine_id, int))
            for machine_id in self.last_machine_by_job
        ):
            raise ValueError(
                "last_machine_by_job entries must be non-boolean integers "
                "referencing known machines"
            )
        if any(
            machine_id is not None and not 0 <= machine_id < machine_count
            for machine_id in self.last_machine_by_job
        ):
            raise ValueError("last_machine_by_job entries must reference known machines")


def create_runtime(instance: InstanceSpec) -> ScheduleRuntime:
    return ScheduleRuntime(
        instance=instance,
        next_op_index=[0 for _ in instance.jobs],
        timelines=[MachineTimeline(machine.machine_id) for machine in instance.machines],
        machines=[MachineRuntime(machine.machine_id) for machine in instance.machines],
        last_machine_by_job=[None for _ in instance.jobs],
    )
