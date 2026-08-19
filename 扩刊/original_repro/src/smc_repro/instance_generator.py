from __future__ import annotations

import math
import random

import numpy as np

from smc_repro.schemas import InstanceSpec, JobSpec, MachineSpec, OperationSpec


def generate_legacy_instance(
    *,
    instance_id: str,
    instance_seed: int,
    failure_seed: int,
    machine_count: int,
    new_job_count: int,
    mean_interarrival: float,
    initial_job_count: int = 5,
) -> InstanceSpec:
    if instance_seed < 0 or failure_seed < 0:
        raise ValueError("seeds must be non-negative")
    if machine_count < 3:
        raise ValueError("legacy generator requires at least three machines")
    if new_job_count < 0 or initial_job_count <= 0:
        raise ValueError("invalid job counts")
    if not math.isfinite(mean_interarrival):
        raise ValueError("mean_interarrival must be finite")
    if mean_interarrival <= 0:
        raise ValueError("mean_interarrival must be positive")

    py_rng = random.Random(instance_seed)
    # RandomState intentionally matches the legacy np.random.seed()/np.random.exponential stream.
    np_rng = np.random.RandomState(instance_seed)
    total_jobs = initial_job_count + new_job_count
    operation_counts = [py_rng.randint(1, 20) for _ in range(total_jobs)]

    operations_by_job: list[tuple[OperationSpec, ...]] = []
    estimated_work: list[float] = []

    for job_id, operation_count in enumerate(operation_counts):
        operations: list[OperationSpec] = []
        job_mean_work = 0.0
        for op_id in range(operation_count):
            eligible_count = py_rng.randint(1, machine_count - 2) + 1
            machine_ids = list(range(machine_count))
            py_rng.shuffle(machine_ids)
            eligible = set(machine_ids[:eligible_count])
            proc_times: list[float | None] = []
            positive_times: list[float] = []
            for machine_id in range(machine_count):
                if machine_id in eligible:
                    value = float(py_rng.randint(1, 50))
                    proc_times.append(value)
                    positive_times.append(value)
                else:
                    proc_times.append(None)
            job_mean_work += sum(positive_times) / len(positive_times)
            operations.append(
                OperationSpec(job_id=job_id, op_id=op_id, proc_times=tuple(proc_times))
            )
        operations_by_job.append(tuple(operations))
        estimated_work.append(job_mean_work)

    arrivals = [0 for _ in range(initial_job_count)]
    if new_job_count:
        intervals = np_rng.exponential(mean_interarrival, size=new_job_count)
        integer_intervals = np.asarray(intervals, dtype=np.int64)
        arrivals.extend(np.cumsum(integer_intervals).astype(int).tolist())

    urgencies = [py_rng.randint(1, 3) for _ in range(total_jobs)]
    jobs: list[JobSpec] = []
    for job_id in range(total_jobs):
        arrival = float(arrivals[job_id])
        due = float(int(arrival + (0.2 + 0.5 * urgencies[job_id]) * estimated_work[job_id]))
        jobs.append(
            JobSpec(
                job_id=job_id,
                arrival_time=arrival,
                due_date=due,
                urgency=urgencies[job_id],
                operations=operations_by_job[job_id],
                weight={1: 3.0, 2: 2.0, 3: 1.0}[urgencies[job_id]],
            )
        )

    machines: list[MachineSpec] = []
    for machine_id in range(machine_count):
        machines.append(
            MachineSpec(
                machine_id=machine_id,
                setup_time=float(py_rng.randint(1, 50)),
                cm_duration=float(py_rng.randint(1, 99)),
                eta=500.0,
                beta=2.0,
                pm_duration_ratio=0.5,
            )
        )

    return InstanceSpec(
        instance_id=instance_id,
        instance_seed=instance_seed,
        failure_seed=failure_seed,
        jobs=tuple(jobs),
        machines=tuple(machines),
        metadata={
            "generator": "legacy_compatible_v1",
            "initial_job_count": initial_job_count,
            "new_job_count": new_job_count,
            "machine_count": machine_count,
            "mean_interarrival": mean_interarrival,
            "urgency_semantics": "1=high,2=medium,3=low",
        },
    )
