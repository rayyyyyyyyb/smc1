from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn

from smc_repro.schemas import InstanceSpec, JobSpec, MachineSpec, OperationSpec

INSTANCE_SCHEMA_VERSION = 1


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"nonstandard JSON constant is not allowed: {value}")


def instance_to_dict(instance: InstanceSpec) -> dict[str, Any]:
    return {
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "instance_seed": instance.instance_seed,
        "failure_seed": instance.failure_seed,
        "metadata": dict(instance.metadata),
        "machines": [
            {
                "machine_id": machine.machine_id,
                "setup_time": machine.setup_time,
                "cm_duration": machine.cm_duration,
                "eta": machine.eta,
                "beta": machine.beta,
                "pm_duration_ratio": machine.pm_duration_ratio,
            }
            for machine in instance.machines
        ],
        "jobs": [
            {
                "job_id": job.job_id,
                "arrival_time": job.arrival_time,
                "due_date": job.due_date,
                "urgency": job.urgency,
                "weight": job.weight,
                "operations": [
                    {
                        "job_id": operation.job_id,
                        "op_id": operation.op_id,
                        "proc_times": list(operation.proc_times),
                    }
                    for operation in job.operations
                ],
            }
            for job in instance.jobs
        ],
    }


def instance_from_dict(data: dict[str, Any]) -> InstanceSpec:
    version = data.get("schema_version")
    if version != INSTANCE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported instance schema version: {version!r}; expected {INSTANCE_SCHEMA_VERSION}"
        )
    try:
        machines = tuple(
            MachineSpec(
                machine_id=int(item["machine_id"]),
                setup_time=float(item["setup_time"]),
                cm_duration=float(item["cm_duration"]),
                eta=float(item["eta"]),
                beta=float(item["beta"]),
                pm_duration_ratio=float(item["pm_duration_ratio"]),
            )
            for item in data["machines"]
        )
        jobs: list[JobSpec] = []
        for item in data["jobs"]:
            operations = tuple(
                OperationSpec(
                    job_id=int(operation["job_id"]),
                    op_id=int(operation["op_id"]),
                    proc_times=tuple(
                        None if value is None else float(value)
                        for value in operation["proc_times"]
                    ),
                )
                for operation in item["operations"]
            )
            jobs.append(
                JobSpec(
                    job_id=int(item["job_id"]),
                    arrival_time=float(item["arrival_time"]),
                    due_date=float(item["due_date"]),
                    urgency=int(item["urgency"]),
                    operations=operations,
                    weight=float(item["weight"]),
                )
            )
        metadata_raw = data.get("metadata", {})
        if not isinstance(metadata_raw, dict):
            raise TypeError("metadata must be an object")
        metadata = {str(key): value for key, value in metadata_raw.items()}
        return InstanceSpec(
            instance_id=str(data["instance_id"]),
            instance_seed=int(data["instance_seed"]),
            failure_seed=int(data["failure_seed"]),
            jobs=tuple(jobs),
            machines=machines,
            metadata=metadata,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid instance payload: {exc}") from exc


def save_instance(instance: InstanceSpec, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        instance_to_dict(instance),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_handle,
            mode="wb",
            mtime=0,
        ) as gzip_handle:
            gzip_handle.write(payload)


def load_instance(path: Path) -> InstanceSpec:
    path = Path(path)
    try:
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            data = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"failed to read instance file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("instance JSON root must be an object")
    return instance_from_dict(data)


def instance_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
