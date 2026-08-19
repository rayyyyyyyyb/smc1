# SMC Original Reproduction — Gate 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the immutable legacy audit, deterministic instance banks, typed scheduling records, explicit timelines, reliability mathematics, schedule metrics, and validator needed before any SMC model is retrained.

**Architecture:** Keep `code/`, `code1/`, and `code2/` byte-for-byte frozen. Create one installable `original_repro/` package. This plan deliberately stops before rules, environment transitions, rewards, agents, or training; those will receive a separate plan after Gate 1 review.

**Tech Stack:** Python 3.11; official PyTorch 2.10.0 CUDA 12.8 wheel; NumPy; SciPy; pandas; Matplotlib; PyYAML; pytest; Ruff; mypy.

**Spec:** `docs/superpowers/specs/2026-08-19-smc-original-reproduction-design.md`.

## Global Constraints

- Do not modify, delete, rename, move, or auto-format any file under `code/`, `code1/`, or `code2/`.
- Do not add GNN, PPO, Gymnasium, Hydra, PyTorch Geometric, DGL, or external datasets.
- Do not implement scheduling rules, environment `step`, rewards, replay buffers, neural networks, or training in Gate 1.
- Use local RNG objects for instance generation; do not mutate Python or NumPy global RNG state.
- Use JSON gzip with deterministic gzip headers; do not use pickle for instances.
- Use TDD and create one commit per task.
- Run the complete verification suite after Task 5, then stop for review.

## Gate 1 File Tree

```text
original_repro/
  pyproject.toml
  README.md
  src/smc_repro/
    __init__.py
    schemas.py
    seeding.py
    instance_generator.py
    instance_io.py
    timeline.py
    reliability.py
    metrics.py
    validator.py
    scripts/
      __init__.py
      verify_hardware.py
      audit_legacy_outputs.py
      build_instance_banks.py
  tests/
    test_package_import.py
    test_legacy_audit.py
    test_legacy_immutable.py
    test_schemas.py
    test_seeding.py
    test_common_random_numbers.py
    test_instance_generator.py
    test_instance_io.py
    test_reliability.py
    test_timeline.py
    test_metrics.py
    test_validator.py
```

---

## Task 1: Create the Installable Project and Verify the RTX 5090

**Files:**
- Create: `original_repro/pyproject.toml`
- Create: `original_repro/README.md`
- Create: `original_repro/src/smc_repro/__init__.py`
- Create: `original_repro/src/smc_repro/scripts/__init__.py`
- Create: `original_repro/src/smc_repro/scripts/verify_hardware.py`
- Create: `original_repro/tests/test_package_import.py`
- Modify: `.gitignore`

**Produces:** an editable `smc_repro` package and a hardware verification command.

- [ ] **Step 1: Create directories**

```bash
mkdir -p original_repro/src/smc_repro/scripts original_repro/tests
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=75", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "smc-repro"
version = "0.1.0"
description = "Audited reproduction of the original SMC DL-DDQN scheduling study"
requires-python = ">=3.11,<3.13"
dependencies = [
  "numpy>=2.0,<3",
  "scipy>=1.14,<2",
  "pandas>=2.2,<3",
  "matplotlib>=3.9,<4",
  "PyYAML>=6,<7",
  "tqdm>=4.66,<5",
]

[project.optional-dependencies]
dev = [
  "pytest>=8,<10",
  "pytest-cov>=5,<7",
  "ruff>=0.8,<1",
  "mypy>=1.13,<2",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["smc_repro"]
```

- [ ] **Step 3: Create package files**

```python
# original_repro/src/smc_repro/__init__.py
__version__ = "0.1.0"
```

```python
# original_repro/src/smc_repro/scripts/__init__.py
"""Executable modules for the SMC reproduction project."""
```

- [ ] **Step 4: Write the failing package test**

```python
# original_repro/tests/test_package_import.py
import smc_repro


def test_package_version_is_defined() -> None:
    assert smc_repro.__version__ == "0.1.0"
```

- [ ] **Step 5: Create the hardware verifier**

```python
# original_repro/src/smc_repro/scripts/verify_hardware.py
from __future__ import annotations

import json
import platform
import sys

import torch


def collect_hardware_info() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; expected the connected RTX 5090.")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    compiled_arches = torch.cuda.get_arch_list()
    required_arch = f"sm_{capability[0]}{capability[1]}"

    vector = torch.tensor([1.0, 2.0, 3.0], device=device)
    result = torch.sum(vector * vector)
    torch.cuda.synchronize(device)
    if float(result.item()) != 14.0:
        raise RuntimeError(f"unexpected CUDA smoke-test result: {result.item()}")

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "compiled_arches": compiled_arches,
        "reported_device_arch": required_arch,
        "cuda_smoke_result": float(result.item()),
    }


def main() -> None:
    print(json.dumps(collect_hardware_info(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Add local/generated files to `.gitignore`**

Append:

```gitignore
# SMC reproduction local environment and generated artifacts
original_repro/.venv/
original_repro/hardware.json
original_repro/environment.lock.txt
original_repro/artifacts/banks/**/*.json.gz
original_repro/artifacts/runs/
original_repro/artifacts/summaries/
original_repro/artifacts/figures/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
```

- [ ] **Step 7: Install on Ubuntu/Linux**

```bash
cd original_repro
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

Windows PowerShell equivalent:

```powershell
cd original_repro
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

- [ ] **Step 8: Verify import and hardware**

```bash
python -m pytest tests/test_package_import.py -q
python -m smc_repro.scripts.verify_hardware | tee hardware.json
python -m pip freeze > environment.lock.txt
```

Acceptance:

- test passes;
- `cuda_available` is true;
- `device_name` identifies the connected RTX 5090;
- the reported compute capability and compiled architecture list are saved for audit;
- CUDA smoke result is 14.0;
- no unsupported-architecture warning is emitted.

- [ ] **Step 9: Create README**

```markdown
# SMC Original-Conference Reproduction

This package is the audited, reproducible implementation of the original SMC DL-DDQN study.
The legacy directories `../code`, `../code1`, and `../code2` are frozen evidence and must not
be edited during the reproduction phase.

## Environment

Install the official PyTorch CUDA wheel first, then install this project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m smc_repro.scripts.verify_hardware
```

## Data

No external dataset is required for the original-paper reproduction. Synthetic train and test
instance banks are generated by `smc_repro.scripts.build_instance_banks` and stored with
stable seeds, JSON schemas, and SHA-256 hashes.
```

- [ ] **Step 10: Commit**

```bash
cd ..
git add .gitignore original_repro/pyproject.toml original_repro/README.md \
  original_repro/src/smc_repro/__init__.py \
  original_repro/src/smc_repro/scripts/__init__.py \
  original_repro/src/smc_repro/scripts/verify_hardware.py \
  original_repro/tests/test_package_import.py
git commit -m "build: add reproducible SMC package environment"
```

---

## Task 2: Freeze and Hash the Legacy Snapshot

**Files:**
- Create: `docs/superpowers/specs/2026-08-19-smc-original-reproduction-design.md`
- Create: `docs/audit/legacy_manifest.json`
- Create: `original_repro/src/smc_repro/scripts/audit_legacy_outputs.py`
- Create: `original_repro/tests/test_legacy_audit.py`
- Create: `original_repro/tests/test_legacy_immutable.py`

**Produces:** a byte-level manifest of all files under the three legacy directories.

- [ ] **Step 1: Copy the supplied design document**

```bash
mkdir -p docs/superpowers/specs docs/audit
cp 2026-08-19-smc-original-reproduction-design.md \
  docs/superpowers/specs/2026-08-19-smc-original-reproduction-design.md
```

- [ ] **Step 2: Write the audit unit test**

```python
# tests/test_legacy_audit.py
import hashlib
import json

from smc_repro.scripts.audit_legacy_outputs import audit_legacy_outputs


def test_audit_records_every_legacy_file(tmp_path) -> None:
    expected = {}
    for dirname in ("code", "code1", "code2"):
        path = tmp_path / dirname / "sample.txt"
        path.parent.mkdir(parents=True)
        payload = dirname.encode("utf-8")
        path.write_bytes(payload)
        expected[f"{dirname}/sample.txt"] = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    output = tmp_path / "manifest.json"
    manifest = audit_legacy_outputs(tmp_path, output)
    assert manifest["files"] == expected
    assert json.loads(output.read_text(encoding="utf-8"))["files"] == expected
```

- [ ] **Step 3: Write the repository immutability test**

```python
# original_repro/tests/test_legacy_immutable.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path


LEGACY_DIRS = ("code", "code1", "code2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_legacy_manifest_matches_files() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "docs" / "audit" / "legacy_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    observed: dict[str, dict[str, object]] = {}
    for dirname in LEGACY_DIRS:
        for path in sorted((repo_root / dirname).rglob("*")):
            if path.is_file():
                relative_path = path.relative_to(repo_root).as_posix()
                observed[relative_path] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }

    assert manifest["files"] == observed
```

- [ ] **Step 4: Run tests and verify they fail because the audit module/manifest is absent**

```bash
cd original_repro
python -m pytest tests/test_legacy_audit.py tests/test_legacy_immutable.py -q
```

- [ ] **Step 5: Implement audit script**

```python
# src/smc_repro/scripts/audit_legacy_outputs.py
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LEGACY_DIRS = ("code", "code1", "code2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_legacy_outputs(root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    files: dict[str, dict[str, object]] = {}
    for dirname in LEGACY_DIRS:
        legacy_dir = root / dirname
        if not legacy_dir.is_dir():
            raise FileNotFoundError(f"missing legacy directory: {legacy_dir}")
        for path in sorted(legacy_dir.rglob("*")):
            if path.is_file():
                relative_path = path.relative_to(root).as_posix()
                files[relative_path] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "legacy_directories": list(LEGACY_DIRS),
        "files": files,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_legacy_outputs(args.repo_root, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Generate manifest and rerun tests**

```bash
python -m smc_repro.scripts.audit_legacy_outputs \
  --repo-root .. \
  --output ../docs/audit/legacy_manifest.json
python -m pytest tests/test_legacy_audit.py tests/test_legacy_immutable.py -q
git diff --exit-code -- ../code ../code1 ../code2
```

- [ ] **Step 7: Commit**

```bash
cd ..
git add docs/superpowers/specs docs/audit \
  original_repro/src/smc_repro/scripts/audit_legacy_outputs.py \
  original_repro/tests/test_legacy_audit.py \
  original_repro/tests/test_legacy_immutable.py
git commit -m "chore: freeze legacy SMC snapshot"
```

---

## Task 3: Add Typed Schemas and Deterministic Random Streams

**Files:**
- Create: `original_repro/src/smc_repro/schemas.py`
- Create: `original_repro/src/smc_repro/seeding.py`
- Create: `original_repro/tests/test_schemas.py`
- Create: `original_repro/tests/test_seeding.py`
- Create: `original_repro/tests/test_common_random_numbers.py`

**Produces:** validated instance/schedule records and deterministic global/keyed RNG helpers.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schemas.py
import pytest

from smc_repro.schemas import IntervalType, OperationSpec, ScheduleInterval


def test_operation_rejects_empty_eligibility() -> None:
    with pytest.raises(ValueError, match="eligible machine"):
        OperationSpec(job_id=0, op_id=0, proc_times=(None, None))


def test_process_interval_requires_job_and_operation() -> None:
    with pytest.raises(ValueError, match="PROCESS"):
        ScheduleInterval(0, 0.0, 1.0, IntervalType.PROCESS)


def test_interval_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="end"):
        ScheduleInterval(0, 2.0, 1.0, IntervalType.PM)
```

```python
# tests/test_seeding.py
from smc_repro.seeding import keyed_uniform


def test_keyed_uniform_is_repeatable_and_bounded() -> None:
    a = keyed_uniform(7, "failure", "instance-1", 2, 3, 4)
    b = keyed_uniform(7, "failure", "instance-1", 2, 3, 4)
    assert a == b
    assert 0.0 <= a < 1.0


def test_keyed_uniform_changes_when_key_changes() -> None:
    assert keyed_uniform(7, "x", 1) != keyed_uniform(7, "x", 2)


def test_set_global_seed_repeats_python_numpy_and_torch() -> None:
    import random

    import numpy as np
    import torch

    from smc_repro.seeding import set_global_seed

    set_global_seed(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    set_global_seed(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert first == second
```

```python
# tests/test_common_random_numbers.py
from smc_repro.seeding import keyed_uniform


def test_unrelated_random_calls_do_not_change_failure_draw() -> None:
    expected = keyed_uniform(13, "process_failure", "i-1", 4, 2, 7)
    for index in range(10_000):
        keyed_uniform(13, "unrelated", index)
    observed = keyed_uniform(13, "process_failure", "i-1", 4, 2, 7)
    assert observed == expected
```

- [ ] **Step 2: Verify imports fail**

```bash
cd original_repro
python -m pytest tests/test_schemas.py tests/test_seeding.py \
  tests/test_common_random_numbers.py -q
```

- [ ] **Step 3: Implement schemas**

```python
# src/smc_repro/schemas.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class IntervalType(StrEnum):
    PROCESS = "PROCESS"
    SETUP = "SETUP"
    PM = "PM"
    CM = "CM"


@dataclass(frozen=True)
class OperationSpec:
    job_id: int
    op_id: int
    proc_times: tuple[float | None, ...]

    def __post_init__(self) -> None:
        if self.job_id < 0 or self.op_id < 0:
            raise ValueError("job_id and op_id must be non-negative")
        eligible = [value for value in self.proc_times if value is not None]
        if not eligible:
            raise ValueError("operation must have at least one eligible machine")
        if any(value is not None and value <= 0 for value in self.proc_times):
            raise ValueError("eligible processing times must be positive")

    @property
    def eligible_machines(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.proc_times) if value is not None)

    def processing_time(self, machine_id: int) -> float:
        if machine_id < 0 or machine_id >= len(self.proc_times):
            raise KeyError(f"unknown machine {machine_id}")
        value = self.proc_times[machine_id]
        if value is None:
            raise KeyError(
                f"machine {machine_id} is not eligible for operation {(self.job_id, self.op_id)}"
            )
        return float(value)


@dataclass(frozen=True)
class JobSpec:
    job_id: int
    arrival_time: float
    due_date: float
    urgency: int
    operations: tuple[OperationSpec, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("job_id must be non-negative")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be non-negative")
        if self.due_date < self.arrival_time:
            raise ValueError("due_date must not precede arrival_time")
        if self.urgency not in (1, 2, 3):
            raise ValueError("urgency must be 1, 2, or 3")
        if not self.operations:
            raise ValueError("job must contain at least one operation")
        for expected_op_id, operation in enumerate(self.operations):
            if operation.job_id != self.job_id or operation.op_id != expected_op_id:
                raise ValueError("operations must use matching sequential job/op ids")
        if self.weight <= 0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True)
class MachineSpec:
    machine_id: int
    setup_time: float
    cm_duration: float
    eta: float = 500.0
    beta: float = 2.0
    pm_duration_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if self.setup_time < 0 or self.cm_duration <= 0:
            raise ValueError("invalid setup or corrective-maintenance duration")
        if self.eta <= 0 or self.beta <= 0:
            raise ValueError("Weibull eta and beta must be positive")
        if not 0 < self.pm_duration_ratio <= 1:
            raise ValueError("pm_duration_ratio must be in (0, 1]")

    @property
    def pm_duration(self) -> float:
        return self.cm_duration * self.pm_duration_ratio


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: str
    instance_seed: int
    failure_seed: int
    jobs: tuple[JobSpec, ...]
    machines: tuple[MachineSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must not be empty")
        if self.instance_seed < 0 or self.failure_seed < 0:
            raise ValueError("seeds must be non-negative")
        if not self.jobs or not self.machines:
            raise ValueError("instance must contain jobs and machines")
        if tuple(job.job_id for job in self.jobs) != tuple(range(len(self.jobs))):
            raise ValueError("job ids must be contiguous from zero")
        if tuple(machine.machine_id for machine in self.machines) != tuple(
            range(len(self.machines))
        ):
            raise ValueError("machine ids must be contiguous from zero")
        machine_count = len(self.machines)
        for job in self.jobs:
            for operation in job.operations:
                if len(operation.proc_times) != machine_count:
                    raise ValueError("processing-time vectors must match machine count")


@dataclass(frozen=True)
class ScheduleInterval:
    machine_id: int
    start: float
    end: float
    interval_type: IntervalType
    job_id: int | None = None
    op_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        if self.interval_type is IntervalType.PROCESS:
            if self.job_id is None or self.op_id is None:
                raise ValueError("PROCESS interval requires job_id and op_id")
        elif self.job_id is not None or self.op_id is not None:
            raise ValueError("non-PROCESS intervals must not carry job_id/op_id")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: ScheduleInterval, eps: float = 1e-9) -> bool:
        if self.machine_id != other.machine_id:
            return False
        return self.start < other.end - eps and other.start < self.end - eps
```

- [ ] **Step 4: Implement seeding**

```python
# src/smc_repro/seeding.py
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def keyed_uniform(base_seed: int, *keys: object) -> float:
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    payload = "|".join(map(str, (base_seed, *keys))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / float(1 << 64)
```

- [ ] **Step 5: Verify**

```bash
python -m pytest tests/test_schemas.py tests/test_seeding.py \
  tests/test_common_random_numbers.py -q
python -m ruff check src/smc_repro/schemas.py src/smc_repro/seeding.py \
  tests/test_schemas.py tests/test_seeding.py tests/test_common_random_numbers.py
python -m mypy src/smc_repro/schemas.py src/smc_repro/seeding.py
```

- [ ] **Step 6: Commit**

```bash
cd ..
git add original_repro/src/smc_repro/schemas.py \
  original_repro/src/smc_repro/seeding.py \
  original_repro/tests/test_schemas.py \
  original_repro/tests/test_seeding.py \
  original_repro/tests/test_common_random_numbers.py
git commit -m "feat: add typed scheduling schemas and deterministic streams"
```

---

## Task 4: Add Legacy-Compatible Instance Generation, Serialization, and Banks

**Files:**
- Create: `original_repro/src/smc_repro/instance_generator.py`
- Create: `original_repro/src/smc_repro/instance_io.py`
- Create: `original_repro/src/smc_repro/scripts/build_instance_banks.py`
- Create: `original_repro/tests/test_instance_generator.py`
- Create: `original_repro/tests/test_instance_io.py`

**Produces:** deterministic synthetic instances, deterministic JSON-gzip bytes, 540 test instances, and fixed training banks.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_instance_generator.py
import random

import numpy as np

from smc_repro.instance_generator import generate_legacy_instance


def _generate():
    return generate_legacy_instance(
        instance_id="x",
        instance_seed=101,
        failure_seed=201,
        machine_count=8,
        new_job_count=10,
        mean_interarrival=50.0,
    )


def test_generator_is_repeatable() -> None:
    assert _generate() == _generate()


def test_generator_matches_legacy_ranges() -> None:
    instance = _generate()
    assert len(instance.jobs) == 15
    assert len(instance.machines) == 8
    for job in instance.jobs:
        assert 1 <= len(job.operations) <= 20
        for operation in job.operations:
            assert 2 <= len(operation.eligible_machines) <= 7
            for machine_id in operation.eligible_machines:
                assert 1 <= operation.processing_time(machine_id) <= 50


def test_generator_does_not_change_global_rng_state() -> None:
    random.seed(999)
    np.random.seed(999)
    expected_py = random.random()
    expected_np = float(np.random.random())
    random.seed(999)
    np.random.seed(999)
    _generate()
    assert random.random() == expected_py
    assert float(np.random.random()) == expected_np
```

```python
# tests/test_instance_io.py
import gzip
import json

import pytest

from smc_repro.instance_generator import generate_legacy_instance
from smc_repro.instance_io import load_instance, save_instance


def _instance():
    return generate_legacy_instance(
        instance_id="roundtrip",
        instance_seed=103,
        failure_seed=203,
        machine_count=8,
        new_job_count=10,
        mean_interarrival=50.0,
    )


def test_json_gzip_round_trip(tmp_path) -> None:
    path = tmp_path / "roundtrip.json.gz"
    save_instance(_instance(), path)
    assert load_instance(path) == _instance()


def test_identical_payload_has_identical_compressed_bytes(tmp_path) -> None:
    a = tmp_path / "a.json.gz"
    b = tmp_path / "nested" / "b.json.gz"
    save_instance(_instance(), a)
    save_instance(_instance(), b)
    assert a.read_bytes() == b.read_bytes()


def test_unsupported_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"schema_version": 99}, handle)
    with pytest.raises(ValueError, match="unsupported"):
        load_instance(path)


def test_malformed_gzip_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json.gz"
    path.write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="failed to read"):
        load_instance(path)
```

- [ ] **Step 2: Verify imports fail**

```bash
cd original_repro
python -m pytest tests/test_instance_generator.py tests/test_instance_io.py -q
```

- [ ] **Step 3: Implement generator**

```python
# src/smc_repro/instance_generator.py
from __future__ import annotations

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
```

- [ ] **Step 4: Implement deterministic serialization**

```python
# src/smc_repro/instance_io.py
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from smc_repro.schemas import InstanceSpec, JobSpec, MachineSpec, OperationSpec


INSTANCE_SCHEMA_VERSION = 1


def instance_to_dict(instance: InstanceSpec) -> dict[str, Any]:
    return {
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "instance_id": instance.instance_id,
        "instance_seed": instance.instance_seed,
        "failure_seed": instance.failure_seed,
        "metadata": instance.metadata,
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
            data = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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
```

- [ ] **Step 5: Implement bank builder**

```python
# src/smc_repro/scripts/build_instance_banks.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from smc_repro.instance_generator import generate_legacy_instance
from smc_repro.instance_io import instance_sha256, save_instance


MACHINES = (8, 12, 16)
MEAN_INTERARRIVALS = (50, 100, 150)
NEW_JOB_COUNTS = (10, 20, 30)
SCENARIOS = tuple(
    (machine_count, mean_interarrival, new_job_count)
    for machine_count in MACHINES
    for mean_interarrival in MEAN_INTERARRIVALS
    for new_job_count in NEW_JOB_COUNTS
)


def _entry(
    output_root: Path,
    path: Path,
    instance_id: str,
    instance_seed: int,
    failure_seed: int,
) -> dict[str, object]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "instance_id": instance_id,
        "instance_seed": instance_seed,
        "failure_seed": failure_seed,
        "sha256": instance_sha256(path),
    }


def build_instance_banks(
    *,
    output_root: Path,
    test_repetitions: int,
    train_seeds: tuple[int, ...],
    train_episodes: int,
    base_seed: int,
) -> dict[str, Any]:
    if test_repetitions <= 0 or train_episodes <= 0:
        raise ValueError("test_repetitions and train_episodes must be positive")
    if base_seed < 0 or any(seed < 0 for seed in train_seeds):
        raise ValueError("seeds must be non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for scenario_index, (machine_count, mean_interarrival, new_job_count) in enumerate(SCENARIOS):
        scenario_id = f"m{machine_count:02d}_j{new_job_count:02d}_e{mean_interarrival:03d}"
        for repetition in range(test_repetitions):
            instance_id = f"test_{scenario_id}_rep{repetition:02d}"
            instance_seed = base_seed + 1_000_000 + scenario_index * 10_000 + repetition
            failure_seed = base_seed + 2_000_000 + scenario_index * 10_000 + repetition
            instance = generate_legacy_instance(
                instance_id=instance_id,
                instance_seed=instance_seed,
                failure_seed=failure_seed,
                machine_count=machine_count,
                new_job_count=new_job_count,
                mean_interarrival=float(mean_interarrival),
            )
            path = output_root / "test" / scenario_id / f"{instance_id}.json.gz"
            save_instance(instance, path)
            entries.append(_entry(output_root, path, instance_id, instance_seed, failure_seed))

    for train_seed in train_seeds:
        parameter_rng = random.Random(base_seed + 3_000_000 + train_seed)
        for episode in range(train_episodes):
            machine_count = parameter_rng.randint(8, 18)
            mean_interarrival = parameter_rng.randint(50, 200)
            new_job_count = parameter_rng.randint(10, 30)
            instance_id = f"train_seed{train_seed:03d}_ep{episode:04d}"
            instance_seed = base_seed + 4_000_000 + train_seed * 100_000 + episode
            failure_seed = base_seed + 5_000_000 + train_seed * 100_000 + episode
            instance = generate_legacy_instance(
                instance_id=instance_id,
                instance_seed=instance_seed,
                failure_seed=failure_seed,
                machine_count=machine_count,
                new_job_count=new_job_count,
                mean_interarrival=float(mean_interarrival),
            )
            path = output_root / "train" / f"seed_{train_seed:03d}" / f"{instance_id}.json.gz"
            save_instance(instance, path)
            entries.append(_entry(output_root, path, instance_id, instance_seed, failure_seed))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "base_seed": base_seed,
        "scenario_order": [
            {
                "machine_count": machine_count,
                "mean_interarrival": mean_interarrival,
                "new_job_count": new_job_count,
            }
            for machine_count, mean_interarrival, new_job_count in SCENARIOS
        ],
        "test_repetitions": test_repetitions,
        "train_seeds": list(train_seeds),
        "train_episodes": train_episodes,
        "files": entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--test-repetitions", type=int, default=20)
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20260819)
    args = parser.parse_args()
    build_instance_banks(
        output_root=args.output_root,
        test_repetitions=args.test_repetitions,
        train_seeds=tuple(args.train_seeds),
        train_episodes=args.train_episodes,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run unit tests**

```bash
python -m pytest tests/test_instance_generator.py tests/test_instance_io.py -q
```

- [ ] **Step 7: Build complete banks twice**

```bash
rm -rf /tmp/smc-bank-a /tmp/smc-bank-b
python -m smc_repro.scripts.build_instance_banks \
  --output-root /tmp/smc-bank-a \
  --test-repetitions 20 \
  --train-seeds 0 1 2 3 4 \
  --train-episodes 200 \
  --base-seed 20260819
python -m smc_repro.scripts.build_instance_banks \
  --output-root /tmp/smc-bank-b \
  --test-repetitions 20 \
  --train-seeds 0 1 2 3 4 \
  --train-episodes 200 \
  --base-seed 20260819
cmp /tmp/smc-bank-a/manifest.json /tmp/smc-bank-b/manifest.json
python - <<'PYCOUNT'
import hashlib
import json
from pathlib import Path

path = Path('/tmp/smc-bank-a/manifest.json')
manifest = json.loads(path.read_text(encoding='utf-8'))
assert len(manifest['files']) == 1540, len(manifest['files'])
print('instance_count=', len(manifest['files']))
print('manifest_sha256=', hashlib.sha256(path.read_bytes()).hexdigest())
PYCOUNT
```

Expected: `cmp` succeeds and instance count is 1540.

- [ ] **Step 8: Quality checks and commit**

```bash
python -m ruff check src/smc_repro/instance_generator.py \
  src/smc_repro/instance_io.py \
  src/smc_repro/scripts/build_instance_banks.py \
  tests/test_instance_generator.py tests/test_instance_io.py
python -m mypy src/smc_repro/instance_generator.py \
  src/smc_repro/instance_io.py \
  src/smc_repro/scripts/build_instance_banks.py
cd ..
git add original_repro/src/smc_repro/instance_generator.py \
  original_repro/src/smc_repro/instance_io.py \
  original_repro/src/smc_repro/scripts/build_instance_banks.py \
  original_repro/tests/test_instance_generator.py \
  original_repro/tests/test_instance_io.py
git commit -m "feat: add deterministic SMC instance banks"
```

Do not commit the 1540 generated `.json.gz` files. Keep `manifest.json` from a release bank only after repository review.

---

## Task 5: Add Timelines, Reliability, Metrics, and Validation

**Files:**
- Create: `original_repro/src/smc_repro/timeline.py`
- Create: `original_repro/src/smc_repro/reliability.py`
- Create: `original_repro/src/smc_repro/metrics.py`
- Create: `original_repro/src/smc_repro/validator.py`
- Create: `original_repro/tests/test_timeline.py`
- Create: `original_repro/tests/test_reliability.py`
- Create: `original_repro/tests/test_metrics.py`
- Create: `original_repro/tests/test_validator.py`

**Produces:** a single aligned interval representation, conditional Weibull failure probability, independently recomputable final metrics, and a complete-schedule validator.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reliability.py
import pytest

from smc_repro.reliability import weibull_cdf, weibull_interval_failure_probability


def test_interval_probability_zero_duration_is_zero() -> None:
    assert weibull_interval_failure_probability(100.0, 0.0, 500.0, 2.0) == 0.0


def test_interval_probability_increases_with_age_and_duration() -> None:
    young = weibull_interval_failure_probability(50.0, 10.0, 500.0, 2.0)
    old = weibull_interval_failure_probability(300.0, 10.0, 500.0, 2.0)
    longer = weibull_interval_failure_probability(300.0, 30.0, 500.0, 2.0)
    assert 0.0 <= young < old < longer <= 1.0


def test_conditional_probability_matches_survival_ratio() -> None:
    age, duration, eta, beta = 100.0, 40.0, 500.0, 2.0
    expected = 1.0 - (1.0 - weibull_cdf(age + duration, eta, beta)) / (
        1.0 - weibull_cdf(age, eta, beta)
    )
    assert weibull_interval_failure_probability(age, duration, eta, beta) == pytest.approx(expected)
```

```python
# tests/test_timeline.py
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
```

```python
# tests/test_metrics.py
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
```

```python
# tests/test_validator.py
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
```

- [ ] **Step 2: Verify imports fail**

```bash
cd original_repro
python -m pytest tests/test_reliability.py tests/test_timeline.py \
  tests/test_metrics.py tests/test_validator.py -q
```

- [ ] **Step 3: Implement reliability**

```python
# src/smc_repro/reliability.py
from __future__ import annotations

import math


def _validate(age: float, duration: float, eta: float, beta: float) -> None:
    if age < 0 or duration < 0:
        raise ValueError("age and duration must be non-negative")
    if eta <= 0 or beta <= 0:
        raise ValueError("eta and beta must be positive")


def weibull_cdf(age: float, eta: float = 500.0, beta: float = 2.0) -> float:
    _validate(age, 0.0, eta, beta)
    return 1.0 - math.exp(-((age / eta) ** beta))


def weibull_interval_failure_probability(
    age: float,
    duration: float,
    eta: float = 500.0,
    beta: float = 2.0,
) -> float:
    _validate(age, duration, eta, beta)
    if duration == 0.0:
        return 0.0
    cumulative_hazard_increment = ((age + duration) / eta) ** beta - (age / eta) ** beta
    value = 1.0 - math.exp(-cumulative_hazard_increment)
    return min(1.0, max(0.0, value))


def health_from_effective_age(
    age: float,
    eta: float = 500.0,
    beta: float = 2.0,
) -> float:
    return 100.0 * (1.0 - weibull_cdf(age, eta, beta))
```

- [ ] **Step 4: Implement timeline**

```python
# src/smc_repro/timeline.py
from __future__ import annotations

from dataclasses import dataclass, field

from smc_repro.schemas import ScheduleInterval


@dataclass
class MachineTimeline:
    machine_id: int
    intervals: list[ScheduleInterval] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")

    def ordered(self) -> list[ScheduleInterval]:
        return sorted(
            self.intervals,
            key=lambda item: (item.start, item.end, item.interval_type.value),
        )

    def add(self, interval: ScheduleInterval) -> None:
        if interval.machine_id != self.machine_id:
            raise ValueError("interval machine does not match timeline")
        for existing in self.intervals:
            if existing.overlaps(interval):
                raise ValueError(f"timeline overlap: {existing} vs {interval}")
        self.intervals.append(interval)
        self.intervals.sort(
            key=lambda item: (item.start, item.end, item.interval_type.value)
        )

    @property
    def available_time(self) -> float:
        return max((interval.end for interval in self.intervals), default=0.0)

    def earliest_feasible_start(self, duration: float, earliest: float) -> float:
        if duration < 0 or earliest < 0:
            raise ValueError("duration and earliest must be non-negative")
        cursor = earliest
        for interval in self.ordered():
            if interval.end <= cursor:
                continue
            if cursor + duration <= interval.start + 1e-9:
                return cursor
            cursor = max(cursor, interval.end)
        return cursor
```

- [ ] **Step 5: Implement validator**

```python
# src/smc_repro/validator.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Iterable

from smc_repro.schemas import InstanceSpec, IntervalType, ScheduleInterval


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...]


def validate_schedule(
    instance: InstanceSpec,
    intervals: Iterable[ScheduleInterval],
    *,
    require_complete: bool,
) -> ValidationReport:
    schedule = tuple(intervals)
    errors: list[str] = []
    machine_count = len(instance.machines)
    by_machine: dict[int, list[ScheduleInterval]] = defaultdict(list)
    process_by_operation: dict[tuple[int, int], list[ScheduleInterval]] = defaultdict(list)

    for interval in schedule:
        if not 0 <= interval.machine_id < machine_count:
            errors.append(f"invalid machine id {interval.machine_id}")
            continue
        if interval.end < interval.start:
            errors.append(f"negative interval duration: {interval}")
        by_machine[interval.machine_id].append(interval)

        if interval.interval_type is not IntervalType.PROCESS:
            continue
        assert interval.job_id is not None and interval.op_id is not None
        key = (interval.job_id, interval.op_id)
        process_by_operation[key].append(interval)
        if not 0 <= interval.job_id < len(instance.jobs):
            errors.append(f"unknown job for PROCESS interval: {key}")
            continue
        job = instance.jobs[interval.job_id]
        if not 0 <= interval.op_id < len(job.operations):
            errors.append(f"unknown operation for PROCESS interval: {key}")
            continue
        operation = job.operations[interval.op_id]
        if interval.machine_id not in operation.eligible_machines:
            errors.append(f"ineligible machine {interval.machine_id} for operation {key}")
        if interval.start < job.arrival_time - 1e-9:
            errors.append(
                f"arrival violation for job {job.job_id}: {interval.start} < {job.arrival_time}"
            )

    for machine_id, machine_intervals in by_machine.items():
        ordered = sorted(
            machine_intervals,
            key=lambda item: (item.start, item.end, item.interval_type.value),
        )
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.overlaps(current):
                errors.append(
                    f"machine {machine_id} overlap: {previous} vs {current}"
                )

    expected = {
        (job.job_id, operation.op_id)
        for job in instance.jobs
        for operation in job.operations
    }
    for key in sorted(expected):
        count = len(process_by_operation.get(key, ()))
        if count > 1:
            errors.append(f"operation {key} processed {count} times")
        if require_complete and count == 0:
            errors.append(f"missing operation {key}")

    for key in sorted(set(process_by_operation) - expected):
        errors.append(f"unexpected operation {key}")

    unique_intervals = {
        key: values[0]
        for key, values in process_by_operation.items()
        if len(values) == 1 and key in expected
    }
    for job in instance.jobs:
        for op_id in range(1, len(job.operations)):
            previous_key = (job.job_id, op_id - 1)
            current_key = (job.job_id, op_id)
            if previous_key in unique_intervals and current_key in unique_intervals:
                previous = unique_intervals[previous_key]
                current = unique_intervals[current_key]
                if current.start < previous.end - 1e-9:
                    errors.append(
                        f"precedence violation {previous_key}->{current_key}: "
                        f"{current.start} < {previous.end}"
                    )

    return ValidationReport(ok=not errors, errors=tuple(errors))
```

- [ ] **Step 6: Implement metrics**

```python
# src/smc_repro/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from smc_repro.schemas import InstanceSpec, IntervalType, ScheduleInterval
from smc_repro.validator import validate_schedule


@dataclass(frozen=True)
class ScheduleMetrics:
    makespan: float
    paper_trave: float
    total_tardiness: float
    mean_tardiness: float
    weighted_tardiness: float
    tardy_rate: float
    paper_uave: float
    standard_utilization: float
    availability_adjusted_utilization: float
    total_process_time: float
    total_setup_time: float
    total_pm_time: float
    total_cm_time: float
    pm_count: int
    cm_count: int


def compute_schedule_metrics(
    instance: InstanceSpec,
    intervals: Iterable[ScheduleInterval],
) -> ScheduleMetrics:
    schedule = tuple(intervals)
    report = validate_schedule(instance, schedule, require_complete=True)
    if not report.ok:
        raise ValueError("invalid or incomplete schedule:\n" + "\n".join(report.errors))

    process_by_operation = {
        (interval.job_id, interval.op_id): interval
        for interval in schedule
        if interval.interval_type is IntervalType.PROCESS
    }
    job_completions: dict[int, float] = {}
    process_by_job: dict[int, float] = {}
    tardiness_by_job: dict[int, float] = {}

    for job in instance.jobs:
        final_key = (job.job_id, len(job.operations) - 1)
        final_interval = process_by_operation[final_key]
        completion = final_interval.end
        process_work = sum(
            process_by_operation[(job.job_id, operation.op_id)].duration
            for operation in job.operations
        )
        job_completions[job.job_id] = completion
        process_by_job[job.job_id] = process_work
        tardiness_by_job[job.job_id] = max(0.0, completion - job.due_date)

    makespan = max(job_completions.values())
    total_tardiness = sum(tardiness_by_job.values())
    mean_tardiness = total_tardiness / len(instance.jobs)
    weighted_tardiness = sum(
        job.weight * tardiness_by_job[job.job_id] for job in instance.jobs
    )
    tardy_rate = sum(value > 1e-9 for value in tardiness_by_job.values()) / len(instance.jobs)
    paper_trave = sum(
        tardiness_by_job[job.job_id] / max(process_by_job[job.job_id], 1e-12)
        for job in instance.jobs
    ) / len(instance.jobs)

    process_time = sum(
        interval.duration
        for interval in schedule
        if interval.interval_type is IntervalType.PROCESS
    )
    setup_time = sum(
        interval.duration
        for interval in schedule
        if interval.interval_type is IntervalType.SETUP
    )
    pm_intervals = tuple(
        interval for interval in schedule if interval.interval_type is IntervalType.PM
    )
    cm_intervals = tuple(
        interval for interval in schedule if interval.interval_type is IntervalType.CM
    )
    pm_time = sum(interval.duration for interval in pm_intervals)
    cm_time = sum(interval.duration for interval in cm_intervals)

    per_machine_paper_utilization: list[float] = []
    for machine in instance.machines:
        machine_process = tuple(
            interval
            for interval in schedule
            if interval.machine_id == machine.machine_id
            and interval.interval_type is IntervalType.PROCESS
        )
        busy = sum(interval.duration for interval in machine_process)
        final_process_end = max((interval.end for interval in machine_process), default=0.0)
        per_machine_paper_utilization.append(
            0.0 if final_process_end <= 0.0 else busy / final_process_end
        )
    paper_uave = sum(per_machine_paper_utilization) / len(instance.machines)

    total_capacity = len(instance.machines) * makespan
    standard_utilization = 0.0 if total_capacity <= 0 else process_time / total_capacity
    available_capacity = total_capacity - setup_time - pm_time - cm_time
    availability_adjusted_utilization = (
        0.0 if available_capacity <= 0 else process_time / available_capacity
    )

    return ScheduleMetrics(
        makespan=makespan,
        paper_trave=paper_trave,
        total_tardiness=total_tardiness,
        mean_tardiness=mean_tardiness,
        weighted_tardiness=weighted_tardiness,
        tardy_rate=tardy_rate,
        paper_uave=paper_uave,
        standard_utilization=standard_utilization,
        availability_adjusted_utilization=availability_adjusted_utilization,
        total_process_time=process_time,
        total_setup_time=setup_time,
        total_pm_time=pm_time,
        total_cm_time=cm_time,
        pm_count=len(pm_intervals),
        cm_count=len(cm_intervals),
    )
```

- [ ] **Step 7: Run the complete Gate 1 suite**

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
```

- [ ] **Step 8: Confirm legacy directories remain unchanged**

```bash
python -m smc_repro.scripts.audit_legacy_outputs \
  --repo-root .. \
  --output ../docs/audit/legacy_manifest_after_gate1.json
cmp ../docs/audit/legacy_manifest.json \
  ../docs/audit/legacy_manifest_after_gate1.json
git diff --exit-code -- ../code ../code1 ../code2
```

Expected: both comparisons succeed.

- [ ] **Step 9: Commit**

```bash
cd ..
git add original_repro/src/smc_repro/reliability.py \
  original_repro/src/smc_repro/timeline.py \
  original_repro/src/smc_repro/validator.py \
  original_repro/src/smc_repro/metrics.py \
  original_repro/tests/test_reliability.py \
  original_repro/tests/test_timeline.py \
  original_repro/tests/test_metrics.py \
  original_repro/tests/test_validator.py \
  docs/audit/legacy_manifest_after_gate1.json
git commit -m "feat: add audited timeline reliability and metrics core"
```

- [ ] **Step 10: Stop for review**

Do not implement any Task beyond Gate 1. Collect:

```bash
git status --short
git log --oneline -5
git diff HEAD~5..HEAD --stat
git diff --exit-code -- code code1 code2
cd original_repro
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m smc_repro.scripts.verify_hardware
```

Report exact test counts, bank count, manifest SHA-256, legacy manifest file count, five commit SHAs, deviations, and unresolved issues. Upload the repository for external review before any rules or model code is added.

---

## Self-Review Record

This Gate 1 reference implementation was syntax-compiled and its executable unit tests were run in an isolated reference tree before the plan was issued. The validated reference suite contained 26 passing tests, including deterministic compressed-byte equality and a full two-build manifest comparison. Codex must still run the suite in the target Python 3.11 + CUDA environment because exact cross-version bitwise reproducibility is not assumed.
