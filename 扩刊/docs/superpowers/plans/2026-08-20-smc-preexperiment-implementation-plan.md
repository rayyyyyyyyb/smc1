# SMC Pre-Experiment Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the completed Gate 1 foundation into a clean-clone-reproducible, fully tested implementation of the original SMC DL-DDQN method and all required baselines, ending at an end-to-end preflight gate immediately before full experiments.

**Architecture:** Keep `code/`, `code1/`, and `code2/` frozen as legacy evidence. Harden the audited `扩刊/original_repro/` foundation, regenerate and verify the synthetic instance bank, then add strict profile configuration, named runtime state, dispatching-rule libraries, a constructive scheduling environment, agents, checkpoints, and tiny smoke runs. Do not add full experiment orchestration, long training, paper plots, GNN, PPO, or endogenous PM actions in this phase.

**Tech Stack:** Python 3.11, PyTorch 2.10.0+cu128, NumPy, SciPy, pandas, PyYAML, pytest, Ruff, mypy; one NVIDIA RTX 5090 for CUDA smoke and agent smoke training.

**Spec:** `扩刊/docs/superpowers/specs/2026-08-20-smc-preexperiment-readiness-design.md`. Before implementation, copy `2026-08-20-smc-gate1-audit-report.md` to that spec path without changing its contents.

## Global Constraints

- Work from repository `main` only after recording the current HEAD; use a new Git worktree or a clearly isolated branch for implementation.
- Do not modify, reformat, move, or delete any file under repository-root `code/`, `code1/`, or `code2/`.
- Preserve the historical 120-file local legacy manifests as archival evidence; do not use them as clean-clone equality tests.
- New source remains under `扩刊/original_repro/src/smc_repro/`.
- Python is exactly 3.11.x; PyTorch is exactly `2.10.0+cu128` on the RTX 5090 environment.
- Keep the committed release-bank reference manifest SHA-256 exactly `68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515`.
- No external dataset is needed or downloaded in this phase; materialize the original synthetic bank from the generator.
- Do not introduce GNN, PyTorch Geometric, DGL, PPO, Gymnasium, Hydra, Ray, W&B, PROCESS/PM/WAIT action nodes, or a new paper method.
- Do not run the full 200-episode × 5-seed reproduction or the complete 540-instance evaluation.
- Smoke training is capped at 5 episodes per profile and smoke evaluation at 2 test instances per profile/method.
- All stochastic processes use separate streams: instance generation, policy exploration/tie-breaking, failures, wear, CM recovery, and bootstrap/statistics.
- A missing checkpoint is a hard `FileNotFoundError`; evaluation loads always force epsilon to exactly `0.0`.
- Every completed schedule is validated before metrics are accepted.
- Every YAML configuration is strict: missing required keys and unknown keys are hard errors.
- Use TDD: write a failing test, run it, implement the smallest change, rerun focused tests, then full tests.
- Each task ends with one focused commit and an updated line in `扩刊/all.md` or a new phase execution log.
- Do not claim a command passed unless its fresh output is captured in the final preflight report.

---

## Locked Target File Structure

```text
扩刊/
  CODEX_PHASE2_PREEXPERIMENT_PROMPT.md
  all.md
  docs/
    audit/
      legacy_local_full_manifest.json
      legacy_local_full_manifest_after_gate1.json
      legacy_tracked_manifest.json
      environment_5090_resolved.json
    superpowers/
      specs/
        2026-08-20-smc-preexperiment-readiness-design.md
      plans/
        2026-08-20-smc-preexperiment-implementation-plan.md
  original_repro/
    README.md
    pyproject.toml
    configs/
      legacy_snapshot.yaml
      paper_repro.yaml
      corrected_smc.yaml
      smoke.yaml
      ambiguities.json
    artifacts/
      banks/
        release/manifest.json
        materialized/                 # generated, ignored
      preflight/                      # generated, ignored except schema docs
    src/smc_repro/
      __init__.py
      config.py
      schemas.py
      seeding.py
      instance_generator.py
      instance_io.py
      timeline.py
      reliability.py
      metrics.py
      validator.py
      runtime.py
      observations.py
      rewards.py
      environment.py
      experiment_contract.py
      rules/
        __init__.py
        base.py
        legacy.py
        paper.py
        classical.py
      agents/
        __init__.py
        networks.py
        replay.py
        checkpoint.py
        dual_ddqn.py
        tabular.py
      scripts/
        audit_legacy_outputs.py
        build_instance_banks.py
        verify_instance_bank.py
        capture_environment.py
        clean_worktree_gate.py
        preflight.py
    tests/
      test_legacy_audit.py
      test_legacy_immutable.py
      test_schemas.py
      test_seeding.py
      test_instance_bank_verification.py
      test_metrics.py
      test_validator.py
      test_config.py
      test_observations.py
      test_rewards.py
      test_legacy_rules.py
      test_paper_rules.py
      test_classical_rules.py
      test_environment.py
      test_common_random_numbers.py
      test_networks.py
      test_double_dqn_targets.py
      test_checkpoint.py
      test_tabular_agents.py
      test_preflight.py
```

---

# Gate 1.5 — Make the Foundation Portable and Experiment-Safe

## Task 0: Split Local-Full and Git-Tracked Legacy Audits

**Files:**
- Copy: `2026-08-20-smc-gate1-audit-report.md` → `扩刊/docs/superpowers/specs/2026-08-20-smc-preexperiment-readiness-design.md`
- Rename: `扩刊/docs/audit/legacy_manifest.json` → `扩刊/docs/audit/legacy_local_full_manifest.json`
- Rename: `扩刊/docs/audit/legacy_manifest_after_gate1.json` → `扩刊/docs/audit/legacy_local_full_manifest_after_gate1.json`
- Create: `扩刊/docs/audit/legacy_tracked_manifest.json`
- Modify: `扩刊/original_repro/src/smc_repro/scripts/audit_legacy_outputs.py`
- Modify: `扩刊/original_repro/tests/test_legacy_immutable.py`
- Modify: `扩刊/original_repro/tests/test_legacy_audit.py`

**Interfaces:**
- Consumes: repository root and `code`, `code1`, `code2`.
- Produces: `AuditScope`, `audit_legacy_outputs(root, output_path, scope)`, and a tracked-only manifest that is valid in any clone.
- Produces: a historical local-full manifest that remains archival and is not used as a clean-clone equality requirement.

- [ ] **Step 1: Record the starting point and preserve the supplied audit**

PowerShell:

```powershell
$repo = (git rev-parse --show-toplevel).Trim()
git status --short
git log --oneline -8
New-Item -ItemType Directory -Force "$repo\扩刊\docs\superpowers\specs" | Out-Null
Copy-Item "$repo\2026-08-20-smc-gate1-audit-report.md" `
  "$repo\扩刊\docs\superpowers\specs\2026-08-20-smc-preexperiment-readiness-design.md"
```

Bash:

```bash
repo="$(git rev-parse --show-toplevel)"
git status --short
git log --oneline -8
mkdir -p "$repo/扩刊/docs/superpowers/specs"
cp "$repo/2026-08-20-smc-gate1-audit-report.md" \
  "$repo/扩刊/docs/superpowers/specs/2026-08-20-smc-preexperiment-readiness-design.md"
```

- [ ] **Step 2: Demonstrate the current clean-worktree failure before changing code**

Use the existing RTX 5090 Python environment; do not install another torch wheel.

PowerShell:

```powershell
$repo = (git rev-parse --show-toplevel).Trim()
$wt = Join-Path $env:TEMP ("smc-gate15-red-" + [guid]::NewGuid().ToString("N"))
git worktree add --detach $wt HEAD
try {
  $env:PYTHONPATH = Join-Path $wt "扩刊\original_repro\src"
  & "C:\Users\LXT\smc_gate1_env\Scripts\python.exe" -m pytest `
    (Join-Path $wt "扩刊\original_repro\tests\test_legacy_immutable.py") -q
  if ($LASTEXITCODE -eq 0) {
    throw "Expected the current clean-worktree legacy test to fail before the fix."
  }
} finally {
  Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
  git worktree remove --force $wt
}
```

Expected: the legacy-manifest equality test fails because ignored/untracked files in the 120-file manifest are absent.

- [ ] **Step 3: Rename the historical full manifests**

```bash
git mv 扩刊/docs/audit/legacy_manifest.json \
  扩刊/docs/audit/legacy_local_full_manifest.json
git mv 扩刊/docs/audit/legacy_manifest_after_gate1.json \
  扩刊/docs/audit/legacy_local_full_manifest_after_gate1.json
cmp 扩刊/docs/audit/legacy_local_full_manifest.json \
  扩刊/docs/audit/legacy_local_full_manifest_after_gate1.json
```

Expected: `cmp` exits 0. The two 120-file historical manifests remain byte-identical.

- [ ] **Step 4: Replace the audit implementation with explicit scopes**

Use this complete implementation:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

LEGACY_DIRS = ("code", "code1", "code2")


class AuditScope(StrEnum):
    TRACKED = "tracked"
    ALL_LOCAL = "all_local"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_legacy_paths(root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", *LEGACY_DIRS],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    paths: list[Path] = []
    for raw_value in completed.stdout.split(b"\0"):
        if not raw_value:
            continue
        git_path = PurePosixPath(os.fsdecode(raw_value))
        path = root.joinpath(*git_path.parts)
        if not path.is_file():
            raise FileNotFoundError(f"tracked legacy file is missing: {path}")
        paths.append(path)
    return tuple(sorted(paths, key=lambda item: item.relative_to(root).as_posix()))


def _all_local_legacy_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for dirname in LEGACY_DIRS:
        legacy_dir = root / dirname
        if not legacy_dir.is_dir():
            raise FileNotFoundError(f"missing legacy directory: {legacy_dir}")
        paths.extend(path for path in legacy_dir.rglob("*") if path.is_file())
    return tuple(sorted(paths, key=lambda item: item.relative_to(root).as_posix()))


def audit_legacy_outputs(
    root: Path,
    output_path: Path,
    scope: AuditScope,
) -> dict[str, Any]:
    root = root.resolve()
    if scope is AuditScope.TRACKED:
        source_paths = _tracked_legacy_paths(root)
    else:
        source_paths = _all_local_legacy_paths(root)

    files: dict[str, dict[str, object]] = {}
    for path in source_paths:
        relative_path = path.relative_to(root).as_posix()
        files[relative_path] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "scope": scope.value,
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
    parser.add_argument(
        "--scope",
        type=AuditScope,
        choices=tuple(AuditScope),
        required=True,
    )
    args = parser.parse_args()
    audit_legacy_outputs(args.repo_root, args.output, args.scope)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write tracked-scope tests before regenerating the manifest**

Add tests that create a temporary Git repository containing one tracked file and one ignored file. The tracked audit must include only the tracked file, while the all-local audit must include both. Use the following complete test helper and tests:

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from smc_repro.scripts.audit_legacy_outputs import AuditScope, audit_legacy_outputs


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_tracked_scope_ignores_untracked_runtime_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(root, "init")
    for dirname in ("code", "code1", "code2"):
        (root / dirname).mkdir()
    (root / ".gitignore").write_text("PM.txt\n__pycache__/\n", encoding="utf-8")
    (root / "code" / "DQN.py").write_text("print('tracked')\n", encoding="utf-8")
    (root / "code" / "PM.txt").write_text("runtime log\n", encoding="utf-8")
    _run_git(root, "add", ".gitignore", "code/DQN.py")

    tracked = audit_legacy_outputs(
        root,
        root / "tracked.json",
        AuditScope.TRACKED,
    )
    local = audit_legacy_outputs(
        root,
        root / "local.json",
        AuditScope.ALL_LOCAL,
    )

    assert set(tracked["files"]) == {"code/DQN.py"}
    assert "code/PM.txt" in local["files"]
    assert "code/PM.txt" not in tracked["files"]
```

- [ ] **Step 6: Generate the tracked manifest and update the repository equality test**

```bash
cd 扩刊/original_repro
python -m smc_repro.scripts.audit_legacy_outputs \
  --repo-root ../.. \
  --scope tracked \
  --output ../docs/audit/legacy_tracked_manifest.json
```

Update `test_legacy_immutable.py` so it invokes `audit_legacy_outputs(..., AuditScope.TRACKED)` into `tmp_path` and compares the generated manifest to `legacy_tracked_manifest.json`. Do not recursively scan all local files in that test.

Required assertions:

```python
assert committed_manifest["scope"] == "tracked"
assert generated_manifest == committed_manifest
assert all(not key.endswith("PM.txt") for key in committed_manifest["files"])
assert all("/__pycache__/" not in key for key in committed_manifest["files"])
assert all("/.idea/" not in key for key in committed_manifest["files"])
```

- [ ] **Step 7: Run focused and full tests**

```bash
cd 扩刊/original_repro
python -m pytest tests/test_legacy_audit.py tests/test_legacy_immutable.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
```

- [ ] **Step 8: Commit the tracked-manifest fix**

A detached worktree only sees committed files, so commit before running the clean-worktree proof.

```bash
git add 扩刊/docs/audit \
  扩刊/docs/superpowers/specs/2026-08-20-smc-preexperiment-readiness-design.md \
  扩刊/original_repro/src/smc_repro/scripts/audit_legacy_outputs.py \
  扩刊/original_repro/tests/test_legacy_audit.py \
  扩刊/original_repro/tests/test_legacy_immutable.py
git commit -m "fix: make legacy audit portable across clean clones"
```

- [ ] **Step 9: Prove the committed fix in a new clean worktree**

PowerShell:

```powershell
$repo = (git rev-parse --show-toplevel).Trim()
$wt = Join-Path $env:TEMP ("smc-gate15-green-" + [guid]::NewGuid().ToString("N"))
git worktree add --detach $wt HEAD
try {
  $env:PYTHONPATH = Join-Path $wt "扩刊\original_repro\src"
  & "C:\Users\LXT\smc_gate1_env\Scripts\python.exe" -m pytest `
    (Join-Path $wt "扩刊\original_repro\tests") -q
  if ($LASTEXITCODE -ne 0) {
    throw "Clean-worktree pytest failed."
  }
} finally {
  Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
  git worktree remove --force $wt
}
```

If the clean-worktree test fails, return to root-cause investigation, apply only the confirmed fix, rerun the local suite, and amend this task commit before continuing.

---

## Task 1: Harden Random Keys, Metadata, and Interval Invariants

**Files:**
- Modify: `扩刊/original_repro/src/smc_repro/seeding.py`
- Modify: `扩刊/original_repro/src/smc_repro/schemas.py`
- Modify: `扩刊/original_repro/src/smc_repro/instance_io.py`
- Modify: `扩刊/original_repro/src/smc_repro/timeline.py`
- Modify: `扩刊/original_repro/tests/test_seeding.py`
- Modify: `扩刊/original_repro/tests/test_common_random_numbers.py`
- Modify: `扩刊/original_repro/tests/test_schemas.py`
- Modify: `扩刊/original_repro/tests/test_instance_io.py`
- Modify: `扩刊/original_repro/tests/test_timeline.py`

**Interfaces:**
- Produces: collision-resistant `keyed_uniform(base_seed, *keys)` with stable typed encoding.
- Produces: read-only scalar metadata mappings.
- Produces: strictly positive recorded intervals and strictly positive gap-search durations.

- [ ] **Step 1: Add failing random-key collision tests**

```python
import pytest

from smc_repro.seeding import keyed_uniform


def test_keyed_uniform_distinguishes_delimiter_placement() -> None:
    assert keyed_uniform(7, "a|b", "c") != keyed_uniform(7, "a", "b|c")


def test_keyed_uniform_distinguishes_value_types() -> None:
    assert keyed_uniform(7, 1) != keyed_uniform(7, "1")
    assert keyed_uniform(7, True) != keyed_uniform(7, 1)


def test_keyed_uniform_rejects_nonfinite_float_keys() -> None:
    with pytest.raises(ValueError, match="finite"):
        keyed_uniform(7, float("nan"))


def test_keyed_uniform_rejects_unsupported_key_types() -> None:
    with pytest.raises(TypeError, match="random-stream keys"):
        keyed_uniform(7, [1, 2])
```

Run:

```bash
python -m pytest tests/test_seeding.py tests/test_common_random_numbers.py -q
```

Expected: at least the delimiter and type tests fail against the current implementation.

- [ ] **Step 2: Replace the key encoding**

Use this implementation in `seeding.py` while retaining `set_global_seed()`:

```python
import hashlib
import math
import struct


def _encode_key(value: object) -> bytes:
    if value is None:
        tag = b"N"
        body = b""
    elif isinstance(value, bool):
        tag = b"B"
        body = b"1" if value else b"0"
    elif isinstance(value, int):
        tag = b"I"
        body = str(value).encode("ascii")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("floating-point random-stream keys must be finite")
        tag = b"F"
        body = value.hex().encode("ascii")
    elif isinstance(value, str):
        tag = b"S"
        body = value.encode("utf-8")
    else:
        raise TypeError(
            "random-stream keys must be None, bool, int, finite float, or str; "
            f"got {type(value).__name__}"
        )
    return tag + struct.pack(">Q", len(body)) + body


def keyed_uniform(base_seed: int, *keys: object) -> float:
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    payload = b"".join(_encode_key(value) for value in (base_seed, *keys))
    digest = hashlib.blake2b(payload, digest_size=8, person=b"smc-crn1").digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / float(1 << 64)
```

- [ ] **Step 3: Add failing metadata immutability tests**

```python
import pytest

from smc_repro.schemas import InstanceSpec, MachineSpec, JobSpec, OperationSpec


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
```

- [ ] **Step 4: Implement scalar read-only metadata**

In `schemas.py`, add:

```python
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias

MetadataScalar: TypeAlias = str | int | float | bool | None


def _freeze_metadata(
    metadata: Mapping[str, MetadataScalar],
) -> Mapping[str, MetadataScalar]:
    copied: dict[str, MetadataScalar] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError("metadata values must be JSON scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("floating-point metadata values must be finite")
        copied[key] = value
    return MappingProxyType(copied)
```

Change both metadata annotations to:

```python
metadata: Mapping[str, MetadataScalar] = field(default_factory=dict)
```

At the end of `InstanceSpec.__post_init__` and `ScheduleInterval.__post_init__` call:

```python
object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
```

In `instance_io.instance_to_dict()` serialize with:

```text
"metadata": dict(instance.metadata),
```

- [ ] **Step 5: Add positive-duration tests for every recorded interval and timeline query**

```python
import pytest

from smc_repro.schemas import IntervalType, ScheduleInterval
from smc_repro.timeline import MachineTimeline


@pytest.mark.parametrize("interval_type", [IntervalType.SETUP, IntervalType.PM, IntervalType.CM])
def test_nonprocess_recorded_intervals_require_positive_duration(
    interval_type: IntervalType,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        ScheduleInterval(0, 2.0, 2.0, interval_type)


def test_timeline_rejects_zero_duration_search() -> None:
    with pytest.raises(ValueError, match="positive"):
        MachineTimeline(0).earliest_feasible_start(0.0, 0.0)
```

Update schema validation to reject `end <= start` for every interval. An environment with `setup_time == 0` must omit the SETUP interval rather than record a zero-duration interval.

Update timeline validation to:

```python
if duration <= 0 or earliest < 0:
    raise ValueError("duration must be positive and earliest must be non-negative")
```

- [ ] **Step 6: Run all relevant gates**

```bash
python -m pytest tests/test_seeding.py tests/test_common_random_numbers.py \
  tests/test_schemas.py tests/test_instance_io.py tests/test_timeline.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
```

- [ ] **Step 7: Confirm the committed instance reference manifest did not change**

```bash
python - <<'PY'
from hashlib import sha256
from pathlib import Path

path = Path("artifacts/banks/release/manifest.json")
actual = sha256(path.read_bytes()).hexdigest()
expected = "68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515"
assert actual == expected, (actual, expected)
print(actual)
PY
```

- [ ] **Step 8: Commit**

```bash
git add 扩刊/original_repro/src/smc_repro/seeding.py \
  扩刊/original_repro/src/smc_repro/schemas.py \
  扩刊/original_repro/src/smc_repro/instance_io.py \
  扩刊/original_repro/src/smc_repro/timeline.py \
  扩刊/original_repro/tests
git commit -m "fix: harden deterministic streams and immutable schemas"
```

---

## Task 2: Harden Metrics and Maintenance-Interval Validation

**Files:**
- Modify: `扩刊/original_repro/src/smc_repro/metrics.py`
- Modify: `扩刊/original_repro/src/smc_repro/validator.py`
- Modify: `扩刊/original_repro/tests/test_metrics.py`
- Modify: `扩刊/original_repro/tests/test_validator.py`

**Interfaces:**
- Produces: bounded, independently recomputable utilization metrics.
- Produces: `mean_flow_time`, `on_time_rate`, `total_downtime`, and `failure_count`.
- Produces: semantic validation for SETUP, PM, CM, and the final scheduling horizon.

- [ ] **Step 1: Add failing horizon and utilization-bound tests**

Use a complete hand instance with two machines and add a PM interval after all processing. The validator must reject it. Also add a direct unit test for the horizon-duration helper.

```python
import pytest

from smc_repro.metrics import duration_within_horizon
from smc_repro.schemas import IntervalType, ScheduleInterval


def test_duration_within_horizon_clips_both_ends() -> None:
    interval = ScheduleInterval(0, 8.0, 14.0, IntervalType.PM)
    assert duration_within_horizon(interval, 10.0) == pytest.approx(2.0)


def test_duration_within_horizon_returns_zero_after_horizon() -> None:
    interval = ScheduleInterval(0, 11.0, 12.0, IntervalType.PM)
    assert duration_within_horizon(interval, 10.0) == 0.0
```

- [ ] **Step 2: Add failing semantic-duration tests**

Required cases:

```text
1. SETUP duration differs from machine.setup_time -> error.
2. PM duration differs from machine.pm_duration -> error.
3. CM duration differs from machine.cm_duration -> error.
4. Complete schedule contains SETUP/PM/CM ending after makespan -> error.
5. setup_time == 0 and no SETUP interval -> valid.
```

Use error substrings `setup duration`, `PM duration`, `CM duration`, and `after final process horizon`.

- [ ] **Step 3: Extend `ScheduleMetrics`**

The dataclass must contain exactly these fields in this order:

```python
@dataclass(frozen=True)
class ScheduleMetrics:
    makespan: float
    paper_trave: float
    total_tardiness: float
    mean_tardiness: float
    weighted_tardiness: float
    tardy_rate: float
    on_time_rate: float
    mean_flow_time: float
    paper_uave: float
    standard_utilization: float
    availability_adjusted_utilization: float
    total_process_time: float
    total_setup_time: float
    total_pm_time: float
    total_cm_time: float
    total_downtime: float
    pm_count: int
    cm_count: int
    failure_count: int
```

For this original-conference model, `failure_count == cm_count`. Keep both names because later analyses distinguish event meaning from interval count.

- [ ] **Step 4: Implement horizon clipping and metric assertions**

Add:

```python
def duration_within_horizon(interval: ScheduleInterval, horizon: float) -> float:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    left = max(0.0, interval.start)
    right = min(horizon, interval.end)
    return max(0.0, right - left)
```

Calculate downtime only within `[0, makespan]`. Compute:

```python
on_time_rate = 1.0 - tardy_rate
mean_flow_time = sum(
    job_completions[job.job_id] - job.arrival_time for job in instance.jobs
) / len(instance.jobs)
total_downtime = pm_time + cm_time
```

After computing utilization, use explicit assertions rather than clipping:

```python
for name, value in (
    ("paper_uave", paper_uave),
    ("standard_utilization", standard_utilization),
    ("availability_adjusted_utilization", availability_adjusted_utilization),
):
    if not 0.0 <= value <= 1.0 + 1e-9:
        raise ValueError(f"{name} is outside [0, 1]: {value}")
```

- [ ] **Step 5: Implement semantic interval validation**

Inside `validate_schedule()`, after confirming a valid machine id, compare non-PROCESS duration to the machine specification with tolerance `1e-9`:

```python
machine = instance.machines[interval.machine_id]
if interval.interval_type is IntervalType.SETUP:
    expected_duration = machine.setup_time
    label = "setup duration"
elif interval.interval_type is IntervalType.PM:
    expected_duration = machine.pm_duration
    label = "PM duration"
elif interval.interval_type is IntervalType.CM:
    expected_duration = machine.cm_duration
    label = "CM duration"
else:
    expected_duration = None
    label = ""

if expected_duration is not None and abs(interval.duration - expected_duration) > 1e-9:
    errors.append(
        f"{label} mismatch on machine {interval.machine_id}: "
        f"{interval.duration} != {expected_duration}"
    )
```

When `require_complete=True`, determine `process_horizon` from the latest PROCESS end. Report any non-PROCESS interval with `end > process_horizon + 1e-9` as `after final process horizon`.

- [ ] **Step 6: Run tests and static checks**

```bash
python -m pytest tests/test_metrics.py tests/test_validator.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
```

- [ ] **Step 7: Commit**

```bash
git add 扩刊/original_repro/src/smc_repro/metrics.py \
  扩刊/original_repro/src/smc_repro/validator.py \
  扩刊/original_repro/tests/test_metrics.py \
  扩刊/original_repro/tests/test_validator.py
git commit -m "fix: enforce maintenance semantics and bounded metrics"
```

---

## Task 3: Materialize and Verify the Synthetic Instance Bank; Commit an Exact Environment Snapshot

**Files:**
- Create: `扩刊/original_repro/src/smc_repro/scripts/verify_instance_bank.py`
- Create: `扩刊/original_repro/src/smc_repro/scripts/capture_environment.py`
- Create: `扩刊/original_repro/tests/test_instance_bank_verification.py`
- Create: `扩刊/docs/audit/environment_5090_resolved.json` by running the script
- Modify: `扩刊/original_repro/README.md`

**Interfaces:**
- Consumes: committed reference `artifacts/banks/release/manifest.json`.
- Produces: locally generated `artifacts/banks/materialized/` with 1540 gzip files.
- Produces: `verify_instance_bank(reference_manifest, bank_root, expected_manifest_sha256) -> BankVerificationReport`.
- Produces: machine-portable environment JSON without editable absolute paths.

- [ ] **Step 1: Confirm that no external dataset is required**

Add this exact scope note to README:

```text
The original SMC conference reproduction uses only programmatically generated synthetic dynamic-FJSP instances. Do not download Brandimarte, Hurink, OR-Library, Taillard, or other external benchmark sets in this phase. Those benchmarks belong to the later GNN-upgrade evaluation and must not be mixed into the original-paper results.
```

- [ ] **Step 2: Write failing bank-verification tests**

The tests must cover:

```text
1. Exact valid bank passes.
2. Missing gzip fails and names the missing relative path.
3. Modified gzip fails and reports SHA mismatch.
4. Generated manifest bytes differ from reference -> fail.
5. Wrong expected reference-manifest SHA -> fail.
6. Extra gzip not listed in manifest -> fail.
7. Absolute, backslash, or `..` manifest paths fail before any file access.
```

Create a tiny reference bank in `tmp_path` with two deterministic gzip files by calling `save_instance()`. Do not include committed 1540-file data in unit tests.

- [ ] **Step 3: Implement `verify_instance_bank.py`**

Use this public report type and verification contract:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from smc_repro.instance_io import instance_sha256, load_instance


@dataclass(frozen=True)
class BankVerificationReport:
    reference_manifest: str
    reference_manifest_sha256: str
    generated_manifest: str
    generated_manifest_sha256: str
    expected_file_count: int
    verified_file_count: int
    ok: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_instance_bank(
    reference_manifest: Path,
    bank_root: Path,
    expected_manifest_sha256: str,
) -> BankVerificationReport:
    reference_manifest = reference_manifest.resolve()
    bank_root = bank_root.resolve()
    generated_manifest = bank_root / "manifest.json"

    actual_reference_sha = _sha256(reference_manifest)
    if actual_reference_sha != expected_manifest_sha256:
        raise ValueError(
            "reference manifest SHA-256 mismatch: "
            f"{actual_reference_sha} != {expected_manifest_sha256}"
        )
    if not generated_manifest.is_file():
        raise FileNotFoundError(f"generated manifest is missing: {generated_manifest}")

    reference_bytes = reference_manifest.read_bytes()
    generated_bytes = generated_manifest.read_bytes()
    if generated_bytes != reference_bytes:
        raise ValueError("generated manifest is not byte-identical to the reference manifest")

    manifest = json.loads(reference_bytes.decode("utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("bank manifest 'files' must be a list")

    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each bank manifest entry must be an object")
        relative_path = entry.get("path")
        expected_sha = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha, str):
            raise ValueError("bank entry requires string path and sha256")
        if relative_path in expected_paths:
            raise ValueError(f"duplicate bank path in manifest: {relative_path}")
        expected_paths.add(relative_path)
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ValueError(f"unsafe bank path in manifest: {relative_path!r}")
        path = (bank_root / Path(*pure_path.parts)).resolve()
        if not path.is_relative_to(bank_root):
            raise ValueError(f"bank path escapes root: {relative_path!r}")
        if not path.is_file():
            raise FileNotFoundError(f"bank instance is missing: {relative_path}")
        actual_sha = instance_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"bank instance SHA-256 mismatch for {relative_path}: "
                f"{actual_sha} != {expected_sha}"
            )
        load_instance(path)

    observed_paths = {
        path.relative_to(bank_root).as_posix()
        for path in bank_root.rglob("*.json.gz")
        if path.is_file()
    }
    extra_paths = sorted(observed_paths - expected_paths)
    if extra_paths:
        raise ValueError(f"bank contains unlisted gzip files: {extra_paths[:10]}")

    return BankVerificationReport(
        reference_manifest=str(reference_manifest),
        reference_manifest_sha256=actual_reference_sha,
        generated_manifest=str(generated_manifest),
        generated_manifest_sha256=_sha256(generated_manifest),
        expected_file_count=len(expected_paths),
        verified_file_count=len(observed_paths),
        ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_instance_bank(
        args.reference,
        args.bank_root,
        args.expected_manifest_sha256,
    )
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Materialize the full bank locally**

PowerShell:

```powershell
cd 扩刊\original_repro
Remove-Item -Recurse -Force artifacts\banks\materialized -ErrorAction SilentlyContinue
python -m smc_repro.scripts.build_instance_banks `
  --output-root artifacts\banks\materialized `
  --test-repetitions 20 `
  --train-seeds 0 1 2 3 4 `
  --train-episodes 200 `
  --base-seed 20260819
python -m smc_repro.scripts.verify_instance_bank `
  --reference artifacts\banks\release\manifest.json `
  --bank-root artifacts\banks\materialized `
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 `
  --report artifacts\preflight\bank_verification.json
```

Bash:

```bash
cd 扩刊/original_repro
rm -rf artifacts/banks/materialized
python -m smc_repro.scripts.build_instance_banks \
  --output-root artifacts/banks/materialized \
  --test-repetitions 20 \
  --train-seeds 0 1 2 3 4 \
  --train-episodes 200 \
  --base-seed 20260819
python -m smc_repro.scripts.verify_instance_bank \
  --reference artifacts/banks/release/manifest.json \
  --bank-root artifacts/banks/materialized \
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 \
  --report artifacts/preflight/bank_verification.json
```

Expected report:

```json
{
  "expected_file_count": 1540,
  "verified_file_count": 1540,
  "ok": true
}
```

- [ ] **Step 5: Implement `capture_environment.py`**

The committed snapshot must contain only portable values and must be generated by this exact implementation:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import torch


_NAME_SEPARATORS = re.compile(r"[-_.]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_name(name: str) -> str:
    normalized = _NAME_SEPARATORS.sub("-", name).lower().strip("-")
    if not normalized:
        raise ValueError(f"invalid installed distribution name: {name!r}")
    return normalized


def _installed_packages() -> list[dict[str, str]]:
    versions: dict[str, str] = {}
    display_names: dict[str, str] = {}
    for distribution in distributions():
        raw_name = distribution.metadata.get("Name")
        if raw_name is None:
            continue
        name = str(raw_name).strip()
        normalized = _normalized_name(name)
        version = str(distribution.version)
        previous = versions.get(normalized)
        if previous is not None and previous != version:
            raise RuntimeError(
                "conflicting installed versions for normalized package "
                f"{normalized!r}: {previous!r} versus {version!r}"
            )
        versions[normalized] = version
        display_names.setdefault(normalized, name)
    return [
        {"name": display_names[normalized], "version": versions[normalized]}
        for normalized in sorted(versions)
    ]


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"unexpected git commit value: {commit!r}")
    return commit


def collect_environment(repo_root: Path, bank_manifest: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    bank_manifest = bank_manifest.resolve()
    if not bank_manifest.is_file():
        raise FileNotFoundError(f"bank manifest is missing: {bank_manifest}")

    required_environment = {
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, expected in required_environment.items():
        observed = os.environ.get(name)
        if observed != expected:
            raise RuntimeError(
                f"{name} must be set before Python starts: expected {expected!r}, "
                f"observed {observed!r}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; expected the connected RTX 5090")
    device = torch.device("cuda:0")
    vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    cuda_smoke_result = float(torch.sum(vector * vector).item())
    torch.cuda.synchronize(device)
    if cuda_smoke_result != 14.0:
        raise RuntimeError(f"unexpected CUDA smoke result: {cuda_smoke_result}")

    capability = torch.cuda.get_device_capability(device)
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "compiled_cuda_arches": list(torch.cuda.get_arch_list()),
        "cuda_smoke_result": cuda_smoke_result,
        "PYTHONHASHSEED": required_environment["PYTHONHASHSEED"],
        "CUBLAS_WORKSPACE_CONFIG": required_environment["CUBLAS_WORKSPACE_CONFIG"],
        "bank_manifest_sha256": _sha256(bank_manifest),
        "packages": _installed_packages(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = collect_environment(args.repo_root, args.bank_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
```

The snapshot deliberately uses `importlib.metadata.distributions()` rather than `pip freeze`, because editable freeze lines contain local absolute paths. It rejects duplicate normalized package names with conflicting versions and refuses to run unless the deterministic environment variables were set before interpreter startup. `git_commit` records the source commit from which this machine snapshot was captured; after later implementation commits it is historical provenance and is **not** required to equal the final experiment-code HEAD. Every run contract records the live current HEAD separately.

Required command:

```powershell
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m smc_repro.scripts.capture_environment `
  --repo-root ..\.. `
  --bank-manifest artifacts\banks\release\manifest.json `
  --output ..\docs\audit\environment_5090_resolved.json
```

The script must verify a real CUDA tensor result of 14.0, as `verify_hardware.py` does.

- [ ] **Step 6: Run tests and inspect the environment snapshot**

```bash
python -m pytest tests/test_instance_bank_verification.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python - <<'PY'
import json
from pathlib import Path

path = Path("../docs/audit/environment_5090_resolved.json")
data = json.loads(path.read_text(encoding="utf-8"))
assert data["gpu_name"] == "NVIDIA GeForce RTX 5090"
assert data["torch_version"].startswith("2.10.0")
assert data["bank_manifest_sha256"] == (
    "68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515"
)
assert data["PYTHONHASHSEED"] == "0"
assert data["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
print("package_count=", len(data["packages"]))
PY
```

- [ ] **Step 7: Update README with exact materialization/preflight commands**

The README must state:

```text
- Reference manifest is committed; 1540 gzip files are generated locally.
- Expected manifest SHA is fixed.
- Exact PowerShell and Bash commands are provided.
- External benchmark data is deferred to the GNN phase.
- Formal runs must record the bank-manifest SHA.
```

- [ ] **Step 8: Commit source, tests, README, and portable environment metadata**

```bash
git add 扩刊/original_repro/src/smc_repro/scripts/verify_instance_bank.py \
  扩刊/original_repro/src/smc_repro/scripts/capture_environment.py \
  扩刊/original_repro/tests/test_instance_bank_verification.py \
  扩刊/original_repro/README.md \
  扩刊/docs/audit/environment_5090_resolved.json
git commit -m "feat: materialize verified banks and capture runtime metadata"
```

Do not commit `artifacts/banks/materialized/**/*.json.gz` or `artifacts/preflight/bank_verification.json`.

---

# Gate 2 — Lock Reproduction Profiles, State, Rewards, and Rules

## Task 4: Add Strict Reproduction Profiles and the Ambiguity Register

**Files:**
- Create: `扩刊/original_repro/src/smc_repro/config.py`
- Create: `扩刊/original_repro/configs/legacy_snapshot.yaml`
- Create: `扩刊/original_repro/configs/paper_repro.yaml`
- Create: `扩刊/original_repro/configs/corrected_smc.yaml`
- Create: `扩刊/original_repro/configs/smoke.yaml`
- Create: `扩刊/original_repro/configs/ambiguities.json`
- Create: `扩刊/original_repro/tests/test_config.py`

**Interfaces:**
- Produces: strict `ReproductionProfile` and `load_profile(path) -> ReproductionProfile`.
- Produces: canonical `profile_sha256(profile) -> str`.
- Fixes every paper/code ambiguity before agent implementation.

- [ ] **Step 1: Create the ambiguity register**

Use this exact content:

```json
{
  "schema_version": 1,
  "items": [
    {
      "id": "A-001",
      "topic": "state feature order",
      "paper": ["crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "tr_std"],
      "source_code": ["u_ave", "u_std", "crj_ave", "crj_std", "tr_ave", "tr_std"],
      "resolution": {
        "legacy_snapshot": "source_code",
        "paper_repro": "paper",
        "corrected_smc": "paper"
      }
    },
    {
      "id": "A-002",
      "topic": "lower network reward context",
      "paper": "says chosen reward one-hot but also states seven-dimensional input",
      "source_code": "one scalar equal to upper-network max Q",
      "resolution": {
        "legacy_snapshot": "max_q_scalar",
        "paper_repro": "reward_id_scalar",
        "corrected_smc": "reward_id_scalar"
      },
      "future_sensitivity": ["max_q_scalar", "reward_id_one_hot"]
    },
    {
      "id": "A-003",
      "topic": "network depth",
      "paper": "two 10-unit hidden layers in the upper network and two 50-unit hidden layers in the lower network",
      "source_code": "three 10-unit upper hidden layers and seven 50-unit lower hidden layers",
      "resolution": {
        "legacy_snapshot": "source_code",
        "paper_repro": "paper",
        "corrected_smc": "paper"
      }
    },
    {
      "id": "A-004",
      "topic": "partial-episode TR state",
      "paper": "final-completion formula is not fully defined for unfinished jobs",
      "source_code": "workload pressure using OPT plus ETL versus due window",
      "resolution": {
        "legacy_snapshot": "legacy_workload_pressure",
        "paper_repro": "legacy_workload_pressure",
        "corrected_smc": "projected_completion_tardiness_ratio"
      },
      "note": "paper_repro uses the only executable source interpretation for state/reward, while final evaluation uses the paper completion-time metric"
    },
    {
      "id": "A-005",
      "topic": "classic-rule decoding",
      "paper": "FIFO/EDD/MRT/SPT/LPT names are given, but neither the machine decoder nor the exact SPT/LPT workload scope is uniquely specified",
      "resolution": "FIFO uses arrival time; EDD uses absolute due date; MRT uses total remaining nominal work; SPT/LPT use the next operation's mean eligible-machine nominal time; all five use earliest predicted completion under the active profile and are labelled FIFO+ECT, EDD+ECT, MRT+ECT, SPT+ECT, LPT+ECT",
      "note": "predicted completion includes setup only when the active profile enables source tool-change semantics and never samples PM/CM inside a rule score"
    },
    {
      "id": "A-006",
      "topic": "failure timing",
      "paper_and_code": "corrective maintenance is sampled before processing and delays the operation; no within-operation interruption is implemented",
      "resolution": {
        "legacy_snapshot": "legacy_prestart_cdf",
        "paper_repro": "legacy_prestart_cdf",
        "corrected_smc": "prestart_conditional_interval_risk"
      },
      "note": "do not claim physical within-operation breakdown interruption in the original-conference rerun"
    },
    {
      "id": "A-007",
      "topic": "initial observation",
      "paper": "the method section defines the state features but does not state that the first observation is an all-zero vector",
      "source_code": "training starts from an explicit six-dimensional zero vector rather than querying the environment",
      "resolution": {
        "legacy_snapshot": "zero",
        "paper_repro": "environment",
        "corrected_smc": "environment"
      },
      "future_sensitivity": ["zero", "environment"]
    },
    {
      "id": "A-008",
      "topic": "paper A2 priority for unfinished jobs",
      "paper": "uses completion/tardiness notation that is not uniquely defined for an unfinished job at a constructive decision epoch",
      "source_code": "uses the current constructive completion proxy and a differently parenthesized urgency expression",
      "resolution": {
        "legacy_snapshot": "source_code",
        "paper_repro": "use decision_time as the overdue completion proxy and reproduce the printed non-overdue expression",
        "corrected_smc": "use decision_time as the overdue completion proxy and projected/remaining-work values for state diagnostics"
      },
      "note": "all three implementations must be separately named and unit-tested; do not hide this choice behind Rule 4/5/6 labels"
    },
    {
      "id": "A-009",
      "topic": "embedded local insertion",
      "paper": "claims insertion into the earliest feasible idle slot",
      "source_code": "sets earliest_start no earlier than the machine tail, which makes historical gaps unreachable in normal execution",
      "resolution": {
        "legacy_snapshot": "tail_append",
        "paper_repro": "tail_append_with_discrepancy_recorded",
        "corrected_smc": "tail_append"
      },
      "note": "retrospective insertion is deferred because health-dependent durations and maintenance would require chronological replay of later machine events; do not implement an internally inconsistent partial fix"
    },
    {
      "id": "A-010",
      "topic": "setup and tool-change time",
      "paper": "the formulation assumes setup times are negligible and the printed B1/B2 equations do not include setup",
      "source_code": "change_cutter() adds a machine-specific tool-change delay when either the job changes machine or the machine changes job",
      "resolution": {
        "legacy_snapshot": "source_tool_change",
        "paper_repro": "none",
        "corrected_smc": "source_tool_change"
      },
      "note": "paper_repro emits no SETUP interval; corrected_smc keeps explicit source tool-change costs as an audited correction"
    },
    {
      "id": "A-011",
      "topic": "urgency-level semantics",
      "paper": "Table I states 1=highest and 3=lowest, while the experimental table caption reverses the wording",
      "source_code": "the due-window formula and rule denominators behave as 1=high, 2=medium, 3=low",
      "resolution": "all profiles use 1=high, 2=medium, 3=low and preserve that meaning in metadata, tests, weights, and rule formulas"
    },
    {
      "id": "A-012",
      "topic": "absolute due date versus due window",
      "paper": "the printed equation omits the arrival-time addition",
      "source_code": "initial jobs use arrival zero and dynamic jobs use arrival + urgency-scaled estimated work",
      "resolution": "all stored due dates are absolute; for every job use arrival_time + (0.2 + 0.5*urgency)*estimated_work, and subtract arrival only when a relative due window is required"
    }
  ]
}
```

- [ ] **Step 2: Write strict-loader tests**

Required tests:

```text
1. All four committed YAML files load.
2. Unknown top-level key fails.
3. Unknown nested key fails.
4. Missing required field fails.
5. lower_context=reward_id_one_hot implies lower_input_dim=8.
6. max_q_scalar and reward_id_scalar imply lower_input_dim=7.
7. Feature order contains exactly six unique known names.
8. Setup mode is one of `none` or `source_tool_change`; paper_repro is `none`.
9. Urgency semantics are locked to 1=high, 2=medium, 3=low.
10. Probability/health thresholds and epsilon are in valid ranges.
11. YAML round-trip to canonical dict yields stable profile SHA.
12. `ambiguities.json` has unique IDs A-001 through A-012 and every profile resolution is explicit.
```

- [ ] **Step 3: Implement enums and profile dataclasses**

`config.py` must define these enums:

```python
class ProfileName(StrEnum):
    LEGACY_SNAPSHOT = "legacy_snapshot"
    PAPER_REPRO = "paper_repro"
    CORRECTED_SMC = "corrected_smc"


class InitialObservationMode(StrEnum):
    ZERO = "zero"
    ENVIRONMENT = "environment"


class TRFeatureMode(StrEnum):
    LEGACY_WORKLOAD_PRESSURE = "legacy_workload_pressure"
    PROJECTED_COMPLETION_TARDINESS_RATIO = "projected_completion_tardiness_ratio"


class RuleSetName(StrEnum):
    LEGACY = "legacy"
    PAPER = "paper"


class SetupMode(StrEnum):
    NONE = "none"
    SOURCE_TOOL_CHANGE = "source_tool_change"


class LowerContextMode(StrEnum):
    MAX_Q_SCALAR = "max_q_scalar"
    REWARD_ID_SCALAR = "reward_id_scalar"
    REWARD_ID_ONE_HOT = "reward_id_one_hot"


class FailureMode(StrEnum):
    LEGACY_PRESTART_CDF = "legacy_prestart_cdf"
    PRESTART_CONDITIONAL_INTERVAL_RISK = "prestart_conditional_interval_risk"


class WearMode(StrEnum):
    LEGACY_PER_OPERATION = "legacy_per_operation"
    EFFECTIVE_AGE = "effective_age"
```

The top-level profile contains typed nested dataclasses for `state`, `architecture`, `reliability`, `scheduling`, `reward`, and `training`. It must expose:

```python
@property
def lower_input_dim(self) -> int:
    return 8 if self.architecture.lower_context is LowerContextMode.REWARD_ID_ONE_HOT else 7
```

Use a manual key-set check for each mapping before construction. Do not silently ignore unknown YAML fields.

Canonical hash:

```python
def profile_sha256(profile: ReproductionProfile) -> str:
    payload = json.dumps(
        profile.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
```

- [ ] **Step 4: Create `legacy_snapshot.yaml`**

```yaml
schema_version: 1
profile: legacy_snapshot
state:
  order: [u_ave, u_std, crj_ave, crj_std, tr_ave, tr_std]
  initial_observation: zero
  tr_feature: legacy_workload_pressure
  utilization_feature: paper_uave
architecture:
  upper_hidden: [10, 10, 10]
  lower_hidden: [50, 50, 50, 50, 50, 50, 50]
  lower_context: max_q_scalar
reliability:
  failure_mode: legacy_prestart_cdf
  wear_mode: legacy_per_operation
  pm_enabled: true
  pm_failure_threshold: 0.2
  pm_health_threshold: 30.0
  cm_age_repair_factor: 0.5
  high_load_failure_bias: true
scheduling:
  rule_set: legacy
  setup_mode: source_tool_change
  local_insertion: false
  explicit_nonprocess_intervals: true
reward:
  tardiness_mode: legacy
  utilization_mode: legacy
training:
  episodes: 200
  replay_capacity: 2000
  batch_size: 16
  gamma: 0.95
  learning_rate: 0.001
  target_update_steps: 200
  epsilon_start: 0.6
  epsilon_end: 0.01
  epsilon_decrement: 0.0001
  deterministic: true
```

- [ ] **Step 5: Create `paper_repro.yaml`**

```yaml
schema_version: 1
profile: paper_repro
state:
  order: [crj_ave, crj_std, u_ave, u_std, tr_ave, tr_std]
  initial_observation: environment
  tr_feature: legacy_workload_pressure
  utilization_feature: paper_uave
architecture:
  upper_hidden: [10, 10]
  lower_hidden: [50, 50]
  lower_context: reward_id_scalar
reliability:
  failure_mode: legacy_prestart_cdf
  wear_mode: legacy_per_operation
  pm_enabled: true
  pm_failure_threshold: 0.2
  pm_health_threshold: 30.0
  cm_age_repair_factor: 0.5
  high_load_failure_bias: true
scheduling:
  rule_set: paper
  setup_mode: none
  local_insertion: false
  explicit_nonprocess_intervals: true
reward:
  tardiness_mode: paper
  utilization_mode: paper
training:
  episodes: 200
  replay_capacity: 2000
  batch_size: 16
  gamma: 0.95
  learning_rate: 0.001
  target_update_steps: 200
  epsilon_start: 0.6
  epsilon_end: 0.01
  epsilon_decrement: 0.0001
  deterministic: true
```

- [ ] **Step 6: Create `corrected_smc.yaml`**

```yaml
schema_version: 1
profile: corrected_smc
state:
  order: [crj_ave, crj_std, u_ave, u_std, tr_ave, tr_std]
  initial_observation: environment
  tr_feature: projected_completion_tardiness_ratio
  utilization_feature: standard_utilization
architecture:
  upper_hidden: [10, 10]
  lower_hidden: [50, 50]
  lower_context: reward_id_scalar
reliability:
  failure_mode: prestart_conditional_interval_risk
  wear_mode: effective_age
  pm_enabled: true
  pm_failure_threshold: 0.2
  pm_health_threshold: 30.0
  cm_age_repair_factor: 0.5
  high_load_failure_bias: false
scheduling:
  rule_set: paper
  setup_mode: source_tool_change
  local_insertion: false
  explicit_nonprocess_intervals: true
reward:
  tardiness_mode: paper
  utilization_mode: paper
training:
  episodes: 200
  replay_capacity: 2000
  batch_size: 16
  gamma: 0.95
  learning_rate: 0.001
  target_update_steps: 200
  epsilon_start: 0.6
  epsilon_end: 0.01
  epsilon_decrement: 0.0001
  deterministic: true
```

- [ ] **Step 7: Create `smoke.yaml`**

This file is an override, not an independent scientific profile:

```yaml
schema_version: 1
training:
  episodes: 3
  replay_capacity: 128
  batch_size: 4
  target_update_steps: 5
  epsilon_start: 0.2
  epsilon_end: 0.0
  epsilon_decrement: 0.1
```

The loader must support a base profile plus one strict override. Unknown override keys remain errors.

- [ ] **Step 8: Run tests and commit**

```bash
python -m pytest tests/test_config.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
git add 扩刊/original_repro/src/smc_repro/config.py \
  扩刊/original_repro/configs \
  扩刊/original_repro/tests/test_config.py
git commit -m "feat: lock strict SMC reproduction profiles"
```

---

## Task 5: Add Runtime State, Named Observations, and Reward Functions

**Files:**
- Create: `扩刊/original_repro/src/smc_repro/runtime.py`
- Create: `扩刊/original_repro/src/smc_repro/observations.py`
- Create: `扩刊/original_repro/src/smc_repro/rewards.py`
- Create: `扩刊/original_repro/tests/test_observations.py`
- Create: `扩刊/original_repro/tests/test_rewards.py`

**Interfaces:**
- Produces: `MachineRuntime`, `ScheduleRuntime`, `ScheduleObservation`.
- Produces: observation vectorization by named profile order.
- Produces: legacy and paper reward functions with explicit zero/equality behavior.

- [ ] **Step 1: Define mutable runtime objects**

`runtime.py` must define:

```python
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
```

Constructor helper:

```python
def create_runtime(instance: InstanceSpec) -> ScheduleRuntime:
    return ScheduleRuntime(
        instance=instance,
        next_op_index=[0 for _ in instance.jobs],
        timelines=[MachineTimeline(machine.machine_id) for machine in instance.machines],
        machines=[MachineRuntime(machine.machine_id) for machine in instance.machines],
        last_machine_by_job=[None for _ in instance.jobs],
    )
```

Required invariants:

```text
next_op_index[j] in [0, len(job.operations)]
machine/runtime ids are contiguous
last_machine_by_job has one entry per job
runtime instance is never replaced mid-episode
```

- [ ] **Step 2: Define named observation output**

`observations.py` must define:

```python
FEATURE_NAMES = (
    "crj_ave",
    "crj_std",
    "u_ave",
    "u_std",
    "tr_ave",
    "tr_std",
)


@dataclass(frozen=True)
class ScheduleObservation:
    crj_ave: float
    crj_std: float
    u_ave: float
    u_std: float
    tr_ave: float
    tr_std: float

    def vector(self, order: tuple[str, ...]) -> np.ndarray:
        if len(order) != 6 or set(order) != set(FEATURE_NAMES):
            raise ValueError("state feature order must contain each known feature exactly once")
        values = np.asarray([getattr(self, name) for name in order], dtype=np.float32)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("state vector must contain six finite values")
        return values
```

- [ ] **Step 3: Implement observation formulas**

Use these exact per-job definitions:

```text
assigned_work_i:
  Sum of PROCESS durations already scheduled for job i.

remaining_nominal_work_i:
  Sum over unscheduled operations of the mean positive eligible nominal processing time.

completion_ratio_i:
  assigned_work / (assigned_work + remaining_nominal_work), or 0 when denominator is 0.

legacy_workload_pressure_i:
  max(0, [assigned_work + remaining_nominal_work - (due-arrival)] /
         [assigned_work + remaining_nominal_work]), or 0 when denominator is 0.

projected_completion_i:
  If no operation is assigned, arrival_time + remaining_nominal_work.
  Otherwise, end time of the latest assigned operation for this job + remaining_nominal_work.

projected_completion_tardiness_ratio_i:
  max(0, projected_completion_i - due_date) /
  max(assigned_work + remaining_nominal_work, 1e-12).
```

Use these machine definitions:

```text
paper utilization per machine:
  PROCESS busy duration / final PROCESS end; 0 for unused machines.

standard current utilization:
  total PROCESS duration / (machine_count * current_horizon);
  current_horizon is max end over all recorded intervals; 0 when no interval exists.
```

Return mean and population standard deviation (`ddof=0`) across jobs/machines. The legacy and paper profiles use `legacy_workload_pressure`; corrected uses projected completion tardiness. The initial zero-vector behavior is applied by the environment only for `legacy_snapshot`, not inside the observation function.

- [ ] **Step 4: Write hand-calculated observation tests**

Required tests:

```text
1. Empty runtime produces finite environment observation.
2. legacy and paper orders contain same named values in different positions.
3. Workload pressure ignores an inserted wait but corrected projected TR increases.
4. Paper Uave and standard utilization differ on a two-machine hand schedule.
5. No magic numeric index is used in reward code.
```

- [ ] **Step 5: Implement reward functions**

Use explicit names:

```python
class RewardMode(IntEnum):
    TARDINESS = 0
    UTILIZATION = 1


def legacy_tardiness_reward(previous: float, current: float) -> int:
    if current < previous:
        return 1
    if current < previous * 1.1:
        return 0
    return -1


def paper_tardiness_reward(previous: float, current: float) -> int:
    return 1 if current < previous else -1


def legacy_utilization_reward(previous: float, current: float) -> int:
    if current > previous:
        return 1
    if current > 0.9 * previous:
        return 0
    return -1


def paper_utilization_reward(previous: float, current: float) -> int:
    if current > previous:
        return 1
    if 0.9 * previous <= current <= previous:
        return 0
    return -1
```

The zero/equality cases are intentionally different:

```text
legacy tardiness: previous=current=0 -> -1
paper tardiness:  previous=current=0 -> -1
legacy utilization: previous=current=0 -> -1
paper utilization:  previous=current=0 -> 0
```

Add one dispatcher:

```text
def transition_reward(
    profile: ReproductionProfile,
    mode: RewardMode,
    previous: ScheduleObservation,
    current: ScheduleObservation,
) -> int:
```

It selects by named attribute, never vector index.

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/test_observations.py tests/test_rewards.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
git add 扩刊/original_repro/src/smc_repro/runtime.py \
  扩刊/original_repro/src/smc_repro/observations.py \
  扩刊/original_repro/src/smc_repro/rewards.py \
  扩刊/original_repro/tests/test_observations.py \
  扩刊/original_repro/tests/test_rewards.py
git commit -m "feat: add named scheduling state and reward profiles"
```

## Task 6: Implement Auditable Legacy, Paper, and Classical Dispatching Rules

**Files:**
- Create: `扩刊/original_repro/src/smc_repro/rules/__init__.py`
- Create: `扩刊/original_repro/src/smc_repro/rules/base.py`
- Create: `扩刊/original_repro/src/smc_repro/rules/legacy.py`
- Create: `扩刊/original_repro/src/smc_repro/rules/paper.py`
- Create: `扩刊/original_repro/src/smc_repro/rules/classical.py`
- Create: `扩刊/original_repro/tests/test_legacy_rules.py`
- Create: `扩刊/original_repro/tests/test_paper_rules.py`
- Create: `扩刊/original_repro/tests/test_classical_rules.py`

**Interfaces:**
- Consumes: immutable `RuleContext` views built by the environment; rules do not mutate runtime or timelines.
- Produces: `DispatchDecision(job_id, op_id, machine_id, rule_name)` for one legal next-operation assignment.
- Produces: nine source-compatible rules, nine paper-defined rules, and five labelled classical baselines with one locked machine selector.

- [ ] **Step 1: Define immutable views and deterministic tie-breaking in `rules/base.py`**

Use these public records:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from smc_repro.seeding import keyed_uniform


class JobSelector(StrEnum):
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    FIFO = "FIFO"
    EDD = "EDD"
    MRT = "MRT"
    SPT = "SPT"
    LPT = "LPT"


class MachineSelector(StrEnum):
    EARLIEST_START = "earliest_start"
    EARLIEST_COMPLETION = "earliest_completion"
    RANDOM = "random"


@dataclass(frozen=True)
class JobRuleView:
    job_id: int
    op_id: int
    arrival_time: float
    due_date: float
    urgency: int
    decision_time: float
    latest_process_end: float
    operation_count: int
    completed_operation_count: int
    completion_ratio_by_count: float
    completion_ratio_by_work: float
    processed_work: float
    remaining_nominal_work: float
    next_operation_mean_processing_time: float


@dataclass(frozen=True)
class PairRuleView:
    job_id: int
    op_id: int
    machine_id: int
    earliest_start: float
    estimated_completion: float


@dataclass(frozen=True)
class RuleContext:
    instance_id: str
    decision_index: int
    policy_seed: int
    jobs: tuple[JobRuleView, ...]
    pairs: tuple[PairRuleView, ...]

    def __post_init__(self) -> None:
        if self.decision_index < 0 or self.policy_seed < 0:
            raise ValueError("decision_index and policy_seed must be non-negative")
        if not self.jobs or not self.pairs:
            raise ValueError("rule context must contain ready jobs and legal pairs")
        job_keys = {(job.job_id, job.op_id) for job in self.jobs}
        if len(job_keys) != len(self.jobs):
            raise ValueError("ready job views must be unique")
        if any((pair.job_id, pair.op_id) not in job_keys for pair in self.pairs):
            raise ValueError("every pair must refer to a ready job view")


@dataclass(frozen=True)
class DispatchDecision:
    job_id: int
    op_id: int
    machine_id: int
    rule_name: str
```

Add helpers with deterministic job/machine-id tie-breaking:

```python
def argmin_job(
    jobs: tuple[JobRuleView, ...],
    score,
) -> JobRuleView:
    return min(jobs, key=lambda job: (float(score(job)), job.job_id))


def argmax_job(
    jobs: tuple[JobRuleView, ...],
    score,
) -> JobRuleView:
    return min(jobs, key=lambda job: (-float(score(job)), job.job_id))


def keyed_choice(items: tuple[object, ...], base_seed: int, *keys: object) -> object:
    if not items:
        raise ValueError("cannot select from an empty sequence")
    value = keyed_uniform(base_seed, *keys)
    index = min(int(value * len(items)), len(items) - 1)
    return items[index]


def pairs_for_job(context: RuleContext, job: JobRuleView) -> tuple[PairRuleView, ...]:
    pairs = tuple(
        pair
        for pair in context.pairs
        if pair.job_id == job.job_id and pair.op_id == job.op_id
    )
    if not pairs:
        raise ValueError(f"no legal machine pair for job {job.job_id} operation {job.op_id}")
    return tuple(sorted(pairs, key=lambda pair: pair.machine_id))


def select_machine(
    context: RuleContext,
    job: JobRuleView,
    selector: MachineSelector,
    *,
    namespace: str,
) -> PairRuleView:
    pairs = pairs_for_job(context, job)
    if selector is MachineSelector.EARLIEST_START:
        return min(pairs, key=lambda pair: (pair.earliest_start, pair.machine_id))
    if selector is MachineSelector.EARLIEST_COMPLETION:
        return min(pairs, key=lambda pair: (pair.estimated_completion, pair.machine_id))
    if selector is MachineSelector.RANDOM:
        selected = keyed_choice(
            pairs,
            context.policy_seed,
            "rule_machine",
            namespace,
            context.instance_id,
            context.decision_index,
            job.job_id,
            job.op_id,
        )
        if not isinstance(selected, PairRuleView):
            raise TypeError("keyed machine selection returned an invalid object")
        return selected
    raise AssertionError(f"unsupported machine selector: {selector}")
```

Do not import `ScheduleRuntime`, `MachineTimeline`, NumPy, Python `random`, or torch in any rule module.

- [ ] **Step 2: Implement the nine source-compatible rules in `rules/legacy.py`**

Lock the source behavior as follows:

```text
Legacy A1:
  Minimize completed_operation_count / operation_count.

Legacy A2, no tardy ready job:
  Minimize (latest_process_end + decision_time - due_date) / urgency.

Legacy A2, at least one tardy ready job:
  Tardy means due_date < decision_time.
  Minimize due_date - decision_time / (4 - urgency), preserving source parentheses.

Legacy A3:
  Select a ready job through keyed policy randomness.

Legacy B1:
  Minimize estimated_completion. This corresponds to source Rule 1/4/7.

Legacy B2:
  Minimize earliest_start. This corresponds to source Rule 2/5/8.

Legacy B3:
  Select a legal machine through keyed policy randomness.
```

Use this table; names are deliberately explicit rather than relying only on integers:

```python
LEGACY_COMPOSITE_RULES = {
    0: (JobSelector.A1, MachineSelector.EARLIEST_COMPLETION, "legacy_A1_ECT"),
    1: (JobSelector.A1, MachineSelector.EARLIEST_START, "legacy_A1_EST"),
    2: (JobSelector.A1, MachineSelector.RANDOM, "legacy_A1_RANDOM"),
    3: (JobSelector.A2, MachineSelector.EARLIEST_COMPLETION, "legacy_A2_ECT"),
    4: (JobSelector.A2, MachineSelector.EARLIEST_START, "legacy_A2_EST"),
    5: (JobSelector.A2, MachineSelector.RANDOM, "legacy_A2_RANDOM"),
    6: (JobSelector.A3, MachineSelector.EARLIEST_COMPLETION, "legacy_A3_ECT"),
    7: (JobSelector.A3, MachineSelector.EARLIEST_START, "legacy_A3_EST"),
    8: (JobSelector.A3, MachineSelector.RANDOM, "legacy_A3_RANDOM"),
}
```

Implement one public dispatcher:

```text
def dispatch_legacy_rule(context: RuleContext, action_index: int) -> DispatchDecision:
```

It must reject indices outside `0..8`, choose the job, choose the machine, and return the selected next `op_id` from the view. All random namespaces include the action index so job randomness and machine randomness cannot alias.

- [ ] **Step 3: Implement the nine paper-defined rules in `rules/paper.py`**

Lock the paper formulas as follows:

```text
Paper A1:
  Minimize completion_ratio_by_work / (4 - urgency).
  Urgency semantics remain 1=high, 3=low; the denominator gives high urgency more priority.

Paper A2, at least one tardy ready job:
  Tardy means due_date < decision_time.
  Maximize (decision_time - due_date) / urgency.

Paper A2, no tardy ready job:
  Minimize (due_date - processed_work) / max(remaining_nominal_work, 1e-12).
  This intentionally reproduces the printed DDL-OPT over ETL expression rather than silently
  replacing it with a conventional slack formula.

Paper A3:
  Select a ready job through keyed policy randomness.

Paper B1:
  Minimize earliest_start.

Paper B2:
  Minimize estimated_completion.

Paper B3:
  Select a legal machine through keyed policy randomness.
```

Use this table:

```python
PAPER_COMPOSITE_RULES = {
    0: (JobSelector.A1, MachineSelector.EARLIEST_START, "paper_A1_B1_EST"),
    1: (JobSelector.A1, MachineSelector.EARLIEST_COMPLETION, "paper_A1_B2_ECT"),
    2: (JobSelector.A1, MachineSelector.RANDOM, "paper_A1_B3_RANDOM"),
    3: (JobSelector.A2, MachineSelector.EARLIEST_START, "paper_A2_B1_EST"),
    4: (JobSelector.A2, MachineSelector.EARLIEST_COMPLETION, "paper_A2_B2_ECT"),
    5: (JobSelector.A2, MachineSelector.RANDOM, "paper_A2_B3_RANDOM"),
    6: (JobSelector.A3, MachineSelector.EARLIEST_START, "paper_A3_B1_EST"),
    7: (JobSelector.A3, MachineSelector.EARLIEST_COMPLETION, "paper_A3_B2_ECT"),
    8: (JobSelector.A3, MachineSelector.RANDOM, "paper_A3_B3_RANDOM"),
}
```

Implement:

```text
def dispatch_paper_rule(context: RuleContext, action_index: int) -> DispatchDecision:
```

- [ ] **Step 4: Implement labelled classical rules in `rules/classical.py`**

The paper does not uniquely specify machine assignment for FIFO/EDD/MRT/SPT/LPT. The ambiguity register locks all five to earliest-completion machine selection, and names must expose that choice:

```python
class ClassicalRule(StrEnum):
    FIFO_ECT = "FIFO+ECT"
    EDD_ECT = "EDD+ECT"
    MRT_ECT = "MRT+ECT"
    SPT_ECT = "SPT+ECT"
    LPT_ECT = "LPT+ECT"
```

Job priorities:

```text
FIFO+ECT: minimize (arrival_time, job_id).
EDD+ECT:  minimize (due_date, job_id).
MRT+ECT:  maximize (remaining_nominal_work, then lower job_id).
SPT+ECT:  minimize (next_operation_mean_processing_time, job_id).
LPT+ECT:  maximize (next_operation_mean_processing_time, then lower job_id).
```

Implement:

```text
def dispatch_classical_rule(
    context: RuleContext,
    rule: ClassicalRule,
) -> DispatchDecision:
```

- [ ] **Step 5: Write hand-calculated rule tests before implementation**

Create one fixture with three ready jobs and three machines whose values force every selector to a known answer. Tests must cover:

```text
1. Legacy A1 selects lowest completed-operation ratio and lower job id on ties.
2. Legacy A2 reproduces both the non-tardy and tardy source formulas exactly.
3. Paper A1 uses urgency-weighted work completion, not count completion.
4. Paper A2 reproduces both printed branches exactly.
5. Legacy B1 is ECT while paper B1 is EST.
6. Paper B2 is ECT.
7. All random choices are repeatable for the same policy seed and decision index.
8. Changing decision index changes at least one selection over a bounded fixture sweep.
9. No selected machine is outside the legal pair list.
10. FIFO/EDD/MRT/SPT/LPT each select the hand-computed job and then ECT machine.
11. Invalid action index and empty pair/job contexts fail loudly.
12. Every action index 0..8 has a unique stable rule name.
```

Do not assert that two individual hash draws must differ; instead sweep at least 64 decision indices and assert that both candidates are selected at least once.

- [ ] **Step 6: Export the public API and run all gates**

`rules/__init__.py` exports only:

```python
from smc_repro.rules.base import DispatchDecision, RuleContext
from smc_repro.rules.classical import ClassicalRule, dispatch_classical_rule
from smc_repro.rules.legacy import dispatch_legacy_rule
from smc_repro.rules.paper import dispatch_paper_rule

__all__ = [
    "ClassicalRule",
    "DispatchDecision",
    "RuleContext",
    "dispatch_classical_rule",
    "dispatch_legacy_rule",
    "dispatch_paper_rule",
]
```

Run:

```bash
python -m pytest tests/test_legacy_rules.py tests/test_paper_rules.py \
  tests/test_classical_rules.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
```

- [ ] **Step 7: Commit**

```bash
git add 扩刊/original_repro/src/smc_repro/rules \
  扩刊/original_repro/tests/test_legacy_rules.py \
  扩刊/original_repro/tests/test_paper_rules.py \
  扩刊/original_repro/tests/test_classical_rules.py
git commit -m "feat: add audited SMC dispatching rule libraries"
```

---

# Gate 3 — Build the Original Constructive Environment and Agents

## Task 7: Implement the Profile-Controlled Constructive Scheduling Environment

**Files:**
- Create: `扩刊/original_repro/src/smc_repro/environment.py`
- Create: `扩刊/original_repro/src/smc_repro/experiment_contract.py`
- Create: `扩刊/original_repro/tests/test_environment.py`
- Extend: `扩刊/original_repro/tests/test_common_random_numbers.py`

**Interfaces:**
- Consumes: `InstanceSpec`, `ReproductionProfile`, one rule action index, and separated policy/failure/wear/repair seeds.
- Produces: `reset()`, `build_rule_context()`, `step_rule(action_index)`, `final_result()`, final intervals, validator report, and metrics.
- Preserves the original constructive/list-scheduling interpretation. It does not claim a physical event-driven SMDP.

- [ ] **Step 1: Define the environment API and result records**

Use:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from smc_repro.metrics import ScheduleMetrics
from smc_repro.observations import ScheduleObservation
from smc_repro.rules.base import DispatchDecision, RuleContext
from smc_repro.schemas import InstanceSpec, ScheduleInterval
from smc_repro.validator import ValidationReport


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    named_observation: ScheduleObservation
    reward: int
    done: bool
    decision: DispatchDecision
    emitted_intervals: tuple[ScheduleInterval, ...]
    info: Mapping[str, str | int | float | bool | None]


@dataclass(frozen=True)
class EpisodeResult:
    instance_id: str
    profile_name: str
    intervals: tuple[ScheduleInterval, ...]
    validation: ValidationReport
    metrics: ScheduleMetrics
    decisions: int
```

`SchedulingEnvironment` constructor:

```text
def __init__(
    self,
    instance: InstanceSpec,
    profile: ReproductionProfile,
    *,
    policy_seed: int,
    failure_seed: int | None = None,
    wear_seed: int | None = None,
    repair_seed: int | None = None,
) -> None:
```

Seed defaults are derived from `instance.failure_seed` through distinct integer offsets, but callers may supply explicit values. The offsets are recorded in read-only `info`/episode metadata. Construct `StepResult.info` through the same defensive scalar-mapping freeze used by interval metadata; never expose a mutable dictionary. Never use global `random`, global NumPy RNG, or torch RNG in environment dynamics.

Public methods:

```text
def reset(self) -> tuple[np.ndarray, ScheduleObservation]:
def build_rule_context(self) -> RuleContext:
def step_rule(self, action_index: int, reward_mode: RewardMode) -> StepResult:
def final_result(self) -> EpisodeResult:
def is_done(self) -> bool:
```

`final_result()` must fail before completion, call `validate_schedule(require_complete=True)`, and reject every invalid schedule before computing metrics.

- [ ] **Step 2: Build rule views without mutating the runtime**

At each constructive decision:

```text
1. ready jobs = unfinished jobs with arrival_time <= decision_time.
2. If none are ready, advance decision_time to the minimum future arrival among unfinished jobs.
3. Do not fall back to a future job without advancing time.
4. For each ready job, expose only its next operation.
5. For each eligible machine, compute a side-effect-free candidate plan.
```

The candidate estimator returns:

```python
@dataclass(frozen=True)
class CandidatePlan:
    job_id: int
    op_id: int
    machine_id: int
    predecessor_end: float
    setup_required: bool
    setup_duration: float
    process_nominal_duration: float
    process_estimated_duration: float
    earliest_start: float
    estimated_completion: float
```

Candidate estimates must use the current degradation factor but must not sample PM, CM, wear, or repair. Under `setup_mode=source_tool_change`, `setup_required` is true when either the selected job previously used another machine or the selected machine previously processed another job, exactly matching the source `change_cutter()` predicate; under `setup_mode=none`, it is always false and no SETUP duration enters the score or schedule. Candidate `earliest_start` includes arrival, predecessor completion, and the machine timeline. `estimated_completion` equals `earliest_start + setup_duration + process_estimated_duration`, so it reproduces source ECT when setup is enabled and the paper B2 equation when paper_repro disables setup. All three original-conference profiles append at the timeline tail. The paper's claimed historical-gap insertion is recorded in ambiguity A-009 but is not partially "fixed" here: retrospective insertion would change the chronological health/maintenance history of already scheduled operations and therefore requires a full replay engine, which belongs to the later event-driven upgrade.

- [ ] **Step 3: Lock the exact interval order and maintenance semantics**

For all profiles, preserve the source ordering so profile differences do not silently include an unrelated maintenance-order redesign:

```text
SETUP (only when `setup_mode=source_tool_change`, `setup_required` is true, and setup_time > 0)
PM check and optional PM
CM risk check and optional CM
PROCESS
wear/update
```

The source behavior checks maintenance after applying the setup delay but before processing. The rerun records that behavior explicitly. Do not claim within-operation interruption.

For each selected decision, compute the complete interval bundle and every resulting machine/job state change in local temporary values. Validate the bundle on a cloned timeline first; only after validation may the environment atomically replace the timeline and commit health, age, usage, degradation factor, counts, last-job/last-machine fields, next-op index, decision time, and decision index. This makes the entire step transactional: a failed step cannot partially mutate either intervals or runtime state.

Use one helper:

```text
def _append_interval_bundle_transactionally(
    timeline: MachineTimeline,
    intervals: tuple[ScheduleInterval, ...],
) -> None:
```

It copies the current interval tuple into a temporary `MachineTimeline`, adds every candidate interval, and replaces the original only after all additions succeed. Expose a dedicated `replace_intervals_for_transaction()` method on `MachineTimeline` rather than mutating a private list from the environment.

- [ ] **Step 4: Implement source-compatible degradation and maintenance**

For `legacy_snapshot` and `paper_repro`:

```text
failure probability before an operation:
  Weibull CDF at usage_time.

PM trigger:
  current failure_probability > 0.2 OR health < 30, when pm_enabled.

PM duration:
  machine.cm_duration * machine.pm_duration_ratio.

PM recovery:
  health=100, usage_time=0, effective_age=0, degradation_factor=1.0;
  immediately before PROCESS, recompute the source formula, which yields 1.01 at health 100.

High-load failure bias:
  Reproduce the source `CTK` branch using each machine's latest PROCESS end only, not the
  full timeline availability that may include SETUP/PM/CM. Mark a machine high-load when
  its latest PROCESS end is at or above the 90th percentile of all machines' latest PROCESS ends. A high-load machine fails when min(u_primary, u_secondary) < CDF;
  otherwise it fails when u_primary < CDF.

CM duration:
  machine.cm_duration.

CM recovery:
  add a keyed uniform health recovery in [20,40], cap at 90;
  usage_time *= cm_age_repair_factor;
  effective_age mirrors usage_time.

Degradation factor before processing:
  1 + round(0.01 * exp(0.05 * (100 - round(health, 1))), 2).

Wear after processing:
  usage_time += actual process duration;
  health -= keyed uniform in [4,8], clipped to [0,100];
  effective_age=usage_time;
  recompute degradation factor for the next decision.
```

Stable keyed draws:

```python
u_primary = keyed_uniform(
    failure_seed,
    "failure_primary",
    instance_id,
    job_id,
    op_id,
    machine_id,
)
u_secondary = keyed_uniform(
    failure_seed,
    "failure_secondary",
    instance_id,
    job_id,
    op_id,
    machine_id,
)
wear = 4.0 + 4.0 * keyed_uniform(
    wear_seed,
    "wear",
    instance_id,
    job_id,
    op_id,
    machine_id,
)
recovery = 20.0 + 20.0 * keyed_uniform(
    repair_seed,
    "cm_recovery",
    instance_id,
    job_id,
    op_id,
    machine_id,
)
```

For `corrected_smc`:

```text
failure probability:
  weibull_interval_failure_probability(effective_age, estimated actual process duration).

high-load bias:
  disabled by profile.

PM recovery:
  effective_age=usage_time=0, health=100, degradation_factor=1.0;
  immediately before PROCESS, recompute the degradation formula.

CM recovery:
  effective_age *= cm_age_repair_factor;
  usage_time=effective_age;
  health=health_from_effective_age(effective_age);

Wear:
  effective_age += actual process duration;
  usage_time=effective_age;
  health=health_from_effective_age(effective_age);
  recompute degradation factor.
```

Use the same keyed primary draw for the same `(instance, job, operation, machine)` across algorithms, regardless of the order in which an algorithm schedules that operation. Policy calls, decision indices, and unrelated random events must not move the failure/wear/repair draw.

- [ ] **Step 5: Define the exact clock and update rules**

After committing a PROCESS interval:

```text
next_op_index[job_id] += 1
machine.last_job_id = job_id
last_machine_by_job[job_id] = machine_id
machine.process_count += 1
machine.pm_count/cm_count updated from emitted intervals
decision_index += 1
decision_time = minimum timeline available_time over machines
```

Before constructing the next context, if no unfinished arrived job exists at `decision_time`, advance to the next unfinished-job arrival. A job is considered constructively completed once all of its operations have PROCESS intervals; no artificial post-completion maintenance is sampled.

The initial observation behavior is profile-controlled:

```text
legacy_snapshot: return a six-zero vector from reset while retaining a named environment observation.
paper_repro: return the environment-derived vector.
corrected_smc: return the environment-derived vector.
```

The reward compares the vector-independent named observations before and after the action. The first legacy transition compares against the six-zero named surrogate:

```python
ScheduleObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 6: Attach complete scalar metadata to every interval**

Because interval metadata is restricted to JSON scalar values, record only:

```text
schema_version
profile
event_id
decision_index
rule_name
selected_job_id
selected_op_id
health_before
health_after
effective_age_before
effective_age_after
failure_probability
failure_draw_primary
failure_draw_secondary (null when unused)
pm_triggered
cm_triggered
nominal_processing_time
degradation_factor
```

Use the exact stable event identifier:

```python
event_id = (
    f"{self.instance.instance_id}:d{self.runtime.decision_index:06d}:"
    f"{interval_type.value.lower()}"
)
```

There is at most one interval of each type in one constructive decision, so this format is unique. SETUP, PM, and CM intervals must use this `event_id` and the configured duration. PROCESS duration must be at least nominal processing time and match the metadata within tolerance.

- [ ] **Step 7: Define `experiment_contract.py` before any training code**

Create:

```python
@dataclass(frozen=True)
class RunContract:
    schema_version: int
    git_commit: str
    profile_name: str
    profile_sha256: str
    bank_manifest_sha256: str
    method: str
    train_seed: int
    policy_seed: int
    failure_stream_version: str
    environment_metadata_path: str
```

Functions:

```text
def collect_git_commit(repo_root: Path, *, allow_dirty: bool = False) -> str:

def build_run_contract(
    repo_root: Path,
    profile: ReproductionProfile,
    *,
    bank_manifest_sha256: str,
    method: str,
    train_seed: int,
    policy_seed: int,
    environment_metadata_path: Path,
    allow_dirty: bool = False,
) -> RunContract:

def contract_sha256(contract: RunContract) -> str:
```

`collect_git_commit()` must call `git rev-parse HEAD` and reject a dirty tree for formal runs. Preflight may allow dirty mode only through an explicit `allow_dirty=True` argument recorded in the report. `environment_metadata_path` must be stored as a repository-relative POSIX path; reject absolute paths, backslashes, empty components, and `..`. The environment snapshot's own capture commit is provenance and need not equal the contract's current `git_commit`.

- [ ] **Step 8: Write focused environment tests before implementation**

Required tests:

```text
1. One job/one machine/no setup/no failure produces one exact PROCESS interval.
2. A job change emits one SETUP interval immediately before the maintenance/process bundle in source-tool-change profiles, while paper_repro emits none.
3. PM threshold emits a PM interval and restores profile-specific state.
4. A forced keyed failure emits CM before PROCESS and uses the configured duration.
5. Same selected process action gets the same primary failure draw after 10,000 unrelated keyed draws.
6. Corrected interval risk increases with process duration; legacy CDF does not depend on candidate duration.
7. All original-conference profiles append at the machine tail, and no profile silently performs retrospective gap insertion.
8. A failed transactional step leaves the original timeline and every runtime field byte-for-byte/value-for-value equivalent.
9. No PM or CM is generated after the final PROCESS interval.
10. Every legacy and paper action index completes a fixed small instance with a valid schedule.
11. Every classic rule completes the same instance with a valid schedule.
12. Reset behavior differs exactly as configured while named observations remain available.
13. All interval metadata values are JSON scalar values.
14. Two repeated episodes with identical contract and seeds produce identical serialized intervals and metrics.
15. Changing only policy seed may change random rules but does not change keyed failure values for matching actions.
16. Invalid rule action, ineligible machine, repeated completed job, and step-after-done all fail loudly.
17. The high-load percentile uses latest PROCESS ends and is unaffected by a PM/CM interval that extends only nonprocess availability.
```

For forced-event tests, choose seeds by scanning `keyed_uniform()` values in the test until a draw lies below or above a fixed threshold. Do not monkeypatch production reliability functions.

- [ ] **Step 9: Run all environment gates**

```bash
python -m pytest tests/test_environment.py tests/test_common_random_numbers.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
```

Run a non-learning deterministic episode for every rule on one generated instance:

```bash
python - <<'PY'
from pathlib import Path

from smc_repro.config import load_profile
from smc_repro.environment import SchedulingEnvironment
from smc_repro.instance_io import load_instance
from smc_repro.rewards import RewardMode

instance = load_instance(
    Path("artifacts/banks/materialized/test/m08_j10_e050/test_m08_j10_e050_rep00.json.gz")
)
for profile_path in ("configs/legacy_snapshot.yaml", "configs/paper_repro.yaml"):
    profile = load_profile(Path(profile_path))
    for action_index in range(9):
        env = SchedulingEnvironment(instance, profile, policy_seed=action_index)
        env.reset()
        while not env.is_done():
            env.step_rule(action_index, RewardMode.TARDINESS)
        result = env.final_result()
        assert result.validation.ok
        assert result.decisions == sum(len(job.operations) for job in instance.jobs)
        print(profile.profile.value, action_index, result.metrics.makespan)
PY
```

- [ ] **Step 10: Commit**

```bash
git add 扩刊/original_repro/src/smc_repro/environment.py \
  扩刊/original_repro/src/smc_repro/experiment_contract.py \
  扩刊/original_repro/src/smc_repro/timeline.py \
  扩刊/original_repro/tests/test_environment.py \
  扩刊/original_repro/tests/test_common_random_numbers.py
git commit -m "feat: add profile-controlled SMC scheduling environment"
```

---

## Task 8: Implement DL-DDQN, Vanilla-DQN Targets, Tabular Baselines, and Strict Checkpoints

**Files:**
- Create: `扩刊/original_repro/src/smc_repro/agents/__init__.py`
- Create: `扩刊/original_repro/src/smc_repro/agents/networks.py`
- Create: `扩刊/original_repro/src/smc_repro/agents/replay.py`
- Create: `扩刊/original_repro/src/smc_repro/agents/checkpoint.py`
- Create: `扩刊/original_repro/src/smc_repro/agents/dual_ddqn.py`
- Create: `扩刊/original_repro/src/smc_repro/agents/tabular.py`
- Modify: `扩刊/original_repro/src/smc_repro/rewards.py`
- Create: `扩刊/original_repro/tests/test_networks.py`
- Create: `扩刊/original_repro/tests/test_double_dqn_targets.py`
- Create: `扩刊/original_repro/tests/test_checkpoint.py`
- Create: `扩刊/original_repro/tests/test_tabular_agents.py`

**Interfaces:**
- Consumes: six-dimensional profile-ordered observations and nine rule actions.
- Produces: one dual-layer value agent configurable as Double DQN or vanilla DQN, complete resume checkpoints, and Q-learning/SARSA baselines.
- Does not create full experiment runners or paper result files.

- [ ] **Step 1: Implement configurable MLPs and lower-context construction**

`agents/networks.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from smc_repro.config import LowerContextMode


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or not hidden_dims:
            raise ValueError("network dimensions must be positive and hidden_dims non-empty")
        if any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden dimensions must be positive")
        dimensions = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("Q-network input must be rank two [batch, features]")
        return self.model(values)
```

Context construction:

```python
def lower_input_dim(mode: LowerContextMode) -> int:
    return 8 if mode is LowerContextMode.REWARD_ID_ONE_HOT else 7


def build_lower_context(
    states: torch.Tensor,
    upper_q_values: torch.Tensor,
    reward_ids: torch.Tensor,
    mode: LowerContextMode,
) -> torch.Tensor:
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("states must have shape [batch, 6]")
    if upper_q_values.shape != (states.shape[0], 2):
        raise ValueError("upper_q_values must have shape [batch, 2]")
    if reward_ids.shape != (states.shape[0],):
        raise ValueError("reward_ids must have shape [batch]")
    if torch.any((reward_ids < 0) | (reward_ids > 1)):
        raise ValueError("reward ids must be 0 or 1")
    if mode is LowerContextMode.MAX_Q_SCALAR:
        context = torch.max(upper_q_values, dim=1, keepdim=True).values
    elif mode is LowerContextMode.REWARD_ID_SCALAR:
        context = reward_ids.to(dtype=states.dtype).unsqueeze(1)
    elif mode is LowerContextMode.REWARD_ID_ONE_HOT:
        context = torch.nn.functional.one_hot(reward_ids, num_classes=2).to(states.dtype)
    else:
        raise AssertionError(f"unsupported lower context: {mode}")
    return torch.cat((states, context), dim=1)
```

Tests must prove the exact hidden-layer counts from each YAML profile and exact context tensor values for all three modes.

- [ ] **Step 2: Implement an isolated replay buffer**

`agents/replay.py`:

```python
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    rule_action: int
    reward: float
    next_state: np.ndarray
    reward_id: int
    done: bool

    def __post_init__(self) -> None:
        for name, value in (("state", self.state), ("next_state", self.next_state)):
            copied = np.array(value, dtype=np.float32, copy=True)
            if copied.shape != (6,) or not np.all(np.isfinite(copied)):
                raise ValueError(f"{name} must be a finite float32 vector with shape (6,)")
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if not 0 <= self.rule_action < 9:
            raise ValueError("rule_action must be in [0, 8]")
        if self.reward_id not in (0, 1):
            raise ValueError("reward_id must be 0 or 1")
        if not np.isfinite(self.reward):
            raise ValueError("reward must be finite")


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0 or seed < 0:
            raise ValueError("capacity must be positive and seed non-negative")
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)
        self.capacity = capacity

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        if batch_size <= 0 or batch_size > len(self._items):
            raise ValueError("invalid replay sample size")
        return tuple(self._rng.sample(tuple(self._items), batch_size))

    def state_dict(self) -> dict[str, object]:
        serialized_items = tuple(
            {
                "state": item.state.tolist(),
                "rule_action": item.rule_action,
                "reward": item.reward,
                "next_state": item.next_state.tolist(),
                "reward_id": item.reward_id,
                "done": item.done,
            }
            for item in self._items
        )
        return {
            "schema_version": 1,
            "capacity": self.capacity,
            "rng_state": self._rng.getstate(),
            "items": serialized_items,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if state.get("schema_version") != 1 or state.get("capacity") != self.capacity:
            raise ValueError("incompatible replay-buffer checkpoint")
        raw_items = state.get("items")
        rng_state = state.get("rng_state")
        if not isinstance(raw_items, tuple) or not isinstance(rng_state, tuple):
            raise ValueError("invalid replay-buffer checkpoint payload")
        restored: list[Transition] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise ValueError("invalid replay-buffer item")
            restored.append(
                Transition(
                    state=np.asarray(raw_item["state"], dtype=np.float32),
                    rule_action=int(raw_item["rule_action"]),
                    reward=float(raw_item["reward"]),
                    next_state=np.asarray(raw_item["next_state"], dtype=np.float32),
                    reward_id=int(raw_item["reward_id"]),
                    done=bool(raw_item["done"]),
                )
            )
        self._items.clear()
        self._items.extend(restored)
        self._rng.setstate(rng_state)
```

The replay checkpoint uses only tensors and Python primitive containers so that the final checkpoint can be loaded with `weights_only=True`. Do not save `Transition` instances directly and do not fall back to `weights_only=False`.

- [ ] **Step 3: Isolate target computation and test it numerically**

`agents/dual_ddqn.py` must expose:

```python
def value_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    gamma: float,
    *,
    double_dqn: bool,
) -> torch.Tensor:
    if double_dqn:
        selected = torch.argmax(online_next_q, dim=1)
        bootstrap = target_next_q.gather(1, selected.unsqueeze(1)).squeeze(1)
    else:
        bootstrap = torch.max(target_next_q, dim=1).values
    return rewards + gamma * bootstrap * (~dones).to(rewards.dtype)
```

Hand test:

```python
online = torch.tensor([[1.0, 9.0], [8.0, 2.0]])
target = torch.tensor([[7.0, 3.0], [4.0, 6.0]])
rewards = torch.tensor([2.0, -1.0])
dones = torch.tensor([False, True])
```

Expected:

```text
Double DQN: [2 + gamma*3, -1]
Vanilla DQN: [2 + gamma*7, -1]
```

Add a nine-action test for the lower network and assert target tensors have no gradient.

- [ ] **Step 4: Implement the dual-layer agent without changing the original learning identity**

Public records:

```python
@dataclass(frozen=True)
class AgentDecision:
    rule_action: int
    reward_mode: RewardMode
    epsilon: float
    exploratory: bool


@dataclass(frozen=True)
class UpdateReport:
    upper_loss: float
    lower_loss: float
    epsilon: float
    global_update_step: int
```

Constructor:

```text
def __init__(
    self,
    profile: ReproductionProfile,
    *,
    seed: int,
    device: torch.device,
    double_dqn: bool,
) -> None:
```

Network initialization must depend only on `seed` and must not consume or leak the caller's global torch RNG state. Construct all four networks inside `torch.random.fork_rng(...)`, call `torch.manual_seed(seed)` inside that context, and call `torch.cuda.manual_seed_all(seed)` when the selected device is CUDA. After constructing the two online networks, create/copy targets deterministically. Add a test that intervening global torch draws do not change parameters for two agents with the same seed, while a different seed changes at least one parameter.

Required networks:

```text
upper_online: input 6, profile upper hidden list, output 2
upper_target: same
lower_online: input derived from lower context, profile lower hidden list, output 9
lower_target: same
Adam optimizer per online network using profile learning rate
MSE loss, matching the source/paper
```

Use a local `random.Random(seed)` for joint epsilon exploration. One Bernoulli controls both levels, preserving source behavior:

```text
if explore:
  reward_id uniformly from {0,1}
  rule_action uniformly from the integer set 0 through 8
else:
  reward_id = argmax upper_online(state)
  lower_context = configured context built with that reward_id
  rule_action = argmax lower_online(lower_context)
```

After every training-mode decision:

```python
epsilon = max(profile.training.epsilon_end, epsilon - epsilon_decrement)
```

Evaluation mode never decrements and always uses `epsilon=0.0`.

For replay updates:

```text
1. Sample with ReplayBuffer's local RNG.
2. Train upper online on stored reward_id.
3. Build upper target through `value_target()`.
4. Build current lower context:
   - max_q_scalar: detached current upper-online max Q;
   - reward-id modes: stored current reward_id.
5. For the next lower context:
   - compute next reward id as argmax of upper_online(next_state);
   - max_q_scalar uses detached next upper-online max Q;
   - scalar/one-hot modes use the computed next reward id.
6. Train lower online on stored rule action with the same scalar transition reward.
7. When `global_update_step % target_update_steps == 0`, copy both online networks
   to targets before the update, matching the source's first-update synchronization.
8. Increment global update step only when an optimizer update occurs.
```

Do not add prioritized replay, dueling heads, n-step return, Huber loss, AMP, gradient accumulation, or reward normalization in this phase.

- [ ] **Step 5: Add exact checkpoint schema and load semantics**

`agents/checkpoint.py` defines `CHECKPOINT_SCHEMA_VERSION = 1` and functions:

```python
def save_checkpoint(
    path: Path,
    agent: DualLayerValueAgent,
    contract: RunContract,
) -> str:
    """Write atomically and return checkpoint SHA-256."""


def load_checkpoint(
    path: Path,
    agent: DualLayerValueAgent,
    expected_contract: RunContract,
    *,
    for_training: bool,
) -> str:
    """Load strictly, return SHA-256, and force epsilon=0 when not training."""
```

Payload fields:

```text
schema_version
contract (canonical primitive dict)
contract_sha256
profile (canonical primitive dict)
profile_sha256
double_dqn
upper_online / upper_target state dicts
lower_online / lower_target state dicts
upper_optimizer / lower_optimizer state dicts
global_update_step
decision_count
epsilon
agent_rng_state
replay_state
torch_cpu_rng_state
torch_cuda_rng_states (empty tuple when CUDA unavailable)
```

Rules:

```text
- Missing path -> FileNotFoundError.
- File SHA is computed before loading.
- `torch.load(..., weights_only=True)` is mandatory for self-produced checkpoints.
- Schema, contract hash, profile hash, double_dqn flag, and network dimensions must match.
- Atomic write uses a sibling temporary file, fsync, then os.replace.
- for_training=True restores optimizers, replay, RNG, global step, decision count, epsilon.
- for_training=False loads network weights only, clears replay, switches networks to eval, and
  forces epsilon exactly 0.0 regardless of the checkpoint value.
```

Required checkpoint tests:

```text
1. Missing checkpoint raises FileNotFoundError.
2. Evaluation load forces epsilon 0 after a checkpoint saved with epsilon > 0.
3. Training resume restores the next exploratory decision and the next replay sample exactly.
4. A one-update uninterrupted run equals save/load/resume within exact tensor equality on CPU.
5. Contract/profile mismatch fails before any agent state is mutated.
6. Corrupted bytes and wrong schema fail clearly.
7. Checkpoint SHA reported by function matches independent hashlib computation.
8. `weights_only=True` succeeds on the final payload.
```

- [ ] **Step 6: Add the source joint reward for tabular-paper replication**

In `rewards.py`:

```python
def legacy_joint_reward(
    previous: ScheduleObservation,
    current: ScheduleObservation,
) -> int:
    if current.tr_ave < previous.tr_ave:
        return 1
    if current.tr_ave < previous.tr_ave * 1.1:
        return 0
    if current.u_ave > previous.u_ave:
        return 1
    if current.u_ave > previous.u_ave * 0.9:
        return 0
    return -1
```

This is retained only for the source/paper-style tabular comparison. The later apples-to-apples protocol will also run fixed reward modes through the same environment; do not conflate the two result families.

- [ ] **Step 7: Implement Q-learning and SARSA with auditable discretization**

`agents/tabular.py`:

```python
class TabularAlgorithm(StrEnum):
    Q_LEARNING = "q_learning"
    SARSA = "sarsa"


class TabularRewardProtocol(StrEnum):
    LEGACY_JOINT = "legacy_joint"
    FIXED_TARDINESS = "fixed_tardiness"
    FIXED_UTILIZATION = "fixed_utilization"
```

Discretizer:

```python
def discretize_six_feature_state(state: np.ndarray) -> int:
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("tabular state must be a finite vector with shape (6,)")
    mean_value = float(np.mean(np.clip(state, 0.0, 1.0)))
    return min(9, int(mean_value * 10.0))
```

Agent defaults reproduce the existing baseline unless a test/experiment config overrides them:

```text
Q table: zeros with shape (10,9)
learning rate: 0.1
gamma: 0.95
epsilon: 0.2
epsilon multiplicative decay: 0.995
minimum epsilon: 0.01
```

Use one private `random.Random(seed)` for exploration and random tie-breaking. When exploiting, choose uniformly among all maximal-Q actions rather than always action 0; the change removes an accidental initial Rule-1 bias and must be labelled `corrected_tie_break`. Add an optional `first_argmax` mode only for exact source sensitivity, not as the default scientific baseline.

Updates:

```text
Q-learning:
  Q[s,a] += alpha * (r + gamma*max_a' Q[s',a']*(not done) - Q[s,a])

SARSA:
  Q[s,a] += alpha * (r + gamma*Q[s',a_next]*(not done) - Q[s,a])
```

State, Q table, epsilon, local RNG, algorithm, and tie mode must have primitive state-dict serialization and strict restore tests.

- [ ] **Step 8: Write agent tests before completing implementations**

Minimum tests:

```text
1. Profile architecture creates the exact paper/source layer counts and dimensions.
2. Same agent seed gives identical initial parameters despite intervening global torch draws; a different seed changes at least one parameter.
3. Replay transitions defensively copy state arrays and expose them read-only.
4. All lower-context modes produce exact tensors and dimensions.
5. Double and vanilla targets match hand calculations.
6. Terminal transitions do not bootstrap.
7. Epsilon=1 explores both reward ids and all nine actions over a bounded deterministic sweep.
8. Epsilon=0 decisions equal direct network argmax.
9. Epsilon never drops below configured end.
10. Replay sampling is repeatable after state restore.
11. One optimizer step changes at least one online parameter and no target parameter between syncs.
12. Target sync copies both networks exactly.
13. Checkpoint tests listed above all pass.
14. Discretizer maps 0 to state 0 and all ones to state 9.
15. Q-learning and SARSA hand updates match exact values.
16. Corrected tie breaking samples every tied action over a bounded sweep; source mode picks first.
17. No tabular or deep-agent method mutates the environment's failure/wear streams.
```

- [ ] **Step 9: Run a three-episode agent smoke without producing scientific results**

Use the first three instances from `train/seed_000`. For each profile:

```text
DDQN (`double_dqn=True`)
Vanilla DQN target (`double_dqn=False`)
```

For paper profile only, also run Q-learning and SARSA. This smoke is a functionality check; save outputs under ignored `artifacts/preflight/agent_smoke/` and never quote them as experiment results.

Acceptance:

```text
- every episode completes;
- every schedule validates;
- no NaN/Inf in observations, rewards, losses, Q tables, parameters, or metrics;
- checkpoint round-trip passes;
- evaluation load reports epsilon exactly 0;
- repeated CPU smoke with identical seeds is byte-identical after canonical JSON serialization.
```

- [ ] **Step 10: Run all gates and commit**

```bash
python -m pytest tests/test_networks.py tests/test_double_dqn_targets.py \
  tests/test_checkpoint.py tests/test_tabular_agents.py -q
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
```

Commit:

```bash
git add 扩刊/original_repro/src/smc_repro/agents \
  扩刊/original_repro/src/smc_repro/rewards.py \
  扩刊/original_repro/tests/test_networks.py \
  扩刊/original_repro/tests/test_double_dqn_targets.py \
  扩刊/original_repro/tests/test_checkpoint.py \
  扩刊/original_repro/tests/test_tabular_agents.py
git commit -m "feat: add reproducible SMC value and tabular agents"
```

---

# Final Preflight — Prove the Repository Is Ready for Experiment Drivers

## Task 9: Add Clean-Worktree and End-to-End Preflight Gates

**Files:**
- Create: `扩刊/original_repro/src/smc_repro/scripts/clean_worktree_gate.py`
- Create: `扩刊/original_repro/src/smc_repro/scripts/preflight.py`
- Create: `扩刊/original_repro/tests/test_preflight.py`
- Modify: `扩刊/original_repro/README.md`
- Modify: `扩刊/all.md`

**Interfaces:**
- Consumes: committed source, committed profiles/manifests/environment snapshot, materialized 1540-instance bank, and the existing RTX 5090 Python environment.
- Produces: `artifacts/preflight/preflight_report.json` and `artifacts/preflight/clean_worktree_report.json`.
- Ends this phase. It does not create formal training/evaluation sweeps, aggregation, statistical tests, or paper plots.

- [ ] **Step 1: Implement a clean-worktree gate that uses only Git-tracked content**

`clean_worktree_gate.py` must:

```text
1. Resolve repository root through `git rev-parse --show-toplevel`.
2. Reject a dirty source tree before starting.
3. Create a detached temporary Git worktree at the current HEAD.
4. Use the explicitly provided Python executable; do not create/install another environment.
5. Set PYTHONPATH to `<temporary>/扩刊/original_repro/src` so imports come from the worktree.
6. Set PYTHONHASHSEED=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching Python.
7. Run, from `<temporary>/扩刊/original_repro`:
   - `python -m pytest -q`
   - `python -m ruff check src tests`
   - `python -m mypy src/smc_repro`
   - `python -m compileall -q src tests`
8. Run the tracked legacy audit and compare it with `legacy_tracked_manifest.json`.
9. Record command, exit code, duration, and SHA-256 of stdout/stderr in JSON.
10. Always remove the temporary worktree in a finally block.
11. Exit non-zero if any command fails or cleanup fails.
```

Public function:

```text
def run_clean_worktree_gate(
    repo_root: Path,
    python_executable: Path,
    report_path: Path,
) -> dict[str, object]:
```

Do not copy ignored files, local manifests, materialized banks, virtual environments, checkpoints, or preflight artifacts into the temporary worktree.

- [ ] **Step 2: Write the clean-worktree test with a tiny temporary Git repository**

The test creates a temporary Git repository containing:

```text
code/tracked.txt
code/ignored.bin (ignored and not committed)
tracked manifest containing only code/tracked.txt
one tiny passing pytest test
```

Then it invokes the lower-level worktree/audit function and proves the ignored file is not required. Also test:

```text
- dirty repository is rejected;
- failing subprocess returns a failed report and raises;
- temporary worktree is removed after failure;
- paths containing non-ASCII characters work.
```

Do not run the complete real repository gate from unit tests; the real gate is an explicit final command.

- [ ] **Step 3: Implement canonical serialization helpers for preflight evidence**

In `preflight.py`:

```python
def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
```

Episode serialization includes interval fields in timeline order, metric fields through `dataclasses.asdict`, profile/contract hashes, decisions, reward sequence, loss sequence, checkpoint SHA, and evaluation epsilon. It excludes timestamps and wall-clock durations from the deterministic payload; timings are recorded separately.

- [ ] **Step 4: Implement the preflight pipeline in a fixed order**

CLI:

```bash
python -m smc_repro.scripts.preflight \
  --repo-root ../.. \
  --bank-root artifacts/banks/materialized \
  --reference-manifest artifacts/banks/release/manifest.json \
  --environment-metadata ../docs/audit/environment_5090_resolved.json \
  --output artifacts/preflight/preflight_report.json \
  --device cuda:0
```

The script performs these gates in order and stops at the first failure while still writing a failed report:

```text
P00 repository cleanliness and current Git SHA
P01 environment metadata and live RTX 5090 CUDA tensor smoke; validate the snapshot's versions/device/bank SHA but treat its `git_commit` as capture provenance, not as the live HEAD
P02 tracked legacy manifest equality
P03 release-manifest SHA equality
P04 materialized bank verification: 1540/1540 files
P05 strict load of legacy_snapshot, paper_repro, corrected_smc and smoke override
P06 non-learning completion of one small instance under 18 composite rules
P07 non-learning completion under five classical rules
P08 3-episode DDQN smoke for each of the three profiles
P09 3-episode vanilla-DQN-target smoke for each of the three profiles
P10 3-episode Q-learning and SARSA smoke under paper_repro
P11 strict checkpoint training-resume and evaluation-load round trip
P12 epsilon=0 evaluation on two fixed test instances for every deep smoke checkpoint
P13 schedule validation and finite-metric check for every episode
P14 repeated deterministic evaluation equality
P15 no forbidden tracked/generated artifacts and no legacy diff
```

Use the following fixed bank entries:

```text
train:
  train/seed_000/train_seed000_ep0000.json.gz
  train/seed_000/train_seed000_ep0001.json.gz
  train/seed_000/train_seed000_ep0002.json.gz

evaluation:
  test/m08_j10_e050/test_m08_j10_e050_rep00.json.gz
  test/m08_j10_e050/test_m08_j10_e050_rep01.json.gz
```

Fixed seeds:

```text
agent/train seed: 61000 + profile index*100 + method index
policy seed:      62000 + episode index
failure seed:     use instance.failure_seed
wear seed:        instance.failure_seed + 10_000_000
repair seed:      instance.failure_seed + 20_000_000
replay seed:      agent seed + 1
```

Profile order is exactly:

```text
legacy_snapshot
paper_repro
corrected_smc
```

Method order is exactly:

```text
ddqn
vanilla_dqn_target
q_learning
sarsa
```

- [ ] **Step 5: Define preflight scientific-safety assertions**

Every episode must satisfy:

```text
- number of decisions equals total number of operations;
- validator.ok is true;
- exactly one PROCESS interval per operation;
- all observations and rewards are finite;
- all losses, network parameters, target values, and Q-table values are finite;
- makespan > 0;
- standard and paper utilization lie in [0,1] within 1e-9 tolerance;
- PM/CM/setup/process durations are positive;
- evaluation epsilon is exactly 0.0;
- missing checkpoints are never silently accepted;
- run contract carries current Git SHA, profile SHA, and reference-bank SHA;
- no file under repository-root code/code1/code2 changed;
- no generated bank/checkpoint/result file is staged or tracked accidentally.
```

Repeated deterministic evaluation:

```text
- load the same checkpoint twice into fresh agents;
- evaluate the same instance with identical environment streams;
- canonical episode JSON bytes and SHA-256 must be identical;
- do not require training on GPU to be bitwise identical across separate processes;
  reproducibility is checked through same-process checkpoint/evaluation replay and recorded seeds.
```

- [ ] **Step 6: Define the preflight report schema**

Top level:

```json
{
  "schema_version": 1,
  "status": "passed",
  "git_commit": "40-hex-sha",
  "bank_manifest_sha256": "68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515",
  "environment_metadata_sha256": "64-hex-sha",
  "device": "NVIDIA GeForce RTX 5090",
  "gates": [],
  "episodes": [],
  "checkpoint_roundtrips": [],
  "determinism_checks": [],
  "generated_artifacts": [],
  "warnings": []
}
```

Each gate contains:

```text
name
status
started_at_utc
ended_at_utc
duration_seconds
evidence_sha256
message
```

Timestamps and durations are excluded from the report's separate `scientific_payload_sha256`; that hash covers only deterministic profile, contract, bank, episode, and checkpoint evidence.

- [ ] **Step 7: Write preflight unit tests**

Tests use tiny hand instances and CPU unless a test specifically checks device selection. Cover:

```text
1. Canonical JSON ordering and NaN rejection.
2. A passing gate produces stable evidence hash.
3. A failed gate writes `status=failed` before raising.
4. Profile/method/seed ordering is exact.
5. Fixed bank paths are validated and cannot escape the bank root.
6. Invalid schedule is rejected before metrics enter the report.
7. Evaluation epsilon other than zero fails.
8. Repeated deterministic episode payloads compare byte-for-byte.
9. Generated artifacts outside ignored roots fail.
10. A mock missing checkpoint is a hard failure.
```

- [ ] **Step 8: Update README with the exact no-download/data preparation statement**

Add a section titled `Original-conference data preparation`:

```markdown
The original SMC study uses synthetic instances; no public benchmark download is required for
this reproduction phase. The repository commits the reference manifest, not the 1540 compressed
instances. Generate and verify them locally before preflight with the commands below.
```

Include the full builder/verifier/preflight commands from Tasks 3 and 9 for both PowerShell and Bash. State explicitly that Brandimarte/Hurink and other external benchmarks are deferred until the later GNN-generalization phase.

- [ ] **Step 9: Commit the preflight implementation before running the real clean gate**

```bash
git add 扩刊/original_repro/src/smc_repro/scripts/clean_worktree_gate.py \
  扩刊/original_repro/src/smc_repro/scripts/preflight.py \
  扩刊/original_repro/tests/test_preflight.py \
  扩刊/original_repro/README.md \
  扩刊/all.md
git commit -m "test: add clean-clone and end-to-end preflight gates"
```

- [ ] **Step 10: Run the complete quality suite on the RTX 5090 host**

PowerShell:

```powershell
$repo = (git rev-parse --show-toplevel).Trim()
cd "$repo\扩刊\original_repro"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
python -m smc_repro.scripts.verify_hardware
python -m smc_repro.scripts.verify_instance_bank `
  --reference artifacts\banks\release\manifest.json `
  --bank-root artifacts\banks\materialized `
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 `
  --report artifacts\preflight\bank_verification.json
python -m smc_repro.scripts.preflight `
  --repo-root ..\.. `
  --bank-root artifacts\banks\materialized `
  --reference-manifest artifacts\banks\release\manifest.json `
  --environment-metadata ..\docs\audit\environment_5090_resolved.json `
  --output artifacts\preflight\preflight_report.json `
  --device cuda:0
$pythonExe = (Get-Command python).Source
python -m smc_repro.scripts.clean_worktree_gate `
  --repo-root ..\.. `
  --python-executable $pythonExe `
  --report artifacts\preflight\clean_worktree_report.json
```

Bash:

```bash
repo="$(git rev-parse --show-toplevel)"
cd "$repo/扩刊/original_repro"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m compileall -q src tests
python -m smc_repro.scripts.verify_hardware
python -m smc_repro.scripts.verify_instance_bank \
  --reference artifacts/banks/release/manifest.json \
  --bank-root artifacts/banks/materialized \
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 \
  --report artifacts/preflight/bank_verification.json
python -m smc_repro.scripts.preflight \
  --repo-root ../.. \
  --bank-root artifacts/banks/materialized \
  --reference-manifest artifacts/banks/release/manifest.json \
  --environment-metadata ../docs/audit/environment_5090_resolved.json \
  --output artifacts/preflight/preflight_report.json \
  --device cuda:0
python -m smc_repro.scripts.clean_worktree_gate \
  --repo-root ../.. \
  --python-executable "$(command -v python)" \
  --report artifacts/preflight/clean_worktree_report.json
```

- [ ] **Step 11: Verify the final reports and repository state**

```bash
python - <<'PY'
import json
from pathlib import Path

for name in ("preflight_report.json", "clean_worktree_report.json"):
    path = Path("artifacts/preflight") / name
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "passed", (name, data)
    print(name, data.get("scientific_payload_sha256", data.get("report_sha256")))
PY

git status --short
git diff --exit-code -- ../../code ../../code1 ../../code2
git ls-files artifacts/banks/materialized artifacts/preflight
```

Acceptance:

```text
- both reports exist and pass;
- `git status --short` is empty;
- legacy diff exits 0;
- `git ls-files` prints no generated bank or preflight files;
- reference manifest SHA remains fixed;
- all 1540 materialized files verify;
- all formal preflight evaluation loads report epsilon 0;
- clean detached worktree passes without local ignored legacy files.
```

- [ ] **Step 12: Append the exact final evidence to `扩刊/all.md` and commit only that log update**

Record without paraphrasing:

```text
starting HEAD and ending HEAD
all Task 0-Task 9 commit SHAs and subjects
pytest total/pass/fail and duration
Ruff/mypy/compileall results
live GPU/torch/CUDA details
reference manifest SHA and 1540/1540 verification
pre-log HEAD and its preflight scientific payload SHA
pre-log clean-worktree report SHA
number of deep/tabular smoke episodes
checkpoint round-trip count
all deviations from this plan
all remaining issues
```

Then:

```bash
git add 扩刊/all.md
git commit -m "docs: archive pre-experiment readiness evidence"
```

The values committed above are explicitly the **pre-log-commit** evidence. A report whose deterministic payload includes the current Git SHA cannot have its own final hash committed into that same Git history without changing the SHA again. Therefore, at the new documentation HEAD, rerun both the full preflight and the clean-worktree gate so the final committed HEAD—not its parent—is freshly verified. Keep the final reports ignored and local, print their independent file SHA-256 values, and provide the complete reports with the next review request. Do not amend the evidence commit merely to chase a self-referential final hash.

```bash
repo="$(git rev-parse --show-toplevel)"
cd "$repo/扩刊/original_repro"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m smc_repro.scripts.preflight \
  --repo-root ../.. \
  --bank-root artifacts/banks/materialized \
  --reference-manifest artifacts/banks/release/manifest.json \
  --environment-metadata ../docs/audit/environment_5090_resolved.json \
  --output artifacts/preflight/preflight_report.json \
  --device cuda:0
python -m smc_repro.scripts.clean_worktree_gate \
  --repo-root ../.. \
  --python-executable "$(command -v python)" \
  --report artifacts/preflight/clean_worktree_report.json
git status --short
```

Both reports must show `status: passed`, and `git status --short` must remain empty. Also print `git rev-parse HEAD` and independent SHA-256 values for both final report files; these are the authoritative final-HEAD evidence supplied externally, not values claimed to be embedded in `all.md`.

- [ ] **Step 13: Stop**

Do not implement:

```text
formal training sweep scripts
540-instance method evaluation loops
parallel GPU scheduling
result aggregation
confidence intervals or statistical tests
paper tables/figures
GNN/PPO/new action spaces
external benchmark download
```

Upload the repository and provide the two ignored JSON reports separately or paste their complete content into the next review request. The next assistant cycle will audit this readiness state and then produce the formal experiment-code plan.

---

# Final Acceptance Checklist

The repository is ready for formal experiment-driver implementation only when every item below is true:

- [ ] Historical local-full manifests are preserved but excluded from clean-clone equality tests.
- [ ] Git-tracked legacy manifest passes in a detached clean worktree.
- [ ] Typed random keys distinguish delimiters and value types.
- [ ] Metadata is defensively copied and read-only.
- [ ] All recorded intervals have positive duration.
- [ ] Metrics clip to the scheduling horizon and validator rejects post-horizon maintenance/setup.
- [ ] The 1540-instance synthetic bank is materialized and verified against the committed reference SHA.
- [ ] A portable RTX 5090 environment snapshot is committed.
- [ ] Three strict profiles and every paper/code ambiguity are committed.
- [ ] State and rewards use named fields, not magic indices.
- [ ] Source, paper, and classical rules pass hand-calculated tests.
- [ ] The constructive environment emits explicit SETUP/PM/CM/PROCESS intervals and validates every completed schedule.
- [ ] Paper/source local-insertion discrepancy is recorded; no internally inconsistent retrospective insertion is enabled.
- [ ] Common random numbers are keyed by instance/job/operation/machine and do not depend on decision order or policy RNG call order.
- [ ] DL-DDQN, vanilla-DQN targets, Q-learning, and SARSA complete smoke episodes without non-finite values.
- [ ] Checkpoints support exact training resume and force epsilon 0 in evaluation.
- [ ] Full pytest, Ruff, mypy, compileall, bank verification, preflight, and clean-worktree gates pass.
- [ ] No full scientific experiment has been run or reported as a result.

---

# Plan Self-Review Record

Before delivery, this plan was checked for:

```text
- scope coverage from Gate 1 portability through the final preflight;
- exact separation of legacy_snapshot, paper_repro, and corrected_smc;
- exact 90th-percentile source high-load branch;
- exact source tool-change predicate using both job-machine and machine-job changes;
- Python 3.11-compatible type syntax;
- deterministic gzip/bank SHA preservation;
- lower-network context dimensional consistency;
- no silent checkpoint fallback;
- no external dataset requirement for the original-paper rerun;
- no formal experiment or GNN implementation in this phase.
```

Codex must still execute every test and gate in the target repository and RTX 5090 environment. The embedded code is an implementation contract, not evidence that the target repository already passes.
