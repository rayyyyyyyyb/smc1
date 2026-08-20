from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform as platform_module
import re
import stat as stat_module
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

import numpy as np
import torch

from smc_repro.agents.checkpoint import load_checkpoint, save_checkpoint
from smc_repro.agents.dual_ddqn import DualLayerValueAgent
from smc_repro.agents.networks import build_lower_context
from smc_repro.agents.replay import ReplayBuffer, Transition
from smc_repro.agents.tabular import TabularAgent, TabularAlgorithm
from smc_repro.config import ReproductionProfile, load_profile, profile_sha256
from smc_repro.environment import EpisodeResult, SchedulingEnvironment
from smc_repro.experiment_contract import (
    RunContract,
    build_run_contract,
    collect_git_commit,
)
from smc_repro.experiment_contract import contract_sha256 as run_contract_sha256
from smc_repro.instance_io import load_instance
from smc_repro.rewards import RewardMode, legacy_joint_reward
from smc_repro.rules.classical import ClassicalRule
from smc_repro.schemas import InstanceSpec, IntervalType, ScheduleInterval
from smc_repro.scripts.audit_legacy_outputs import AuditScope, audit_legacy_outputs
from smc_repro.scripts.verify_instance_bank import verify_instance_bank

EXPECTED_BANK_MANIFEST_SHA256 = (
    "68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515"
)
PROFILE_ORDER = ("legacy_snapshot", "paper_repro", "corrected_smc")
METHOD_ORDER = ("ddqn", "vanilla_dqn_target", "q_learning", "sarsa")
GATE_NAMES = tuple(f"P{index:02d}" for index in range(16))
TRAIN_BANK_PATHS = (
    "train/seed_000/train_seed000_ep0000.json.gz",
    "train/seed_000/train_seed000_ep0001.json.gz",
    "train/seed_000/train_seed000_ep0002.json.gz",
)
EVALUATION_BANK_PATHS = (
    "test/m08_j10_e050/test_m08_j10_e050_rep00.json.gz",
    "test/m08_j10_e050/test_m08_j10_e050_rep01.json.gz",
)
_PROFILE_FILES = {
    "legacy_snapshot": "legacy_snapshot.yaml",
    "paper_repro": "paper_repro.yaml",
    "corrected_smc": "corrected_smc.yaml",
}
_GATE_MESSAGES = {
    "P00": "repository cleanliness and current Git SHA",
    "P01": "environment metadata and live RTX 5090 CUDA smoke",
    "P02": "tracked legacy manifest equality",
    "P03": "release-manifest SHA equality",
    "P04": "materialized bank verification",
    "P05": "strict profile and smoke-override loading",
    "P06": "18 composite-rule non-learning completions",
    "P07": "five classical-rule non-learning completions",
    "P08": "three-profile DDQN smoke",
    "P09": "three-profile vanilla-target DQN smoke",
    "P10": "paper-profile Q-learning and SARSA smoke",
    "P11": "strict checkpoint training/evaluation round trips",
    "P12": "epsilon-zero fixed-instance evaluation",
    "P13": "schedule and finite-metric validation",
    "P14": "deterministic evaluation replay equality",
    "P15": "forbidden-artifact and legacy-diff protection",
}
_ENVIRONMENT_FIELDS = {
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "bank_manifest_sha256",
    "captured_at_utc",
    "compiled_cuda_arches",
    "compute_capability",
    "cuda_available",
    "cuda_smoke_result",
    "git_commit",
    "gpu_name",
    "packages",
    "platform",
    "python_implementation",
    "python_version",
    "schema_version",
    "torch_cuda_runtime",
    "torch_version",
}
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


class PreflightError(RuntimeError):
    """Raised only after the failed preflight report is persisted."""


class PreflightInputError(ValueError):
    """Raised before gate execution when no safe report destination exists."""


@dataclasses.dataclass(frozen=True)
class LiveEnvironmentFacts:
    torch_version: str
    torch_cuda_runtime: str
    gpu_name: str
    compute_capability: tuple[int, int]
    compiled_cuda_arches: tuple[str, ...]
    cuda_smoke_result: float
    python_version: str
    python_implementation: str
    platform: str


@dataclasses.dataclass(frozen=True)
class _ValidatedPreflightPaths:
    repo_root: Path
    bank_root: Path
    reference_manifest: Path
    environment_metadata: Path
    output: Path


def canonical_json_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"value is not valid canonical JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _same_lexical_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(first)) == os.path.normcase(str(second))


def _input_error(message: str) -> PreflightInputError:
    return PreflightInputError(f"{message}; no report was written")


def _reject_reparse_components(path: Path, repo_root: Path, name: str) -> None:
    if not path.is_relative_to(repo_root):
        raise _input_error(f"{name} must remain lexically inside the repository")
    current = repo_root
    for component in path.relative_to(repo_root).parts:
        current /= component
        try:
            status = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise _input_error(f"unable to inspect {name} path component: {exc}") from exc
        attributes = getattr(status, "st_file_attributes", 0)
        reparse_attribute = getattr(
            stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        )
        if stat_module.S_ISLNK(status.st_mode) or attributes & reparse_attribute:
            raise _input_error(f"{name} must not contain symlink or reparse components")


def _validate_preflight_paths(
    *,
    repo_root: Path,
    bank_root: Path,
    reference_manifest: Path,
    environment_metadata: Path,
    output: Path,
) -> _ValidatedPreflightPaths:
    """Validate every public path before any gate or report write."""
    for name, value in (
        ("repo_root", repo_root),
        ("bank_root", bank_root),
        ("reference_manifest", reference_manifest),
        ("environment_metadata", environment_metadata),
        ("output", output),
    ):
        if not isinstance(value, Path):
            raise _input_error(f"{name} must be a Path")

    repo_lexical = _absolute_without_symlink_resolution(repo_root)
    if not repo_lexical.is_dir():
        raise _input_error("repo_root must be an existing directory")
    repo_resolved = repo_lexical.resolve(strict=True)
    project_lexical = repo_resolved / "扩刊/original_repro"
    expansion_lexical = repo_resolved / "扩刊"
    expected_bank = project_lexical / "artifacts/banks/materialized"
    expected_reference = project_lexical / "artifacts/banks/release/manifest.json"
    expected_environment = expansion_lexical / "docs/audit/environment_5090_resolved.json"
    preflight_root = project_lexical / "artifacts/preflight"

    supplied_bank = _absolute_without_symlink_resolution(bank_root)
    supplied_reference = _absolute_without_symlink_resolution(reference_manifest)
    supplied_environment = _absolute_without_symlink_resolution(environment_metadata)
    supplied_output = _absolute_without_symlink_resolution(output)
    for name, supplied, expected in (
        ("bank_root", supplied_bank, expected_bank),
        ("reference_manifest", supplied_reference, expected_reference),
        ("environment_metadata", supplied_environment, expected_environment),
    ):
        if not _same_lexical_path(supplied, expected):
            raise _input_error(f"{name} must use the repository-controlled path")

    if not supplied_output.is_relative_to(preflight_root) or supplied_output == preflight_root:
        raise _input_error("output must be a file below the controlled preflight root")
    if supplied_output.exists() and supplied_output.is_dir():
        raise _input_error("output must not be a directory")

    for name, supplied in (
        ("bank_root", supplied_bank),
        ("reference_manifest", supplied_reference),
        ("environment_metadata", supplied_environment),
        ("preflight root", preflight_root),
        ("output", supplied_output),
    ):
        _reject_reparse_components(supplied, repo_resolved, name)

    try:
        bank_resolved = supplied_bank.resolve(strict=True)
        reference_resolved = supplied_reference.resolve(strict=True)
        environment_resolved = supplied_environment.resolve(strict=True)
        preflight_resolved = preflight_root.resolve(strict=False)
        output_resolved = supplied_output.resolve(strict=False)
    except OSError as exc:
        raise _input_error(f"unable to resolve preflight paths: {exc}") from exc
    if not _same_lexical_path(preflight_resolved, preflight_root):
        raise _input_error(
            "controlled preflight root must resolve to its exact canonical path"
        )
    if not bank_resolved.is_dir():
        raise _input_error("bank_root must be an existing directory")
    if not reference_resolved.is_file():
        raise _input_error("reference_manifest must be an existing file")
    if not environment_resolved.is_file():
        raise _input_error("environment_metadata must be an existing file")
    for name, resolved in (
        ("bank_root", bank_resolved),
        ("reference_manifest", reference_resolved),
        ("environment_metadata", environment_resolved),
        ("preflight root", preflight_resolved),
        ("output", output_resolved),
    ):
        if not resolved.is_relative_to(repo_resolved):
            raise _input_error(f"{name} resolves outside the repository")
    if not output_resolved.is_relative_to(preflight_resolved):
        raise _input_error("output resolves outside the controlled preflight root")
    if not bank_resolved.is_relative_to(project_lexical.resolve(strict=True)):
        raise _input_error("bank_root resolves outside the controlled project")

    return _ValidatedPreflightPaths(
        repo_root=repo_resolved,
        bank_root=bank_resolved,
        reference_manifest=reference_resolved,
        environment_metadata=environment_resolved,
        output=output_resolved,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _scientific_payload(report: Mapping[str, object]) -> dict[str, object]:
    deterministic_fields = (
        "git_commit",
        "bank_manifest_sha256",
        "environment_metadata_sha256",
        "device",
        "profile_order",
        "method_order",
        "train_bank_paths",
        "evaluation_bank_paths",
        "training_plan",
        "environment_evidence",
        "bank_verification",
        "contracts",
        "episodes",
        "checkpoint_roundtrips",
        "determinism_checks",
        "generated_artifacts",
    )
    gates = cast(Sequence[Mapping[str, object]], report.get("gates", ()))
    return {
        **{name: report.get(name) for name in deterministic_fields},
        "gate_evidence": [
            {
                "name": gate.get("name"),
                "status": gate.get("status"),
                "evidence_sha256": gate.get("evidence_sha256"),
                "message": gate.get("message"),
            }
            for gate in gates
        ],
    }


def _write_report(path: Path, report: dict[str, object]) -> None:
    report["scientific_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(_scientific_payload(report))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _base_report(initial: Mapping[str, object]) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "git_commit": "",
        "bank_manifest_sha256": EXPECTED_BANK_MANIFEST_SHA256,
        "environment_metadata_sha256": "",
        "device": "",
        "profile_order": list(PROFILE_ORDER),
        "method_order": list(METHOD_ORDER),
        "train_bank_paths": list(TRAIN_BANK_PATHS),
        "evaluation_bank_paths": list(EVALUATION_BANK_PATHS),
        "training_plan": [
            {
                "profile": profile,
                "method": method,
                "agent_train_seed": seed,
                "replay_seed": seed + 1,
            }
            for profile, method, seed in planned_training_cells()
        ],
        "environment_evidence": {},
        "bank_verification": {},
        "gates": [],
        "contracts": [],
        "episodes": [],
        "checkpoint_roundtrips": [],
        "determinism_checks": [],
        "generated_artifacts": [],
        "warnings": [],
    }
    report.update(initial)
    return report


def run_gate_sequence(
    report_path: Path,
    initial_report: Mapping[str, object],
    gates: Sequence[tuple[str, Callable[[], object]]],
) -> dict[str, object]:
    """Run in order, persist after every gate, and stop after the first failure."""
    report = (
        cast(dict[str, object], initial_report)
        if isinstance(initial_report, dict)
        and initial_report.get("schema_version") == 1
        and "gates" in initial_report
        else _base_report(initial_report)
    )
    gate_records = cast(list[dict[str, object]], report["gates"])
    for name, gate in gates:
        started = _utc_now()
        start = time.perf_counter()
        try:
            evidence = gate()
            evidence_hash = sha256_bytes(canonical_json_bytes(evidence))
            gate_records.append(
                {
                    "name": name,
                    "status": "passed",
                    "started_at_utc": started,
                    "ended_at_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - start,
                    "evidence_sha256": evidence_hash,
                    "message": _GATE_MESSAGES.get(name, f"{name} passed"),
                }
            )
            _write_report(report_path, report)
        except Exception as exc:
            failure_evidence = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            gate_records.append(
                {
                    "name": name,
                    "status": "failed",
                    "started_at_utc": started,
                    "ended_at_utc": _utc_now(),
                    "duration_seconds": time.perf_counter() - start,
                    "evidence_sha256": sha256_bytes(
                        canonical_json_bytes(failure_evidence)
                    ),
                    "message": str(exc),
                }
            )
            report["status"] = "failed"
            _write_report(report_path, report)
            raise PreflightError(f"{name} failed: {exc}") from exc
    report["status"] = "passed"
    _write_report(report_path, report)
    return report


def planned_training_cells() -> tuple[tuple[str, str, int], ...]:
    profile_index = {name: index for index, name in enumerate(PROFILE_ORDER)}
    method_index = {name: index for index, name in enumerate(METHOD_ORDER)}
    cells: list[tuple[str, str, int]] = []
    for method in METHOD_ORDER[:2]:
        for profile in PROFILE_ORDER:
            cells.append(
                (
                    profile,
                    method,
                    61000 + profile_index[profile] * 100 + method_index[method],
                )
            )
    for method in METHOD_ORDER[2:]:
        cells.append(
            (
                "paper_repro",
                method,
                61000
                + profile_index["paper_repro"] * 100
                + method_index[method],
            )
        )
    return tuple(cells)


def validate_fixed_bank_paths(
    bank_root: Path,
    relative_paths: Sequence[str],
    *,
    require_files: bool = True,
) -> tuple[Path, ...]:
    root = bank_root.resolve()
    resolved: list[Path] = []
    for relative in relative_paths:
        if not isinstance(relative, str):
            raise ValueError("bank paths must be strings")
        pure = PurePosixPath(relative)
        windows = PureWindowsPath(relative)
        if (
            pure.is_absolute()
            or windows.drive
            or windows.root
            or "\\" in relative
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != relative
        ):
            raise ValueError(f"bank path is unsafe or noncanonical: {relative!r}")
        target = root.joinpath(*pure.parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"bank path escapes bank root: {relative!r}")
        if require_files and not target.is_file():
            raise FileNotFoundError(f"fixed bank instance is missing: {relative}")
        resolved.append(target)
    return tuple(resolved)


def validate_evaluation_epsilon(value: object) -> None:
    if type(value) not in (int, float) or float(cast(int | float, value)) != 0.0:
        raise ValueError("evaluation epsilon must be exactly 0.0")


def _required_string(metadata: Mapping[str, object], field: str) -> str:
    value = metadata[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"environment metadata {field} must be a non-empty string")
    return value


def validate_environment_snapshot(
    metadata: Mapping[str, object],
    live: LiveEnvironmentFacts,
) -> dict[str, object]:
    if not isinstance(metadata, Mapping) or not all(
        isinstance(key, str) for key in metadata
    ):
        raise ValueError("environment metadata must be a string-keyed object")
    if set(metadata) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment metadata fields are incomplete or unknown")
    if type(metadata["schema_version"]) is not int or metadata["schema_version"] != 1:
        raise ValueError("environment metadata schema_version must be the integer 1")
    capture_commit = metadata["git_commit"]
    if not isinstance(capture_commit, str) or _COMMIT_PATTERN.fullmatch(
        capture_commit
    ) is None:
        raise ValueError(
            "environment metadata git_commit must be a canonical captured 40-hex SHA"
        )
    if metadata["cuda_available"] is not True:
        raise ValueError("environment metadata cuda_available must be true")
    if metadata["bank_manifest_sha256"] != EXPECTED_BANK_MANIFEST_SHA256:
        raise ValueError("environment metadata bank manifest SHA mismatch")
    if metadata["PYTHONHASHSEED"] != "0":
        raise ValueError("environment metadata PYTHONHASHSEED must equal 0")
    if metadata["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError(
            "environment metadata CUBLAS_WORKSPACE_CONFIG must equal :4096:8"
        )

    captured_at = _required_string(metadata, "captured_at_utc")
    try:
        captured_datetime = datetime.fromisoformat(captured_at)
    except ValueError as exc:
        raise ValueError(
            "environment metadata captured_at_utc must be ISO-8601"
        ) from exc
    if captured_datetime.tzinfo is None:
        raise ValueError("environment metadata captured_at_utc must include a timezone")

    capability_raw = metadata["compute_capability"]
    if (
        not isinstance(capability_raw, list)
        or len(capability_raw) != 2
        or any(type(value) is not int or value < 0 for value in capability_raw)
    ):
        raise ValueError(
            "environment metadata compute_capability must contain two non-negative integers"
        )
    capability = (cast(int, capability_raw[0]), cast(int, capability_raw[1]))
    arches_raw = metadata["compiled_cuda_arches"]
    if (
        not isinstance(arches_raw, list)
        or not arches_raw
        or not all(isinstance(value, str) and value for value in arches_raw)
        or len(set(cast(list[str], arches_raw))) != len(arches_raw)
    ):
        raise ValueError(
            "environment metadata compiled_cuda_arches must be unique non-empty strings"
        )
    arches = tuple(cast(list[str], arches_raw))
    smoke_raw = metadata["cuda_smoke_result"]
    if isinstance(smoke_raw, bool) or not isinstance(smoke_raw, (int, float)):
        raise ValueError("environment metadata cuda_smoke_result must be finite")
    smoke = float(smoke_raw)
    if not math.isfinite(smoke) or smoke != 14.0:
        raise ValueError("environment metadata cuda_smoke_result must equal 14.0")

    packages_raw = metadata["packages"]
    if not isinstance(packages_raw, list) or not packages_raw:
        raise ValueError("environment metadata packages must be a non-empty list")
    package_names: list[str] = []
    for package in packages_raw:
        if not isinstance(package, dict) or set(package) != {"name", "version"}:
            raise ValueError("environment metadata package records are invalid")
        name = package["name"]
        version = package["version"]
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError("environment metadata package records are invalid")
        package_names.append(name.casefold().replace("_", "-"))
    if package_names != sorted(package_names) or len(package_names) != len(
        set(package_names)
    ):
        raise ValueError("environment metadata packages must be sorted and unique")

    captured_facts = {
        "torch_version": _required_string(metadata, "torch_version"),
        "torch_cuda_runtime": _required_string(metadata, "torch_cuda_runtime"),
        "gpu_name": _required_string(metadata, "gpu_name"),
        "compute_capability": capability,
        "compiled_cuda_arches": arches,
        "cuda_smoke_result": smoke,
        "python_version": _required_string(metadata, "python_version"),
        "python_implementation": _required_string(metadata, "python_implementation"),
        "platform": _required_string(metadata, "platform"),
    }
    live_facts = dataclasses.asdict(live)
    for field, observed in captured_facts.items():
        expected = live_facts[field]
        if observed != expected:
            raise ValueError(f"environment metadata live mismatch for {field}")
    if "RTX 5090" not in cast(str, captured_facts["gpu_name"]):
        raise ValueError("environment metadata gpu_name must identify the RTX 5090")
    if "sm_120" not in arches:
        raise ValueError("environment metadata compiled arches must include sm_120")

    return {
        "schema_version": 1,
        "capture_git_commit": capture_commit,
        "capture_git_commit_is_provenance_only": True,
        "cuda_available": True,
        "bank_manifest_sha256": EXPECTED_BANK_MANIFEST_SHA256,
        **captured_facts,
    }


def require_checkpoint(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint is missing: {path}")
    return path


def _transition_primitive(transition: Transition) -> dict[str, object]:
    if type(transition) is not Transition:
        raise TypeError("replay samples must contain Transition records")
    return {
        "state": transition.state.tolist(),
        "rule_action": transition.rule_action,
        "reward": transition.reward,
        "next_state": transition.next_state.tolist(),
        "reward_id": transition.reward_id,
        "done": transition.done,
    }


def compare_replay_samples(
    first: Sequence[Transition],
    second: Sequence[Transition],
) -> dict[str, object]:
    first_payload = [_transition_primitive(item) for item in first]
    second_payload = [_transition_primitive(item) for item in second]
    first_bytes = canonical_json_bytes(first_payload)
    second_bytes = canonical_json_bytes(second_payload)
    if first_bytes != second_bytes:
        raise ValueError(
            "replay samples differ in complete Transition primitive fields"
        )
    digest = sha256_bytes(first_bytes)
    return {
        "equal": True,
        "sample_size": len(first_payload),
        "sample_sha256": digest,
    }


def compare_deterministic_episode_payloads(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> dict[str, object]:
    first_bytes = canonical_json_bytes(dict(first))
    second_bytes = canonical_json_bytes(dict(second))
    first_sha = sha256_bytes(first_bytes)
    second_sha = sha256_bytes(second_bytes)
    if first_bytes != second_bytes:
        raise ValueError(
            "deterministic episode payloads are not byte-for-byte identical"
        )
    return {
        "identical": True,
        "first_sha256": first_sha,
        "second_sha256": second_sha,
        "byte_count": len(first_bytes),
    }


def validate_generated_artifacts(
    repo_root: Path,
    paths: Sequence[Path],
    *,
    require_git_ignored: bool = True,
) -> tuple[str, ...]:
    root = repo_root.resolve()
    allowed_roots = (
        (root / "扩刊/original_repro/artifacts/banks/materialized").resolve(),
        (root / "扩刊/original_repro/artifacts/preflight").resolve(),
    )
    relative_paths: list[str] = []
    for path in paths:
        target = path.resolve()
        if not any(target.is_relative_to(allowed) for allowed in allowed_roots):
            raise ValueError(f"generated artifact is outside allowed ignored roots: {path}")
        relative = target.relative_to(root).as_posix()
        if require_git_ignored:
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", "--", relative],
                cwd=root,
                check=False,
            )
            if ignored.returncode != 0:
                raise ValueError(f"generated artifact is not Git-ignored: {relative}")
        relative_paths.append(relative)
    return tuple(relative_paths)


def _finite_sequence(values: Sequence[object], name: str) -> list[float]:
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain only finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain only finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite numbers")
        converted.append(number)
    return converted


def _interval_dict(interval: ScheduleInterval) -> dict[str, object]:
    return {
        "machine_id": interval.machine_id,
        "start": interval.start,
        "end": interval.end,
        "duration": interval.duration,
        "interval_type": interval.interval_type.value,
        "job_id": interval.job_id,
        "op_id": interval.op_id,
        "metadata": dict(interval.metadata),
    }


def episode_record(
    result: EpisodeResult | object,
    *,
    profile_sha256: str,
    contract_sha256: str,
    rewards: Sequence[object],
    losses: Sequence[object],
    checkpoint_sha256: str | None,
    evaluation_epsilon: float | None,
    observations: Sequence[Sequence[object]],
    target_values: Sequence[object],
    contract: RunContract | None = None,
    decision_trace: Sequence[Mapping[str, object]] = (),
    total_operations: int | None = None,
    network_parameters: Sequence[object] = (),
    q_table_values: Sequence[object] = (),
    method: str = "",
    phase: str = "",
    episode_index: int = 0,
    stream_seeds: Mapping[str, int] | None = None,
) -> dict[str, object]:
    episode = cast(EpisodeResult, result)
    validation = episode.validation
    if not validation.ok:
        errors = validation.errors
        raise ValueError("schedule validation failed: " + "; ".join(errors))

    intervals = tuple(episode.intervals)
    ordered_intervals = tuple(
        sorted(
            intervals,
            key=lambda item: (
                item.machine_id,
                item.start,
                item.end,
                item.interval_type.value,
            ),
        )
    )
    if any(interval.duration <= 0.0 for interval in ordered_intervals):
        raise ValueError("all schedule interval durations must be positive")
    process_intervals = tuple(
        interval
        for interval in ordered_intervals
        if interval.interval_type is IntervalType.PROCESS
    )
    decisions = episode.decisions
    expected_operations = decisions if total_operations is None else total_operations
    if decisions != expected_operations:
        raise ValueError("number of decisions must equal total number of operations")
    process_keys = {(item.job_id, item.op_id) for item in process_intervals}
    if len(process_intervals) != expected_operations or len(process_keys) != expected_operations:
        raise ValueError("exactly one PROCESS interval is required per operation")

    metrics = episode.metrics
    if not dataclasses.is_dataclass(metrics) or isinstance(metrics, type):
        raise TypeError("episode metrics must be a dataclass instance")
    metrics_dict = dataclasses.asdict(metrics)
    _finite_sequence(list(metrics_dict.values()), "metric fields")
    makespan = float(metrics_dict["makespan"])
    if makespan <= 0.0:
        raise ValueError("episode makespan must be positive")
    for field in ("standard_utilization", "paper_uave"):
        value = float(metrics_dict[field])
        if not -1e-9 <= value <= 1.0 + 1e-9:
            raise ValueError(f"{field} must be in [0, 1] within tolerance")

    flattened_observations = [value for row in observations for value in row]
    rewards_values = _finite_sequence(rewards, "reward sequence")
    losses_values = _finite_sequence(losses, "loss sequence")
    observations_values = _finite_sequence(
        flattened_observations, "observation sequence"
    )
    targets_values = _finite_sequence(target_values, "target values")
    parameter_values = _finite_sequence(network_parameters, "network parameters")
    q_values = _finite_sequence(q_table_values, "Q-table values")
    if evaluation_epsilon is not None:
        validate_evaluation_epsilon(evaluation_epsilon)
    for name, digest in (
        ("profile_sha256", profile_sha256),
        ("contract_sha256", contract_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name} must be a canonical SHA-256")
    if checkpoint_sha256 is not None and (
        len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError("checkpoint_sha256 must be a canonical SHA-256")
    if contract is not None:
        if (
            contract.profile_sha256 != profile_sha256
            or contract_sha256 != run_contract_sha256(contract)
        ):
            raise ValueError("run contract profile/contract SHA mismatch")

    return {
        "instance_id": episode.instance_id,
        "profile": episode.profile_name,
        "method": method,
        "phase": phase,
        "episode_index": episode_index,
        "stream_seeds": dict(stream_seeds or {}),
        "profile_sha256": profile_sha256,
        "contract": dataclasses.asdict(contract) if contract is not None else None,
        "contract_sha256": contract_sha256,
        "decision_count": decisions,
        "total_operations": expected_operations,
        "decisions": [dict(item) for item in decision_trace],
        "intervals": [_interval_dict(item) for item in ordered_intervals],
        "metrics": metrics_dict,
        "validator": {"ok": True, "errors": []},
        "observations": [
            _finite_sequence(row, "observation row") for row in observations
        ],
        "rewards": rewards_values,
        "losses": losses_values,
        "target_values": targets_values,
        "network_parameter_sha256": sha256_bytes(
            canonical_json_bytes(parameter_values)
        ),
        "q_table_sha256": sha256_bytes(canonical_json_bytes(q_values)),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_epsilon": evaluation_epsilon,
        "canonical_observation_value_count": len(observations_values),
    }


def _total_operations(instance: InstanceSpec) -> int:
    return sum(len(job.operations) for job in instance.jobs)


def _network_values(agent: DualLayerValueAgent) -> list[float]:
    values: list[float] = []
    for network in (
        agent.upper_online,
        agent.upper_target,
        agent.lower_online,
        agent.lower_target,
    ):
        for parameter in network.parameters():
            values.extend(parameter.detach().cpu().reshape(-1).tolist())
    _finite_sequence(values, "network parameters")
    return values


def _target_values(agent: DualLayerValueAgent, state: np.ndarray) -> list[float]:
    tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        upper = agent.upper_target(tensor)
        reward_ids = torch.zeros(1, dtype=torch.int64, device=agent.device)
        context = build_lower_context(
            tensor,
            upper,
            reward_ids,
            agent.profile.architecture.lower_context,
        )
        lower = agent.lower_target(context)
    values = cast(
        list[float],
        torch.cat((upper.reshape(-1), lower.reshape(-1))).detach().cpu().tolist(),
    )
    _finite_sequence(values, "target values")
    return values


def _new_environment(
    instance: InstanceSpec,
    profile: ReproductionProfile,
    policy_seed: int,
) -> SchedulingEnvironment:
    return SchedulingEnvironment(
        instance,
        profile,
        policy_seed=policy_seed,
        failure_seed=instance.failure_seed,
        wear_seed=instance.failure_seed + 10_000_000,
        repair_seed=instance.failure_seed + 20_000_000,
    )


class _PreflightRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        bank_root: Path,
        reference_manifest: Path,
        environment_metadata: Path,
        output: Path,
        device: str,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.project_root = self.repo_root / "扩刊/original_repro"
        self.bank_root = bank_root.resolve()
        self.reference_manifest = reference_manifest.resolve()
        self.environment_metadata = environment_metadata.resolve()
        self.output = output.resolve()
        self.requested_device = device
        self.report = _base_report({"device": device})
        self.git_commit = ""
        self.profiles: dict[str, ReproductionProfile] = {}
        self.train_instances: tuple[InstanceSpec, ...] = ()
        self.evaluation_instances: tuple[InstanceSpec, ...] = ()
        self.deep_runs: dict[
            tuple[str, str],
            tuple[DualLayerValueAgent, RunContract, Path, str, np.ndarray],
        ] = {}

    def append_report(self, field: str, value: object) -> None:
        items = cast(list[object], self.report[field])
        items.append(value)

    def contract(self, profile: ReproductionProfile, method: str, seed: int) -> RunContract:
        contract = build_run_contract(
            self.repo_root,
            profile,
            bank_manifest_sha256=EXPECTED_BANK_MANIFEST_SHA256,
            method=method,
            train_seed=seed,
            policy_seed=62000,
            environment_metadata_path=self.environment_metadata.relative_to(
                self.repo_root
            ),
        )
        self.append_report("contracts", dataclasses.asdict(contract))
        return contract

    def p00(self) -> object:
        self.git_commit = collect_git_commit(self.repo_root)
        self.report["git_commit"] = self.git_commit
        self._record_generated(self.output)
        return {"git_commit": self.git_commit, "clean": True}

    def p01(self) -> object:
        if os.environ.get("PYTHONHASHSEED") != "0":
            raise RuntimeError("PYTHONHASHSEED must equal 0")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
        metadata_raw: object = json.loads(
            self.environment_metadata.read_text(encoding="utf-8")
        )
        if not isinstance(metadata_raw, dict):
            raise ValueError("environment metadata must be a JSON object")
        metadata = cast(dict[str, object], metadata_raw)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; expected RTX 5090")
        device = torch.device(self.requested_device)
        if device.type != "cuda":
            raise ValueError("formal preflight device must be CUDA")
        vector = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = float(torch.sum(vector * vector).item())
        torch.cuda.synchronize(device)
        name = torch.cuda.get_device_name(device)
        capability = list(torch.cuda.get_device_capability(device))
        if "RTX 5090" not in name:
            raise RuntimeError(f"expected RTX 5090, observed {name}")
        if result != 14.0:
            raise RuntimeError("CUDA tensor smoke did not return 14.0")
        cuda_runtime = torch.version.cuda
        if not isinstance(cuda_runtime, str) or not cuda_runtime:
            raise RuntimeError("live torch CUDA runtime version is unavailable")
        live = LiveEnvironmentFacts(
            torch_version=str(torch.__version__),
            torch_cuda_runtime=cuda_runtime,
            gpu_name=name,
            compute_capability=(capability[0], capability[1]),
            compiled_cuda_arches=tuple(torch.cuda.get_arch_list()),
            cuda_smoke_result=result,
            python_version=platform_module.python_version(),
            python_implementation=platform_module.python_implementation(),
            platform=platform_module.platform(),
        )
        evidence = validate_environment_snapshot(metadata, live)
        metadata_sha = _sha256_file(self.environment_metadata)
        self.report["environment_metadata_sha256"] = metadata_sha
        self.report["device"] = name
        evidence = {**evidence, "environment_metadata_sha256": metadata_sha}
        self.report["environment_evidence"] = evidence
        return evidence

    def p02(self) -> object:
        expected_path = self.repo_root / "扩刊/docs/audit/legacy_tracked_manifest.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        output = self.output.parent / "legacy_tracked_preflight.json"
        observed = audit_legacy_outputs(self.repo_root, output, AuditScope.TRACKED)
        if observed != expected:
            raise ValueError("tracked legacy audit differs from committed manifest")
        self._record_generated(output)
        return {
            "equal": True,
            "file_count": len(cast(dict[str, object], observed["files"])),
            "manifest_sha256": sha256_bytes(canonical_json_bytes(observed)),
        }

    def p03(self) -> object:
        observed = _sha256_file(self.reference_manifest)
        if observed != EXPECTED_BANK_MANIFEST_SHA256:
            raise ValueError("release manifest SHA-256 mismatch")
        self.report["bank_manifest_sha256"] = observed
        return {"manifest_sha256": observed}

    def p04(self) -> object:
        verified = verify_instance_bank(
            self.reference_manifest,
            self.bank_root,
            EXPECTED_BANK_MANIFEST_SHA256,
        )
        if (
            not verified.ok
            or verified.expected_file_count != 1540
            or verified.verified_file_count != 1540
        ):
            raise ValueError("materialized bank verification is not 1540/1540")
        train_paths = validate_fixed_bank_paths(self.bank_root, TRAIN_BANK_PATHS)
        evaluation_paths = validate_fixed_bank_paths(
            self.bank_root, EVALUATION_BANK_PATHS
        )
        self.train_instances = tuple(load_instance(path) for path in train_paths)
        self.evaluation_instances = tuple(
            load_instance(path) for path in evaluation_paths
        )
        evidence = {
            "reference_manifest": self.reference_manifest.relative_to(
                self.repo_root
            ).as_posix(),
            "reference_manifest_sha256": verified.reference_manifest_sha256,
            "generated_manifest": (self.bank_root / "manifest.json")
            .relative_to(self.repo_root)
            .as_posix(),
            "generated_manifest_sha256": verified.generated_manifest_sha256,
            "expected_file_count": verified.expected_file_count,
            "verified_file_count": verified.verified_file_count,
            "ok": verified.ok,
        }
        self.report["bank_verification"] = evidence
        return evidence

    def p05(self) -> object:
        config_root = self.project_root / "configs"
        loaded: dict[str, str] = {}
        for profile_name in PROFILE_ORDER:
            profile = load_profile(
                config_root / _PROFILE_FILES[profile_name],
                config_root / "smoke.yaml",
            )
            if profile.profile.value != profile_name or profile.training.episodes != 3:
                raise ValueError(f"strict smoke profile mismatch: {profile_name}")
            self.profiles[profile_name] = profile
            loaded[profile_name] = profile_sha256(profile)
        return {"profile_sha256": loaded, "smoke_episodes": 3}

    def _episode(
        self,
        result: EpisodeResult,
        *,
        profile: ReproductionProfile,
        contract: RunContract,
        method: str,
        phase: str,
        episode_index: int,
        rewards: Sequence[object],
        losses: Sequence[object],
        observations: Sequence[Sequence[object]],
        target_values: Sequence[object] = (),
        parameters: Sequence[object] = (),
        q_values: Sequence[object] = (),
        checkpoint_sha: str | None = None,
        evaluation_epsilon: float | None = None,
        decisions: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        instance = result_instance(result, self)
        return episode_record(
            result,
            profile_sha256=profile_sha256(profile),
            contract_sha256=run_contract_sha256(contract),
            rewards=rewards,
            losses=losses,
            checkpoint_sha256=checkpoint_sha,
            evaluation_epsilon=evaluation_epsilon,
            observations=observations,
            target_values=target_values,
            contract=contract,
            decision_trace=decisions,
            total_operations=_total_operations(instance),
            network_parameters=parameters,
            q_table_values=q_values,
            method=method,
            phase=phase,
            episode_index=episode_index,
            stream_seeds={
                "agent_train_seed": contract.train_seed,
                "policy_seed": 62000 + episode_index,
                "failure_seed": instance.failure_seed,
                "wear_seed": instance.failure_seed + 10_000_000,
                "repair_seed": instance.failure_seed + 20_000_000,
                "replay_seed": contract.train_seed + 1,
            },
        )

    def _record_generated(self, path: Path) -> None:
        relative = validate_generated_artifacts(self.repo_root, (path,))[0]
        generated = cast(list[str], self.report["generated_artifacts"])
        if relative not in generated:
            generated.append(relative)

    def p06(self) -> object:
        instance = self.train_instances[0]
        count = 0
        for profile_name in PROFILE_ORDER[:2]:
            profile = self.profiles[profile_name]
            for action in range(9):
                contract = self.contract(profile, f"non_learning_rule_{action}", 61000)
                env = _new_environment(instance, profile, 62000)
                state, _ = env.reset()
                observations: list[list[float]] = [state.tolist()]
                rewards: list[float] = []
                decisions: list[dict[str, object]] = []
                while not env.is_done():
                    step = env.step_rule(action, RewardMode.TARDINESS)
                    rewards.append(step.reward)
                    observations.append(step.observation.tolist())
                    decisions.append(
                        {
                            "action": action,
                            "reward_mode": int(RewardMode.TARDINESS),
                            "rule_name": step.decision.rule_name,
                            "job_id": step.decision.job_id,
                            "op_id": step.decision.op_id,
                            "machine_id": step.decision.machine_id,
                        }
                    )
                record = self._episode(
                    env.final_result(),
                    profile=profile,
                    contract=contract,
                    method=f"rule_{action}",
                    phase="non_learning_composite",
                    episode_index=0,
                    rewards=rewards,
                    losses=(),
                    observations=observations,
                    evaluation_epsilon=0.0,
                    decisions=decisions,
                )
                self.append_report("episodes", record)
                count += 1
        if count != 18:
            raise AssertionError("composite-rule preflight must produce 18 episodes")
        return {"completed": count}

    def p07(self) -> object:
        instance = self.train_instances[0]
        profile = self.profiles["paper_repro"]
        completed: list[str] = []
        for rule in ClassicalRule:
            contract = self.contract(profile, f"classical_{rule.name.lower()}", 61100)
            env = _new_environment(instance, profile, 62000)
            state, _ = env.reset()
            observations: list[list[float]] = [state.tolist()]
            rewards: list[float] = []
            decisions: list[dict[str, object]] = []
            while not env.is_done():
                step = env._step_classical(rule, RewardMode.TARDINESS)
                rewards.append(step.reward)
                observations.append(step.observation.tolist())
                decisions.append(
                    {
                        "action": rule.value,
                        "reward_mode": int(RewardMode.TARDINESS),
                        "rule_name": step.decision.rule_name,
                        "job_id": step.decision.job_id,
                        "op_id": step.decision.op_id,
                        "machine_id": step.decision.machine_id,
                    }
                )
            record = self._episode(
                env.final_result(),
                profile=profile,
                contract=contract,
                method=rule.value,
                phase="non_learning_classical",
                episode_index=0,
                rewards=rewards,
                losses=(),
                observations=observations,
                evaluation_epsilon=0.0,
                decisions=decisions,
            )
            self.append_report("episodes", record)
            completed.append(rule.value)
        if len(completed) != 5:
            raise AssertionError("classical preflight must produce five episodes")
        return {"completed": completed}

    def _deep_cell(self, profile_name: str, method: str, seed: int) -> object:
        profile = self.profiles[profile_name]
        double_dqn = method == "ddqn"
        agent = DualLayerValueAgent(
            profile,
            seed=seed,
            device=torch.device(self.requested_device),
            double_dqn=double_dqn,
        )
        agent.replay = ReplayBuffer(profile.training.replay_capacity, seed + 1)
        contract = self.contract(profile, method, seed)
        records: list[dict[str, object]] = []
        final_state = np.zeros(6, dtype=np.float32)
        for episode_index, instance in enumerate(self.train_instances):
            env = _new_environment(instance, profile, 62000 + episode_index)
            state, _ = env.reset()
            observations: list[list[float]] = [state.tolist()]
            rewards: list[float] = []
            losses: list[float] = []
            decisions: list[dict[str, object]] = []
            while not env.is_done():
                decision = agent.decide(state, training=True)
                step = env.step_rule(decision.rule_action, decision.reward_mode)
                agent.remember(
                    Transition(
                        state=state,
                        rule_action=decision.rule_action,
                        reward=step.reward,
                        next_state=step.observation,
                        reward_id=int(decision.reward_mode),
                        done=step.done,
                    )
                )
                update = agent.update()
                if update is not None:
                    losses.extend((update.upper_loss, update.lower_loss))
                decisions.append(
                    {
                        "action": decision.rule_action,
                        "reward_mode": int(decision.reward_mode),
                        "epsilon": decision.epsilon,
                        "exploratory": decision.exploratory,
                        "rule_name": step.decision.rule_name,
                        "job_id": step.decision.job_id,
                        "op_id": step.decision.op_id,
                        "machine_id": step.decision.machine_id,
                    }
                )
                rewards.append(step.reward)
                state = step.observation
                observations.append(state.tolist())
            final_state = state.copy()
            records.append(
                self._episode(
                    env.final_result(),
                    profile=profile,
                    contract=contract,
                    method=method,
                    phase="training_smoke",
                    episode_index=episode_index,
                    rewards=rewards,
                    losses=losses,
                    observations=observations,
                    target_values=_target_values(agent, state),
                    parameters=_network_values(agent),
                    decisions=decisions,
                )
            )
        checkpoint = self.output.parent / "checkpoints" / f"{profile_name}_{method}.pt"
        checkpoint_sha = save_checkpoint(checkpoint, agent, contract)
        self._record_generated(checkpoint)
        for record in records:
            record["checkpoint_sha256"] = checkpoint_sha
            self.append_report("episodes", record)
        self.deep_runs[(profile_name, method)] = (
            agent,
            contract,
            checkpoint,
            checkpoint_sha,
            final_state,
        )
        return {
            "profile": profile_name,
            "method": method,
            "agent_seed": seed,
            "replay_seed": seed + 1,
            "episodes": 3,
            "checkpoint_sha256": checkpoint_sha,
        }

    def p08(self) -> object:
        results = []
        for profile_index, profile_name in enumerate(PROFILE_ORDER):
            results.append(
                self._deep_cell(profile_name, "ddqn", 61000 + profile_index * 100)
            )
        return results

    def p09(self) -> object:
        results = []
        for profile_index, profile_name in enumerate(PROFILE_ORDER):
            results.append(
                self._deep_cell(
                    profile_name,
                    "vanilla_dqn_target",
                    61000 + profile_index * 100 + 1,
                )
            )
        return results

    def _tabular_cell(self, method: str, seed: int) -> object:
        algorithm = TabularAlgorithm(method)
        profile = self.profiles["paper_repro"]
        agent = TabularAgent(algorithm, seed=seed)
        contract = self.contract(profile, method, seed)
        for episode_index, instance in enumerate(self.train_instances):
            env = _new_environment(instance, profile, 62000 + episode_index)
            state, previous_named = env.reset()
            action: int | None = None
            observations: list[list[float]] = [state.tolist()]
            rewards: list[float] = []
            q_updates: list[float] = []
            decisions: list[dict[str, object]] = []
            while not env.is_done():
                if action is None:
                    action = agent.select_action(state, training=True)
                step = env.step_rule(action, RewardMode.TARDINESS)
                joint_reward = legacy_joint_reward(
                    previous_named, step.named_observation
                )
                next_action = (
                    agent.select_action(step.observation, training=True)
                    if algorithm is TabularAlgorithm.SARSA and not step.done
                    else None
                )
                q_updates.append(
                    agent.update(
                        state,
                        action,
                        joint_reward,
                        step.observation,
                        step.done,
                        next_action=next_action,
                    )
                )
                decisions.append(
                    {
                        "action": action,
                        "reward_protocol": "legacy_joint",
                        "epsilon": agent.epsilon,
                        "rule_name": step.decision.rule_name,
                        "job_id": step.decision.job_id,
                        "op_id": step.decision.op_id,
                        "machine_id": step.decision.machine_id,
                    }
                )
                rewards.append(joint_reward)
                state = step.observation
                previous_named = step.named_observation
                observations.append(state.tolist())
                action = next_action
            record = self._episode(
                env.final_result(),
                profile=profile,
                contract=contract,
                method=method,
                phase="training_smoke",
                episode_index=episode_index,
                rewards=rewards,
                losses=q_updates,
                observations=observations,
                q_values=agent.q_table.reshape(-1).tolist(),
                decisions=decisions,
            )
            self.append_report("episodes", record)
            agent.decay_epsilon()
        state_dict = agent.state_dict()
        restored = TabularAgent(algorithm, seed=0)
        restored.load_state_dict(state_dict)
        if not np.array_equal(restored.q_table, agent.q_table):
            raise ValueError("tabular state round trip mismatch")
        return {
            "method": method,
            "agent_seed": seed,
            "episodes": 3,
            "reward_protocol": "legacy_joint",
            "state_round_trip": True,
        }

    def p10(self) -> object:
        return [
            self._tabular_cell("q_learning", 61102),
            self._tabular_cell("sarsa", 61103),
        ]

    def p11(self) -> object:
        evidence: list[dict[str, object]] = []
        for profile_name in PROFILE_ORDER:
            for method in METHOD_ORDER[:2]:
                original, contract, checkpoint, checkpoint_sha, final_state = self.deep_runs[
                    (profile_name, method)
                ]
                require_checkpoint(checkpoint)
                resumed = DualLayerValueAgent(
                    original.profile,
                    seed=original.seed,
                    device=original.device,
                    double_dqn=original.double_dqn,
                )
                loaded_sha = load_checkpoint(
                    checkpoint, resumed, contract, for_training=True
                )
                if loaded_sha != checkpoint_sha:
                    raise ValueError("training checkpoint SHA mismatch")
                if resumed.epsilon != original.epsilon or len(resumed.replay) != len(
                    original.replay
                ):
                    raise ValueError("training-resume state mismatch")
                sample_a = original.replay.sample(original.profile.training.batch_size)
                sample_b = resumed.replay.sample(resumed.profile.training.batch_size)
                replay_comparison = compare_replay_samples(sample_a, sample_b)
                decision_a = original.decide(final_state, training=True)
                decision_b = resumed.decide(final_state, training=True)
                if decision_a != decision_b:
                    raise ValueError("training-resume next decision mismatch")
                transition = Transition(
                    state=final_state,
                    rule_action=decision_a.rule_action,
                    reward=0.0,
                    next_state=final_state,
                    reward_id=int(decision_a.reward_mode),
                    done=False,
                )
                original.remember(transition)
                resumed.remember(transition)
                update_a = original.update()
                update_b = resumed.update()
                if update_a != update_b:
                    raise ValueError("training-resume next update mismatch")
                for original_network, resumed_network in (
                    (original.upper_online, resumed.upper_online),
                    (original.upper_target, resumed.upper_target),
                    (original.lower_online, resumed.lower_online),
                    (original.lower_target, resumed.lower_target),
                ):
                    for original_parameter, resumed_parameter in zip(
                        original_network.parameters(),
                        resumed_network.parameters(),
                        strict=True,
                    ):
                        if not torch.equal(original_parameter, resumed_parameter):
                            raise ValueError(
                                "training-resume next-update tensors differ"
                            )
                evaluation = DualLayerValueAgent(
                    original.profile,
                    seed=original.seed,
                    device=original.device,
                    double_dqn=original.double_dqn,
                )
                eval_sha = load_checkpoint(
                    checkpoint, evaluation, contract, for_training=False
                )
                validate_evaluation_epsilon(evaluation.epsilon)
                if eval_sha != checkpoint_sha or len(evaluation.replay) != 0:
                    raise ValueError("evaluation-load checkpoint state mismatch")
                item = {
                    "profile": profile_name,
                    "method": method,
                    "checkpoint_sha256": checkpoint_sha,
                    "training_load_sha256": loaded_sha,
                    "evaluation_load_sha256": eval_sha,
                    "next_decision_equal": True,
                    "next_replay_sample_equal": True,
                    "next_replay_sample_sha256": replay_comparison[
                        "sample_sha256"
                    ],
                    "next_update_equal": True,
                    "next_update_tensors_equal": True,
                    "evaluation_epsilon": evaluation.epsilon,
                }
                evidence.append(item)
                self.append_report("checkpoint_roundtrips", item)
        if len(evidence) != 6:
            raise AssertionError("checkpoint round-trip count must be six")
        return evidence

    def _evaluate(
        self,
        profile_name: str,
        method: str,
        instance: InstanceSpec,
        episode_index: int,
        *,
        append: bool,
    ) -> dict[str, object]:
        trained, contract, checkpoint, checkpoint_sha, _ = self.deep_runs[
            (profile_name, method)
        ]
        require_checkpoint(checkpoint)
        agent = DualLayerValueAgent(
            trained.profile,
            seed=trained.seed,
            device=trained.device,
            double_dqn=trained.double_dqn,
        )
        loaded_sha = load_checkpoint(checkpoint, agent, contract, for_training=False)
        if loaded_sha != checkpoint_sha:
            raise ValueError("evaluation checkpoint SHA mismatch")
        validate_evaluation_epsilon(agent.epsilon)
        env = _new_environment(instance, trained.profile, 62000 + episode_index)
        state, _ = env.reset()
        observations: list[list[float]] = [state.tolist()]
        rewards: list[float] = []
        decisions: list[dict[str, object]] = []
        while not env.is_done():
            decision = agent.decide(state, training=False)
            validate_evaluation_epsilon(decision.epsilon)
            step = env.step_rule(decision.rule_action, decision.reward_mode)
            decisions.append(
                {
                    "action": decision.rule_action,
                    "reward_mode": int(decision.reward_mode),
                    "epsilon": decision.epsilon,
                    "exploratory": decision.exploratory,
                    "rule_name": step.decision.rule_name,
                    "job_id": step.decision.job_id,
                    "op_id": step.decision.op_id,
                    "machine_id": step.decision.machine_id,
                }
            )
            rewards.append(step.reward)
            state = step.observation
            observations.append(state.tolist())
        record = self._episode(
            env.final_result(),
            profile=trained.profile,
            contract=contract,
            method=method,
            phase="evaluation",
            episode_index=episode_index,
            rewards=rewards,
            losses=(),
            observations=observations,
            target_values=_target_values(agent, state),
            parameters=_network_values(agent),
            checkpoint_sha=checkpoint_sha,
            evaluation_epsilon=agent.epsilon,
            decisions=decisions,
        )
        if append:
            self.append_report("episodes", record)
        return record

    def p12(self) -> object:
        count = 0
        for profile_name in PROFILE_ORDER:
            for method in METHOD_ORDER[:2]:
                for episode_index, instance in enumerate(self.evaluation_instances):
                    self._evaluate(
                        profile_name, method, instance, episode_index, append=True
                    )
                    count += 1
        if count != 12:
            raise AssertionError("deep evaluation episode count must be 12")
        return {"evaluation_episodes": count, "epsilon": 0.0}

    def p13(self) -> object:
        episodes = cast(list[dict[str, object]], self.report["episodes"])
        for record in episodes:
            validator = cast(dict[str, object], record["validator"])
            if validator.get("ok") is not True:
                raise ValueError("an episode has invalid schedule evidence")
            if record["decision_count"] != record["total_operations"]:
                raise ValueError("an episode decision count is invalid")
            contract = cast(dict[str, object] | None, record["contract"])
            if contract is None or (
                contract.get("git_commit") != self.git_commit
                or contract.get("profile_sha256") != record["profile_sha256"]
                or contract.get("bank_manifest_sha256")
                != EXPECTED_BANK_MANIFEST_SHA256
            ):
                raise ValueError("an episode run contract has invalid provenance")
            metrics = cast(dict[str, object], record["metrics"])
            _finite_sequence(list(metrics.values()), "reported metric fields")
            if record["evaluation_epsilon"] is not None:
                validate_evaluation_epsilon(record["evaluation_epsilon"])
        return {"validated_episode_count": len(episodes)}

    def p14(self) -> object:
        checks: list[dict[str, object]] = []
        for profile_name in PROFILE_ORDER:
            for method in METHOD_ORDER[:2]:
                for episode_index, instance in enumerate(self.evaluation_instances):
                    first = self._evaluate(
                        profile_name, method, instance, episode_index, append=False
                    )
                    second = self._evaluate(
                        profile_name, method, instance, episode_index, append=False
                    )
                    comparison = compare_deterministic_episode_payloads(first, second)
                    item = {
                        "profile": profile_name,
                        "method": method,
                        "instance_id": instance.instance_id,
                        **comparison,
                    }
                    checks.append(item)
                    self.append_report("determinism_checks", item)
        if len(checks) != 12:
            raise AssertionError("determinism check count must be 12")
        return checks

    def p15(self) -> object:
        generated = cast(list[str], self.report["generated_artifacts"])
        generated_paths = tuple(self.repo_root / path for path in generated)
        validate_generated_artifacts(self.repo_root, generated_paths)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise ValueError(f"repository became dirty during preflight: {status}")
        legacy_diff = subprocess.run(
            ["git", "diff", "--exit-code", "--", "code", "code1", "code2"],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
        )
        if legacy_diff.returncode != 0:
            raise ValueError("legacy directories differ from HEAD")
        tracked_output = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.repo_root,
            check=True,
            capture_output=True,
        ).stdout
        tracked_paths = tuple(
            os.fsdecode(raw).replace("\\", "/")
            for raw in tracked_output.split(b"\0")
            if raw
        )
        generated_roots = (
            "扩刊/original_repro/artifacts/banks/materialized/",
            "扩刊/original_repro/artifacts/preflight/",
            "扩刊/original_repro/artifacts/runs/",
            "扩刊/original_repro/artifacts/summaries/",
            "扩刊/original_repro/artifacts/figures/",
        )
        tracked_generated = sorted(
            path
            for path in tracked_paths
            if path.startswith(generated_roots)
            or path.casefold().endswith((".pt", ".pth", ".ckpt"))
            or (
                path.startswith("扩刊/original_repro/artifacts/banks/")
                and path.casefold().endswith(".json.gz")
            )
        )
        if tracked_generated:
            raise ValueError(f"generated artifacts are tracked: {tracked_generated}")
        current_audit = audit_legacy_outputs(
            self.repo_root,
            self.output.parent / "legacy_tracked_p15.json",
            AuditScope.TRACKED,
        )
        expected = json.loads(
            (self.repo_root / "扩刊/docs/audit/legacy_tracked_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if current_audit != expected:
            raise ValueError("legacy manifest changed during preflight")
        self._record_generated(self.output.parent / "legacy_tracked_p15.json")
        return {
            "repository_clean": True,
            "legacy_diff": "zero",
            "tracked_generated_artifacts": [],
            "generated_artifact_count": len(generated),
        }


def result_instance(result: EpisodeResult, runner: _PreflightRunner) -> InstanceSpec:
    for instance in (*runner.train_instances, *runner.evaluation_instances):
        if instance.instance_id == result.instance_id:
            return instance
    raise ValueError(f"episode references unknown instance: {result.instance_id}")


def run_preflight(
    *,
    repo_root: Path,
    bank_root: Path,
    reference_manifest: Path,
    environment_metadata: Path,
    output: Path,
    device: str,
) -> dict[str, object]:
    paths = _validate_preflight_paths(
        repo_root=repo_root,
        bank_root=bank_root,
        reference_manifest=reference_manifest,
        environment_metadata=environment_metadata,
        output=output,
    )
    runner = _PreflightRunner(
        repo_root=paths.repo_root,
        bank_root=paths.bank_root,
        reference_manifest=paths.reference_manifest,
        environment_metadata=paths.environment_metadata,
        output=paths.output,
        device=device,
    )
    gates: tuple[tuple[str, Callable[[], object]], ...] = (
        ("P00", runner.p00),
        ("P01", runner.p01),
        ("P02", runner.p02),
        ("P03", runner.p03),
        ("P04", runner.p04),
        ("P05", runner.p05),
        ("P06", runner.p06),
        ("P07", runner.p07),
        ("P08", runner.p08),
        ("P09", runner.p09),
        ("P10", runner.p10),
        ("P11", runner.p11),
        ("P12", runner.p12),
        ("P13", runner.p13),
        ("P14", runner.p14),
        ("P15", runner.p15),
    )
    return run_gate_sequence(paths.output, runner.report, gates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--environment-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run_preflight(
        repo_root=args.repo_root,
        bank_root=args.bank_root,
        reference_manifest=args.reference_manifest,
        environment_metadata=args.environment_metadata,
        output=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
