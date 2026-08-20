from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath, PurePosixPath

from smc_repro.config import ProfileName, ReproductionProfile, profile_sha256

FAILURE_STREAM_VERSION = "smc-crn1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


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

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("schema_version must be the integer 1")
        if not isinstance(self.git_commit, str) or not _GIT_COMMIT_PATTERN.fullmatch(
            self.git_commit
        ):
            raise ValueError("git_commit must be a canonical lowercase Git object ID")
        if not isinstance(self.profile_name, str) or not self.profile_name.strip():
            raise ValueError("profile_name must be a non-empty string")
        try:
            ProfileName(self.profile_name)
        except ValueError as exc:
            raise ValueError("profile_name must identify a locked reproduction profile") from exc
        _validate_sha256("profile_sha256", self.profile_sha256)
        _validate_sha256("bank_manifest_sha256", self.bank_manifest_sha256)
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        _validate_seed("train_seed", self.train_seed)
        _validate_seed("policy_seed", self.policy_seed)
        if self.failure_stream_version != FAILURE_STREAM_VERSION:
            raise ValueError(
                f"failure_stream_version must be {FAILURE_STREAM_VERSION!r}"
            )
        _validate_canonical_path_string(self.environment_metadata_path)


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")


def _validate_seed(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_canonical_path_string(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("environment_metadata_path must not be empty")
    if "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(
            "environment_metadata_path must be a repository-relative POSIX path"
        )
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError(
            "environment_metadata_path must not contain empty, dot, or dotdot components"
        )


def _repository_relative_posix_path(path: PurePath) -> str:
    if not isinstance(path, PurePath):
        raise ValueError("environment_metadata_path must be a path")
    if path.is_absolute() or path.anchor:
        raise ValueError("environment_metadata_path must be repository-relative")
    components = path.parts
    if not components or any(component in {"", ".", ".."} for component in components):
        raise ValueError(
            "environment_metadata_path must not contain empty, dot, or dotdot components"
        )
    if any("\\" in component or "/" in component for component in components):
        raise ValueError("environment_metadata_path must not contain backslashes")
    canonical = PurePosixPath(*components).as_posix()
    _validate_canonical_path_string(canonical)
    return canonical


def collect_git_commit(repo_root: Path, *, allow_dirty: bool = False) -> str:
    if not isinstance(repo_root, Path) or not repo_root.is_dir():
        raise ValueError("repo_root must be an existing directory")
    if type(allow_dirty) is not bool:
        raise ValueError("allow_dirty must be a boolean")
    try:
        commit_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_process = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"unable to inspect Git repository: {exc}") from exc
    commit = commit_process.stdout.strip()
    if not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise RuntimeError("git rev-parse HEAD returned a noncanonical object ID")
    if status_process.stdout.strip() and not allow_dirty:
        raise RuntimeError("formal run requires a clean Git worktree; repository is dirty")
    return commit


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
    if not isinstance(profile, ReproductionProfile):
        raise TypeError("profile must be a ReproductionProfile")
    return RunContract(
        schema_version=1,
        git_commit=collect_git_commit(repo_root, allow_dirty=allow_dirty),
        profile_name=profile.profile.value,
        profile_sha256=profile_sha256(profile),
        bank_manifest_sha256=bank_manifest_sha256,
        method=method,
        train_seed=train_seed,
        policy_seed=policy_seed,
        failure_stream_version=FAILURE_STREAM_VERSION,
        environment_metadata_path=_repository_relative_posix_path(
            environment_metadata_path
        ),
    )


def contract_sha256(contract: RunContract) -> str:
    if type(contract) is not RunContract:
        raise TypeError("contract must be a RunContract")
    payload = json.dumps(
        asdict(contract),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
