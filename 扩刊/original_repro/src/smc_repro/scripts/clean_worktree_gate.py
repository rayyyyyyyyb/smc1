from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path


class CleanWorktreeGateError(RuntimeError):
    """Raised after a failed clean-worktree report has been persisted."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    without_hash = dict(report)
    without_hash.pop("report_sha256", None)
    canonical = (
        json.dumps(
            without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    report["report_sha256"] = _sha256(canonical)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _command_record(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> tuple[dict[str, object], bytes, bytes]:
    started_at = _utc_now()
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except OSError as exc:
        exit_code = -1
        stdout = b""
        stderr = str(exc).encode("utf-8", errors="replace")
    duration = time.perf_counter() - start
    record: dict[str, object] = {
        "command": [str(part) for part in command],
        "cwd": str(cwd),
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "duration_seconds": duration,
        "exit_code": exit_code,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
    }
    return record, stdout, stderr


def _resolve_repository(repo_root: Path) -> tuple[Path, dict[str, object]]:
    record, stdout, stderr = _command_record(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=repo_root,
    )
    if record["exit_code"] != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git rev-parse failed: {message}")
    try:
        resolved = Path(os.fsdecode(stdout.strip())).resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError("git rev-parse returned an invalid repository root") from exc
    return resolved, record


def _tracked_manifest(
    worktree: Path,
    legacy_directories: tuple[str, ...],
) -> dict[str, object]:
    listed = subprocess.run(
        ["git", "-C", str(worktree), "ls-files", "-z", "--", *legacy_directories],
        check=True,
        capture_output=True,
    )
    files: dict[str, dict[str, object]] = {}
    for raw_path in sorted(item for item in listed.stdout.split(b"\0") if item):
        relative = os.fsdecode(raw_path).replace("\\", "/")
        blob = subprocess.run(
            ["git", "-C", str(worktree), "cat-file", "blob", f":{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        files[relative] = {
            "size_bytes": len(blob),
            "sha256": _sha256(blob),
        }
    return {
        "schema_version": 2,
        "scope": "tracked",
        "legacy_directories": list(legacy_directories),
        "files": files,
    }


def _default_commands(
    python_executable: Path,
    worktree: Path,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    audit_output = worktree / ".preflight_legacy_tracked.json"
    return (
        ("pytest", ("-m", "pytest", "-q")),
        ("ruff", ("-m", "ruff", "check", "src", "tests")),
        ("mypy", ("-m", "mypy", "src/smc_repro")),
        ("compileall", ("-m", "compileall", "-q", "src", "tests")),
        (
            "legacy_audit",
            (
                "-m",
                "smc_repro.scripts.audit_legacy_outputs",
                "--repo-root",
                str(worktree),
                "--output",
                str(audit_output),
                "--scope",
                "tracked",
            ),
        ),
    )


def run_detached_worktree_gate(
    repo_root: Path,
    python_executable: Path,
    report_path: Path,
    *,
    commands: Sequence[tuple[str, Sequence[str]]] | None = None,
    project_relative: Path = Path("扩刊/original_repro"),
    tracked_manifest_relative: Path = Path("扩刊/docs/audit/legacy_tracked_manifest.json"),
    legacy_directories: tuple[str, ...] = ("code", "code1", "code2"),
) -> dict[str, object]:
    """Run quality gates in a detached worktree and always persist cleanup evidence."""
    if not isinstance(repo_root, Path) or not repo_root.is_dir():
        raise ValueError("repo_root must be an existing directory")
    if not isinstance(python_executable, Path) or not python_executable.is_file():
        raise ValueError("python_executable must be an existing file")
    if not isinstance(report_path, Path):
        raise TypeError("report_path must be a Path")
    if not legacy_directories or any(
        not isinstance(item, str) or not item for item in legacy_directories
    ):
        raise ValueError("legacy_directories must contain non-empty names")

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "failed",
        "git_commit": None,
        "source_repository": str(repo_root.resolve()),
        "python_executable": str(python_executable.resolve()),
        "temporary_worktree": "",
        "temporary_worktree_removed": True,
        "setup_commands": [],
        "commands": [],
        "legacy_audit": None,
        "cleanup_commands": [],
        "message": "gate did not complete",
    }
    temporary_parent: Path | None = None
    temporary_worktree: Path | None = None
    worktree_registered = False
    failure: str | None = None
    cleanup_failure: str | None = None

    try:
        resolved_repo, rev_parse_record = _resolve_repository(repo_root.resolve())
        setup_records = report["setup_commands"]
        assert isinstance(setup_records, list)
        setup_records.append(rev_parse_record)

        status_record, stdout, stderr = _command_record(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=resolved_repo,
        )
        setup_records.append(status_record)
        if status_record["exit_code"] != 0:
            raise RuntimeError(
                "git status failed: "
                + stderr.decode("utf-8", errors="replace").strip()
            )
        if stdout.strip():
            raise RuntimeError("dirty source repository is not permitted")

        head_record, head_stdout, head_stderr = _command_record(
            ("git", "rev-parse", "HEAD"), cwd=resolved_repo
        )
        setup_records.append(head_record)
        if head_record["exit_code"] != 0:
            raise RuntimeError(
                "git rev-parse HEAD failed: "
                + head_stderr.decode("utf-8", errors="replace").strip()
            )
        commit = head_stdout.decode("ascii", errors="strict").strip()
        if len(commit) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise RuntimeError("git rev-parse HEAD returned a noncanonical object ID")
        report["git_commit"] = commit

        temporary_parent = Path(
            tempfile.mkdtemp(prefix="smc-clean-worktree-", dir=resolved_repo.parent)
        )
        temporary_worktree = temporary_parent / "detached-扩刊"
        report["temporary_worktree"] = str(temporary_worktree)
        add_record, _, add_stderr = _command_record(
            (
                "git",
                "worktree",
                "add",
                "--detach",
                str(temporary_worktree),
                commit,
            ),
            cwd=resolved_repo,
        )
        setup_records.append(add_record)
        if add_record["exit_code"] != 0:
            raise RuntimeError(
                "git worktree add failed: "
                + add_stderr.decode("utf-8", errors="replace").strip()
            )
        worktree_registered = True

        project_root = (temporary_worktree / project_relative).resolve()
        if not project_root.is_dir() or not project_root.is_relative_to(
            temporary_worktree.resolve()
        ):
            raise RuntimeError(f"worktree project directory is missing: {project_relative}")
        manifest_path = (temporary_worktree / tracked_manifest_relative).resolve()
        if not manifest_path.is_file() or not manifest_path.is_relative_to(
            temporary_worktree.resolve()
        ):
            raise RuntimeError(
                f"tracked legacy manifest is missing: {tracked_manifest_relative}"
            )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(
            (temporary_worktree / "扩刊/original_repro/src").resolve()
            if project_relative == Path("扩刊/original_repro")
            else (project_root / "src").resolve()
        )
        env["PYTHONHASHSEED"] = "0"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        selected_commands = (
            _default_commands(python_executable, temporary_worktree)
            if commands is None
            else tuple((name, tuple(arguments)) for name, arguments in commands)
        )
        command_records = report["commands"]
        assert isinstance(command_records, list)
        for name, arguments in selected_commands:
            command = (str(python_executable.resolve()), *arguments)
            command_record, _, command_stderr = _command_record(
                command,
                cwd=project_root,
                env=env,
            )
            command_record["name"] = name
            command_records.append(command_record)
            if command_record["exit_code"] != 0:
                tail = command_stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"{name} failed with exit {command_record['exit_code']}: {tail}")

        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit_output = temporary_worktree / ".preflight_legacy_tracked.json"
        if commands is None:
            observed = json.loads(audit_output.read_text(encoding="utf-8"))
        else:
            observed = _tracked_manifest(temporary_worktree, legacy_directories)
        if observed != expected:
            raise RuntimeError("tracked legacy audit differs from legacy_tracked_manifest.json")
        audit_evidence = {
            "expected_manifest": tracked_manifest_relative.as_posix(),
            "file_count": len(observed.get("files", {})),
            "evidence_sha256": _sha256(
                (
                    json.dumps(
                        observed,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            ),
            "equal": True,
        }
        report["legacy_audit"] = audit_evidence
        report["status"] = "passed"
        report["message"] = "all detached clean-worktree gates passed"
    except Exception as exc:
        failure = str(exc)
        report["status"] = "failed"
        report["message"] = failure
    finally:
        cleanup_records = report["cleanup_commands"]
        assert isinstance(cleanup_records, list)
        if temporary_worktree is not None and worktree_registered:
            cleanup_record, _, cleanup_stderr = _command_record(
                ("git", "worktree", "remove", "--force", str(temporary_worktree)),
                cwd=repo_root.resolve(),
            )
            cleanup_records.append(cleanup_record)
            if cleanup_record["exit_code"] != 0:
                cleanup_failure = (
                    "git worktree cleanup failed: "
                    + cleanup_stderr.decode("utf-8", errors="replace").strip()
                )
        if temporary_parent is not None and temporary_parent.exists():
            try:
                shutil.rmtree(temporary_parent)
            except OSError as exc:
                cleanup_failure = cleanup_failure or f"temporary cleanup failed: {exc}"
        removed = temporary_worktree is None or not temporary_worktree.exists()
        report["temporary_worktree_removed"] = removed
        if not removed:
            cleanup_failure = cleanup_failure or "temporary worktree still exists after cleanup"
        if cleanup_failure is not None:
            report["status"] = "failed"
            report["message"] = cleanup_failure
        _write_report(report_path.resolve(), report)

    if report["status"] != "passed":
        raise CleanWorktreeGateError(str(report["message"]))
    return report


def run_clean_worktree_gate(
    repo_root: Path,
    python_executable: Path,
    report_path: Path,
) -> dict[str, object]:
    return run_detached_worktree_gate(repo_root, python_executable, report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run_clean_worktree_gate(args.repo_root, args.python_executable, args.report)


if __name__ == "__main__":
    main()
