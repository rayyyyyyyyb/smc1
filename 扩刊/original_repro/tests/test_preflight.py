from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import smc_repro.scripts.clean_worktree_gate as clean_gate_module
import smc_repro.scripts.preflight as preflight_module
from smc_repro.agents.replay import Transition
from smc_repro.config import load_profile
from smc_repro.scripts.clean_worktree_gate import (
    CleanWorktreeGateError,
    run_detached_worktree_gate,
)
from smc_repro.scripts.preflight import (
    EVALUATION_BANK_PATHS,
    GATE_NAMES,
    TRAIN_BANK_PATHS,
    PreflightError,
    canonical_json_bytes,
    compare_deterministic_episode_payloads,
    episode_record,
    planned_training_cells,
    require_checkpoint,
    run_gate_sequence,
    sha256_bytes,
    validate_evaluation_epsilon,
    validate_fixed_bank_paths,
    validate_generated_artifacts,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tiny_repository(tmp_path: Path, *, passing: bool = True) -> tuple[Path, Path]:
    repo = tmp_path / "仓库-扩刊"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "preflight@example.invalid")
    _git(repo, "config", "user.name", "Preflight Test")
    (repo / ".gitignore").write_text("code/ignored.bin\n", encoding="utf-8")
    code = repo / "code"
    code.mkdir()
    tracked = code / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    (code / "ignored.bin").write_bytes(b"ignored-local-only")
    manifest = {
        "schema_version": 2,
        "scope": "tracked",
        "legacy_directories": ["code"],
        "files": {
            "code/tracked.txt": {
                "size_bytes": len(tracked.read_bytes()),
                "sha256": hashlib.sha256(tracked.read_bytes()).hexdigest(),
            }
        },
    }
    manifest_path = repo / "legacy_tracked_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    test_dir = repo / "tests"
    test_dir.mkdir()
    test_dir.joinpath("test_tiny.py").write_text(
        "from pathlib import Path\n\n"
        "def test_tiny():\n"
        "    assert not Path('code/ignored.bin').exists()\n"
        "    assert "
        + ("True" if passing else "False")
        + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tiny")
    return repo, manifest_path


def test_canonical_json_orders_keys_uses_utf8_and_rejects_nan() -> None:
    assert canonical_json_bytes({"z": 1, "汉": [2, 3]}) == (
        '{"z":1,"汉":[2,3]}\n'.encode()
    )
    assert sha256_bytes(b"payload") == hashlib.sha256(b"payload").hexdigest()
    with pytest.raises(ValueError, match="JSON"):
        canonical_json_bytes({"invalid": float("nan")})


def test_gate_sequence_is_fixed_and_stable(tmp_path: Path) -> None:
    report_a = tmp_path / "a.json"
    report_b = tmp_path / "b.json"
    seen: list[str] = []

    def make_gate(name: str):  # type: ignore[no-untyped-def]
        def gate() -> dict[str, object]:
            seen.append(name)
            return {"gate": name, "value": 7}

        return gate

    definitions = tuple((name, make_gate(name)) for name in GATE_NAMES)
    first = run_gate_sequence(report_a, {"git_commit": "a" * 40}, definitions)
    seen.clear()
    second = run_gate_sequence(report_b, {"git_commit": "a" * 40}, definitions)
    assert seen == list(GATE_NAMES)
    assert [gate["name"] for gate in first["gates"]] == list(GATE_NAMES)
    assert first["scientific_payload_sha256"] == second["scientific_payload_sha256"]
    assert [gate["evidence_sha256"] for gate in first["gates"]] == [
        gate["evidence_sha256"] for gate in second["gates"]
    ]


def test_failed_gate_persists_report_before_raise_and_stops(tmp_path: Path) -> None:
    report_path = tmp_path / "failed.json"
    seen: list[str] = []

    def passes() -> dict[str, bool]:
        seen.append("P00")
        return {"ok": True}

    def fails() -> dict[str, bool]:
        seen.append("P01")
        raise RuntimeError("deliberate failure")

    def must_not_run() -> dict[str, bool]:
        seen.append("P02")
        return {"ok": True}

    with pytest.raises(PreflightError, match="P01"):
        run_gate_sequence(
            report_path,
            {"git_commit": "b" * 40},
            (("P00", passes), ("P01", fails), ("P02", must_not_run)),
        )
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert [gate["name"] for gate in persisted["gates"]] == ["P00", "P01"]
    assert persisted["gates"][-1]["status"] == "failed"
    assert "deliberate failure" in persisted["gates"][-1]["message"]
    assert seen == ["P00", "P01"]


def test_profile_method_seed_plan_is_exact() -> None:
    assert planned_training_cells() == (
        ("legacy_snapshot", "ddqn", 61000),
        ("paper_repro", "ddqn", 61100),
        ("corrected_smc", "ddqn", 61200),
        ("legacy_snapshot", "vanilla_dqn_target", 61001),
        ("paper_repro", "vanilla_dqn_target", 61101),
        ("corrected_smc", "vanilla_dqn_target", 61201),
        ("paper_repro", "q_learning", 61102),
        ("paper_repro", "sarsa", 61103),
    )


def test_fixed_bank_paths_are_exact_and_contained(tmp_path: Path) -> None:
    bank_root = tmp_path / "bank"
    bank_root.mkdir()
    all_paths = TRAIN_BANK_PATHS + EVALUATION_BANK_PATHS
    for relative in all_paths:
        path = bank_root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    resolved = validate_fixed_bank_paths(bank_root, all_paths)
    assert tuple(path.relative_to(bank_root).as_posix() for path in resolved) == all_paths
    with pytest.raises(ValueError, match="escapes|canonical"):
        validate_fixed_bank_paths(bank_root, ("../escape.json.gz",), require_files=False)


def test_invalid_schedule_is_rejected_before_metrics_are_read() -> None:
    class ExplodingMetrics:
        @property
        def makespan(self) -> float:
            raise AssertionError("metrics must not be read")

    result = SimpleNamespace(
        instance_id="tiny",
        profile_name="paper_repro",
        validation=SimpleNamespace(ok=False, errors=("bad schedule",)),
        metrics=ExplodingMetrics(),
        intervals=(),
        decisions=1,
    )
    with pytest.raises(ValueError, match="schedule validation failed"):
        episode_record(
            result,
            profile_sha256="c" * 64,
            contract_sha256="d" * 64,
            rewards=(),
            losses=(),
            checkpoint_sha256=None,
            evaluation_epsilon=None,
            observations=(),
            target_values=(),
        )


def test_evaluation_epsilon_must_be_exactly_zero() -> None:
    validate_evaluation_epsilon(0.0)
    with pytest.raises(ValueError, match="exactly 0.0"):
        validate_evaluation_epsilon(1e-12)


def test_deterministic_episode_comparison_is_byte_exact() -> None:
    first = {"profile": "paper_repro", "rewards": [1, -1], "epsilon": 0.0}
    second = {"epsilon": 0.0, "rewards": [1, -1], "profile": "paper_repro"}
    evidence = compare_deterministic_episode_payloads(first, second)
    assert evidence["identical"] is True
    assert evidence["first_sha256"] == evidence["second_sha256"]
    with pytest.raises(ValueError, match="byte-for-byte"):
        compare_deterministic_episode_payloads(first, {**second, "epsilon": 0.1})


def test_generated_artifacts_must_stay_under_ignored_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    allowed = repo / "扩刊/original_repro/artifacts/preflight/checkpoints/model.pt"
    validate_generated_artifacts(repo, (allowed,), require_git_ignored=False)
    with pytest.raises(ValueError, match="outside allowed ignored roots"):
        validate_generated_artifacts(
            repo,
            (repo / "扩刊/original_repro/results/formal.json",),
            require_git_ignored=False,
        )


def test_missing_checkpoint_is_a_hard_failure(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        require_checkpoint(tmp_path / "missing.pt")


def test_clean_worktree_uses_only_tracked_files_and_supports_non_ascii(
    tmp_path: Path,
) -> None:
    repo, manifest = _tiny_repository(tmp_path)
    report_path = tmp_path / "clean-report.json"
    ignored = repo / "code/ignored.bin"
    report = run_detached_worktree_gate(
        repo,
        Path(sys.executable),
        report_path,
        commands=(("pytest", ("-m", "pytest", "-q")),),
        project_relative=Path("."),
        tracked_manifest_relative=manifest.relative_to(repo),
        legacy_directories=("code",),
    )
    assert report["status"] == "passed"
    assert ignored.is_file()
    assert report["temporary_worktree_removed"] is True
    assert report["commands"][0]["exit_code"] == 0
    assert set(report["commands"][0]) >= {
        "command",
        "exit_code",
        "duration_seconds",
        "stdout_sha256",
        "stderr_sha256",
    }
    assert not Path(report["temporary_worktree"]).exists()


def test_clean_worktree_rejects_dirty_repository_and_persists_failure(
    tmp_path: Path,
) -> None:
    repo, manifest = _tiny_repository(tmp_path)
    (repo / "dirty.txt").write_text("dirty", encoding="utf-8")
    report_path = tmp_path / "dirty-report.json"
    with pytest.raises(CleanWorktreeGateError, match="dirty"):
        run_detached_worktree_gate(
            repo,
            Path(sys.executable),
            report_path,
            commands=(),
            project_relative=Path("."),
            tracked_manifest_relative=manifest.relative_to(repo),
            legacy_directories=("code",),
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["temporary_worktree_removed"] is True


def test_clean_worktree_failure_is_reported_and_always_cleaned_up(
    tmp_path: Path,
) -> None:
    repo, manifest = _tiny_repository(tmp_path, passing=False)
    report_path = tmp_path / "failed-clean-report.json"
    with pytest.raises(CleanWorktreeGateError, match="pytest"):
        run_detached_worktree_gate(
            repo,
            Path(sys.executable),
            report_path,
            commands=(("pytest", ("-m", "pytest", "-q")),),
            project_relative=Path("."),
            tracked_manifest_relative=manifest.relative_to(repo),
            legacy_directories=("code",),
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["commands"][0]["exit_code"] != 0
    assert report["temporary_worktree_removed"] is True
    assert not Path(report["temporary_worktree"]).exists()


def _preflight_layout(repo: Path) -> dict[str, Path | str]:
    project = repo / "扩刊/original_repro"
    bank = project / "artifacts/banks/materialized"
    reference = project / "artifacts/banks/release/manifest.json"
    environment = repo / "扩刊/docs/audit/environment_5090_resolved.json"
    output = project / "artifacts/preflight/preflight_report.json"
    bank.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    reference.write_text("{}\n", encoding="utf-8")
    environment.parent.mkdir(parents=True)
    environment.write_text("{}\n", encoding="utf-8")
    output.parent.mkdir(parents=True)
    return {
        "repo_root": repo,
        "bank_root": bank,
        "reference_manifest": reference,
        "environment_metadata": environment,
        "output": output,
        "device": "cuda:0",
    }


def test_invalid_preflight_output_never_overwrites_tracked_or_protected_files(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    arguments = _preflight_layout(repo)
    victims = [
        repo / "code/tracked.txt",
        repo / "code1/tracked.txt",
        repo / "code2/tracked.txt",
        tmp_path / "outside/report.json",
    ]
    for victim in victims:
        victim.parent.mkdir(parents=True, exist_ok=True)
        victim.write_bytes(b"ORIGINAL\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "preflight@example.invalid")
    _git(repo, "config", "user.name", "Preflight Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tracked victims")
    for victim in victims:
        with pytest.raises(
            preflight_module.PreflightInputError, match="no report was written"
        ):
            preflight_module.run_preflight(
                **{**arguments, "output": victim}  # type: ignore[arg-type]
            )
        assert victim.read_bytes() == b"ORIGINAL\n"
        assert _git(repo, "status", "--short") == ""


def test_preflight_output_symlink_cannot_escape_controlled_root(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    arguments = _preflight_layout(repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = cast(Path, arguments["output"]).parent / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    escaped_report = link / "report.json"
    with pytest.raises(
        preflight_module.PreflightInputError, match="no report was written"
    ):
        preflight_module.run_preflight(
            **{**arguments, "output": escaped_report}  # type: ignore[arg-type]
        )
    assert not (outside / "report.json").exists()


def test_preflight_root_symlink_to_protected_repo_path_cannot_overwrite_victim(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    arguments = _preflight_layout(repo)
    victim = repo / "code/victim.json"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"TRACKED ORIGINAL\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "preflight@example.invalid")
    _git(repo, "config", "user.name", "Preflight Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tracked protected victim")

    preflight_root = cast(Path, arguments["output"]).parent
    preflight_root.rmdir()
    try:
        preflight_root.symlink_to(victim.parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    arguments["output"] = preflight_root / victim.name
    status_before = _git(repo, "status", "--short")

    caught: Exception | None = None
    try:
        preflight_module.run_preflight(**arguments)  # type: ignore[arg-type]
    except Exception as exc:
        caught = exc

    assert victim.read_bytes() == b"TRACKED ORIGINAL\n"
    assert isinstance(caught, preflight_module.PreflightInputError)
    assert "no report was written" in str(caught)
    assert _git(repo, "status", "--short") == status_before


def test_preflight_runner_contract_records_repo_relative_environment_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    arguments = _preflight_layout(repo)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "preflight@example.invalid")
    _git(repo, "config", "user.name", "Preflight Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "runner contract fixture")
    environment = cast(Path, arguments["environment_metadata"])
    assert environment.is_absolute()
    profile_path = Path(__file__).resolve().parents[1] / "configs/paper_repro.yaml"

    runner = preflight_module._PreflightRunner(  # type: ignore[attr-defined]
        **arguments  # type: ignore[arg-type]
    )
    contract = runner.contract(load_profile(profile_path), "ddqn", 61000)

    assert (
        contract.environment_metadata_path
        == "扩刊/docs/audit/environment_5090_resolved.json"
    )


def _environment_snapshot() -> tuple[dict[str, object], object]:
    snapshot_path = Path(__file__).resolve().parents[2] / (
        "docs/audit/environment_5090_resolved.json"
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    facts = preflight_module.LiveEnvironmentFacts(
        torch_version=str(snapshot["torch_version"]),
        torch_cuda_runtime=str(snapshot["torch_cuda_runtime"]),
        gpu_name=str(snapshot["gpu_name"]),
        compute_capability=tuple(snapshot["compute_capability"]),
        compiled_cuda_arches=tuple(snapshot["compiled_cuda_arches"]),
        cuda_smoke_result=float(snapshot["cuda_smoke_result"]),
        python_version=str(snapshot["python_version"]),
        python_implementation=str(snapshot["python_implementation"]),
        platform=str(snapshot["platform"]),
    )
    return snapshot, facts


def test_environment_snapshot_accepts_old_canonical_capture_provenance() -> None:
    snapshot, facts = _environment_snapshot()
    snapshot["git_commit"] = "1" * 40
    evidence = preflight_module.validate_environment_snapshot(snapshot, facts)
    assert evidence["capture_git_commit"] == "1" * 40
    assert evidence["capture_git_commit"] != "dd9bb0354e9302b9a2ba7a630168ca393f713348"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 999),
        ("git_commit", None),
        ("git_commit", "A" * 40),
        ("cuda_available", False),
        ("torch_version", "0.0.invalid"),
        ("gpu_name", "not-an-rtx-5090"),
    ),
)
def test_environment_snapshot_strictly_rejects_invalid_schema_and_device_facts(
    field: str,
    value: object,
) -> None:
    snapshot, facts = _environment_snapshot()
    snapshot[field] = value
    with pytest.raises(ValueError, match="environment metadata"):
        preflight_module.validate_environment_snapshot(snapshot, facts)


def _transition(**changes: object) -> Transition:
    values: dict[str, object] = {
        "state": np.zeros(6, dtype=np.float32),
        "rule_action": 0,
        "reward": 0.0,
        "next_state": np.zeros(6, dtype=np.float32),
        "reward_id": 0,
        "done": False,
    }
    values.update(changes)
    return Transition(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changed",
    (
        {"rule_action": 1},
        {"reward": 1.0},
        {"next_state": np.ones(6, dtype=np.float32)},
        {"reward_id": 1},
        {"done": True},
    ),
)
def test_replay_sample_comparison_checks_every_transition_field(
    changed: dict[str, object],
) -> None:
    original = _transition()
    different = _transition(**changed)
    assert np.array_equal(original.state, different.state)
    with pytest.raises(ValueError, match="complete Transition"):
        preflight_module.compare_replay_samples((original,), (different,))
    evidence = preflight_module.compare_replay_samples((original,), (original,))
    assert evidence["equal"] is True


def test_clean_worktree_default_commands_lock_interpreter_environment_and_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, manifest = _tiny_repository(tmp_path)
    report_path = tmp_path / "default-report.json"
    original_runner = clean_gate_module._command_record
    intercepted: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    def intercept(
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> tuple[dict[str, object], bytes, bytes]:
        if command[0] != str(Path(sys.executable).resolve()):
            return original_runner(command, cwd=cwd, env=env)
        assert env is not None
        intercepted.append((command, cwd, env))
        if "smc_repro.scripts.audit_legacy_outputs" in command:
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                json.dumps(
                    clean_gate_module._tracked_manifest(
                        Path(command[command.index("--repo-root") + 1]),
                        ("code",),
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return (
            {
                "command": list(command),
                "cwd": str(cwd),
                "started_at_utc": "fixture",
                "ended_at_utc": "fixture",
                "duration_seconds": 0.0,
                "exit_code": 0,
                "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            },
            b"",
            b"",
        )

    monkeypatch.setattr(clean_gate_module, "_command_record", intercept)
    report = run_detached_worktree_gate(
        repo,
        Path(sys.executable),
        report_path,
        project_relative=Path("."),
        tracked_manifest_relative=manifest.relative_to(repo),
        legacy_directories=("code",),
    )
    assert report["status"] == "passed"
    temporary = Path(report["temporary_worktree"])
    interpreter = str(Path(sys.executable).resolve())
    assert [entry[0] for entry in intercepted] == [
        (interpreter, "-m", "pytest", "-q"),
        (interpreter, "-m", "ruff", "check", "src", "tests"),
        (interpreter, "-m", "mypy", "src/smc_repro"),
        (interpreter, "-m", "compileall", "-q", "src", "tests"),
        (
            interpreter,
            "-m",
            "smc_repro.scripts.audit_legacy_outputs",
            "--repo-root",
            str(temporary),
            "--output",
            str(temporary / ".preflight_legacy_tracked.json"),
            "--scope",
            "tracked",
        ),
    ]
    for command, cwd, env in intercepted:
        assert command[0] == interpreter
        assert cwd == temporary
        assert env["PYTHONPATH"] == str(temporary / "src")
        assert env["PYTHONHASHSEED"] == "0"
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_clean_worktree_cleanup_command_failure_is_persisted_and_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, manifest = _tiny_repository(tmp_path)
    report_path = tmp_path / "cleanup-failed.json"
    original_runner = clean_gate_module._command_record

    def fail_cleanup(
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> tuple[dict[str, object], bytes, bytes]:
        if command[1:4] == ("worktree", "remove", "--force"):
            return (
                {
                    "command": list(command),
                    "cwd": str(cwd),
                    "started_at_utc": "fixture",
                    "ended_at_utc": "fixture",
                    "duration_seconds": 0.0,
                    "exit_code": 1,
                    "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                    "stderr_sha256": hashlib.sha256(b"cleanup failed").hexdigest(),
                },
                b"",
                b"cleanup failed",
            )
        return original_runner(command, cwd=cwd, env=env)

    monkeypatch.setattr(clean_gate_module, "_command_record", fail_cleanup)
    with pytest.raises(CleanWorktreeGateError, match="cleanup failed"):
        run_detached_worktree_gate(
            repo,
            Path(sys.executable),
            report_path,
            commands=(),
            project_relative=Path("."),
            tracked_manifest_relative=manifest.relative_to(repo),
            legacy_directories=("code",),
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["cleanup_commands"][-1]["exit_code"] == 1
    assert report["temporary_worktree_removed"] is True
    assert not Path(report["temporary_worktree"]).exists()
