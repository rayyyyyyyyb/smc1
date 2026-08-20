from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from smc_repro.scripts import audit_legacy_outputs as audit_module


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


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
    manifest = audit_module.audit_legacy_outputs(
        tmp_path,
        output,
        audit_module.AuditScope.ALL_LOCAL,
    )
    assert manifest["schema_version"] == 2
    assert manifest["scope"] == "all_local"
    assert manifest["legacy_directories"] == ["code", "code1", "code2"]
    assert manifest["files"] == expected
    assert json.loads(output.read_text(encoding="utf-8"))["files"] == expected
    serialized = output.read_bytes()
    assert serialized.endswith(b"\n")
    assert b"\r\n" not in serialized


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

    tracked = audit_module.audit_legacy_outputs(
        root,
        root / "tracked.json",
        audit_module.AuditScope.TRACKED,
    )
    local = audit_module.audit_legacy_outputs(
        root,
        root / "local.json",
        audit_module.AuditScope.ALL_LOCAL,
    )

    assert set(tracked["files"]) == {"code/DQN.py"}
    assert "code/PM.txt" in local["files"]
    assert "code/PM.txt" not in tracked["files"]


def test_tracked_scope_hashes_canonical_index_blob_across_line_endings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(root, "init")
    _run_git(root, "config", "core.autocrlf", "false")
    for dirname in ("code", "code1", "code2"):
        (root / dirname).mkdir()
    tracked_path = root / "code" / "policy.txt"
    canonical_payload = b"first line\nsecond line\n"
    tracked_path.write_bytes(canonical_payload)
    _run_git(root, "add", "code/policy.txt")

    lf_checkout = audit_module.audit_legacy_outputs(
        root,
        root / "lf.json",
        audit_module.AuditScope.TRACKED,
    )
    _run_git(root, "config", "core.autocrlf", "true")
    tracked_path.write_bytes(canonical_payload.replace(b"\n", b"\r\n"))
    crlf_checkout = audit_module.audit_legacy_outputs(
        root,
        root / "crlf.json",
        audit_module.AuditScope.TRACKED,
    )

    assert crlf_checkout == lf_checkout
    assert crlf_checkout["files"]["code/policy.txt"] == {
        "size_bytes": len(canonical_payload),
        "sha256": hashlib.sha256(canonical_payload).hexdigest(),
    }
