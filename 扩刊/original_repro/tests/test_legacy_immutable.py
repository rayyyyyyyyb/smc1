from __future__ import annotations

import json
import subprocess
from pathlib import Path

from smc_repro.scripts import audit_legacy_outputs as audit_module


def test_legacy_manifest_matches_files(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    manifest_path = project_root / "docs" / "audit" / "legacy_tracked_manifest.json"
    assert manifest_path.is_file()
    clean_diff = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--exit-code",
            "--",
            "code",
            "code1",
            "code2",
        ],
        capture_output=True,
        text=True,
    )
    assert clean_diff.returncode == 0, clean_diff.stdout + clean_diff.stderr
    committed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated_manifest = audit_module.audit_legacy_outputs(
        repo_root,
        tmp_path / "legacy_tracked_manifest.json",
        audit_module.AuditScope.TRACKED,
    )

    assert committed_manifest["scope"] == "tracked"
    assert generated_manifest == committed_manifest
    assert all(not key.endswith("PM.txt") for key in committed_manifest["files"])
    assert all("/__pycache__/" not in key for key in committed_manifest["files"])
    assert all("/.idea/" not in key for key in committed_manifest["files"])
