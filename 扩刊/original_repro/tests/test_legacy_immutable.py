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
    repo_root = Path(__file__).resolve().parents[3]
    project_root = repo_root / "扩刊"
    manifest_path = project_root / "docs" / "audit" / "legacy_manifest.json"
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
