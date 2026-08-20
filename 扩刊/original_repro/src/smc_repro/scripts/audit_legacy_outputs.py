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


def _tracked_legacy_blobs(root: Path) -> tuple[tuple[str, bytes], ...]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", *LEGACY_DIRS],
        check=True,
        capture_output=True,
    )
    blobs: list[tuple[str, bytes]] = []
    for raw_value in completed.stdout.split(b"\0"):
        if not raw_value:
            continue
        git_path = PurePosixPath(os.fsdecode(raw_value))
        relative_path = git_path.as_posix()
        blob = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob", f":{relative_path}"],
            check=True,
            capture_output=True,
        ).stdout
        blobs.append((relative_path, blob))
    return tuple(sorted(blobs, key=lambda item: item[0]))


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
    files: dict[str, dict[str, object]] = {}
    if scope is AuditScope.TRACKED:
        for relative_path, blob in _tracked_legacy_blobs(root):
            files[relative_path] = {
                "size_bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
    else:
        for path in _all_local_legacy_paths(root):
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
    output_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
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
