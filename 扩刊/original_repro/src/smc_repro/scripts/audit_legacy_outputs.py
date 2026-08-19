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
