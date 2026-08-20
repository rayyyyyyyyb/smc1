from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import UTC, datetime
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
        raw_name = distribution.metadata["Name"]
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
        "captured_at_utc": datetime.now(UTC).isoformat(),
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
