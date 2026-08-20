from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from smc_repro.instance_io import instance_sha256, load_instance

_LOWER_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class BankVerificationReport:
    reference_manifest: str
    reference_manifest_sha256: str
    generated_manifest: str
    generated_manifest_sha256: str
    expected_file_count: int
    verified_file_count: int
    ok: bool


@dataclass(frozen=True)
class _ValidatedBankEntry:
    relative_path: str
    expected_sha256: str
    resolved_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_instance_bank(
    reference_manifest: Path,
    bank_root: Path,
    expected_manifest_sha256: str,
) -> BankVerificationReport:
    reference_manifest = reference_manifest.resolve()
    bank_root = bank_root.resolve()
    generated_manifest = bank_root / "manifest.json"

    actual_reference_sha = _sha256(reference_manifest)
    if actual_reference_sha != expected_manifest_sha256:
        raise ValueError(
            "reference manifest SHA-256 mismatch: "
            f"{actual_reference_sha} != {expected_manifest_sha256}"
        )
    if not generated_manifest.is_file():
        raise FileNotFoundError(f"generated manifest is missing: {generated_manifest}")

    reference_bytes = reference_manifest.read_bytes()
    generated_bytes = generated_manifest.read_bytes()
    if generated_bytes != reference_bytes:
        raise ValueError("generated manifest is not byte-identical to the reference manifest")

    manifest: object = json.loads(reference_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("bank manifest root must be an object")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError(
            f"unsupported bank manifest schema version: {schema_version!r}; expected 1"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("bank manifest 'files' must be a list")

    raw_paths: set[str] = set()
    normalized_paths: set[str] = set()
    casefold_paths: dict[str, str] = {}
    lexical_entries: list[tuple[str, str, PurePosixPath]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each bank manifest entry must be an object")
        relative_path = entry.get("path")
        expected_sha = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha, str):
            raise ValueError("bank entry requires string path and sha256")
        if relative_path in raw_paths:
            raise ValueError(f"duplicate bank path in manifest: {relative_path}")
        raw_paths.add(relative_path)
        pure_path = PurePosixPath(relative_path)
        windows_path = PureWindowsPath(relative_path)
        if (
            pure_path.is_absolute()
            or not pure_path.parts
            or "\\" in relative_path
            or bool(windows_path.drive)
            or bool(windows_path.root)
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ValueError(f"unsafe bank path in manifest: {relative_path!r}")
        normalized_path = pure_path.as_posix()
        if normalized_path != relative_path:
            raise ValueError(
                f"bank path is not canonical POSIX spelling: {relative_path!r}"
            )
        if normalized_path in normalized_paths:
            raise ValueError(f"normalized bank path collision: {relative_path!r}")
        normalized_paths.add(normalized_path)
        casefold_path = normalized_path.casefold()
        previous_casefold = casefold_paths.get(casefold_path)
        if previous_casefold is not None:
            raise ValueError(
                "bank path casefold collision: "
                f"{previous_casefold!r} versus {relative_path!r}"
            )
        casefold_paths[casefold_path] = relative_path
        if _LOWER_HEX_SHA256.fullmatch(expected_sha) is None:
            raise ValueError(
                f"bank entry requires lower-case 64-hex SHA-256: {relative_path!r}"
            )
        lexical_entries.append((relative_path, expected_sha, pure_path))

    validated_entries: list[_ValidatedBankEntry] = []
    resolved_targets: dict[str, str] = {}
    for relative_path, expected_sha, pure_path in lexical_entries:
        resolved_path = (bank_root / Path(*pure_path.parts)).resolve()
        if not resolved_path.is_relative_to(bank_root):
            raise ValueError(f"bank path escapes root: {relative_path!r}")
        resolved_key = str(resolved_path).casefold()
        previous_target = resolved_targets.get(resolved_key)
        if previous_target is not None:
            raise ValueError(
                "bank path resolved-target collision: "
                f"{previous_target!r} versus {relative_path!r}"
            )
        resolved_targets[resolved_key] = relative_path
        validated_entries.append(
            _ValidatedBankEntry(relative_path, expected_sha, resolved_path)
        )

    verified_count = 0
    for entry in validated_entries:
        if not entry.resolved_path.is_file():
            relative_path = entry.relative_path
            raise FileNotFoundError(f"bank instance is missing: {relative_path}")
        actual_sha = instance_sha256(entry.resolved_path)
        if actual_sha != entry.expected_sha256:
            raise ValueError(
                f"bank instance SHA-256 mismatch for {entry.relative_path}: "
                f"{actual_sha} != {entry.expected_sha256}"
            )
        load_instance(entry.resolved_path)
        verified_count += 1

    observed_paths = {
        path.relative_to(bank_root).as_posix()
        for path in bank_root.rglob("*.json.gz")
        if path.is_file()
    }
    expected_paths = {entry.relative_path for entry in validated_entries}
    extra_paths = sorted(observed_paths - expected_paths)
    if extra_paths:
        raise ValueError(f"bank contains unlisted gzip files: {extra_paths[:10]}")
    unobserved_paths = sorted(expected_paths - observed_paths)
    if unobserved_paths:
        raise ValueError(f"bank manifest paths were not observed exactly: {unobserved_paths[:10]}")
    expected_count = len(validated_entries)
    observed_count = len(observed_paths)
    if verified_count != expected_count or observed_count != expected_count:
        raise ValueError(
            "bank verification count mismatch: "
            f"expected={expected_count}, verified={verified_count}, observed={observed_count}"
        )

    return BankVerificationReport(
        reference_manifest=str(reference_manifest),
        reference_manifest_sha256=actual_reference_sha,
        generated_manifest=str(generated_manifest),
        generated_manifest_sha256=_sha256(generated_manifest),
        expected_file_count=expected_count,
        verified_file_count=verified_count,
        ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_instance_bank(
        args.reference,
        args.bank_root,
        args.expected_manifest_sha256,
    )
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
