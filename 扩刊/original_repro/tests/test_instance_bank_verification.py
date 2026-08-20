import hashlib
import importlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from smc_repro.instance_generator import generate_legacy_instance
from smc_repro.instance_io import instance_sha256, save_instance
from smc_repro.scripts.build_instance_banks import build_instance_banks

VerifyInstanceBank = Callable[[Path, Path, str], Any]


def _verify_instance_bank() -> VerifyInstanceBank:
    module_name = "smc_repro.scripts.verify_instance_bank"
    assert importlib.util.find_spec(module_name) is not None, (
        "verify_instance_bank module must exist"
    )
    module = importlib.import_module(module_name)
    return module.verify_instance_bank


def _instance(instance_id: str, seed: int):
    return generate_legacy_instance(
        instance_id=instance_id,
        instance_seed=seed,
        failure_seed=seed + 100,
        machine_count=8,
        new_job_count=10,
        mean_interarrival=50.0,
    )


def _write_manifest(path: Path, manifest: object) -> bytes:
    rendered = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered)
    return rendered


def _rewrite_manifests(
    reference: Path,
    bank_root: Path,
    manifest: object,
) -> str:
    reference_bytes = _write_manifest(reference, manifest)
    _write_manifest(bank_root / "manifest.json", manifest)
    return hashlib.sha256(reference_bytes).hexdigest()


def _entries(manifest: dict[str, object]) -> list[dict[str, str]]:
    entries = manifest["files"]
    assert isinstance(entries, list)
    assert all(isinstance(entry, dict) for entry in entries)
    return entries


def _symlink_or_skip(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlinks are unavailable on this platform: {exc}")


def _tiny_bank(tmp_path: Path) -> tuple[Path, Path, str, dict[str, object]]:
    bank_root = tmp_path / "materialized"
    relative_paths = (
        "test/tiny_a.json.gz",
        "train/seed_000/tiny_b.json.gz",
    )
    entries: list[dict[str, str]] = []
    for index, relative_path in enumerate(relative_paths):
        path = bank_root / Path(*relative_path.split("/"))
        save_instance(_instance(f"tiny_{index}", 10 + index), path)
        entries.append({"path": relative_path, "sha256": instance_sha256(path)})

    manifest: dict[str, object] = {"schema_version": 1, "files": entries}
    reference_manifest = tmp_path / "release" / "manifest.json"
    reference_bytes = _write_manifest(reference_manifest, manifest)
    _write_manifest(bank_root / "manifest.json", manifest)
    expected_sha = hashlib.sha256(reference_bytes).hexdigest()
    return reference_manifest, bank_root, expected_sha, manifest


def test_exact_valid_bank_passes(tmp_path: Path) -> None:
    reference, bank_root, expected_sha, _ = _tiny_bank(tmp_path)

    report = _verify_instance_bank()(reference, bank_root, expected_sha)

    assert report.expected_file_count == 2
    assert report.verified_file_count == 2
    assert len(list(bank_root.rglob("*.json.gz"))) == 2
    assert report.ok is True


def test_missing_gzip_names_missing_relative_path(tmp_path: Path) -> None:
    reference, bank_root, expected_sha, _ = _tiny_bank(tmp_path)
    missing_path = bank_root / "test" / "tiny_a.json.gz"
    missing_path.unlink()

    with pytest.raises(FileNotFoundError, match=r"test/tiny_a\.json\.gz"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_modified_gzip_reports_sha_mismatch(tmp_path: Path) -> None:
    reference, bank_root, expected_sha, _ = _tiny_bank(tmp_path)
    modified_path = bank_root / "test" / "tiny_a.json.gz"
    modified_path.write_bytes(modified_path.read_bytes() + b"modified")

    with pytest.raises(
        ValueError,
        match=r"SHA-256 mismatch for test/tiny_a\.json\.gz",
    ):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_generated_manifest_must_be_byte_identical(tmp_path: Path) -> None:
    reference, bank_root, expected_sha, _ = _tiny_bank(tmp_path)
    generated_manifest = bank_root / "manifest.json"
    generated_manifest.write_bytes(generated_manifest.read_bytes() + b" ")

    with pytest.raises(ValueError, match="not byte-identical"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_wrong_expected_reference_manifest_sha_fails(tmp_path: Path) -> None:
    reference, bank_root, _, _ = _tiny_bank(tmp_path)

    with pytest.raises(ValueError, match="reference manifest SHA-256 mismatch"):
        _verify_instance_bank()(reference, bank_root, "0" * 64)


def test_extra_unlisted_gzip_fails(tmp_path: Path) -> None:
    reference, bank_root, expected_sha, _ = _tiny_bank(tmp_path)
    extra_path = bank_root / "extra" / "unlisted.json.gz"
    save_instance(_instance("extra", 99), extra_path)

    with pytest.raises(ValueError, match=r"extra/unlisted\.json\.gz"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        ".",
        "/absolute.json.gz",
        "C:/windows-drive.json.gz",
        r"test\backslash.json.gz",
        "../escape.json.gz",
    ],
)
def test_unsafe_manifest_paths_fail_before_instance_file_access(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    manifest["files"] = [{"path": unsafe_path, "sha256": "0" * 64}]
    reference_bytes = _write_manifest(reference, manifest)
    _write_manifest(bank_root / "manifest.json", manifest)
    expected_sha = hashlib.sha256(reference_bytes).hexdigest()

    with pytest.raises(ValueError, match="unsafe bank path"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_late_unsafe_path_is_rejected_before_earlier_instance_content(
    tmp_path: Path,
) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    first_path = bank_root / "test" / "tiny_a.json.gz"
    first_path.write_bytes(first_path.read_bytes() + b"modified")
    _entries(manifest)[-1] = {"path": "../late-escape.json.gz", "sha256": "0" * 64}
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="unsafe bank path"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_late_outside_symlink_is_rejected_before_earlier_instance_content(
    tmp_path: Path,
) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    first_path = bank_root / "test" / "tiny_a.json.gz"
    first_path.write_bytes(first_path.read_bytes() + b"modified")
    outside = tmp_path / "outside.json.gz"
    save_instance(_instance("outside", 301), outside)
    link = bank_root / "test" / "outside-link.json.gz"
    _symlink_or_skip(link, outside)
    _entries(manifest).append(
        {"path": "test/outside-link.json.gz", "sha256": instance_sha256(outside)}
    )
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="escapes root"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_manifest_root_must_be_object(tmp_path: Path) -> None:
    reference, bank_root, _, _ = _tiny_bank(tmp_path)
    expected_sha = _rewrite_manifests(reference, bank_root, [])

    with pytest.raises(ValueError, match="root must be an object"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_manifest_schema_version_must_be_exactly_one(tmp_path: Path) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    manifest["schema_version"] = 2
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="unsupported bank manifest schema version"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_duplicate_raw_path_is_rejected(tmp_path: Path) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    first_entry = dict(_entries(manifest)[0])
    _entries(manifest).append(first_entry)
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="duplicate bank path"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_noncanonical_posix_alias_is_rejected(tmp_path: Path) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    first_entry = dict(_entries(manifest)[0])
    first_entry["path"] = "test/./tiny_a.json.gz"
    _entries(manifest).append(first_entry)
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="canonical POSIX"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_casefold_path_collision_is_rejected(tmp_path: Path) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    first_entry = dict(_entries(manifest)[0])
    first_entry["path"] = "TEST/TINY_A.JSON.GZ"
    _entries(manifest).append(first_entry)
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="casefold collision"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_resolved_symlink_target_collision_is_rejected(tmp_path: Path) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    original = bank_root / "test" / "tiny_a.json.gz"
    alias = bank_root / "test" / "alias.json.gz"
    _symlink_or_skip(alias, original)
    _entries(manifest).append(
        {"path": "test/alias.json.gz", "sha256": instance_sha256(original)}
    )
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="resolved-target collision"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_manifest_sha_must_be_lowercase_64_hex(tmp_path: Path) -> None:
    reference, bank_root, _, manifest = _tiny_bank(tmp_path)
    _entries(manifest)[0]["sha256"] = "A" * 64
    expected_sha = _rewrite_manifests(reference, bank_root, manifest)

    with pytest.raises(ValueError, match="lower-case 64-hex SHA-256"):
        _verify_instance_bank()(reference, bank_root, expected_sha)


def test_builder_writes_crlf_manifest_when_text_writer_is_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def posix_write_text(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> int:
        posix_bytes = data.replace("\r\n", "\n").encode(
            encoding or "utf-8",
            errors or "strict",
        )
        return path.write_bytes(posix_bytes)

    monkeypatch.setattr(Path, "write_text", posix_write_text)
    output_root = tmp_path / "portable-bank"

    build_instance_banks(
        output_root=output_root,
        test_repetitions=1,
        train_seeds=(),
        train_episodes=1,
        base_seed=20260819,
    )

    manifest_bytes = (output_root / "manifest.json").read_bytes()
    assert manifest_bytes.endswith(b"\r\n")
    assert b"\n" not in manifest_bytes.replace(b"\r\n", b"")
