import gzip
import json

import pytest

from smc_repro.instance_generator import generate_legacy_instance
from smc_repro.instance_io import load_instance, save_instance


def _instance():
    return generate_legacy_instance(
        instance_id="roundtrip",
        instance_seed=103,
        failure_seed=203,
        machine_count=8,
        new_job_count=10,
        mean_interarrival=50.0,
    )


def test_json_gzip_round_trip(tmp_path) -> None:
    path = tmp_path / "roundtrip.json.gz"
    save_instance(_instance(), path)
    assert load_instance(path) == _instance()


def test_identical_payload_has_identical_compressed_bytes(tmp_path) -> None:
    a = tmp_path / "a.json.gz"
    b = tmp_path / "nested" / "b.json.gz"
    save_instance(_instance(), a)
    save_instance(_instance(), b)
    assert a.read_bytes() == b.read_bytes()


def test_unsupported_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"schema_version": 99}, handle)
    with pytest.raises(ValueError, match="unsupported"):
        load_instance(path)


def test_malformed_gzip_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json.gz"
    path.write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="failed to read"):
        load_instance(path)


def test_malformed_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad-json.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('{"schema_version": 1,')
    with pytest.raises(ValueError, match="failed to read"):
        load_instance(path)
