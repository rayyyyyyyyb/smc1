import hashlib
import json

from smc_repro.scripts.audit_legacy_outputs import audit_legacy_outputs


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
    manifest = audit_legacy_outputs(tmp_path, output)
    assert manifest["files"] == expected
    assert json.loads(output.read_text(encoding="utf-8"))["files"] == expected
