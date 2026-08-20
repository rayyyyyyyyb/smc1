# SMC Original-Conference Reproduction

This package is the audited, reproducible implementation of the original SMC DL-DDQN study.
The legacy directories `../../code`, `../../code1`, and `../../code2` are frozen evidence and must not
be edited during the reproduction phase.

## Environment

Install the official PyTorch CUDA wheel first, then install this project:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src/smc_repro
python -m smc_repro.scripts.verify_hardware
```

## Data

The original SMC conference reproduction uses only programmatically generated synthetic dynamic-FJSP instances. Do not download Brandimarte, Hurink, OR-Library, Taillard, or other external benchmark sets in this phase. Those benchmarks belong to the later GNN-upgrade evaluation and must not be mixed into the original-paper results.

The reference manifest is committed at `artifacts/banks/release/manifest.json`; the 1540 gzip
instance files are generated locally. The expected reference-manifest SHA-256 is fixed at
`68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515`.
External benchmark data remains deferred to the later GNN phase. Every formal run must record
the bank-manifest SHA in its provenance.

PowerShell:

```powershell
cd 扩刊\original_repro
Remove-Item -Recurse -Force artifacts\banks\materialized -ErrorAction SilentlyContinue
python -m smc_repro.scripts.build_instance_banks `
  --output-root artifacts\banks\materialized `
  --test-repetitions 20 `
  --train-seeds 0 1 2 3 4 `
  --train-episodes 200 `
  --base-seed 20260819
python -m smc_repro.scripts.verify_instance_bank `
  --reference artifacts\banks\release\manifest.json `
  --bank-root artifacts\banks\materialized `
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 `
  --report artifacts\preflight\bank_verification.json
```

Bash:

```bash
cd 扩刊/original_repro
rm -rf artifacts/banks/materialized
python -m smc_repro.scripts.build_instance_banks \
  --output-root artifacts/banks/materialized \
  --test-repetitions 20 \
  --train-seeds 0 1 2 3 4 \
  --train-episodes 200 \
  --base-seed 20260819
python -m smc_repro.scripts.verify_instance_bank \
  --reference artifacts/banks/release/manifest.json \
  --bank-root artifacts/banks/materialized \
  --expected-manifest-sha256 68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515 \
  --report artifacts/preflight/bank_verification.json
```
