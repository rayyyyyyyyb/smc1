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

## Original-conference data preparation

The original SMC study uses synthetic instances; no public benchmark download is required for
this reproduction phase. The repository commits the reference manifest, not the 1540 compressed
instances. Generate and verify them locally before preflight with the commands below.

The expected reference-manifest SHA-256 is fixed at
`68a2fd2420a8710743b28db1b234b69dc2ecf96046d14950abac22cee7cd1515`.
Brandimarte, Hurink, OR-Library, Taillard, and all other external benchmarks are explicitly
deferred until the later GNN-generalization phase; they must not be downloaded or mixed into the
original-conference reproduction.

PowerShell:

```powershell
$repo = (git rev-parse --show-toplevel).Trim()
Set-Location "$repo\扩刊\original_repro"
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
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
python -m smc_repro.scripts.preflight `
  --repo-root ..\.. `
  --bank-root artifacts\banks\materialized `
  --reference-manifest artifacts\banks\release\manifest.json `
  --environment-metadata ..\docs\audit\environment_5090_resolved.json `
  --output artifacts\preflight\preflight_report.json `
  --device cuda:0
$pythonExe = (Get-Command python).Source
python -m smc_repro.scripts.clean_worktree_gate `
  --repo-root ..\.. `
  --python-executable $pythonExe `
  --report artifacts\preflight\clean_worktree_report.json
```

Bash:

```bash
repo="$(git rev-parse --show-toplevel)"
cd "$repo/扩刊/original_repro"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
python -m smc_repro.scripts.preflight \
  --repo-root ../.. \
  --bank-root artifacts/banks/materialized \
  --reference-manifest artifacts/banks/release/manifest.json \
  --environment-metadata ../docs/audit/environment_5090_resolved.json \
  --output artifacts/preflight/preflight_report.json \
  --device cuda:0
python -m smc_repro.scripts.clean_worktree_gate \
  --repo-root ../.. \
  --python-executable "$(command -v python)" \
  --report artifacts/preflight/clean_worktree_report.json
```
