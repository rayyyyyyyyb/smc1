from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from smc_repro.instance_generator import generate_legacy_instance
from smc_repro.instance_io import instance_sha256, save_instance

MACHINES = (8, 12, 16)
MEAN_INTERARRIVALS = (50, 100, 150)
NEW_JOB_COUNTS = (10, 20, 30)
SCENARIOS = tuple(
    (machine_count, mean_interarrival, new_job_count)
    for machine_count in MACHINES
    for mean_interarrival in MEAN_INTERARRIVALS
    for new_job_count in NEW_JOB_COUNTS
)


def _entry(
    output_root: Path,
    path: Path,
    instance_id: str,
    instance_seed: int,
    failure_seed: int,
) -> dict[str, object]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "instance_id": instance_id,
        "instance_seed": instance_seed,
        "failure_seed": failure_seed,
        "sha256": instance_sha256(path),
    }


def build_instance_banks(
    *,
    output_root: Path,
    test_repetitions: int,
    train_seeds: tuple[int, ...],
    train_episodes: int,
    base_seed: int,
) -> dict[str, Any]:
    if test_repetitions <= 0 or train_episodes <= 0:
        raise ValueError("test_repetitions and train_episodes must be positive")
    if base_seed < 0 or any(seed < 0 for seed in train_seeds):
        raise ValueError("seeds must be non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []

    for scenario_index, (machine_count, mean_interarrival, new_job_count) in enumerate(
        SCENARIOS
    ):
        scenario_id = f"m{machine_count:02d}_j{new_job_count:02d}_e{mean_interarrival:03d}"
        for repetition in range(test_repetitions):
            instance_id = f"test_{scenario_id}_rep{repetition:02d}"
            instance_seed = base_seed + 1_000_000 + scenario_index * 10_000 + repetition
            failure_seed = base_seed + 2_000_000 + scenario_index * 10_000 + repetition
            instance = generate_legacy_instance(
                instance_id=instance_id,
                instance_seed=instance_seed,
                failure_seed=failure_seed,
                machine_count=machine_count,
                new_job_count=new_job_count,
                mean_interarrival=float(mean_interarrival),
            )
            path = output_root / "test" / scenario_id / f"{instance_id}.json.gz"
            save_instance(instance, path)
            entries.append(_entry(output_root, path, instance_id, instance_seed, failure_seed))

    for train_seed in train_seeds:
        parameter_rng = random.Random(base_seed + 3_000_000 + train_seed)
        for episode in range(train_episodes):
            machine_count = parameter_rng.randint(8, 18)
            mean_interarrival = parameter_rng.randint(50, 200)
            new_job_count = parameter_rng.randint(10, 30)
            instance_id = f"train_seed{train_seed:03d}_ep{episode:04d}"
            instance_seed = base_seed + 4_000_000 + train_seed * 100_000 + episode
            failure_seed = base_seed + 5_000_000 + train_seed * 100_000 + episode
            instance = generate_legacy_instance(
                instance_id=instance_id,
                instance_seed=instance_seed,
                failure_seed=failure_seed,
                machine_count=machine_count,
                new_job_count=new_job_count,
                mean_interarrival=float(mean_interarrival),
            )
            path = output_root / "train" / f"seed_{train_seed:03d}" / f"{instance_id}.json.gz"
            save_instance(instance, path)
            entries.append(_entry(output_root, path, instance_id, instance_seed, failure_seed))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "base_seed": base_seed,
        "scenario_order": [
            {
                "machine_count": machine_count,
                "mean_interarrival": mean_interarrival,
                "new_job_count": new_job_count,
            }
            for machine_count, mean_interarrival, new_job_count in SCENARIOS
        ],
        "test_repetitions": test_repetitions,
        "train_seeds": list(train_seeds),
        "train_episodes": train_episodes,
        "files": entries,
    }
    manifest_path = output_root / "manifest.json"
    rendered_manifest = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).replace("\n", "\r\n") + "\r\n"
    manifest_path.write_bytes(rendered_manifest.encode("utf-8"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--test-repetitions", type=int, default=20)
    parser.add_argument("--train-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20260819)
    args = parser.parse_args()
    build_instance_banks(
        output_root=args.output_root,
        test_repetitions=args.test_repetitions,
        train_seeds=tuple(args.train_seeds),
        train_episodes=args.train_episodes,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
