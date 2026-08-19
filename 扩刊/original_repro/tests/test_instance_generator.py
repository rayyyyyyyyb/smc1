import random

import numpy as np
import pytest

from smc_repro.instance_generator import generate_legacy_instance


def _generate():
    return generate_legacy_instance(
        instance_id="x",
        instance_seed=101,
        failure_seed=201,
        machine_count=8,
        new_job_count=10,
        mean_interarrival=50.0,
    )


def test_generator_is_repeatable() -> None:
    assert _generate() == _generate()


def test_generator_matches_legacy_ranges() -> None:
    instance = _generate()
    assert len(instance.jobs) == 15
    assert len(instance.machines) == 8
    for job in instance.jobs:
        assert 1 <= len(job.operations) <= 20
        for operation in job.operations:
            assert 2 <= len(operation.eligible_machines) <= 7
            for machine_id in operation.eligible_machines:
                assert 1 <= operation.processing_time(machine_id) <= 50


def test_generator_does_not_change_global_rng_state() -> None:
    random.seed(999)
    np.random.seed(999)
    expected_py = random.random()
    expected_np = float(np.random.random())
    random.seed(999)
    np.random.seed(999)
    _generate()
    assert random.random() == expected_py
    assert float(np.random.random()) == expected_np


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_generator_rejects_non_finite_mean_interarrival(value: float) -> None:
    with pytest.raises(ValueError, match="mean_interarrival.*finite"):
        generate_legacy_instance(
            instance_id="bad-scale",
            instance_seed=101,
            failure_seed=201,
            machine_count=8,
            new_job_count=10,
            mean_interarrival=value,
        )
