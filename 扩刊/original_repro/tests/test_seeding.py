from smc_repro.seeding import keyed_uniform


def test_keyed_uniform_is_repeatable_and_bounded() -> None:
    a = keyed_uniform(7, "failure", "instance-1", 2, 3, 4)
    b = keyed_uniform(7, "failure", "instance-1", 2, 3, 4)
    assert a == b
    assert 0.0 <= a < 1.0


def test_keyed_uniform_changes_when_key_changes() -> None:
    assert keyed_uniform(7, "x", 1) != keyed_uniform(7, "x", 2)


def test_set_global_seed_repeats_python_numpy_and_torch() -> None:
    import random

    import numpy as np
    import torch

    from smc_repro.seeding import set_global_seed

    set_global_seed(123)
    first = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    set_global_seed(123)
    second = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert first == second
