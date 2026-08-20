import pytest

from smc_repro.seeding import keyed_uniform


def test_keyed_uniform_is_repeatable_and_bounded() -> None:
    a = keyed_uniform(7, "failure", "instance-1", 2, 3, 4)
    b = keyed_uniform(7, "failure", "instance-1", 2, 3, 4)
    assert a == b
    assert 0.0 <= a < 1.0


def test_keyed_uniform_changes_when_key_changes() -> None:
    assert keyed_uniform(7, "x", 1) != keyed_uniform(7, "x", 2)


def test_keyed_uniform_distinguishes_delimiter_placement() -> None:
    assert keyed_uniform(7, "a|b", "c") != keyed_uniform(7, "a", "b|c")


def test_keyed_uniform_distinguishes_value_types() -> None:
    assert keyed_uniform(7, 1) != keyed_uniform(7, "1")
    assert keyed_uniform(7, True) != keyed_uniform(7, 1)


def test_keyed_uniform_rejects_nonfinite_float_keys() -> None:
    with pytest.raises(ValueError, match="finite"):
        keyed_uniform(7, float("nan"))


def test_keyed_uniform_rejects_unsupported_key_types() -> None:
    with pytest.raises(TypeError, match="random-stream keys"):
        keyed_uniform(7, [1, 2])


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
