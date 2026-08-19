from smc_repro.seeding import keyed_uniform


def test_unrelated_random_calls_do_not_change_failure_draw() -> None:
    expected = keyed_uniform(13, "process_failure", "i-1", 4, 2, 7)
    for index in range(10_000):
        keyed_uniform(13, "unrelated", index)
    observed = keyed_uniform(13, "process_failure", "i-1", 4, 2, 7)
    assert observed == expected
