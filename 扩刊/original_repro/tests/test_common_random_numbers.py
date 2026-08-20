from smc_repro.seeding import keyed_uniform


def test_unrelated_random_calls_do_not_change_failure_draw() -> None:
    expected = keyed_uniform(13, "failure_primary", "i-1", 4, 2, 7)
    for index in range(10_000):
        keyed_uniform(13, "unrelated", index)
    observed = keyed_uniform(13, "failure_primary", "i-1", 4, 2, 7)
    assert observed == expected


def test_failure_wear_and_repair_namespaces_are_stable_and_distinct() -> None:
    keys = ("i-1", 4, 2, 7)
    draws = {
        namespace: keyed_uniform(13, namespace, *keys)
        for namespace in (
            "failure_primary",
            "failure_secondary",
            "wear",
            "cm_recovery",
        )
    }

    assert len(set(draws.values())) == 4
    assert draws == {
        namespace: keyed_uniform(13, namespace, *keys) for namespace in draws
    }
