from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from smc_repro.rules.base import DispatchDecision
from smc_repro.rules.legacy import dispatch_legacy_rule
from smc_repro.rules.paper import dispatch_paper_rule
from tests.test_legacy_rules import make_hand_context, make_tardy_context


def test_paper_a1_uses_urgency_weighted_work_completion_not_count() -> None:
    context = make_hand_context()

    paper = dispatch_paper_rule(context, 0)
    legacy = dispatch_legacy_rule(context, 1)

    # Work/urgency scores are .60/3, .15/1, and .40/2, selecting job 1.
    # Count completion ties jobs 0 and 1 and therefore selects lower job 0.
    assert paper == DispatchDecision(1, 1, 2, "paper_A1_B1_EST")
    assert legacy.job_id == 0


def test_paper_a2_reproduces_printed_non_tardy_branch() -> None:
    context = make_hand_context()

    decision = dispatch_paper_rule(context, 3)

    # Printed scores are (18-12)/8, (20-3)/17, and (16-4)/6.
    # A conventional slack formula would incorrectly choose job 1.
    assert decision == DispatchDecision(0, 2, 0, "paper_A2_B1_EST")


def test_paper_a2_reproduces_printed_tardy_branch() -> None:
    context = make_tardy_context(make_hand_context())

    decision = dispatch_paper_rule(context, 4)

    # Tardy scores are (10-6)/1, (10-9)/3, and (10-12)/2, so job 0 wins.
    assert decision == DispatchDecision(0, 2, 1, "paper_A2_B2_ECT")


def test_legacy_b1_is_ect_while_paper_b1_is_est_and_paper_b2_is_ect() -> None:
    context = make_hand_context()
    shared_a1_jobs = (
        replace(context.jobs[0], completion_ratio_by_work=0.0),
        replace(context.jobs[1], completion_ratio_by_work=0.9),
        replace(context.jobs[2], completion_ratio_by_work=0.8),
    )
    context = replace(context, jobs=shared_a1_jobs)

    legacy_b1 = dispatch_legacy_rule(context, 0)
    paper_b1 = dispatch_paper_rule(context, 0)
    paper_b2 = dispatch_paper_rule(context, 1)

    assert legacy_b1 == DispatchDecision(0, 2, 1, "legacy_A1_ECT")
    assert paper_b1 == DispatchDecision(0, 2, 0, "paper_A1_B1_EST")
    assert paper_b2 == DispatchDecision(0, 2, 1, "paper_A1_B2_ECT")


def test_paper_random_choices_are_repeatable_and_namespace_separated() -> None:
    contexts = tuple(make_hand_context(decision_index=index) for index in range(128))

    first = tuple(dispatch_paper_rule(context, 8) for context in contexts)
    second = tuple(dispatch_paper_rule(context, 8) for context in contexts)
    machine_ids = {
        action: {dispatch_paper_rule(context, action).machine_id for context in contexts}
        for action in (2, 5)
    }
    job_ids = {
        action: {dispatch_paper_rule(context, action).job_id for context in contexts}
        for action in (6, 7, 8)
    }
    action_eight_machines = {job_id: set() for job_id in range(3)}
    for decision in first:
        action_eight_machines[decision.job_id].add(decision.machine_id)
    paper_action_6 = tuple(dispatch_paper_rule(context, 6).job_id for context in contexts)
    paper_action_7 = tuple(dispatch_paper_rule(context, 7).job_id for context in contexts)
    legacy_action_6 = tuple(dispatch_legacy_rule(context, 6).job_id for context in contexts)
    other_seed = tuple(
        dispatch_paper_rule(replace(context, policy_seed=1730), 6).job_id
        for context in contexts
    )

    assert first == second
    assert len(set(first)) > 1
    assert machine_ids == {2: {0, 2}, 5: {0, 1}}
    assert job_ids == {6: {0, 1, 2}, 7: {0, 1, 2}, 8: {0, 1, 2}}
    assert action_eight_machines == {0: {0, 1}, 1: {0, 2}, 2: {1, 2}}
    assert paper_action_6 != paper_action_7
    assert paper_action_6 != legacy_action_6
    assert paper_action_6 != other_seed


def test_paper_action_names_are_unique_and_stable() -> None:
    expected = (
        "paper_A1_B1_EST",
        "paper_A1_B2_ECT",
        "paper_A1_B3_RANDOM",
        "paper_A2_B1_EST",
        "paper_A2_B2_ECT",
        "paper_A2_B3_RANDOM",
        "paper_A3_B1_EST",
        "paper_A3_B2_ECT",
        "paper_A3_B3_RANDOM",
    )
    context = make_hand_context()

    actual = tuple(dispatch_paper_rule(context, action).rule_name for action in range(9))

    assert actual == expected
    assert len(set(actual)) == 9


@pytest.mark.parametrize("action_index", [-1, 9])
def test_paper_rejects_invalid_action_indices(action_index: int) -> None:
    with pytest.raises(ValueError, match="0..8"):
        dispatch_paper_rule(make_hand_context(), action_index)


@pytest.mark.parametrize(
    "action_index",
    [False, True, 0.0, 8.0, float("nan"), float("inf"), "0", None, [], {}],
)
def test_paper_rejects_non_integer_action_indices(action_index: object) -> None:
    with pytest.raises(ValueError, match="integer in 0..8"):
        dispatch_paper_rule(make_hand_context(), action_index)  # type: ignore[arg-type]


def test_all_paper_dispatches_preserve_a_deep_context_snapshot() -> None:
    context = make_hand_context()

    for action_index in range(9):
        before = copy.deepcopy(context)
        dispatch_paper_rule(context, action_index)
        assert context == before
