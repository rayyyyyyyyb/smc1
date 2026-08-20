from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from smc_repro.rules.base import DispatchDecision
from smc_repro.rules.classical import ClassicalRule, dispatch_classical_rule
from tests.test_legacy_rules import make_hand_context


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (ClassicalRule.FIFO_ECT, DispatchDecision(1, 1, 0, "FIFO+ECT")),
        (ClassicalRule.EDD_ECT, DispatchDecision(2, 3, 2, "EDD+ECT")),
        (ClassicalRule.MRT_ECT, DispatchDecision(1, 1, 0, "MRT+ECT")),
        (ClassicalRule.SPT_ECT, DispatchDecision(2, 3, 2, "SPT+ECT")),
        (ClassicalRule.LPT_ECT, DispatchDecision(0, 2, 1, "LPT+ECT")),
    ],
)
def test_classical_rules_choose_hand_computed_job_then_ect_machine(
    rule: ClassicalRule,
    expected: DispatchDecision,
) -> None:
    assert dispatch_classical_rule(make_hand_context(), rule) == expected


def test_classical_maximizers_use_lower_job_id_on_ties() -> None:
    context = make_hand_context()
    tied_jobs = (
        replace(
            context.jobs[0],
            remaining_nominal_work=17.0,
            next_operation_mean_processing_time=7.0,
        ),
        replace(context.jobs[1], next_operation_mean_processing_time=7.0),
        context.jobs[2],
    )
    context = replace(context, jobs=tied_jobs)

    mrt = dispatch_classical_rule(context, ClassicalRule.MRT_ECT)
    lpt = dispatch_classical_rule(context, ClassicalRule.LPT_ECT)

    assert mrt == DispatchDecision(0, 2, 1, "MRT+ECT")
    assert lpt == DispatchDecision(0, 2, 1, "LPT+ECT")


def test_classical_dispatch_does_not_mutate_context() -> None:
    context = make_hand_context()

    for rule in ClassicalRule:
        before = copy.deepcopy(context)
        dispatch_classical_rule(context, rule)
        assert context == before


def test_classical_rejects_an_unsupported_rule() -> None:
    with pytest.raises(AssertionError, match="unsupported classical rule"):
        dispatch_classical_rule(make_hand_context(), "FIFO+ECT")  # type: ignore[arg-type]
