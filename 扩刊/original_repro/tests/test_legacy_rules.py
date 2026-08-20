from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import smc_repro.rules as rules
from smc_repro.rules.base import (
    DispatchDecision,
    JobRuleView,
    MachineSelector,
    PairRuleView,
    RuleContext,
    keyed_choice,
    select_machine,
)
from smc_repro.rules.legacy import dispatch_legacy_rule


def make_hand_context(
    *,
    decision_index: int = 0,
    policy_seed: int = 1729,
) -> RuleContext:
    """Return the shared three-job/three-machine hand-calculated fixture."""
    jobs = (
        JobRuleView(
            job_id=0,
            op_id=2,
            arrival_time=2.0,
            due_date=18.0,
            urgency=1,
            decision_time=10.0,
            latest_process_end=4.0,
            operation_count=4,
            completed_operation_count=1,
            completion_ratio_by_count=0.25,
            completion_ratio_by_work=0.60,
            processed_work=12.0,
            remaining_nominal_work=8.0,
            next_operation_mean_processing_time=7.0,
        ),
        JobRuleView(
            job_id=1,
            op_id=1,
            arrival_time=0.0,
            due_date=20.0,
            urgency=3,
            decision_time=10.0,
            latest_process_end=8.0,
            operation_count=4,
            completed_operation_count=1,
            completion_ratio_by_count=0.25,
            completion_ratio_by_work=0.15,
            processed_work=3.0,
            remaining_nominal_work=17.0,
            next_operation_mean_processing_time=5.0,
        ),
        JobRuleView(
            job_id=2,
            op_id=3,
            arrival_time=4.0,
            due_date=16.0,
            urgency=2,
            decision_time=10.0,
            latest_process_end=6.0,
            operation_count=4,
            completed_operation_count=2,
            completion_ratio_by_count=0.50,
            completion_ratio_by_work=0.40,
            processed_work=4.0,
            remaining_nominal_work=6.0,
            next_operation_mean_processing_time=3.0,
        ),
    )
    pairs = (
        PairRuleView(
            job_id=0,
            op_id=2,
            machine_id=0,
            earliest_start=5.0,
            estimated_completion=12.0,
        ),
        PairRuleView(
            job_id=0,
            op_id=2,
            machine_id=1,
            earliest_start=7.0,
            estimated_completion=9.0,
        ),
        PairRuleView(
            job_id=1,
            op_id=1,
            machine_id=0,
            earliest_start=8.0,
            estimated_completion=9.0,
        ),
        PairRuleView(
            job_id=1,
            op_id=1,
            machine_id=2,
            earliest_start=4.0,
            estimated_completion=11.0,
        ),
        PairRuleView(
            job_id=2,
            op_id=3,
            machine_id=1,
            earliest_start=3.0,
            estimated_completion=8.0,
        ),
        PairRuleView(
            job_id=2,
            op_id=3,
            machine_id=2,
            earliest_start=3.0,
            estimated_completion=6.0,
        ),
    )
    return RuleContext(
        instance_id="hand-three-by-three",
        decision_index=decision_index,
        policy_seed=policy_seed,
        jobs=jobs,
        pairs=pairs,
    )


def make_tardy_context(context: RuleContext) -> RuleContext:
    due_dates = (6.0, 9.0, 12.0)
    return replace(
        context,
        jobs=tuple(
            replace(job, due_date=due_date)
            for job, due_date in zip(context.jobs, due_dates, strict=True)
        ),
    )


def test_legacy_a1_uses_count_ratio_and_lower_job_id_tie_break() -> None:
    context = make_hand_context()

    decision = dispatch_legacy_rule(context, 0)

    assert decision == DispatchDecision(0, 2, 1, "legacy_A1_ECT")


def test_legacy_a2_reproduces_non_tardy_source_formula() -> None:
    context = make_hand_context()

    ect = dispatch_legacy_rule(context, 3)
    est = dispatch_legacy_rule(context, 4)

    assert ect == DispatchDecision(0, 2, 1, "legacy_A2_ECT")
    assert est == DispatchDecision(0, 2, 0, "legacy_A2_EST")


def test_legacy_a2_preserves_tardy_source_parentheses() -> None:
    context = make_tardy_context(make_hand_context())

    decision = dispatch_legacy_rule(context, 3)

    # Source scores are 6-10/3, 9-10/1, and 12-10/2, so job 1 wins.
    # The conventional (due-now)/(4-urgency) rewrite would incorrectly choose job 0.
    assert decision == DispatchDecision(1, 1, 0, "legacy_A2_ECT")


def test_legacy_b1_is_ect_and_b2_is_est() -> None:
    context = make_hand_context()

    b1 = dispatch_legacy_rule(context, 0)
    b2 = dispatch_legacy_rule(context, 1)

    assert (b1.job_id, b1.machine_id) == (0, 1)
    assert (b2.job_id, b2.machine_id) == (0, 0)


def test_machine_ties_choose_the_lower_machine_id() -> None:
    context = make_hand_context()
    job = context.jobs[2]

    pair = select_machine(
        context,
        job,
        MachineSelector.EARLIEST_START,
        namespace="tie-proof",
    )

    assert (pair.job_id, pair.machine_id) == (2, 1)


def test_legacy_random_choices_repeat_and_sweep_all_candidates() -> None:
    contexts = tuple(make_hand_context(decision_index=index) for index in range(128))

    first = tuple(dispatch_legacy_rule(context, 8) for context in contexts)
    second = tuple(dispatch_legacy_rule(context, 8) for context in contexts)
    random_machine_ids = {
        action: {dispatch_legacy_rule(context, action).machine_id for context in contexts}
        for action in (2, 5)
    }
    random_job_ids = {
        action: {dispatch_legacy_rule(context, action).job_id for context in contexts}
        for action in (6, 7, 8)
    }
    action_eight_machines = {job_id: set() for job_id in range(3)}
    for decision in first:
        action_eight_machines[decision.job_id].add(decision.machine_id)

    assert first == second
    assert len(set(first)) > 1
    assert random_machine_ids == {2: {0, 1}, 5: {0, 1}}
    assert random_job_ids == {6: {0, 1, 2}, 7: {0, 1, 2}, 8: {0, 1, 2}}
    assert action_eight_machines == {0: {0, 1}, 1: {0, 2}, 2: {1, 2}}


def test_every_legacy_decision_uses_a_legal_pair() -> None:
    for decision_index in range(64):
        context = make_hand_context(decision_index=decision_index)
        legal = {
            (pair.job_id, pair.op_id, pair.machine_id) for pair in context.pairs
        }
        for action_index in range(9):
            decision = dispatch_legacy_rule(context, action_index)
            assert (decision.job_id, decision.op_id, decision.machine_id) in legal


def test_legacy_action_names_are_unique_and_stable() -> None:
    expected = (
        "legacy_A1_ECT",
        "legacy_A1_EST",
        "legacy_A1_RANDOM",
        "legacy_A2_ECT",
        "legacy_A2_EST",
        "legacy_A2_RANDOM",
        "legacy_A3_ECT",
        "legacy_A3_EST",
        "legacy_A3_RANDOM",
    )
    context = make_hand_context()

    actual = tuple(dispatch_legacy_rule(context, action).rule_name for action in range(9))

    assert actual == expected
    assert len(set(actual)) == 9


@pytest.mark.parametrize("action_index", [-1, 9])
def test_legacy_rejects_invalid_action_indices(action_index: int) -> None:
    with pytest.raises(ValueError, match="0..8"):
        dispatch_legacy_rule(make_hand_context(), action_index)


@pytest.mark.parametrize(
    "action_index",
    [False, True, 0.0, 8.0, float("nan"), float("inf"), "0", None, [], {}],
)
def test_legacy_rejects_non_integer_action_indices(action_index: object) -> None:
    with pytest.raises(ValueError, match="integer in 0..8"):
        dispatch_legacy_rule(make_hand_context(), action_index)  # type: ignore[arg-type]


def test_context_rejects_empty_or_orphaned_views() -> None:
    context = make_hand_context()

    with pytest.raises(ValueError, match="ready jobs and legal pairs"):
        replace(context, jobs=(), pairs=())
    with pytest.raises(ValueError, match="ready jobs and legal pairs"):
        replace(context, pairs=())
    with pytest.raises(ValueError, match="every pair"):
        replace(context, pairs=(replace(context.pairs[0], job_id=99),))


def test_context_rejects_negative_duplicate_and_nonfinite_inputs() -> None:
    context = make_hand_context()

    with pytest.raises(ValueError, match="non-negative"):
        replace(context, decision_index=-1)
    with pytest.raises(ValueError, match="non-negative"):
        replace(context, policy_seed=-1)
    with pytest.raises(ValueError, match="ready job views must be unique"):
        replace(context, jobs=context.jobs + (context.jobs[0],))
    with pytest.raises(ValueError, match="legal machine pairs must be unique"):
        replace(context, pairs=context.pairs + (context.pairs[0],))
    with pytest.raises(ValueError, match="finite"):
        replace(context.jobs[0], due_date=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        replace(context.pairs[0], earliest_start=float("inf"))


@pytest.mark.parametrize("instance_id", [None, "", 0, False, [], {}])
def test_context_requires_a_non_empty_string_instance_id(instance_id: object) -> None:
    with pytest.raises(ValueError, match="instance_id must be a non-empty string"):
        replace(make_hand_context(), instance_id=instance_id)


@pytest.mark.parametrize("field_name", ["decision_index", "policy_seed"])
@pytest.mark.parametrize(
    "value",
    [False, True, 0.0, 1.5, float("nan"), float("inf"), -1, "0", None, [], {}],
)
def test_context_requires_non_bool_non_negative_integer_indices(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field_name} must be a non-negative integer"):
        replace(make_hand_context(), **{field_name: value})


def test_context_detaches_from_mutable_job_and_pair_sequence_aliases() -> None:
    original = make_hand_context()
    jobs = list(original.jobs)
    pairs = list(original.pairs)
    context = RuleContext(
        instance_id=original.instance_id,
        decision_index=original.decision_index,
        policy_seed=original.policy_seed,
        jobs=jobs,  # type: ignore[arg-type]
        pairs=pairs,  # type: ignore[arg-type]
    )
    snapshot = copy.deepcopy(context)

    jobs.clear()
    pairs.clear()

    assert isinstance(context.jobs, tuple)
    assert isinstance(context.pairs, tuple)
    assert context == snapshot


def test_context_rejects_wrong_job_and_pair_view_element_types() -> None:
    context = make_hand_context()

    with pytest.raises(TypeError, match="jobs must contain only JobRuleView"):
        replace(context, jobs=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pairs must contain only PairRuleView"):
        replace(context, pairs=(object(),))  # type: ignore[arg-type]


def test_views_and_decisions_are_frozen_and_all_legacy_dispatches_preserve_context() -> None:
    context = make_hand_context()

    for action_index in range(9):
        before = copy.deepcopy(context)
        dispatch_legacy_rule(context, action_index)
        assert context == before

    decision = dispatch_legacy_rule(context, 8)
    with pytest.raises(FrozenInstanceError):
        context.jobs[0].due_date = 0.0
    with pytest.raises(FrozenInstanceError):
        context.pairs[0].machine_id = 99
    with pytest.raises(FrozenInstanceError):
        context.decision_index = 99
    with pytest.raises(FrozenInstanceError):
        decision.machine_id = 99


def test_missing_selected_job_pair_and_empty_keyed_choice_fail_loudly() -> None:
    context = make_hand_context()
    without_job_zero_pairs = replace(
        context,
        pairs=tuple(pair for pair in context.pairs if pair.job_id != 0),
    )

    with pytest.raises(ValueError, match="no legal machine pair"):
        dispatch_legacy_rule(without_job_zero_pairs, 0)
    with pytest.raises(ValueError, match="empty sequence"):
        keyed_choice((), 1, "empty")


def test_rule_modules_have_no_forbidden_direct_imports() -> None:
    rule_dir = Path(__file__).parents[1] / "src" / "smc_repro" / "rules"
    forbidden_roots = {"numpy", "random", "torch"}
    forbidden_modules = {"smc_repro.runtime", "smc_repro.timeline"}
    violations: list[str] = []

    for path in sorted(rule_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            for module_name in imported:
                root = module_name.split(".", maxsplit=1)[0]
                if root in forbidden_roots or module_name in forbidden_modules:
                    violations.append(f"{path.name}:{node.lineno}:{module_name}")

    assert violations == []


def test_rules_package_exports_only_the_locked_public_api() -> None:
    assert rules.__all__ == [
        "ClassicalRule",
        "DispatchDecision",
        "RuleContext",
        "dispatch_classical_rule",
        "dispatch_legacy_rule",
        "dispatch_paper_rule",
    ]
