from __future__ import annotations

from smc_repro.rules.base import (
    DispatchDecision,
    JobRuleView,
    JobSelector,
    MachineSelector,
    RuleContext,
    argmin_job,
    keyed_choice,
    select_machine,
)

LEGACY_COMPOSITE_RULES: dict[int, tuple[JobSelector, MachineSelector, str]] = {
    0: (JobSelector.A1, MachineSelector.EARLIEST_COMPLETION, "legacy_A1_ECT"),
    1: (JobSelector.A1, MachineSelector.EARLIEST_START, "legacy_A1_EST"),
    2: (JobSelector.A1, MachineSelector.RANDOM, "legacy_A1_RANDOM"),
    3: (JobSelector.A2, MachineSelector.EARLIEST_COMPLETION, "legacy_A2_ECT"),
    4: (JobSelector.A2, MachineSelector.EARLIEST_START, "legacy_A2_EST"),
    5: (JobSelector.A2, MachineSelector.RANDOM, "legacy_A2_RANDOM"),
    6: (JobSelector.A3, MachineSelector.EARLIEST_COMPLETION, "legacy_A3_ECT"),
    7: (JobSelector.A3, MachineSelector.EARLIEST_START, "legacy_A3_EST"),
    8: (JobSelector.A3, MachineSelector.RANDOM, "legacy_A3_RANDOM"),
}


def _select_job(
    context: RuleContext,
    selector: JobSelector,
    action_index: int,
) -> JobRuleView:
    if selector is JobSelector.A1:
        return argmin_job(
            context.jobs,
            lambda job: job.completed_operation_count / job.operation_count,
        )
    if selector is JobSelector.A2:
        if any(job.due_date < job.decision_time for job in context.jobs):
            return argmin_job(
                context.jobs,
                lambda job: job.due_date - job.decision_time / (4 - job.urgency),
            )
        return argmin_job(
            context.jobs,
            lambda job: (
                job.latest_process_end + job.decision_time - job.due_date
            )
            / job.urgency,
        )
    if selector is JobSelector.A3:
        jobs = tuple(sorted(context.jobs, key=lambda job: job.job_id))
        selected = keyed_choice(
            jobs,
            context.policy_seed,
            "legacy_job",
            action_index,
            context.instance_id,
            context.decision_index,
        )
        if not isinstance(selected, JobRuleView):
            raise TypeError("keyed job selection returned an invalid object")
        return selected
    raise AssertionError(f"unsupported legacy job selector: {selector}")


def dispatch_legacy_rule(context: RuleContext, action_index: int) -> DispatchDecision:
    if type(action_index) is not int or not 0 <= action_index <= 8:
        raise ValueError("legacy action_index must be an integer in 0..8")
    job_selector, machine_selector, rule_name = LEGACY_COMPOSITE_RULES[action_index]

    job = _select_job(context, job_selector, action_index)
    pair = select_machine(
        context,
        job,
        machine_selector,
        namespace=f"legacy_action_{action_index}",
    )
    return DispatchDecision(
        job_id=job.job_id,
        op_id=job.op_id,
        machine_id=pair.machine_id,
        rule_name=rule_name,
    )
