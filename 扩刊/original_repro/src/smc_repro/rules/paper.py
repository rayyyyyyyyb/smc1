from __future__ import annotations

from smc_repro.rules.base import (
    DispatchDecision,
    JobRuleView,
    JobSelector,
    MachineSelector,
    RuleContext,
    argmax_job,
    argmin_job,
    keyed_choice,
    select_machine,
)

PAPER_COMPOSITE_RULES: dict[int, tuple[JobSelector, MachineSelector, str]] = {
    0: (JobSelector.A1, MachineSelector.EARLIEST_START, "paper_A1_B1_EST"),
    1: (JobSelector.A1, MachineSelector.EARLIEST_COMPLETION, "paper_A1_B2_ECT"),
    2: (JobSelector.A1, MachineSelector.RANDOM, "paper_A1_B3_RANDOM"),
    3: (JobSelector.A2, MachineSelector.EARLIEST_START, "paper_A2_B1_EST"),
    4: (JobSelector.A2, MachineSelector.EARLIEST_COMPLETION, "paper_A2_B2_ECT"),
    5: (JobSelector.A2, MachineSelector.RANDOM, "paper_A2_B3_RANDOM"),
    6: (JobSelector.A3, MachineSelector.EARLIEST_START, "paper_A3_B1_EST"),
    7: (JobSelector.A3, MachineSelector.EARLIEST_COMPLETION, "paper_A3_B2_ECT"),
    8: (JobSelector.A3, MachineSelector.RANDOM, "paper_A3_B3_RANDOM"),
}


def _select_job(
    context: RuleContext,
    selector: JobSelector,
    action_index: int,
) -> JobRuleView:
    if selector is JobSelector.A1:
        return argmin_job(
            context.jobs,
            lambda job: job.completion_ratio_by_work / (4 - job.urgency),
        )
    if selector is JobSelector.A2:
        if any(job.due_date < job.decision_time for job in context.jobs):
            return argmax_job(
                context.jobs,
                lambda job: (job.decision_time - job.due_date) / job.urgency,
            )
        return argmin_job(
            context.jobs,
            lambda job: (job.due_date - job.processed_work)
            / max(job.remaining_nominal_work, 1e-12),
        )
    if selector is JobSelector.A3:
        jobs = tuple(sorted(context.jobs, key=lambda job: job.job_id))
        selected = keyed_choice(
            jobs,
            context.policy_seed,
            "paper_job",
            action_index,
            context.instance_id,
            context.decision_index,
        )
        if not isinstance(selected, JobRuleView):
            raise TypeError("keyed job selection returned an invalid object")
        return selected
    raise AssertionError(f"unsupported paper job selector: {selector}")


def dispatch_paper_rule(context: RuleContext, action_index: int) -> DispatchDecision:
    if type(action_index) is not int or not 0 <= action_index <= 8:
        raise ValueError("paper action_index must be an integer in 0..8")
    job_selector, machine_selector, rule_name = PAPER_COMPOSITE_RULES[action_index]

    job = _select_job(context, job_selector, action_index)
    pair = select_machine(
        context,
        job,
        machine_selector,
        namespace=f"paper_action_{action_index}",
    )
    return DispatchDecision(
        job_id=job.job_id,
        op_id=job.op_id,
        machine_id=pair.machine_id,
        rule_name=rule_name,
    )
