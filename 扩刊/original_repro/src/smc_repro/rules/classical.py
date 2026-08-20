from __future__ import annotations

from enum import StrEnum

from smc_repro.rules.base import (
    DispatchDecision,
    MachineSelector,
    RuleContext,
    argmax_job,
    argmin_job,
    select_machine,
)


class ClassicalRule(StrEnum):
    FIFO_ECT = "FIFO+ECT"
    EDD_ECT = "EDD+ECT"
    MRT_ECT = "MRT+ECT"
    SPT_ECT = "SPT+ECT"
    LPT_ECT = "LPT+ECT"


def dispatch_classical_rule(
    context: RuleContext,
    rule: ClassicalRule,
) -> DispatchDecision:
    if rule is ClassicalRule.FIFO_ECT:
        job = argmin_job(context.jobs, lambda item: item.arrival_time)
    elif rule is ClassicalRule.EDD_ECT:
        job = argmin_job(context.jobs, lambda item: item.due_date)
    elif rule is ClassicalRule.MRT_ECT:
        job = argmax_job(context.jobs, lambda item: item.remaining_nominal_work)
    elif rule is ClassicalRule.SPT_ECT:
        job = argmin_job(
            context.jobs,
            lambda item: item.next_operation_mean_processing_time,
        )
    elif rule is ClassicalRule.LPT_ECT:
        job = argmax_job(
            context.jobs,
            lambda item: item.next_operation_mean_processing_time,
        )
    else:
        raise AssertionError(f"unsupported classical rule: {rule}")

    pair = select_machine(
        context,
        job,
        MachineSelector.EARLIEST_COMPLETION,
        namespace=f"classical_{rule.value}",
    )
    return DispatchDecision(
        job_id=job.job_id,
        op_id=job.op_id,
        machine_id=pair.machine_id,
        rule_name=rule.value,
    )
