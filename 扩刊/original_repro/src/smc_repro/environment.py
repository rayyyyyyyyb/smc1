from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from smc_repro.config import (
    FailureMode,
    InitialObservationMode,
    ReproductionProfile,
    RuleSetName,
    SetupMode,
    WearMode,
)
from smc_repro.metrics import ScheduleMetrics, compute_schedule_metrics
from smc_repro.observations import ScheduleObservation, compute_observation
from smc_repro.reliability import (
    health_from_effective_age,
    weibull_cdf,
    weibull_interval_failure_probability,
)
from smc_repro.rewards import RewardMode, transition_reward
from smc_repro.rules import (
    ClassicalRule,
    dispatch_classical_rule,
    dispatch_legacy_rule,
    dispatch_paper_rule,
)
from smc_repro.rules.base import (
    DispatchDecision,
    JobRuleView,
    PairRuleView,
    RuleContext,
)
from smc_repro.runtime import MachineRuntime, ScheduleRuntime, create_runtime
from smc_repro.schemas import (
    InstanceSpec,
    IntervalType,
    MetadataScalar,
    ScheduleInterval,
    _freeze_metadata,
)
from smc_repro.seeding import keyed_uniform
from smc_repro.timeline import MachineTimeline
from smc_repro.validator import ValidationReport, validate_schedule

FAILURE_SEED_OFFSET = 0
WEAR_SEED_OFFSET = 1
REPAIR_SEED_OFFSET = 2


@dataclass(frozen=True)
class CandidatePlan:
    job_id: int
    op_id: int
    machine_id: int
    predecessor_end: float
    setup_required: bool
    setup_duration: float
    process_nominal_duration: float
    process_estimated_duration: float
    earliest_start: float
    estimated_completion: float


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    named_observation: ScheduleObservation
    reward: int
    done: bool
    decision: DispatchDecision
    emitted_intervals: tuple[ScheduleInterval, ...]
    info: Mapping[str, str | int | float | bool | None]


@dataclass(frozen=True)
class EpisodeResult:
    instance_id: str
    profile_name: str
    intervals: tuple[ScheduleInterval, ...]
    validation: ValidationReport
    metrics: ScheduleMetrics
    decisions: int


def _append_interval_bundle_transactionally(
    timeline: MachineTimeline,
    intervals: tuple[ScheduleInterval, ...],
) -> None:
    replacement = MachineTimeline(timeline.machine_id, timeline.intervals)
    for interval in intervals:
        replacement.add(interval)
    timeline.replace_intervals_for_transaction(replacement.intervals)


class SchedulingEnvironment:
    def __init__(
        self,
        instance: InstanceSpec,
        profile: ReproductionProfile,
        *,
        policy_seed: int,
        failure_seed: int | None = None,
        wear_seed: int | None = None,
        repair_seed: int | None = None,
    ) -> None:
        self._validate_seed("policy_seed", policy_seed)
        for name, seed in (
            ("failure_seed", failure_seed),
            ("wear_seed", wear_seed),
            ("repair_seed", repair_seed),
        ):
            if seed is not None:
                self._validate_seed(name, seed)
        self.instance = instance
        self.profile = profile
        self.policy_seed = policy_seed
        self._failure_seed_offset: int | None = (
            FAILURE_SEED_OFFSET if failure_seed is None else None
        )
        self._wear_seed_offset: int | None = WEAR_SEED_OFFSET if wear_seed is None else None
        self._repair_seed_offset: int | None = (
            REPAIR_SEED_OFFSET if repair_seed is None else None
        )
        self.failure_seed = (
            instance.failure_seed + FAILURE_SEED_OFFSET
            if failure_seed is None
            else failure_seed
        )
        self.wear_seed = (
            instance.failure_seed + WEAR_SEED_OFFSET if wear_seed is None else wear_seed
        )
        self.repair_seed = (
            instance.failure_seed + REPAIR_SEED_OFFSET
            if repair_seed is None
            else repair_seed
        )
        self.runtime: ScheduleRuntime = create_runtime(instance)
        self._previous_named_observation = compute_observation(self.runtime, self.profile)

    @staticmethod
    def _validate_seed(name: str, value: int) -> None:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    def reset(self) -> tuple[np.ndarray, ScheduleObservation]:
        self.runtime = create_runtime(self.instance)
        named = compute_observation(self.runtime, self.profile)
        if self.profile.state.initial_observation is InitialObservationMode.ZERO:
            vector = np.zeros(6, dtype=np.float32)
            self._previous_named_observation = ScheduleObservation(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            vector = named.vector(self.profile.state.order)
            self._previous_named_observation = named
        return vector, named

    def is_done(self) -> bool:
        self.runtime.validate()
        return all(
            next_op_index == len(job.operations)
            for job, next_op_index in zip(
                self.instance.jobs,
                self.runtime.next_op_index,
                strict=True,
            )
        )

    def _ready_job_ids(self) -> tuple[int, ...]:
        unfinished = tuple(
            job.job_id
            for job, next_op_index in zip(
                self.instance.jobs,
                self.runtime.next_op_index,
                strict=True,
            )
            if next_op_index < len(job.operations)
        )
        ready = tuple(
            job_id
            for job_id in unfinished
            if self.instance.jobs[job_id].arrival_time <= self.runtime.decision_time
        )
        if ready or not unfinished:
            return ready
        self.runtime.decision_time = min(
            self.instance.jobs[job_id].arrival_time for job_id in unfinished
        )
        return tuple(
            job_id
            for job_id in unfinished
            if self.instance.jobs[job_id].arrival_time <= self.runtime.decision_time
        )

    def _process_intervals_for_job(self, job_id: int) -> tuple[ScheduleInterval, ...]:
        return tuple(
            interval
            for timeline in self.runtime.timelines
            for interval in timeline.intervals
            if interval.interval_type is IntervalType.PROCESS
            and interval.job_id == job_id
        )

    def _estimate_candidate(
        self,
        job_id: int,
        op_id: int,
        machine_id: int,
    ) -> CandidatePlan:
        job = self.instance.jobs[job_id]
        operation = job.operations[op_id]
        nominal = operation.processing_time(machine_id)
        machine_runtime = self.runtime.machines[machine_id]
        predecessor_end = 0.0
        if op_id > 0:
            predecessor_intervals = tuple(
                interval
                for interval in self._process_intervals_for_job(job_id)
                if interval.op_id == op_id - 1
            )
            if len(predecessor_intervals) != 1:
                raise ValueError("candidate operation requires exactly one predecessor")
            predecessor_end = predecessor_intervals[0].end
        previous_machine = self.runtime.last_machine_by_job[job_id]
        setup_required = (
            self.profile.scheduling.setup_mode is SetupMode.SOURCE_TOOL_CHANGE
            and (
                (previous_machine is not None and previous_machine != machine_id)
                or (
                    machine_runtime.last_job_id is not None
                    and machine_runtime.last_job_id != job_id
                )
            )
        )
        setup_duration = (
            self.instance.machines[machine_id].setup_time if setup_required else 0.0
        )
        estimated_duration = nominal * machine_runtime.degradation_factor
        earliest_start = max(
            job.arrival_time,
            predecessor_end,
            self.runtime.timelines[machine_id].available_time,
        )
        return CandidatePlan(
            job_id=job_id,
            op_id=op_id,
            machine_id=machine_id,
            predecessor_end=predecessor_end,
            setup_required=setup_required,
            setup_duration=setup_duration,
            process_nominal_duration=nominal,
            process_estimated_duration=estimated_duration,
            earliest_start=earliest_start,
            estimated_completion=(
                earliest_start + setup_duration + estimated_duration
            ),
        )

    def build_rule_context(self) -> RuleContext:
        if self.is_done():
            raise RuntimeError("episode is complete")
        ready_job_ids = self._ready_job_ids()
        jobs: list[JobRuleView] = []
        pairs: list[PairRuleView] = []
        for job_id in ready_job_ids:
            job = self.instance.jobs[job_id]
            op_id = self.runtime.next_op_index[job_id]
            operation = job.operations[op_id]
            process_intervals = self._process_intervals_for_job(job_id)
            processed_work = sum(interval.duration for interval in process_intervals)
            remaining_nominal_work = sum(
                sum(
                    duration
                    for duration in remaining_operation.proc_times
                    if duration is not None
                )
                / len(remaining_operation.eligible_machines)
                for remaining_operation in job.operations[op_id:]
            )
            total_work = processed_work + remaining_nominal_work
            next_mean = sum(
                duration for duration in operation.proc_times if duration is not None
            ) / len(operation.eligible_machines)
            jobs.append(
                JobRuleView(
                    job_id=job_id,
                    op_id=op_id,
                    arrival_time=job.arrival_time,
                    due_date=job.due_date,
                    urgency=job.urgency,
                    decision_time=self.runtime.decision_time,
                    latest_process_end=max(
                        (interval.end for interval in process_intervals),
                        default=0.0,
                    ),
                    operation_count=len(job.operations),
                    completed_operation_count=op_id,
                    completion_ratio_by_count=op_id / len(job.operations),
                    completion_ratio_by_work=(
                        processed_work / total_work if total_work > 0.0 else 0.0
                    ),
                    processed_work=processed_work,
                    remaining_nominal_work=remaining_nominal_work,
                    next_operation_mean_processing_time=next_mean,
                )
            )
            for machine_id in operation.eligible_machines:
                candidate = self._estimate_candidate(job_id, op_id, machine_id)
                pairs.append(
                    PairRuleView(
                        job_id=job_id,
                        op_id=op_id,
                        machine_id=machine_id,
                        earliest_start=candidate.earliest_start,
                        estimated_completion=candidate.estimated_completion,
                    )
                )
        return RuleContext(
            instance_id=self.instance.instance_id,
            decision_index=self.runtime.decision_index,
            policy_seed=self.policy_seed,
            jobs=tuple(jobs),
            pairs=tuple(pairs),
        )

    def step_rule(self, action_index: int, reward_mode: RewardMode) -> StepResult:
        if self.is_done():
            raise RuntimeError("episode is complete")
        if type(action_index) is not int or not 0 <= action_index <= 8:
            raise ValueError("action_index must be an integer in 0..8")
        decision_time_before = self.runtime.decision_time
        try:
            context = self.build_rule_context()
            if self.profile.scheduling.rule_set is RuleSetName.LEGACY:
                decision = dispatch_legacy_rule(context, action_index)
            elif self.profile.scheduling.rule_set is RuleSetName.PAPER:
                decision = dispatch_paper_rule(context, action_index)
            else:
                raise AssertionError(
                    f"unsupported rule set: {self.profile.scheduling.rule_set}"
                )
            return self._step_decision(
                decision,
                reward_mode,
                action_index=action_index,
                context=context,
            )
        except Exception:
            self.runtime.decision_time = decision_time_before
            raise

    def _step_classical(
        self,
        rule: ClassicalRule,
        reward_mode: RewardMode,
    ) -> StepResult:
        if self.is_done():
            raise RuntimeError("episode is complete")
        if type(rule) is not ClassicalRule:
            raise ValueError("rule must be a ClassicalRule")
        decision_time_before = self.runtime.decision_time
        try:
            context = self.build_rule_context()
            return self._step_decision(
                dispatch_classical_rule(context, rule),
                reward_mode,
                context=context,
            )
        except Exception:
            self.runtime.decision_time = decision_time_before
            raise

    def _step_decision(
        self,
        decision: DispatchDecision,
        reward_mode: RewardMode,
        *,
        action_index: int | None = None,
        context: RuleContext | None = None,
    ) -> StepResult:
        if self.is_done():
            raise RuntimeError("episode is complete")
        if type(decision) is not DispatchDecision:
            raise TypeError("decision must be a DispatchDecision")
        if type(reward_mode) is not RewardMode:
            raise ValueError("reward_mode must be a RewardMode")
        for name, value in (
            ("job_id", decision.job_id),
            ("op_id", decision.op_id),
            ("machine_id", decision.machine_id),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"decision {name} must be a non-negative integer")
        if not isinstance(decision.rule_name, str) or not decision.rule_name:
            raise ValueError("decision rule_name must be a non-empty string")
        if context is None:
            decision_time_before = self.runtime.decision_time
            try:
                active_context = self.build_rule_context()
            finally:
                self.runtime.decision_time = decision_time_before
        else:
            active_context = context
        ready_keys = {(job.job_id, job.op_id) for job in active_context.jobs}
        selected_key = (decision.job_id, decision.op_id)
        if selected_key not in ready_keys:
            raise ValueError(
                "selected job operation is not the next operation or the job is completed"
            )
        legal_pairs = {
            (pair.job_id, pair.op_id, pair.machine_id) for pair in active_context.pairs
        }
        if (decision.job_id, decision.op_id, decision.machine_id) not in legal_pairs:
            raise ValueError("selected machine is not eligible for the operation")

        candidate = self._estimate_candidate(
            decision.job_id,
            decision.op_id,
            decision.machine_id,
        )
        machine_id = decision.machine_id
        machine_spec = self.instance.machines[machine_id]
        current_machine = self.runtime.machines[machine_id]
        self._validate_machine_state(current_machine)
        health_before = current_machine.health
        age_before = current_machine.effective_age
        health = health_before
        effective_age = age_before
        usage_time = current_machine.usage_time
        cursor = candidate.earliest_start
        raw_intervals: list[tuple[IntervalType, float, float]] = []

        if candidate.setup_required and candidate.setup_duration > 0.0:
            setup_end = cursor + candidate.setup_duration
            raw_intervals.append((IntervalType.SETUP, cursor, setup_end))
            cursor = setup_end

        initial_risk = self._failure_probability_values(
            machine_id,
            effective_age,
            usage_time,
            candidate.process_estimated_duration,
        )
        pm_triggered = self.profile.reliability.pm_enabled and (
            initial_risk > self.profile.reliability.pm_failure_threshold
            or health < self.profile.reliability.pm_health_threshold
        )
        if pm_triggered:
            pm_end = cursor + machine_spec.pm_duration
            raw_intervals.append((IntervalType.PM, cursor, pm_end))
            cursor = pm_end
            health = 100.0
            effective_age = 0.0
            usage_time = 0.0

        process_degradation = self._degradation_factor(health)
        estimated_actual_duration = (
            candidate.process_nominal_duration * process_degradation
        )
        failure_probability = self._failure_probability_values(
            machine_id,
            effective_age,
            usage_time,
            estimated_actual_duration,
        )
        failure_draw_primary = keyed_uniform(
            self.failure_seed,
            "failure_primary",
            self.instance.instance_id,
            decision.job_id,
            decision.op_id,
            machine_id,
        )
        failure_draw_secondary: float | None = None
        if (
            self.profile.reliability.high_load_failure_bias
            and self._is_high_load(machine_id)
        ):
            failure_draw_secondary = keyed_uniform(
                self.failure_seed,
                "failure_secondary",
                self.instance.instance_id,
                decision.job_id,
                decision.op_id,
                machine_id,
            )
            cm_triggered = (
                min(failure_draw_primary, failure_draw_secondary)
                < failure_probability
            )
        else:
            cm_triggered = failure_draw_primary < failure_probability

        if cm_triggered:
            cm_end = cursor + machine_spec.cm_duration
            raw_intervals.append((IntervalType.CM, cursor, cm_end))
            cursor = cm_end
            if self.profile.reliability.wear_mode is WearMode.LEGACY_PER_OPERATION:
                recovery = 20.0 + 20.0 * keyed_uniform(
                    self.repair_seed,
                    "cm_recovery",
                    self.instance.instance_id,
                    decision.job_id,
                    decision.op_id,
                    machine_id,
                )
                health = min(90.0, health + recovery)
                usage_time *= self.profile.reliability.cm_age_repair_factor
                effective_age = usage_time
            elif self.profile.reliability.wear_mode is WearMode.EFFECTIVE_AGE:
                effective_age *= self.profile.reliability.cm_age_repair_factor
                usage_time = effective_age
                health = health_from_effective_age(
                    effective_age,
                    machine_spec.eta,
                    machine_spec.beta,
                )
            else:
                raise AssertionError(
                    f"unsupported wear mode: {self.profile.reliability.wear_mode}"
                )
            process_degradation = self._degradation_factor(health)

        process_duration = candidate.process_nominal_duration * process_degradation
        process_end = cursor + process_duration
        raw_intervals.append((IntervalType.PROCESS, cursor, process_end))

        if self.profile.reliability.wear_mode is WearMode.LEGACY_PER_OPERATION:
            usage_time += process_duration
            wear = 4.0 + 4.0 * keyed_uniform(
                self.wear_seed,
                "wear",
                self.instance.instance_id,
                decision.job_id,
                decision.op_id,
                machine_id,
            )
            health = min(100.0, max(0.0, health - wear))
            effective_age = usage_time
        elif self.profile.reliability.wear_mode is WearMode.EFFECTIVE_AGE:
            effective_age += process_duration
            usage_time = effective_age
            health = health_from_effective_age(
                effective_age,
                machine_spec.eta,
                machine_spec.beta,
            )
        else:
            raise AssertionError(
                f"unsupported wear mode: {self.profile.reliability.wear_mode}"
            )
        next_degradation = self._degradation_factor(health)
        self._validate_local_state(
            health,
            effective_age,
            usage_time,
            next_degradation,
        )

        metadata_base: dict[str, MetadataScalar] = {
            "schema_version": self.profile.schema_version,
            "profile": self.profile.profile.value,
            "decision_index": self.runtime.decision_index,
            "rule_name": decision.rule_name,
            "selected_job_id": decision.job_id,
            "selected_op_id": decision.op_id,
            "health_before": health_before,
            "health_after": health,
            "effective_age_before": age_before,
            "effective_age_after": effective_age,
            "failure_probability": failure_probability,
            "failure_draw_primary": failure_draw_primary,
            "failure_draw_secondary": failure_draw_secondary,
            "pm_triggered": pm_triggered,
            "cm_triggered": cm_triggered,
            "nominal_processing_time": candidate.process_nominal_duration,
            "degradation_factor": process_degradation,
        }
        emitted: list[ScheduleInterval] = []
        for interval_type, start, end in raw_intervals:
            metadata = dict(metadata_base)
            metadata["event_id"] = (
                f"{self.instance.instance_id}:d{self.runtime.decision_index:06d}:"
                f"{interval_type.value.lower()}"
            )
            if interval_type is IntervalType.PROCESS:
                emitted.append(
                    ScheduleInterval(
                        machine_id,
                        start,
                        end,
                        interval_type,
                        decision.job_id,
                        decision.op_id,
                        metadata,
                    )
                )
            else:
                emitted.append(
                    ScheduleInterval(
                        machine_id,
                        start,
                        end,
                        interval_type,
                        metadata=metadata,
                    )
                )
        emitted_intervals = tuple(emitted)

        proposed_timelines = [
            MachineTimeline(timeline.machine_id, timeline.intervals)
            for timeline in self.runtime.timelines
        ]
        _append_interval_bundle_transactionally(
            proposed_timelines[machine_id],
            emitted_intervals,
        )
        proposed_schedule = tuple(
            interval
            for timeline in proposed_timelines
            for interval in timeline.intervals
        )
        validation = validate_schedule(
            self.instance,
            proposed_schedule,
            require_complete=False,
        )
        if not validation.ok:
            raise ValueError(
                "invalid transactional interval bundle:\n" + "\n".join(validation.errors)
            )

        proposed_machines = [self._clone_machine(machine) for machine in self.runtime.machines]
        proposed_machine = proposed_machines[machine_id]
        proposed_machine.health = health
        proposed_machine.effective_age = effective_age
        proposed_machine.usage_time = usage_time
        proposed_machine.degradation_factor = next_degradation
        proposed_machine.last_job_id = decision.job_id
        proposed_machine.pm_count += int(pm_triggered)
        proposed_machine.cm_count += int(cm_triggered)
        proposed_machine.process_count += 1
        proposed_next_op_index = list(self.runtime.next_op_index)
        proposed_next_op_index[decision.job_id] += 1
        proposed_last_machine = list(self.runtime.last_machine_by_job)
        proposed_last_machine[decision.job_id] = machine_id
        proposed_runtime = ScheduleRuntime(
            instance=self.instance,
            next_op_index=proposed_next_op_index,
            timelines=proposed_timelines,
            machines=proposed_machines,
            last_machine_by_job=proposed_last_machine,
            decision_time=min(timeline.available_time for timeline in proposed_timelines),
            decision_index=self.runtime.decision_index + 1,
        )
        current_named = compute_observation(proposed_runtime, self.profile)
        current_vector = current_named.vector(self.profile.state.order)
        reward = transition_reward(
            self.profile,
            reward_mode,
            self._previous_named_observation,
            current_named,
        )
        done = all(
            next_op_index == len(job.operations)
            for job, next_op_index in zip(
                self.instance.jobs,
                proposed_next_op_index,
                strict=True,
            )
        )
        info = _freeze_metadata(
            {
                "schema_version": self.profile.schema_version,
                "instance_id": self.instance.instance_id,
                "profile": self.profile.profile.value,
                "decision_index": self.runtime.decision_index,
                "action_index": action_index,
                "rule_name": decision.rule_name,
                "policy_seed": self.policy_seed,
                "failure_seed": self.failure_seed,
                "wear_seed": self.wear_seed,
                "repair_seed": self.repair_seed,
                "failure_seed_offset": self._failure_seed_offset,
                "wear_seed_offset": self._wear_seed_offset,
                "repair_seed_offset": self._repair_seed_offset,
            }
        )

        result = StepResult(
            observation=current_vector,
            named_observation=current_named,
            reward=reward,
            done=done,
            decision=decision,
            emitted_intervals=emitted_intervals,
            info=info,
        )
        self.runtime = proposed_runtime
        self._previous_named_observation = current_named
        return result

    def final_result(self) -> EpisodeResult:
        if not self.is_done():
            raise RuntimeError("episode is not complete")
        intervals = tuple(
            interval
            for timeline in self.runtime.timelines
            for interval in timeline.intervals
        )
        validation = validate_schedule(
            self.instance,
            intervals,
            require_complete=True,
        )
        if not validation.ok:
            raise ValueError("invalid completed schedule:\n" + "\n".join(validation.errors))
        metrics = compute_schedule_metrics(self.instance, intervals)
        return EpisodeResult(
            instance_id=self.instance.instance_id,
            profile_name=self.profile.profile.value,
            intervals=intervals,
            validation=validation,
            metrics=metrics,
            decisions=self.runtime.decision_index,
        )

    def _failure_probability(self, machine_id: int, process_duration: float) -> float:
        machine = self.runtime.machines[machine_id]
        return self._failure_probability_values(
            machine_id,
            machine.effective_age,
            machine.usage_time,
            process_duration,
        )

    def _is_high_load(self, machine_id: int) -> bool:
        if type(machine_id) is not int or not 0 <= machine_id < len(self.instance.machines):
            raise ValueError("machine_id must reference a known machine")
        latest_process_ends = [
            max(
                (
                    interval.end
                    for interval in timeline.intervals
                    if interval.interval_type is IntervalType.PROCESS
                ),
                default=0.0,
            )
            for timeline in self.runtime.timelines
        ]
        threshold = float(np.percentile(np.asarray(latest_process_ends), 90.0))
        return latest_process_ends[machine_id] >= threshold

    def _failure_probability_values(
        self,
        machine_id: int,
        effective_age: float,
        usage_time: float,
        process_duration: float,
    ) -> float:
        if type(machine_id) is not int or not 0 <= machine_id < len(self.instance.machines):
            raise ValueError("machine_id must reference a known machine")
        if not all(
            math.isfinite(value)
            for value in (effective_age, usage_time, process_duration)
        ):
            raise ValueError("reliability state and process duration must be finite")
        if effective_age < 0.0 or usage_time < 0.0 or process_duration < 0.0:
            raise ValueError("reliability state and process duration must be non-negative")
        machine = self.instance.machines[machine_id]
        if self.profile.reliability.failure_mode is FailureMode.LEGACY_PRESTART_CDF:
            return weibull_cdf(usage_time, machine.eta, machine.beta)
        if (
            self.profile.reliability.failure_mode
            is FailureMode.PRESTART_CONDITIONAL_INTERVAL_RISK
        ):
            return weibull_interval_failure_probability(
                effective_age,
                process_duration,
                machine.eta,
                machine.beta,
            )
        raise AssertionError(
            f"unsupported failure mode: {self.profile.reliability.failure_mode}"
        )

    @staticmethod
    def _degradation_factor(health: float) -> float:
        if not math.isfinite(health):
            raise ValueError("machine health must be finite")
        if not 0.0 <= health <= 100.0:
            raise ValueError("machine health must be in [0, 100]")
        return 1.0 + round(0.01 * math.exp(0.05 * (100.0 - round(health, 1))), 2)

    @staticmethod
    def _validate_machine_state(machine: MachineRuntime) -> None:
        SchedulingEnvironment._validate_local_state(
            machine.health,
            machine.effective_age,
            machine.usage_time,
            machine.degradation_factor,
        )

    @staticmethod
    def _validate_local_state(
        health: float,
        effective_age: float,
        usage_time: float,
        degradation_factor: float,
    ) -> None:
        if not all(
            math.isfinite(value)
            for value in (health, effective_age, usage_time, degradation_factor)
        ):
            raise ValueError("machine runtime floating-point fields must be finite")
        if not 0.0 <= health <= 100.0:
            raise ValueError("machine health must be in [0, 100]")
        if effective_age < 0.0 or usage_time < 0.0:
            raise ValueError("machine ages must be non-negative")
        if degradation_factor <= 0.0:
            raise ValueError("machine degradation factor must be positive")

    @staticmethod
    def _clone_machine(machine: MachineRuntime) -> MachineRuntime:
        return MachineRuntime(
            machine_id=machine.machine_id,
            health=machine.health,
            effective_age=machine.effective_age,
            usage_time=machine.usage_time,
            degradation_factor=machine.degradation_factor,
            last_job_id=machine.last_job_id,
            pm_count=machine.pm_count,
            cm_count=machine.cm_count,
            process_count=machine.process_count,
        )
