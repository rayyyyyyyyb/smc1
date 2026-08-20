from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from smc_repro.config import ReproductionProfile, load_profile, profile_sha256
from smc_repro.environment import (
    CandidatePlan,
    EpisodeResult,
    SchedulingEnvironment,
    StepResult,
    _append_interval_bundle_transactionally,
)
from smc_repro.experiment_contract import (
    FAILURE_STREAM_VERSION,
    RunContract,
    build_run_contract,
    collect_git_commit,
    contract_sha256,
)
from smc_repro.observations import ScheduleObservation
from smc_repro.reliability import health_from_effective_age
from smc_repro.rewards import RewardMode
from smc_repro.rules import ClassicalRule, DispatchDecision
from smc_repro.schemas import (
    InstanceSpec,
    IntervalType,
    JobSpec,
    MachineSpec,
    OperationSpec,
    ScheduleInterval,
)
from smc_repro.seeding import keyed_uniform
from smc_repro.timeline import MachineTimeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
METADATA_KEYS = {
    "schema_version",
    "profile",
    "event_id",
    "decision_index",
    "rule_name",
    "selected_job_id",
    "selected_op_id",
    "health_before",
    "health_after",
    "effective_age_before",
    "effective_age_after",
    "failure_probability",
    "failure_draw_primary",
    "failure_draw_secondary",
    "pm_triggered",
    "cm_triggered",
    "nominal_processing_time",
    "degradation_factor",
}


def _profile(name: str) -> ReproductionProfile:
    return load_profile(CONFIG_ROOT / f"{name}.yaml")


def _single_job_instance(
    *,
    instance_id: str = "single",
    arrival: float = 0.0,
    nominal: float = 4.0,
    setup: float = 0.0,
    cm_duration: float = 6.0,
    eta: float = 1.0e12,
    beta: float = 2.0,
) -> InstanceSpec:
    return InstanceSpec(
        instance_id,
        101,
        202,
        (
            JobSpec(
                0,
                arrival,
                arrival + 100.0,
                1,
                (OperationSpec(0, 0, (nominal,)),),
            ),
        ),
        (MachineSpec(0, setup, cm_duration, eta=eta, beta=beta),),
    )


def _two_job_one_machine_instance(*, setup: float = 2.0) -> InstanceSpec:
    return InstanceSpec(
        "two-job-one-machine",
        303,
        404,
        (
            JobSpec(0, 0.0, 100.0, 1, (OperationSpec(0, 0, (2.0,)),)),
            JobSpec(1, 0.0, 100.0, 1, (OperationSpec(1, 0, (3.0,)),)),
        ),
        (MachineSpec(0, setup, 4.0, eta=1.0e12, beta=2.0),),
    )


def _small_policy_instance() -> InstanceSpec:
    return InstanceSpec(
        "policy-small",
        505,
        606,
        (
            JobSpec(
                0,
                0.0,
                80.0,
                1,
                (
                    OperationSpec(0, 0, (2.0, 3.0)),
                    OperationSpec(0, 1, (4.0, 2.0)),
                ),
            ),
            JobSpec(
                1,
                3.0,
                90.0,
                2,
                (
                    OperationSpec(1, 0, (3.0, 2.0)),
                    OperationSpec(1, 1, (2.0, 4.0)),
                ),
            ),
        ),
        (
            MachineSpec(0, 1.0, 4.0, eta=1.0e12, beta=2.0),
            MachineSpec(1, 1.0, 4.0, eta=1.0e12, beta=2.0),
        ),
    )


def _float_snapshot(value: float) -> str:
    return float(value).hex()


def _runtime_snapshot(environment: SchedulingEnvironment) -> tuple[object, ...]:
    runtime = environment.runtime
    machines = tuple(
        (
            id(machine),
            machine.machine_id,
            _float_snapshot(machine.health),
            _float_snapshot(machine.effective_age),
            _float_snapshot(machine.usage_time),
            _float_snapshot(machine.degradation_factor),
            machine.last_job_id,
            machine.pm_count,
            machine.cm_count,
            machine.process_count,
        )
        for machine in runtime.machines
    )
    return (
        id(runtime),
        id(runtime.instance),
        type(runtime.next_op_index),
        id(runtime.next_op_index),
        tuple(runtime.next_op_index),
        type(runtime.timelines),
        id(runtime.timelines),
        tuple(
            (id(timeline), timeline.machine_id, timeline.intervals)
            for timeline in runtime.timelines
        ),
        type(runtime.machines),
        id(runtime.machines),
        machines,
        type(runtime.last_machine_by_job),
        id(runtime.last_machine_by_job),
        tuple(runtime.last_machine_by_job),
        _float_snapshot(runtime.decision_time),
        runtime.decision_index,
        id(environment._previous_named_observation),
        environment._previous_named_observation,
    )


def _scan_seed(namespace: str, instance: InstanceSpec, *, below: float) -> int:
    for seed in range(100_000):
        draw = keyed_uniform(seed, namespace, instance.instance_id, 0, 0, 0)
        if draw < below:
            return seed
    raise AssertionError("bounded seed scan did not find a keyed draw")


def _run_composite_episode(
    profile_name: str,
    action_index: int,
) -> EpisodeResult:
    environment = SchedulingEnvironment(
        _small_policy_instance(),
        _profile(profile_name),
        policy_seed=action_index,
    )
    environment.reset()
    while not environment.is_done():
        environment.step_rule(action_index, RewardMode.TARDINESS)
    return environment.final_result()


def _init_git_repository(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Task 7 Test"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "task7@example.invalid"],
        cwd=path,
        check=True,
    )
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_public_result_records_are_frozen_and_candidate_fields_are_exact() -> None:
    candidate = CandidatePlan(0, 1, 2, 3.0, True, 4.0, 5.0, 6.0, 7.0, 17.0)

    assert tuple(field.name for field in dataclasses.fields(CandidatePlan)) == (
        "job_id",
        "op_id",
        "machine_id",
        "predecessor_end",
        "setup_required",
        "setup_duration",
        "process_nominal_duration",
        "process_estimated_duration",
        "earliest_start",
        "estimated_completion",
    )
    assert tuple(field.name for field in dataclasses.fields(StepResult)) == (
        "observation",
        "named_observation",
        "reward",
        "done",
        "decision",
        "emitted_intervals",
        "info",
    )
    assert tuple(field.name for field in dataclasses.fields(EpisodeResult)) == (
        "instance_id",
        "profile_name",
        "intervals",
        "validation",
        "metrics",
        "decisions",
    )
    with pytest.raises(FrozenInstanceError):
        candidate.earliest_start = 0.0


def test_one_job_one_machine_emits_one_exact_process_interval() -> None:
    instance = _single_job_instance()
    environment = SchedulingEnvironment(instance, _profile("paper_repro"), policy_seed=7)

    reset_vector, named = environment.reset()
    result = environment.step_rule(0, RewardMode.TARDINESS)

    assert reset_vector.shape == (6,)
    assert named == ScheduleObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.done
    assert [interval.interval_type for interval in result.emitted_intervals] == [
        IntervalType.PROCESS
    ]
    process = result.emitted_intervals[0]
    assert (process.machine_id, process.job_id, process.op_id, process.start) == (0, 0, 0, 0.0)
    assert process.end == pytest.approx(4.04)
    assert process.metadata["nominal_processing_time"] == 4.0
    assert process.metadata["degradation_factor"] == pytest.approx(1.01)
    with pytest.raises(TypeError):
        result.info["policy_seed"] = 99


def test_context_advances_to_arrival_exposes_only_next_operation_and_is_side_effect_free() -> None:
    instance = InstanceSpec(
        "ready-next-only",
        11,
        12,
        (
            JobSpec(
                0,
                5.0,
                50.0,
                1,
                (
                    OperationSpec(0, 0, (2.0, None)),
                    OperationSpec(0, 1, (3.0, 4.0)),
                ),
            ),
        ),
        (MachineSpec(0, 0.0, 4.0), MachineSpec(1, 0.0, 4.0)),
    )
    environment = SchedulingEnvironment(instance, _profile("paper_repro"), policy_seed=1)
    environment.reset()

    context = environment.build_rule_context()
    after_clock_advance = _runtime_snapshot(environment)
    repeated = environment.build_rule_context()

    assert environment.runtime.decision_time == 5.0
    assert [(job.job_id, job.op_id) for job in context.jobs] == [(0, 0)]
    assert [(pair.job_id, pair.op_id, pair.machine_id) for pair in context.pairs] == [(0, 0, 0)]
    assert repeated == context
    assert _runtime_snapshot(environment) == after_clock_advance


def test_candidate_estimator_uses_current_degradation_setup_predicate_and_tail() -> None:
    instance = _two_job_one_machine_instance(setup=2.0)
    environment = SchedulingEnvironment(instance, _profile("legacy_snapshot"), policy_seed=2)
    environment.reset()
    environment.runtime.timelines[0].add(
        ScheduleInterval(0, 10.0, 12.0, IntervalType.PM)
    )
    environment.runtime.machines[0].degradation_factor = 1.5
    environment.runtime.machines[0].last_job_id = 1
    before = _runtime_snapshot(environment)

    plan = environment._estimate_candidate(0, 0, 0)

    assert plan == CandidatePlan(0, 0, 0, 0.0, True, 2.0, 2.0, 3.0, 12.0, 17.0)
    assert _runtime_snapshot(environment) == before


def test_source_job_change_emits_setup_immediately_before_process_but_paper_emits_none() -> None:
    instance = _two_job_one_machine_instance()
    legacy = SchedulingEnvironment(instance, _profile("legacy_snapshot"), policy_seed=3)
    paper = SchedulingEnvironment(instance, _profile("paper_repro"), policy_seed=3)
    legacy.reset()
    paper.reset()

    legacy.step_rule(0, RewardMode.TARDINESS)
    legacy_second = legacy.step_rule(0, RewardMode.TARDINESS)
    paper.step_rule(0, RewardMode.TARDINESS)
    paper_second = paper.step_rule(0, RewardMode.TARDINESS)

    assert [item.interval_type for item in legacy_second.emitted_intervals] == [
        IntervalType.SETUP,
        IntervalType.PROCESS,
    ]
    assert legacy_second.emitted_intervals[0].end == legacy_second.emitted_intervals[1].start
    assert [item.interval_type for item in paper_second.emitted_intervals] == [
        IntervalType.PROCESS
    ]


@pytest.mark.parametrize("profile_name", ["legacy_snapshot", "paper_repro", "corrected_smc"])
def test_every_profile_appends_at_tail_without_retrospective_gap_insertion(
    profile_name: str,
) -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(cm_duration=4.0),
        _profile(profile_name),
        policy_seed=4,
    )
    environment.reset()
    environment.runtime.timelines[0].add(ScheduleInterval(0, 10.0, 12.0, IntervalType.PM))

    plan = environment._estimate_candidate(0, 0, 0)

    assert plan.earliest_start == 12.0


@pytest.mark.parametrize("profile_name", ["legacy_snapshot", "corrected_smc"])
def test_pm_threshold_emits_pm_and_restores_profile_state(profile_name: str) -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(),
        _profile(profile_name),
        policy_seed=5,
    )
    environment.reset()
    machine = environment.runtime.machines[0]
    machine.health = 20.0
    machine.usage_time = 100.0
    machine.effective_age = 100.0

    result = environment.step_rule(0, RewardMode.UTILIZATION)
    machine = environment.runtime.machines[0]

    assert [item.interval_type for item in result.emitted_intervals] == [
        IntervalType.PM,
        IntervalType.PROCESS,
    ]
    assert machine.pm_count == 1
    assert machine.cm_count == 0
    assert machine.usage_time == pytest.approx(result.emitted_intervals[-1].duration)
    assert machine.effective_age == pytest.approx(machine.usage_time)
    if profile_name == "corrected_smc":
        spec = environment.instance.machines[0]
        assert machine.health == pytest.approx(
            health_from_effective_age(machine.effective_age, spec.eta, spec.beta)
        )
    else:
        assert 92.0 <= machine.health <= 96.0


def test_forced_keyed_failure_emits_configured_cm_before_process() -> None:
    instance = _single_job_instance(eta=500.0)
    profile = replace(
        _profile("legacy_snapshot"),
        reliability=replace(_profile("legacy_snapshot").reliability, pm_enabled=False),
    )
    failure_seed = _scan_seed("failure_primary", instance, below=0.5)
    environment = SchedulingEnvironment(
        instance,
        profile,
        policy_seed=6,
        failure_seed=failure_seed,
    )
    environment.reset()
    machine = environment.runtime.machines[0]
    machine.health = 50.0
    machine.usage_time = 500.0
    machine.effective_age = 500.0

    result = environment.step_rule(0, RewardMode.TARDINESS)
    machine = environment.runtime.machines[0]

    assert [item.interval_type for item in result.emitted_intervals] == [
        IntervalType.CM,
        IntervalType.PROCESS,
    ]
    assert result.emitted_intervals[0].duration == instance.machines[0].cm_duration
    assert result.emitted_intervals[0].end == result.emitted_intervals[1].start
    assert machine.cm_count == 1


def test_corrected_interval_risk_increases_with_duration_while_legacy_cdf_does_not() -> None:
    instance = _single_job_instance(eta=100.0)
    legacy = SchedulingEnvironment(instance, _profile("legacy_snapshot"), policy_seed=7)
    corrected = SchedulingEnvironment(instance, _profile("corrected_smc"), policy_seed=7)
    legacy.reset()
    corrected.reset()
    for environment in (legacy, corrected):
        environment.runtime.machines[0].usage_time = 40.0
        environment.runtime.machines[0].effective_age = 40.0

    assert legacy._failure_probability(0, 1.0) == legacy._failure_probability(0, 20.0)
    assert corrected._failure_probability(0, 20.0) > corrected._failure_probability(0, 1.0)


def test_transactional_bundle_replacement_preserves_original_on_late_overlap() -> None:
    original = ScheduleInterval(0, 0.0, 2.0, IntervalType.PROCESS, 0, 0)
    timeline = MachineTimeline(0, (original,))
    first = ScheduleInterval(0, 2.0, 4.0, IntervalType.PM)
    second = ScheduleInterval(0, 3.0, 7.0, IntervalType.CM)

    with pytest.raises(ValueError, match="overlap"):
        _append_interval_bundle_transactionally(timeline, (first, second))

    assert timeline.intervals == (original,)


def test_failed_nonfinite_step_preserves_timeline_and_every_runtime_field() -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(),
        _profile("paper_repro"),
        policy_seed=8,
    )
    environment.reset()
    environment.runtime.machines[0].health = float("nan")
    before = _runtime_snapshot(environment)

    with pytest.raises(ValueError, match="finite"):
        environment.step_rule(0, RewardMode.TARDINESS)

    assert _runtime_snapshot(environment) == before


def test_failed_step_rolls_back_ready_job_clock_advancement() -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(arrival=5.0),
        _profile("paper_repro"),
        policy_seed=8,
    )
    environment.reset()
    environment.runtime.machines[0].health = float("nan")
    before = _runtime_snapshot(environment)

    with pytest.raises(ValueError, match="finite"):
        environment.step_rule(0, RewardMode.TARDINESS)

    assert _runtime_snapshot(environment) == before


def test_late_commit_failure_preserves_whole_live_environment_snapshot() -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(),
        _profile("paper_repro"),
        policy_seed=8,
    )
    environment.reset()
    environment.runtime.next_op_index = tuple(  # type: ignore[assignment]
        environment.runtime.next_op_index
    )
    before = _runtime_snapshot(environment)

    with pytest.raises((TypeError, ValueError), match="list|item assignment"):
        environment.step_rule(0, RewardMode.TARDINESS)

    assert _runtime_snapshot(environment) == before


@pytest.mark.parametrize(
    "attribute",
    ["next_op_index", "timelines", "machines", "last_machine_by_job"],
)
def test_runtime_rejects_non_list_state_containers(attribute: str) -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(),
        _profile("paper_repro"),
        policy_seed=8,
    )
    value = getattr(environment.runtime, attribute)
    setattr(environment.runtime, attribute, tuple(value))

    with pytest.raises(ValueError, match=rf"{attribute} must be a list"):
        environment.runtime.validate()


def test_unrepresentable_large_time_process_duration_aborts_whole_step() -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(arrival=1.0e16, nominal=4.0),
        _profile("paper_repro"),
        policy_seed=8,
    )
    environment.reset()
    before = _runtime_snapshot(environment)

    with pytest.raises(ValueError, match="PROCESS duration metadata mismatch"):
        environment.step_rule(0, RewardMode.TARDINESS)

    assert _runtime_snapshot(environment) == before


def test_final_operation_has_no_post_process_pm_or_cm() -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(),
        _profile("legacy_snapshot"),
        policy_seed=9,
    )
    environment.reset()

    step = environment.step_rule(0, RewardMode.TARDINESS)
    episode = environment.final_result()

    assert step.done
    assert episode.intervals[-1].interval_type is IntervalType.PROCESS
    assert not any(
        interval.interval_type in {IntervalType.PM, IntervalType.CM}
        and interval.start >= episode.intervals[-1].end
        for interval in episode.intervals
    )


@pytest.mark.parametrize("profile_name", ["legacy_snapshot", "paper_repro"])
@pytest.mark.parametrize("action_index", range(9))
def test_every_composite_action_completes_with_a_valid_schedule(
    profile_name: str,
    action_index: int,
) -> None:
    result = _run_composite_episode(profile_name, action_index)

    assert result.validation.ok
    assert result.decisions == 4


@pytest.mark.parametrize("rule", tuple(ClassicalRule))
def test_every_classical_rule_completes_with_a_valid_schedule(rule: ClassicalRule) -> None:
    environment = SchedulingEnvironment(
        _small_policy_instance(),
        _profile("corrected_smc"),
        policy_seed=10,
    )
    environment.reset()
    while not environment.is_done():
        environment._step_classical(rule, RewardMode.TARDINESS)

    result = environment.final_result()

    assert result.validation.ok
    assert result.decisions == 4


def test_reset_mode_changes_vector_only_and_legacy_first_reward_uses_zero_surrogate() -> None:
    instance = _single_job_instance(arrival=0.0, nominal=4.0)
    legacy = SchedulingEnvironment(instance, _profile("legacy_snapshot"), policy_seed=11)
    paper = SchedulingEnvironment(instance, _profile("paper_repro"), policy_seed=11)

    legacy_vector, legacy_named = legacy.reset()
    paper_vector, paper_named = paper.reset()
    legacy_step = legacy.step_rule(0, RewardMode.UTILIZATION)

    assert np.array_equal(legacy_vector, np.zeros(6, dtype=np.float32))
    assert legacy_named == paper_named
    assert np.array_equal(paper_vector, paper_named.vector(_profile("paper_repro").state.order))
    assert legacy_step.reward == 1


def test_complete_interval_bundle_has_exact_order_unique_event_ids_and_scalar_metadata() -> None:
    instance = InstanceSpec(
        "all-events",
        71,
        72,
        (
            JobSpec(0, 0.0, 100.0, 1, (OperationSpec(0, 0, (4.0,)),)),
            JobSpec(1, 0.0, 100.0, 1, (OperationSpec(1, 0, (4.0,)),)),
        ),
        (MachineSpec(0, 2.0, 6.0, eta=1.0, beta=2.0),),
    )
    failure_seed = _scan_seed("failure_primary", instance, below=0.99)
    environment = SchedulingEnvironment(
        instance,
        _profile("corrected_smc"),
        policy_seed=12,
        failure_seed=failure_seed,
    )
    environment.reset()
    machine = environment.runtime.machines[0]
    machine.last_job_id = 1
    machine.health = 20.0
    machine.effective_age = 10.0
    machine.usage_time = 10.0

    step = environment._step_decision(
        DispatchDecision(0, 0, 0, "metadata_probe"),
        RewardMode.TARDINESS,
    )

    assert [interval.interval_type for interval in step.emitted_intervals] == [
        IntervalType.SETUP,
        IntervalType.PM,
        IntervalType.CM,
        IntervalType.PROCESS,
    ]
    event_ids = [interval.metadata["event_id"] for interval in step.emitted_intervals]
    assert event_ids == [
        "all-events:d000000:setup",
        "all-events:d000000:pm",
        "all-events:d000000:cm",
        "all-events:d000000:process",
    ]
    assert len(event_ids) == len(set(event_ids))
    for interval in step.emitted_intervals:
        assert set(interval.metadata) == METADATA_KEYS
        assert all(
            value is None or isinstance(value, (str, int, float, bool))
            for value in interval.metadata.values()
        )
        json.dumps(dict(interval.metadata), sort_keys=True, allow_nan=False)


def test_identical_seeds_produce_identical_serialized_intervals_and_metrics() -> None:
    results = [_run_composite_episode("legacy_snapshot", 8) for _ in range(2)]

    def serialized(result: EpisodeResult) -> str:
        payload = [
            {
                "machine_id": interval.machine_id,
                "start": interval.start,
                "end": interval.end,
                "type": interval.interval_type.value,
                "job_id": interval.job_id,
                "op_id": interval.op_id,
                "metadata": dict(interval.metadata),
            }
            for interval in result.intervals
        ]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    assert serialized(results[0]) == serialized(results[1])
    assert results[0].metrics == results[1].metrics


def test_policy_seed_does_not_change_failure_draw_for_matching_selected_action() -> None:
    instance = _single_job_instance(eta=100.0)
    observed: list[float] = []
    for policy_seed in (1, 999):
        environment = SchedulingEnvironment(
            instance,
            _profile("paper_repro"),
            policy_seed=policy_seed,
            failure_seed=44,
        )
        environment.reset()
        environment.runtime.machines[0].usage_time = 20.0
        environment.runtime.machines[0].effective_age = 20.0
        result = environment._step_decision(
            DispatchDecision(0, 0, 0, "matching_action"),
            RewardMode.TARDINESS,
        )
        observed.append(float(result.emitted_intervals[-1].metadata["failure_draw_primary"]))

    assert observed[0] == observed[1]


@pytest.mark.parametrize("bad_action", [-1, 9, True, 1.0, "1"])
def test_invalid_rule_actions_fail_loudly_without_mutation(bad_action: object) -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(),
        _profile("paper_repro"),
        policy_seed=13,
    )
    environment.reset()
    before = _runtime_snapshot(environment)

    with pytest.raises(ValueError, match="action_index"):
        environment.step_rule(bad_action, RewardMode.TARDINESS)  # type: ignore[arg-type]

    assert _runtime_snapshot(environment) == before


def test_ineligible_and_repeated_decisions_and_step_after_done_fail_loudly() -> None:
    instance = InstanceSpec(
        "adversarial-actions",
        81,
        82,
        (
            JobSpec(0, 0.0, 100.0, 1, (OperationSpec(0, 0, (2.0, None)),)),
            JobSpec(1, 0.0, 100.0, 1, (OperationSpec(1, 0, (2.0, None)),)),
        ),
        (MachineSpec(0, 0.0, 4.0), MachineSpec(1, 0.0, 4.0)),
    )
    environment = SchedulingEnvironment(instance, _profile("paper_repro"), policy_seed=14)
    environment.reset()
    before = _runtime_snapshot(environment)

    with pytest.raises(ValueError, match="eligible"):
        environment._step_decision(
            DispatchDecision(0, 0, 1, "ineligible"),
            RewardMode.TARDINESS,
        )
    assert _runtime_snapshot(environment) == before

    environment._step_decision(
        DispatchDecision(0, 0, 0, "complete"),
        RewardMode.TARDINESS,
    )
    completed = _runtime_snapshot(environment)
    with pytest.raises(ValueError, match="next operation|completed"):
        environment._step_decision(
            DispatchDecision(0, 0, 0, "repeat"),
            RewardMode.TARDINESS,
        )
    assert _runtime_snapshot(environment) == completed
    environment._step_decision(
        DispatchDecision(1, 0, 0, "complete-second"),
        RewardMode.TARDINESS,
    )
    with pytest.raises(RuntimeError, match="complete"):
        environment.step_rule(0, RewardMode.TARDINESS)


def test_high_load_percentile_uses_latest_process_end_not_nonprocess_availability() -> None:
    instance = InstanceSpec(
        "high-load-process-only",
        91,
        92,
        (
            JobSpec(0, 0.0, 100.0, 1, (OperationSpec(0, 0, (10.0, None)),)),
            JobSpec(1, 0.0, 100.0, 1, (OperationSpec(1, 0, (None, 20.0)),)),
        ),
        (
            MachineSpec(0, 0.0, 180.0, eta=500.0, beta=2.0),
            MachineSpec(1, 0.0, 4.0, eta=500.0, beta=2.0),
        ),
    )
    environment = SchedulingEnvironment(instance, _profile("legacy_snapshot"), policy_seed=15)
    environment.reset()
    environment.runtime.timelines[0].add(
        ScheduleInterval(0, 0.0, 10.0, IntervalType.PROCESS, 0, 0)
    )
    environment.runtime.timelines[1].add(
        ScheduleInterval(1, 0.0, 20.0, IntervalType.PROCESS, 1, 0)
    )

    assert not environment._is_high_load(0)
    assert environment._is_high_load(1)
    environment.runtime.timelines[0].add(ScheduleInterval(0, 10.0, 100.0, IntervalType.PM))
    assert not environment._is_high_load(0)
    assert environment._is_high_load(1)


def test_final_result_rejects_incomplete_or_invalid_schedule_before_metrics() -> None:
    environment = SchedulingEnvironment(
        _single_job_instance(setup=1.0),
        _profile("paper_repro"),
        policy_seed=16,
    )
    environment.reset()
    with pytest.raises(RuntimeError, match="complete"):
        environment.final_result()

    environment.step_rule(0, RewardMode.TARDINESS)
    process_end = environment.runtime.timelines[0].available_time
    environment.runtime.timelines[0].add(
        ScheduleInterval(0, process_end, process_end + 1.0, IntervalType.SETUP)
    )
    with pytest.raises(ValueError, match="invalid"):
        environment.final_result()


def test_default_seed_offsets_are_distinct_recorded_and_info_is_scalar_frozen() -> None:
    instance = _single_job_instance()
    environment = SchedulingEnvironment(instance, _profile("paper_repro"), policy_seed=17)
    environment.reset()

    step = environment.step_rule(0, RewardMode.TARDINESS)

    offsets = (
        step.info["failure_seed_offset"],
        step.info["wear_seed_offset"],
        step.info["repair_seed_offset"],
    )
    assert len(set(offsets)) == 3
    assert step.info["failure_seed"] == instance.failure_seed + offsets[0]
    assert step.info["wear_seed"] == instance.failure_seed + offsets[1]
    assert step.info["repair_seed"] == instance.failure_seed + offsets[2]
    assert all(
        value is None or isinstance(value, (str, int, float, bool))
        for value in step.info.values()
    )
    with pytest.raises(TypeError):
        step.info["failure_seed"] = 0


def test_machine_timeline_transaction_replacement_is_atomic_and_defensive() -> None:
    first = ScheduleInterval(0, 0.0, 2.0, IntervalType.PROCESS, 0, 0)
    replacement = [first]
    timeline = MachineTimeline(0)

    timeline.replace_intervals_for_transaction(replacement)
    replacement.clear()

    assert timeline.intervals == (first,)
    with pytest.raises(ValueError, match="overlap"):
        timeline.replace_intervals_for_transaction(
            (
                first,
                ScheduleInterval(0, 1.0, 3.0, IntervalType.PM),
            )
        )
    assert timeline.intervals == (first,)


def test_run_contract_clean_dirty_gate_and_canonical_hash(tmp_path: Path) -> None:
    commit = _init_git_repository(tmp_path)
    profile = _profile("paper_repro")

    assert collect_git_commit(tmp_path) == commit
    contract = build_run_contract(
        tmp_path,
        profile,
        bank_manifest_sha256="a" * 64,
        method="paper_A1_B2_ECT",
        train_seed=101,
        policy_seed=202,
        environment_metadata_path=Path("artifacts/environment_5090_resolved.json"),
    )

    assert contract == RunContract(
        schema_version=1,
        git_commit=commit,
        profile_name="paper_repro",
        profile_sha256=profile_sha256(profile),
        bank_manifest_sha256="a" * 64,
        method="paper_A1_B2_ECT",
        train_seed=101,
        policy_seed=202,
        failure_stream_version=FAILURE_STREAM_VERSION,
        environment_metadata_path="artifacts/environment_5090_resolved.json",
    )
    assert len(contract_sha256(contract)) == 64
    assert contract_sha256(contract) == contract_sha256(contract)

    (tmp_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        collect_git_commit(tmp_path)
    assert collect_git_commit(tmp_path, allow_dirty=True) == commit


@pytest.mark.parametrize(
    "invalid_path",
    [
        Path(),
        Path("../environment.json"),
        Path("artifacts/../environment.json"),
        Path("C:/absolute/environment.json"),
        PurePosixPath("artifacts\\environment.json"),
    ],
)
def test_run_contract_rejects_noncanonical_environment_paths(
    tmp_path: Path,
    invalid_path: Path,
) -> None:
    _init_git_repository(tmp_path)

    with pytest.raises(ValueError, match="environment_metadata_path"):
        build_run_contract(
            tmp_path,
            _profile("paper_repro"),
            bank_manifest_sha256="b" * 64,
            method="paper_A1_B2_ECT",
            train_seed=1,
            policy_seed=2,
            environment_metadata_path=invalid_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("git_commit", "not-a-commit"),
        ("profile_name", "unknown_profile"),
        ("profile_sha256", "f" * 63),
        ("bank_manifest_sha256", "g" * 64),
        ("method", ""),
        ("train_seed", -1),
        ("policy_seed", True),
        ("failure_stream_version", "unknown"),
        ("environment_metadata_path", "../environment.json"),
    ],
)
def test_run_contract_record_rejects_invalid_fields(field: str, value: object) -> None:
    valid = RunContract(
        schema_version=1,
        git_commit="1" * 40,
        profile_name="paper_repro",
        profile_sha256="2" * 64,
        bank_manifest_sha256="3" * 64,
        method="paper_A1_B2_ECT",
        train_seed=1,
        policy_seed=2,
        failure_stream_version=FAILURE_STREAM_VERSION,
        environment_metadata_path="artifacts/environment.json",
    )

    with pytest.raises(ValueError):
        replace(valid, **{field: value})


def test_contract_hash_matches_independent_known_json_bytes_and_digest() -> None:
    base = RunContract(
        schema_version=1,
        git_commit="1" * 40,
        profile_name="paper_repro",
        profile_sha256="2" * 64,
        bank_manifest_sha256="3" * 64,
        method="paper_A1_B2_ECT",
        train_seed=1,
        policy_seed=2,
        failure_stream_version=FAILURE_STREAM_VERSION,
        environment_metadata_path="artifacts/environment.json",
    )
    expected_payload = (
        b'{"bank_manifest_sha256":"'
        b"3333333333333333333333333333333333333333333333333333333333333333"
        b'","environment_metadata_path":"artifacts/environment.json"'
        b',"failure_stream_version":"smc-crn1","git_commit":"'
        b"1111111111111111111111111111111111111111"
        b'","method":"paper_A1_B2_ECT","policy_seed":2'
        b',"profile_name":"paper_repro","profile_sha256":"'
        b"2222222222222222222222222222222222222222222222222222222222222222"
        b'","schema_version":1,"train_seed":1}'
    )
    observed_payload = json.dumps(
        dataclasses.asdict(base),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert observed_payload == expected_payload
    assert contract_sha256(base) == (
        "927247183abb7fdbac6e97fc00e40a1f7128c020775c1d3a01da897c01957239"
    )
