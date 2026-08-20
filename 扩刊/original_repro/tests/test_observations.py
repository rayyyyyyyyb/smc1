from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from smc_repro.config import ReproductionProfile, load_profile
from smc_repro.observations import ScheduleObservation, compute_observation
from smc_repro.runtime import ScheduleRuntime, create_runtime
from smc_repro.schemas import (
    InstanceSpec,
    IntervalType,
    JobSpec,
    MachineSpec,
    OperationSpec,
    ScheduleInterval,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _single_job_instance(*, due_date: float = 1.0) -> InstanceSpec:
    return InstanceSpec(
        "single",
        11,
        12,
        (
            JobSpec(
                0,
                0.0,
                due_date,
                1,
                (OperationSpec(0, 0, (4.0,)),),
            ),
        ),
        (MachineSpec(0, 0.0, 2.0),),
    )


def _two_job_instance() -> InstanceSpec:
    return InstanceSpec(
        "two-job",
        21,
        22,
        (
            JobSpec(
                0,
                0.0,
                4.0,
                1,
                (
                    OperationSpec(0, 0, (2.0, None)),
                    OperationSpec(0, 1, (2.0, 4.0)),
                ),
            ),
            JobSpec(
                1,
                1.0,
                4.0,
                2,
                (OperationSpec(1, 0, (None, 3.0)),),
            ),
        ),
        (MachineSpec(0, 0.0, 2.0), MachineSpec(1, 0.0, 2.0)),
    )


def _profile(name: str) -> ReproductionProfile:
    return load_profile(CONFIG_ROOT / f"{name}.yaml")


def test_create_runtime_has_contiguous_mutable_state_and_stable_instance() -> None:
    instance = _two_job_instance()

    runtime = create_runtime(instance)

    assert runtime.instance is instance
    assert runtime.next_op_index == [0, 0]
    assert [timeline.machine_id for timeline in runtime.timelines] == [0, 1]
    assert [machine.machine_id for machine in runtime.machines] == [0, 1]
    assert runtime.last_machine_by_job == [None, None]
    assert runtime.decision_time == 0.0
    assert runtime.decision_index == 0

    runtime.next_op_index[0] = 1
    runtime.machines[0].health = 75.0
    runtime.last_machine_by_job[0] = 0
    assert runtime.next_op_index[0] == 1
    assert runtime.machines[0].health == 75.0
    assert runtime.last_machine_by_job[0] == 0

    with pytest.raises(AttributeError, match="instance"):
        runtime.instance = _single_job_instance()


def test_empty_runtime_returns_finite_environment_observation_not_legacy_zero() -> None:
    runtime = create_runtime(_single_job_instance(due_date=1.0))

    observation = compute_observation(runtime, _profile("legacy_snapshot"))
    vector = observation.vector(_profile("legacy_snapshot").state.order)

    assert np.all(np.isfinite(vector))
    assert observation == ScheduleObservation(
        crj_ave=0.0,
        crj_std=0.0,
        u_ave=0.0,
        u_std=0.0,
        tr_ave=0.75,
        tr_std=0.0,
    )
    assert not np.array_equal(vector, np.zeros(6, dtype=np.float32))


def test_observation_matches_hand_calculated_job_and_machine_values() -> None:
    runtime = create_runtime(_two_job_instance())
    runtime.next_op_index[0] = 1
    runtime.last_machine_by_job[0] = 0
    runtime.timelines[0].add(
        ScheduleInterval(0, 1.0, 3.0, IntervalType.PROCESS, 0, 0)
    )

    legacy = compute_observation(runtime, _profile("paper_repro"))
    corrected = compute_observation(runtime, _profile("corrected_smc"))

    assert legacy.crj_ave == pytest.approx(0.2)
    assert legacy.crj_std == pytest.approx(0.2)
    assert legacy.u_ave == pytest.approx(1.0 / 3.0)
    assert legacy.u_std == pytest.approx(1.0 / 3.0)
    assert legacy.tr_ave == pytest.approx(0.1)
    assert legacy.tr_std == pytest.approx(0.1)
    assert corrected.crj_ave == pytest.approx(0.2)
    assert corrected.crj_std == pytest.approx(0.2)
    assert corrected.u_ave == pytest.approx(1.0 / 3.0)
    assert corrected.u_std == pytest.approx(1.0 / 3.0)
    assert corrected.tr_ave == pytest.approx(0.2)
    assert corrected.tr_std == pytest.approx(0.2)


def test_legacy_and_paper_orders_permute_the_same_named_values() -> None:
    runtime = create_runtime(_two_job_instance())
    observation = compute_observation(runtime, _profile("paper_repro"))
    legacy_order = _profile("legacy_snapshot").state.order
    paper_order = _profile("paper_repro").state.order

    legacy_vector = observation.vector(legacy_order)
    paper_vector = observation.vector(paper_order)

    assert legacy_order != paper_order
    assert {
        name: float(legacy_vector[index]) for index, name in enumerate(legacy_order)
    } == pytest.approx(
        {name: float(paper_vector[index]) for index, name in enumerate(paper_order)}
    )


def test_inserted_wait_leaves_workload_pressure_but_increases_projected_tr() -> None:
    instance = InstanceSpec(
        "wait",
        31,
        32,
        (
            JobSpec(
                0,
                0.0,
                8.0,
                1,
                (
                    OperationSpec(0, 0, (2.0,)),
                    OperationSpec(0, 1, (2.0,)),
                ),
            ),
        ),
        (MachineSpec(0, 0.0, 2.0),),
    )
    early = create_runtime(instance)
    delayed = create_runtime(instance)
    for runtime, start in ((early, 0.0), (delayed, 10.0)):
        runtime.next_op_index[0] = 1
        runtime.timelines[0].add(
            ScheduleInterval(0, start, start + 2.0, IntervalType.PROCESS, 0, 0)
        )

    early_legacy = compute_observation(early, _profile("paper_repro"))
    delayed_legacy = compute_observation(delayed, _profile("paper_repro"))
    early_corrected = compute_observation(early, _profile("corrected_smc"))
    delayed_corrected = compute_observation(delayed, _profile("corrected_smc"))

    assert early_legacy.tr_ave == 0.0
    assert delayed_legacy.tr_ave == early_legacy.tr_ave
    assert early_corrected.tr_ave == 0.0
    assert delayed_corrected.tr_ave == pytest.approx(1.5)


def test_paper_and_standard_utilization_differ_on_two_machine_schedule() -> None:
    instance = InstanceSpec(
        "utilization",
        41,
        42,
        (
            JobSpec(0, 0.0, 10.0, 1, (OperationSpec(0, 0, (2.0, None)),)),
            JobSpec(1, 0.0, 10.0, 1, (OperationSpec(1, 0, (None, 2.0)),)),
        ),
        (MachineSpec(0, 0.0, 2.0), MachineSpec(1, 0.0, 2.0)),
    )
    runtime = create_runtime(instance)
    runtime.next_op_index[:] = [1, 1]
    runtime.timelines[0].add(
        ScheduleInterval(0, 0.0, 2.0, IntervalType.PROCESS, 0, 0)
    )
    runtime.timelines[1].add(
        ScheduleInterval(1, 4.0, 6.0, IntervalType.PROCESS, 1, 0)
    )

    paper = compute_observation(runtime, _profile("paper_repro"))
    standard = compute_observation(runtime, _profile("corrected_smc"))

    assert paper.u_ave == pytest.approx(2.0 / 3.0)
    assert paper.u_std == pytest.approx(1.0 / 3.0)
    assert standard.u_ave == pytest.approx(1.0 / 3.0)
    assert standard.u_std == 0.0


def test_singleton_population_standard_deviations_are_zero() -> None:
    observation = compute_observation(
        create_runtime(_single_job_instance(due_date=10.0)),
        _profile("corrected_smc"),
    )

    assert observation.crj_std == 0.0
    assert observation.u_std == 0.0
    assert observation.tr_std == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda runtime: runtime.next_op_index.__setitem__(0, -1), "operation bounds"),
        (lambda runtime: runtime.next_op_index.__setitem__(0, 3), "operation bounds"),
        (lambda runtime: runtime.timelines.reverse(), "timeline machine ids"),
        (lambda runtime: runtime.machines.reverse(), "runtime machine ids"),
        (lambda runtime: runtime.last_machine_by_job.pop(), "one entry per job"),
        (lambda runtime: runtime.last_machine_by_job.__setitem__(0, 2), "known machines"),
    ],
)
def test_observation_rejects_runtime_invariant_violations(
    mutation: Callable[[ScheduleRuntime], object], message: str
) -> None:
    runtime = create_runtime(_two_job_instance())
    mutation(runtime)

    with pytest.raises(ValueError, match=message):
        compute_observation(runtime, _profile("paper_repro"))


@pytest.mark.parametrize("machine_id", [0.0, False, "0"])
def test_observation_rejects_non_integer_timeline_machine_ids(
    machine_id: object,
) -> None:
    runtime = create_runtime(_two_job_instance())
    runtime.timelines[0].machine_id = machine_id  # type: ignore[assignment]

    with pytest.raises(ValueError, match="timeline machine ids must be non-boolean integers"):
        compute_observation(runtime, _profile("paper_repro"))


@pytest.mark.parametrize("machine_id", [0.0, False, "0"])
def test_observation_rejects_non_integer_runtime_machine_ids(
    machine_id: object,
) -> None:
    runtime = create_runtime(_two_job_instance())
    runtime.machines[0].machine_id = machine_id  # type: ignore[assignment]

    with pytest.raises(ValueError, match="runtime machine ids must be non-boolean integers"):
        compute_observation(runtime, _profile("paper_repro"))


@pytest.mark.parametrize("machine_id", [0.5, False, "0"])
def test_observation_rejects_non_integer_last_machine_references(
    machine_id: object,
) -> None:
    runtime = create_runtime(_two_job_instance())
    runtime.last_machine_by_job[0] = machine_id  # type: ignore[assignment]

    with pytest.raises(
        ValueError,
        match="last_machine_by_job entries must be non-boolean integers",
    ):
        compute_observation(runtime, _profile("paper_repro"))


@pytest.mark.parametrize(
    "order",
    [
        ("crj_ave", "crj_std", "u_ave", "u_std", "tr_ave"),
        ("crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "tr_ave"),
        ("crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "unknown"),
    ],
)
def test_vector_rejects_non_permutation_feature_orders(order: tuple[str, ...]) -> None:
    observation = ScheduleObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="each known feature exactly once"):
        observation.vector(order)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_vector_rejects_non_finite_observation_values(value: float) -> None:
    observation = ScheduleObservation(value, 0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="six finite values"):
        observation.vector(
            ("crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "tr_std")
        )


def test_schedule_observation_is_frozen() -> None:
    observation = ScheduleObservation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(FrozenInstanceError):
        observation.tr_ave = 1.0
