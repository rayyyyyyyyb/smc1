from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from smc_repro.agents import (
    DualLayerValueAgent,
    TabularAgent,
    TabularAlgorithm,
    TabularRewardProtocol,
    TieBreakMode,
    Transition,
    discretize_six_feature_state,
)
from smc_repro.config import ReproductionProfile, load_profile
from smc_repro.environment import SchedulingEnvironment
from smc_repro.observations import ScheduleObservation
from smc_repro.rewards import legacy_joint_reward
from smc_repro.schemas import InstanceSpec, JobSpec, MachineSpec, OperationSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _profile() -> ReproductionProfile:
    return load_profile(CONFIG_ROOT / "paper_repro.yaml")


def _observation(*, tardiness: float, utilization: float) -> ScheduleObservation:
    return ScheduleObservation(0.0, 0.0, utilization, 0.0, tardiness, 0.0)


@pytest.mark.parametrize(
    ("current_tardiness", "current_utilization", "expected"),
    [
        (9.0, 0.1, 1),
        (10.5, 0.1, 0),
        (12.0, 0.6, 1),
        (12.0, 0.46, 0),
        (12.0, 0.4, -1),
    ],
)
def test_legacy_joint_reward_preserves_exact_source_branch_order(
    current_tardiness: float,
    current_utilization: float,
    expected: int,
) -> None:
    previous = _observation(tardiness=10.0, utilization=0.5)
    current = _observation(
        tardiness=current_tardiness,
        utilization=current_utilization,
    )

    assert legacy_joint_reward(previous, current) == expected


def test_tabular_protocol_names_are_exact_and_nonoverlapping() -> None:
    assert tuple(TabularAlgorithm) == (
        TabularAlgorithm.Q_LEARNING,
        TabularAlgorithm.SARSA,
    )
    assert tuple(TabularRewardProtocol) == (
        TabularRewardProtocol.LEGACY_JOINT,
        TabularRewardProtocol.FIXED_TARDINESS,
        TabularRewardProtocol.FIXED_UTILIZATION,
    )
    assert TieBreakMode.CORRECTED_TIE_BREAK.value == "corrected_tie_break"
    assert TieBreakMode.FIRST_ARGMAX.value == "first_argmax"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (np.zeros(6, dtype=np.float32), 0),
        (np.ones(6, dtype=np.float32), 9),
        (np.full(6, 0.59, dtype=np.float32), 5),
        (np.full(6, 2.0, dtype=np.float32), 9),
        (np.full(6, -2.0, dtype=np.float32), 0),
    ],
)
def test_discretizer_maps_six_features_to_exact_auditable_bins(
    state: np.ndarray,
    expected: int,
) -> None:
    assert discretize_six_feature_state(state) == expected


@pytest.mark.parametrize("algorithm", tuple(TabularAlgorithm))
def test_q_learning_and_sarsa_hand_updates_match_exact_equations(
    algorithm: TabularAlgorithm,
) -> None:
    agent = TabularAgent(
        algorithm,
        seed=7,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=0.0,
    )
    state = np.zeros(6, dtype=np.float32)
    next_state = np.ones(6, dtype=np.float32)
    agent.q_table[9, 3] = 2.0

    updated = agent.update(
        state,
        action=4,
        reward=1.0,
        next_state=next_state,
        done=False,
        next_action=3 if algorithm is TabularAlgorithm.SARSA else None,
    )

    assert updated == pytest.approx(0.29)
    assert agent.q_table[0, 4] == pytest.approx(0.29)


@pytest.mark.parametrize("algorithm", tuple(TabularAlgorithm))
def test_tabular_terminal_update_masks_all_bootstrap_values(
    algorithm: TabularAlgorithm,
) -> None:
    agent = TabularAgent(algorithm, seed=9, epsilon=0.0)
    state = np.zeros(6, dtype=np.float32)
    next_state = np.ones(6, dtype=np.float32)
    agent.q_table[9, :] = 1000.0

    observed = agent.update(
        state,
        action=2,
        reward=-1.0,
        next_state=next_state,
        done=True,
    )

    assert observed == pytest.approx(-0.1)


def test_corrected_tie_break_samples_every_tied_action_but_source_picks_first() -> None:
    state = np.zeros(6, dtype=np.float32)
    corrected = TabularAgent(
        TabularAlgorithm.Q_LEARNING,
        seed=44,
        epsilon=0.0,
        tie_break=TieBreakMode.CORRECTED_TIE_BREAK,
    )
    source = TabularAgent(
        TabularAlgorithm.Q_LEARNING,
        seed=44,
        epsilon=0.0,
        tie_break=TieBreakMode.FIRST_ARGMAX,
    )

    corrected_actions = {
        corrected.select_action(state, training=False) for _ in range(256)
    }
    source_actions = {source.select_action(state, training=False) for _ in range(32)}

    assert corrected_actions == set(range(9))
    assert source_actions == {0}


def test_tabular_primitive_state_restore_is_strict_and_continues_local_rng() -> None:
    state = np.zeros(6, dtype=np.float32)
    agent = TabularAgent(
        TabularAlgorithm.Q_LEARNING,
        seed=101,
        epsilon=0.2,
        tie_break=TieBreakMode.CORRECTED_TIE_BREAK,
    )
    agent.q_table[0, 2:5] = 3.0
    agent.decay_epsilon()
    saved = agent.state_dict()
    expected_actions = [agent.select_action(state, training=True) for _ in range(40)]
    restored = TabularAgent(
        TabularAlgorithm.Q_LEARNING,
        seed=999,
        epsilon=0.2,
        tie_break=TieBreakMode.CORRECTED_TIE_BREAK,
    )

    restored.load_state_dict(saved)
    observed_actions = [restored.select_action(state, training=True) for _ in range(40)]

    assert observed_actions == expected_actions
    assert np.array_equal(restored.q_table, np.asarray(saved["q_table"]))
    assert restored.epsilon == saved["epsilon"]

    incompatible = TabularAgent(TabularAlgorithm.SARSA, seed=1)
    before = incompatible.state_dict()
    with pytest.raises(ValueError, match="algorithm"):
        incompatible.load_state_dict(saved)
    assert incompatible.state_dict() == before


def test_tabular_epsilon_decay_is_multiplicative_and_respects_minimum() -> None:
    agent = TabularAgent(
        TabularAlgorithm.Q_LEARNING,
        seed=3,
        epsilon=0.02,
        epsilon_decay=0.5,
        minimum_epsilon=0.01,
    )

    observed = [agent.decay_epsilon() for _ in range(4)]

    assert observed == [0.01, 0.01, 0.01, 0.01]


def test_deep_and_tabular_agent_methods_do_not_mutate_environment_crn_streams() -> None:
    instance = InstanceSpec(
        "stream-isolation",
        111,
        222,
        (
            JobSpec(
                0,
                0.0,
                100.0,
                1,
                (OperationSpec(0, 0, (2.0,)),),
            ),
        ),
        (MachineSpec(0, 0.0, 4.0),),
    )
    environment = SchedulingEnvironment(instance, _profile(), policy_seed=333)
    state, _ = environment.reset()
    before = (
        environment.failure_seed,
        environment.wear_seed,
        environment.repair_seed,
        environment.runtime.decision_index,
        tuple(environment.runtime.next_op_index),
        tuple(timeline.intervals for timeline in environment.runtime.timelines),
    )
    deep = DualLayerValueAgent(
        _profile(),
        seed=444,
        device=torch.device("cpu"),
        double_dqn=True,
    )
    tabular = TabularAgent(TabularAlgorithm.Q_LEARNING, seed=555)

    deep.decide(state, training=True)
    tabular_action = tabular.select_action(state, training=True)
    tabular.update(state, tabular_action, 1.0, state, done=True)

    after = (
        environment.failure_seed,
        environment.wear_seed,
        environment.repair_seed,
        environment.runtime.decision_index,
        tuple(environment.runtime.next_op_index),
        tuple(timeline.intervals for timeline in environment.runtime.timelines),
    )
    assert after == before


def test_transition_huge_integer_boundaries_raise_controlled_value_errors() -> None:
    huge = 10**1000
    ordinary_state = np.zeros(6, dtype=np.float32)
    huge_state = np.asarray([huge, 0, 0, 0, 0, 0], dtype=object)

    with pytest.raises(ValueError, match="state"):
        Transition(huge_state, 0, 0.0, ordinary_state, 0, False)
    with pytest.raises(ValueError, match="reward"):
        Transition(ordinary_state, 0, huge, ordinary_state, 0, False)


def test_tabular_huge_integer_boundaries_raise_controlled_value_errors() -> None:
    huge = 10**1000
    ordinary_state = np.zeros(6, dtype=np.float32)
    huge_state = np.asarray([huge, 0, 0, 0, 0, 0], dtype=object)

    with pytest.raises(ValueError, match="learning_rate"):
        TabularAgent(
            TabularAlgorithm.Q_LEARNING,
            seed=1,
            learning_rate=huge,
        )
    agent = TabularAgent(TabularAlgorithm.Q_LEARNING, seed=1)
    with pytest.raises(ValueError, match="reward"):
        agent.update(ordinary_state, 0, huge, ordinary_state, done=True)
    with pytest.raises(ValueError, match="tabular state"):
        discretize_six_feature_state(huge_state)

    state = agent.state_dict()
    state["epsilon"] = huge
    before = agent.state_dict()
    with pytest.raises(ValueError, match="epsilon"):
        agent.load_state_dict(state)
    assert agent.state_dict() == before
