from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest
import yaml

from smc_repro.config import (
    URGENCY_SEMANTICS,
    FailureMode,
    InitialObservationMode,
    LowerContextMode,
    ProfileName,
    ReproductionProfile,
    RuleSetName,
    SetupMode,
    TRFeatureMode,
    WearMode,
    load_profile,
    profile_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
LEGACY_PATH = CONFIG_ROOT / "legacy_snapshot.yaml"
PAPER_PATH = CONFIG_ROOT / "paper_repro.yaml"
CORRECTED_PATH = CONFIG_ROOT / "corrected_smc.yaml"
SMOKE_PATH = CONFIG_ROOT / "smoke.yaml"
AMBIGUITIES_PATH = CONFIG_ROOT / "ambiguities.json"

EXPECTED_PROFILES: dict[str, dict[str, Any]] = {
    "legacy_snapshot": {
        "schema_version": 1,
        "profile": "legacy_snapshot",
        "state": {
            "order": ["u_ave", "u_std", "crj_ave", "crj_std", "tr_ave", "tr_std"],
            "initial_observation": "zero",
            "tr_feature": "legacy_workload_pressure",
            "utilization_feature": "paper_uave",
        },
        "architecture": {
            "upper_hidden": [10, 10, 10],
            "lower_hidden": [50, 50, 50, 50, 50, 50, 50],
            "lower_context": "max_q_scalar",
        },
        "reliability": {
            "failure_mode": "legacy_prestart_cdf",
            "wear_mode": "legacy_per_operation",
            "pm_enabled": True,
            "pm_failure_threshold": 0.2,
            "pm_health_threshold": 30.0,
            "cm_age_repair_factor": 0.5,
            "high_load_failure_bias": True,
        },
        "scheduling": {
            "rule_set": "legacy",
            "setup_mode": "source_tool_change",
            "local_insertion": False,
            "explicit_nonprocess_intervals": True,
        },
        "reward": {"tardiness_mode": "legacy", "utilization_mode": "legacy"},
        "training": {
            "episodes": 200,
            "replay_capacity": 2000,
            "batch_size": 16,
            "gamma": 0.95,
            "learning_rate": 0.001,
            "target_update_steps": 200,
            "epsilon_start": 0.6,
            "epsilon_end": 0.01,
            "epsilon_decrement": 0.0001,
            "deterministic": True,
        },
    },
    "paper_repro": {
        "schema_version": 1,
        "profile": "paper_repro",
        "state": {
            "order": ["crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "tr_std"],
            "initial_observation": "environment",
            "tr_feature": "legacy_workload_pressure",
            "utilization_feature": "paper_uave",
        },
        "architecture": {
            "upper_hidden": [10, 10],
            "lower_hidden": [50, 50],
            "lower_context": "reward_id_scalar",
        },
        "reliability": {
            "failure_mode": "legacy_prestart_cdf",
            "wear_mode": "legacy_per_operation",
            "pm_enabled": True,
            "pm_failure_threshold": 0.2,
            "pm_health_threshold": 30.0,
            "cm_age_repair_factor": 0.5,
            "high_load_failure_bias": True,
        },
        "scheduling": {
            "rule_set": "paper",
            "setup_mode": "none",
            "local_insertion": False,
            "explicit_nonprocess_intervals": True,
        },
        "reward": {"tardiness_mode": "paper", "utilization_mode": "paper"},
        "training": {
            "episodes": 200,
            "replay_capacity": 2000,
            "batch_size": 16,
            "gamma": 0.95,
            "learning_rate": 0.001,
            "target_update_steps": 200,
            "epsilon_start": 0.6,
            "epsilon_end": 0.01,
            "epsilon_decrement": 0.0001,
            "deterministic": True,
        },
    },
    "corrected_smc": {
        "schema_version": 1,
        "profile": "corrected_smc",
        "state": {
            "order": ["crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "tr_std"],
            "initial_observation": "environment",
            "tr_feature": "projected_completion_tardiness_ratio",
            "utilization_feature": "standard_utilization",
        },
        "architecture": {
            "upper_hidden": [10, 10],
            "lower_hidden": [50, 50],
            "lower_context": "reward_id_scalar",
        },
        "reliability": {
            "failure_mode": "prestart_conditional_interval_risk",
            "wear_mode": "effective_age",
            "pm_enabled": True,
            "pm_failure_threshold": 0.2,
            "pm_health_threshold": 30.0,
            "cm_age_repair_factor": 0.5,
            "high_load_failure_bias": False,
        },
        "scheduling": {
            "rule_set": "paper",
            "setup_mode": "source_tool_change",
            "local_insertion": False,
            "explicit_nonprocess_intervals": True,
        },
        "reward": {"tardiness_mode": "paper", "utilization_mode": "paper"},
        "training": {
            "episodes": 200,
            "replay_capacity": 2000,
            "batch_size": 16,
            "gamma": 0.95,
            "learning_rate": 0.001,
            "target_update_steps": 200,
            "epsilon_start": 0.6,
            "epsilon_end": 0.01,
            "epsilon_decrement": 0.0001,
            "deterministic": True,
        },
    },
}

PROFILE_PATHS = {
    "legacy_snapshot": LEGACY_PATH,
    "paper_repro": PAPER_PATH,
    "corrected_smc": CORRECTED_PATH,
}


def _base_mapping() -> dict[str, Any]:
    raw = yaml.safe_load(PAPER_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_yaml(tmp_path: Path, data: dict[str, Any], name: str = "profile.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_yaml_text(tmp_path: Path, text: str, name: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _assert_duplicate_error(error: pytest.ExceptionInfo[ValueError], path: str, key: str) -> None:
    message = str(error.value)
    assert f"{path}: duplicate key {key!r}" in message
    assert re.search(r"line \d+, column \d+", message)


def _mutated_yaml(tmp_path: Path, dotted_path: str, value: Any) -> Path:
    data = copy.deepcopy(_base_mapping())
    parts = dotted_path.split(".")
    target = data
    for part in parts[:-1]:
        child = target[part]
        assert isinstance(child, dict)
        target = child
    target[parts[-1]] = value
    return _write_yaml(tmp_path, data)


def _missing_yaml(tmp_path: Path, dotted_path: str) -> Path:
    data = copy.deepcopy(_base_mapping())
    parts = dotted_path.split(".")
    target = data
    for part in parts[:-1]:
        child = target[part]
        assert isinstance(child, dict)
        target = child
    del target[parts[-1]]
    return _write_yaml(tmp_path, data)


def _reverse_mappings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_mappings(item) for key, item in reversed(value.items())}
    if isinstance(value, list):
        return [_reverse_mappings(item) for item in value]
    return value


def test_all_four_yaml_files_load_as_three_exact_scientific_profiles() -> None:
    for profile_name, path in PROFILE_PATHS.items():
        profile = load_profile(path)
        assert profile.to_dict() == EXPECTED_PROFILES[profile_name]

    smoke = load_profile(PAPER_PATH, SMOKE_PATH)
    assert smoke.profile is ProfileName.PAPER_REPRO
    assert smoke.training.episodes == 3
    assert smoke.training.replay_capacity == 128
    assert smoke.training.batch_size == 4
    assert smoke.training.gamma == 0.95
    assert smoke.training.target_update_steps == 5
    assert smoke.training.epsilon_start == 0.2
    assert smoke.training.epsilon_end == 0.0
    assert smoke.training.epsilon_decrement == 0.1

    with pytest.raises(ValueError, match=r"\$\.profile"):
        load_profile(SMOKE_PATH)
    with pytest.raises(ValueError):
        ProfileName("smoke")


def test_profile_is_strictly_typed_with_prescribed_enums_and_nested_dataclasses() -> None:
    assert issubclass(ProfileName, StrEnum)
    assert [member.value for member in ProfileName] == [
        "legacy_snapshot",
        "paper_repro",
        "corrected_smc",
    ]
    assert [member.value for member in InitialObservationMode] == ["zero", "environment"]
    assert [member.value for member in TRFeatureMode] == [
        "legacy_workload_pressure",
        "projected_completion_tardiness_ratio",
    ]
    assert [member.value for member in RuleSetName] == ["legacy", "paper"]
    assert [member.value for member in SetupMode] == ["none", "source_tool_change"]
    assert [member.value for member in LowerContextMode] == [
        "max_q_scalar",
        "reward_id_scalar",
        "reward_id_one_hot",
    ]
    assert [member.value for member in FailureMode] == [
        "legacy_prestart_cdf",
        "prestart_conditional_interval_risk",
    ]
    assert [member.value for member in WearMode] == [
        "legacy_per_operation",
        "effective_age",
    ]

    profile = load_profile(CORRECTED_PATH)
    assert isinstance(profile, ReproductionProfile)
    assert is_dataclass(profile)
    for section in (
        profile.state,
        profile.architecture,
        profile.reliability,
        profile.scheduling,
        profile.reward,
        profile.training,
    ):
        assert is_dataclass(section)
    assert profile.profile is ProfileName.CORRECTED_SMC
    assert profile.state.initial_observation is InitialObservationMode.ENVIRONMENT
    assert profile.state.tr_feature is TRFeatureMode.PROJECTED_COMPLETION_TARDINESS_RATIO
    assert profile.architecture.lower_context is LowerContextMode.REWARD_ID_SCALAR
    assert profile.reliability.failure_mode is FailureMode.PRESTART_CONDITIONAL_INTERVAL_RISK
    assert profile.reliability.wear_mode is WearMode.EFFECTIVE_AGE
    assert profile.scheduling.rule_set is RuleSetName.PAPER
    assert profile.scheduling.setup_mode is SetupMode.SOURCE_TOOL_CHANGE


def test_unknown_top_level_key_fails_with_path(tmp_path: Path) -> None:
    path = _mutated_yaml(tmp_path, "unexpected", "value")
    with pytest.raises(ValueError, match=r"\$\.unexpected"):
        load_profile(path)


def test_unknown_nested_key_fails_with_path(tmp_path: Path) -> None:
    path = _mutated_yaml(tmp_path, "training.optimizer", "adam")
    with pytest.raises(ValueError, match=r"\$\.training\.optimizer"):
        load_profile(path)


@pytest.mark.parametrize(
    ("key", "original", "replacement"),
    [
        (
            "schema_version",
            "schema_version: 1\n",
            "schema_version: 1\nschema_version: 2\n",
        ),
        (
            "profile",
            "profile: paper_repro\n",
            "profile: paper_repro\nprofile: paper_repro\n",
        ),
        (
            "training",
            "training:\n",
            "training:\ntraining:\n",
        ),
    ],
)
def test_duplicate_root_keys_are_rejected_before_construction(
    tmp_path: Path, key: str, original: str, replacement: str
) -> None:
    text = PAPER_PATH.read_text(encoding="utf-8").replace(original, replacement, 1)
    path = _write_yaml_text(tmp_path, text, "duplicate-root.yaml")

    with pytest.raises(ValueError) as error:
        load_profile(path)

    _assert_duplicate_error(error, f"$.{key}", key)


def test_duplicate_nested_base_key_is_rejected_even_when_values_match(tmp_path: Path) -> None:
    text = PAPER_PATH.read_text(encoding="utf-8").replace(
        "  episodes: 200\n",
        "  episodes: 200\n  episodes: 200\n",
        1,
    )
    path = _write_yaml_text(tmp_path, text, "duplicate-nested.yaml")

    with pytest.raises(ValueError) as error:
        load_profile(path)

    _assert_duplicate_error(error, "$.training.episodes", "episodes")


def test_duplicate_nested_smoke_override_key_is_rejected(tmp_path: Path) -> None:
    text = SMOKE_PATH.read_text(encoding="utf-8").replace(
        "  episodes: 3\n",
        "  episodes: 3\n  episodes: 4\n",
        1,
    )
    path = _write_yaml_text(tmp_path, text, "duplicate-smoke.yaml")

    with pytest.raises(ValueError) as error:
        load_profile(PAPER_PATH, path)

    _assert_duplicate_error(error, "$.training.episodes", "episodes")


def test_cyclic_alias_reaches_normal_dimension_validation(tmp_path: Path) -> None:
    text = PAPER_PATH.read_text(encoding="utf-8").replace(
        "  upper_hidden: [10, 10]\n",
        "  upper_hidden: &cycle [*cycle]\n",
        1,
    )
    path = _write_yaml_text(tmp_path, text, "cyclic-alias.yaml")

    with pytest.raises(
        ValueError,
        match=r"\$\.architecture\.upper_hidden\[0\]: expected an integer",
    ):
        load_profile(path)


def test_ordinary_sequence_alias_remains_valid(tmp_path: Path) -> None:
    text = PAPER_PATH.read_text(encoding="utf-8").replace(
        "  upper_hidden: [10, 10]\n  lower_hidden: [50, 50]\n",
        "  upper_hidden: &hidden [10, 10]\n  lower_hidden: *hidden\n",
        1,
    )
    path = _write_yaml_text(tmp_path, text, "ordinary-alias.yaml")

    profile = load_profile(path)

    assert profile.architecture.upper_hidden == (10, 10)
    assert profile.architecture.lower_hidden == (10, 10)


def test_mapping_merge_key_remains_valid(tmp_path: Path) -> None:
    text = PAPER_PATH.read_text(encoding="utf-8").replace(
        "scheduling:\n"
        "  rule_set: paper\n"
        "  setup_mode: none\n"
        "  local_insertion: false\n"
        "  explicit_nonprocess_intervals: true\n",
        "scheduling:\n"
        "  <<: &scheduling_defaults\n"
        "    rule_set: paper\n"
        "    setup_mode: none\n"
        "    local_insertion: false\n"
        "  explicit_nonprocess_intervals: true\n",
        1,
    )
    path = _write_yaml_text(tmp_path, text, "mapping-merge.yaml")

    profile = load_profile(path)

    assert profile.scheduling.rule_set is RuleSetName.PAPER
    assert profile.scheduling.setup_mode is SetupMode.NONE
    assert profile.scheduling.local_insertion is False
    assert profile.scheduling.explicit_nonprocess_intervals is True


def test_duplicate_key_with_alias_value_remains_rejected(tmp_path: Path) -> None:
    text = PAPER_PATH.read_text(encoding="utf-8").replace(
        "  order: [crj_ave, crj_std, u_ave, u_std, tr_ave, tr_std]\n",
        "  order: &feature_order [crj_ave, crj_std, u_ave, u_std, tr_ave, tr_std]\n"
        "  order: *feature_order\n",
        1,
    )
    path = _write_yaml_text(tmp_path, text, "duplicate-alias.yaml")

    with pytest.raises(ValueError) as error:
        load_profile(path)

    _assert_duplicate_error(error, "$.state.order", "order")


@pytest.mark.parametrize(
    "dotted_path",
    [
        "schema_version",
        "profile",
        "state",
        "state.order",
        "architecture.lower_context",
        "reliability.pm_enabled",
        "scheduling.setup_mode",
        "reward.tardiness_mode",
        "training.gamma",
    ],
)
def test_missing_required_field_fails_with_path(tmp_path: Path, dotted_path: str) -> None:
    path = _missing_yaml(tmp_path, dotted_path)
    expected_path = "$" + "".join(f".{part}" for part in dotted_path.split("."))
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        load_profile(path)


def test_reward_id_one_hot_implies_eight_lower_inputs(tmp_path: Path) -> None:
    path = _mutated_yaml(tmp_path, "architecture.lower_context", "reward_id_one_hot")
    assert load_profile(path).lower_input_dim == 8


@pytest.mark.parametrize("mode", ["max_q_scalar", "reward_id_scalar"])
def test_scalar_contexts_imply_seven_lower_inputs(tmp_path: Path, mode: str) -> None:
    path = _mutated_yaml(tmp_path, "architecture.lower_context", mode)
    assert load_profile(path).lower_input_dim == 7


@pytest.mark.parametrize(
    "order",
    [
        ["crj_ave", "crj_std", "u_ave", "u_std", "tr_ave"],
        ["crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "tr_ave"],
        ["crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "unknown"],
    ],
)
def test_feature_order_requires_six_unique_known_names(tmp_path: Path, order: list[str]) -> None:
    path = _mutated_yaml(tmp_path, "state.order", order)
    with pytest.raises(ValueError, match=r"\$\.state\.order"):
        load_profile(path)


def test_setup_mode_is_strict_and_paper_repro_disables_setup(tmp_path: Path) -> None:
    assert load_profile(PAPER_PATH).scheduling.setup_mode is SetupMode.NONE
    assert load_profile(LEGACY_PATH).scheduling.setup_mode is SetupMode.SOURCE_TOOL_CHANGE
    path = _mutated_yaml(tmp_path, "scheduling.setup_mode", "implicit")
    with pytest.raises(ValueError, match=r"\$\.scheduling\.setup_mode"):
        load_profile(path)


def test_urgency_semantics_are_locked_and_cannot_be_overridden(tmp_path: Path) -> None:
    expected = {1: "high", 2: "medium", 3: "low"}
    assert dict(URGENCY_SEMANTICS) == expected
    assert dict(load_profile(PAPER_PATH).urgency_semantics) == expected

    override = {
        "schema_version": 1,
        "urgency_semantics": {1: "low", 2: "medium", 3: "high"},
    }
    path = _write_yaml(tmp_path, override, "override.yaml")
    with pytest.raises(ValueError, match=r"\$\.urgency_semantics"):
        load_profile(PAPER_PATH, path)


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    [
        ("reliability.pm_failure_threshold", -0.01),
        ("reliability.pm_failure_threshold", 1.01),
        ("reliability.pm_health_threshold", -0.01),
        ("reliability.pm_health_threshold", 100.01),
        ("reliability.cm_age_repair_factor", -0.01),
        ("reliability.cm_age_repair_factor", 1.01),
        ("training.episodes", 0),
        ("training.replay_capacity", 0),
        ("training.batch_size", 0),
        ("training.gamma", -0.01),
        ("training.gamma", 1.01),
        ("training.learning_rate", 0.0),
        ("training.target_update_steps", 0),
        ("training.epsilon_start", -0.01),
        ("training.epsilon_start", 1.01),
        ("training.epsilon_end", -0.01),
        ("training.epsilon_end", 1.01),
        ("training.epsilon_decrement", -0.01),
        ("training.epsilon_decrement", 1.01),
    ],
)
def test_probability_health_training_and_epsilon_ranges_are_validated(
    tmp_path: Path, dotted_path: str, value: Any
) -> None:
    path = _mutated_yaml(tmp_path, dotted_path, value)
    expected_path = "$" + "".join(f".{part}" for part in dotted_path.split("."))
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        load_profile(path)


def test_canonical_round_trip_and_sha_ignore_input_mapping_order(tmp_path: Path) -> None:
    profile = load_profile(CORRECTED_PATH)
    expected_payload = json.dumps(
        EXPECTED_PROFILES["corrected_smc"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected_sha = hashlib.sha256(expected_payload).hexdigest()
    assert profile_sha256(profile) == expected_sha

    round_trip_path = _write_yaml(tmp_path, profile.to_dict(), "round_trip.yaml")
    round_trip = load_profile(round_trip_path)
    assert round_trip.to_dict() == profile.to_dict()
    assert profile_sha256(round_trip) == expected_sha

    reversed_path = _write_yaml(
        tmp_path,
        _reverse_mappings(EXPECTED_PROFILES["corrected_smc"]),
        "reversed.yaml",
    )
    reordered = load_profile(reversed_path)
    assert reordered.to_dict() == profile.to_dict()
    assert profile_sha256(reordered) == expected_sha


def test_ambiguity_register_has_exact_unique_ids_and_explicit_resolutions() -> None:
    data = json.loads(AMBIGUITIES_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    items = data["items"]
    ids = [item["id"] for item in items]
    assert ids == [f"A-{number:03d}" for number in range(1, 13)]
    assert len(ids) == len(set(ids))

    expected_resolutions: dict[str, str | dict[str, str]] = {
        "A-001": {
            "legacy_snapshot": "source_code",
            "paper_repro": "paper",
            "corrected_smc": "paper",
        },
        "A-002": {
            "legacy_snapshot": "max_q_scalar",
            "paper_repro": "reward_id_scalar",
            "corrected_smc": "reward_id_scalar",
        },
        "A-003": {
            "legacy_snapshot": "source_code",
            "paper_repro": "paper",
            "corrected_smc": "paper",
        },
        "A-004": {
            "legacy_snapshot": "legacy_workload_pressure",
            "paper_repro": "legacy_workload_pressure",
            "corrected_smc": "projected_completion_tardiness_ratio",
        },
        "A-005": (
            "FIFO uses arrival time; EDD uses absolute due date; MRT uses total remaining "
            "nominal work; SPT/LPT use the next operation's mean eligible-machine nominal "
            "time; all five use earliest predicted completion under the active profile and "
            "are labelled FIFO+ECT, EDD+ECT, MRT+ECT, SPT+ECT, LPT+ECT"
        ),
        "A-006": {
            "legacy_snapshot": "legacy_prestart_cdf",
            "paper_repro": "legacy_prestart_cdf",
            "corrected_smc": "prestart_conditional_interval_risk",
        },
        "A-007": {
            "legacy_snapshot": "zero",
            "paper_repro": "environment",
            "corrected_smc": "environment",
        },
        "A-008": {
            "legacy_snapshot": "source_code",
            "paper_repro": (
                "use decision_time as the overdue completion proxy and reproduce the printed "
                "non-overdue expression"
            ),
            "corrected_smc": (
                "use decision_time as the overdue completion proxy and projected/remaining-work "
                "values for state diagnostics"
            ),
        },
        "A-009": {
            "legacy_snapshot": "tail_append",
            "paper_repro": "tail_append_with_discrepancy_recorded",
            "corrected_smc": "tail_append",
        },
        "A-010": {
            "legacy_snapshot": "source_tool_change",
            "paper_repro": "none",
            "corrected_smc": "source_tool_change",
        },
        "A-011": (
            "all profiles use 1=high, 2=medium, 3=low and preserve that meaning in metadata, "
            "tests, weights, and rule formulas"
        ),
        "A-012": (
            "all stored due dates are absolute; for every job use arrival_time + "
            "(0.2 + 0.5*urgency)*estimated_work, and subtract arrival only when a relative "
            "due window is required"
        ),
    }
    assert {item["id"]: item["resolution"] for item in items} == expected_resolutions
    profile_names = {"legacy_snapshot", "paper_repro", "corrected_smc"}
    for item in items:
        resolution = item["resolution"]
        if isinstance(resolution, dict):
            assert set(resolution) == profile_names
            assert all(isinstance(value, str) and value for value in resolution.values())
        else:
            assert isinstance(resolution, str) and resolution


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    [
        ("profile", "smoke"),
        ("state.initial_observation", "cached"),
        ("state.tr_feature", "final_tardiness"),
        ("state.utilization_feature", "busy_ratio"),
        ("architecture.lower_context", "reward_vector"),
        ("reliability.failure_mode", "within_operation"),
        ("reliability.wear_mode", "calendar_age"),
        ("scheduling.rule_set", "mixed"),
        ("scheduling.setup_mode", "implicit"),
        ("reward.tardiness_mode", "corrected"),
        ("reward.utilization_mode", "corrected"),
    ],
)
def test_invalid_enums_and_locked_string_modes_fail_with_paths(
    tmp_path: Path, dotted_path: str, value: str
) -> None:
    path = _mutated_yaml(tmp_path, dotted_path, value)
    expected_path = "$" + "".join(f".{part}" for part in dotted_path.split("."))
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        load_profile(path)


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    [
        ("schema_version", True),
        ("architecture.upper_hidden", [True, 10]),
        ("reliability.pm_failure_threshold", True),
        ("training.episodes", True),
        ("training.gamma", True),
    ],
)
def test_bool_is_never_accepted_as_an_integer_or_float(
    tmp_path: Path, dotted_path: str, value: Any
) -> None:
    path = _mutated_yaml(tmp_path, dotted_path, value)
    expected_path = "$" + "".join(f".{part}" for part in dotted_path.split("."))
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        load_profile(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "dotted_path",
    [
        "reliability.pm_failure_threshold",
        "reliability.pm_health_threshold",
        "reliability.cm_age_repair_factor",
        "training.gamma",
        "training.learning_rate",
        "training.epsilon_start",
        "training.epsilon_end",
        "training.epsilon_decrement",
    ],
)
def test_non_finite_numbers_fail_with_paths(
    tmp_path: Path, dotted_path: str, value: float
) -> None:
    path = _mutated_yaml(tmp_path, dotted_path, value)
    expected_path = "$" + "".join(f".{part}" for part in dotted_path.split("."))
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        load_profile(path)


@pytest.mark.parametrize(
    ("dotted_path", "value"),
    [
        ("architecture.upper_hidden", []),
        ("architecture.upper_hidden", [10, 0]),
        ("architecture.lower_hidden", []),
        ("architecture.lower_hidden", [50, -1]),
    ],
)
def test_network_dimensions_must_be_nonempty_positive_integers(
    tmp_path: Path, dotted_path: str, value: list[int]
) -> None:
    path = _mutated_yaml(tmp_path, dotted_path, value)
    expected_path = "$" + "".join(f".{part}" for part in dotted_path.split("."))
    with pytest.raises(ValueError, match=re.escape(expected_path)):
        load_profile(path)


def test_incompatible_explicit_lower_input_dimension_is_rejected(tmp_path: Path) -> None:
    path = _mutated_yaml(tmp_path, "architecture.lower_input_dim", 8)
    with pytest.raises(ValueError, match=r"\$\.architecture\.lower_input_dim"):
        load_profile(path)


def test_override_is_deep_strict_and_cannot_change_profile_identity(tmp_path: Path) -> None:
    unknown_top = _write_yaml(
        tmp_path,
        {"schema_version": 1, "notes": "quick"},
        "unknown_top.yaml",
    )
    with pytest.raises(ValueError, match=r"\$\.notes"):
        load_profile(PAPER_PATH, unknown_top)

    unknown_nested = _write_yaml(
        tmp_path,
        {"schema_version": 1, "training": {"optimizer": "adam"}},
        "unknown_nested.yaml",
    )
    with pytest.raises(ValueError, match=r"\$\.training\.optimizer"):
        load_profile(PAPER_PATH, unknown_nested)

    changed_profile = _write_yaml(
        tmp_path,
        {"schema_version": 1, "profile": "corrected_smc"},
        "changed_profile.yaml",
    )
    with pytest.raises(ValueError, match=r"\$\.profile"):
        load_profile(PAPER_PATH, changed_profile)


def test_override_requires_schema_version_one(tmp_path: Path) -> None:
    missing = _write_yaml(tmp_path, {"training": {"episodes": 3}}, "missing.yaml")
    with pytest.raises(ValueError, match=r"\$\.schema_version"):
        load_profile(PAPER_PATH, missing)

    wrong = _write_yaml(
        tmp_path,
        {"schema_version": 2, "training": {"episodes": 3}},
        "wrong.yaml",
    )
    with pytest.raises(ValueError, match=r"\$\.schema_version"):
        load_profile(PAPER_PATH, wrong)


def test_yaml_root_and_nested_sections_must_be_mappings(tmp_path: Path) -> None:
    root = tmp_path / "root.yaml"
    root.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\$"):
        load_profile(root)

    path = _mutated_yaml(tmp_path, "training", ["not", "a", "mapping"])
    with pytest.raises(ValueError, match=r"\$\.training"):
        load_profile(path)
