from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeVar, cast

import yaml  # type: ignore[import-untyped]


class ProfileName(StrEnum):
    LEGACY_SNAPSHOT = "legacy_snapshot"
    PAPER_REPRO = "paper_repro"
    CORRECTED_SMC = "corrected_smc"


class InitialObservationMode(StrEnum):
    ZERO = "zero"
    ENVIRONMENT = "environment"


class TRFeatureMode(StrEnum):
    LEGACY_WORKLOAD_PRESSURE = "legacy_workload_pressure"
    PROJECTED_COMPLETION_TARDINESS_RATIO = "projected_completion_tardiness_ratio"


class RuleSetName(StrEnum):
    LEGACY = "legacy"
    PAPER = "paper"


class SetupMode(StrEnum):
    NONE = "none"
    SOURCE_TOOL_CHANGE = "source_tool_change"


class LowerContextMode(StrEnum):
    MAX_Q_SCALAR = "max_q_scalar"
    REWARD_ID_SCALAR = "reward_id_scalar"
    REWARD_ID_ONE_HOT = "reward_id_one_hot"


class FailureMode(StrEnum):
    LEGACY_PRESTART_CDF = "legacy_prestart_cdf"
    PRESTART_CONDITIONAL_INTERVAL_RISK = "prestart_conditional_interval_risk"


class WearMode(StrEnum):
    LEGACY_PER_OPERATION = "legacy_per_operation"
    EFFECTIVE_AGE = "effective_age"


URGENCY_SEMANTICS: Mapping[int, str] = MappingProxyType(
    {1: "high", 2: "medium", 3: "low"}
)

_FEATURE_NAMES = frozenset({"crj_ave", "crj_std", "u_ave", "u_std", "tr_ave", "tr_std"})
_UTILIZATION_FEATURES = frozenset({"paper_uave", "standard_utilization"})
_REWARD_MODES = frozenset({"legacy", "paper"})


class _YamlMark(Protocol):
    line: int
    column: int


class _YamlNode(Protocol):
    id: str
    tag: str
    value: object
    start_mark: _YamlMark


def _yaml_child_path(path: str, key: str) -> str:
    if key.isidentifier():
        return _child_path(path, key)
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _reject_duplicate_mapping_keys(
    node: object, path: str, source_name: str, visited: set[int]
) -> None:
    node_identity = id(node)
    if node_identity in visited:
        return
    visited.add(node_identity)
    node_data = cast(_YamlNode, node)
    if node_data.id == "mapping":
        pairs = cast(list[tuple[object, object]], node_data.value)
        seen: dict[tuple[str, str], _YamlMark] = {}
        children: list[tuple[object, str]] = []
        for key_node, value_node in pairs:
            key_data = cast(_YamlNode, key_node)
            key = str(key_data.value)
            fingerprint = (key_data.tag, key)
            child_path = _yaml_child_path(path, key)
            if fingerprint in seen:
                first_mark = seen[fingerprint]
                mark = key_data.start_mark
                raise ValueError(
                    f"{source_name}: {child_path}: duplicate key {key!r} at "
                    f"line {mark.line + 1}, column {mark.column + 1}; first defined at "
                    f"line {first_mark.line + 1}, column {first_mark.column + 1}"
                )
            seen[fingerprint] = key_data.start_mark
            children.append((value_node, child_path))
        for value_node, child_path in children:
            _reject_duplicate_mapping_keys(value_node, child_path, source_name, visited)
    elif node_data.id == "sequence":
        items = cast(list[object], node_data.value)
        for index, item in enumerate(items):
            _reject_duplicate_mapping_keys(
                item, f"{path}[{index}]", source_name, visited
            )


class _StrictSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    def get_single_data(self) -> object:
        node: object | None = self.get_single_node()
        if node is None:
            return None
        source_name = str(getattr(self, "name", "<yaml>"))
        _reject_duplicate_mapping_keys(node, "$", source_name, set())
        return cast(object, self.construct_document(node))


@dataclass(frozen=True)
class StateConfig:
    order: tuple[str, ...]
    initial_observation: InitialObservationMode
    tr_feature: TRFeatureMode
    utilization_feature: str


@dataclass(frozen=True)
class ArchitectureConfig:
    upper_hidden: tuple[int, ...]
    lower_hidden: tuple[int, ...]
    lower_context: LowerContextMode


@dataclass(frozen=True)
class ReliabilityConfig:
    failure_mode: FailureMode
    wear_mode: WearMode
    pm_enabled: bool
    pm_failure_threshold: float
    pm_health_threshold: float
    cm_age_repair_factor: float
    high_load_failure_bias: bool


@dataclass(frozen=True)
class SchedulingConfig:
    rule_set: RuleSetName
    setup_mode: SetupMode
    local_insertion: bool
    explicit_nonprocess_intervals: bool


@dataclass(frozen=True)
class RewardConfig:
    tardiness_mode: str
    utilization_mode: str


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int
    replay_capacity: int
    batch_size: int
    gamma: float
    learning_rate: float
    target_update_steps: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decrement: float
    deterministic: bool


@dataclass(frozen=True)
class ReproductionProfile:
    schema_version: int
    profile: ProfileName
    state: StateConfig
    architecture: ArchitectureConfig
    reliability: ReliabilityConfig
    scheduling: SchedulingConfig
    reward: RewardConfig
    training: TrainingConfig

    @property
    def lower_input_dim(self) -> int:
        return 8 if self.architecture.lower_context is LowerContextMode.REWARD_ID_ONE_HOT else 7

    @property
    def urgency_semantics(self) -> Mapping[int, str]:
        return URGENCY_SEMANTICS

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.value,
            "state": {
                "order": list(self.state.order),
                "initial_observation": self.state.initial_observation.value,
                "tr_feature": self.state.tr_feature.value,
                "utilization_feature": self.state.utilization_feature,
            },
            "architecture": {
                "upper_hidden": list(self.architecture.upper_hidden),
                "lower_hidden": list(self.architecture.lower_hidden),
                "lower_context": self.architecture.lower_context.value,
            },
            "reliability": {
                "failure_mode": self.reliability.failure_mode.value,
                "wear_mode": self.reliability.wear_mode.value,
                "pm_enabled": self.reliability.pm_enabled,
                "pm_failure_threshold": self.reliability.pm_failure_threshold,
                "pm_health_threshold": self.reliability.pm_health_threshold,
                "cm_age_repair_factor": self.reliability.cm_age_repair_factor,
                "high_load_failure_bias": self.reliability.high_load_failure_bias,
            },
            "scheduling": {
                "rule_set": self.scheduling.rule_set.value,
                "setup_mode": self.scheduling.setup_mode.value,
                "local_insertion": self.scheduling.local_insertion,
                "explicit_nonprocess_intervals": (
                    self.scheduling.explicit_nonprocess_intervals
                ),
            },
            "reward": {
                "tardiness_mode": self.reward.tardiness_mode,
                "utilization_mode": self.reward.utilization_mode,
            },
            "training": {
                "episodes": self.training.episodes,
                "replay_capacity": self.training.replay_capacity,
                "batch_size": self.training.batch_size,
                "gamma": self.training.gamma,
                "learning_rate": self.training.learning_rate,
                "target_update_steps": self.training.target_update_steps,
                "epsilon_start": self.training.epsilon_start,
                "epsilon_end": self.training.epsilon_end,
                "epsilon_decrement": self.training.epsilon_decrement,
                "deterministic": self.training.deterministic,
            },
        }


_TOP_LEVEL_KEYS = (
    "schema_version",
    "profile",
    "state",
    "architecture",
    "reliability",
    "scheduling",
    "reward",
    "training",
)
_OVERRIDE_KEYS = (
    "schema_version",
    "state",
    "architecture",
    "reliability",
    "scheduling",
    "reward",
    "training",
)
_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "state": ("order", "initial_observation", "tr_feature", "utilization_feature"),
    "architecture": ("upper_hidden", "lower_hidden", "lower_context"),
    "reliability": (
        "failure_mode",
        "wear_mode",
        "pm_enabled",
        "pm_failure_threshold",
        "pm_health_threshold",
        "cm_age_repair_factor",
        "high_load_failure_bias",
    ),
    "scheduling": (
        "rule_set",
        "setup_mode",
        "local_insertion",
        "explicit_nonprocess_intervals",
    ),
    "reward": ("tardiness_mode", "utilization_mode"),
    "training": (
        "episodes",
        "replay_capacity",
        "batch_size",
        "gamma",
        "learning_rate",
        "target_update_steps",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decrement",
        "deterministic",
    ),
}

EnumT = TypeVar("EnumT", bound=StrEnum)


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}"


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{path}: mapping keys must be strings")
    return cast(dict[str, object], value)


def _check_keys(
    mapping: Mapping[str, object],
    *,
    required: tuple[str, ...],
    allowed: tuple[str, ...],
    path: str,
) -> None:
    for key in mapping:
        if key not in allowed:
            raise ValueError(f"{_child_path(path, key)}: unknown field")
    for key in required:
        if key not in mapping:
            raise ValueError(f"{_child_path(path, key)}: missing required field")


def _enum_value(enum_type: type[EnumT], value: object, path: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{path}: expected one of {choices}") from exc


def _choice(value: object, choices: frozenset[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{path}: expected one of {expected}")
    return value


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path}: expected a boolean")
    return value


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{path}: expected an integer")
    result = value
    if minimum is not None and result < minimum:
        raise ValueError(f"{path}: expected an integer >= {minimum}")
    return result


def _number(value: object, path: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{path}: expected a finite number")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        raise ValueError(f"{path}: expected a finite number")
    return result


def _bounded_number(value: object, path: str, minimum: float, maximum: float) -> float:
    result = _number(value, path)
    if not minimum <= result <= maximum:
        raise ValueError(f"{path}: expected a value in [{minimum}, {maximum}]")
    return result


def _positive_number(value: object, path: str) -> float:
    result = _number(value, path)
    if result <= 0.0:
        raise ValueError(f"{path}: expected a positive number")
    return result


def _hidden_dimensions(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: expected a nonempty list of positive integers")
    dimensions = tuple(
        _integer(item, f"{path}[{index}]", minimum=1) for index, item in enumerate(value)
    )
    return dimensions


def _feature_order(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{path}: expected exactly six feature names")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: feature names must be strings")
    order = tuple(cast(list[str], value))
    if len(set(order)) != 6 or set(order) != _FEATURE_NAMES:
        raise ValueError(f"{path}: expected each known feature exactly once")
    return order


def _section(mapping: Mapping[str, object], name: str) -> dict[str, object]:
    path = _child_path("$", name)
    section = _mapping(mapping[name], path)
    keys = _SECTION_KEYS[name]
    _check_keys(section, required=keys, allowed=keys, path=path)
    return section


def _parse_profile(raw: object) -> ReproductionProfile:
    mapping = _mapping(raw, "$")
    _check_keys(mapping, required=_TOP_LEVEL_KEYS, allowed=_TOP_LEVEL_KEYS, path="$")

    schema_version = _integer(mapping["schema_version"], "$.schema_version")
    if schema_version != 1:
        raise ValueError("$.schema_version: expected 1")

    state = _section(mapping, "state")
    architecture = _section(mapping, "architecture")
    reliability = _section(mapping, "reliability")
    scheduling = _section(mapping, "scheduling")
    reward = _section(mapping, "reward")
    training = _section(mapping, "training")

    replay_capacity = _integer(
        training["replay_capacity"], "$.training.replay_capacity", minimum=1
    )
    batch_size = _integer(training["batch_size"], "$.training.batch_size", minimum=1)

    return ReproductionProfile(
        schema_version=schema_version,
        profile=_enum_value(ProfileName, mapping["profile"], "$.profile"),
        state=StateConfig(
            order=_feature_order(state["order"], "$.state.order"),
            initial_observation=_enum_value(
                InitialObservationMode,
                state["initial_observation"],
                "$.state.initial_observation",
            ),
            tr_feature=_enum_value(TRFeatureMode, state["tr_feature"], "$.state.tr_feature"),
            utilization_feature=_choice(
                state["utilization_feature"],
                _UTILIZATION_FEATURES,
                "$.state.utilization_feature",
            ),
        ),
        architecture=ArchitectureConfig(
            upper_hidden=_hidden_dimensions(
                architecture["upper_hidden"], "$.architecture.upper_hidden"
            ),
            lower_hidden=_hidden_dimensions(
                architecture["lower_hidden"], "$.architecture.lower_hidden"
            ),
            lower_context=_enum_value(
                LowerContextMode,
                architecture["lower_context"],
                "$.architecture.lower_context",
            ),
        ),
        reliability=ReliabilityConfig(
            failure_mode=_enum_value(
                FailureMode, reliability["failure_mode"], "$.reliability.failure_mode"
            ),
            wear_mode=_enum_value(
                WearMode, reliability["wear_mode"], "$.reliability.wear_mode"
            ),
            pm_enabled=_boolean(reliability["pm_enabled"], "$.reliability.pm_enabled"),
            pm_failure_threshold=_bounded_number(
                reliability["pm_failure_threshold"],
                "$.reliability.pm_failure_threshold",
                0.0,
                1.0,
            ),
            pm_health_threshold=_bounded_number(
                reliability["pm_health_threshold"],
                "$.reliability.pm_health_threshold",
                0.0,
                100.0,
            ),
            cm_age_repair_factor=_bounded_number(
                reliability["cm_age_repair_factor"],
                "$.reliability.cm_age_repair_factor",
                0.0,
                1.0,
            ),
            high_load_failure_bias=_boolean(
                reliability["high_load_failure_bias"],
                "$.reliability.high_load_failure_bias",
            ),
        ),
        scheduling=SchedulingConfig(
            rule_set=_enum_value(
                RuleSetName, scheduling["rule_set"], "$.scheduling.rule_set"
            ),
            setup_mode=_enum_value(
                SetupMode, scheduling["setup_mode"], "$.scheduling.setup_mode"
            ),
            local_insertion=_boolean(
                scheduling["local_insertion"], "$.scheduling.local_insertion"
            ),
            explicit_nonprocess_intervals=_boolean(
                scheduling["explicit_nonprocess_intervals"],
                "$.scheduling.explicit_nonprocess_intervals",
            ),
        ),
        reward=RewardConfig(
            tardiness_mode=_choice(
                reward["tardiness_mode"], _REWARD_MODES, "$.reward.tardiness_mode"
            ),
            utilization_mode=_choice(
                reward["utilization_mode"],
                _REWARD_MODES,
                "$.reward.utilization_mode",
            ),
        ),
        training=TrainingConfig(
            episodes=_integer(training["episodes"], "$.training.episodes", minimum=1),
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            gamma=_bounded_number(training["gamma"], "$.training.gamma", 0.0, 1.0),
            learning_rate=_positive_number(
                training["learning_rate"], "$.training.learning_rate"
            ),
            target_update_steps=_integer(
                training["target_update_steps"],
                "$.training.target_update_steps",
                minimum=1,
            ),
            epsilon_start=_bounded_number(
                training["epsilon_start"], "$.training.epsilon_start", 0.0, 1.0
            ),
            epsilon_end=_bounded_number(
                training["epsilon_end"], "$.training.epsilon_end", 0.0, 1.0
            ),
            epsilon_decrement=_bounded_number(
                training["epsilon_decrement"],
                "$.training.epsilon_decrement",
                0.0,
                1.0,
            ),
            deterministic=_boolean(
                training["deterministic"], "$.training.deterministic"
            ),
        ),
    )


def _load_yaml_mapping(path: str | Path) -> dict[str, object]:
    resolved_path = Path(path)
    try:
        with resolved_path.open("r", encoding="utf-8") as stream:
            raw: object = yaml.load(stream, Loader=_StrictSafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{resolved_path}: unable to load YAML: {exc}") from exc
    return _mapping(raw, "$")


def _apply_override(
    base: ReproductionProfile, override: Mapping[str, object]
) -> dict[str, object]:
    _check_keys(
        override,
        required=("schema_version",),
        allowed=_OVERRIDE_KEYS,
        path="$",
    )
    schema_version = _integer(override["schema_version"], "$.schema_version")
    if schema_version != 1:
        raise ValueError("$.schema_version: expected 1")

    merged = base.to_dict()
    for section_name, allowed_keys in _SECTION_KEYS.items():
        if section_name not in override:
            continue
        path = _child_path("$", section_name)
        section_override = _mapping(override[section_name], path)
        _check_keys(section_override, required=(), allowed=allowed_keys, path=path)
        base_section = _mapping(merged[section_name], path)
        merged_section = dict(base_section)
        merged_section.update(section_override)
        merged[section_name] = merged_section
    return merged


def load_profile(
    path: str | Path, override_path: str | Path | None = None
) -> ReproductionProfile:
    base = _parse_profile(_load_yaml_mapping(path))
    if override_path is None:
        return base
    override = _load_yaml_mapping(override_path)
    return _parse_profile(_apply_override(base, override))


def profile_sha256(profile: ReproductionProfile) -> str:
    payload = json.dumps(
        profile.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
