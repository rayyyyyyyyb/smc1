from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from smc_repro.agents.dual_ddqn import DualLayerValueAgent
from smc_repro.config import profile_sha256
from smc_repro.experiment_contract import RunContract, contract_sha256

CHECKPOINT_SCHEMA_VERSION = 1

_PAYLOAD_FIELDS = {
    "schema_version",
    "contract",
    "contract_sha256",
    "profile",
    "profile_sha256",
    "double_dqn",
    "upper_online",
    "upper_target",
    "lower_online",
    "lower_target",
    "upper_optimizer",
    "lower_optimizer",
    "global_update_step",
    "decision_count",
    "epsilon",
    "agent_rng_state",
    "replay_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_contract(contract: RunContract) -> dict[str, object]:
    if type(contract) is not RunContract:
        raise TypeError("contract must be a RunContract")
    return cast(dict[str, object], dataclasses.asdict(contract))


def _checkpoint_payload(
    agent: DualLayerValueAgent,
    contract: RunContract,
) -> dict[str, object]:
    cuda_states: tuple[torch.Tensor, ...]
    if torch.cuda.is_available():
        cuda_states = tuple(torch.cuda.get_rng_state_all())
    else:
        cuda_states = ()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract": _canonical_contract(contract),
        "contract_sha256": contract_sha256(contract),
        "profile": agent.profile.to_dict(),
        "profile_sha256": profile_sha256(agent.profile),
        "double_dqn": agent.double_dqn,
        "upper_online": agent.upper_online.state_dict(),
        "upper_target": agent.upper_target.state_dict(),
        "lower_online": agent.lower_online.state_dict(),
        "lower_target": agent.lower_target.state_dict(),
        "upper_optimizer": agent.upper_optimizer.state_dict(),
        "lower_optimizer": agent.lower_optimizer.state_dict(),
        "global_update_step": agent.global_update_step,
        "decision_count": agent.decision_count,
        "epsilon": agent.epsilon,
        "agent_rng_state": agent.rng_state(),
        "replay_state": agent.replay.state_dict(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_states": cuda_states,
    }


def save_checkpoint(
    path: Path,
    agent: DualLayerValueAgent,
    contract: RunContract,
) -> str:
    """Write atomically and return checkpoint SHA-256."""
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be a Path")
    if not isinstance(agent, DualLayerValueAgent):
        raise TypeError("agent must be a DualLayerValueAgent")
    expected_profile_sha = profile_sha256(agent.profile)
    if (
        contract.profile_name != agent.profile.profile.value
        or contract.profile_sha256 != expected_profile_sha
    ):
        raise ValueError("checkpoint contract and agent profile are incompatible")
    payload = _checkpoint_payload(agent, contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return _file_sha256(path)


def _state_mapping(payload: object, field: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint {field} must be a state dictionary")
    return cast(Mapping[str, object], payload)


def _validate_network_state(
    field: str,
    payload: object,
    expected: Mapping[str, torch.Tensor],
) -> None:
    observed = _state_mapping(payload, field)
    if observed.keys() != expected.keys():
        raise ValueError(f"checkpoint {field} network dimensions are incompatible")
    for key, expected_tensor in expected.items():
        observed_tensor = observed[key]
        if not isinstance(observed_tensor, torch.Tensor):
            raise ValueError(f"checkpoint {field} network state is invalid")
        if (
            observed_tensor.shape != expected_tensor.shape
            or observed_tensor.dtype != expected_tensor.dtype
        ):
            raise ValueError(f"checkpoint {field} network dimensions are incompatible")


def _validate_rng_states(payload: dict[str, object]) -> None:
    cpu_state = payload["torch_cpu_rng_state"]
    cuda_states = payload["torch_cuda_rng_states"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.device.type != "cpu":
        raise ValueError("checkpoint torch CPU RNG state is invalid")
    if not isinstance(cuda_states, tuple) or not all(
        isinstance(item, torch.Tensor) and item.device.type == "cpu"
        for item in cuda_states
    ):
        raise ValueError("checkpoint torch CUDA RNG states are invalid")
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count is incompatible")
        devices = list(range(torch.cuda.device_count()))
    else:
        if cuda_states:
            raise ValueError("checkpoint CUDA RNG states require CUDA")
        devices = []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(cpu_state)
            if cuda_states:
                torch.cuda.set_rng_state_all(list(cuda_states))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint torch RNG state is invalid") from exc


def _prevalidate_payload(
    payload: object,
    agent: DualLayerValueAgent,
    expected_contract: RunContract,
    *,
    for_training: bool,
) -> dict[str, object]:
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) for key in payload
    ):
        raise ValueError("checkpoint payload must be a string-keyed mapping")
    state = cast(dict[str, object], payload)
    if set(state) != _PAYLOAD_FIELDS:
        raise ValueError("checkpoint payload fields are incomplete or unknown")
    if state["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is incompatible")

    expected_profile_sha = profile_sha256(agent.profile)
    if (
        expected_contract.profile_name != agent.profile.profile.value
        or expected_contract.profile_sha256 != expected_profile_sha
    ):
        raise ValueError("expected contract and destination profile are incompatible")
    expected_contract_dict = _canonical_contract(expected_contract)
    expected_contract_sha = contract_sha256(expected_contract)
    if (
        state["contract"] != expected_contract_dict
        or state["contract_sha256"] != expected_contract_sha
    ):
        raise ValueError("checkpoint contract is incompatible")
    if (
        state["profile"] != agent.profile.to_dict()
        or state["profile_sha256"] != expected_profile_sha
    ):
        raise ValueError("checkpoint profile is incompatible")
    if type(state["double_dqn"]) is not bool or (
        state["double_dqn"] is not agent.double_dqn
    ):
        raise ValueError("checkpoint double_dqn flag is incompatible")

    for field in ("global_update_step", "decision_count"):
        value = state[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"checkpoint {field} is invalid")
    epsilon = state["epsilon"]
    if type(epsilon) not in (int, float) or not math.isfinite(cast(float, epsilon)):
        raise ValueError("checkpoint epsilon is invalid")
    if not 0.0 <= float(cast(float, epsilon)) <= 1.0:
        raise ValueError("checkpoint epsilon is invalid")

    for field, module in (
        ("upper_online", agent.upper_online),
        ("upper_target", agent.upper_target),
        ("lower_online", agent.lower_online),
        ("lower_target", agent.lower_target),
    ):
        _validate_network_state(field, state[field], module.state_dict())

    preflight = DualLayerValueAgent(
        agent.profile,
        seed=agent.seed,
        device=agent.device,
        double_dqn=agent.double_dqn,
    )
    try:
        preflight.upper_online.load_state_dict(
            cast(Mapping[str, torch.Tensor], state["upper_online"]), strict=True
        )
        preflight.upper_target.load_state_dict(
            cast(Mapping[str, torch.Tensor], state["upper_target"]), strict=True
        )
        preflight.lower_online.load_state_dict(
            cast(Mapping[str, torch.Tensor], state["lower_online"]), strict=True
        )
        preflight.lower_target.load_state_dict(
            cast(Mapping[str, torch.Tensor], state["lower_target"]), strict=True
        )
        if for_training:
            preflight.upper_optimizer.load_state_dict(
                cast(dict[str, object], state["upper_optimizer"])
            )
            preflight.lower_optimizer.load_state_dict(
                cast(dict[str, object], state["lower_optimizer"])
            )
            preflight.set_rng_state(
                cast(tuple[object, ...], state["agent_rng_state"])
            )
            preflight.replay.load_state_dict(
                cast(dict[str, object], state["replay_state"])
            )
            _validate_rng_states(state)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint state is invalid: {exc}") from exc
    return state


def _load_networks(
    agent: DualLayerValueAgent,
    state: dict[str, object],
) -> None:
    for field, module in (
        ("upper_online", agent.upper_online),
        ("upper_target", agent.upper_target),
        ("lower_online", agent.lower_online),
        ("lower_target", agent.lower_target),
    ):
        module.load_state_dict(
            cast(Mapping[str, torch.Tensor], state[field]), strict=True
        )


def load_checkpoint(
    path: Path,
    agent: DualLayerValueAgent,
    expected_contract: RunContract,
    *,
    for_training: bool,
) -> str:
    """Load strictly, return SHA-256, and force epsilon=0 when not training."""
    if not isinstance(path, Path):
        raise TypeError("checkpoint path must be a Path")
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(agent, DualLayerValueAgent):
        raise TypeError("agent must be a DualLayerValueAgent")
    if type(expected_contract) is not RunContract:
        raise TypeError("expected_contract must be a RunContract")
    if type(for_training) is not bool:
        raise ValueError("for_training must be a boolean")
    checkpoint_sha = _file_sha256(path)
    try:
        raw_payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"unable to load checkpoint with weights_only=True: {exc}"
        ) from exc
    state = _prevalidate_payload(
        raw_payload,
        agent,
        expected_contract,
        for_training=for_training,
    )
    _load_networks(agent, state)
    if for_training:
        agent.upper_optimizer.load_state_dict(
            cast(dict[str, object], state["upper_optimizer"])
        )
        agent.lower_optimizer.load_state_dict(
            cast(dict[str, object], state["lower_optimizer"])
        )
        agent.replay.load_state_dict(
            cast(dict[str, object], state["replay_state"])
        )
        agent.set_rng_state(cast(tuple[object, ...], state["agent_rng_state"]))
        agent.global_update_step = cast(int, state["global_update_step"])
        agent.decision_count = cast(int, state["decision_count"])
        agent.epsilon = float(cast(float, state["epsilon"]))
        agent.upper_online.train()
        agent.upper_target.train()
        agent.lower_online.train()
        agent.lower_target.train()
        torch.random.set_rng_state(cast(torch.Tensor, state["torch_cpu_rng_state"]))
        cuda_states = cast(tuple[torch.Tensor, ...], state["torch_cuda_rng_states"])
        if cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))
    else:
        agent.replay.clear()
        agent.epsilon = 0.0
        agent.upper_online.eval()
        agent.upper_target.eval()
        agent.lower_online.eval()
        agent.lower_target.eval()
    return checkpoint_sha
