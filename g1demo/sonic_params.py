"""Authoritative G1 constants for the SONIC policy, loaded from a generated file.

These values (joint orders, default standing pose, action scales, PD gains) live
in the SONIC C++ deployment header. ``tools/extract_sonic_params.py`` evaluates
that header's own expressions and writes ``configs/g1_sonic_params.json``; this
module loads it and exposes the conversions.

Two joint orderings coexist and mixing them up silently corrupts every target:

``mujoco`` (aka URDF order)
    The order used by MuJoCo, the G1 URDF, and this demo's simulator.
``isaaclab``
    The order used *inside* the SONIC networks. The encoder expects references
    in this order and the decoder emits actions in it.

Two different joint-position conventions also coexist, and they are **not** the
same:

* The encoder's reference input takes **absolute** joint angles.
* The decoder's proprioceptive history takes joint positions **relative to the
  default standing pose** (the C++ subtracts ``default_angles``).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable

import numpy as np

from .paths import sonic_params_path

NUM_JOINTS = 29


class ParamsMismatch(ValueError):
    """Raised when the generated parameter file is inconsistent or stale."""


@lru_cache(maxsize=1)
def params() -> dict:
    """Load and validate ``configs/g1_sonic_params.json`` (cached)."""
    path = sonic_params_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate it with:\n"
            f"  python tools/extract_sonic_params.py --sonic-repo /path/to/GR00T-WholeBodyControl"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))

    def array(key: str) -> np.ndarray:
        values = np.asarray(raw[key], dtype=np.float64)
        if values.shape != (NUM_JOINTS,):
            raise ParamsMismatch(f"{key} has shape {values.shape}, expected ({NUM_JOINTS},)")
        if not np.isfinite(values).all():
            raise ParamsMismatch(f"{key} contains NaN or infinity")
        return values

    for key in ("isaaclab_to_mujoco", "mujoco_to_isaaclab"):
        values = np.asarray(raw[key], dtype=np.int64)
        if sorted(values.tolist()) != list(range(NUM_JOINTS)):
            raise ParamsMismatch(f"{key} is not a permutation of 0..{NUM_JOINTS - 1}")
        raw[key] = values

    for key in ("default_angles", "action_scale", "kp", "kd", "effort_limit",
                "joint_lower", "joint_upper"):
        raw[key] = array(key)
    if (raw["action_scale"] <= 0).any():
        raise ParamsMismatch("action_scale must be strictly positive")
    if (raw["kp"] <= 0).any() or (raw["kd"] < 0).any():
        raise ParamsMismatch("PD gains must be positive (kd non-negative)")
    if (raw["effort_limit"] <= 0).any():
        raise ParamsMismatch("effort_limit must be strictly positive")
    if (raw["joint_lower"] >= raw["joint_upper"]).any():
        raise ParamsMismatch("joint_lower must be strictly below joint_upper")

    names = list(raw["joint_names"])
    if len(names) != NUM_JOINTS or len(set(names)) != NUM_JOINTS:
        raise ParamsMismatch("joint_names must list 29 distinct joints")

    # The two permutations must be exact inverses.
    forward, inverse = raw["isaaclab_to_mujoco"], raw["mujoco_to_isaaclab"]
    if not np.array_equal(inverse[forward], np.arange(NUM_JOINTS)):
        raise ParamsMismatch("isaaclab_to_mujoco and mujoco_to_isaaclab disagree")

    return raw


def joint_names() -> tuple[str, ...]:
    """Joint names in **MuJoCo** order."""
    return tuple(params()["joint_names"])


def name_to_mujoco_index() -> dict[str, int]:
    """Map joint name -> MuJoCo index."""
    return {name: index for index, name in enumerate(joint_names())}


def default_angles() -> np.ndarray:
    """Default standing pose, MuJoCo order, radians."""
    return params()["default_angles"]


def action_scale() -> np.ndarray:
    """Per-joint action scale, MuJoCo order, radians per action unit."""
    return params()["action_scale"]


def kp() -> np.ndarray:
    """Per-joint position gain, MuJoCo order."""
    return params()["kp"]


def kd() -> np.ndarray:
    """Per-joint velocity gain, MuJoCo order."""
    return params()["kd"]


def effort_limit() -> np.ndarray:
    """Per-joint torque limit (N·m), MuJoCo order, as configured in IsaacLab."""
    return params()["effort_limit"]


def joint_lower() -> np.ndarray:
    """Per-joint lower position limit, MuJoCo order, radians."""
    return params()["joint_lower"]


def joint_upper() -> np.ndarray:
    """Per-joint upper position limit, MuJoCo order, radians."""
    return params()["joint_upper"]


def validate_against_contract(joint_names_contract: Iterable[str]) -> None:
    """Check that the contract's joint set matches the SONIC model's."""
    contract_set = set(joint_names_contract)
    sonic_set = set(joint_names())
    if contract_set != sonic_set:
        raise ParamsMismatch(
            "Action contract and SONIC joint lists differ: "
            f"only in contract {sorted(contract_set - sonic_set)}, "
            f"only in SONIC {sorted(sonic_set - contract_set)}"
        )


def mujoco_to_isaaclab(values_mujoco: np.ndarray) -> np.ndarray:
    """Reorder a MuJoCo-order array (…, 29) into IsaacLab order."""
    return np.asarray(values_mujoco)[..., params()["mujoco_to_isaaclab"]]


def isaaclab_to_mujoco(values_isaaclab: np.ndarray) -> np.ndarray:
    """Reorder an IsaacLab-order array (…, 29) into MuJoCo order."""
    return np.asarray(values_isaaclab)[..., params()["isaaclab_to_mujoco"]]


def measured_to_isaaclab_relative(q_mujoco: np.ndarray) -> np.ndarray:
    """Measured joint positions -> IsaacLab order, relative to the default pose.

    This matches the C++: ``q_isaac[i] = q_mujoco[perm[i]] - default[perm[i]]``.
    It is the convention the SONIC **decoder** expects for
    ``his_body_joint_positions``. (The **encoder** wants absolute angles, so do
    not use this for its motion reference.)
    """
    q = np.asarray(q_mujoco, dtype=np.float64)
    if q.shape[-1] != NUM_JOINTS:
        raise ValueError(f"Expected last axis {NUM_JOINTS}, got {q.shape}")
    permutation = params()["mujoco_to_isaaclab"]
    return q[..., permutation] - default_angles()[permutation]


def q_target_from_action(action_isaaclab: np.ndarray) -> np.ndarray:
    """Apply the official SONIC action-to-target convention.

    Straight from ``g1_deploy_onnx_ref.cpp``::

        q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * action_scale[i]

    where ``i`` indexes MuJoCo order and ``action`` is the decoder output in
    IsaacLab order. Returns joint position targets in **MuJoCo** order, radians.

    The result is *not* clamped here: saturation is a real symptom worth
    surfacing to the caller rather than hiding.
    """
    action = np.asarray(action_isaaclab, dtype=np.float64)
    if action.shape[-1] != NUM_JOINTS:
        raise ValueError(f"Expected last axis {NUM_JOINTS}, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Decoder action contains NaN or infinity")
    return default_angles() + action[..., params()["isaaclab_to_mujoco"]] * action_scale()


def target_saturation(q_target_mujoco: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Boolean mask of joints whose target falls outside ``[lower, upper]``."""
    target = np.asarray(q_target_mujoco)
    return (target < np.asarray(lower)) | (target > np.asarray(upper))
