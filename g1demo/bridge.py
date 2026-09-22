"""Export action chunks as SONIC joint-reference CSVs for the official C++ deck.

This is the bridge to NVIDIA's *own* simulator and robot, and it is the shortest
path to the measurement this project still owes: running the same reference
through the official ``deploy.sh sim`` and comparing its action buffer against
:class:`g1demo.sonic.decoder.SonicDecoder`.

The C++ deployment consumes a motion clip, not a token. This writes that clip::

    joint_pos.csv    [T, 29]  IsaacLab order, radians, absolute
    joint_vel.csv    [T, 29]  IsaacLab order, rad/s
    body_pos.csv     [T, 3]   root position (x/y pinned, z supplied)
    body_quat.csv    [T, 4]   root orientation, wxyz
    metadata.txt     frame count, as the loader expects
    hands.json       hand commands, sidecar only

Hands are a sidecar because SONIC's motion references do not drive them.

Unlike the version this replaces, nothing here needs a GR00T-WholeBodyControl
checkout: the joint permutation, limits and timings come from
``configs/g1_sonic_params.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .contract import contract
from .sonic_params import (
    default_angles,
    joint_lower,
    joint_names,
    joint_upper,
    mujoco_to_isaaclab,
)

#: SONIC reference clips run at the policy rate.
FPS = 50


def _write_csv(path: Path, array: np.ndarray, columns: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(array)


def split_chunk(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Split ``[T, action_dim]`` into IsaacLab-ordered joints, root, and hands."""
    c = contract()
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != c.action_dim:
        raise ValueError(f"Expected actions shaped [T, {c.action_dim}], got {array.shape}")
    if len(array) < 2:
        raise ValueError("Need at least two frames to derive velocities")
    if not np.isfinite(array).all():
        raise ValueError("Actions contain NaN or infinity")

    names = joint_names()
    mujoco = array[:, c.flat_indices_for(names)]

    lower, upper = joint_lower(), joint_upper()
    scale = np.maximum(1.0, np.abs(upper))
    tolerance = 1e-4 * scale
    outside = (mujoco < lower - tolerance) | (mujoco > upper + tolerance)
    if outside.any():
        frame, joint = np.argwhere(outside)[0]
        raise ValueError(
            f"Frame {frame}: {names[joint]}={mujoco[frame, joint]:.4f} rad is outside "
            f"the G1 limit [{lower[joint]:.4f}, {upper[joint]:.4f}]"
        )

    # The CSV convention is IsaacLab order; convert from MuJoCo order.
    root = array[:, c["root"].slice]
    hands = {side: array[:, c[f"{side}_end_effector"].slice] for side in ("left", "right")}
    return mujoco_to_isaaclab(mujoco), root, hands


def _root_quaternion(root: np.ndarray) -> np.ndarray:
    """Roll/pitch angles plus integrated yaw rate -> wxyz quaternions."""
    yaw = np.cumsum(root[:, 2]) / FPS
    yaw -= yaw[0]
    roll, pitch = root[:, 0], root[:, 1]
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.column_stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def export(actions: np.ndarray, output: Path, root_height: float = 0.793,
           origin: str = "unspecified") -> Path:
    """Write one SONIC reference clip. Returns the output directory."""
    output = Path(output)
    joints, root, hands = split_chunk(actions)
    output.mkdir(parents=True, exist_ok=True)

    velocity = np.gradient(joints, 1.0 / FPS, axis=0)
    quaternion = _root_quaternion(root)
    position = np.tile([0.0, 0.0, root_height], (len(joints), 1))

    _write_csv(output / "joint_pos.csv", joints, [f"joint_{i}" for i in range(29)])
    _write_csv(output / "joint_vel.csv", velocity, [f"joint_vel_{i}" for i in range(29)])
    _write_csv(output / "body_pos.csv", position, ["body_0_x", "body_0_y", "body_0_z"])
    _write_csv(
        output / "body_quat.csv",
        quaternion,
        ["body_0_w", "body_0_x", "body_0_y", "body_0_z"],
    )
    (output / "metadata.txt").write_text(
        f"Metadata for: {output.name}\nBody part indexes:\n[0]\n"
        f"Total timesteps: {len(joints)}\n"
    )
    (output / "hands.json").write_text(json.dumps(
        {"fps": FPS, "frames": [{side: hands[side][frame].tolist() for side in hands}
                                for frame in range(len(joints))]}, indent=2))
    (output / "bridge_manifest.json").write_text(json.dumps({
        "source_action_shape": list(np.shape(actions)),
        "fps": FPS,
        "joint_order": "isaaclab",
        "default_pose_reference": default_angles().tolist(),
        "root_height_m": root_height,
        "root_policy": "fixed XY; roll/pitch absolute; yaw rate integrated",
        "sonic_mode": "joint reference clip (encode mode 0)",
        "hands": "sidecar only; SONIC references do not drive the hands",
        "action_origin": origin,
        "prerequisite": "requires an encoder+decoder pair matching this reference",
    }, indent=2))
    return output
