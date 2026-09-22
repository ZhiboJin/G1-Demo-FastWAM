"""SONIC v1.1 encoder: motion reference -> 64-D whole-body token.

This mirrors the ``g1`` branch (encode mode 0) of NVIDIA's released ONNX graph.
It is a *deployment* adapter, not a differentiable training module.

Interface (verified against the shipped graph)::

    obs_dict        float32 [1, 1751]   ->   encoded_tokens  float32 [1, 64]

The 64 values are FSQ-quantized motion features. The graph contains the
quantizer, so **never quantize again** downstream.

What goes in
------------
The reference is 10 samples of the 29 body joints, taken every 5 control ticks
(100 ms apart) starting at ``frame``, i.e. a 0.9 s window at 50 Hz. Joint values
are **absolute** angles in **IsaacLab** order -- the SONIC encoder does *not*
subtract the default pose. Joint velocities are derived from the reference
trajectory by centred differences.

The root orientation is supplied as relative rotation matrices (first two
columns, flattened row-wise) against the robot's current heading. Ten of those
make up the 60-value orientation block.

What is intentionally left zero
-------------------------------
The observation vector has one flat structural slot per observation in
``observation_config.yaml``, including branches for other modalities. Mode 0
uses only the first four observations, so the teleop and SMPL branches stay
zero. ``encoder_mode_4`` is written as ``[mode, 0, 0, 0]`` -- the C++
``GatherEncoderMode`` writes the scalar mode followed by zeros, so mode 0 is
four zeros, *not* a one-hot vector.

Hands never enter this graph: they are carried alongside the token.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contract import contract
from ..paths import CHECKPOINTS
from ..sonic_params import (
    joint_lower,
    joint_names,
    joint_upper,
    mujoco_to_isaaclab,
)

#: Ordered encoder observation slots, exactly as ``observation_config.yaml``
#: lists them for ``sonic_v1_1``. Names are checked against the shipped YAML so
#: this can never silently drift from the graph it is feeding.
LAYOUT: tuple[tuple[str, int], ...] = (
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_anchor_orientation_heading_10frame_step5", 60),
    ("motion_anchor_orientation_heading", 6),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_heading_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
)

INPUT_DIM = sum(width for _, width in LAYOUT)  # 1751
TOKEN_DIM = 64
FPS = 50
NUM_FRAMES = 10
STEP = 5
#: Sample offsets, in control ticks, of the 10 reference frames.
OFFSETS = np.arange(NUM_FRAMES) * STEP
#: Frames needed per encode: the last offset plus one.
LOOKAHEAD = int(OFFSETS[-1]) + 1

_MODE = np.flatnonzero([name == "encoder_mode_4" for name, _ in LAYOUT])[0]
_SLOT = {}
_cursor = 0
for _name, _width in LAYOUT:
    _SLOT[_name] = slice(_cursor, _cursor + _width)
    _cursor += _width

DEFAULT_ENCODER = CHECKPOINTS / "model_encoder.onnx"
DEFAULT_OBSERVATION_CONFIG = CHECKPOINTS / "observation_config.yaml"


def reference_rotation(root: np.ndarray, initial_yaw: float) -> np.ndarray:
    """Rotation matrices for the reference root channels.

    ``root`` is ``[T, 3]`` of absolute roll, pitch and yaw **rate** (rad, rad,
    rad/s). Yaw is integrated at :data:`FPS`; the first sample uses
    ``initial_yaw`` so that replanning can preserve world-heading continuity
    instead of silently snapping the reference back to zero.

    Returns ``[T, 3, 3]`` rotation matrices ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    root = np.asarray(root, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"Root block must be [T, 3], got {root.shape}")
    yaw = initial_yaw + np.concatenate(([0.0], np.cumsum(root[:-1, 2]) / FPS))
    roll, pitch = root[:, 0], root[:, 1]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack(
        [
            cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr,
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def heading_yaw(quat_wxyz: np.ndarray) -> float:
    """Yaw of a unit quaternion in ``wxyz`` order."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _unit_quaternion(quat_wxyz) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Robot quaternion must be 4 finite values in wxyz order")
    norm = float(np.linalg.norm(quat))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise ValueError(f"Robot quaternion must be unit length, got |q| = {norm:.6f}")
    return quat / norm


def encode_mode0(root: np.ndarray, robot_quat_wxyz: np.ndarray,
                 initial_reference_yaw: float, frame: int = 0) -> np.ndarray:
    """Build the 60-value orientation reference block for the 10 sampled frames.

    ``root`` spans the whole chunk because yaw is integrated from the chunk's
    first frame; the 10 reference samples are taken afterwards, so a chunk used
    from a later ``frame`` still carries the correct absolute heading.
    """
    rotation = reference_rotation(root, initial_reference_yaw)[frame + OFFSETS]
    yaw = heading_yaw(_unit_quaternion(robot_quat_wxyz))
    c, s = np.cos(yaw), np.sin(yaw)
    inv_heading = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    relative = inv_heading @ rotation
    # NVIDIA: first two columns of R, flattened row-wise (R00,R01,R10,R11,R20,R21).
    return relative[:, :, :2].reshape(-1)


def pack_observation(actions: np.ndarray, robot_quat_wxyz, initial_reference_yaw: float,
                     frame: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pack one encoder input from a physical-unit action chunk.

    ``actions`` is ``[T, action_dim]`` of absolute joint angles (radians), hand
    commands, and root roll/pitch/yaw-rate.

    Returns ``(observation[1, 1751], hands)``. The hands are returned because they
    travel *beside* the token rather than through the graph.
    """
    c = contract()
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != c.action_dim:
        raise ValueError(
            f"Expected actions shaped [T, {c.action_dim}], got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain NaN or infinity")

    if not isinstance(frame, (int, np.integer)) or frame < 0:
        raise ValueError(f"frame must be a non-negative integer, got {frame!r}")
    if frame + OFFSETS[-1] >= len(actions):
        raise ValueError(
            f"SONIC mode 0 needs {LOOKAHEAD} samples from frame {frame}, "
            f"but the chunk has {len(actions)}. No implicit padding is applied."
        )

    # Contract order -> MuJoCo order (one vectorised gather), then IsaacLab order.
    flat = c.flat_indices_for(joint_names())
    mujoco = actions[:, flat]
    _check_limits(mujoco)
    isaac = mujoco_to_isaaclab(mujoco)
    velocity_isaac = np.gradient(isaac, 1.0 / FPS, axis=0)

    # Hands bypass the encoder entirely; they are carried beside the token.
    hands = {
        side: actions[frame, c[f"{side}_end_effector"].slice].astype(np.float32)
        for side in ("left", "right")
    }
    root = actions[:, c["root"].slice]
    orientation = encode_mode0(root, robot_quat_wxyz, initial_reference_yaw, frame)

    obs = np.zeros((1, INPUT_DIM), dtype=np.float32)
    # encoder_mode_4 slot stays all zeros: mode 0 is [0, 0, 0, 0], not one-hot.
    obs[0, _SLOT["motion_joint_positions_10frame_step5"]] = isaac[frame + OFFSETS].reshape(-1)
    obs[0, _SLOT["motion_joint_velocities_10frame_step5"]] = velocity_isaac[
        frame + OFFSETS
    ].reshape(-1)
    obs[0, _SLOT["motion_anchor_orientation_heading_10frame_step5"]] = orientation
    return obs, hands


def _check_limits(mujoco: np.ndarray) -> None:
    """Reject references outside the G1 joint limits (with a small tolerance)."""
    lower, upper = joint_lower(), joint_upper()
    scale = np.maximum(1.0, np.abs(upper))
    tolerance = 1e-4 * scale
    below = mujoco < (lower - tolerance)
    above = mujoco > (upper + tolerance)
    if below.any() or above.any():
        frame, joint = np.argwhere(below | above)[0]
        raise ValueError(
            f"Reference joint '{joint_names()[joint]}' = {mujoco[frame, joint]:.4f} rad "
            f"at frame {frame} is outside the G1 limit "
            f"[{lower[joint]:.4f}, {upper[joint]:.4f}]"
        )


class SonicEncoder:
    """The pretrained SONIC v1.1 encoder graph (encode mode 0, G1)."""

    def __init__(self, encoder_path: Path | str | None = None,
                 observation_config: Path | str | None = None,
                 providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        encoder_path = Path(encoder_path or DEFAULT_ENCODER)
        observation_config = Path(observation_config or DEFAULT_OBSERVATION_CONFIG)
        if not encoder_path.exists():
            raise FileNotFoundError(
                f"Missing SONIC encoder: {encoder_path}\n"
                f"Download it with:  python -m g1demo.cli download-sonic"
            )
        self._check_observation_config(observation_config)

        self.session = ort.InferenceSession(
            str(encoder_path), providers=providers or ["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected a single SONIC encoder input/output")
        self.input, self.output = inputs[0], outputs[0]
        if list(self.input.shape) != [1, INPUT_DIM] or self.input.type != "tensor(float)":
            raise ValueError(
                f"Encoder input mismatch: {self.input.shape}/{self.input.type}, "
                f"expected [1, {INPUT_DIM}] tensor(float)"
            )
        if list(self.output.shape) != [1, TOKEN_DIM]:
            raise ValueError(
                f"Encoder output mismatch: {self.output.shape}, expected [1, {TOKEN_DIM}]"
            )

    @staticmethod
    def _check_observation_config(path: Path) -> None:
        """Confirm the shipped YAML describes the layout this adapter packs."""
        import yaml

        if not path.exists():
            raise FileNotFoundError(f"Missing SONIC observation config: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["encoder"]
        names = [
            item["name"]
            for item in config["encoder_observations"]
            if item.get("enabled", True)
        ]
        expected = [name for name, _ in LAYOUT]
        if names != expected:
            raise ValueError(
                "The shipped observation_config.yaml lists a different encoder layout "
                "than this adapter implements.\n"
                f"  config: {names}\n  code:   {expected}"
            )
        if int(config["dimension"]) != TOKEN_DIM:
            raise ValueError(f"Config token dimension {config['dimension']} != {TOKEN_DIM}")
        if config.get("use_fp16", False):
            raise ValueError("This adapter targets the exact float32 v1.1 deployment layout")

    def encode(self, actions: np.ndarray, robot_quat_wxyz, initial_reference_yaw: float,
               frame: int = 0) -> dict[str, np.ndarray]:
        """Encode one control tick.

        Returns ``{"token": [64], "left_hand": [n], "right_hand": [n]}``. Hand
        commands are passed through untouched; they are not part of the graph.
        """
        obs, hands = pack_observation(
            actions, robot_quat_wxyz, initial_reference_yaw, frame
        )
        token = self.session.run([self.output.name], {self.input.name: obs})[0]
        if token.shape != (1, TOKEN_DIM) or not np.isfinite(token).all():
            raise RuntimeError(f"SONIC encoder returned an invalid token: {token.shape}")
        # The graph already contains the FSQ quantizer. Do not quantize again.
        return {
            "token": token[0].astype(np.float32),
            "left_hand": hands["left"],
            "right_hand": hands["right"],
        }
