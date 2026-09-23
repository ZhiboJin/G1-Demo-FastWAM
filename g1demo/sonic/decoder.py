"""SONIC v1.1 decoder: 64-D token + proprioceptive history -> 29 joint actions.

This mirrors NVIDIA's released ``model_decoder.onnx``::

    obs_dict   float32 [1, 994]   ->   action   float32 [1, 29]

The action is **not** a torque and **not** a joint angle. It is a per-joint
normalized offset that the deployment turns into a position target with::

    q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * action_scale[i]

:func:`g1demo.sonic_params.q_target_from_action` implements exactly that,
including the IsaacLab -> MuJoCo permutation. Actions here are in **IsaacLab**
order, matching the graph's output.

Observation layout (994 values)
-------------------------------
The layout is derived from the ordered list in the shipped
``observation_config.yaml`` together with the per-name widths in the deployment
observation registry; it is not guessed. The sum matches the ONNX-declared
input width.

===========================================  =====  ==========================
Observation                                  Dim    Notes
===========================================  =====  ==========================
``token_state``                                 64  encoder output, FSQ applied
``his_base_angular_velocity_10frame_step1``     30  10 x 3, oldest -> newest
``his_body_joint_positions_10frame_step1``     290  10 x 29, IsaacLab order
``his_body_joint_velocities_10frame_step1``    290  10 x 29
``his_last_actions_10frame_step1``             290  10 x 29
``his_gravity_dir_10frame_step1``               30  10 x 3
===========================================  =====  ==========================

History blocks are time-major with the frame axis running **oldest to newest**.
``his_body_joint_positions`` is measured joint position *relative to the default
standing pose* (the C++ subtracts ``default_angles``); use
:func:`g1demo.sonic_params.measured_to_isaaclab_relative` to build it.

Hands are absent by design: the decoder has no hand outputs, and hand commands
travel on a separate channel in NVIDIA's deployment.

The official deployment computes gravity from each logged frame's quaternion.
Pass ``base_quat_history`` for that behavior; a single current quaternion is
accepted for standalone probes and repeated across the window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..paths import CHECKPOINTS

NUM_FRAMES = 10
NUM_JOINTS = 29
TOKEN_DIM = 64
INPUT_DIM = 994
OUTPUT_DIM = 29
FPS = 50

DEFAULT_DECODER = CHECKPOINTS / "model_decoder.onnx"

#: Ordered decoder observation blocks: (name, frames, width per frame).
LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("token_state", 1, TOKEN_DIM),
    ("his_base_angular_velocity_10frame_step1", NUM_FRAMES, 3),
    ("his_body_joint_positions_10frame_step1", NUM_FRAMES, NUM_JOINTS),
    ("his_body_joint_velocities_10frame_step1", NUM_FRAMES, NUM_JOINTS),
    ("his_last_actions_10frame_step1", NUM_FRAMES, NUM_JOINTS),
    ("his_gravity_dir_10frame_step1", NUM_FRAMES, 3),
)

#: Slice of each block within the flat observation, derived from :data:`LAYOUT`.
SLICES: dict[str, slice] = {}
_cursor = 0
for _name, _frames, _width in LAYOUT:
    SLICES[_name] = slice(_cursor, _cursor + _frames * _width)
    _cursor += _frames * _width
assert _cursor == INPUT_DIM, f"decoder layout sums to {_cursor}, expected {INPUT_DIM}"


def gravity_direction(base_quat_wxyz) -> np.ndarray:
    """World gravity expressed in the body frame: ``R(q)^T @ (0, 0, -1)``.

    Mirrors ``GatherHisGravityDir``, which rotates ``{0,0,-1}`` by the conjugate
    of the base quaternion.
    """
    quat = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Base quaternion must be 4 finite values in wxyz order")
    norm = float(np.linalg.norm(quat))
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise ValueError(f"Base quaternion must be unit length, got |q| = {norm:.6f}")
    w, x, y, z = quat / norm
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return (rotation.T @ np.array([0.0, 0.0, -1.0])).astype(np.float32)


def _history(values, name: str, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (NUM_FRAMES, width):
        raise ValueError(
            f"{name} must have shape ({NUM_FRAMES}, {width}) oldest->newest, "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array.reshape(-1)


def pack_observation(token, joint_pos_history, joint_vel_history,
                     last_action_history, base_ang_vel_history,
                     base_quat_wxyz, base_quat_history=None) -> np.ndarray:
    """Assemble the ``[1, 994]`` decoder input.

    ``joint_pos_history``, ``joint_vel_history`` and ``last_action_history`` are
    ``[10, 29]`` in IsaacLab order, oldest first; ``base_ang_vel_history`` is
    ``[10, 3]``. ``joint_pos_history`` must already be relative to the default
    pose.
    """
    token = np.asarray(token, dtype=np.float32).reshape(-1)
    if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
        raise ValueError(f"Token must be {TOKEN_DIM} finite values, got {token.shape}")

    # The official state logger stores one orientation per historical frame.
    # Retain the single-orientation fallback for standalone verification calls.
    if base_quat_history is None:
        gravity = np.tile(gravity_direction(base_quat_wxyz), (NUM_FRAMES, 1))
    else:
        quats = np.asarray(base_quat_history, dtype=np.float64)
        if quats.shape != (NUM_FRAMES, 4):
            raise ValueError(
                f"base_quat_history must have shape ({NUM_FRAMES}, 4) oldest->newest, "
                f"got {quats.shape}"
            )
        gravity = np.stack([gravity_direction(quat) for quat in quats])

    obs = np.zeros((1, INPUT_DIM), dtype=np.float32)
    obs[0, SLICES["token_state"]] = token
    obs[0, SLICES["his_base_angular_velocity_10frame_step1"]] = _history(
        base_ang_vel_history, "base_ang_vel_history", 3
    )
    obs[0, SLICES["his_body_joint_positions_10frame_step1"]] = _history(
        joint_pos_history, "joint_pos_history", NUM_JOINTS
    )
    obs[0, SLICES["his_body_joint_velocities_10frame_step1"]] = _history(
        joint_vel_history, "joint_vel_history", NUM_JOINTS
    )
    obs[0, SLICES["his_last_actions_10frame_step1"]] = _history(
        last_action_history, "last_action_history", NUM_JOINTS
    )
    obs[0, SLICES["his_gravity_dir_10frame_step1"]] = gravity.reshape(-1)
    return obs


class SonicDecoder:
    """The pretrained SONIC v1.1 ``g1_dyn`` decoder."""

    def __init__(self, decoder_path: Path | str | None = None,
                 providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        decoder_path = Path(decoder_path or DEFAULT_DECODER)
        if not decoder_path.exists():
            raise FileNotFoundError(
                f"Missing SONIC decoder: {decoder_path}\n"
                f"Download it with:  python -m g1demo.cli download-sonic"
            )
        self.session = ort.InferenceSession(
            str(decoder_path), providers=providers or ["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected a single SONIC decoder input/output")
        self.input, self.output = inputs[0], outputs[0]
        if list(self.input.shape) != [1, INPUT_DIM] or self.input.type != "tensor(float)":
            raise ValueError(
                f"Decoder input mismatch: {self.input.shape}/{self.input.type}, "
                f"expected [1, {INPUT_DIM}] tensor(float)"
            )
        if list(self.output.shape) != [1, OUTPUT_DIM]:
            raise ValueError(
                f"Decoder output mismatch: {self.output.shape}, expected [1, {OUTPUT_DIM}]"
            )

    def decode(self, token, joint_pos_history, joint_vel_history,
               last_action_history, base_ang_vel_history,
               base_quat_wxyz, base_quat_history=None) -> np.ndarray:
        """Run one decoder step.

        Returns the raw ``(29,)`` action in **IsaacLab** order. Convert it to a
        position target with :func:`g1demo.sonic_params.q_target_from_action`.
        """
        obs = pack_observation(
            token=token,
            joint_pos_history=joint_pos_history,
            joint_vel_history=joint_vel_history,
            last_action_history=last_action_history,
            base_ang_vel_history=base_ang_vel_history,
            base_quat_wxyz=base_quat_wxyz,
            base_quat_history=base_quat_history,
        )
        action = self.session.run([self.output.name], {self.input.name: obs})[0]
        if action.shape != (1, OUTPUT_DIM) or not np.isfinite(action).all():
            raise RuntimeError(f"SONIC decoder returned an invalid action: {action.shape}")
        return action[0].astype(np.float64)
