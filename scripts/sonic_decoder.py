"""SONIC v1.1 decoder observation packing and closed-loop token->joint execution.

Completes the other half of the demo's SONIC integration:

    encoder: [1,1751] -> [1,64] token           (scripts/sonic_encoder.py)
    decoder: [1,994]  -> [1,29] G1 joint action (this file)

The 994-D layout is not guessed. It is derived from the ordered list in the
authoritative ``observation_config.yaml`` (delivered alongside the ONNX graph)
combined with the per-name dimensions declared in the deployment observation
registry in ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp``
(the table containing ``his_gravity_dir_10frame_step1 -> 30``):

    token_state                              64
    his_base_angular_velocity_10frame_step1  30
    his_body_joint_positions_10frame_step1  290
    his_body_joint_velocities_10frame_step1 290
    his_last_actions_10frame_step1          290
    his_gravity_dir_10frame_step1            30
                                          -----
                                           994

History blocks are ``num_frames x 29``, time-major. The gather functions are
called with their default ``newest_first=false``, and ``StateLogger::GetLatest``
reverses its newest-first ring output when that flag is false, so the frame axis
runs OLDEST -> NEWEST. Joint order is IsaacLab order, because the C++ side
converts the policy output with ``floatarr[isaaclab_to_mujoco[i]]``.

Hand commands are NOT part of this decoder. They remain a separate sidecar, as
in the encoder half, and never enter either graph.

Local source verification used ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/
g1_deploy_onnx_ref.cpp`` (observation registry and the ``GatherHis*`` gatherers)
and ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/state_logger.cpp``
(``GetLatest`` ordering).
"""
from pathlib import Path

import numpy as np

from project_paths import ROOT

NUM_FRAMES = 10
NUM_JOINTS = 29
TOKEN_DIM = 64
DECODER_INPUT_DIM = 994
DECODER_OUTPUT_DIM = 29
FPS = 50

# Default location, mirroring where the encoder graph already lives.
DEFAULT_DECODER = ROOT / "checkpoints/sonic_v1_1/model_decoder.onnx"

# Ordered decoder observation blocks: (name, frames, per-frame width)
LAYOUT = [
    ("token_state", 1, TOKEN_DIM),
    ("his_base_angular_velocity_10frame_step1", NUM_FRAMES, 3),
    ("his_body_joint_positions_10frame_step1", NUM_FRAMES, NUM_JOINTS),
    ("his_body_joint_velocities_10frame_step1", NUM_FRAMES, NUM_JOINTS),
    ("his_last_actions_10frame_step1", NUM_FRAMES, NUM_JOINTS),
    ("his_gravity_dir_10frame_step1", NUM_FRAMES, 3),
]

# Slices derived from LAYOUT so they can never silently drift from it.
SLICES = {}
_cursor = 0
for _name, _frames, _width in LAYOUT:
    _size = _frames * _width
    SLICES[_name] = slice(_cursor, _cursor + _size)
    _cursor += _size
assert _cursor == DECODER_INPUT_DIM, f"layout sums to {_cursor}, expected {DECODER_INPUT_DIM}"


def history_block(values):
    """Validate an (NUM_FRAMES, width) oldest->newest history block and flatten it."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != NUM_FRAMES:
        raise ValueError(f"History block must be [{NUM_FRAMES}, W], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("History block contains NaN or infinity")
    return values.reshape(-1)


def gravity_dir(base_quat_wxyz):
    """Gravity direction in the body frame: R(q)^-1 * (0,0,-1).

    Mirrors ``GatherHisGravityDir``: ``quat_rotate_d(quat_conjugate_d(base_quat),
    {0,0,-1})``. The conjugate of a unit quaternion is its inverse, so this is the
    world gravity vector rotated into the body frame.
    """
    quat = np.asarray(base_quat_wxyz, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Require a finite base quaternion in wxyz order")
    norm = np.linalg.norm(quat)
    if not np.isclose(norm, 1.0, atol=1e-3):
        raise ValueError("Base quaternion must be unit length")
    w, x, y, z = quat / norm
    world = np.array([0.0, 0.0, -1.0])
    r = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    return (r.T @ world).astype(np.float32)


def pack_decoder_observation(token, joint_pos_history, joint_vel_history,
                            last_action_history, base_ang_vel_history,
                            base_quat_wxyz):
    """Build the [1,994] decoder input from a token and oldest->newest history.

    Parameters
    ----------
    token : array-like, shape (64,)
        Output of the pretrained SONIC v1.1 encoder (FSQ already applied).
    joint_pos_history, joint_vel_history, last_action_history :
        array-like, shape (10, 29), IsaacLab joint order, oldest -> newest.
        ``joint_pos_history`` is measured joint position relative to default,
        matching the training observation.
    base_ang_vel_history : array-like, shape (10, 3)
        Measured base angular velocity, oldest -> newest.
    base_quat_wxyz : array-like, shape (4,)
        Current base orientation. Gravity direction is computed from it here
        rather than supplied.
    """
    token = np.asarray(token, dtype=np.float32)
    if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
        raise ValueError(f"Token must be finite with shape ({TOKEN_DIM},)")

    pos = history_block(joint_pos_history)
    vel = history_block(joint_vel_history)
    act = history_block(last_action_history)
    ang = history_block(base_ang_vel_history)
    if ang.size != NUM_FRAMES * 3:
        raise ValueError("Base angular velocity history must be [10, 3]")

    # Deployment gathers gravity for all 10 frames from StateLogger. A caller
    # holding only the current orientation must pass it per frame instead.
    gravity = np.tile(gravity_dir(base_quat_wxyz), (NUM_FRAMES, 1)).reshape(-1)

    obs = np.zeros((1, DECODER_INPUT_DIM), dtype=np.float32)
    obs[0, SLICES["token_state"]] = token
    obs[0, SLICES["his_base_angular_velocity_10frame_step1"]] = ang
    obs[0, SLICES["his_body_joint_positions_10frame_step1"]] = pos
    obs[0, SLICES["his_body_joint_velocities_10frame_step1"]] = vel
    obs[0, SLICES["his_last_actions_10frame_step1"]] = act
    obs[0, SLICES["his_gravity_dir_10frame_step1"]] = gravity
    return obs


class SonicDecoder:
    """Pretrained SONIC v1.1 decoder: 64-D token + state history -> 29 joints.

    Output is the policy's raw action in IsaacLab joint order. It is NOT the final
    motor command: the C++ deployment applies per-index scaling and adds
    ``default_angles`` (``motor_command.q_target = default_angles + action*scale``).
    That scaling convention is deliberately not applied here, because it is not yet
    validated against the official controller.
    """

    def __init__(self, decoder_path=None, providers=None):
        import onnxruntime as ort
        decoder_path = Path(decoder_path or DEFAULT_DECODER)
        if not decoder_path.exists():
            raise FileNotFoundError(
                f"Missing SONIC decoder: {decoder_path}. Download it with "
                f"`python download_from_hf.py --sonic-v1-1` from the SONIC repo, "
                f"or set the path explicitly."
            )
        self.session = ort.InferenceSession(
            str(decoder_path), providers=providers or ["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected a single SONIC decoder input/output")
        self.input, self.output = inputs[0], outputs[0]
        if self.input.shape != [1, DECODER_INPUT_DIM] or self.input.type != "tensor(float)":
            raise ValueError(
                f"Decoder input mismatch: {self.input.shape}/{self.input.type}, "
                f"expected [1,{DECODER_INPUT_DIM}] tensor(float)"
            )
        if self.output.shape != [1, DECODER_OUTPUT_DIM]:
            raise ValueError(f"Decoder output mismatch: {self.output.shape}")

    def decode(self, **obs_kwargs):
        obs = pack_decoder_observation(**obs_kwargs)
        action = self.session.run([self.output.name], {self.input.name: obs})[0]
        if action.shape != (1, DECODER_OUTPUT_DIM) or not np.isfinite(action).all():
            raise RuntimeError("SONIC decoder returned an invalid action")
        return action[0]
