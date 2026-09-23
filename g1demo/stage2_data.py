"""Prepare synchronized G1 episodes for SONIC-latent action learning.

This is a data boundary, not a MotionWAM trainer. It turns physical reference
trajectories into the released SONIC v1.1 64-D FSQ tokens and retains the hand
channels separately. The currently released ONNX graph exposes the token vector,
not MotionWAM's discrete token index, so these are continuous-token targets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import contract
from .sonic.encoder import FPS, LOOKAHEAD, SonicEncoder, heading_yaw


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a scalar string")
    text = str(array.item()).strip()
    if not text:
        raise ValueError(f"{name} cannot be empty")
    return text


def prepare_episode(source: Path | str, output: Path | str,
                    encoder: SonicEncoder | None = None) -> Path:
    """Validate one episode and write aligned observation/latent training rows.

    Input ``.npz`` keys: ``rgb`` uint8 [T,H,W,3], ``joint_pos`` [T,29] in
    MuJoCo order, ``base_quat_wxyz`` [T,4], ``action_ref`` [T,action_dim],
    ``timestamp_s`` [T], scalar strings ``instruction`` and ``task_id``.
    ``action_ref`` is the desired motion trajectory, not the measured joints.
    Output has T-45 labeled frames because SONIC needs 46 reference frames;
    each row retains the physical ``action_ref`` target alongside its token.
    """
    source, output = Path(source), Path(output)
    with np.load(source, allow_pickle=False) as episode:
        required = {"rgb", "joint_pos", "base_quat_wxyz", "action_ref",
                    "timestamp_s", "instruction", "task_id"}
        missing = required - set(episode.files)
        if missing:
            raise ValueError(f"Episode is missing {sorted(missing)}")
        rgb = episode["rgb"]
        joints = np.asarray(episode["joint_pos"], dtype=np.float64)
        quat = np.asarray(episode["base_quat_wxyz"], dtype=np.float64)
        actions = np.asarray(episode["action_ref"], dtype=np.float64)
        timestamp = np.asarray(episode["timestamp_s"], dtype=np.float64)
        instruction = _scalar_string(episode["instruction"], "instruction")
        task_id = _scalar_string(episode["task_id"], "task_id")

    total = len(actions)
    if total < LOOKAHEAD:
        raise ValueError(f"Need at least {LOOKAHEAD} frames, got {total}")
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[0] != total or rgb.shape[-1] != 3:
        raise ValueError("rgb must be uint8 [T,H,W,3]")
    if rgb.shape[1] % 16 or rgb.shape[2] % 16:
        raise ValueError("RGB height and width must be divisible by 16")
    expected = {"joint_pos": (total, 29), "base_quat_wxyz": (total, 4),
                "action_ref": (total, contract().action_dim), "timestamp_s": (total,)}
    for name, array in (("joint_pos", joints), ("base_quat_wxyz", quat),
                        ("action_ref", actions), ("timestamp_s", timestamp)):
        if array.shape != expected[name] or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with shape {expected[name]}")
    if not np.allclose(np.diff(timestamp), 1.0 / FPS, atol=0.002, rtol=0):
        raise ValueError("timestamp_s must be synchronized at 50 Hz (20 ms apart)")
    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-3):
        raise ValueError("base_quat_wxyz must contain unit quaternions")

    encoder = encoder or SonicEncoder()
    rows = total - LOOKAHEAD + 1
    tokens = np.empty((rows, 64), dtype=np.float32)
    left = np.empty((rows, contract().hand_sizes["left"]), dtype=np.float32)
    right = np.empty((rows, contract().hand_sizes["right"]), dtype=np.float32)
    reference_yaw = heading_yaw(quat[0])
    for frame in range(rows):
        packet = encoder.encode(actions, quat[frame], reference_yaw, frame=frame)
        tokens[frame] = packet["token"]
        left[frame] = packet["left_hand"]
        right[frame] = packet["right_hand"]

    # A continuous target for FastWAM's existing action head. This is an
    # engineering baseline, not MotionWAM's discrete-index objective.
    latent_action = np.concatenate((tokens, left, right), axis=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, rgb=rgb[:rows], joint_pos=joints[:rows].astype(np.float32),
        base_quat_wxyz=quat[:rows].astype(np.float32),
        action_ref=actions[:rows].astype(np.float32),
        timestamp_s=timestamp[:rows], sonic_token=tokens,
        left_hand=left, right_hand=right, latent_action=latent_action,
        instruction=np.asarray(instruction), task_id=np.asarray(task_id),
        target_kind=np.asarray("sonic_v1_1_fsq_vector_64"),
        source_frames=np.asarray(total),
    )
    return output
