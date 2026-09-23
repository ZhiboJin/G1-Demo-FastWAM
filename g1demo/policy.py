"""Sources of G1 action chunks.

Everything downstream consumes ``[T, action_dim]`` physical-unit references at
50 Hz, so the control loop does not care whether they came from FastWAM or from a
script. Both sources implement :class:`g1demo.loop.ActionSource`.

* :class:`ScriptedReference` produces smooth, repeatable whole-body motions. These
  are explicitly *not* learned policies; they exist so the SONIC chain and the
  physics loop can be exercised before a G1 checkpoint exists.
* :class:`FastWAMPolicy` wraps the real FastWAM inference API. It requires a
  trained G1 checkpoint, matching normalization statistics and a camera image.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .contract import contract
from .paths import CONFIGS, add_fastwam_to_path
from .sonic.encoder import LOOKAHEAD
from .sonic_params import default_angles, joint_names

DEFAULT_HORIZON = 100  # 2 s at 50 Hz

#: A motion maps (horizon, amplitude) to per-joint target trajectories.
Motion = Callable[[int, float], dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Scripted motions
# ---------------------------------------------------------------------------
def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Raised-cosine ramp from 0 to 1 with zero slope at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _ramp(horizon: int, start: float, end: float) -> np.ndarray:
    """Smoothly blend from ``start`` to ``end`` over the chunk."""
    return start + (end - start) * _smoothstep(np.linspace(0.0, 1.0, horizon))


def _standing(horizon: int, amplitude: float) -> dict[str, np.ndarray]:  # noqa: ARG001
    """Hold the default pose: no joint is overridden.

    The unused arguments are the shared :data:`Motion` signature.
    """
    return {}


def _arms_up(horizon: int, amplitude: float) -> dict[str, np.ndarray]:
    """Raise both arms overhead and hold."""
    return {
        "left_shoulder_pitch_joint": _ramp(horizon, 0.2, 0.2 + 1.2 * amplitude),
        "right_shoulder_pitch_joint": _ramp(horizon, 0.2, 0.2 + 1.2 * amplitude),
        "left_elbow_joint": _ramp(horizon, 0.6, 0.6 + 0.5 * amplitude),
        "right_elbow_joint": _ramp(horizon, 0.6, 0.6 + 0.5 * amplitude),
    }


def _squat(horizon: int, amplitude: float) -> dict[str, np.ndarray]:
    """A whole-body squat: hips, knees and ankles move together."""
    depth = 0.55 * amplitude
    return {
        "left_hip_pitch_joint": _ramp(horizon, -0.312, -0.312 - depth),
        "right_hip_pitch_joint": _ramp(horizon, -0.312, -0.312 - depth),
        "left_knee_joint": _ramp(horizon, 0.669, 0.669 + 1.1 * depth),
        "right_knee_joint": _ramp(horizon, 0.669, 0.669 + 1.1 * depth),
        "left_ankle_pitch_joint": _ramp(horizon, -0.363, -0.363 - 0.55 * depth),
        "right_ankle_pitch_joint": _ramp(horizon, -0.363, -0.363 - 0.55 * depth),
    }


def _wave(horizon: int, amplitude: float) -> dict[str, np.ndarray]:
    """Raise the right arm and wave the elbow, holding the left arm still."""
    phase = np.linspace(0.0, 3.0 * np.pi, horizon)
    raised = _smoothstep(np.linspace(0.0, 1.0, horizon))
    return {
        "right_shoulder_pitch_joint": 0.2 + 1.1 * amplitude * raised,
        "right_elbow_joint": 0.6 + 0.6 * amplitude * raised
        + 0.25 * amplitude * raised * np.sin(phase),
    }


#: Named motions available to ``ScriptedReference`` and the ``--motion`` flag.
MOTIONS: dict[str, Motion] = {
    "standing": _standing,
    "arms_up": _arms_up,
    "squat": _squat,
    "wave": _wave,
}


@dataclass
class ScriptedReference:
    """A scripted whole-body motion, used as a stand-in for FastWAM.

    It emits the same ``[T, action_dim]`` interface a trained policy would, so the
    SONIC encoder, decoder and control loop cannot tell the difference.
    """

    motion: str = "standing"
    horizon: int = DEFAULT_HORIZON
    amplitude: float = 1.0
    name: str = field(default="scripted", init=False)

    def __post_init__(self) -> None:
        if self.motion not in MOTIONS:
            known = ", ".join(sorted(MOTIONS))
            raise ValueError(f"Unknown motion {self.motion!r}. Available: {known}")
        if self.horizon < LOOKAHEAD:
            raise ValueError(
                f"horizon must be at least {LOOKAHEAD} frames for SONIC mode 0, "
                f"got {self.horizon}"
            )
        self.name = f"scripted:{self.motion}"

    def actions(self, sim=None, instruction: str | None = None) -> np.ndarray:  # noqa: ARG002
        """Return ``[horizon, action_dim]`` references starting at the default pose.

        ``sim`` and ``instruction`` are ignored: a scripted source needs no
        observation. They are part of the
        :class:`~g1demo.loop.ActionSource` interface so a learned policy can be
        substituted without changing the control loop.
        """
        c = contract()
        names = joint_names()
        # `flat` maps each MuJoCo-ordered joint to its slot in the flat action
        # vector, so targets are looked up by MuJoCo index, not contract index.
        flat = c.flat_indices_for(names)
        position = {name: index for index, name in enumerate(names)}

        chunk = np.zeros((self.horizon, c.action_dim), dtype=np.float64)
        chunk[:, flat] = default_angles()
        for joint, values in MOTIONS[self.motion](self.horizon, self.amplitude).items():
            if joint not in position:
                raise ValueError(f"Unknown joint {joint!r} in motion {self.motion!r}")
            chunk[:, flat[position[joint]]] = values
        return chunk


# ---------------------------------------------------------------------------
# Real FastWAM
# ---------------------------------------------------------------------------
class FastWAMPolicy:
    """Adapter around the upstream FastWAM model for G1 whole-body references.

    FastWAM predicts one action chunk of ``[T, action_dim]`` from camera images, a
    language instruction and proprioception. Here ``action_dim`` is exactly the G1
    contract width (29 joints + hands + root), because the model's action head is
    sized from the contract. The hand channels it predicts are *not* fed to SONIC;
    they bypass the encoder, matching NVIDIA's deployment.

    No G1 weights or statistics are bundled. The adapter is ready for a trained
    checkpoint but fails clearly until those inputs are supplied.
    """

    def __init__(self, checkpoint: Path | str | None = None,
                 config: Path | str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16",
                 action_horizon: int = DEFAULT_HORIZON,
                 observation_provider: Callable | None = None) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.config = Path(config) if config else CONFIGS / "fastwam_g1.yaml"
        self.device = device
        self.dtype = dtype
        self.action_horizon = action_horizon
        self.observation_provider = observation_provider
        self.model = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._proprio_mean: np.ndarray | None = None
        self._proprio_scale: np.ndarray | None = None
        if action_horizon < LOOKAHEAD:
            raise ValueError(f"action_horizon must be at least {LOOKAHEAD}")

    @property
    def name(self) -> str:
        return f"fastwam:{self.checkpoint.name if self.checkpoint else 'unloaded'}"

    def load(self) -> None:
        """Instantiate the model and load a G1 checkpoint.

        Raises with an actionable message when the pieces are absent, rather than
        failing later inside a forward pass.
        """
        if self.checkpoint is None or not self.checkpoint.exists():
            raise FileNotFoundError(
                "No G1 FastWAM checkpoint. Upstream FastWAM ships LIBERO/RoboTwin "
                "weights only; those action heads are 7/14 wide and cannot drive a "
                f"G1. Train a {contract().action_dim}-wide head first, or use ScriptedReference to exercise "
                "the SONIC half of the pipeline."
            )
        import torch
        import yaml

        add_fastwam_to_path()

        c = contract()
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        for branch in ("video_dit_config", "action_dit_config"):
            width = int(config[branch]["action_dim"])
            if width != c.action_dim:
                raise ValueError(
                    f"configs/fastwam_g1.yaml {branch}.action_dim={width} does not "
                    f"match the action contract width {c.action_dim}. Regenerate the "
                    f"config or fix configs/action_space.json."
                )
        if int(config["proprio_dim"]) != c.num_joints:
            raise ValueError(
                f"configs/fastwam_g1.yaml proprio_dim={config['proprio_dim']} does not "
                f"match the {c.num_joints} measured joints"
            )

        from fastwam.runtime import create_fastwam

        self.model = create_fastwam(
            device=self.device,
            model_dtype=getattr(torch, self.dtype),
            **{key: value for key, value in config.items()
               if key not in {"_target_", "compile_training_denoise"}},
        )
        self.model.load_checkpoint(self.checkpoint)
        self.model.to(self.device).eval()

    def set_normalizer(self, mean, scale) -> None:
        """Supply the per-channel statistics FastWAM's action head was trained with.

        LIBERO and RoboTwin statistics must not be reused: the channels differ.
        """
        c = contract()
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if mean.shape != (c.action_dim,) or scale.shape != (c.action_dim,):
            raise ValueError(
                f"Normalizer must have {c.action_dim} entries, got "
                f"{mean.shape} and {scale.shape}"
            )
        if (scale <= 0).any() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Normalizer mean/scale must be finite with positive scale")
        self._mean, self._scale = mean, scale

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        """Map normalized model output back to physical units."""
        if self._mean is None or self._scale is None:
            raise RuntimeError(
                "No action normalizer set. FastWAM predicts normalized actions; "
                "call set_normalizer() with the training statistics first."
            )
        return np.asarray(actions, dtype=np.float64) * self._scale + self._mean

    def set_proprio_normalizer(self, mean, scale) -> None:
        """Supply training statistics for the 29 measured body joints."""
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if mean.shape != (contract().num_joints,) or scale.shape != mean.shape:
            raise ValueError(f"Proprio normalizer needs {contract().num_joints} entries")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Proprio mean/scale must be finite with positive scale")
        self._proprio_mean, self._proprio_scale = mean, scale

    def predict(self, image: np.ndarray, joint_positions: np.ndarray,
                instruction: str) -> np.ndarray:
        """Return physical-unit G1 references from one RGB frame and joint state.

        ``image`` is uint8 HWC RGB; joints are absolute radians in SONIC's
        MuJoCo order. FastWAM receives RGB in [-1, 1] and normalized proprio.
        """
        if self.model is None:
            raise RuntimeError("Load a trained G1 FastWAM checkpoint first")
        if self._mean is None or self._scale is None:
            raise RuntimeError("Set G1 action normalization statistics first")
        if self._proprio_mean is None or self._proprio_scale is None:
            raise RuntimeError("Set G1 proprio normalization statistics first")
        if not instruction or not instruction.strip():
            raise ValueError("A language instruction is required")
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be uint8 RGB [H,W,3]")
        if image.shape[0] % 16 or image.shape[1] % 16:
            raise ValueError("image height and width must be multiples of 16")
        joints = np.asarray(joint_positions, dtype=np.float64)
        if joints.shape != (contract().num_joints,) or not np.isfinite(joints).all():
            raise ValueError(f"joint_positions must be {contract().num_joints} finite angles")

        import torch

        rgb = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        rgb = rgb.float().div(127.5).sub(1.0)
        proprio = torch.from_numpy(
            ((joints - self._proprio_mean) / self._proprio_scale).astype(np.float32)
        )
        with torch.no_grad():
            result = self.model.infer_action(
                prompt=instruction,
                input_image=rgb,
                action_horizon=self.action_horizon,
                proprio=proprio,
            )
        action = result["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action)
        if action.shape == (1, self.action_horizon, contract().action_dim):
            action = action[0]
        if action.shape != (self.action_horizon, contract().action_dim):
            raise ValueError(f"FastWAM returned malformed action chunk {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("FastWAM returned non-finite actions")
        return self.denormalize(action)

    def actions(self, sim, instruction: str | None = None) -> np.ndarray:
        """Adapt the simulation loop using a caller-supplied RGB camera source."""
        if self.observation_provider is None:
            raise RuntimeError("Pass observation_provider(sim) -> uint8 RGB image")
        return self.predict(self.observation_provider(sim), sim.joint_pos, instruction or "")
