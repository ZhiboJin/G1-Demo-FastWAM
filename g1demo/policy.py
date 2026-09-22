"""Sources of G1 action chunks.

Everything downstream of here consumes ``[T, action_dim]`` physical-unit
references at 50 Hz, so the whole-body controller does not care whether they
came from FastWAM or from a script.

* :class:`FastWAMPolicy` wraps the real FastWAM model. It is the intended
  production source, and is exercised only when G1 weights exist -- upstream has
  not released a G1 checkpoint, so the demo ships scripted sources instead.
* :class:`ScriptedReference` produces smooth, repeatable whole-body motions.
  These are explicitly *not* learned policies; they exist so the SONIC
  encoder/decoder and the physics loop can be validated before training.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import contract
from .paths import CONFIGS, add_fastwam_to_path
from .sonic_params import default_angles, joint_names

DEFAULT_HORIZON = 100  # 2 s at 50 Hz


# ---------------------------------------------------------------------------
# Scripted references
# ---------------------------------------------------------------------------
def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Raised-cosine ramp from 0 to 1 with zero slope at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


@dataclass
class ScriptedReference:
    """A repeating scripted whole-body motion, used as a FastWAM stand-in."""

    motion: str = "standing"
    horizon: int = DEFAULT_HORIZON
    amplitude: float = 1.0
    name: str = "scripted"

    def __post_init__(self) -> None:
        if self.motion not in _MOTIONS:
            raise ValueError(
                f"Unknown motion {self.motion!r}. Available: {', '.join(sorted(_MOTIONS))}"
            )
        if self.horizon < 46:
            raise ValueError(
                f"horizon must be at least 46 frames for SONIC mode 0, got {self.horizon}"
            )
        self.name = f"scripted:{self.motion}"

    def actions(self, sim=None, instruction: str | None = None) -> np.ndarray:
        c = contract()
        names = joint_names()
        # `flat` maps each MuJoCo-ordered joint onto its slot in the flat action
        # vector, so targets must be looked up by MuJoCo index, not contract index.
        flat = c.flat_indices_for(names)
        position = {name: index for index, name in enumerate(names)}

        chunk = np.zeros((self.horizon, c.action_dim), dtype=np.float64)
        chunk[:, flat] = default_angles()
        for joint, values in _MOTIONS[self.motion](self.horizon, self.amplitude).items():
            if joint not in position:
                raise ValueError(f"Unknown joint {joint!r} in motion {self.motion!r}")
            chunk[:, flat[position[joint]]] = values
        return chunk


def _ramp(horizon: int, start: float, end: float) -> np.ndarray:
    """Smoothly blend from ``start`` to ``end`` over the chunk."""
    return start + (end - start) * _smoothstep(np.linspace(0.0, 1.0, horizon))


def _standing(horizon: int, amplitude: float) -> dict[str, np.ndarray]:
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


_MOTIONS = {
    "standing": _standing,
    "arms_up": _arms_up,
    "squat": _squat,
    "wave": _wave,
}


# ---------------------------------------------------------------------------
# Real FastWAM
# ---------------------------------------------------------------------------
class FastWAMPolicy:
    """Adapter around the upstream FastWAM model for G1 whole-body references.

    FastWAM predicts one action chunk of ``[T, action_dim]`` from camera images,
    a language instruction and proprioception. Here ``action_dim`` is exactly the
    G1 contract width (29 joints + hands + root), because the model's action head
    is sized from the contract.

    The hand channels the model predicts are *not* fed to SONIC: they bypass the
    encoder, matching NVIDIA's deployment.

    Not runnable as shipped: upstream has released LIBERO and RoboTwin
    checkpoints but no G1 policy, so there are no weights for a 34-wide G1 action
    head. The class exists so that swapping in a trained checkpoint is a
    one-line change at the call site.
    """

    def __init__(self, checkpoint: Path | str | None = None,
                 config: Path | str | None = None,
                 normalizer_stats: Path | str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16",
                 action_horizon: int = 50) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.config = Path(config) if config else CONFIGS / "fastwam_g1.yaml"
        self.normalizer_stats = Path(normalizer_stats) if normalizer_stats else None
        self.device = device
        self.dtype = dtype
        self.action_horizon = action_horizon
        self.model = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"fastwam:{self.checkpoint.name if self.checkpoint else 'unloaded'}"

    def load(self) -> None:
        """Instantiate the model and load a G1 checkpoint.

        Raises with an actionable message when the pieces are absent, rather than
        failing later inside a forward pass.
        """
        import torch
        import yaml

        if self.checkpoint is None or not self.checkpoint.exists():
            raise FileNotFoundError(
                "No G1 FastWAM checkpoint. Upstream FastWAM ships LIBERO/RoboTwin "
                "weights only; those action heads are 7/14 wide and cannot drive a "
                "G1. Train a 34-wide head first, or use ScriptedReference to "
                "exercise the SONIC half of the pipeline."
            )
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
            torch_dtype=getattr(torch, self.dtype),
            **{key: value for key, value in config.items() if key != "_target_"},
        )
        state = torch.load(self.checkpoint, map_location="cpu")
        self.model.load_checkpoint(state)
        self.model.to(self.device).eval()

    def set_normalizer(self, mean, scale) -> None:
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
        if self._mean is None or self._scale is None:
            raise RuntimeError(
                "No action normalizer set. FastWAM predicts normalized actions; "
                "call set_normalizer() with the training statistics first."
            )
        return np.asarray(actions, dtype=np.float64) * self._scale + self._mean

    def actions(self, sim, instruction: str | None = None) -> np.ndarray:
        """Predict one chunk. Requires :meth:`load` and :meth:`set_normalizer`."""
        raise NotImplementedError(
            "FastWAM inference needs camera observations, a text prompt and fitted "
            "G1 normalization statistics. Wire this up once a G1 checkpoint exists; "
            "the pipeline from the action chunk onwards is already implemented and "
            "is what the scripted reference exercises."
        )
