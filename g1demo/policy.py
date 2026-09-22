"""Sources of G1 action chunks.

Everything downstream consumes ``[T, action_dim]`` physical-unit references at
50 Hz, so the control loop does not care whether they came from FastWAM or from a
script. Both sources implement :class:`g1demo.loop.ActionSource`.

* :class:`ScriptedReference` produces smooth, repeatable whole-body motions. These
  are explicitly *not* learned policies; they exist so the SONIC chain and the
  physics loop can be exercised before a G1 checkpoint exists.
* :class:`FastWAMPolicy` wraps the real FastWAM model. It is the intended
  production source. Upstream has released no G1 checkpoint, so its inference
  entry point is a documented stub rather than a silently broken one.
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

    It emits exactly the same ``[T, 34]`` interface a trained policy would, so the
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

    ``load`` and ``set_normalizer`` are implemented and validated. ``actions`` is a
    documented stub: upstream has released LIBERO and RoboTwin checkpoints whose
    action heads are 7 and 14 wide, and neither can drive a 34-wide G1 head, so
    there is nothing to run yet. Everything downstream of the chunk is implemented
    and is what :class:`ScriptedReference` exercises.
    """

    def __init__(self, checkpoint: Path | str | None = None,
                 config: Path | str | None = None,
                 device: str = "cuda", dtype: str = "bfloat16",
                 action_horizon: int = 50) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.config = Path(config) if config else CONFIGS / "fastwam_g1.yaml"
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
                "G1. Train a 34-wide head first, or use ScriptedReference to exercise "
                "the SONIC half of the pipeline."
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
        self.model.load_checkpoint(torch.load(self.checkpoint, map_location="cpu"))
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

    def actions(self, sim, instruction: str | None = None) -> np.ndarray:
        """Predict one chunk. Requires :meth:`load` and :meth:`set_normalizer`."""
        raise NotImplementedError(
            "FastWAM inference needs camera observations, a text prompt and fitted "
            "G1 normalization statistics. Wire this up once a G1 checkpoint exists: "
            "call self.model.infer_action(...) with the chunk horizon, then "
            "self.denormalize(...). The pipeline from the action chunk onwards is "
            "already implemented, and is what ScriptedReference exercises."
        )
