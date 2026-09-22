"""Simulation backends.

``g1_mujoco``
    Two G1 models — the real one from MuJoCo Menagerie (default) and a mesh-free
    rebuild of NVIDIA's — driven with the SONIC deployment's PD gains.
"""

from .g1_mujoco import (
    CONTROL_HZ,
    GAINS_NATIVE,
    GAINS_SONIC,
    SOURCE_MENAGERIE,
    SOURCE_NVIDIA,
    G1Sim,
    menagerie_model,
    nvidia_meshfree_model,
)

__all__ = [
    "CONTROL_HZ",
    "GAINS_NATIVE",
    "GAINS_SONIC",
    "SOURCE_MENAGERIE",
    "SOURCE_NVIDIA",
    "G1Sim",
    "menagerie_model",
    "nvidia_meshfree_model",
]
