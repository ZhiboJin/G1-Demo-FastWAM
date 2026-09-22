"""Simulation backends.

``g1_mujoco``
    A MuJoCo G1 built from NVIDIA's official MJCF, actuated with the same PD
    gains the SONIC deployment commands.
"""

from .g1_mujoco import CONTROL_HZ, G1Sim, build_model

__all__ = ["CONTROL_HZ", "G1Sim", "build_model"]
