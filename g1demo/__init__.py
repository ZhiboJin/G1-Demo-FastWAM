"""FastWAM -> SONIC -> Unitree G1: whole-body control demo.

The pipeline this package implements::

    FastWAM  --predicts-->  [T, 46] joint + Dex3 hand + root references
                              |
                              +-- 29 body joints + 3 root channels
                              |     -> SONIC v1.1 encoder (incl. FSQ)
                              |     -> 64-D whole-body latent token
                              |     -> SONIC v1.1 decoder -> 29 joint actions
                              |
                              +-- left/right hand commands -> BYPASS the encoder
                                                             and the decoder

The hands bypass SONIC because its graphs have no hand inputs or outputs: in
NVIDIA's deployment the dexterous-hand commands travel on a separate channel
and never enter the policy.

Modules
-------
``contract``
    The 46-D action layout, loaded once from ``configs/action_space.json``.
``sonic_params``
    Authoritative G1 constants extracted from the SONIC C++ deployment.
``sonic``
    The pretrained SONIC v1.1 encoder and decoder.
``sim``
    A MuJoCo G1 model driven with the same PD gains the real controller uses.
"""

__all__ = ["contract", "paths", "sonic_params"]
__version__ = "0.2.0"
