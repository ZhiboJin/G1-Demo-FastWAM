"""The pretrained SONIC v1.1 whole-body controller.

``encoder``
    Motion reference -> 64-D token (see :mod:`g1demo.sonic.encoder`).
``decoder``
    Token + proprioceptive history -> 29 joint actions (see
    :mod:`g1demo.sonic.decoder`).

The two graphs are the released ``sonic_v1_1`` ONNX files; both halves are real
pretrained networks, not stand-ins. What is *not* pretrained is FastWAM's G1
action head, which upstream has not released.
"""

from .decoder import SonicDecoder
from .encoder import SonicEncoder

__all__ = ["SonicDecoder", "SonicEncoder"]
