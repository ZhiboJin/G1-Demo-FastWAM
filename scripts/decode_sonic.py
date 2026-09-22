"""Decode one control tick from a SONIC token plus robot state history; no hardware I/O.

Symmetric counterpart to ``encode_sonic.py``: that file produces the 64-value
token, this one turns the token back into 29 IsaacLab-order joint actions.

The state history is supplied as a single consistent snapshot repeated across the
10-frame window (``--num-frames`` is fixed at 10 by the v1.1 observation layout).
Replace this with real logged history when feeding a live controller.
"""
import argparse

import numpy as np

from g1_model import ROOT
from sonic_decoder import NUM_FRAMES, SonicDecoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True, help=".npz from encode_sonic.py, or raw token .npy")
    p.add_argument("--decoder", default=str(ROOT / "checkpoints/sonic_v1_1/model_decoder.onnx"))
    p.add_argument("--joint-pos", type=float, nargs=29, required=True,
                   help="Measured joint positions relative to default, IsaacLab order")
    p.add_argument("--joint-vel", type=float, nargs=29, default=None,
                   help="Measured joint velocities; defaults to zeros")
    p.add_argument("--last-action", type=float, nargs=29, default=None,
                   help="Previous policy action; defaults to zeros")
    p.add_argument("--base-ang-vel", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--robot-quat", type=float, nargs=4, required=True,
                   help="Current base orientation, wxyz")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    loaded = np.load(a.packet)
    token = loaded["token"] if hasattr(loaded, "files") else loaded
    zeros = np.zeros((NUM_FRAMES, 29), dtype=np.float32)

    def history(values, fallback):
        if values is None:
            return fallback
        return np.tile(np.asarray(values, dtype=np.float32), (NUM_FRAMES, 1))

    decoder = SonicDecoder(a.decoder)
    action = decoder.decode(
        token=token,
        joint_pos_history=history(a.joint_pos, zeros),
        joint_vel_history=history(a.joint_vel, zeros),
        last_action_history=history(a.last_action, zeros),
        base_ang_vel_history=np.tile(np.asarray(a.base_ang_vel, dtype=np.float32), (NUM_FRAMES, 1)),
        base_quat_wxyz=a.robot_quat,
    )
    np.save(a.out, action)
    print(f"action[{action.shape[0]}] range [{action.min():.4f}, {action.max():.4f}] -> {a.out}")


if __name__ == "__main__":
    main()
