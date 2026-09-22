"""Encode one control tick from a physical-unit chunk; no hardware I/O."""
import argparse
import numpy as np
from g1_model import ROOT
from sonic_encoder import SonicEncoder

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--actions", required=True, help="Physical-unit [T,D] .npy at 50 Hz")
    p.add_argument("--encoder", default=str(ROOT / "checkpoints/sonic_v1_1/model_encoder.onnx"))
    p.add_argument("--observation-config", default=str(ROOT / "checkpoints/sonic_v1_1/observation_config.yaml"))
    p.add_argument("--robot-quat", type=float, nargs=4, required=True)
    p.add_argument("--reference-yaw", type=float, required=True, help="World yaw at chunk frame 0")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    encoder = SonicEncoder(a.encoder, a.observation_config)
    result = encoder.encode(np.load(a.actions), a.robot_quat, a.reference_yaw, a.frame)
    np.savez(a.out, **result)
    print({name:list(value.shape) for name,value in result.items()})
