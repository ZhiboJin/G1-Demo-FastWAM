"""OpenHLM-style action packing and safe extraction of G1 joint targets.

This is an interface adapter, not a SONIC controller. Joint positions are absolute radians.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_contract import load_contract


def pack(blocks):
    _, offsets, dim = load_contract()
    action = np.zeros(dim, dtype=np.float32)
    for name, (lo, hi) in offsets.items():
        value = np.asarray(blocks.get(name, np.zeros(hi - lo)), dtype=np.float32)
        if value.shape != (hi - lo,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite vector of length {hi - lo}")
        action[lo:hi] = value
    return action


def unpack(action):
    _, offsets, dim = load_contract()
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (dim,) or not np.all(np.isfinite(action)):
        raise ValueError(f"Action must be a finite vector of length {dim}")
    return {name: action[lo:hi].copy() for name, (lo, hi) in offsets.items()}


def joint_targets(action):
    cfg, _, _ = load_contract()
    blocks = unpack(action)
    return {joint: float(value)
            for group, names in cfg["joint_groups"].items()
            for joint, value in zip(names, blocks[group])}


if __name__ == "__main__":
    cfg, _, dim = load_contract()
    zero = pack({})
    assert len(zero) == dim and len(joint_targets(zero)) == 29
    assert all(np.array_equal(v, np.zeros_like(v)) for v in unpack(zero).values())
    print(f"Validated {dim}-D action and 29 named G1 joints")
