"""Convert OpenHLM-style FastWAM G1 action chunks to SONIC joint references.

Consumes a real FastWAM action array after G1 posttraining, or scripted_actions.npy
for an interface smoke test. This exports SONIC's documented reference format;
it does not run the SONIC policy or synthesize its 64-D latent tokens.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_adapter import unpack
from validate_contract import load_contract
from project_paths import ROOT, sonic_repo

UPSTREAM = sonic_repo()
URDF = UPSTREAM / "gear_sonic/data/robots/g1/g1_29dof.urdf"
MJCF = UPSTREAM / "gear_sonic/data/robots/g1/g1_29dof_old.xml"
MAPPING = UPSTREAM / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
FPS = 50


def g1_schema():
    cfg, _, dim = load_contract()
    robot = ET.parse(URDF).getroot()
    limits = {}
    for joint in robot.findall("joint"):
        limit = joint.find("limit")
        if joint.get("type") == "revolute" and limit is not None:
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    mj_root = ET.parse(MJCF).getroot()
    mj_joints = [joint for joint in mj_root.find("worldbody").iter("joint")
                 if joint.get("type") != "free"]
    mj_names = [joint.get("name") for joint in mj_joints]
    if len(mj_names) != 29 or set(mj_names) != set(limits):
        raise RuntimeError("G1 URDF and SONIC MJCF joint lists differ")
    for joint in mj_joints:
        mj_limit = tuple(float(x) for x in joint.get("range").split())
        if not np.allclose(mj_limit, limits[joint.get("name")], atol=1e-3):
            raise RuntimeError(f"G1 URDF and SONIC MJCF limits differ for {joint.get('name')}")
    contract_names = [n for group in cfg["joint_groups"].values() for n in group]
    if set(contract_names) != set(mj_names):
        raise RuntimeError("G1 action contract and URDF joint lists differ")
    source = MAPPING.read_text(encoding="utf-8")
    match = re.search(r"mujoco_to_isaaclab\s*=\s*\{([^}]+)\}", source)
    if not match:
        raise RuntimeError("SONIC joint mapping not found")
    isaac_in_mujoco = [int(v) for v in re.findall(r"\d+", match.group(1))]
    if sorted(isaac_in_mujoco) != list(range(29)):
        raise RuntimeError("Invalid SONIC joint permutation")
    return cfg, dim, mj_names, limits, isaac_in_mujoco


def as_joint_and_root(actions):
    cfg, dim, mj_names, limits, isaac_in_mujoco = g1_schema()
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != dim or actions.shape[0] < 2:
        raise ValueError(f"Expected at least two frames of shape [T,{dim}]")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain NaN or infinity")
    mj = np.empty((len(actions), 29))
    root = np.empty((len(actions), 3))
    hands = []
    for t, action in enumerate(actions):
        blocks = unpack(action)
        named = {joint: float(value) for group, names in cfg["joint_groups"].items()
                 for joint, value in zip(names, blocks[group])}
        for j, name in enumerate(mj_names):
            value = named[name]
            lo, hi = limits[name]
            if value < lo - 1e-4 or value > hi + 1e-4:
                raise ValueError(f"Frame {t}: {name}={value:.4f} outside URDF [{lo},{hi}]")
            mj[t, j] = value
        root[t] = blocks["root"]
        hands.append({side: blocks[side + "_end_effector"].tolist() for side in ("left", "right")})
    isaac = mj[:, isaac_in_mujoco]
    return isaac, root, hands, mj_names


def write_csv(path, array, columns):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(array)


def export(actions, output, root_height=0.793, origin="unspecified"):
    q, root, hands, _ = as_joint_and_root(actions)
    output.mkdir(parents=True, exist_ok=True)
    vel = np.gradient(q, 1 / FPS, axis=0)
    yaw = np.cumsum(root[:, 2]) / FPS
    yaw -= yaw[0]
    roll, pitch = root[:, 0], root[:, 1]
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.column_stack((cr*cp*cy + sr*sp*sy,
                            sr*cp*cy - cr*sp*sy,
                            cr*sp*cy + sr*cp*sy,
                            cr*cp*sy - sr*sp*cy))
    pos = np.tile([0.0, 0.0, root_height], (len(q), 1))
    write_csv(output / "joint_pos.csv", q, [f"joint_{j}" for j in range(29)])
    write_csv(output / "joint_vel.csv", vel, [f"joint_vel_{j}" for j in range(29)])
    write_csv(output / "body_pos.csv", pos, ["body_0_x", "body_0_y", "body_0_z"])
    write_csv(output / "body_quat.csv", quat, ["body_0_w", "body_0_x", "body_0_y", "body_0_z"])
    (output / "metadata.txt").write_text(
        f"Metadata for: {output.name}\nBody part indexes:\n[0]\nTotal timesteps: {len(q)}\n")
    (output / "hands.json").write_text(json.dumps({"fps": FPS, "frames": hands}, indent=2))
    (output / "bridge_manifest.json").write_text(json.dumps({
        "source_action_shape": list(np.shape(actions)), "fps": FPS,
        "root_height_m": root_height, "root_policy": "fixed XY; roll/pitch absolute; integrate yaw rate",
        "sonic_mode": "joint reference CSV; encoder mode 0",
        "hands": "sidecar only; not controlled by this SONIC reference",
        "action_origin": origin
    }, indent=2))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--actions", type=Path, required=True, help="[T, action_dim] .npy FastWAM output")
    p.add_argument("--out", type=Path, required=True, help="SONIC reference clip directory")
    p.add_argument("--root-height", type=float, default=0.793)
    p.add_argument("--origin", default="unspecified", help="Provenance label, e.g. scripted_ik or fastwam_inference")
    a = p.parse_args()
    print(export(np.load(a.actions), a.out, a.root_height, a.origin))
