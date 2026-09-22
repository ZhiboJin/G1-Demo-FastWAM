"""Kinematic MuJoCo playback of a SONIC joint-reference clip.

This visualizes the exact CSVs SONIC accepts. It does NOT run SONIC's ONNX policy.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_model import build
from sonic_bridge import g1_schema


def read_csv(path, width):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    array = np.asarray([[float(v) for v in row] for row in rows[1:]], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid {path.name}: expected [T,{width}]")
    return array


def play(clip, snapshot):
    _, _, mj_names, _, isaac_in_mujoco = g1_schema()
    q_isaac = read_csv(clip / "joint_pos.csv", 29)
    q_mj = np.empty_like(q_isaac)
    q_mj[:, isaac_in_mujoco] = q_isaac
    root_pos = read_csv(clip / "body_pos.csv", 3)
    root_quat = read_csv(clip / "body_quat.csv", 4)
    if len(root_pos) != len(q_isaac) or len(root_quat) != len(q_isaac):
        raise ValueError("SONIC reference frame counts differ")
    model = mujoco.MjModel.from_xml_path(str(build()))
    data = mujoco.MjData(model)
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if model_names != mj_names:
        raise RuntimeError("Playback model joint order differs from SONIC reference order")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 0.8]
    camera.distance = 2.0
    camera.azimuth = 45
    camera.elevation = -12
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    try:
        for i in np.linspace(0, len(q_mj) - 1, 4, dtype=int):
            data.qpos[:] = q_mj[i]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(frames, axis=1)).save(snapshot)
    report = {"frames": len(q_isaac), "fps": 50, "joint_count": 29,
              "root_quaternion_norm_range": [float(np.linalg.norm(root_quat, axis=1).min()),
                                             float(np.linalg.norm(root_quat, axis=1).max())],
              "playback": "kinematic_MuJoCo_reference_visualization",
              "sonic_policy_executed": False, "snapshot": str(snapshot)}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, default=Path("artifacts/sonic_reference_playback.png"))
    a = p.parse_args()
    play(a.clip, a.snapshot)
