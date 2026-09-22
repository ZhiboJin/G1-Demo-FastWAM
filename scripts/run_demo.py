"""Two reaching tasks using MuJoCo dynamics, wrist Jacobian IK and joint PD."""
import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_model import build
from validate_contract import load_contract
from action_adapter import pack


def run(task, out, viewer=False, snapshot=None, action_out=None):
    model = mujoco.MjModel.from_xml_path(str(build()))
    data = mujoco.MjData(model)
    cfg, offsets, dim = load_contract()
    if model.nu != 29 or model.nq != 29:
        raise RuntimeError(f"Expected fixed-base 29-DoF G1, got nq={model.nq}, nu={model.nu}")
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i for i in range(model.njnt)}
    for name in (n for group in cfg["joint_groups"].values() for n in group):
        if name not in names:
            raise RuntimeError(f"Joint contract mismatch: {name}")
    # One actor per native joint. The source MJCF uses torque motors.
    actuator_for_joint = {}
    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i, 0])
        actuator_for_joint[joint_id] = i
    target_q = np.zeros(model.nq)
    mujoco.mj_forward(model, data)
    sides = [task] if task != "both" else ["left", "right"]
    results = []
    scripted_actions = []
    frames = []
    renderer = mujoco.Renderer(model, height=480, width=640) if snapshot else None
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.8]
    camera.distance = 2.0
    camera.azimuth = 45
    camera.elevation = -12
    launch = None
    if viewer:
        from mujoco import viewer as mj_viewer
        launch = mj_viewer.launch_passive(model, data)
    try:
        for side in sides:
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, side + "_ee")
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, side + "_target")
            # Place the task target near the current wrist so it is reachable.
            start = data.site_xpos[ee_id].copy()
            goal = start + np.array([0.10, 0.04 if side == "left" else -0.04, 0.08])
            model.site_pos[target_id] = goal
            arm_joints = [names[n] for n in cfg["joint_groups"][side + "_arm"]]
            best = float("inf")
            for step in range(1000):
                mujoco.mj_forward(model, data)
                error = goal - data.site_xpos[ee_id]
                best = min(best, float(np.linalg.norm(error)))
                if step % 10 == 0:
                    jac = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jac, None, ee_id)
                    j = jac[:, arm_joints]
                    dq = j.T @ np.linalg.solve(j @ j.T + 0.002 * np.eye(3), 0.35 * error)
                    for idx, delta in zip(arm_joints, dq):
                        lo, hi = model.jnt_range[idx]
                        target_q[idx] = np.clip(target_q[idx] + delta, lo + 0.02, hi - 0.02)
                    scripted_actions.append(pack({
                        group: np.array([target_q[names[name]] for name in joint_names])
                        for group, joint_names in cfg["joint_groups"].items()
                    }))
                for jid in range(model.njnt):
                    aid = actuator_for_joint[jid]
                    torque = 65 * (target_q[jid] - data.qpos[jid]) - 2.8 * data.qvel[jid]
                    data.ctrl[aid] = np.clip(torque, *model.actuator_ctrlrange[aid])
                mujoco.mj_step(model, data)
                if launch:
                    launch.sync()
                    time.sleep(model.opt.timestep)
            distance = float(np.linalg.norm(goal - data.site_xpos[ee_id]))
            if renderer:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
            results.append({"task": f"touch_{side}_target", "distance_m": distance,
                            "best_distance_m": best, "success_at_5cm": distance < 0.05})
        payload = {"controller": "jacobian_ik_pd", "fastwam_inference": False,
                   "sonic_inference": False, "torso_fixed": True,
                   "action_dim": dim, "action_slices": offsets, "results": results}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        if action_out:
            action_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(action_out, np.asarray(scripted_actions, dtype=np.float32))
        if frames:
            from PIL import Image
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.concatenate(frames, axis=1)).save(snapshot)
        print(json.dumps(payload, indent=2))
        return all(row["success_at_5cm"] for row in results)
    finally:
        if launch:
            launch.close()
        if renderer:
            renderer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["left", "right", "both"], default="both")
    p.add_argument("--out", type=Path, default=Path("artifacts/run.json"))
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--snapshot", type=Path, default=Path("artifacts/demo.png"))
    p.add_argument("--action-out", type=Path, default=Path("artifacts/scripted_actions.npy"))
    args = p.parse_args()
    raise SystemExit(0 if run(args.task, args.out, args.viewer, args.snapshot, args.action_out) else 1)
