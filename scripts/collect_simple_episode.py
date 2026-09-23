"""Record one successful SIMPLE G1 MP rollout without changing SIMPLE itself.

This is a raw source episode. Its 43-D mixed AMO/CuRobo/Dex3 targets are NOT
the 46-D body/root/Dex3 reference expected by ``prepare-stage2``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ENV_ID = "simple/G1WholebodyBendPickMP-v0"


def collect(simple_root: Path, output: Path, seed: int, max_steps: int) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(simple_root / "src"))
    import gymnasium as gym
    import simple.envs  # noqa: F401 - registers the environments
    from simple.agents.mp import MotionPlannerAgent
    from simple.mp.curobo import CuRoboPlanner

    env = gym.make(ENV_ID, sim_mode="mujoco", headless=True,
                   render_hz=50, max_episode_steps=max_steps)
    frames: dict[str, list[np.ndarray | float]] = {
        "rgb": [], "joint_pos_43": [], "base_pose_xyz_wxyz": [],
        "action_target_43": [], "timestamp_s": [], "reward": [],
    }
    try:
        obs, info = env.reset(seed=seed)
        task = env.unwrapped.task
        robot = task.robot
        joint_names = tuple(robot.joint_names)
        if len(joint_names) != 43:
            raise ValueError(f"Expected SIMPLE's 43-joint Dex3 G1, got {len(joint_names)}")

        planner = CuRoboPlanner(robot=robot, plan_dt=0.01,
                                plan_batch_size=1, easy_motion_gen=False)
        agent = MotionPlannerAgent(task, planner, plan_batch_size=1)
        success = False
        step_count = 0
        for phase in range(8):
            state = agent.synthesize()
            if state is False:
                break
            while step_count < max_steps:
                try:
                    command = agent.get_action(obs, info)
                except StopIteration:
                    break
                obs, reward, terminated, truncated, info = env.step(command)
                # Match SIMPLE's LeRobot recorder: actuator controls, replacing
                # AMO lower-body torques with desired PD targets and the last
                # 14 channels with desired Dex3 hand joint positions.
                target = np.array([robot.actuators[name].ctrl.item()
                                   for name in joint_names], dtype=np.float32)
                target[:15] = np.asarray(robot.pd_target, dtype=np.float32)
                hand = robot.hand_target_qpos()
                target[-14:] = [hand[name] for name in joint_names[-14:]]
                frames["rgb"].append(obs["head_stereo_left"].copy())
                frames["joint_pos_43"].append(obs["joint_qpos"].copy())
                frames["base_pose_xyz_wxyz"].append(
                    np.asarray(env.unwrapped.mujoco.mjData.qpos[:7]).copy())
                frames["action_target_43"].append(target)
                frames["timestamp_s"].append(float(env.unwrapped.mujoco.mjData.time))
                frames["reward"].append(float(reward))
                step_count += 1
                if terminated or truncated:
                    success = bool(terminated)
                    break
            print(f"phase {phase + 1}: {step_count} steps, "
                  f"reward {frames['reward'][-1]:.3f}", flush=True)
            if success or step_count >= max_steps:
                break

        if not success:
            raise RuntimeError(f"No successful episode in {step_count} steps; nothing saved")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            **{key: np.stack(values) for key, values in frames.items()},
            joint_names=np.asarray(joint_names),
            instruction=np.asarray(task.instruction),
            task_id=np.asarray(ENV_ID),
            target_kind=np.asarray("simple_mp_mixed_amo_curobo_dex3_43"),
            seed=np.asarray(seed),
            success=np.asarray(True),
        )
        print(f"saved {step_count} successful frames to {output}", flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "SIMPLE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()
    collect(args.simple_root, args.out, args.seed, args.max_steps)


if __name__ == "__main__":
    main()
