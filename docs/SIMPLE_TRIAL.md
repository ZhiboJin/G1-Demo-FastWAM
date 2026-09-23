# SIMPLE G1 trial — 2026-09-23

Goal: run `simple/G1WholebodyBendPickMP-v0` in MuJoCo, inspect a real episode,
then convert it for `prepare-stage2`. The environment now runs successfully on
this laptop. One **successful 214-frame raw episode** was recorded; conversion
to the SONIC Stage-2 contract remains open. No training data was fabricated.

## What was tried

- Used the existing Python 3.10 FastWAM environment and official SIMPLE source
  checkout at `robotics/SIMPLE/` (`6d10628794d9c7de4596b4f2afb2c054a637c2bc`).
- Tried `PYTHONPATH=src python -c 'import simple.envs'`. It first failed on
  missing `gymnasium`.
- Installed `gymnasium`, `transforms3d`, `trimesh`, `opencv-python-headless`,
  `dm-control`, `scipy`, and `lxml` in `/tmp/simple_trial_deps` only, then retried
  with `PYTHONPATH=/tmp/simple_trial_deps:src`.
- The import then failed in `simple/robots/mixin.py`: `RuntimeError: curobo not
  installed, uv pip install --groups curobo`. The source imports CuRobo while
  registering robots even if `sim_mode="mujoco"` is planned. SIMPLE's
  installation guide calls CuRobo a must-have dependency.
- Initially, `third_party/curobo` was empty, `nvcc` was not on `PATH`, Git LFS
  was unavailable, and `data/` had no G1 assets. The GPU itself worked:
  `nvidia-smi` saw the RTX 5070 Laptop GPU, and PyTorch 2.7.1+cu128 reported
  `torch.cuda.is_available() == True`. The `CUDA Version: 13.0` line in
  `nvidia-smi` describes driver capability, not an installed compiler.

## Repair and live result

- Created `robotics/SIMPLE/.venv-lite` with Python 3.10 and
  `--system-site-packages`, reusing CUDA-enabled PyTorch without installing
  packages into the FastWAM environment. Installed SIMPLE's MuJoCo-path Python
  dependencies there. Installed CUDA 12.8 `nvcc` and Git LFS in
  `robotics/SIMPLE/.cuda-toolkit` with conda. `nvcc --version` reports 12.8.93.
  The CUDA toolchain's `targets/x86_64-linux/include` must be on `CPATH` when
  building this CuRobo fork; the conda layout lacks those headers at its root.
- Initialized SIMPLE's pinned CuRobo and AMO submodules. Built the pinned
  CuRobo source for `TORCH_CUDA_ARCH_LIST=12.0`, matching the RTX 5070. All
  five CuRobo CUDA extensions import after loading `torch`.
- Used Git LFS to fetch the three AMO checkpoints from the SIMPLE checkout.
  SIMPLE downloaded the G1 robot, HSSD scene, and GraspNet object assets on
  first use. The source checkout remains clean; the local environments are
  ignored through `.git/info/exclude`.
- `gym.make('simple/G1WholebodyBendPickMP-v0', sim_mode='mujoco',
  headless=True, render_hz=50)` now succeeds. `reset(seed=0)` returned
  `joint_qpos (43,)`, four `uint8 (360,640,3)` camera images, instruction
  `bend to pick up the cracker box`, and a valid MuJoCo floating-base pose.
- The actual CuRobo planner initialized; `MotionPlannerAgent.synthesize()`
  reached its first `phase_break`, produced a `loco_command`, and `env.step()`
  completed. A later bounded full-episode run reached reward 0.919 and
  terminated successfully at step 213.
- Warp 1.7.0 logs a `cuDeviceGetUuid` driver-entry warning, then reports the
  RTX 5070 CUDA device. The later CuRobo grasp phase also completed in the
  successful episode, so this warning did not block this task.

## Recorded raw episode

`scripts/collect_simple_episode.py` in this demo repo reproduces the run
without requiring the LeRobot recorder. The successful seed-0 episode is at
`robotics/SIMPLE/data/bend_pick_seed0_raw.npz` (ignored by SIMPLE Git; 25.8 MB).
It contains 214 aligned 50 Hz frames, one `head_stereo_left` RGB view,
`joint_pos_43`, measured floating-base pose, `action_target_43` constructed
like SIMPLE's MP recorder, per-frame reward, timestamps, joint names, task ID,
instruction, and success flag. The final reward is 1.0. The first and last
camera frames were visually checked: the white box moves into the G1's right
hand. The 213 timestamp gaps are all 0.020 s; root quaternion norms are 1.

```bash
cd ~/robotics/SIMPLE
MUJOCO_GL=egl .venv-lite/bin/python \
  ../G1-Demo-FastWAM/scripts/collect_simple_episode.py \
  --out data/bend_pick_seed0_raw.npz --seed 0
```

This is **source data**, not `prepare-stage2` input. It has 43 mixed MP target
channels, no explicit desired root orientation/yaw-rate trajectory, and 14
Dex3 hand joints instead of the current two scalar placeholders. Its head
image is 360×640, so height also needs a resize/crop to a multiple of 16.
Do not rename `action_target_43` to `action_ref` or train a SONIC latent policy
from it without the conversion decisions below.

The earlier temporary package folder `/tmp/simple_trial_deps` was only an
import probe. SIMPLE, SONIC, and `engineai` tracked source files were not edited.

## Source-verified episode mapping

The MP task declares 50 Hz rendering/control and a 640×360 head stereo camera.
Its environment returns `joint_qpos` plus rendered camera images. Its LeRobot
recorder stores `observation.joint_qpos`, `observation.rgb_<camera>`, `action`,
task instruction, AMO diagnostic fields, and environment configuration. It saves
successful episodes only. The `action` value is constructed **after** stepping:
actuator controls, with the first 15 replaced by AMO lower-body joint targets
and the last 14 by Dex3 hand targets. It is not the original CuRobo command and
is not already a SONIC motion reference.

| SIMPLE MP recording | `prepare-stage2` input | Required work |
|---|---|---|
| head camera frame | `rgb [T,H,W,3]` | Select one camera; resize/crop to dimensions divisible by 16; verify timing and color order. |
| `observation.joint_qpos [T,43]` | `joint_pos [T,29]` | Select the 29 body joints by **name** and reorder to SONIC MuJoCo order; verify units. |
| `action [T,43]` | `action_ref [T,34]` | Select/reorder 29 **desired** body joints; derive root roll, pitch, and yaw rate from a recorded desired root trajectory; map hands explicitly. Never use measured positions as desired actions without labeling that approximation. |
| 14 Dex3 hand targets | two scalar hand channels | Current contract cannot preserve dexterous motion. Expand hand contract/model or define and validate a grasp synergy with hardware-compatible units. |
| task text | `instruction` | Preserve the exact per-episode instruction. |
| 50 Hz frames | `timestamp_s` | Confirm actual frame intervals; do not synthesize timestamps for dropped frames. |
| simulator root state | `base_quat_wxyz` | Extract measured floating-base quaternion in `wxyz`; the standard LeRobot recorder does not expose it as a named feature, so capture it separately or recover and verify it from simulator state. |

The absent root orientation/reference and the 14-to-2 hand mismatch prevent a
faithful automatic converter today. A successful SIMPLE episode should first be
audited for field names, shapes, joint order, and timestamps. Then add a
conversion tool with an explicit hand policy and root-reference derivation,
followed by a SONIC encode/decode round trip and a replay comparison.

## Re-run this setup

This laptop now has the local toolchain and a working MuJoCo reset. The key
verification commands are:

```bash
cd ~/robotics/SIMPLE
.cuda-toolkit/bin/nvcc --version
PYTHONPATH=src MUJOCO_GL=egl .venv-lite/bin/python - <<'PY'
import torch, gymnasium as gym, simple.envs
env = gym.make('simple/G1WholebodyBendPickMP-v0',
               sim_mode='mujoco', headless=True, render_hz=50)
obs, info = env.reset(seed=0)
print(env.unwrapped.task.instruction, obs['joint_qpos'].shape)
env.close()
PY
```

For a LeRobot-format episode, install SIMPLE's pinned LeRobot fork and its
remaining recording dependencies, then use MuJoCo mode and one episode:

```bash
cd ~/robotics/SIMPLE
datagen simple/G1WholebodyBendPickMP-v0 \
  --sim-mode mujoco --headless --no-webrtc \
  --num-episodes 1 --render-hz 50 \
  --save-dir data/datagen
```

Check `datagen --help` for installed Typer option spelling before running.
This LeRobot command has **not** been run successfully here. The current
`.venv-lite` does not yet contain LeRobot, but the raw collector above ran a
complete successful episode.
