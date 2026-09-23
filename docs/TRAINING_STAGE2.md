# Stage-2 training foundation

The goal is a language-conditioned G1 demo in an environment: an egocentric RGB
frame and measured robot state go into a policy; the policy emits a whole-body
SONIC command and separate end-effector commands. The existing 46-channel joint
reference demo verifies one execution path. `g1demo/stage2_data.py` now prepares
an alternative latent target for learning.

## What MotionWAM actually trains

The [MotionWAM paper](https://arxiv.org/html/2606.09215) specifies:

1. **Stage 1:** train its Cosmos-Predict2.5 Video DiT on egocentric human and
   humanoid video, with no action loss. This is the expensive stage to skip.
2. **Stage 2:** attach a Motion DiT and jointly update Video DiT and Motion DiT
   on action-labeled humanoid data. Preserve the video objective. The output is
   a SONIC discrete motion code plus continuous end-effector values; the
   published G1 action width is 66 (64 SONIC FSQ values and two grippers).
3. **Stage 3:** fine-tune the full model on task-specific G1 whole-body
   teleoperation episodes. The paper reports 200 episodes per task at 50 Hz.

Its project page does not currently publish a downloadable Stage-1 checkpoint or
training code. The publicly described base is Cosmos-Predict2.5-2B; that base
checkpoint is **not** the Stage-1 egocentric adaptation. FastWAM's Video DiT is
Wan2.2-based and cannot directly load a Cosmos/MotionWAM Video DiT checkpoint.
Reusing FastWAM is feasible as an **engineering variant**, not a reproduction of
MotionWAM Stage 2. Do not label a generic Wan2.2 or released LIBERO FastWAM
checkpoint as a completed MotionWAM Stage-1 checkpoint.

## Selected FastWAM output and data boundary

FastWAM is configured to predict `[T,46]` physical references: 29 body angles,
three root channels, and 7+7 Dex3 hand angles. The SONIC encoder runs **after**
FastWAM prediction; it is not part of FastWAM's output head. The resulting
`[64-token, 7 left hand, 7 right hand]` packet has width 78 for SIMPLE's
external controller. The current repository has no trained G1 policy or
FastWAM-to-SIMPLE transport bridge.

`python -m g1demo.cli prepare-stage2 --episode RAW.npz --out PREPARED.npz`
validates a synchronized 50 Hz episode and runs each 46-frame desired reference
window through the released SONIC v1.1 encoder. It keeps hands separate and
also writes `[sonic_token(64), left_hand(7), right_hand(7)]` as `latent_action`
for downstream inspection and an optional latent-action baseline. With the
current Dex3 contract, that is 78 values. These are FSQ **vectors**, not
MotionWAM's learned discrete index representation. Do not train the selected
46-output FastWAM head on `latent_action`; train it on `action_ref` with matching
normalization statistics. The prepared file now retains the aligned
`action_ref` rows, but a G1 training loader is still needed.
SIMPLE's current SONIC evaluator also expects 78 values: the same 64-token
body channel plus 7 joint commands for each Dex3 hand. The paper's 66-value
two-gripper contract cannot be sent to that evaluator unchanged.

Required raw episode `.npz` keys (all arrays aligned by frame index):

| Key | Shape / meaning |
|---|---|
| `rgb` | `uint8 [T,H,W,3]`, head RGB, H and W divisible by 16 |
| `joint_pos` | `[T,29]` measured body angles, MuJoCo/SONIC deployment order |
| `base_quat_wxyz` | `[T,4]` measured root orientation |
| `action_ref` | `[T,46]` **desired** body/root/Dex3 reference in `action_space.json` order |
| `timestamp_s` | `[T]`, monotonic with 20 ms intervals |
| `instruction`, `task_id` | non-empty scalar strings |

Store, alongside the episode, camera calibration/extrinsics, hand model and
units, body/hand command limits, measured hand state, success flag, simulator
seed, scene/object IDs, original controller, and source licenses. The current
minimal `.npz` cannot substitute for that complete collection record. Split
train and validation by scene and object instance, not adjacent frames of the
same rollout; fit action/proprioception statistics on the training split only.

## Reusing FastWAM

Keep upstream `robotics/FastWAM/` intact. Its Hydra training entry point,
flow-matching action expert, video loader, processors, and text cache are useful.
The local G1 work still needs a dataset adapter from converted episodes to its
LeRobot `RobotVideoDataset` schema, supervised `[T,46]` reference targets,
action/proprioception statistics, and a post-model encoder-to-SIMPLE bridge.
Train a small overfit batch before a full run. The current laptop has an 8 GB GPU and no
Stage-1 or G1 checkpoint installed; full FastWAM training needs larger compute.

## SIMPLE task decision

SIMPLE is a good **simulation and data-source candidate** because it includes
G1 whole-body tasks, MuJoCo physics, Isaac Sim rendering, and SONIC integration.
The clean upstream checkout is `robotics/SIMPLE/` at commit
`6d10628794d9c7de4596b4f2afb2c054a637c2bc`. A successful 214-frame
MuJoCo bend-and-pick episode is documented in `docs/SIMPLE_TRIAL.md`; its raw
NPZ is under `robotics/SIMPLE/data/`. It is not yet SONIC Stage-2 training data.
See `configs/demo_tasks.yaml` for three proposed tasks and language prompts.

There is a crucial controller difference: SIMPLE's five motion-planning G1 tasks
use AMO for the lower body and CuRobo for the arms. Its one documented SONIC task
is `G1WholebodyXMoveBendCarryBoxSonic-v0` and uses teleoperation. Do not combine
their raw actions as if they were already the same SONIC reference format. For
Stage 2, use MP rollouts only after conversion to 29 joint references, hand
commands, 50 Hz camera/state alignment, and round-trip SONIC tests. The MP
recorder writes 43 targets, including 14 Dex3 finger joints, whereas this
repo's present action contract has 14 named Dex3 hand channels. Convert and
reorder them explicitly by joint name, and verify limits before training.
Keep the
SONIC carry-box task as the closest eventual demo, even if its teleop data comes
later. The full SIMPLE installation includes Isaac Sim 4.5, but the local
MuJoCo-only G1 task now runs without Isaac Sim after installing CuRobo, AMO
weights, and task assets.

Sources: [MotionWAM paper](https://arxiv.org/html/2606.09215),
[SIMPLE tasks](https://psi-lab.ai/SIMPLE/docs/tasks/index.html),
[SIMPLE source](https://github.com/physical-superintelligence-lab/SIMPLE).
