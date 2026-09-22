# GEAR-SONIC (GR00T-WholeBodyControl) — Technical Interface Report

Prepared for an engineer writing an adapter that feeds motion commands into SONIC and reads back joint actions.

**Evidence quality labels used below**
- **[MEASURED]** — read directly out of the released artifacts (ONNX graphs parsed with `onnx`, HF file metadata) or the shipped C++/Python source.
- **[DOCS]** — stated by NVIDIA documentation / model card / project page.
- **[INFERRED]** — my reasoning from code or configs, not explicitly documented.
- **[NOT DETERMINED]** — could not find a documented answer.

Primary artifacts inspected: `model_encoder.onnx` / `model_decoder.onnx` (default and `low_latency/` variants) downloaded from Hugging Face and parsed; `observation_config.yaml` for all three variants; the full `gear_sonic/` and `docs/` trees of the GitHub repo; the C++ deployment headers/`g1_deploy_onnx_ref.cpp`.

---

## 0. TL;DR for the adapter author

There are three ways to attach an adapter. Pick per your deployment target:

1. **ZMQ motion streaming into the stock C++ controller** (recommended if the endpoint is the real robot or the MuJoCo sim). Publish one of the documented "pose" protocols (v1 joint-based, v3 SMPL+joint, v4 token-only) on port 5556, topic `pose`. Read back actions from the C++ ZMQ publisher on port 5557, topic `g1_debug`.
2. **Direct ONNX pair** (recommended if you only want the network, no robot stack): `model_encoder.onnx` maps a flat float32 vector `[1, 1762]` → a 64-D token `[1, 64]`; `model_decoder.onnx` maps `[1, 994]` (token + proprioception) → `[1, 29]` normalized joint-position offsets.
3. **Isaac Lab Python eval** (`gear_sonic/eval_agent_trl.py`) using `sonic_release/last.pt` — needs Isaac Lab/Isaac Sim.

The policy output is **not** torques and **not** raw joint angles. It is 29 normalized joint-position offsets; the mapping to hardware is `q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i]`, sent as a PD position target with per-joint `kp`/`kd` and `tau_ff = 0`. Hands are **outside** the policy.

---

## 1. ARCHITECTURE — encoders, token space, decoder

**Documented design [DOCS]** ([project page](https://nvlabs.github.io/GEAR-SONIC/), [training-code reference](https://nvlabs.github.io/GR00T-WholeBodyControl/references/training_code.html)):
> "SONIC employs a universal control policy that seamlessly handles robot motion, human motion, and hybrid motion through a shared latent representation. **Specialized encoders** process diverse motion commands into a **universal token space**."

The training-code reference gives the canonical diagram: N modality encoders → FSQ quantizer → two decoders (`g1_dyn` for actions, `g1_kin` for auxiliary reconstruction only). The module is `UniversalTokenModule`, file `gear_sonic/trl/modules/universal_token_modules.py`, described as "SONIC's action transform module (ATM)".

**Measured module list and checkpoint names [MEASURED]** — from the released ONNX graphs (weight tensor names are preserved from PyTorch):

| Module | Role | Input dim | Hidden | Output | Where |
|---|---|---|---|---|---|
| `module.encoders.g1` | G1 robot-joint reference encoder (encode mode 0) | 640 | 2048-1024-512-512, SiLU | 64 | `model_encoder.onnx` |
| `module.encoders.teleop` | VR 3-point teleoperation encoder (mode 1) | 267 | 2048-1024-512-512, SiLU | 64 | `model_encoder.onnx` |
| `module.encoders.smpl` | SMPL human-motion encoder (mode 2) | 840 (default/v1.1) / 336 (low-latency) | 2048-1024-512-512, SiLU | 64 | `model_encoder.onnx` |
| `quantizer` (×3) | FSQ, `vector_quantize_pytorch.FSQ` | 64 | — | 64 | `model_encoder.onnx` |
| `module.decoders.g1_dyn` | **the deployed decoder**; outputs joint actions | 994 | default: 2048-2048-1024-1024-512-512; low_latency & v1.1: 4096-4096-2048-2048-1024-1024-512-512; SiLU | 29 | `model_decoder.onnx` |
| `module.decoders.g1_kin` | kinematic decoder, reconstruction aux loss **only** | 32 | 2048-1024-512-512, 10 output frames | future motion | *not exported* to the deployment ONNX |

All three encoders are trained jointly with `encoder_sample_probs: {g1: 1.0, smpl: 1.0, teleop: 1.0}` and one is sampled per environment per episode. At deployment exactly one is selected by the `encoder_mode_4` field in the encoder input. **[DOCS/MEASURED]**

The default `g1_dyn` config in the training checkpoint `sonic_release/config.yaml` has `hidden_dims: [2048, 2048, 1024, 1024, 512, 512]`, which matches the exported ONNX exactly (10,184,221 parameters, ~40.9 MB fp32). The `low_latency` and `sonic_v1_1` decoders are the larger 4096-wide MLP (~37.4 M params, ~150 MB). **[MEASURED]**

Yes, there is a real shared-latent design: 3 encoders → one 64-D FSQ token → 1 deployed decoder. The docs warn: "Each ONNX encoder and decoder must be used with its matching observation configuration." **[DOCS]**

---

## 2. ENCODER INPUT

### 2.1 The flat tensor (deployment)

**[MEASURED]** Parsed straight from the released ONNX graphs:

| Variant | Encoder ONNX input | Encoder ONNX output |
|---|---|---|
| Default (top-level `model_encoder.onnx`) | `obs_dict`, float32, **`[1, 1762]`** | `encoded_tokens`, float32, **`[1, 64]`** |
| Low-latency (`low_latency/model_encoder.onnx`) | `obs_dict`, float32, **`[1, 1247]`** | `encoded_tokens`, float32, `[1, 64]` |
| SONIC v1.1 (`sonic_v1_1/model_encoder.onnx`) | `model_encoder.onnx`, documented **1751** [DOCS]; not parsed | `[1, 64]` |

Batch size is fixed at 1. The input is a **single flat concatenation**; there is no per-modality input tensor in the deployed encoder — the active encoder is selected by the mode id inside the flat vector.

### 2.2 Exact layout of the default encoder input (1762) [MEASURED]

Concatenation order == the order of `encoder_observations` in the variant's `observation_config.yaml`. For the **default** variant the released `observation_config.yaml` (downloaded from HF) is:

| # | Observation name | Dim | Per-frame shape | Content |
|---|---|---|---|---|
| 1 | `encoder_mode_4` | 4 | — | `[mode_id, 0, 0, 0]`; 0=G1 joint, 1=teleop, 2=SMPL |
| 2 | `motion_joint_positions_10frame_step5` | 290 | 10 × 29 | future reference joint positions (rad, IsaacLab order) |
| 3 | `motion_joint_velocities_10frame_step5` | 290 | 10 × 29 | future reference joint velocities (rad/s) |
| 4 | `motion_root_z_position_10frame_step5` | 10 | 10 × 1 | root height (m) |
| 5 | `motion_root_z_position` | 1 | 1 | current root height |
| 6 | `motion_anchor_orientation` | 6 | 1 × 6 | heading-corrected relative base rotation, 6-D (first two columns of the 3×3 matrix) |
| 7 | `motion_anchor_orientation_10frame_step5` | 60 | 10 × 6 | same, 10 frames |
| 8 | `motion_joint_positions_lowerbody_10frame_step5` | 120 | 10 × 12 | lower-body joints only, 12 DoF |
| 9 | `motion_joint_velocities_lowerbody_10frame_step5` | 120 | 10 × 12 | lower-body velocities |
| 10 | `vr_3point_local_target` | 9 | 3 × xyz | `[left_wrist, right_wrist, head]` positions in root frame |
| 11 | `vr_3point_local_orn_target` | 12 | 3 × quat | same three points, quaternions wxyz |
| 12 | `smpl_joints_10frame_step1` | 720 | 10 × 24 × 3 | SMPL joint positions, 24 joints |
| 13 | `smpl_anchor_orientation_10frame_step1` | 60 | 10 × 6 | SMPL root orientation 6-D |
| 14 | `motion_joint_positions_wrists_10frame_step1` | 60 | 10 × 6 | the 6 wrist joints only |
| | **Total** | **1762** | | |

Rows 4–5 (`motion_root_z_position*`) appear in `encoder_observations` but in **no** `encoder_modes.required_observations` list, so they are zero-filled padding in the mode-selective path. Removing them gives the v1.1 layout (**1751**) [MEASURED sum + DOCS statement agree].

The **low-latency** layout differs in two ways [MEASURED, from its `observation_config.yaml`]: G1/teleop references switch from `step5` to `step1` (same dims), the root-z entries are absent, `motion_anchor_orientation` moves next to the G1 block, and the SMPL block becomes `smpl_joints_4frame_step1` (288), `smpl_anchor_orientation_4frame_step1` (24), `motion_joint_positions_wrists_4frame_step1` (24) — total 1247.

**Adapter-critical:** because the encoder is a single flat tensor, the layout is positional. You must keep the exact order of the variant's `observation_config.yaml`. Reordering the YAML (which the generic docs say changes offsets) will break a released encoder.

### 2.3 What the joint data actually is

- **Absolute joint angles**, not offsets, and in **IsaacLab joint ordering** (29 joints). Confirmed on three independent paths: the reference-motion format docs (`joint_pos.csv`, "Joint positions in IsaacLab order (29 joints)"), the C++ ZMQ protocol spec ("Joint positions in IsaacLab order (all 29 joints)"), and the training code `TrackingCommand.joint_pos_multi_future` → `motion_lib.get_dof_pos(...)`, which returns the raw `dof` array with no default-pose subtraction. **[MEASURED]**
- **Root translation is NOT in the encoder input except for `root_z`.** There is no root x/y translation observation; heading is handled by the anchor-orientation 6-D term and a runtime `delta_heading` offset. **[MEASURED/DOCS]**
- **Body keypoints** enter via the SMPL encoder (24 SMPL joints × 3) and the teleop encoder (3 points: head + 2 wrists), not via the G1 encoder.

### 2.4 Sequence length and rate

**[DOCS]** Control loop and reference-motion timeline: **50 Hz (20 ms per tick)**. Naming convention `{name}_{N}frame_step{S}`: `N` = frames, `S` = stride in control ticks (so `step5` = 0.1 s).

| Path | Default / v1.1 | Low-latency |
|---|---|---|
| G1 joint encoder (`command_multi_future`) | 10 frames @ step5 → 100 ms spacing, **0.9 s span** | 10 frames @ step1 → 20 ms spacing, 0.18 s span |
| Teleop lower-body | 10 frames @ step5 | 10 frames @ step1 |
| SMPL encoder | 10 frames @ step1 → 20 ms spacing, **~200 ms lookahead** | 4 frames @ step1 → **~80 ms lookahead** |

Note the model card's headline "10 future frames at 20 ms spacing, approximately 200 ms of reference lookahead" refers **specifically to the SMPL encoder** (which uses `step1`). The G1 joint path uses `step5` (100 ms spacing). This resolves an apparent contradiction between the model card and the observation reference. **[MEASURED + INFERRED reconciliation]**

---

## 3. LATENT ("universal token space")

**[DOCS]** "All three models use the SONIC universal-token controller, produce **64-dimensional latent motion tokens**"; "The full action space per inference step is 78-dimensional: 64-dim motion token + 7-dim left hand joints + 7-dim right hand joints."

**[MEASURED]** The encoder ONNX emits exactly `encoded_tokens float32 [1, 64]`, FSQ-quantized.

**[MEASURED/INFERRED]** It is best understood as **2 tokens × 32 dims**, not one 64-D continuous vector:
- `gear_sonic/trl/modules/universal_token_modules.py`: `self.token_dim = self.num_fsq_levels`; `self.token_total_dim = self.token_dim * self.max_num_tokens`; logged as `embedding dim: 64 (num_tokens=2, token_dim=32)`.
- `sonic_v1_1/model_config.yaml` and `sonic_release/config.yaml` both set `num_fsq_levels: 32`, `max_num_tokens: 2`, `fsq_level_list: 32` (broadcast to 32 levels per dimension), quantizer `vector_quantize_pytorch.FSQ`.
- The ONNX export flattens `(B, 2, 32)` → `[1, 64]`. The encoder graph tail contains three FSQ quantizer subgraphs and a `ScatterND`; the graph's final operator reshapes the selected quantized latent to `[1, 64]`.

So: 64 dimensions, each independently quantized to one of 32 levels, arranged as 2 motion tokens of 32 dims. Downstream consumers (VLA, ZMQ protocol v4, `token_state`) treat it as a single flat 64-vector. `encoder.dimension: 64` in the YAML is the value the C++ validates against.

There is also a documented **latent residual** hook: `post_quantization` (default), `pre_quantization`, `pre_quantization_replace` — an external policy can inject corrections into the token space without retraining. **[DOCS]**

---

## 4. DECODER OUTPUT

### 4.1 Shapes

**[MEASURED]** `model_decoder.onnx`: input `obs_dict` float32 **`[1, 994]`**, output **`action` float32 `[1, 29]`**.

Decoder input layout for the default variant (from its `observation_config.yaml`, verified against the C++ `GetObservationRegistry()` dimensions):

| Order | Observation | Dim |
|---|---|---|
| 1 | `token_state` | 64 |
| 2 | `his_base_angular_velocity_10frame_step1` | 30 (3 × 10) |
| 3 | `his_body_joint_positions_10frame_step1` | 290 (29 × 10) |
| 4 | `his_body_joint_velocities_10frame_step1` | 290 |
| 5 | `his_last_actions_10frame_step1` | 290 |
| 6 | `his_gravity_dir_10frame_step1` | 30 |
| | **Total** | **994** |

⚠️ **Stale comment:** the shipped `observation_config.yaml` header says `# Total dimension: 436 (64+12+116+116+116+12)`. That is wrong. The registry dims are 30/290/290/290/30, the sum is **994**, and the released ONNX decoder input is **994**. Do not use the 436 comment. (The docs page's "Minimal (154D — default policy)" example is a *different*, non-token policy and also does not describe the released decoder.) **[MEASURED]**

### 4.2 Action semantics — PD targets, not torques

**[MEASURED]** `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`, `CreatePolicyCommand()`:

```cpp
for (int i = 0; i < G1_NUM_MOTOR; i++) {                      // i = hardware/MuJoCo index
  const double action_value = floatarr[isaaclab_to_mujoco[i]] * g1_action_scale[i];
  last_action[i] = floatarr[i];
  motor_command_tmp.q_target.at(i) = default_angles[i] + action_value;
  motor_command_tmp.tau_ff.at(i)   = 0.0;
  motor_command_tmp.kp.at(i)       = kps[i];
  motor_command_tmp.kd.at(i)       = kds[i];
  motor_command_tmp.dq_target.at(i)= 0.0;
}
```

- The policy emits **29 normalized joint-position offsets**, indexed in **IsaacLab order**.
- Hardware command = **position PD target** `q_target = default_angles + action_scale ⊙ action`, with `dq_target = 0`, `tau_ff = 0`, per-joint `kp`/`kd` from `policy_parameters.hpp`. Sent on Unitree DDS `rt/lowcmd` as `MotorCmd.q/dq/kp/kd/tau`.
- `default_angles` (hardware order) are the G1 standing pose, e.g. `left_hip_pitch=-0.312, left_knee=0.669, left_ankle_pitch=-0.363, left_shoulder_pitch=0.2, left_elbow=0.6, right_shoulder_roll=-0.2`. Full 29-entry array is in `policy_parameters.hpp`. **[MEASURED]**
- Gains and scales are derived from motor armature constants: `stiffness = armature·ω²`, `damping = 2ζ·armature·ω` with ω = 10 Hz·2π, ζ = 2.0, and `action_scale = 0.25·effort_limit/stiffness`. **[MEASURED — documentation in the header]** Numeric values:

| Motor group | Kp (Nm/rad) | Kd (Nm·s/rad) | action_scale |
|---|---|---|---|
| 7520_22 (hip pitch/roll, knee) | 99.098428 | 6.308802 | 0.350661 |
| 7520_14 (hip yaw, waist yaw) | 40.179238 | 2.557890 | 0.547546 |
| 5020 (shoulders, elbow, wrist roll) | 14.250623 | 0.907223 | 0.438577 |
| 4010 (wrist pitch/yaw) | 16.778327 | 1.068142 | 0.074501 |
| ankle pitch/roll, waist roll/pitch | 28.501246 (2×5020) | 1.814446 | 0.438577 |

  `--motor-kp-scale 4,10=1.5` / `--motor-kd-scale 4,10=1.5` (the SONIC v1.1 tuning) multiply hardware indices 4 and 10 = left/right ankle-pitch. **[DOCS]**

### 4.3 Frequency and dimensionality

- Action dim: **29 body DoF** (hand dims are not in the decoder). **[MEASURED]**
- Control loop: **50 Hz**; the C++ runs an input thread at 100 Hz, control at 50 Hz, planner at 10 Hz, command writer at 500 Hz. Encoder + decoder both run each control tick via TensorRT. **[MEASURED, from the class doc-comment]**
- `action_clip_value: 20.0` exists in the env config. **[MEASURED]**

### 4.4 Hands / end-effectors — separate channel

**[MEASURED/DOCS]** Hands are **not** part of the SONIC policy output.

- The C++ holds `left_hand_joint_buffer_` / `right_hand_joint_buffer_` (7 each) fed from the **input** stream (`left_hand_joints` / `right_hand_joints` ZMQ fields, or PICO trigger), and drives Unitree **Dex3** hands directly via DDS topics `rt/dex3/left/cmd` / `rt/dex3/right/cmd` with `MAX_DELTA_Q = 0.25 rad` per write and a runtime `max_close_ratio` (0.2–1.0). **[MEASURED]**
- `ZMQOutputHandler` publishes `last_left_hand_action double[7]` and `last_right_hand_action double[7]` for logging. **[MEASURED]**
- The VLA integration formalizes this: embodiment `UNITREE_G1_SONIC` / `unitree_g1_sonic` has a **78-dim action space = 64-dim motion token + 7 left hand + 7 right hand**, predicted at 2.5 Hz over a 40-step horizon and published at 50 Hz. **[DOCS]**
- Only for the Unitree G1 + Dex3 hand configuration; 43-DoF motion data with `hand_dof_pos` exists in the motion library, but the released `robot.type` is `g1_model_12_dex` with `actions_dim: 29`. **[MEASURED]**

---

## 5. ROBOT

- **Target robot: Unitree G1, 29 body DoF.** Confirmed by `const int G1_NUM_MOTOR = 29`, the 29-entry `default_angles`/`g1_action_scale`/`kps`/`kds`, `G1JointIndex` enum (0–28), and `robot: {type: g1_model_12_dex, actions_dim: 29}`. **[MEASURED]**
- The model card states: "SONIC provides three released **Unitree G1** checkpoints" and "These checkpoints target the **Unitree G1 embodiment**." **All three released checkpoints are G1-specific.** **[DOCS]**
- The **architecture and training pipeline are extensible**: the docs walk through adding a new robot (worked example: Unitree H2, 31 DoF) by adding URDF/MJCF, `robots/<robot>.py` (joint/body index maps, actuator gains, action scale), an order converter, and an experiment YAML. **No other pre-trained checkpoint is released.** **[DOCS]**
- Note the "12_dex" name: model/robot config name for the 29-DoF G1 with 12-DoF-ish Dex3 hand setup, but `actions_dim` remains 29. **[MEASURED, naming interpretation INFERRED]**

---

## 6. DEPLOYMENT

### 6.1 Training stack [DOCS]

- **Isaac Lab 2.3+** (README badge: IsaacLab **2.3.2**; docs badge **2.3.0**), i.e. Isaac Sim underneath.
- GPU: NVIDIA GPU with **CUDA 12.x** ("L40 recommended"); OS Ubuntu 22.04+; **Python 3.11 (required by Isaac Lab)**.
- Training config: `num_envs=4096`, 100,000 iterations, PPO with auxiliary reconstruction/latent losses; **64+ GPUs recommended** for real convergence (paper: 21k GPU hours, 1.2M→42M params, 700 h / 100M+ frames of mocap, Bones-SEED dataset).
- ONNX export via `python gear_sonic/eval_agent_trl.py +checkpoint=... +export_onnx_only=true`, producing `*_smpl.onnx`, `*_g1.onnx`, `*_teleop.onnx` (per-modality encoder+decoder) plus `*_encoder.onnx` (all encoders) and `*_decoder.onnx`.

### 6.2 Deployment stack [DOCS + MEASURED]

- **C++ (primary): `gear_sonic_deploy/`**, built with `just build`. Inference via **NVIDIA TensorRT** (`TRTInference/InferenceEngine.{h,cpp}`), with **CUDA graph capture**, pinned-memory input staging, and FP16 support (`--policy-precision 16`, `--planner-precision 16`, and `encoder.use_fp16`).
- **TensorRT versions are pinned and mandatory:** **10.13 for x86_64**, **10.7 for Jetson / G1 onboard Orin** (requires JetPack 6). Docs: "You **must** use the exact TensorRT versions listed above. Using a different version is known to produce incorrect inference results."
- An **ONNX Runtime path also exists** in the C++ (`include/ort_session.hpp`, `cmake/Findonnxruntime.cmake`) — I did not confirm which models/options route through it. **[INFERRED from source]**
- Requirements: Ubuntu 20.04/22.04/24.04 or Debian-based, CUDA Toolkit, Python 3.8+, Git LFS. Docker image provided for ROS2 Humble (x86_64 CUDA 12.4.1 / driver 550+; Jetson container on CUDA 12.6 host).
- **Inference does not require Isaac Sim.** The released ONNX models run in the C++/TensorRT stack on a plain Ubuntu box or on the robot's Jetson. The PyTorch `.pt` checkpoints are only needed for Isaac Lab eval/fine-tuning. **[DOCS]** Moreover the ONNX files are standard opset-13 graphs and ran under a stock Python `onnxruntime`/`onnx` parse in this investigation. **[MEASURED]**
- **MuJoCo path exists** — see §8.
- **GPU VRAM requirement: [NOT DETERMINED].** No VRAM figure is documented anywhere I found. Model sizes imply small requirements for *inference* (see §7); training VRAM is undocumented (4096 envs on an L40-class GPU is the only hint).

### 6.3 Runtime loop and interfaces [DOCS/MEASURED]

- Input interfaces: `keyboard | gamepad | gamepad_manager | zmq | zmq_manager | ros2 | manager`.
- Output: `--output-type <zmq|ros2|all>`; ZMQ output defaults to port **5557**, topic **`g1_debug`** (plus a `robot_config` topic re-published every ~2 s).
- Reference motion data: a directory of per-motion subfolders with `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, `metadata.txt` (+ optional `body_lin_vel.csv`, `body_ang_vel.csv`, `smpl_joint.csv`, `smpl_pose.csv`, `info.txt`) — all at **50 Hz**, quaternions **wxyz**, joints in **IsaacLab order**, root in column group 0.

---

## 7. AVAILABILITY

| Item | Finding |
|---|---|
| Code | **Public, ungated.** [github.com/NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl). Git LFS required for meshes/ONNX. |
| Checkpoints | **Public, ungated.** HF API reports `"private": false, "gated": false` for [`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC). Docs: "The repository is **public** — no token required for downloading." |
| NGC / EULA | **No NGC account or EULA acceptance needed for the HF repo.** The closest thing to a gate is the separate manual download of the pinned **TensorRT** tarball from NVIDIA Developer (account required), and accepting NVIDIA's license for Isaac Lab. **[DOCS/INFERRED]** |
| License | **Dual:** source code **Apache-2.0**; model weights **NVIDIA Open Model License** (commercial use with attribution, subject to NVIDIA Trustworthy AI terms). The HF repo carries `license: other, license_name: nvidia-open-model-license`. |

**File sizes (exact bytes from the HF API) [MEASURED]:**

| File | Bytes | ≈ |
|---|---|---|
| `model_encoder.onnx` (default) | 50,100,513 | 47.8 MiB |
| `model_decoder.onnx` (default) | 40,900,688 | 39.0 MiB |
| `planner_sonic.onnx` | 773,952,989 | 738.1 MiB |
| `observation_config.yaml` | 2,336 | — |
| **Default deployment set** | **864,956,526** | **≈ 825 MiB** |
| `sonic_release/last.pt` | 469,418,283 | 447.7 MiB |
| `sonic_release/config.yaml` | 28,331 | — |
| `low_latency/model_encoder.onnx` | 45,933,505 | 43.8 MiB |
| `low_latency/model_decoder.onnx` | 149,763,762 | 142.8 MiB |
| `low_latency/last.pt` | 1,125,409,931 | 1.05 GiB |
| `sonic_v1_1/model_encoder.onnx` | 50,117,796 | 47.8 MiB |
| `sonic_v1_1/model_decoder.onnx` | 149,778,638 | 142.8 MiB |
| `sonic_v1_1/last.pt` | 1,138,314,059 | 1.06 GiB |
| `bones_seed_smpl/*.tar.part_*` | 6 × 5,368,709,120 + 88,299,520 | **≈ 30 GiB** |
| Whole HF repo (`usedStorage`) | 37,327,333,666 | ≈ 34.8 GiB |

Download CLI given by the docs:

```bash
python download_from_hf.py                       # deployment: ONNX encoder+decoder+planner
python download_from_hf.py --low-latency
python download_from_hf.py --sonic-v1-1
python download_from_hf.py --training [--no-smpl] # PyTorch .pt (+ optional 30 GB SMPL data)
python download_from_hf.py --sample               # ~4 MB sample motions
```

**Third-party mirror:** [`BlackCatRoboticsAI/g1-sonic-base`](https://huggingface.co/BlackCatRoboticsAI/g1-sonic-base) is a **public, ungated, unofficial** re-upload (created 2026-09-03, 0 likes, 21 downloads) containing the default encoder/decoder/planner, `config.json`, one `bones_seed_smpl` part, and the `low_latency/` set — but **no** `sonic_release/`, no `sonic_v1_1/`, no `sample_data/`. Its `LICENSE` blob hash is identical to NVIDIA's, so it appears to be a straight mirror, not a fork. Prefer `nvidia/GEAR-SONIC`. **[MEASURED]**

---

## 8. SIMULATION

### 8.1 Isaac Lab / Isaac Sim [DOCS]

There is **no registered Gym environment id** for the SONIC tracking task in the documentation. It is a custom Isaac Lab `ManagerBasedRLEnv` composed by Hydra:
- env components: `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py` (robot added via `robot_mapping`, key `"g1_model_12_dex"`), `robots/g1.py`, MDP terms under `manager_env/mdp/`.
- Experiment preset: `+exp=manager/universal_token/all_modes/sonic_release` (also `…/sonic_v1_1`, `…/sonic_bones_seed`, `…/sonic_h2`).
- Eval/demo entry point (opens the Isaac Sim viewer):

```bash
python gear_sonic/eval_agent_trl.py \
    +checkpoint=sonic_release/last.pt +headless=False ++num_envs=1 \
    ++manager_env.observations.policy.enable_corruption=False \
    ++manager_env.observations.tokenizer.enable_corruption=False \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered"
```

Metrics mode (`++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=128`) and video render mode are documented; converged targets are `success_rate > 0.97`, `mpjpe_l < 30 mm`.

### 8.2 MuJoCo — **yes, a documented MuJoCo deployment example exists** [DOCS + MEASURED]

This is the "Sim2Sim" quick start, and it drives the **same C++ ONNX/TensorRT controller** as the real robot:

```bash
# one-time
bash install_scripts/install_mujoco_sim.sh     # creates .venv_sim (MuJoCo, Pinocchio, Unitree SDK2)
# Terminal 1 (host, repo root)
source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py
# Terminal 2 (gear_sonic_deploy/)
bash deploy.sh sim
```

- The MuJoCo sim and the C++ binary talk over **Unitree DDS on the loopback interface** (`lo`, `DOMAIN_ID: 0`); `--disable-crc-check` is required for sim.
- Documented MuJoCo scene: `gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml` (43-DoF = 29 body + hands) selected via `ROBOT_SCENE` in `gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml`; `SIMULATE_DT: 0.005`, `VIEWER_DT: 0.02`.
- `deploy.sh` modes: `sim` and `real`. Keyboard: `]` start, `9` drop robot (MuJoCo window), `T` play motion, `N`/`P` next/prev, `R` restart, `O` emergency stop.
- A standalone motion visualizer exists: `python visualize_motion.py --motion_dir …` and `--realtime_debug_url tcp://localhost:5557`.
- **Limitation:** the MuJoCo path is a *sim2sim bridge for the C++ controller*, not a Python MuJoCo policy runner — you still need the compiled C++ deploy binary (TensorRT) to run it end-to-end. **[DOCS/INFERRED]**

---

## 9. Concrete adapter interfaces (the two that matter)

### 9.1 Publishing motion commands into the C++ controller — ZMQ "pose" protocol [DOCS]

Default publisher endpoint: `--input-type zmq --zmq-host <ip> --zmq-port 5556 --zmq-topic pose`. Enable streaming with `ENTER` in the deployment console. Wire format: `[topic_prefix][1280-byte JSON header][concatenated binary fields]`. Header example:

```json
{ "v": 1, "endian": "le", "count": 100,
  "fields": [ {"name":"joint_pos","dtype":"f32","shape":[100,29]},
              {"name":"body_quat","dtype":"f64","shape":[100,1,4]} ] }
```

Header is **exactly 1280 bytes, null-padded** (this changed in the 2026-03-24 release). `frame_index` must be monotonically increasing; `frame_step` is auto-detected from the first two indices; a gap > 200 frames triggers a catch-up reset; changing protocol version mid-session disables ZMQ mode for safety.

| Protocol | Encode mode | Required fields | Optional |
|---|---|---|---|
| **v1** joint-based | 0 | `joint_pos [N,29]` f32/f64, `joint_vel [N,29]`, `body_quat`, `frame_index` | `smpl_joints`, `smpl_pose`, hands, vr_* |
| **v2** SMPL | 2 | `smpl_joints [N,24,3]`, `smpl_pose [N,21,3]`, `body_quat`, `frame_index` | joint_pos/vel |
| **v3** joint+SMPL | 2 | `joint_pos`, `joint_vel`, `smpl_joints`, `smpl_pose`, `body_quat`, `frame_index` | — |
| **v4** token-only | bypasses encoder | `token_state [D]` (D must equal `encoder.dimension`, i.e. 64) | `frame_index`, `left_hand_joints [7]`, `right_hand_joints [7]`, `body_quat_w [4]` |

Optional across versions: `left_hand_joints [7]`, `right_hand_joints [7]`, `vr_position [9]`, `vr_orientation [12]`, `catch_up` (bool, default true), `heading_increment` (scalar). `vr_compliance` is accepted but **ignored** (compliance is keyboard-controlled).

- **v1 is the SONIC native mode for robot-joint references**; in v3, only the 6 wrist joints in `joint_pos` need real values (indices `[23,24,25,26,27,28]` in IsaacLab order) — SMPL carries the rest.
- `joint_pos` values are **absolute joint angles in IsaacLab order** (§2.3).

### 9.2 Reading actions back — ZMQ `g1_debug` on port 5557 [MEASURED]

msgpack map, published every control tick, `[topic_prefix][msgpack]`, topic prefix `g1_debug` by default. Keys (from the `ZMQOutputHandler` header doc, cross-checked against the packing code):

| Key | Type | Notes |
|---|---|---|
| `index`, `ros_timestamp`, `control_loop_type` | metadata | `control_loop_type` is always `"cpp"` |
| `base_quat [4]`, `base_ang_vel [3]`, `body_torso_quat [4]`, `body_torso_ang_vel [3]` | IMU | wxyz |
| `body_q [29]`, `body_dq [29]` | measured joints | **MuJoCo order**, `body_q` has default-angle offsets added |
| `left_hand_q/dq [7]`, `right_hand_q/dq [7]` | hands | |
| **`last_action [29]`** | commanded body action | **MuJoCo order**, **already scaled and default-offset applied** — i.e. equal to the `q_target` sent to the motors |
| `last_left_hand_action [7]`, `last_right_hand_action [7]` | hands | |
| **`token_state [64]`** | encoder latent | empty array if no encoder |
| `base_trans_target [3]`, `base_quat_target [4]`, `body_q_target [29]` | current reference motion | |
| `vr_3point_position [9]`, `vr_3point_orientation [12]`, `vr_3point_compliance [3]` | VR | |
| `init_base_quat [4]`, `delta_heading` | heading | conditional |

Two order/semantics details worth pinning down because they are easy to get wrong:
- Internally `last_action[i] = policy_action[i]` (policy/IsaacLab index order, raw normalized value) and that is what the `his_last_actions_10frame_step1` observation feeds back. **[MEASURED — `CreatePolicyCommand()` + `GatherHisLastActions()`]**
- The **ZMQ-published** `last_action` is remapped to MuJoCo order and rescaled: `last_action_mujoco[i] = state.last_action[isaaclab_to_mujoco[i]] * g1_action_scale[i] + default_angles[i]`. So it equals the actual PD position target per motor. **[MEASURED — `zmq_output_handler.hpp`]**

### 9.3 Pure-Python ONNX path [MEASURED]

```
encoder: obs_dict float32[1,1762]  -> encoded_tokens float32[1,64]   (opset 13, batch fixed 1)
decoder: obs_dict float32[1,994]   -> action float32[1,29]
```
The decoder needs the 64-D token **plus** the 930-D proprioception history (12/116/116/116/12 blocks as listed in §4.1). If you only stream tokens (protocol v4), you still must supply the robot-state history yourself. Note the raw ONNX `action` is the *normalized* policy output — apply §4.2 to get motor targets.

---

## 10. Explicitly NOT determined / caveats

1. **GPU VRAM requirements** — not documented anywhere I found, for either training or deployment.
2. **Isaac Sim version pin** — docs only say "Isaac Lab 2.3+"; the README badge says 2.3.2 and the docs badge 2.3.0. No explicit Isaac Sim version is stated.
3. **Exact `sonic_v1_1` encoder ONNX shape** — documented as 1751 in its `observation_config.yaml`; I did not download that file to parse it (the default and low-latency encoders were parsed directly). The 1751 figure is consistent with the default 1762 minus the two unused root-z terms.
4. **`g1_kin` decoder weights** are not in the released ONNX files — reconstruction-only, training-time. `g1_kin` output dims were read from config, not from an artifact.
5. **Which code paths use ONNX Runtime vs TensorRT** in the C++ (`ort_session.hpp` exists alongside `TRTInference`) — I did not trace this. `freq_test` is documented as an ONNX load sanity check.
6. **`sonic_release/config.yaml`'s `model_config` mismatch for v1.1:** `sonic_v1_1/model_config.yaml` lists `g1_dyn` hidden dims `[4096,4096,2048,2048,1024,1024,512,512]`, which **matches** the v1.1 ONNX; but the same file's config also shows `num_steps_per_env: 24`, `num_envs: 1`, etc. from a debug/export run. Treat `config.yaml` as the resolved training config and the ONNX as ground truth for architecture.
7. **The "436" total-dimension comment** in the default `observation_config.yaml` is stale and wrong (real value 994). Similarly, the docs page's "154D default policy" example does not describe the released token-based decoder.
8. **The low-latency `observation_config.yaml` comment** says the 4-frame SMPL observations "require a rebuild to load" — the registry entries `smpl_joints_4frame_step1`, `smpl_anchor_orientation_4frame_step1`, `motion_joint_positions_wrists_4frame_step1` **are present in current `main`**, so a normal `just build` suffices. The comment is stale.
9. **Hand/gripper control is not part of the learned policy** — it is a pass-through input channel to the Dex3 driver. If you need the policy to control the hands, that requires a separate policy or VLA (the 78-D action space).
10. **Nothing about SONIC is gated on NGC.** Only the separate download of the exact TensorRT tarball (NVIDIA Developer account) and Isaac Lab's own license acceptance sit outside the public HF/GitHub flow.

---

## Sources

- Project page: https://nvlabs.github.io/GEAR-SONIC/
- Docs home: https://nvlabs.github.io/GR00T-WholeBodyControl/
- Model card: https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html
- Download models: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html
- Installation (deployment): https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html
- Installation (training): https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html
- Quick start: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html
- Training guide: https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html
- Training on new embodiments: https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/new_embodiments.html
- Training code structure (universal token module): https://nvlabs.github.io/GR00T-WholeBodyControl/references/training_code.html
- Observation configuration: https://nvlabs.github.io/GR00T-WholeBodyControl/references/observation_config.html
- C++ deployment program flow: https://nvlabs.github.io/GR00T-WholeBodyControl/references/deployment_code.html
- Motion reference data: https://nvlabs.github.io/GR00T-WholeBodyControl/references/motion_reference.html
- Kinematic planner ONNX reference: https://nvlabs.github.io/GR00T-WholeBodyControl/references/planner_onnx.html
- Coordinate/rotation conventions: https://nvlabs.github.io/GR00T-WholeBodyControl/references/conventions.html
- ZMQ streaming tutorial (protocols v1–v4): https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html
- VLA workflow (latent actions, 78-D space): https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html
- VLA inference (action space): https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html
- Data collection (LeRobot channels): https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html
- License: https://nvlabs.github.io/GR00T-WholeBodyControl/resources/license.html
- Paper: https://arxiv.org/abs/2511.07820 · https://www.science.org/doi/10.1126/scirobotics.aed4592
- Code: https://github.com/NVlabs/GR00T-WholeBodyControl · raw README: https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/main/README.md
- HF model: https://huggingface.co/nvidia/GEAR-SONIC · manifest https://huggingface.co/nvidia/GEAR-SONIC/blob/main/config.json
- HF mirror: https://huggingface.co/BlackCatRoboticsAI/g1-sonic-base
- Raw configs parsed: https://huggingface.co/nvidia/GEAR-SONIC/raw/main/observation_config.yaml · `low_latency/observation_config.yaml` · `sonic_v1_1/observation_config.yaml` · `sonic_v1_1/model_config.yaml` · `sonic_release/config.yaml`
- Raw source parsed: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/{src/g1_deploy_onnx_ref.cpp, include/policy_parameters.hpp, include/robot_parameters.hpp, include/dex3_hands.hpp, include/output_interface/zmq_output_handler.hpp, include/input_interface/zmq_endpoint_interface.hpp, include/input_interface/zmq_packed_message_subscriber.hpp, include/input_interface/streamed_motion_merger.hpp}`; `gear_sonic/{trl/modules/universal_token_modules.py, envs/manager_env/robots/g1.py, envs/manager_env/mdp/commands.py, utils/motion_lib/motion_lib_base.py, utils/inference_helpers.py}`
