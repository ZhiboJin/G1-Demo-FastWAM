# Implemented FastWAM G1 model and SONIC boundary

The proposed path is valid: predict explicit joint references, encode them using
the pretrained G1 encoder/FSQ, and keep hand commands separate. The encoder needs
a motion window and current robot orientation, not just a single 29-value pose.
The 64 latent coordinates are learned motion features; they do not name joints.

```text
RGB + instruction/text embeddings + measured G1 joints
          |
FastWAM video expert + ActionDiT + MoT
          |
denoised [T,34] -> per-channel denormalization
          |
          +-- 29 named joints + 3 root channels
          |       -> joint permutation, derived velocities, relative rotations
          |       -> pretrained SONIC v1.1 G1 encoder INCLUDING FSQ
          |       -> 64-value token
          |
          +-- configurable left/right hands -> bypass encoder unchanged

Deployment boundary (implemented in Python, not yet executed as control):
64-value token + robot state history -> SONIC decoder -> [1,29] joint action
separate hand commands -> hardware-specific hand controller
```

The decoder half is now implemented. `scripts/sonic_decoder.py` assembles the
994-value observation and executes the pretrained graph; `scripts/verify_decoder.py`
runs encoder and decoder in sequence and passes 16/16 checks. What remains is
executing that action in a physics loop, which needs the Linux C++ deployment.

## Decoder observation (994 values)

The layout is derived from `observation_config.yaml`'s ordered list plus the
per-name dimensions in the deployment observation registry
(`g1_deploy_onnx_ref.cpp`, the table declaring `his_gravity_dir_10frame_step1 -> 30`).
The sum equals the ONNX-declared input width:

| Observation | Dim | Notes |
|---|---|---|
| `token_state` | 64 | the encoder's output; FSQ already applied |
| `his_base_angular_velocity_10frame_step1` | 30 | 10 x 3, oldest -> newest |
| `his_body_joint_positions_10frame_step1` | 290 | 10 x 29, IsaacLab order |
| `his_body_joint_velocities_10frame_step1` | 290 | 10 x 29 |
| `his_last_actions_10frame_step1` | 290 | 10 x 29 |
| `his_gravity_dir_10frame_step1` | 30 | 10 x 3, `R(q)^T @ (0,0,-1)` |

The frame axis runs oldest -> newest: the `GatherHis*` functions pass
`newest_first=false`, and `StateLogger::GetLatest` reverses its newest-first ring in
that case. Joint order is IsaacLab order, because the C++ side indexes the policy
output with `isaaclab_to_mujoco`.

The decoder output is the policy's raw action, **not** a motor command. C++ applies
`q_target = default_angles + action * scale`. That scaling is deliberately not applied
in Python because it is **not yet validated**: decoded magnitudes reach `|action| ~ 9 rad`
for out-of-distribution scripted references. Confirming the semantics requires running
the official Linux sim2sim and comparing our Python output against the C++ action buffer
for the same reference frame and identical state history.

## Action contract

`configs/action_space.json` is authoritative. With one value per hand:

| Slice | Meaning | Routing |
|---|---|---|
| 0:7 | Left arm joint positions | Encoder |
| 7:8 | Left hand command | Bypass |
| 8:15 | Right arm joint positions | Encoder |
| 15:16 | Right hand command | Bypass |
| 16:22 | Left leg joint positions | Encoder |
| 22:28 | Right leg joint positions | Encoder |
| 28:31 | Waist joint positions | Encoder |
| 31:34 | Absolute roll/pitch, yaw rate | Encoder orientation reference |

Joint positions are absolute radians; root units are rad, rad, rad/s. Hand units
remain placeholders until hardware is selected. `g1_model.action_names()` names
every scalar. Changing hand sizes updates the head and adapter slices, but requires
rebuilding weights and normalization statistics. The root block is retained from
the existing demo: joint angles alone do not specify world root orientation.
This contract has no explicit root translation/height target. Locomotion quality
is untested. SONIC's final joint commands need not equal its input references.

## Model

The model configuration now lives in `configs/fastwam_g1.yaml`. The Python factory
loads it directly and checks action dimensions against `action_space.json`.
It follows the upstream `fastwam.runtime.create_fastwam` model schema with no
dataset interpolations. Edit both `action_dim` values when hand sizes change.
`load_text_encoder: false` requires precomputed context/context_mask; set it true
to load the text encoder for prompt strings. Run `scripts/build_g1_architecture.py`
to refresh the derived JSON. This model YAML does not supply a training dataset,
camera processor or fitted normalization statistics.

`scripts/g1_model.py` provides `build_fastwam()`, using the actual sibling FastWAM
implementation. It retains the official Wan2.2 video expert, 30-layer MoT,
24 attention heads and 1024-wide ActionDiT. Action input/output projections use
D = 32 + left_hand_size + right_hand_size. No replacement transformer or 64-D
FastWAM head is introduced. Channels are modeled jointly per time step. During
denoising they represent flow; physical action semantics apply after integration
and denormalization, and must be learned from matching labels.

The upstream backbone loader excludes `action_encoder.*` and `head.*`, so G1
projections start untrained. The production factory uses official Wan weights and
the interpolated ActionDiT backbone. These large weights were not downloaded.
Production expert dimensions were allocated on PyTorch's meta device; a reduced
instance of the real ActionDiT was tested with actual forward/backward tensors.
The saved tiny checkpoint is random and explicitly labeled UNTRAINED.

Default proprioception is 29 measured joint positions in MuJoCo order. A richer
state requires an explicit `proprio_dim` and matching data semantics.
`ActionNormalizer` requires statistics with exactly matching named channels;
LIBERO statistics must not be reused. `G1Policy.predict()` calls upstream
`infer_action` then denormalizes. `encode_at()` produces a token plus separate
hand commands using the current robot quaternion.

## SONIC v1.1

`scripts/sonic_encoder.py` supports only the exact float32 v1.1 deployment layout.
It checks the ordered YAML configuration and actual ONNX signatures: input
`obs_dict [1,1751]`, output `encoded_tokens [1,64]`. FSQ is inside this pretrained
graph; do not quantize a second time. Default/low-latency variants are rejected.

G1 references use offsets `[0,5,...,45]` at 50 Hz: a 0.9-second window. Each encoded
tick requires at least 46 remaining samples. With a 50-frame prediction, five ticks
are available before more lookahead is needed. Short chunks are rejected without
padding. This imposes an inference latency requirement not yet benchmarked on the
target GPU. Longer horizons trade more lookahead for prediction difficulty.

Joint positions and velocities occupy separate time-major blocks in IsaacLab
order. Velocities use centered differences internally and one-sided differences
at endpoints. Rotation features are the first two matrix columns flattened
row-wise. The caller supplies reference world yaw at chunk frame zero; yaw rate
at t advances yaw from t to t+1. Replans must preserve world-yaw continuity.
Reference orientation is normalized by current robot heading. Unused teleop/SMPL
branches are zero-filled. Hands never enter the observation buffer.

This adapter uses actions directly. The older CSV bridge retains its independent
export convention and is not used to construct these encoder observations.
The NPZ output has named `token`, `left_hand`, `right_hand` arrays: currently 64+1+1.
NVIDIA's stock VLA 78-D protocol expects 7+7 hand joints; this packet cannot be sent
unchanged to that protocol. Hardware routing and closed-loop decoding remain future work.

## Run on this machine

From `fastwam_g1_demo`:

```powershell
.venv-model/Scripts/python.exe scripts/build_g1_architecture.py --save-tiny
.venv-model/Scripts/python.exe scripts/verify_model.py
.venv-model/Scripts/python.exe scripts/encode_sonic.py --actions artifacts/scripted_actions.npy --robot-quat 1 0 0 0 --reference-yaw 0 --out artifacts/sonic_packet.npz
```

The last command uses the actual pretrained encoder with scripted references,
not trained FastWAM output. The verification JSON records the encoder SHA-256.
Weights are in `checkpoints/sonic_v1_1`; `.venv-model` is the isolated CPU environment.
On another machine install `requirements-model.txt` and download the encoder and
matching observation YAML from NVIDIA below. Use upstream FastWAM's supported
CUDA environment for full-size inference/training. Dataset conversion, training,
normalization estimation and free-base closed-loop evaluation remain out of scope.

## Sources

- [Official FastWAM](https://github.com/yuantianyuan01/FastWAM)
- [Official SONIC v1.1 weights and configuration](https://huggingface.co/nvidia/GEAR-SONIC/tree/main/sonic_v1_1)
- [SONIC training / FSQ architecture](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html)
- [SONIC hand protocol](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html)

Local source verification used the observation registry and gatherers in
`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`, the v1.1
observation YAML, and `gear_sonic/envs/manager_env/mdp/observations.py`.
