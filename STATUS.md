# Continuation record — 2026-09-22

## Decoder milestone: full encoder -> token -> decoder chain now runs

The SONIC v1.1 **decoder** was the missing half of the integration. It has been
downloaded, its interface derived from official source, implemented and verified.

- `checkpoints/sonic_v1_1/model_decoder.onnx` downloaded (149,778,638 bytes,
  SHA-256 `34BAE8570D4A4421A5391A5C2BEFD745D4A02D182EC539E5F9DA44C091C67509`),
  alongside a refreshed `model_encoder.onnx` and `observation_config.yaml`.
  `planner_sonic.onnx` (773,952,989 bytes) is also available in the SONIC checkout.
- Added `scripts/sonic_decoder.py` and the symmetric CLI `scripts/decode_sonic.py`.
- Added `scripts/verify_decoder.py`: **16/16 checks pass**, running the real
  encoder and the real decoder in sequence over 31 control ticks. `verify_model.py`
  still passes 8/8 afterward.
- `G1Policy` gained `decode_at()` and a composite `step()`, so the full
  `predict -> encode_at -> decode_at` interface is now reachable from one object.

### The 994-D decoder observation layout is derived, not guessed

From `observation_config.yaml`'s ordered list plus the per-name dimensions in the
deployment observation registry
(`gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`, the table
declaring `his_gravity_dir_10frame_step1 -> 30`). The sum equals the ONNX-declared 994:

| Observation | Dim |
|---|---|
| `token_state` | 64 |
| `his_base_angular_velocity_10frame_step1` | 30 |
| `his_body_joint_positions_10frame_step1` | 290 |
| `his_body_joint_velocities_10frame_step1` | 290 |
| `his_last_actions_10frame_step1` | 290 |
| `his_gravity_dir_10frame_step1` | 30 |

Two conventions were confirmed from source rather than assumed: the frame axis runs
**oldest -> newest** (the gatherers pass `newest_first=false`, and
`StateLogger::GetLatest` reverses its newest-first ring in that case), and the 29
joints are in **IsaacLab order** (C++ indexes the output with `isaaclab_to_mujoco`).

### Open question that must not be quietly closed

**Decoder action magnitudes are not validated.** Decoded output reaches
`|action| ~ 9 rad` for both the scripted references and a synthetic zero state. That
is expected for out-of-distribution scripted references feeding a human-motion-trained
encoder, but it does **not** confirm the C++ convention
`q_target = default_angles + action * scale`. This is the same "compare our adapter's
tokens against official packing for the same trajectory/state" step described below,
now extendable to the decoder side.

Resolution procedure: run the official Linux C++ `deploy.sh sim`, then compare our
Python decoder output against the C++ action buffer for the same reference frame with
identical state history. Until that comparison exists, do not claim the action
semantics are correct.

Also still synthetic here: the decoder's state history. Real deployment fills it from
`StateLogger` at 50 Hz.

### Reproduce the decoder milestone

```powershell
.venv-model/Scripts/python.exe scripts/verify_decoder.py
.venv-model/Scripts/python.exe scripts/decode_sonic.py --packet artifacts/sonic_packet.npz --joint-pos 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 --robot-quat 1 0 0 0 --out artifacts/decoded_action.npy
```

## Persistent requirement: Linux training and simulation

The user requires all project code to remain runnable on their Linux machine for
later training. Use portable Python/pathlib and configurable repository paths;
avoid Windows-only runtime dependencies. Recreate virtual environments on Linux
(do not copy `.venv-model`); use upstream-supported CUDA dependencies for full
FastWAM training. The current CPU verification was run on Windows, so Linux/GPU
compatibility is a requirement, not yet a tested claim. Generated JSON currently
contains machine-resolved checkpoint paths: regenerate it on Linux using
`scripts/build_g1_architecture.py`. Preserve sibling repository layout or make
paths configurable before moving directories.

Next simulation verification: run NVIDIA's official free-base G1 MuJoCo sim2sim
with matched SONIC v1.1 encoder, decoder and observation YAML, first using official
reference motions. Verify the robot is released from support and controlled by
physics and policy outputs, rather than assigning reference qpos. Record falls,
joint tracking error, root attitude/height, action saturation and control timing.
Only after this baseline passes, compare our adapter's tokens against official
packing for the same trajectory/state and test those tokens in the control loop.
Do not use anchored-arm scripted references as assumed feasible whole-body motions.
Task objects, cameras, hands, reset logic and task-success checks come after the
controller baseline. No full closed-loop simulation was executed in this session.

Official workflow: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html
Matched checkpoints: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html

## Latest update: dedicated G1 YAML

Added `configs/fastwam_g1.yaml`, a model-only configuration using the official
`fastwam.runtime.create_fastwam` schema, without LIBERO/RoboTwin interpolations.
`scripts/g1_model.py` now loads this YAML for model construction, including
schedulers, loss settings and text-encoder selection. Both action dimensions are
34; proprioception is 29. A mismatch with `action_space.json` is rejected.
When changing hand sizes, update both action dimensions in the YAML as well.
This is not a complete training task/data configuration; those remain deferred.
Rebuild the derived JSON with `scripts/build_g1_architecture.py` after YAML edits.

## Resume here — latest user decisions and verified interface

User goal: build FastWAM for explicit G1 joint-space predictions, then feed the
trajectory through **the pretrained G1 encoder inside GEAR SONIC**, including its
FSQ, to produce a 64-D whole-body motion token. Additional left/right hand outputs
must bypass that encoder. The user confirmed configurable hand sizes, currently
one scalar per hand. Do not replace this design with direct FastWAM latent prediction.
Architecture and encoder integration were requested first; dataset preparation
and training code are deferred until the user asks.

Official documentation was checked after implementation. For **SONIC v1.1,
G1 mode 0**, the required motion inputs are:

| Input | Shape | Convention |
|---|---|---|
| Reference joint positions | 10 x 29 | Radians, IsaacLab joint order |
| Reference joint velocities | 10 x 29 | Radians/second, IsaacLab joint order |
| Reference root orientation | 10 x 6 | Relative to current robot heading |

These are 640 motion values. Samples occur at control-frame offsets
`[0,5,10,15,20,25,30,35,40,45]` at 50 Hz: 100 ms spacing, 0.9-second span.
The full multi-modality ONNX input is `[1,1751]`, including the mode selector and
unused modality slots; its output is `[1,64]`. The adapter zero-fills unused
branches. Hands are not included. Root roll/pitch/yaw-rate are **our adapter's
output representation**, not three raw inputs accepted by the official encoder;
the adapter converts them to orientation features. Joint velocities are derived
from the predicted position trajectory, not separate FastWAM output channels.

Sources checked:

- [Official SONIC v1.1 observation configuration](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/policy/sonic_v1_1/observation_config.yaml)
- [Official observation reference](https://nvlabs.github.io/GR00T-WholeBodyControl/references/observation_config.html)
- [Checkpoint variants and reference timing](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/download_models.html)

### Current evidence and limits

- Eight checks passed in `scripts/verify_model.py`, including actual pretrained
  encoder execution and unchanged tokens when only hand outputs change.
- The existing 200-frame CSV bridge verification also passed.
- `artifacts/sonic_packet.npz` contains a real encoder token and separate hand
  arrays from **scripted IK references**; it is not learned FastWAM task output.
- Full production expert dimensions were instantiated on the meta device;
  a reduced actual ActionDiT passed forward/backward tests. The saved tiny
  checkpoint is random/untrained. Full-size FastWAM inference was not run.
- No G1-trained FastWAM checkpoint, learned task success, SONIC decoder loop,
  physical hand controller, or hardware deployment has been demonstrated.
- Our adapter rejects insufficient lookahead (46 samples required at each tick).
  NVIDIA's general deployment reference describes repeating the last frame;
  rejection is our explicit adapter choice, not an official SONIC requirement.
- A 50-frame chunk supports five successive encoded ticks without replenishment;
  inference latency/replanning cadence remains to be measured.
- The current 64+1+1 packet is not NVIDIA's stock 78-D VLA packet, which assumes
  seven joint commands per hand. Hardware hand mapping remains unresolved.

### Continue from these files

- `docs/model_architecture.md`: detailed implemented design and boundaries.
- `scripts/g1_model.py`: upstream FastWAM factory, named outputs, normalization,
  `G1Policy.predict()` and `encode_at()`.
- `scripts/sonic_encoder.py`: exact v1.1 observation packing and ONNX encoder.
- `configs/action_space.json`: authoritative editable physical output contract.
- `configs/fastwam_g1_resolved.json`: generated production architecture settings.
- `artifacts/model_verification.json`: verification result and encoder hash.

Reproduce from `C:\Users\Justin\Desktop\pjt\fastwam_g1_demo`:

```powershell
.venv-model/Scripts/python.exe scripts/verify_model.py
.venv-model/Scripts/python.exe scripts/encode_sonic.py --actions artifacts/scripted_actions.npy --robot-quat 1 0 0 0 --reference-yaw 0 --out artifacts/sonic_packet.npz
```

If continuing architecture work, first review the implementation and the above
boundaries. If the user next requests training/datasets, inspect actual episode
schemas, units, joint ordering, camera timing and root/hand availability before
choosing normalization or writing converters. Do not assume a public G1 dataset
matches this contract just because it contains 29 joints. Closed-loop SONIC
decoder validation is a separate remaining integration milestone.

## Architecture milestone (latest)

- Added `scripts/g1_model.py`: real upstream FastWAM factory, explicit G1 action
  names, per-channel normalization and policy-to-encoder interface.
- Preserved 34 outputs: 29 joint positions, two configurable hand values and
  three root channels. Hands bypass the encoder, as requested.
- Added exact SONIC v1.1 observation packing and ONNX execution; downloaded the
  official pretrained encoder and matching YAML into `checkpoints/sonic_v1_1/`.
- Created an isolated CPU `.venv-model`. Production architecture allocated on
  meta; real reduced ActionDiT forward/backward tested. No pretrained full-size
  FastWAM weights or trained G1 model were created.
- Real SONIC encoder produced a 64-value token from scripted references. Hand
  bypass, joint permutation, velocity, heading, horizon and limit checks pass.
- See `artifacts/model_verification.json`, `configs/fastwam_g1_resolved.json`,
  and [model_architecture.md](docs/model_architecture.md).
- Dataset/training work is deferred by request. Decoder execution, hardware hand
  routing and closed-loop free-base validation remain unimplemented.

The sections below record the earlier IK/CSV milestone. Their missing-encoder
statements refer to that earlier stage; the new encoder exists in this demo,
not in the sibling SONIC checkout. The base Python environment remains unchanged.

## Completed

- Fresh folder created without changing sibling projects.
- Research decisions and upstream links recorded in README and `docs/`.
- Editable OpenHLM-style G1 action contract: 29 joints + 1 left hand placeholder + 1 right hand placeholder + 3 root = 34 values.
- Generated mesh-free MuJoCo G1 scene from local official-derived G1 XML because checked-out STL files are Git LFS pointers.
- Two task demo (`touch_left_target`, `touch_right_target`) using wrist Jacobian IK and joint PD. Last run stored in `artifacts/run.json`; both met a 5 cm endpoint criterion.
- Rendered task snapshots stored in `artifacts/demo.png`.
- Contract pack/unpack and joint mapping implemented.
- FastWAM-style action array → SONIC reference bridge implemented in `scripts/sonic_bridge.py`; validates the G1 URDF and uses SONIC's own joint permutation. Generated 200-frame clip in `artifacts/sonic_reference/scripted_reaches/`.
- Round-trip, quaternion, and URDF-limit checks pass via `scripts/verify_bridge.py`. Reference CSVs were played kinematically in MuJoCo; montage in `artifacts/sonic_reference_playback.png`.

## Earlier IK milestone limits (historical)

- This is an anchored-torso arm-reaching demo; it does not walk, grasp, or use SONIC/FastWAM inference. The target markers have no contact mechanics; task success measures wrist-to-marker distance.
- Generated geometry is only a visualization approximation. No camera dataset, downloaded model weights, or trained G1 checkpoint exists in this folder.
- No GPU PyTorch environment is installed in the active Python 3.13 environment. Upstream FastWAM specifies Python 3.10 with CUDA PyTorch and is a separate setup.
- The action source in this proof is scripted IK, not FastWAM inference. The SONIC ONNX controller did not run: this checkout has no ONNX weights, its meshes are LFS pointers, and the deployment stack is Linux/DDS oriented.

## Earlier proposed next steps (superseded by the resume notes above)

1. Choose physical hand/end-effector hardware and edit `configs/action_space.json` `end_effectors.*.size`. Re-run `python scripts/validate_contract.py` and update dataset conversion accordingly.
2. On a Linux machine, obtain real G1 meshes and SONIC ONNX weights using NVIDIA's documented download/setup steps, run the official SONIC G1 MuJoCo smoke test, then point `deploy.sh --motion-data` at `artifacts/sonic_reference/` (copy it into the SONIC repo if needed). Verify actual controller tracking and record the checkpoint variant.
3. Inspect one OpenHLM `g1_HLM-12_full_teleop` episode and build the converter/validator to the common episode format in `docs/datasets.md`. Check 34-D semantics, camera timestamps, and units rather than assuming from the dataset label.
4. Create a FastWAM G1 data config and train on the selected public subsets, initialized from pretrained Wan2.2 video DiT and interpolated ActionDiT. Record dataset stats and held-out task results. This is the first true FastWAM-on-G1 milestone.
5. Gather or select matching teleoperation episodes, fine-tune, and connect the trained output through a validated SONIC motion-reference adapter in free-base MuJoCo. Evaluate reach, grasp, navigation, and place tasks separately.

## Reproduce verification

```powershell
cd C:\Users\Justin\Desktop\pjt\fastwam_g1_demo
python scripts/validate_contract.py
python scripts/action_adapter.py
python scripts/run_demo.py --task both --out artifacts/run.json
python scripts/sonic_bridge.py --actions artifacts/scripted_actions.npy --out artifacts/sonic_reference/scripted_reaches --origin scripted_ik
python scripts/verify_bridge.py
python scripts/play_sonic_reference.py --clip artifacts/sonic_reference/scripted_reaches
```
