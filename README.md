# FastWAM × Unitree G1 simulation demo

## Model architecture and real SONIC encoder (new)

The G1 FastWAM factory and SONIC v1.1 encoder integration are now implemented.
FastWAM uses 29 named joint channels, configurable left/right hand channels
(currently one each), and the existing three root-orientation channels: 34 total.
The hand channels bypass the encoder. The actual downloaded NVIDIA encoder/FSQ
produces 64-value tokens; a sample is saved in `artifacts/sonic_packet.npz`.
Read [the implemented model architecture](docs/model_architecture.md) for exact
semantics, code entry points, reference timing and reproducible commands.

This is architecture/interface validation, not a trained FastWAM G1 policy.
The original simulation below still uses scripted IK. Full-size FastWAM weights,
G1 training, and closed-loop C++ SONIC controller execution have not been performed.

## SONIC decoder and the complete token chain (new)

Both halves of the pretrained SONIC v1.1 path now run in Python. The decoder graph
(`checkpoints/sonic_v1_1/model_decoder.onnx`, 150 MB) was added along with
`scripts/sonic_decoder.py`, a symmetric CLI `scripts/decode_sonic.py`, and
`scripts/verify_decoder.py`, which executes encoder and decoder in sequence over 31
control ticks and passes 16/16 checks. `G1Policy` now exposes `decode_at()` and a
composite `step()`:

```text
predict -> [T,34] physical actions -> encode_at -> 64-D token + hands
                                   -> decode_at -> [1,29] joint action
```

The 994-value decoder observation is derived from official source, not guessed; see
[the decoder section](docs/model_architecture.md#decoder-observation-994-values).
**The decoded action magnitudes are not yet validated** — output reaches
`|action| ~ 9 rad` on out-of-distribution scripted references, so the C++ convention
`q_target = default_angles + action * scale` still needs the official Linux sim2sim
to confirm. Do not treat the decoded action as a verified motor command.

```powershell
.venv-model/Scripts/python.exe scripts/verify_decoder.py
```


This folder is a **runnable MuJoCo task and action-contract baseline**, plus a staged path to a trained FastWAM humanoid policy. The current controller is Jacobian IK, **not FastWAM or SONIC inference**. It reaches two visual targets with the left and right G1 wrists. The torso is fixed to make the first manipulation milestone reproducible without a balance policy. See [STATUS.md](STATUS.md) before continuing.

## Run the first demo

From this folder on the present machine (Python 3.13 with `mujoco` and `numpy` already installed):

```powershell
python scripts/run_demo.py --task both --out artifacts/run.json
```

Use `--task left` or `--task right` for individual tasks. The command writes `artifacts/demo.png` with snapshots of both completed reaches. Add `--viewer` for the interactive MuJoCo viewer. Install `requirements-demo.txt` in a clean environment if needed. The script generates `assets/g1_meshfree.xml` from the existing local NVIDIA G1 model at `../GR00T-WholeBodyControl-main/GR00T-WholeBodyControl-main/gear_sonic/data/robots/g1/g1_29dof_old.xml`. The generated model preserves G1 joints, axes, limits, body transforms, and inertias, while replacing unavailable LFS meshes with simple visual capsules. It is an approximation and is not suitable for sim-to-real metrics. The generated file and rollout are reproducible artifacts; the upstream XML remains the authoritative model.

## Action contract

`configs/action_space.json` is the single place to edit dimensions. It exposes all 29 G1 joints plus two editable end-effector blocks and a three-value root command, matching the **34-value OpenHLM-style ordering**. `end_effectors.left.size` and `end_effectors.right.size` are explicitly marked `EDIT_ME`; currently each is `1` (a gripper placeholder). Change them for your actual hands, then rerun `python scripts/validate_contract.py`. The simulator demo consumes joint positions only; it does not pretend the placeholder grippers or root command are physically actuated.

## FastWAM action → SONIC reference bridge

The demo writes **script-generated** `[200,34]` actions to `artifacts/scripted_actions.npy` so the interface can be exercised before G1 FastWAM training. The bridge accepts the same `.npy` shape from a future FastWAM G1 inference run:

```powershell
python scripts/sonic_bridge.py --actions artifacts/scripted_actions.npy --out artifacts/sonic_reference/scripted_reaches --origin scripted_ik
python scripts/verify_bridge.py
python scripts/play_sonic_reference.py --clip artifacts/sonic_reference/scripted_reaches
```

The converter validates names and limits against the **G1 URDF** in the local SONIC checkout, applies SONIC's own IsaacLab joint permutation, and emits its documented `joint_pos.csv`, `joint_vel.csv`, `body_pos.csv`, `body_quat.csv`, and `metadata.txt` at 50 Hz. It keeps hand placeholders in `hands.json` because SONIC joint references do not drive them. Playback is kinematic, **not SONIC controller execution**. A trained G1 FastWAM checkpoint and C++ controller execution are still needed.

## Getting started on a fresh clone (Linux or Windows)

Nothing large is stored in git. The ONNX graphs are re-downloaded and the derived
config is regenerated locally.

```bash
# 1. Python environment for the demo + model checks
python -m venv .venv-model
source .venv-model/bin/activate          # Windows: .venv-model\Scripts\activate
pip install -r requirements-model.txt
pip install -r requirements-demo.txt
pip install huggingface_hub

# 2. Fetch the pretrained SONIC v1.1 encoder + decoder (~200 MB, public)
python scripts/download_sonic_checkpoints.py

# 3. Regenerate the machine-specific derived config
python scripts/build_g1_architecture.py

# 4. Verify
python scripts/validate_contract.py
python scripts/verify_model.py           # 8 checks
python scripts/verify_decoder.py         # 16 checks
python scripts/run_demo.py --task both --out artifacts/run.json
```

`scripts/verify_model.py` needs `artifacts/scripted_actions.npy`, which step 4's
`run_demo.py` writes. Run the demo once before the model checks on a fresh clone.

`scripts/verify_bridge.py` and `scripts/sonic_bridge.py` additionally require a
local NVIDIA GEAR-SONIC checkout (for the G1 URDF and SONIC's joint permutation).
Point the demo at it with an environment variable instead of relying on a sibling
directory layout:

```bash
export SONIC_REPO=/path/to/GR00T-WholeBodyControl
export FASTWAM_REPO=/path/to/FastWAM        # only needed for tests that build the real model
```

`scripts/project_paths.py` resolves both, falling back to sibling directories.

## Repository layout

```text
configs/     action_space.json (authoritative contract), fastwam_g1.yaml
docs/        model_architecture.md, architecture.md, datasets.md
scripts/     contract + demo + bridge + encoder + decoder + verification
assets/      g1_meshfree.xml (generated from the upstream G1 MJCF)
artifacts/   run outputs and verification reports (mostly regenerated)

checkpoints/ NOT in git - run scripts/download_sonic_checkpoints.py
```

## Training stages

1. **Video DiT initialization, no pretraining run.** Use the upstream FastWAM Wan2.2 TI2V 5B initialization and its ActionDiT interpolation script. The released LIBERO checkpoint is useful as a code smoke test, but its action head and normalization are for LIBERO and cannot directly drive G1.
2. **Posttrain on public robot data.** Start with OpenHLM G1 whole-body data, then add NVIDIA synthetic G1 locomanipulation for task variety. Consider Unitree UnifoLM WBT subsets after inspecting individual schemas and licenses. Convert episodes into aligned camera, proprioception, 34-D action, mask, timestamp, and instruction records; normalize per embodiment and retain a held-out task split. This converter and GPU training are the next implementation milestone.
3. **Fine-tune on matching teleoperation episodes.** Capture or select G1 task demonstrations with the same camera/action contract, and fine-tune at lower rate. Validate in simulation before any hardware work.

SONIC is appropriate as the eventual balance and whole-body execution layer, but its released interface accepts motion references/tokens rather than this raw OpenHLM-style 34-D action unchanged. `docs/architecture.md` records the adapter boundary and the required validation. Do not label the current IK controller as SONIC.

## Key upstream sources

- [FastWAM code and checkpoints](https://github.com/yuantianyuan01/FastWAM)
- [OpenHLM action mapping and deployment example](https://github.com/OpenHLM-project/OpenHLM/blob/main/src/openpi4OpenHLM/examples/unitree_g1/sonic_g1_env.py)
- [OpenHLM G1 dataset and model descriptions](https://github.com/OpenHLM-project/OpenHLM)
- [NVIDIA GEAR SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [NVIDIA G1 locomanipulation dataset](https://huggingface.co/datasets/nvidia/g1_locomanip_dataset)
- [Unitree UnifoLM WBT collection](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset)
