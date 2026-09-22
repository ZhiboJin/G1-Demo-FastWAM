# Architecture and integration boundary

> Historical IK/CSV design. The subsequently implemented FastWAM factory and
> real SONIC v1.1 encoder integration are documented in
> [model_architecture.md](model_architecture.md), which supersedes the model and
> SONIC-choice sections below. The simulator described here is still scripted IK.

```text
instruction + head/wrist RGB + G1 proprioception
        │
        ▼
FastWAM (Wan2.2 video DiT + ActionDiT initialized from existing weights)
        │  OpenHLM-style action chunk, currently 34 values/frame
        ▼
action_adapter.py → named 29 joint targets + 2 hand blocks + root block
        │
        ├─ current demo: fixed-torso joint PD, wrist IK supplies targets
        └─ implemented: URDF-validated SONIC joint-reference CSV export
                 └─ current: kinematic MuJoCo reference visualization
                 └─ future: SONIC ONNX controller → free-base G1 MuJoCo
```

The current sim uses the 29-DoF G1 joint model and an anchored torso. It has neither camera observations for model inference nor an action-trained FastWAM checkpoint. The action adapter has no implicit hand actuation. `sonic_bridge.py` exports the 34-D trajectory into the joint-reference format documented by NVIDIA for SONIC encoder mode 0. This is a batch interface, so it does not yet provide closed-loop online control. Directly sending the 34 values as SONIC latent tokens is not valid: NVIDIA's VLA route uses 64-D motion tokens plus 14 hand values.

## Exact action ordering

With one value per hand, offsets are left arm `[0:7]`, left hand `[7:8]`, right arm `[8:15]`, right hand `[15:16]`, left leg `[16:22]`, right leg `[22:28]`, waist `[28:31]`, root `[31:34]`. This follows the [OpenHLM G1 deployment example](https://github.com/OpenHLM-project/OpenHLM/blob/main/src/openpi4OpenHLM/examples/unitree_g1/sonic_g1_env.py). The hand block sizes are user-editable in `configs/action_space.json`. If either changes, every later offset changes automatically. The root block is roll, pitch, yaw angular velocity. The baseline does not command it.

## FastWAM changes required for training

Use the existing sibling `../FastWAM` checkout as the upstream implementation. Its `configs/data/libero_2cam_lerobot_v30.yaml` assumes a 7-D action and 8-D state. A G1 data config must set `shape_meta.action` and `processor.action_output_dim` to the action contract dimension; state size should be computed from the chosen proprioception encoding. The processor's delta-action mask also needs an explicit per-dimension decision: absolute joint targets and hand positions should not inherit LIBERO's default 7-D mask. Keep missing hand channels masked rather than invented when co-training across embodiments. The released LIBERO/RoboTwin checkpoint action heads and dataset statistics are not compatible with G1 without adaptation and training.

## SONIC choice

Use NVIDIA's released GEAR-SONIC controller for the later free-base milestone. Its published G1 variants include default and `sonic_v1_1`, with matching ONNX deployment files. Start with the default release for joint-reference tracking; use `sonic_v1_1` only after checking its heading-normalized conventions. The local `../GR00T-WholeBodyControl-main/GR00T-WholeBodyControl-main` checkout contains SONIC code, but its mesh assets are Git LFS pointers, ONNX checkpoints are absent, and its Linux/DDS runtime is not installed here. The implemented bridge checks 50 Hz, URDF limits, and SONIC joint ordering; the actual ONNX controller must still be run and measured in the official MuJoCo loop.
