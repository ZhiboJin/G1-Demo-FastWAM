# G1 demo: start here

For a fresh clone, follow [INSTALL.md](INSTALL.md) first.

This is the **G1-Demo-FastWAM** git repository at
`robotics/G1-Demo-FastWAM/`. Its sibling `robotics/FastWAM/` is a separate checkout
of the upstream FastWAM research code. This repository pins the unmodified
official SONIC and MuJoCo Menagerie sources as Git submodules under `third_party/`.
For the standing demo after cloning, populate only the G1 model with
`git submodule update --init --depth 1 third_party/mujoco_menagerie`.
The full SONIC source submodule is optional for source comparison and URDF tools;
the demo uses its released ONNX graphs, downloaded separately.
The local `g1demo/sonic/` folder is a small adapter to SONIC's released ONNX
encoder and decoder, not a replacement for SONIC.
This minimal standing loop is separate from SIMPLE's SONIC environment; there
is currently no FastWAM-to-SIMPLE inference bridge in this repository.

## What runs now

```text
scripted reference [T,46] -> SONIC encoder -> 64-D token
    -> SONIC decoder + measured state history -> 29 body joint targets -> MuJoCo G1
                         └─ left/right end-effector commands -> callback
```

`python -m g1demo.cli verify` checks the contract and ONNX graphs. Run
`python -m g1demo.cli demo --motion standing --action-gain 0.5` from this folder
for the current simulation. See `README.md` for installation and other motions.
The 0.5 gain is a simulation workaround described in `docs/sim2sim_gap.md`, not
a validated setting for physical hardware.

The intended learned path is:

```text
language instruction + uint8 RGB camera frame + 29 measured body angles
    -> FastWAMPolicy.predict() -> [T,46] physical-unit reference
    -> same SONIC encoder/decoder and end-effector routing
```

The 46 values are 29 absolute body angles, three root orientation channels,
and seven Dex3 joint targets per hand. SONIC encodes the body and root reference;
its graphs have no hand inputs or outputs. `WholeBodyController.run(on_hand_command=...)`
emits both 7-joint hand arrays each tick. The local 29-DoF MuJoCo model does not
contain hands, so this callback has no physical hand effect yet.

## Where to look

| Path | Purpose |
|---|---|
| `configs/action_space.json` | Names, order and dimensions of the G1 action vector |
| `configs/fastwam_g1.yaml` | Model architecture for a future trained G1 checkpoint |
| `g1demo/policy.py` | Scripted reference and real FastWAM inference adapter |
| `g1demo/loop.py` | 50 Hz orchestration and body/hand command boundary |
| `g1demo/sonic/` | SONIC ONNX input packing and inference |
| `g1demo/sonic_params.py` | Joint-order and action-to-target conversions |
| `g1demo/sim/` | MuJoCo G1 plant, body only |
| `third_party/GR00T-WholeBodyControl/` | Pinned, unmodified official SONIC source |
| `third_party/mujoco_menagerie/` | Pinned G1 MuJoCo model and meshes |
| `g1demo/bridge.py` | Export a reference clip for NVIDIA's C++ simulator |
| `docs/architecture.md` | Detailed tensor and control conventions |
| `docs/PROGRESS.md` | Decisions, completed checks, and next tasks |
| `docs/SONIC_COMPARISON.md` | Measured Python/official-source comparison and runtime gap |
| `docs/TRAINING_STAGE2.md` | MotionWAM stages, SIMPLE tasks and data contract |

## What is still required

1. Train a G1 FastWAM checkpoint with this exact action order and fit **separate**
   46-channel action and 29-channel proprioception mean/scale statistics. The
   released LIBERO/RoboTwin heads are incompatible. `FastWAMPolicy.load()`,
   `set_normalizer()`, and `set_proprio_normalizer()` take these inputs.
2. Supply a synchronized RGB camera frame to `FastWAMPolicy.predict()` or an
   `observation_provider(sim)` to `actions()`. The image must be RGB uint8 with
   height and width divisible by 16. Select camera, crop, and resolution during
   training and keep them the same at inference.
3. Validate the Dex3 hand joint limits, 50 Hz command transport, and emergency
   behavior. The 7+7 channel order is in `configs/action_space.json`; implement
   the hand callback for real hardware or SIMPLE's hand model.
4. Compare the Python SONIC command against NVIDIA's `deploy.sh sim` on the
   same reference and state trace. Resolve the MuJoCo plant gap before trying
   free-base motion or sending commands to the physical G1.

There is no trained G1 FastWAM policy or verified real-robot command transport in
this repository yet. The working demo establishes the contract and SONIC path;
it does not establish task success or physical deployment readiness.
