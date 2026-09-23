# FastWAM × Unitree G1 — whole-body control through SONIC

**New clone? Follow [INSTALL.md](INSTALL.md)** for the tested standing demo.
Then read [START_HERE.md](START_HERE.md) for the repository map and the remaining
work before robot deployment.
The current comparison and training work is in
[docs/SONIC_COMPARISON.md](docs/SONIC_COMPARISON.md) and
[docs/TRAINING_STAGE2.md](docs/TRAINING_STAGE2.md).

Deploying **FastWAM** (a world-action model) onto a **Unitree G1** humanoid, using
NVIDIA's **SONIC** (GEAR-SONIC) controller as the whole-body execution layer, and
demonstrating the chain in MuJoCo.

```text
     instruction + cameras + proprioception
                     │
                 FastWAM                      predicts an action chunk
                     │
             [T, 34] physical references
             ├──────────────────────────────┐
             │ 29 body joints + 3 root      │  left / right hand commands
             ▼                              ▼
   SONIC v1.1 encoder  (incl. FSQ)      BYPASSED by SONIC
             │
      64-D whole-body token
             │
   SONIC v1.1 decoder
             │
      29 joint actions  (IsaacLab order)
             │
   q_target = default_angles + action * scale      ← NVIDIA's convention
             │
      Unitree G1  (50 Hz position PD)
```

The hands bypass SONIC because its graphs have no hand inputs or outputs. In
NVIDIA's own deployment the dexterous-hand commands travel on a separate channel
and never enter the policy.

This diagram is the **small local standing test**, not the whole GEAR-SONIC
deployment. The official stack also handles input streaming, state/history,
the optional kinematic planner, controller timing, TensorRT inference, motor
commands, and simulator/robot transport. SIMPLE's SONIC task uses that external
controller. Its policy evaluation expects a 78-value action per step:
`[SONIC token (64), left Dex3 hand (7), right Dex3 hand (7)]`; it sends the token
over ZMQ and receives low-level commands back through its Unitree bridge. This
repository does not yet connect FastWAM to that SIMPLE interface. See
[docs/architecture.md](docs/architecture.md) for the two paths.

## Status — read this first

| Piece | State |
|---|---|
| 34-D action contract, joint ordering, limits | **Working**, verified against NVIDIA's source |
| SONIC v1.1 encoder (1751 → 64, incl. FSQ) | **Working** — real released ONNX graph |
| SONIC v1.1 decoder (994 → 29) | **Working** — real released ONNX graph |
| `q_target = default + action * scale` convention | **Resolved from source**; previously an open question |
| MuJoCo G1 plant with the deployment's PD gains | **Working**, but see the limitation below |
| Closed loop, free base | **Runs**; stable for the full 5 s run at reduced effective gain |
| FastWAM → G1 inference | **Adapter implemented, untested with G1 weights** — training and statistics are still needed |

**The honest limitation.** Driving the SONIC decoder against a MuJoCo G1 built
from NVIDIA's shipped MJCF does not balance at the deployment's nominal gain. The
MuJoCo plant is much stiffer than the IsaacLab plant the policy was trained
against, so at gain 1.0 the loop diverges and the robot falls, while at an
effective gain of 0.5 it stands for the full 5 s run. Separately, the MuJoCo plant
cannot even hold the default pose passively: the ankle sags under the
ground-reaction moment until the body pitches over. Both are *plant fidelity*
problems, not convention problems — the first control tick matches the open-loop
prediction exactly, and every tensor layout has been checked against the C++.

Full detail, measurements and the fix path: [`docs/sim2sim_gap.md`](docs/sim2sim_gap.md).

## See it run

| Artefact | What it shows |
|---|---|
| `artifacts/standing.mp4` | the G1 holding a standing reference for 5 s under SONIC control (gain 0.5) |
| `artifacts/standing_gain1.0.mp4` | the same run at NVIDIA's nominal gain — falls at 1.1 s |
| `artifacts/standing_gain0.5.png` | pelvis height, decoder output, tracking error, commanded targets |
| `artifacts/standing_gain0.5_token.png` | the 64-D FSQ token the encoder produced |

Rendered on the laptop GPU with `MUJOCO_GL=egl` (use `glfw` when a display exists).

## Which G1

Two models are wired in, selected with `--source`:

| `--source` | What it is |
|---|---|
| `menagerie` (default) | the official **Unitree G1 29-DoF rev 1.0** from MuJoCo Menagerie — real collision meshes, 33.3 kg, checkered scene. This is a real G1. |
| `nvidia` | a rebuild of the MJCF shipped inside GEAR-SONIC. Its STLs are Git-LFS pointers, so a plain clone cannot load it; the rebuild keeps the kinematics, inertias, limits and foot contact spheres and replaces the meshes with primitives. It renders as a stick figure. |

Both expose the 29 actuators in the same order as SONIC's policy parameters, so
neither needs a permutation.

`--model-gains` picks the servos:

| `--model-gains` | kp | Result unaided |
|---|---|---|
| `sonic` (default) | 14–99, from SONIC's C++ header | topples — too soft to hold the pose |
| `native` | 500, Menagerie's own | stands |

That difference is the substance of the sim2sim gap: the policy was trained
against a plant it cannot stabilise here without calibration.

**Compute reality.** CUDA works here, so the simulation renders on the GPU. The
7.5 GiB of VRAM is *not* enough for FastWAM itself: it is a ~6 B parameter model,
about 12 GB in bf16 before activations, and upstream ships no quantized build.
Running or fine-tuning FastWAM therefore needs a larger GPU; everything up to and
including the SONIC chain runs fine on CPU.

**If GPU commands fail with `cuInit` error 304** and `nvidia-smi` says "Failed to
initialize NVML", check whether you are inside a sandbox that restricts device
access. That is not a driver fault: the same commands work in a normal shell.


## Quickstart

```bash
# 0. Clone your development repository.
git clone https://github.com/ZhiboJin/G1-Demo-FastWAM.git
cd G1-Demo-FastWAM

# 1. G1 MuJoCo model. The full SONIC source is optional for this demo.
git submodule update --init --depth 1 third_party/mujoco_menagerie

# 2. Environment (kept inside the repo; the demo needs no GPU)
python3 -m venv --system-site-packages .venv-model
.venv-model/bin/pip install -r requirements.txt

# 3. Pretrained SONIC v1.1 encoder + decoder (~200 MB, public, ungated).
#    If ~/.cache is not writable, keep Hugging Face's cache in the repo:
export HF_HOME=$PWD/.cache/huggingface
.venv-model/bin/python -m g1demo.cli download-sonic

# 4. Verify the contract and the SONIC chain (no physics, no GPU)
.venv-model/bin/python -m g1demo.cli verify          # 16/16 checks

# 5. Closed-loop simulation (add MUJOCO_GL=egl --video to record it)
.venv-model/bin/python -m g1demo.cli demo --motion standing --action-gain 0.5

# 6. Tests
.venv-model/bin/python -m unittest discover -s tests
```

If the download reports `Unknown scheme for proxy URL ... socks://`, this
machine's `ALL_PROXY` setting is incompatible with the Hugging Face client.
With an HTTP proxy already configured, retry step 3 as
`env -u ALL_PROXY -u all_proxy HF_HOME=$PWD/.cache/huggingface .venv-model/bin/python -m g1demo.cli download-sonic`.

Path resolution lives in `g1demo/paths.py`. `MENAGERIE_REPO`, `SONIC_REPO` and
`FASTWAM_REPO` override the default locations. This checkout lives beside
`FastWAM/` and `SIMPLE/`; SONIC and Menagerie are pinned submodules in this
repository. SIMPLE supplies simulation tasks and data; the original FastWAM
checkout supplies the policy code when a trained checkpoint is available.
Locally, the repositories are siblings:
`/home/justin/robotics/{G1-Demo-FastWAM,FastWAM,SIMPLE}`. A GitHub clone of this
repository contains neither sibling; SIMPLE must be cloned separately to run
its embodied-task environments.
If you need NVIDIA's full SONIC source for the C++ comparison or URDF tools,
run `git submodule update --init --depth 1 third_party/GR00T-WholeBodyControl`.
The standing demo uses SONIC's downloaded ONNX graphs and does not need that
large source checkout.

## Commands

| Command | What it does |
|---|---|
| `verify` | 16 checks over the contract, encoder, decoder and action convention |
| `demo` | Closed-loop MuJoCo run; `--mode free\|anchored`, `--plot`, `--video` |
| `gain-sweep` | Calibrates the effective loop gain of the MuJoCo plant |
| `download-sonic` | Fetches and hash-checks the SONIC v1.1 ONNX graphs |
| `extract-params` | Regenerates `configs/g1_sonic_params.json` from SONIC's C++ header |
| `bridge-export` | Writes a SONIC reference clip for NVIDIA's own C++ simulator |

## Layout

```text
g1demo/
  contract.py        the 34-D action layout — the single source of truth
  sonic_params.py    G1 constants, and the joint-order conversions
  paths.py           where the repos, checkpoints and outputs live
  sonic/
    encoder.py       motion reference -> 64-D token
    decoder.py       token + proprioception -> 29 joint actions
  sim/g1_mujoco.py   G1 MuJoCo plant (Menagerie or mesh-free NVIDIA model)
  loop.py            the closed control loop
  policy.py          FastWAM adapter + scripted stand-in references
  bridge.py          SONIC reference-clip export for NVIDIA's C++ stack
  download.py        fetches and hash-checks the ONNX graphs
  cli.py             command line entry points
configs/
  action_space.json       editable action contract (hand sizes live here)
  fastwam_g1.yaml         FastWAM model config for the G1
  g1_sonic_params.json    generated; do not hand-edit
tools/extract_sonic_params.py
tests/test_pipeline.py
docs/
```

## Reading guide

The modules are layered so each answers exactly one question. Read in this order
and you follow the data path from end to end:

| File | Question it answers |
|---|---|
| `g1demo/contract.py` | What does FastWAM predict, and which parts bypass SONIC? |
| `tools/extract_sonic_params.py` | Where do the G1 constants come from? |
| `g1demo/sonic_params.py` | How do the three joint orderings convert into each other? |
| `g1demo/sonic/encoder.py` | How does a motion reference become a 64-D token? |
| `g1demo/sonic/decoder.py` | How does a token become 29 joint actions? |
| `g1demo/sim/g1_mujoco.py` | What robot is it running on? |
| `g1demo/loop.py` | What happens in one 50 Hz control tick? |
| `g1demo/policy.py` | Where does the scripted or learned reference come from? |
| `g1demo/cli.py` | What can I run? |

`g1demo/loop.py`'s module docstring is the whole pipeline in six lines, so it is
the fastest place to start.

By task:

| I want to… | Go to |
|---|---|
| understand the control path | `g1demo/loop.py` |
| change the hands or add a joint | `configs/action_space.json`, then `g1demo/contract.py` |
| change how a reference is encoded | `g1demo/sonic/encoder.py` |
| debug a joint-ordering problem | `g1demo/sonic_params.py` |
| understand why balance fails | `docs/sim2sim_gap.md` |
| drive NVIDIA's own simulator | `g1demo/bridge.py` (`cli bridge-export`) |
| swap in a trained FastWAM | `g1demo/policy.py` |
| regenerate constants after a SONIC update | `cli extract-params` |

Style is enforced by `ruff` (`python -m ruff check`), configured in
`pyproject.toml`.

## Where the constants come from

Nothing here guesses a magic number. The joint permutations, default standing
pose, action scales, PD gains, effort limits and joint limits are *evaluated from
NVIDIA's own C++ source* by `tools/extract_sonic_params.py`, which parses their
expressions rather than transcribing 150+ numbers by hand and cross-checks the
MJCF against the URDF. The generated file records the source's SHA-256. The
encoder and decoder then need **no** SONIC checkout at all.

## Upstream

- [FastWAM](https://github.com/yuantianyuan01/FastWAM) — the policy
- [SONIC / GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) — the controller
- [SONIC project page](https://nvlabs.github.io/GEAR-SONIC/) · [paper](https://arxiv.org/abs/2511.07820) · [docs](https://nvlabs.github.io/GR00T-WholeBodyControl/)
- [SONIC v1.1 weights](https://huggingface.co/nvidia/GEAR-SONIC)

## What is needed next

1. **Close the sim2sim gap** — run NVIDIA's own `deploy.sh sim` and compare its
   action buffer against `SonicDecoder` for the same reference and state history.
   `bridge-export` writes the reference clip that step needs. This is the single
   measurement that turns the decoder output into a verified motor command.
2. **Train a G1 FastWAM head.** Upstream ships LIBERO and RoboTwin checkpoints
   whose action heads are 7- and 14-wide; neither can drive a 34-wide G1 head.
   `FastWAMPolicy.predict()` is wired to the upstream inference API and needs a
   G1 checkpoint plus action/proprioception statistics and an RGB camera frame.
3. **Choose the hands.** `configs/action_space.json` currently carries one scalar
   per hand as a placeholder. Choose the actual command size, update both
   `action_dim` values in `configs/fastwam_g1.yaml`, and supply a driver via
   `on_hand_command`; this simulator has no hand actuators.
