# FastWAM × Unitree G1 — whole-body control through SONIC

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

## Status — read this first

| Piece | State |
|---|---|
| 34-D action contract, joint ordering, limits | **Working**, verified against NVIDIA's source |
| SONIC v1.1 encoder (1751 → 64, incl. FSQ) | **Working** — real released ONNX graph |
| SONIC v1.1 decoder (994 → 29) | **Working** — real released ONNX graph |
| `q_target = default + action * scale` convention | **Resolved from source**; previously an open question |
| MuJoCo G1 plant with the deployment's PD gains | **Working**, but see the limitation below |
| Closed loop, free base | **Runs**; stable for the full 5 s run at reduced effective gain |
| FastWAM → G1 inference | **Not possible yet** — upstream has released no G1 checkpoint |

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

## Quickstart

```bash
# 1. Environment (kept inside the repo; the demo needs no GPU)
python3 -m venv --system-site-packages .venv-model
.venv-model/bin/pip install -r requirements.txt

# 2. Pretrained SONIC v1.1 encoder + decoder (~200 MB, public, ungated).
#    If ~/.cache is not writable, keep Hugging Face's cache in the repo:
export HF_HOME=$PWD/.cache/huggingface
.venv-model/bin/python -m g1demo.cli download-sonic

# 3. Verify the contract and the SONIC chain (no physics, no GPU)
.venv-model/bin/python -m g1demo.cli verify          # 16/16 checks

# 4. Closed-loop simulation
.venv-model/bin/python -m g1demo.cli demo --motion standing --action-gain 0.5

# 5. Tests
.venv-model/bin/python -m unittest discover -s tests
```

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
  sim/g1_mujoco.py   mesh-free G1 plant with the deployment's PD gains
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
| `g1demo/policy.py` | Where does the reference come from? |
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
   `FastWAMPolicy` is the drop-in point once such a checkpoint exists.
3. **Choose the hands.** `configs/action_space.json` currently carries one scalar
   per hand as a placeholder; change `end_effectors.*.size` and the contract, model
   width and routing follow automatically.
