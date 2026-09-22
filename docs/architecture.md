# Architecture

## The chain

```text
instruction + cameras + measured joints
        │
    FastWAM  (Wan2.2 video expert + ActionDiT + MoT)
        │
   [T, 34] action chunk, physical units, 50 Hz
        │
        ├─ 29 body joints (rad) ─┐         ┌─ left hand  (size from config)
        └─ 3 root channels ──────┤         └─ right hand (size from config)
                                 │                    │
                                 ▼                    │  BYPASS
                    SONIC v1.1 encoder (mode 0)       │
                    1751 → 64, FSQ inside             │
                                 │                    │
                    64-D whole-body token             │
                                 │                    │
                    SONIC v1.1 decoder                │
                    994 → 29                          │
                                 │                    │
         q_target = default + action * scale          │
                                 │                    │
                                 ▼                    ▼
                    Unitree G1 body (29 DoF)     hand controller
```

Everything left of the G1 is Python and runs on CPU. FastWAM itself needs CUDA;
no upstream G1 checkpoint exists, so the demo substitutes a scripted reference that
emits the identical `[T, 34]` interface.

## The 34-D action contract

`configs/action_space.json` is authoritative. With one scalar per hand:

| Slice | Meaning | Routed to |
|---|---|---|
| `0:7` | left arm joints | encoder |
| `7:8` | left hand | **bypass** |
| `8:15` | right arm joints | encoder |
| `15:16` | right hand | **bypass** |
| `16:22` | left leg joints | encoder |
| `22:28` | right leg joints | encoder |
| `28:31` | waist joints | encoder |
| `31:34` | root roll, pitch, yaw-rate | encoder orientation reference |

Joint values are absolute radians. Root units are rad, rad, rad/s. Hand units are
placeholders until hardware is chosen.

`g1demo/contract.py` parses this once per process and derives every slice, so
changing a hand size updates the packing, the FastWAM action width and the model
config check together. `FastWAMPolicy.load()` refuses a YAML whose `action_dim`
disagrees with the contract.

## Three joint orderings, and why that is the main hazard

| Order | Used by |
|---|---|
| **contract** | `action_space.json` — grouped by limb |
| **MuJoCo** (URDF) | the simulator, the MJCF, `default_angles`, PD gains |
| **IsaacLab** | *inside* both SONIC networks |

Mixing these silently corrupts every target, so the conversions live in exactly one
place, `g1demo/sonic_params.py`, and are derived from NVIDIA's own
`isaaclab_to_mujoco` / `mujoco_to_isaaclab` arrays rather than re-derived. They are
checked to be exact inverses at load time.

Two joint-position conventions also coexist:

* the encoder's reference input takes **absolute** angles;
* the decoder's proprioceptive history takes positions **relative to
  `default_angles`** (the C++ subtracts it).

`measured_to_isaaclab_relative()` implements the second; `pack_observation()` for
the encoder deliberately does not use it.

## Encoder

`g1demo/sonic/encoder.py`. Real released ONNX graph, verified `[1, 1751] → [1, 64]`.

* 10 reference frames at control offsets `[0, 5, …, 45]`, i.e. a 0.9 s window at
  50 Hz; the caller must supply 46 samples of lookahead, and short chunks are
  rejected rather than padded.
* The 64 outputs are FSQ-quantized **inside the graph**. They are integer multiples
  of 1/16. Never quantize again.
* `encoder_mode_4` is written as `[mode, 0, 0, 0]` — the C++ `GatherEncoderMode`
  writes the scalar mode then zeros, so mode 0 is four zeros, *not* one-hot.
* Orientation is the first two columns of the relative rotation matrix, flattened
  row-wise, matching `GatherMotionAnchorOrientationMutiFrame`.
* Teleop and SMPL slots stay zero: they are not in mode 0's required observations.
* Hands never enter the graph.

## Decoder

`g1demo/sonic/decoder.py`. Real released ONNX graph, verified `[1, 994] → [1, 29]`.

| Observation | Dim |
|---|---|
| `token_state` | 64 |
| `his_base_angular_velocity_10frame_step1` | 30 |
| `his_body_joint_positions_10frame_step1` | 290 |
| `his_body_joint_velocities_10frame_step1` | 290 |
| `his_last_actions_10frame_step1` | 290 |
| `his_gravity_dir_10frame_step1` | 30 |

Frame axis runs oldest → newest; joints are IsaacLab order; `his_last_actions` is
the raw normalized policy output, not the scaled target.

The action is **not** a torque and **not** an angle. NVIDIA's deployment applies:

```cpp
q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i];
```

with `action_scale = 0.25 * effort_limit / stiffness`. This is taken verbatim from
`policy_parameters.hpp` and `CreatePolicyCommand()` in
`g1_deploy_onnx_ref.cpp`, and
`g1demo.sonic_params.q_target_from_action()` reproduces it exactly (asserted
element-wise in the test suite).

## Plant

`g1demo/sim/g1_mujoco.py` rebuilds the official G1 MJCF without its LFS meshes. The
visual meshes upstream are `contype=0`; the only colliding geometry is four small
spheres per foot, which are primitives and survive untouched. Actuators are
position servos with the deployment's `kp`/`kd`, so the torque law matches
`kp*(q_target - q) - kd*qdot`.

## Loop

`g1demo/loop.py`, one tick at 50 Hz:

1. read the G1 state
2. encode the reference chunk → token
3. decode token + 10 frames of history → 29 raw actions
4. `q_target = default + gain * action * scale`, clipped to the joint limits
5. command the servos, step physics 4 × 5 ms
6. push the new measurement into the history

Hand commands are returned alongside each step and never touch steps 2–4.

## Constant provenance

`tools/extract_sonic_params.py` parses `policy_parameters.hpp`, evaluates its own
C++ constant expressions in a restricted evaluator, and writes
`configs/g1_sonic_params.json` with the source SHA-256. It also extracts joint
limits from the MJCF and cross-checks them against the URDF, and recovers each
joint's effort limit from the action-scale expression that references it.

Consequence: the encoder, decoder, sim and contract check need **no**
GEAR-SONIC checkout — only the vendored C++ header once, at generation time.
