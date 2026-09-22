# The sim2sim gap

This document records exactly what was measured when the SONIC decoder was driven
against a MuJoCo G1. It exists so the next person does not have to re-derive it,
and so nothing here is mistaken for a validated result.

## Summary

The Python chain is correct. The **plant** is not a faithful stand-in for
IsaacLab, and the SONIC decoder was trained against IsaacLab. At the deployment's
nominal gain the closed loop diverges and the robot falls. It stands for the full
run only once the effective action gain is reduced.

All figures below are on the **real G1** (MuJoCo Menagerie, Unitree G1 29-DoF
rev 1.0, 33.3 kg), free base, standing reference, 250 ticks = 5 s at 50 Hz.

| Data | Value |
|---|---|
| Control tick 0 | `max abs action = 0.437`, matching the open-loop prediction exactly |
| Tick 0 `q_target - default_angles` | max `0.153 rad`, 0/29 joints saturated |
| SONIC gains, action gain 1.0 | falls at tick 56 (1.1 s), 11.2% saturation |
| SONIC gains, action gain **0.5** | **stands 250 ticks**, mean tracking error `0.056 rad`, **0.0% saturation** |
| SONIC gains, action gain 0.35 | stands 250 ticks, `0.061 rad` |
| Native gains (kp=500), action gain 1.0 | falls at tick 44 |
| Native gains (kp=500), action gain **0.25** | **stands 250 ticks**, mean tracking error `0.007 rad` |
| Plant holding the default pose, no policy, SONIC gains | topples in < 1 s |
| Plant holding the default pose, no policy, native gains | stable (settles to `0.754 m`, 3 mm drift) |

Two things follow. The plant's own stiffness changes how much the decoder's
output must be attenuated, and a stiffer plant tolerates a *lower* gain, not a
higher one — consistent with the loop being gain-limited rather than
torque-limited.

Reproduce:

```bash
python -m g1demo.cli gain-sweep --source menagerie --model-gains sonic  --ticks 250
python -m g1demo.cli gain-sweep --source menagerie --model-gains native --ticks 250
python -m g1demo.cli demo --source menagerie --model-gains sonic --action-gain 0.5
```

## What is *not* the cause

These were all checked and eliminated, which is what makes the surviving
explanation credible.

| Hypothesis | Test | Result |
|---|---|---|
| Wrong ISAACLAB ↔ MuJoCo joint permutation | elementwise comparison against the C++ `q_target` expression | correct; permutations are exact inverses |
| Wrong encoder input convention | compare against `GatherMotionJointPositionsMultiFrame` | correct — absolute angles, IsaacLab order |
| Wrong anchor-orientation layout | compare against `GatherMotionAnchorOrientationMutiFrame` | correct — first two columns, row-wise |
| Mode indicator should be one-hot | read `GatherEncoderMode` | it is `[mode, 0, 0, 0]`; zero-filling mode 0 is correct |
| `last_actions` should be scaled | read `CreatePolicyCommand()` | it is the raw normalized output in IsaacLab order |
| Decoder history should be absolute joint angles | read `state_logger.cpp:296` | it is relative to `default_angles`; implemented that way |
| Relative pose is statically unstable | project the CoM | CoM `x = 0.029` lies inside the foot span |
| Model mass is wrong | `body_mass.sum()` | `33.3 kg` (Menagerie) / `35.1 kg` (NVIDIA MJCF), both G1-plausible |
| Physics timestep too coarse | sweep 1–5 ms × Euler/implicitfast | identical outcome at every setting |
| Contact geometry too point-like | replace foot spheres with a sole box | did not help; moved the standing height off nominal |
| Integrator choice | Euler vs `implicitfast` | neither stabilises it |
| Hidden motor gain scaling | `apply_motor_gain_scales` | defaults are `nullopt`; no scaling applied |
| Wrong model | run both independent G1 models | both behave the same way; the effect is gains, not geometry |

## The two real findings

### 1. At SONIC's gains the plant cannot hold the default pose passively

Commanding `q_target = default_angles` and nothing else, the robot topples within
about a second. Tracking the mechanism:

```text
t=0.00s  h=0.732  pitch=+0.006  worst: left_ankle_pitch  -0.363 -> -0.566  (sag 0.203)
t=0.50s  h=0.684  pitch=+0.358  worst: left_ankle_pitch  -0.363 -> -0.837  (sag 0.474)
t=1.00s  h=0.041  pitch=+1.493  fallen
```

The ankle pitch sags under the ground-reaction moment, the body pitches forward,
the CoM moves ahead of the ankle, and the sag runs away. A `P` term is what
resists it, so the joint is only stable if `kp_ankle` is large enough.

Measuring the torque the real G1 actually needs to stand, and the sag that torque
implies at SONIC's gains:

| Joint | Torque to stand | Sag at SONIC kp | Sag at Menagerie kp=500 |
|---|---|---|---|
| knee | 10.9 N·m | 0.110 rad | 0.022 rad |
| ankle pitch | 5.6 N·m | 0.196 rad | 0.011 rad |

At 0.196 rad of ankle sag the CoM passes the support polygon and the robot goes
over. Menagerie's own `kp=500` holds it. So this is a **gain** mismatch, not a
geometry, mass or contact bug — and running the same test on either G1 model
gives the same answer.

This is *not* evidence the deployment gains are wrong. On the real robot the
policy commands an offset to compensate, and the ankle has the headroom: 5.6 N·m
needed against a 25 N·m limit. It is evidence that the learned offset does not
transfer to this plant.

### 2. The plant's stiffness sets how much the decoder output must be attenuated

At action gain 1.0 the decoder drives the plant into divergence and `|action|`
grows to ~15. The decoder's *outputs* are identical whatever the plant; only the
response differs. And the stiffer the plant, the *lower* the usable gain:

| Plant servos | Gain that stands |
|---|---|
| SONIC kp (14–99) | 0.5 |
| Menagerie kp=500 | 0.25 |

A stiffer plant needing a *smaller* action scale is the signature of a
gain-limited loop, not a torque-limited one: MuJoCo's ideal position servos have
no actuator lag or velocity limit, so the loop has more gain and less phase margin
than the IsaacLab actuators the policy was trained against.

## Why this is not surprising

* The shipped MJCF (`g1_29dof_old.xml`) describes the **older** G1 hardware; the
  policy header describes the newer one and says so explicitly
  ("old is 7520_14 new is 7520_22"). For the hip pitch and ankles the two disagree
  on effort limit (139 vs 88, and 25 vs 50 N·m).
* MuJoCo and PhysX differ in contact stiffness, friction and solver behaviour.
* NVIDIA's own sim2sim path does **not** use a hand-built MuJoCo plant. It runs
  their C++ deployment binary (`deploy.sh sim`) against `run_sim_loop.py` over DDS,
  with TensorRT and the matching scene. That is the reference implementation.

## The fix path, in order

1. **Run NVIDIA's own sim2sim and diff the actions.** This is the decisive
   measurement and it needs their stack (Linux, CUDA, TensorRT 10.13). Export a
   reference clip, run the same frame through both, and compare the action buffers
   element-wise for identical state history:

   ```bash
   python -m g1demo.cli bridge-export --motion standing --horizon 200 --out artifacts/sonic_reference
   cd $SONIC_REPO
   bash install_scripts/install_mujoco_sim.sh
   python gear_sonic/scripts/run_sim_loop.py      # terminal 1
   cd gear_sonic_deploy && bash deploy.sh sim     # terminal 2
   ```

   If the two agree, our Python path is validated and the remaining gap is purely
   the MuJoCo plant. If they disagree, the difference localises the bug.

2. **Calibrate the plant to IsaacLab** rather than adding a fudge factor: match the
   actuator model (effort limit, velocity limit, first-order lag) and the contact
   parameters. `G1Sim` exposes `policy_effort_limits`, `foot_sole` and `integrator`
   as switches precisely so these can be ablated instead of assumed.

3. **Or use IsaacLab directly.** Since SONIC is an IsaacLab policy, training and
   evaluating in IsaacLab removes the transfer question entirely; MuJoCo then
   becomes a visualisation rather than a validation environment.

## What would count as resolved

* The Python decoder's action buffer matches the C++ controller's for the same
  reference and state history.
* The free-base plant holds a reference at gain **1.0** — NVIDIA's convention,
  unmodified — for a run long enough to be meaningful.
* A learned FastWAM G1 checkpoint drives the loop instead of a scripted reference.

Until then, `action_gain < 1.0` is a **calibration knob, not a result**, and the
decoded action is not a verified motor command.
