# The sim2sim gap

This document records exactly what was measured when the SONIC decoder was driven
against a MuJoCo G1. It exists so the next person does not have to re-derive it,
and so nothing here is mistaken for a validated result.

## Summary

The Python chain is correct. The **plant** is not yet a faithful stand-in for
IsaacLab, and the SONIC decoder was trained against IsaacLab. At the deployment's
nominal gain the closed loop diverges and the robot falls. At an effective gain of
0.5 it stands for the full run.

| Data | Value |
|---|---|
| Control tick 0, free base | `max abs action = 0.437`, matching the open-loop prediction exactly |
| Tick 0 `q_target - default_angles` | max `0.153 rad`, 0/29 joints saturated |
| Free base, standing reference, gain 1.0 | falls at tick 82 (1.64 s), `max abs action 12.5`, 10.7% saturation |
| Free base, standing reference, gain 0.5 | **stands for 250 ticks (5.0 s)**, settles at `0.758 m`, mean tracking error `0.058 rad`, **0.0% saturation** |
| Free base, standing reference, gain 0.25 | falls at tick 109 |
| Plant holding the default pose with no policy | topples in < 1 s |
| Gain needed for the plant to hold the default pose | roughly **8×** the shipped `kp`/`kd` |

Reproduce:

```bash
python -m g1demo.cli gain-sweep --motion standing --ticks 250
python -m g1demo.cli demo --motion standing --action-gain 0.5 --ticks 250 --plot artifacts/run.png
python -m unittest tests.test_pipeline.TestSimulator
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
| Relative pose is statically unstable | project the CoM | CoM `x = 0.029` lies inside the foot span `[-0.051, 0.119]` |
| Model mass is wrong | `body_mass.sum()` | `35.1 kg`, matching the real G1 |
| Physics timestep too coarse | sweep 1–5 ms × Euler/implicitfast | identical outcome at every setting |
| Contact geometry too point-like | replace foot spheres with a sole box | did not help; moved the standing height off nominal |
| Integrator choice | Euler vs `implicitfast` | neither stabilises it |
| Hidden motor gain scaling | `apply_motor_gain_scales` | defaults are `nullopt`; no scaling applied |

## The two real findings

### 1. The MuJoCo plant cannot hold the default pose passively

Commanding `q_target = default_angles` and nothing else, the robot topples within
about a second. Tracking the mechanism:

```text
t=0.00s  h=0.732  pitch=+0.006  worst: left_ankle_pitch  -0.363 -> -0.566  (sag 0.203)
t=0.50s  h=0.684  pitch=+0.358  worst: left_ankle_pitch  -0.363 -> -0.837  (sag 0.474)
t=1.00s  h=0.041  pitch=+1.493  fallen
```

The ankle pitch sags under the ground-reaction moment, the body pitches forward,
the CoM moves further ahead of the ankle, and the sag runs away. The `P` term is
what resists it, so the equilibrium is only stable if `kp_ankle` is large enough —
and the shipped value (`28.5 N·m/rad`, i.e. `2 × STIFFNESS_5020`) is not, for this
model. Roughly 8× the shipped gains are needed before the plant stands on its own.

This is *not* evidence that the gains are wrong: the real robot is balanced by the
policy, which learns to command the offset that holds the pose. It is evidence that
the plant and the training plant differ enough that the learned offset does not
transfer.

### 2. Effective loop gain is about 2× too high

At gain 1.0 the decoded action drives the plant into divergence, with `|action|`
growing to ~15. Halving the applied action makes the same loop stand for the full
5 s. Since the decoder's *outputs* are identical in both cases, the difference is
purely how the plant responds — consistent with MuJoCo's ideal position servos
being much stiffer and faster than IsaacLab's actuator model, which applies effort
limits, velocity limits and a first-order lag.

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
