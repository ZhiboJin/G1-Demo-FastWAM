# Continuation log

## 2026-09-23

- Confirmed this is the requested `ZhiboJin/G1-Demo-FastWAM` checkout. It was
  nested under upstream FastWAM at the start of the session. Neither the
  engineai trees nor the SONIC source checkout were modified.
- Moved this demo checkout to `robotics/G1-Demo-FastWAM/`, beside the upstream
  `robotics/FastWAM/` checkout, and extended path discovery for the existing
  SONIC and Menagerie checkouts under `FastWAM/third_party/`.
- Preserved the existing SONIC v1.1 ONNX path and 34-channel action contract.
  SONIC handles 29 body joints and root reference; hand channels bypass it.
- Replaced the FastWAM inference stub with an explicit RGB + measured joints +
  instruction adapter. Fixed the checkpoint loader to pass a path to upstream
  `load_checkpoint()` and corrected the `model_dtype` argument. Enabled the text
  encoder in the G1 config so direct language prompts can be encoded.
- Added a per-tick end-effector callback and validation. This is a software
  command boundary, not a G1 hand-driver implementation.
- Added `START_HERE.md` to explain the two git repositories, current demo, and
  missing training/hardware pieces.
- Verified against the requested GitHub branch: local `main` already had five
  commits beyond remote `main` before this work. Changes in this session remain
  uncommitted and were not pushed.
- Checks: 58 unit tests pass (one known sim2sim test is an expected failure),
  Ruff passes, `verify` passes 16/16 SONIC checks, and a 10-tick free-base
  scripted standing demo runs without falling at simulation gain 0.5.
- The 16 `verify` checks are local checks in `g1demo/cli.py`, not an official
  NVIDIA SONIC certification or hardware validation.

## Next session

1. Record the actual G1 camera and hand hardware configuration and choose a
   50 Hz command transport. Define hand units and safe limits before wiring it.
2. Build a G1 training dataset in the 34-channel contract and save action and
   proprioception statistics with the checkpoint. Revisit the hand dimensions
   first if the hardware needs more than one value per side.
3. Run a trained checkpoint through `FastWAMPolicy.predict()` and a camera
   provider; verify the output width, normalization and 46-frame SONIC lookahead.
4. Compare to official SONIC `deploy.sh sim`, then validate control and safety
   on a supported physical G1 setup.

## 2026-09-23 — comparison and Stage-2 follow-up

- Re-generated G1 constants from NVIDIA's clean SONIC C++ checkout. The output
  matches `configs/g1_sonic_params.json` byte for byte. Re-ran a 250-tick
  free-base standing comparison: action gain 1.0 falls at tick 56; gain 0.5
  completes 250 ticks. Full result and reproducible commands are in
  `docs/SONIC_COMPARISON.md`.
- Official `deploy.sh sim` parity remains outstanding: this machine lacks a
  built C++ deployment binary, `.venv_sim`, deployed model files, and visible
  TensorRT libraries. Added optional `demo --trace` output for aligned future
  comparisons. Do not claim Python/official runtime parity yet.
- Read the MotionWAM three-stage paper and documented the distinction between
  its Stage-2 SONIC latent target and this repo's 34-D joint-reference path.
  The paper's Stage-1 egocentric checkpoint/code was not found publicly;
  FastWAM's Wan2.2 initialization is a different model from MotionWAM's Cosmos
  Video DiT. See `docs/TRAINING_STAGE2.md`.
- Added `prepare-stage2` to validate synchronized 50 Hz G1 episodes and derive
  64-D SONIC FSQ-vector + hand targets. This is a data-preparation baseline,
  not a trained Stage-2 policy or an exact MotionWAM index objective.
- Cloned the clean official SIMPLE checkout to sibling `robotics/SIMPLE/` and
  selected three task candidates in `configs/demo_tasks.yaml`. SIMPLE is not
  installed or run; its MP tasks use AMO/CuRobo, while its one documented SONIC
  task is a carry-box teleoperation task.

### Next concrete work

1. Obtain an actual compatible Stage-1 Video DiT checkpoint or explicitly choose
   the FastWAM/Wan2.2 approximation; do not mix Cosmos and Wan weights.
2. Install official SONIC sim2sim on suitable compute, record `g1_debug` on the
   same reference and state history, and compare token/raw action/target traces.
3. Install SIMPLE with its large Isaac Sim dependencies on suitable hardware,
   run `G1WholebodyTabletopGraspMP-v0` data generation, and inspect its saved
   fields and action convention before mapping an episode to `prepare-stage2`.
4. Build a FastWAM LeRobot loader for the 66-D latent baseline; first overfit a
   tiny real dataset and then validate on held-out scenes and objects.

## 2026-09-23 — SIMPLE execution trial

- Attempted the MuJoCo-only `G1WholebodyBendPickMP-v0` path from the official
  checkout using the existing Python 3.10 environment. Installed small missing
  import dependencies into `/tmp/simple_trial_deps`, outside both repositories.
- Import still stops at missing CuRobo, before task reset. CuRobo checkout is
  empty; `nvcc` is not on `PATH`, and Git LFS and SIMPLE task assets are absent
  here. The NVIDIA GPU and PyTorch CUDA runtime work. No
  episode was generated. See `docs/SIMPLE_TRIAL.md` for the exact trial.
- Audited SIMPLE's MP LeRobot recorder: its G1 whole-body data uses 43 joint
  targets with 14 Dex3 hand joints. Our stage-2 contract requires a 29-joint
  SONIC reference, root motion, and currently two scalar hands. A converter
  needs an explicit hand representation and root-reference source; direct
  ingestion would silently mislabel actions.
- Next: run one real SIMPLE episode on a prepared workstation, inspect its
  schema and simulator root state, then implement and validate the converter.

## 2026-09-23 — SIMPLE CUDA/CuRobo repair

- Confirmed the RTX 5070 and PyTorch CUDA runtime work. Installed `nvcc`
  12.8.93 and Git LFS under `robotics/SIMPLE/.cuda-toolkit`; built pinned
  CuRobo for sm_120 in `robotics/SIMPLE/.venv-lite`. All five CUDA extensions
  import. Materialized the three AMO Git LFS checkpoints.
- SIMPLE's G1 bend-and-pick environment now creates and resets in headless
  MuJoCo. It returned four 640×360 RGB views, 43 measured joints, a natural
  language instruction, and the root pose. The MP agent synthesized an initial
  locomotion phase and MuJoCo completed one planned control step.
- SIMPLE tracked source is clean; engineai and SONIC were untouched. At this
  checkpoint, a full episode and LeRobot recording remained outstanding. See
  `docs/SIMPLE_TRIAL.md` for verification commands and the action mapping.

## 2026-09-23 — first complete SIMPLE rollout

- Ran `simple/G1WholebodyBendPickMP-v0` through both MP phases with CuRobo and
  MuJoCo. The seed-0 episode terminated successfully after 213 steps, reward
  0.919. A second seed-0 run using the new
  `scripts/collect_simple_episode.py` recorded 214 successful frames with final
  reward 1.0 in `robotics/SIMPLE/data/bend_pick_seed0_raw.npz` (25.8 MB).
- Validated 20 ms timestamps, 43-channel measured joints and mixed MP targets,
  root quaternion norms, nonblank 640×360 head RGB, and exact instruction.
  Visually checked first/last frames. SIMPLE source and submodules are clean.
- The raw episode cannot yet feed `prepare-stage2`: it lacks desired root
  trajectory, has 14 Dex3 hand joints versus two scalar hand placeholders, and
  needs an image crop/resize. Next choose the hand representation and explicit
  root-reference derivation before writing a converter and SONIC round-trip.

## 2026-09-23 — repository layout for GitHub

- Moved the clean, official SONIC and MuJoCo Menagerie checkouts out of
  upstream `FastWAM/third_party/` and registered both as pinned Git submodules
  in this G1 demo repository's `third_party/`. Their source was not edited.
- The G1 demo repo is the only GitHub push target. Upstream FastWAM remains a
  sibling policy checkout; SIMPLE remains a sibling simulator/data checkout.
  Engineai directories were not touched.
- Clone this repository with `--recurse-submodules` or run
  `git submodule update --init` after cloning. SONIC ONNX checkpoints and SIMPLE
  rollouts are local downloaded/generated data, not GitHub commit contents.
