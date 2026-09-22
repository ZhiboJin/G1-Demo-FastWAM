# Running this repository on the Linux machine

Everything here runs from a fresh clone. Nothing large is stored in git: the
pretrained SONIC graphs (~200 MB) are downloaded, and the derived config is
regenerated locally so no Windows absolute paths leak into your Linux checkout.

## 1. Clone and set up

```bash
git clone https://github.com/ZhiboJin/G1-Demo-FastWAM.git
cd G1-Demo-FastWAM

python3 -m venv .venv-model
source .venv-model/bin/activate
pip install --upgrade pip
pip install -r requirements-model.txt
pip install -r requirements-demo.txt
pip install huggingface_hub
```

## 2. Fetch the pretrained SONIC v1.1 graphs

```bash
python scripts/download_sonic_checkpoints.py
```

This writes `model_encoder.onnx` (50 MB), `model_decoder.onnx` (150 MB) and
`observation_config.yaml` into `checkpoints/sonic_v1_1/`, verifying each SHA-256
against the build this repository was validated on. If a hash differs, upstream
republished something and you must re-run the verification before trusting the
interface.

Add `--planner-out /path/to/gear_sonic_deploy/planner/target_vel/V2` if you also
want the 774 MB kinematic planner for the C++ deployment.

## 3. Clone the two upstream repositories

These are needed by the bridge checks (G1 URDF, SONIC joint permutation) and by
any test that builds the real FastWAM model. Any layout works.

```bash
mkdir -p ~/pjt && cd ~/pjt
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
git -C GR00T-WholeBodyControl lfs install --local
git -C GR00T-WholeBodyControl lfs pull          # real G1 meshes instead of LFS pointers
git clone https://github.com/yuantianyuan01/FastWAM.git

export SONIC_REPO=$HOME/pjt/GR00T-WholeBodyControl
export FASTWAM_REPO=$HOME/pjt/FastWAM
```

Put those two exports in your shell profile. `scripts/project_paths.py` uses them
and falls back to sibling directories only when they are unset.

## 4. Regenerate the machine-specific config and verify

```bash
python scripts/build_g1_architecture.py    # writes configs/fastwam_g1_resolved.json
python scripts/validate_contract.py
python scripts/run_demo.py --task both --out artifacts/run.json
python scripts/verify_model.py             # 8 checks
python scripts/verify_decoder.py           # 16 checks
```

Run `run_demo.py` before `verify_model.py`: the model checks load
`artifacts/scripted_actions.npy`, which the demo writes. On a fresh clone that file
is committed, so this only matters after you regenerate it.

Expected: both demo reaches succeed at < 5 cm, all 8 model checks pass, all 16
decoder checks pass. `configs/fastwam_g1_resolved.json` is gitignored and will show
up as untracked — that is intended.

## 5. What still needs the Linux C++ stack

The Python side reaches the full interface
`predict -> encode_at -> decode_at`. Executing that action on a physics-simulated
robot needs the official GEAR-SONIC deployment, which is Linux/C++/TensorRT/DDS:

```bash
# In the GR00T-WholeBodyControl checkout, follow its own quickstart:
bash install_scripts/install_mujoco_sim.sh
python gear_sonic/scripts/run_sim_loop.py     # terminal 1 (host python env)
cd gear_sonic_deploy && bash deploy.sh sim    # terminal 2 (C++ binary)
```

### The one measurement that actually closes the loop

Our Python decoder and the C++ controller must agree. Run the same reference frame
with identical state history through both, and compare our
`SonicDecoder.decode(...)` output against the C++ action buffer element-wise.

This settles the question left open in `STATUS.md`: the C++ convention is
`motor_command.q_target = default_angles + action * scale`, and our decoded action
magnitudes (|action| ~ 9 rad on out-of-distribution scripted references) are **not
yet validated** against it. Until that comparison exists, do not treat the decoded
action as a verified motor command.

## 6. Notes and gotchas

- **Line endings.** `.gitattributes` forces LF. If you copied files from Windows by
  hand instead of cloning, convert them (`dos2unix`) or Python imports may still work
  but diffs will be noisy.
- **`checkpoints/`, `configs/fastwam_g1_resolved.json`, `artifacts/model_verification.json`
  and `artifacts/closedloop_verification.json` are gitignored.** The first two are
  re-fetched/regenerated; the last two are per-run reports that would otherwise create
  a diff on every verification run.
- **Python version.** This was validated on 3.13 with CPU PyTorch 2.14. The scripts
  avoid newer syntax and use `pathlib` throughout, so 3.10+ should work, but 3.10 has
  not been tested here. Upstream FastWAM itself specifies Python 3.10 + CUDA PyTorch
  2.7.1+cu128, so keep a **separate** environment for full FastWAM work — do not
  install FastWAM into `.venv-model`.
- **On the 48 GB RTX 4090.** Full FastWAM 5B inference (~10 GB of bf16 weights) is
  feasible, as is training the ActionDiT head and input/output projections with the
  backbone frozen. Full 5B post-training needs DeepSpeed ZeRO-2/3 plus gradient
  checkpointing; upstream ships `scripts/accelerate_configs/` and
  `scripts/ds_configs/` for exactly that.
- **The demo controller is scripted Jacobian IK, not FastWAM and not SONIC.** The
  fixed torso, disabled collisions, and capsule geometry mean it is not usable for
  sim-to-real metrics. See `STATUS.md` for the full list of boundaries.
