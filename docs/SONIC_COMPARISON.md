# SONIC comparison, 2026-09-23

## Completed comparisons

| Check | Result |
|---|---|
| Official C++ constants versus demo constants | Regenerated `g1_sonic_params.json` from the clean NVlabs checkout; byte-for-byte identical (`cmp` exit 0). Source header SHA-256: `b9332adf07c2c9b75c9b1e0756e57c7a1c2a890d8bb0aa53c3f1905fb739b791`. |
| ONNX provenance | Local SONIC v1.1 encoder/decoder SHA-256 hashes match NVIDIA's Hugging Face file metadata at repository revision `6733128a3d8a523b1418b06bca3cdf61c8b0987f`: encoder `fb97de22…365bae`, decoder `34bae857…091c67509`. `g1demo/download.py` downloads these files from `nvidia/GEAR-SONIC`. |
| Observation YAML | Downloaded `sonic_v1_1/observation_config.yaml` is byte-for-byte identical to the file in the pinned official GitHub submodule (`cmp` exit 0). |
| Decoder gravity history | Audited `GatherHisGravityDir` in official C++: it rotates gravity by each logged frame's quaternion. The Python loop now retains ten quaternions and packs ten distinct gravity directions. A standalone call with only one quaternion still repeats it as an explicit fallback. |
| Python input/output sanity | `python -m g1demo.cli verify`: 16/16 local checks, including G1 ordering, token shape, decoder shape, and C++ action-to-target expression. |
| Free-base MuJoCo standing reference, 250 ticks | At NVIDIA action gain 1.0: fall at tick 56, mean tracking error 0.3561 rad, 11.2% saturated joint-ticks. At simulation gain 0.5: 250 ticks without a fall, mean tracking error 0.0557 rad, 0% saturation. |

Reproduce the gain comparison from this repo:

```bash
.venv-model/bin/python -m g1demo.cli gain-sweep --motion standing \
  --mode free --source menagerie --model-gains sonic --ticks 250 \
  --action-gains 1.0 0.5
```

To preserve per-tick Python tensors for a future official comparison, run
`demo --motion standing --ticks 250 --action-gain 1.0 --trace
artifacts/python_standing_trace.npz`. The trace contains token, raw decoder
action, target, measured joints, base quaternion and hands. It still needs an
official trace collected from the **same input state history** before a valid
element-wise output comparison is possible.

The ONNX computations are NVIDIA's exact released graphs. The surrounding
Python observation construction is an adapter, not a copy of the C++ runtime.
For example, this demo derives reference joint velocities with `numpy.gradient`
from its joint trajectory, whereas the official C++ reads stored motion-file
velocities. Its heading-state update and reference playback also need the
same-input replay below. Therefore a successful local `verify` run is not proof
of byte-identical encoder inputs or tick-by-tick decoder actions in deployment.

## Official runtime comparison still outstanding

This machine does **not** currently have the official C++ comparison runtime:
the SONIC checkout has no built deployment binary, no `.venv_sim`, no deployed
ONNX files in its `gear_sonic_deploy/policy/sonic_v1_1/`, and no TensorRT
libraries visible to `ldconfig`. Therefore the table above does **not** show a
same-state, tick-by-tick comparison against `deploy.sh sim`. It supports the
joint constants and Python chain; it does not establish motor-command parity or
the cause of the sim2sim gap.

When the official runtime is installed, do the controlled comparison in this
order:

1. Export `standing` with `bridge-export --horizon 200`; feed exactly that clip
   to the official simulator.
2. Log the official `g1_debug` stream: token, 29 measured joint angles and
   velocities, base quaternion/angular velocity, and `last_action`/joint target
   at every 50 Hz tick.
3. Replay the **same measured state history and reference frames** through the
   Python encoder and decoder. Align by frame index; compare token and raw
   decoder action before comparing scaled targets. Separate reference packing,
   state-history packing, inference, and actuator differences.
4. Only after those agree, compare closed-loop free-base behavior at gain 1.0.

NVIDIA's documented setup uses `run_sim_loop.py` and `bash deploy.sh sim` in two
terminals: https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html

See `docs/sim2sim_gap.md` for the prior plant analysis. Its claim that the Python
chain is correct is a source-level conclusion; official runtime parity remains
open until the same-state replay above is performed.
