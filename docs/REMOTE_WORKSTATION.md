# GPU workstation status

This records the second-machine bring-up for the G1 demo. It is a development
environment, not a validated robot deployment.

## Locations and compute

| Purpose | Path |
|---|---|
| This repository | `/home/justin/projects/G1-Demo-FastWAM` |
| Existing upstream FastWAM | `/home/justin/FastWAM` |
| FastWAM path expected by `g1demo.paths` | `/home/justin/projects/FastWAM` (symlink) |
| SIMPLE source snapshot | `/home/justin/projects/SIMPLE` |

The workstation reports an RTX 4090 with 47.5 GiB usable VRAM, 62 GiB RAM,
and about 1.2 TB free disk. PyTorch 2.7.1+cu128 detects the GPU. `nvcc` 12.4
is installed. These checks establish hardware availability, not training
readiness or task success.

## Installed and verified

- The G1 repository is cloned from the user's GitHub at
  `10a3c648dc3592554cfa395bf99cbcd3b6db79da`.
- `.venv-model` is an isolated Python 3.10 virtual environment that inherits
  packages from the pre-existing `/home/justin/miniconda3/envs/fastwam`.
  ONNX Runtime 1.23.2 CPU provider was copied from the locally tested G1
  environment and its transfer checksum verified.
- Upstream FastWAM has the released LIBERO policy and ActionDiT initialization.
  There is no trained G1 FastWAM checkpoint or G1 normalization statistics.
- The pinned Menagerie G1 assets were copied with a matching SHA-256 after the
  remote GitHub submodule clone reset. Tests report `48 passed, 12 skipped,
  1 xfailed`. The submodule remains formally uninitialized, so a future
  successful `git submodule update --init` is still desirable.
- SIMPLE source is a compact snapshot at
  `6d10628794d9c7de4596b4f2afb2c054a637c2bc`, excluding large policy
  assets and submodules. It supports reading interfaces only; it cannot launch
  a SIMPLE task or the full SONIC controller as installed.

## Work still required

1. Copy or download both official SONIC v1.1 ONNX graphs into
   `checkpoints/sonic_v1_1/`, verify their SHA-256 hashes through
   `.venv-model/bin/python -m g1demo.cli download-sonic`, then run `verify`
   and the scripted standing demo. The graphs are roughly 200 MB together.
2. Complete a full SIMPLE checkout, initialize its pinned
   `third_party/GR00T-WholeBodyControl` fork, download matching model files,
   build its C++/TensorRT controller, and run a SIMPLE SONIC task. The source
   snapshot alone does not meet this requirement.
3. Prepare labeled G1/SIMPLE episodes, G1 action/proprioception statistics,
   and a suitable Video DiT initialization before Stage-2 training. The
   released LIBERO FastWAM weights do not provide a language-to-G1 demo.

The remote cannot currently reach Hugging Face directly (`Network is
unreachable`). GitHub transfers have been slow or reset, and Tailscale reports
a DERP relay instead of a direct connection. This is the present obstacle to
installing the large checkpoints; adding GPU memory alone does not solve it.

No EngineAI project or files were changed during this bring-up.
