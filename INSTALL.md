# Quick installation: G1 standing simulation

This sets up the **working scripted SONIC + MuJoCo demo**. It runs on CPU and
does not require CUDA, SIMPLE, a FastWAM checkout, or the full SONIC source.
You need Git, Python with `venv` and `pip`, and internet access for Python
packages and the public SONIC v1.1 checkpoints (about 200 MB). The steps were
tested from a fresh clone with Python 3.14.6.

Run these commands in a terminal:

```bash
git clone https://github.com/ZhiboJin/G1-Demo-FastWAM.git
cd G1-Demo-FastWAM
git submodule update --init --depth 1 third_party/mujoco_menagerie

python3 -m venv --system-site-packages .venv-model
.venv-model/bin/python -m pip install -r requirements.txt

export HF_HOME=$PWD/.cache/huggingface
.venv-model/bin/python -m g1demo.cli download-sonic
.venv-model/bin/python -m g1demo.cli verify
.venv-model/bin/python -m g1demo.cli demo --motion standing --action-gain 0.5
```

`verify` should report **16/16 passed**. The demo should report **250 ticks**
and `fell False`; its plots and run report go into `artifacts/`. The simulation
gain of `0.5` is a workaround for the current MuJoCo plant, not a setting
validated on a physical G1.

If the checkpoint download fails with `Unknown scheme for proxy URL ...
socks://` and you already have an HTTP proxy configured, retry it with:

```bash
env -u ALL_PROXY -u all_proxy HF_HOME=$PWD/.cache/huggingface \
  .venv-model/bin/python -m g1demo.cli download-sonic
```

If the simulator cannot find `unitree_g1/scene.xml`, rerun the Menagerie
submodule command above. If `verify` cannot find an ONNX model, rerun
`download-sonic`. For the optional full SONIC source comparison, run
`git submodule update --init --depth 1 third_party/GR00T-WholeBodyControl`.

This install **does not yet run language-directed tasks**. A G1-trained FastWAM
checkpoint and normalization statistics are still needed; real camera, hand,
and robot command interfaces are not wired. See [START_HERE.md](START_HERE.md)
for the repository map and [docs/PROGRESS.md](docs/PROGRESS.md) for next steps.
