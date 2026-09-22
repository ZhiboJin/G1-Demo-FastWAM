"""Fetch the pretrained SONIC v1.1 ONNX graphs from Hugging Face.

The two graphs are ~200 MB, are not stored in git, and are public and ungated,
so a fresh clone downloads them once.

Hugging Face writes to ``$HF_HOME`` (``~/.cache/huggingface`` by default). On a
machine where that path is not writable, point ``HF_HOME`` at somewhere that is::

    HF_HOME=$PWD/.cache/huggingface python -m g1demo.cli download-sonic
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

REPO_ID = "nvidia/GEAR-SONIC"
DEST_NAME = "sonic_v1_1"

#: (path inside the Hugging Face repo, local filename)
FILES: tuple[tuple[str, str], ...] = (
    ("sonic_v1_1/model_encoder.onnx", "model_encoder.onnx"),
    ("sonic_v1_1/model_decoder.onnx", "model_decoder.onnx"),
    ("sonic_v1_1/observation_config.yaml", "observation_config.yaml"),
)

#: SHA-256 of the build this adapter was verified against. A mismatch is
#: reported rather than ignored: upstream may have republished the graphs, and
#: the tensor layouts this code depends on would need re-checking.
EXPECTED_SHA256 = {
    "model_encoder.onnx": "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae",
    "model_decoder.onnx": "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509",
}

#: The release tag these graphs come from.
VARIANT = "sonic_v1_1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sonic(dest: Path | None = None, force: bool = False,
                   planner_out: Path | None = None) -> int:
    """Download encoder + decoder + observation config. Returns a process code."""
    from .paths import CHECKPOINTS

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required:  pip install huggingface_hub")
        return 2

    dest = Path(dest) if dest else CHECKPOINTS
    dest.mkdir(parents=True, exist_ok=True)
    unexpected: list[str] = []

    for remote, local_name in FILES:
        target = dest / local_name
        if target.exists() and not force:
            print(f"  [keep]  {local_name} ({target.stat().st_size:,} bytes)")
        else:
            print(f"  [fetch] {remote} ...")
            cached = hf_hub_download(repo_id=REPO_ID, filename=remote)
            shutil.copyfile(cached, target)
            print(f"  [ok]    {local_name} ({target.stat().st_size:,} bytes)")

        expected = EXPECTED_SHA256.get(local_name)
        if expected:
            actual = sha256(target)
            if actual == expected:
                print("          sha256 matches the verified build")
            else:
                print(f"          sha256 {actual}")
                print(f"          expected {expected}")
                print("          NOTE: differs from the verified build. Upstream may have")
                print("                republished; re-run `verify` before trusting the")
                print("                tensor layouts.")
                unexpected.append(local_name)

    if planner_out is not None:
        planner_out = Path(planner_out)
        planner_out.mkdir(parents=True, exist_ok=True)
        remote, local_name = "planner_sonic.onnx", "planner_sonic.onnx"
        target = planner_out / local_name
        if not target.exists() or force:
            print(f"  [fetch] {remote} (774 MB, this takes a while) ...")
            shutil.copyfile(hf_hub_download(repo_id=REPO_ID, filename=remote), target)
        print(f"  [ok]    {target}")

    print(f"\nSONIC {VARIANT} graphs in: {dest}")
    if unexpected:
        print(f"WARNING: {unexpected} did not match the verified hashes.")
        return 1
    return 0
