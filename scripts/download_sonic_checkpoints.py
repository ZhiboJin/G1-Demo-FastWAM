"""Fetch the pretrained SONIC v1.1 graphs into ``checkpoints/sonic_v1_1/``.

The ONNX graphs are ~200 MB, are not stored in git, and are public on Hugging
Face, so a fresh clone (for example on the Linux machine) runs this once.

This is deliberately independent of the upstream SONIC repository's own
``download_from_hf.py``, so the demo stays usable without checking out the whole
GEAR-SONIC tree just to obtain two files.

    pip install huggingface_hub
    python scripts/download_sonic_checkpoints.py

Use ``--decoder-only`` to fetch just the decoder, and ``--planner-out`` to also
place ``planner_sonic.onnx`` (774 MB) somewhere for the C++ deployment.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from project_paths import ROOT

REPO_ID = "nvidia/GEAR-SONIC"

# (path inside the HF repo, local filename)
FILES = [
    ("sonic_v1_1/model_encoder.onnx", "model_encoder.onnx"),
    ("sonic_v1_1/model_decoder.onnx", "model_decoder.onnx"),
    ("sonic_v1_1/observation_config.yaml", "observation_config.yaml"),
]
PLANNER = ("planner_sonic.onnx", "planner_sonic.onnx")

# Recorded when this demo's decoder/encoder integration was verified.
EXPECTED_SHA256 = {
    "model_encoder.onnx": "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae",
    "model_decoder.onnx": "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509",
}
DEST = ROOT / "checkpoints/sonic_v1_1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-only", action="store_true",
                        help="Fetch only model_decoder.onnx")
    parser.add_argument("--planner-out", type=Path, default=None,
                        help="Also download planner_sonic.onnx into this directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required:  pip install huggingface_hub", file=sys.stderr)
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    wanted = [FILES[1]] if args.decoder_only else FILES
    failures = []

    for remote, local_name in wanted:
        target = DEST / local_name
        if target.exists() and not args.force:
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
                print(f"          sha256 matches the verified build")
            else:
                print(f"          sha256 {actual}")
                print(f"          NOTE: differs from the build this demo was verified "
                      f"against ({expected}). Upstream may have republished; re-run "
                      f"scripts/verify_decoder.py before trusting the interface.")
                failures.append(local_name)

    if args.planner_out:
        args.planner_out.mkdir(parents=True, exist_ok=True)
        remote, local_name = PLANNER
        target = args.planner_out / local_name
        if not target.exists() or args.force:
            print(f"  [fetch] {remote} (774 MB, may take a while) ...")
            shutil.copyfile(hf_hub_download(repo_id=REPO_ID, filename=remote), target)
        print(f"  [ok]    {target}")

    print(f"\ncheckpoints in: {DEST}")
    print("Next:  python scripts/verify_decoder.py")
    if failures:
        print(f"\nWARNING: hashes differed for {failures}. Re-verify before use.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
