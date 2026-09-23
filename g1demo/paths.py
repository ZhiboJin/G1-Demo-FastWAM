"""Path resolution for the G1 demo and the upstream checkouts it depends on.

Every path is either inside this repository or comes from an environment
variable, so the demo never depends on a machine-specific absolute path.

Environment variables (all optional):

``FASTWAM_REPO``
    Upstream FastWAM checkout (must contain ``src/fastwam/runtime.py``).
    Defaults to ``../FastWAM``, and also accepts the repository's own parent
    directory when this demo is checked out *inside* a FastWAM workspace.
``SONIC_REPO``
    NVlabs/GR00T-WholeBodyControl checkout. Defaults to this repository's
    ``third_party/GR00T-WholeBodyControl`` submodule. Only needed by tools
    reading NVIDIA's source or G1 URDF; ONNX inference uses the pinned local
    constants and downloaded checkpoints.

``HF_HOME`` may also need to be set to a writable directory when downloading the
pretrained SONIC graphs; ``download_sonic_checkpoints.py`` respects whatever
Hugging Face resolves.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = ROOT / "checkpoints/sonic_v1_1"
CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"


class MissingDependency(FileNotFoundError):
    """Raised when an optional upstream checkout cannot be located."""


def _resolve(env_name: str, candidates: list[Path], marker: str, purpose: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / marker).exists():
            raise MissingDependency(
                f"{env_name}={candidate} does not look like the expected checkout "
                f"(missing {marker})."
            )
        return candidate

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / marker).exists():
            return resolved

    checked = "\n  ".join(str(c) for c in candidates)
    raise MissingDependency(
        f"Could not locate {purpose}. Set {env_name} to its checkout path.\n"
        f"  Looked for {marker} under:\n  {checked}"
    )


def fastwam_repo() -> Path:
    """Upstream FastWAM checkout providing the ``fastwam`` package."""
    return _resolve(
        "FASTWAM_REPO",
        [
            # Sibling layout: robotics/FastWAM next to robotics/G1-Demo-FastWAM.
            ROOT.parent / "FastWAM",
            # Also accept an older nested checkout.
            ROOT.parent,
        ],
        "src/fastwam/runtime.py",
        "the upstream FastWAM checkout (needed only for real model inference)",
    )


def sonic_repo() -> Path:
    """NVlabs/GR00T-WholeBodyControl checkout (G1 URDF and C++ reference)."""
    return _resolve(
        "SONIC_REPO",
        [
            ROOT / "third_party/GR00T-WholeBodyControl",
            ROOT.parent / "GR00T-WholeBodyControl",
        ],
        "gear_sonic_deploy/deploy.sh",
        "the NVlabs GEAR-SONIC checkout (needed only by the URDF bridge tools)",
    )


def menagerie_repo() -> Path:
    """google-deepmind/mujoco_menagerie checkout (the real G1 with meshes).

    NVIDIA's G1 MJCF ships its STL files as Git-LFS pointers, so a normal clone
    cannot render it. Menagerie carries the same robot with the meshes committed.
    """
    return _resolve(
        "MENAGERIE_REPO",
        [
            ROOT / "third_party/mujoco_menagerie",
            ROOT.parent / "mujoco_menagerie",
        ],
        "unitree_g1/scene.xml",
        "the MuJoCo Menagerie checkout",
    )


def menagerie_g1_scene() -> Path:
    """The Menagerie Unitree G1 scene: real meshes, contacts and lighting."""
    return menagerie_repo() / "unitree_g1/scene.xml"


def sonic_params_path() -> Path:
    return CONFIGS / "g1_sonic_params.json"


def add_fastwam_to_path() -> Path:
    """Put the upstream FastWAM ``src`` directory on ``sys.path`` and return it."""
    import sys

    repo = fastwam_repo()
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return repo
