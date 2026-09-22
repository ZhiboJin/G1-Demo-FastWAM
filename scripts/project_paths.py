"""Resolve upstream checkouts on Windows or Linux.

Environment variables override the default sibling repository layout.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _repo(env_name, candidates, marker):
    paths = [Path(os.environ[env_name]).expanduser()] if os.environ.get(env_name) else [ROOT.parent / p for p in candidates]
    for path in paths:
        resolved = path.resolve()
        if (resolved / marker).exists():
            return resolved
    raise FileNotFoundError(
        f"Cannot find upstream repository ({marker}). Set {env_name} to its checkout path. "
        f"Checked: {', '.join(str(p) for p in paths)}"
    )


def fastwam_repo():
    return _repo("FASTWAM_REPO", ["FastWAM"], "src/fastwam/runtime.py")


def sonic_repo():
    return _repo("SONIC_REPO", ["GR00T-WholeBodyControl", "GR00T-WholeBodyControl-main/GR00T-WholeBodyControl-main"],
                 "gear_sonic_deploy/deploy.sh")
