"""Round-trip and failure checks for the FastWAM-action → SONIC-reference bridge."""
import csv
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sonic_bridge import as_joint_and_root, export, g1_schema


def csv_values(path):
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    return np.asarray(rows[1:], dtype=float)


if __name__ == "__main__":
    actions = np.load(Path(__file__).resolve().parents[1] / "artifacts/scripted_actions.npy")
    q_isaac, _, _, _ = as_joint_and_root(actions)
    _, _, _, _, isaac_in_mujoco = g1_schema()
    with tempfile.TemporaryDirectory() as tmp:
        clip = export(actions, Path(tmp) / "clip", origin="scripted_ik_test")
        q_csv = csv_values(clip / "joint_pos.csv")
        assert np.allclose(q_csv, q_isaac, atol=1e-6)
        q_mj = np.empty_like(q_csv)
        q_mj[:, isaac_in_mujoco] = q_csv
        assert np.allclose(q_mj[:, 0], actions[:, 16], atol=1e-6)
        assert np.allclose(q_mj[:, 15], actions[:, 0], atol=1e-6)
        assert np.allclose(np.linalg.norm(csv_values(clip / "body_quat.csv"), axis=1), 1.0)
        broken = actions.copy()
        broken[0, 16] = 99.0
        try:
            as_joint_and_root(broken)
        except ValueError as exc:
            assert "left_hip_pitch_joint" in str(exc)
        else:
            raise AssertionError("URDF limit check failed")
    print(f"Verified {len(actions)} frames, SONIC joint permutation, root quaternions, and URDF limits")
