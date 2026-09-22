"""SONIC v1.1 G1-mode observation packing and real ONNX encoder execution.

    Explicit joint references -> pretrained encoder INCLUDING FSQ -> 64 floats.
    Hand commands bypass the encoder. Not a differentiable training adapter.
"""
from pathlib import Path
import numpy as np
from sonic_bridge import as_joint_and_root

# Exact deployment order from policy/sonic_v1_1/observation_config.yaml.
LAYOUT = [
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_anchor_orientation_heading_10frame_step5", 60),
    ("motion_anchor_orientation_heading", 6),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_heading_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]
INPUT_DIM = sum(size for _, size in LAYOUT)  # 1751 in the released v1.1 graph.
FPS = 50
OFFSETS = np.arange(10) * 5


def reference_rotation(root, initial_yaw):
    """Rz(yaw) Ry(pitch) Rx(roll); first sample uses supplied world yaw.

    Yaw rate at frame t applies to [t,t+1); replan callers supply the new
    reference's initial world yaw instead of silently resetting heading to zero.
    """
    yaw = initial_yaw + np.r_[0., np.cumsum(root[:-1, 2]) / FPS]
    r, p = root[:, 0], root[:, 1]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(yaw), np.sin(yaw)
    return np.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                     sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                     -sp, cp*sr, cp*cr], axis=-1).reshape(-1, 3, 3)


def pack_observation(actions, robot_quat_wxyz, initial_reference_yaw, frame=0):
    q, root, hands, _ = as_joint_and_root(actions)
    if not isinstance(frame, (int, np.integer)) or frame < 0 or frame + OFFSETS[-1] >= len(q):
        raise ValueError("SONIC v1.1 needs frame..frame+45 at 50 Hz (46 samples); no implicit padding")
    quat = np.asarray(robot_quat_wxyz, dtype=float)
    if (quat.shape != (4,) or not np.isfinite(quat).all()
            or not np.isclose(np.linalg.norm(quat), 1., atol=1e-3)
            or not np.isfinite(initial_reference_yaw)):
        raise ValueError("Require finite reference yaw and unit robot quaternion in wxyz order")
    w, x, y, z = quat / np.linalg.norm(quat)
    heading = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = np.cos(heading), np.sin(heading)
    inv_heading = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    rotation = inv_heading @ reference_rotation(root, initial_reference_yaw)[frame + OFFSETS]
    # NVIDIA uses the FIRST TWO COLUMNS flattened ROW-WISE, not two concatenated columns.
    orientation = rotation[:, :, :2].reshape(-1)
    velocity = np.gradient(q, 1/FPS, axis=0)
    obs = np.zeros((1, INPUT_DIM), dtype=np.float32)
    obs[0, 4:294] = q[frame + OFFSETS].reshape(-1)
    obs[0, 294:584] = velocity[frame + OFFSETS].reshape(-1)
    obs[0, 584:644] = orientation
    # mode 0 plus padding and unused teleop/SMPL branches remain zero.
    return obs, {side: np.asarray(hands[frame][side], dtype=np.float32) for side in ("left", "right")}


class SonicEncoder:
    def __init__(self, encoder_path, observation_config, providers=None):
        import yaml
        import onnxruntime as ort
        cfg = yaml.safe_load(Path(observation_config).read_text())["encoder"]
        names = [item["name"] for item in cfg["encoder_observations"] if item.get("enabled", True)]
        if names != [name for name, _ in LAYOUT] or cfg["dimension"] != 64 or cfg.get("use_fp16", False):
            raise ValueError("This adapter supports the exact float32 SONIC v1.1 deployment layout only")
        self.session = ort.InferenceSession(str(encoder_path), providers=providers or ["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected a single SONIC encoder input/output")
        self.input, self.output = inputs[0], outputs[0]
        if self.input.shape != [1, INPUT_DIM] or self.input.type != "tensor(float)":
            raise ValueError(f"Encoder input mismatch: {self.input.shape}, expected [1,{INPUT_DIM}]")
        if self.output.shape != [1, 64] or self.output.type != "tensor(float)":
            raise ValueError(f"Encoder output mismatch: {self.output.shape}")

    def encode(self, actions, robot_quat_wxyz, initial_reference_yaw, frame=0):
        obs, hands = pack_observation(actions, robot_quat_wxyz, initial_reference_yaw, frame)
        token = self.session.run([self.output.name], {self.input.name: obs})[0]
        if token.shape != (1,64) or not np.isfinite(token).all():
            raise RuntimeError("SONIC returned invalid tokens")
        # Encoder graph already contains FSQ. Never quantize a second time.
        return {"token": token[0], "left_hand": hands["left"], "right_hand": hands["right"]}
