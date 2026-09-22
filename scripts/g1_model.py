"""G1 embodiment of upstream FastWAM; no replacement transformer or latent head."""
from pathlib import Path
import sys
import numpy as np

from validate_contract import load_contract
from project_paths import ROOT, fastwam_repo

FASTWAM = fastwam_repo()
MODEL_CONFIG = ROOT / "configs/fastwam_g1.yaml"
sys.path.insert(0, str(FASTWAM / "src"))


def action_names():
    cfg, _, _ = load_contract()
    names = []
    for block in cfg["order"]:
        if block in cfg["joint_groups"]:
            names.extend(cfg["joint_groups"][block])
        elif block == "root":
            names.extend(["root_roll_rad", "root_pitch_rad", "root_yaw_rate_rad_s"])
        else:
            side = block.split("_")[0]
            names.extend(f"{side}_hand_{i}" for i in range(cfg["end_effectors"][side]["size"]))
    return names


def model_kwargs(proprio_dim=None):
    """Load the dedicated G1 YAML and translate runtime keys to factory kwargs."""
    import yaml
    config = yaml.safe_load(MODEL_CONFIG.read_text())
    dim = len(action_names())
    video = dict(config["video_dit_config"])
    action = dict(config["action_dit_config"])
    if config.pop("_target_") != "fastwam.runtime.create_fastwam":
        raise ValueError("Expected the upstream FastWAM runtime target")
    for expert in (video, action):
        if expert["action_dim"] != dim:
            raise ValueError("Update G1 YAML action_dim to match action_space.json")
    kwargs = dict(config)
    for key in ("video_scheduler", "action_scheduler", "loss"):
        kwargs.pop(key)
    kwargs.update(
        model_id=config["model_id"], tokenizer_model_id=config["tokenizer_model_id"],
        tokenizer_max_len=config["tokenizer_max_len"],
        proprio_dim=config["proprio_dim"] if proprio_dim is None else proprio_dim,
        video_dit_config=video, action_dit_config=action,
        action_dit_pretrained_path=str(FASTWAM / config["action_dit_pretrained_path"]),
    )
    for branch in ("video", "action"):
        scheduler = config[branch + "_scheduler"]
        for key in ("train_shift", "infer_shift", "num_train_timesteps"):
            kwargs[branch + "_" + key] = scheduler[key]
    kwargs["loss_lambda_video"] = config["loss"].get("lambda_video", 1.)
    kwargs["loss_lambda_action"] = config["loss"].get("lambda_action", 1.)
    return kwargs


def build_fastwam(proprio_dim=None, device="cuda", torch_dtype=None, **overrides):
    """Instantiate real pretrained FastWAM. Requires official Wan/ActionDiT weights.

    The G1 input projection and output head are new, untrained parameters.
    Upstream ActionDiT's backbone-only loader deliberately excludes both.
    """
    import torch
    from fastwam.models.wan22.fastwam import FastWAM
    kwargs = model_kwargs(proprio_dim)
    kwargs.update(overrides)
    return FastWAM.from_wan22_pretrained(
        device=device, torch_dtype=torch_dtype or torch.bfloat16, **kwargs
    )


def build_action_expert(tiny=False):
    """Actual upstream ActionDiT, optionally reduced for CPU architecture tests."""
    from fastwam.models.wan22.action_dit import ActionDiT
    kwargs = model_kwargs()["action_dit_config"]
    if tiny:
        kwargs.update(hidden_dim=32, ffn_dim=64, text_dim=32, freq_dim=16,
                      num_heads=2, attn_head_dim=16, num_layers=2)
    return ActionDiT(**kwargs)


class ActionNormalizer:
    """Explicit physical-unit boundary. Statistics must match the exact named layout."""
    def __init__(self, names, mean, scale):
        if list(names) != action_names():
            raise ValueError("Normalizer action names/order differ from the G1 contract")
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        if (self.mean.shape != (len(names),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)):
            raise ValueError("Invalid per-dimension mean/scale")

    def decode(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != len(self.mean) or not np.isfinite(actions).all():
            raise ValueError("Expected finite [T, action_dim] normalized actions")
        return actions * self.scale + self.mean


class G1Policy:
    """FastWAM action-only inference -> physical references -> SONIC FSQ + hands.

    No robot transport or motor control is performed here. Call encode_at for each
    control tick with fresh robot orientation and sufficient remaining lookahead,
    then decode_at to execute the token through the pretrained SONIC decoder.

    The composite ``step`` covers the whole interface:

        predict -> denormalized [T,34] -> encode_at -> 64-D token + hands
                                        -> decode_at -> 29 joint actions

    :meth:`decode_at` returns the policy's raw action, not a motor command. The C++
    controller applies per-index scaling and adds ``default_angles``; that scaling
    convention is deliberately not applied here because it is not yet validated
    against the official deployment.
    """
    def __init__(self, fastwam, normalizer, sonic, decoder=None):
        if fastwam.action_expert.action_dim != len(action_names()):
            raise ValueError("FastWAM action head does not match the G1 contract")
        self.fastwam, self.normalizer, self.sonic = fastwam, normalizer, sonic
        self.decoder = decoder

    def predict(self, **inference_inputs):
        result = self.fastwam.infer_action(**inference_inputs)
        return self.normalizer.decode(result["action"].detach().cpu().numpy())

    def encode_at(self, actions, robot_quat_wxyz, initial_reference_yaw, frame=0):
        return self.sonic.encode(actions, robot_quat_wxyz, initial_reference_yaw, frame)

    def decode_at(self, token, joint_pos_history, joint_vel_history,
                  last_action_history, base_ang_vel_history, robot_quat_wxyz):
        """Execute one token through the pretrained decoder. Requires `decoder`."""
        if self.decoder is None:
            raise RuntimeError(
                "No SONIC decoder attached. Construct SonicDecoder("
                "ROOT/'checkpoints/sonic_v1_1/model_decoder.onnx') and pass it in."
            )
        return self.decoder.decode(
            token=token, joint_pos_history=joint_pos_history,
            joint_vel_history=joint_vel_history, last_action_history=last_action_history,
            base_ang_vel_history=base_ang_vel_history, base_quat_wxyz=robot_quat_wxyz,
        )

    def step(self, actions, robot_quat_wxyz, initial_reference_yaw, state, frame=0):
        """One full control tick: physical actions -> token -> joint action.

        ``state`` must provide the decoder's oldest->newest history as
        ``joint_pos`` [10,29], ``joint_vel`` [10,29], ``last_action`` [10,29] and
        ``base_ang_vel`` [10,3]. Real deployment fills these from StateLogger; the
        demo synthesizes them.
        """
        packet = self.encode_at(actions, robot_quat_wxyz, initial_reference_yaw, frame)
        action = self.decode_at(
            token=packet["token"],
            joint_pos_history=state["joint_pos"],
            joint_vel_history=state["joint_vel"],
            last_action_history=state["last_action"],
            base_ang_vel_history=state["base_ang_vel"],
            robot_quat_wxyz=robot_quat_wxyz,
        )
        return {"action": action, "token": packet["token"],
                "left_hand": packet["left_hand"], "right_hand": packet["right_hand"]}
