"""CPU architecture and pretrained SONIC integration tests; no task training."""
import hashlib
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from g1_model import action_names, model_kwargs, build_action_expert, ActionNormalizer
from sonic_encoder import SonicEncoder, pack_observation, INPUT_DIM, reference_rotation
from sonic_bridge import as_joint_and_root, g1_schema
from validate_contract import load_contract

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    def test_real_action_dit_backward(self):
        torch.manual_seed(7)
        model = build_action_expert(tiny=True)
        dim = len(action_names())
        actions = torch.randn(2, 50, dim, requires_grad=True)
        output = model(actions, torch.tensor([.2,.5]), torch.randn(2,4,32))
        self.assertEqual(tuple(output.shape), (2,50,dim))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(actions.grad).all())
        self.assertTrue(torch.all(model.head.weight.grad.abs().sum(dim=1) > 0))

    def test_production_meta_architecture(self):
        # Allocate the actual production architecture on meta, without 5B weights.
        from fastwam.models.wan22.wan_video_dit import WanVideoDiT
        from fastwam.models.wan22.action_dit import ActionDiT
        from fastwam.models.wan22.mot import MoT
        cfg = model_kwargs()
        with torch.device("meta"):
            video = WanVideoDiT(**cfg["video_dit_config"])
            action = ActionDiT(**cfg["action_dit_config"])
            mot = MoT(mixtures={"video":video, "action":action})
        self.assertEqual(action.head.out_features, len(action_names()))
        self.assertEqual(action.action_encoder.in_features, len(action_names()))
        self.assertEqual(len(video.blocks), len(action.blocks))
        self.assertEqual(video.num_heads, action.num_heads)
        self.assertGreater(sum(p.numel() for p in mot.parameters()), 1_000_000_000)

    def test_normalization_names(self):
        n = len(action_names())
        normalizer = ActionNormalizer(action_names(), np.ones(n), np.full(n,2))
        np.testing.assert_equal(normalizer.decode(np.ones((2,n))), np.full((2,n),3))
        with self.assertRaises(ValueError):
            ActionNormalizer(action_names()[::-1], np.zeros(n), np.ones(n))

    def test_configurable_hand_dimensions(self):
        import yaml
        from g1_model import MODEL_CONFIG
        cfg, _, _ = load_contract()
        cfg["end_effectors"]["left"]["size"] = 7
        cfg["end_effectors"]["right"]["size"] = 7
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "contract.json"
            path.write_text(json.dumps(cfg))
            model_cfg = yaml.safe_load(MODEL_CONFIG.read_text())
            for branch in ("video_dit_config", "action_dit_config"):
                model_cfg[branch]["action_dim"] = 46
            model_path = Path(folder) / "model.yaml"
            model_path.write_text(yaml.safe_dump(model_cfg))
            with patch("validate_contract.CONFIG", path), patch("g1_model.MODEL_CONFIG", model_path):
                self.assertEqual(len(action_names()), 46)
                self.assertEqual(build_action_expert(tiny=True).head.out_features, 46)
                a = np.zeros((50,46))
                _, slices, _ = load_contract()
                for side in ("left", "right"):
                    lo, hi = slices[side+"_end_effector"]
                    a[:,lo:hi] = np.arange(7)
                obs, hands = pack_observation(a, [1,0,0,0], 0.)
                np.testing.assert_equal(hands["left"], np.arange(7))
                self.assertEqual(obs.shape, (1,1751))


class SonicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions = np.load(ROOT / "artifacts/scripted_actions.npy")
        cls.encoder = SonicEncoder(ROOT / "checkpoints/sonic_v1_1/model_encoder.onnx",
                                   ROOT / "checkpoints/sonic_v1_1/observation_config.yaml")

    def test_named_joint_mapping_and_velocity(self):
        actions = self.actions.copy()
        cfg, slices, _ = load_contract()
        index = slices["left_arm"][0]
        actions[:,index] = .001 * np.arange(len(actions))
        obs, _ = pack_observation(actions, [1,0,0,0], 0.)
        _, _, mj_names, _, permutation = g1_schema()
        isaac_names = [mj_names[i] for i in permutation]
        joint = isaac_names.index(cfg["joint_groups"]["left_arm"][0])
        np.testing.assert_allclose(obs[0,4:294].reshape(10,29)[:,joint], .005*np.arange(10), atol=1e-7)
        np.testing.assert_allclose(obs[0,294:584].reshape(10,29)[:,joint], .05, atol=1e-7)
        self.assertEqual(obs.shape, (1,1751))

    def test_heading_rotation_and_yaw_integration(self):
        a = self.actions.copy()
        _, slices, _ = load_contract()
        root = slices["root"][0]
        a[:,root:root+3] = [0,0,1]
        obs, _ = pack_observation(a, [np.cos(.3),0,0,np.sin(.3)], .6)
        rotation = obs[0,584:644].reshape(10,6)
        np.testing.assert_allclose(rotation[0], [1,0,0,1,0,0], atol=1e-7)
        angle = .1
        np.testing.assert_allclose(rotation[1], [np.cos(angle),-np.sin(angle),np.sin(angle),np.cos(angle),0,0], atol=1e-7)

    def test_hands_bypass_actual_encoder(self):
        a = self.actions.copy()
        _, slices, _ = load_contract()
        first = self.encoder.encode(a, [1,0,0,0], 0.)
        for block in ("left_end_effector", "right_end_effector"):
            lo, hi = slices[block]
            a[:,lo:hi] = .73
        second = self.encoder.encode(a, [1,0,0,0], 0.)
        np.testing.assert_array_equal(first["token"], second["token"])
        np.testing.assert_allclose(second["left_hand"], .73)
        self.assertEqual(first["token"].shape, (64,))
        self.assertTrue(np.isfinite(first["token"]).all())

    def test_invalid_references_rejected(self):
        with self.assertRaises(ValueError):
            pack_observation(self.actions[:45], [1,0,0,0], 0.)
        with self.assertRaises(ValueError):
            pack_observation(self.actions, [0,0,0,0], 0.)
        a = self.actions.copy()
        a[0,0] = 100
        with self.assertRaises(ValueError):
            pack_observation(a, [1,0,0,0], 0.)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    checkpoint = ROOT / "checkpoints/sonic_v1_1/model_encoder.onnx"
    report = {"tests_run":result.testsRun, "failures":len(result.failures),
              "errors":len(result.errors), "passed":result.wasSuccessful(),
              "action_names":action_names(), "action_dim":len(action_names()),
              "sonic_input_dim":INPUT_DIM, "token_dim":64,
              "sonic_encoder_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "source":"https://huggingface.co/nvidia/GEAR-SONIC/tree/main/sonic_v1_1",
              "limits":"Random tiny ActionDiT + production meta allocation + real pretrained SONIC encoder. No trained G1 FastWAM or closed-loop controller test."}
    (ROOT / "artifacts/model_verification.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
