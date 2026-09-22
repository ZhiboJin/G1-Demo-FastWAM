"""Resolve production configuration; optionally save a tiny untrained test head."""
import argparse
import json
from g1_model import ROOT, action_names, model_kwargs, build_action_expert

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-tiny", action="store_true")
    args = parser.parse_args()
    config = {"architecture":"upstream FastWAM Wan2.2 + ActionDiT + MoT",
              "action_names":action_names(), "action_dim":len(action_names()),
              "action_horizon":50, "reference_fps":50,
              "proprioception":"29 absolute joint positions in MuJoCo order",
              "sonic_variant":"sonic_v1_1", "sonic_token_dim":64,
              "hand_routing":"bypass encoder; separate named outputs",
              "factory_kwargs":model_kwargs()}
    path = ROOT / "configs/fastwam_g1_resolved.json"
    path.write_text(json.dumps(config, indent=2))
    print(path)
    if args.save_tiny:
        import torch
        torch.manual_seed(7)
        model = build_action_expert(tiny=True)
        output = ROOT / "artifacts/tiny_action_dit_UNTRAINED.pt"
        torch.save({"state_dict":model.state_dict(), "action_names":action_names(),
                    "purpose":"CPU architecture test ONLY; random weights; not robot control"}, output)
        print(output)
