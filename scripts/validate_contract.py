import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "action_space.json"


def load_contract():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["joint_size_each"] == 1, "This G1 contract uses one scalar per joint"
    groups = cfg["joint_groups"]
    joint_names = [name for group in groups.values() for name in group]
    assert len(joint_names) == len(set(joint_names)) == 29
    offsets = {}
    offset = 0
    for block in cfg["order"]:
        if block.endswith("_end_effector"):
            size = cfg["end_effectors"][block.split("_")[0]]["size"]
        elif block == "root":
            size = cfg["root_size"]
        else:
            size = len(groups[block])
        assert isinstance(size, int) and size >= 0, block
        offsets[block] = [offset, offset + size]
        offset += size
    return cfg, offsets, offset


if __name__ == "__main__":
    _, offsets, size = load_contract()
    print(json.dumps({"action_dim": size, "slices": offsets}, indent=2))
