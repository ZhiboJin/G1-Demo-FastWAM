"""The G1 action contract: the single source of truth for what FastWAM predicts.

The contract is ``configs/action_space.json``. It defines a flat action vector
made of named blocks and, for each block, where it goes:

* ``encoder`` blocks (the 29 body joints) are turned into a SONIC motion
  reference and encoded into the shared 64-D whole-body token.
* ``bypass`` blocks (the hands) skip SONIC entirely, exactly as its deployment
  does: SONIC's graphs have no hand inputs or outputs.

Nothing else in this package re-derives the layout. If you change the hand
dimensions in ``action_space.json``, the slices and packing helpers follow.
Update the model's two ``action_dim`` values as well; ``FastWAMPolicy.load``
checks that its configuration matches this contract.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .paths import CONFIGS

CONTRACT_PATH = CONFIGS / "action_space.json"

#: Blocks routed into the SONIC encoder.
ROUTE_ENCODER = "encoder"
#: Blocks that bypass SONIC and are handled by downstream hardware code.
ROUTE_BYPASS = "bypass"


@dataclass(frozen=True)
class Block:
    """A contiguous slice of the flat action vector."""

    name: str
    start: int
    stop: int
    route: str

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class Contract:
    """The resolved G1 action contract."""

    blocks: tuple[Block, ...]
    joint_groups: Mapping[str, tuple[str, ...]]
    joint_names: tuple[str, ...]
    hand_sizes: Mapping[str, int]
    _by_name: Mapping[str, Block]

    # -- dimensions -------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return self.blocks[-1].stop

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def root_size(self) -> int:
        return self["root"].size

    def __getitem__(self, name: str) -> Block:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"Unknown action block {name!r}. Known blocks: "
                f"{', '.join(b.name for b in self.blocks)}"
            ) from None

    # -- packing ----------------------------------------------------------
    def pack(self, blocks: Mapping[str, Iterable[float]]) -> np.ndarray:
        """Build a flat action vector from named blocks.

        Missing blocks are zero-filled, which is what the demo wants for the
        root and hand placeholders. Present blocks must be finite and the right
        length; silently truncating a malformed block would corrupt the encoder
        reference, so it is rejected instead.
        """
        action = np.zeros(self.action_dim, dtype=np.float32)
        for name, values in blocks.items():
            block = self[name]
            array = np.asarray(values, dtype=np.float32).reshape(-1)
            if array.shape != (block.size,):
                raise ValueError(
                    f"Block {name!r} must have {block.size} values, got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"Block {name!r} contains NaN or infinity")
            action[block.slice] = array
        return action

    def unpack(self, action: Iterable[float]) -> dict[str, np.ndarray]:
        """Split a flat action vector into named blocks."""
        array = np.asarray(action, dtype=np.float32)
        if array.shape != (self.action_dim,):
            raise ValueError(
                f"Action must have shape ({self.action_dim},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Action contains NaN or infinity")
        return {block.name: array[block.slice].copy() for block in self.blocks}

    # -- named joints -----------------------------------------------------
    def named_joints(self, action: Iterable[float]) -> dict[str, float]:
        """Map a flat action vector to ``{joint_name: value}`` (radians)."""
        blocks = self.unpack(action)
        return {
            joint: float(value)
            for group, names in self.joint_groups.items()
            for joint, value in zip(names, blocks[group], strict=True)
        }

    def action_names(self) -> list[str]:
        """Human-readable name for every scalar in the action vector."""
        names: list[str] = []
        for block in self.blocks:
            if block.name in self.joint_groups:
                names.extend(self.joint_groups[block.name])
            elif block.name == "root":
                names.extend(["root_roll_rad", "root_pitch_rad", "root_yaw_rate_rad_s"])
            else:  # hand blocks
                side = block.name.split("_")[0]
                size = self.hand_sizes[side]
                names.extend(
                    f"{side}_hand_{i}" if size > 1 else f"{side}_hand"
                    for i in range(size)
                )
        return names

    def flat_indices_for(self, joint_order: Iterable[str]) -> np.ndarray:
        """Flat action indices of the joints named in ``joint_order``.

        Lets a caller reorder the 29 joints into any convention (MuJoCo, IsaacLab,
        …) without re-deriving the block layout, and without a Python loop per
        frame.
        """
        position: dict[str, int] = {}
        for block in self.blocks:
            if block.name in self.joint_groups:
                for offset, joint in enumerate(self.joint_groups[block.name]):
                    position[joint] = block.start + offset

        order = list(joint_order)
        missing = [name for name in order if name not in position]
        if missing:
            raise KeyError(f"Joints not present in the contract: {missing}")
        if len(set(order)) != len(order):
            raise ValueError("joint_order contains duplicates")
        return np.asarray([position[name] for name in order], dtype=np.int64)


def _load(path: Path) -> Contract:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("joint_size_each") != 1:
        raise ValueError("This contract assumes one scalar per G1 joint")

    groups = {name: tuple(joints) for name, joints in raw["joint_groups"].items()}
    joint_names = tuple(name for joints in groups.values() for name in joints)
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError(
            f"Expected 29 distinct G1 joints, got {len(joint_names)} "
            f"({len(set(joint_names))} distinct)"
        )

    hand_sizes = {side: int(cfg["size"]) for side, cfg in raw["end_effectors"].items()}
    if set(hand_sizes) != {"left", "right"}:
        raise ValueError("Contract must define left and right end effectors")

    blocks: list[Block] = []
    cursor = 0
    for name in raw["order"]:
        if name in groups:
            size = len(groups[name])
            route = ROUTE_ENCODER
        elif name == "root":
            size = int(raw["root_size"])
            route = ROUTE_ENCODER
        else:
            side = name.split("_")[0]
            if not name.endswith("_end_effector") or side not in hand_sizes:
                raise ValueError(f"Unrecognised contract block {name!r}")
            size = hand_sizes[side]
            route = ROUTE_BYPASS
        if size < 0:
            raise ValueError(f"Block {name!r} has negative size {size}")
        blocks.append(Block(name=name, start=cursor, stop=cursor + size, route=route))
        cursor += size

    if cursor != 29 + sum(hand_sizes.values()) + int(raw["root_size"]):
        raise ValueError("Contract block sizes do not add up to the action dimension")

    return Contract(
        blocks=tuple(blocks),
        joint_groups=groups,
        joint_names=joint_names,
        hand_sizes=hand_sizes,
        _by_name={block.name: block for block in blocks},
    )


@lru_cache(maxsize=1)
def contract(path: Path | None = None) -> Contract:
    """Return the resolved action contract (cached; parsed once per process)."""
    return _load(path or CONTRACT_PATH)
