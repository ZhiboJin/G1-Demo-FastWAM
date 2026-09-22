"""Closed-loop whole-body control: reference -> SONIC encoder -> decoder -> G1.

One control tick at 50 Hz does exactly this::

    1. read the G1 state from the simulator
    2. encode the reference chunk            -> 64-D whole-body token
    3. decode token + 10 frames of history   -> 29 raw actions (IsaacLab order)
    4. q_target = default + action * scale    (MuJoCo order)
    5. command the position servos and step physics
    6. push the new measurement into the history

Step 4 is NVIDIA's own convention, taken from ``policy_parameters.hpp`` and
``CreatePolicyCommand()``; see :func:`g1demo.sonic_params.q_target_from_action`.

Hand commands never enter steps 2-4. They are returned alongside each step so a
hardware layer can route them, which is what NVIDIA's deployment does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .contract import contract
from .sonic.decoder import FPS as SONIC_FPS
from .sonic.decoder import NUM_FRAMES, SonicDecoder
from .sonic.encoder import SonicEncoder, heading_yaw
from .sonic_params import (
    default_angles,
    isaaclab_to_mujoco,
    joint_lower,
    joint_names,
    joint_upper,
    measured_to_isaaclab_relative,
    q_target_from_action,
)
from .sim.g1_mujoco import CONTROL_HZ, G1Sim


class ActionSource(Protocol):
    """Anything that can propose a G1 action chunk."""

    @property
    def name(self) -> str:
        """Short label recorded in the run report."""

    def actions(self, sim: G1Sim, instruction: str | None = None) -> np.ndarray:
        """Return ``[T, action_dim]`` physical-unit references at 50 Hz."""


@dataclass
class StepRecord:
    """Everything observable about one control tick."""

    tick: int
    frame: int
    token: np.ndarray
    raw_action: np.ndarray          # 29, IsaacLab order, decoder output
    q_target: np.ndarray            # 29, MuJoCo order, radians
    q_measured: np.ndarray          # 29, MuJoCo order, radians
    saturated: np.ndarray           # 29 bool, target outside joint limits
    base_height: float
    base_quat_wxyz: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray


@dataclass
class History:
    """The 10-frame proprioceptive window the SONIC decoder expects.

    All joint arrays are in **IsaacLab** order. ``joint_pos`` is relative to the
    default standing pose, which is what the C++ state logger feeds the policy.
    """

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    last_action: np.ndarray
    base_ang_vel: np.ndarray

    @classmethod
    def from_state(cls, sim: G1Sim) -> "History":
        position = measured_to_isaaclab_relative(sim.joint_pos)
        velocity = isaaclab_to_mujoco(sim.joint_vel)  # order only; default is constant
        return cls(
            joint_pos=np.repeat(position[None, :], NUM_FRAMES, axis=0),
            joint_vel=np.repeat(velocity[None, :], NUM_FRAMES, axis=0),
            last_action=np.zeros((NUM_FRAMES, 29), dtype=np.float64),
            base_ang_vel=np.repeat(sim.base_ang_vel_body[None, :], NUM_FRAMES, axis=0),
        )

    def push(self, sim: G1Sim, raw_action: np.ndarray) -> None:
        """Append the newest measurement, dropping the oldest frame."""
        position = measured_to_isaaclab_relative(sim.joint_pos)
        velocity = isaaclab_to_mujoco(sim.joint_vel)
        self.joint_pos = np.vstack((self.joint_pos[1:], position))
        self.joint_vel = np.vstack((self.joint_vel[1:], velocity))
        self.last_action = np.vstack((self.last_action[1:], raw_action))
        self.base_ang_vel = np.vstack(
            (self.base_ang_vel[1:], sim.base_ang_vel_body)
        )


@dataclass
class RunSummary:
    """Aggregate outcome of a closed-loop run."""

    ticks: int
    seconds: float
    fell: bool
    min_base_height: float
    final_base_height: float
    mean_joint_tracking_error: float
    max_joint_tracking_error: float
    saturation_fraction: float
    max_abs_raw_action: float
    token_norm_range: tuple[float, float]
    action_gain: float = 1.0
    per_joint_rms_error: np.ndarray = field(repr=False, default=None)

    def as_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "seconds": round(self.seconds, 3),
            "action_gain": self.action_gain,
            "fell": self.fell,
            "min_base_height_m": round(self.min_base_height, 4),
            "final_base_height_m": round(self.final_base_height, 4),
            "mean_joint_tracking_error_rad": round(self.mean_joint_tracking_error, 4),
            "max_joint_tracking_error_rad": round(self.max_joint_tracking_error, 4),
            "saturation_fraction": round(self.saturation_fraction, 4),
            "max_abs_raw_action": round(self.max_abs_raw_action, 4),
            "token_norm_range": [round(value, 4) for value in self.token_norm_range],
        }


class WholeBodyController:
    """Runs the reference -> token -> action loop against a :class:`G1Sim`."""

    def __init__(self, encoder: SonicEncoder | None = None,
                 decoder: SonicDecoder | None = None,
                 control_hz: int = CONTROL_HZ,
                 action_gain: float = 1.0) -> None:
        if control_hz != SONIC_FPS:
            raise ValueError(
                f"SONIC v1.1 is a {SONIC_FPS} Hz policy; control_hz={control_hz} "
                f"would invalidate the reference timings and the 10-frame history."
            )
        if action_gain <= 0:
            raise ValueError(f"action_gain must be positive, got {action_gain}")
        self.encoder = encoder or SonicEncoder()
        self.decoder = decoder or SonicDecoder()
        self.control_hz = control_hz
        #: Scales the decoded action before it becomes a joint target. ``1.0`` is
        #: NVIDIA's convention and the value that should be used with their
        #: IsaacLab plant. Values below 1.0 reduce the effective loop gain, which
        #: a much stiffer MuJoCo servo otherwise turns into an unstable balance
        #: loop; see docs/sim2sim_gap.md.
        self.action_gain = float(action_gain)
        for side in ("left", "right"):
            if contract().hand_sizes[side] != 1:
                raise NotImplementedError(
                    "This demo routes one scalar per hand; update "
                    "configs/action_space.json and the routing layer for multi-DoF hands."
                )

    def run(self, sim: G1Sim, source: ActionSource, ticks: int = 100,
                 instruction: str | None = None, reference_yaw: float | None = None,
                 replan_every: int | None = None,
                 on_step=None, stop_on_fall: bool = True) -> tuple[RunSummary, list[StepRecord]]:
        """Run the closed loop and return a summary plus per-tick records."""
        if reference_yaw is None:
            reference_yaw = heading_yaw(sim.base_quat_wxyz)

        history = History.from_state(sim)
        records: list[StepRecord] = []

        chunk = np.asarray(source.actions(sim, instruction), dtype=np.float64)
        frame = 0
        # SONIC needs 46 samples of lookahead per encode; replan before running out.
        horizon_ticks = len(chunk) - 45
        if horizon_ticks <= 0:
            raise ValueError(
                f"Action chunk of {len(chunk)} frames is too short: SONIC mode 0 needs "
                f"at least 46 samples, and the demo replans before exhausting the chunk."
            )
        if replan_every is None:
            replan_every = horizon_ticks

        errors: list[np.ndarray] = []
        heights: list[float] = []
        saturated_total = 0
        tokens: list[float] = []

        for tick in range(ticks):
            if frame + 45 >= len(chunk):
                chunk = np.asarray(source.actions(sim, instruction), dtype=np.float64)
                frame = 0

            packet = self.encoder.encode(
                chunk, sim.base_quat_wxyz, reference_yaw, frame
            )
            raw_action = self.decoder.decode(
                token=packet["token"],
                joint_pos_history=history.joint_pos,
                joint_vel_history=history.joint_vel,
                last_action_history=history.last_action,
                base_ang_vel_history=history.base_ang_vel,
                base_quat_wxyz=sim.base_quat_wxyz,
            )
            q_target = q_target_from_action(raw_action * self.action_gain)
            saturated = (q_target < joint_lower()) | (q_target > joint_upper())
            # The real controller clamps at the actuator; do the same so the
            # servo is never asked for an unreachable target.
            q_command = np.clip(q_target, joint_lower(), joint_upper())

            sim.set_targets(q_command)
            sim.advance()

            measured = sim.joint_pos
            record = StepRecord(
                tick=tick,
                frame=frame,
                token=packet["token"],
                raw_action=raw_action,
                q_target=q_target,
                q_measured=measured,
                saturated=saturated,
                base_height=float(sim.base_position[2]),
                base_quat_wxyz=sim.base_quat_wxyz,
                left_hand=packet["left_hand"],
                right_hand=packet["right_hand"],
            )
            records.append(record)
            if on_step is not None:
                on_step(record, sim)

            errors.append(np.abs(measured - q_command))
            heights.append(record.base_height)
            saturated_total += int(saturated.sum())
            tokens.append(float(np.linalg.norm(packet["token"])))

            history.push(sim, raw_action)
            frame += 1

            if stop_on_fall and sim.is_fallen():
                break

        error_stack = np.stack(errors)
        summary = RunSummary(
            ticks=len(records),
            seconds=len(records) / self.control_hz,
            fell=bool(sim.is_fallen()),
            min_base_height=float(min(heights)),
            final_base_height=float(heights[-1]),
            mean_joint_tracking_error=float(error_stack.mean()),
            max_joint_tracking_error=float(error_stack.max()),
            saturation_fraction=saturated_total / max(1, len(records) * 29),
            max_abs_raw_action=float(
                max(float(np.abs(record.raw_action).max()) for record in records)
            ),
            token_norm_range=(float(min(tokens)), float(max(tokens))),
            action_gain=self.action_gain,
            per_joint_rms_error=np.sqrt((error_stack ** 2).mean(axis=0)),
        )
        return summary, records


def joint_error_table(summary: RunSummary) -> list[tuple[str, float]]:
    """Per-joint RMS tracking error, largest first."""
    if summary.per_joint_rms_error is None:
        return []
    order = np.argsort(-summary.per_joint_rms_error)
    names = joint_names()
    return [(names[index], float(summary.per_joint_rms_error[index])) for index in order]
