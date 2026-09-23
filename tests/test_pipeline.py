"""End-to-end checks for the contract, the SONIC chain, the plant and the loop.

Run with::

    python -m unittest discover -s tests -v

The SONIC tests need the pretrained graphs (``python -m g1demo.cli
download-sonic``); they skip with a clear message when those are absent.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g1demo import sonic_params as sp  # noqa: E402
from g1demo.contract import contract  # noqa: E402
from g1demo.paths import CHECKPOINTS  # noqa: E402

HAVE_SONIC = (CHECKPOINTS / "model_encoder.onnx").exists() and (
    CHECKPOINTS / "model_decoder.onnx"
).exists()
SONIC_SKIP = "SONIC v1.1 graphs not downloaded (run: python -m g1demo.cli download-sonic)"


def default_chunk(frames: int = 60) -> np.ndarray:
    """A static reference holding the default standing pose."""
    c = contract()
    chunk = np.zeros((frames, c.action_dim), dtype=np.float64)
    chunk[:, c.flat_indices_for(sp.joint_names())] = sp.default_angles()
    return chunk


class TestContract(unittest.TestCase):
    def test_widths(self):
        c = contract()
        self.assertEqual(c.num_joints, 29)
        self.assertEqual(c.action_dim, 29 + sum(c.hand_sizes.values()) + c.root_size)

    def test_blocks_are_contiguous_and_ordered(self):
        cursor = 0
        for block in contract().blocks:
            self.assertEqual(block.start, cursor)
            self.assertEqual(block.stop, block.start + block.size)
            cursor = block.stop
        self.assertEqual(cursor, contract().action_dim)

    def test_pack_unpack_round_trip(self):
        c = contract()
        action = default_chunk(1)[0]
        blocks = c.unpack(action)
        # pack() narrows to float32, so compare at float32 precision.
        np.testing.assert_allclose(c.pack(blocks), action, rtol=0, atol=1e-6)

    def test_pack_rejects_wrong_length_and_nonfinite(self):
        c = contract()
        with self.assertRaises(ValueError):
            c.pack({"left_arm": np.zeros(3)})
        with self.assertRaises(ValueError):
            c.pack({"left_arm": np.full(7, np.nan)})

    def test_unpack_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            contract().unpack(np.zeros(3))

    def test_named_joints_covers_every_joint(self):
        c = contract()
        named = c.named_joints(default_chunk(1)[0])
        self.assertEqual(set(named), set(sp.joint_names()))
        self.assertEqual(len(named), 29)

    def test_action_names_match_width(self):
        c = contract()
        self.assertEqual(len(c.action_names()), c.action_dim)

    def test_flat_indices_map_contract_joint_to_its_slot(self):
        """The contract groups joints differently from MuJoCo, so this must reorder."""
        c = contract()
        names = sp.joint_names()
        flat = c.flat_indices_for(names)
        self.assertEqual(len(flat), 29)
        for probe_index, probe_name in (
            (0, "left_hip_pitch_joint"),
            (3, "left_knee_joint"),
            (22, "right_shoulder_pitch_joint"),
            (14, "waist_pitch_joint"),
        ):
            self.assertEqual(names[probe_index], probe_name)
            action = np.zeros(c.action_dim)
            action[flat[probe_index]] = 1.234
            named = c.named_joints(action)
            self.assertAlmostEqual(named[probe_name], 1.234)
            moved = [joint for joint, value in named.items() if abs(value) > 1e-12]
            self.assertEqual(moved, [probe_name], "exactly one joint should move")

    def test_hands_are_bypass_and_joints_are_encoder_routed(self):
        c = contract()
        for side in ("left", "right"):
            self.assertEqual(c[f"{side}_end_effector"].route, "bypass")
        for group in c.joint_groups:
            self.assertEqual(c[group].route, "encoder")


class TestSonicParams(unittest.TestCase):
    def test_permutations_are_inverses(self):
        forward = sp.params()["isaaclab_to_mujoco"]
        inverse = sp.params()["mujoco_to_isaaclab"]
        np.testing.assert_array_equal(inverse[forward], np.arange(29))
        np.testing.assert_array_equal(forward[inverse], np.arange(29))

    def test_limits_bracket_the_default_pose(self):
        self.assertTrue((sp.default_angles() > sp.joint_lower()).all())
        self.assertTrue((sp.default_angles() < sp.joint_upper()).all())

    def test_gains_and_scales_match_the_header_formula(self):
        # action_scale = 0.25 * effort_limit / stiffness, and kp is the stiffness,
        # except for the ankles and waist roll/pitch where kp is doubled.
        ratio = 0.25 * sp.effort_limit() / sp.kp()
        doubled = np.isclose(sp.kp(), 2.0 * (0.25 * sp.effort_limit() / sp.action_scale()))
        np.testing.assert_allclose(
            sp.action_scale()[~doubled], ratio[~doubled], rtol=1e-9
        )
        self.assertEqual(int(doubled.sum()), 6)  # 2 ankles + waist roll/pitch, per side

    def test_contract_joint_names_match_sonic(self):
        sp.validate_against_contract(contract().joint_names)

    def test_q_target_matches_the_cpp_expression(self):
        """q_target[i] = default[i] + action[isaaclab_to_mujoco[i]] * scale[i]."""
        action = np.linspace(-2.0, 2.0, 29)
        permutation = sp.params()["isaaclab_to_mujoco"]
        manual = sp.default_angles() + action[permutation] * sp.action_scale()
        np.testing.assert_allclose(sp.q_target_from_action(action), manual)

    def test_q_target_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            sp.q_target_from_action(np.zeros(7))
        with self.assertRaises(ValueError):
            sp.q_target_from_action(np.full(29, np.nan))

    def test_measured_relative_history_subtracts_the_default_pose(self):
        q = sp.default_angles()
        relative = sp.measured_to_isaaclab_relative(q)
        np.testing.assert_allclose(relative, 0.0, atol=1e-12)
        moved = q.copy()
        moved[0] += 0.5
        # MuJoCo joint 0 is the left hip pitch; confirm it lands in IsaacLab order.
        isaac_index = int(np.flatnonzero(sp.params()["mujoco_to_isaaclab"] == 0)[0])
        self.assertAlmostEqual(sp.measured_to_isaaclab_relative(moved)[isaac_index], 0.5)


class TestEncoderMath(unittest.TestCase):
    """Encoder packing rules that do not need the ONNX graph."""

    def test_layout_sums_to_declared_width(self):
        from g1demo.sonic.encoder import INPUT_DIM, LAYOUT

        self.assertEqual(sum(width for _, width in LAYOUT), INPUT_DIM)
        self.assertEqual(INPUT_DIM, 1751)

    def test_mode_zero_is_not_one_hot(self):
        from g1demo.sonic.encoder import pack_observation

        obs, _ = pack_observation(default_chunk(), [1, 0, 0, 0], 0.0)
        np.testing.assert_array_equal(obs[0, :4], np.zeros(4))

    def test_orientation_is_identity_for_level_robot_at_matching_yaw(self):
        from g1demo.sonic.encoder import pack_observation

        obs, _ = pack_observation(default_chunk(), [1, 0, 0, 0], 0.0)
        np.testing.assert_allclose(obs[0, 584:590], [1, 0, 0, 1, 0, 0], atol=1e-9)

    def test_robot_heading_is_removed_from_the_reference(self):
        """A 90 deg robot yaw and a 90 deg reference yaw must cancel out."""
        from g1demo.sonic.encoder import pack_observation

        half = np.pi / 4
        quat = [np.cos(half), 0.0, 0.0, np.sin(half)]
        obs, _ = pack_observation(default_chunk(), quat, np.pi / 2)
        np.testing.assert_allclose(obs[0, 584:590], [1, 0, 0, 1, 0, 0], atol=1e-9)

    def test_yaw_rate_integrates_across_frames(self):
        from g1demo.sonic.encoder import FPS, OFFSETS, pack_observation

        c = contract()
        chunk = default_chunk()
        chunk[:, c["root"].slice] = [0.0, 0.0, 1.0]  # 1 rad/s yaw
        obs, _ = pack_observation(chunk, [1, 0, 0, 0], 0.0)
        # Sample k of the orientation block sits at OFFSETS[k] control ticks,
        # i.e. OFFSETS[k]/FPS seconds of integrated yaw.
        for k in (1, 3, 7):
            angle = OFFSETS[k] / FPS
            np.testing.assert_allclose(
                obs[0, 584 + k * 6 : 584 + k * 6 + 6],
                [np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle), 0, 0],
                atol=1e-6,
                err_msg=f"sample {k} (frame {OFFSETS[k]})",
            )

    def test_short_reference_is_rejected(self):
        from g1demo.sonic.encoder import pack_observation

        with self.assertRaises(ValueError):
            pack_observation(default_chunk(45), [1, 0, 0, 0], 0.0)

    def test_out_of_limit_reference_is_rejected(self):
        from g1demo.sonic.encoder import pack_observation

        chunk = default_chunk()
        chunk[0, 0] = 100.0
        with self.assertRaises(ValueError):
            pack_observation(chunk, [1, 0, 0, 0], 0.0)

    def test_unit_quaternion_enforced(self):
        from g1demo.sonic.encoder import pack_observation

        with self.assertRaises(ValueError):
            pack_observation(default_chunk(), [0, 0, 0, 0], 0.0)


class TestDecoderMath(unittest.TestCase):
    def test_layout_sums_to_declared_width(self):
        from g1demo.sonic.decoder import INPUT_DIM, LAYOUT

        self.assertEqual(sum(f * w for _, f, w in LAYOUT), INPUT_DIM)
        self.assertEqual(INPUT_DIM, 994)

    def test_gravity_is_downward_in_body_frame_when_level(self):
        from g1demo.sonic.decoder import gravity_direction

        np.testing.assert_allclose(gravity_direction([1, 0, 0, 0]), [0, 0, -1], atol=1e-9)

    def test_gravity_rotates_with_the_body(self):
        """Pitching forward 90 deg puts world-down along the body's +x axis."""
        from g1demo.sonic.decoder import gravity_direction

        half = np.pi / 4
        quat = [np.cos(half), 0.0, np.sin(half), 0.0]  # 90 deg about +y
        np.testing.assert_allclose(gravity_direction(quat), [1, 0, 0], atol=1e-9)

    def test_pack_rejects_wrong_history_shape(self):
        from g1demo.sonic.decoder import pack_observation

        with self.assertRaises(ValueError):
            pack_observation(np.zeros(64), np.zeros((5, 29)), np.zeros((10, 29)),
                             np.zeros((10, 29)), np.zeros((10, 3)), [1, 0, 0, 0])


@unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
class TestSonicGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from g1demo.sonic.decoder import SonicDecoder
        from g1demo.sonic.encoder import SonicEncoder

        cls.encoder = SonicEncoder()
        cls.decoder = SonicDecoder()
        cls.chunk = default_chunk()

    def test_token_shape_and_quantization(self):
        packet = self.encoder.encode(self.chunk, [1, 0, 0, 0], 0.0)
        self.assertEqual(packet["token"].shape, (64,))
        # FSQ with 32 levels: every coordinate is an integer multiple of 1/16.
        np.testing.assert_allclose(
            packet["token"] * 16, np.round(packet["token"] * 16), atol=1e-5
        )

    def test_hands_bypass_the_encoder(self):
        first = self.encoder.encode(self.chunk, [1, 0, 0, 0], 0.0)
        altered = self.chunk.copy()
        altered[:, 7] = 0.73
        altered[:, 15] = -0.42
        second = self.encoder.encode(altered, [1, 0, 0, 0], 0.0)
        np.testing.assert_array_equal(first["token"], second["token"])
        np.testing.assert_allclose(second["left_hand"], 0.73)
        np.testing.assert_allclose(second["right_hand"], -0.42)

    def test_decoder_produces_bounded_targets_for_an_in_distribution_reference(self):
        from g1demo.sonic_params import q_target_from_action

        packet = self.encoder.encode(self.chunk, [1, 0, 0, 0], 0.0)
        history = sp.measured_to_isaaclab_relative(np.tile(sp.default_angles(), (10, 1)))
        raw = self.decoder.decode(
            packet["token"], history, np.zeros((10, 29)), np.zeros((10, 29)),
            np.zeros((10, 3)), [1, 0, 0, 0],
        )
        self.assertEqual(raw.shape, (29,))
        target = q_target_from_action(raw)
        within = (target >= sp.joint_lower()) & (target <= sp.joint_upper())
        self.assertTrue(within.all(), f"{(~within).sum()} joints outside limits")
        self.assertLess(np.abs(target - sp.default_angles()).max(), 0.5)

    def test_lookahead_is_enforced_at_later_frames(self):
        with self.assertRaises(ValueError):
            self.encoder.encode(self.chunk, [1, 0, 0, 0], 0.0, frame=20)


class TestSimulator(unittest.TestCase):
    def setUp(self):
        # A fresh plant per test: these mutate physics state.
        from g1demo.sim import G1Sim

        self.sim = G1Sim()

    def test_model_dimensions(self):
        self.assertEqual(self.sim.model.nu, 29)
        self.assertEqual(self.sim.model.njnt, 30)  # 29 joints + free root

    def test_is_the_real_g1(self):
        """The model must be the actual G1, not a stand-in."""
        from g1demo import sonic_params as sp

        # Unitree quotes about 35 kg for the 29-DoF G1; Menagerie's rev 1.0 is
        # 33.34 kg because it carries no dexterous hands.
        self.assertGreater(self.sim.model.body_mass.sum(), 30.0)
        self.assertLess(self.sim.model.body_mass.sum(), 40.0)
        names = [
            mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(1, self.sim.model.njnt)
        ]
        # Same joints, in the same order, as SONIC's policy parameters.
        self.assertEqual(names, list(sp.joint_names()))

    def test_starts_at_the_default_pose_and_in_contact(self):
        np.testing.assert_allclose(self.sim.joint_pos, sp.default_angles(), atol=0.06)
        self.assertGreater(self.sim.base_position[2], 0.70)
        # Settling under gravity leaves a small residual pitch (~0.01 rad).
        np.testing.assert_allclose(self.sim.base_quat_wxyz, [1, 0, 0, 0], atol=1e-2)

    def test_control_tick_is_twenty_milliseconds(self):
        """One advance() must equal one 50 Hz tick, whatever the model timestep."""
        achieved = self.sim.steps_per_control * self.sim.physics_dt
        self.assertAlmostEqual(achieved, 1.0 / 50, places=6)

    def test_rejects_wrong_target_length(self):
        with self.assertRaises(ValueError):
            self.sim.set_targets(np.zeros(5))

    def test_rejects_unknown_source_and_gains(self):
        from g1demo.sim import G1Sim

        with self.assertRaises(ValueError):
            G1Sim(source="shadow")
        with self.assertRaises(ValueError):
            G1Sim(gains="magic")

    def test_model_sources_agree_on_joint_order(self):
        from g1demo import sonic_params as sp
        from g1demo.sim import GAINS_NATIVE, SOURCE_NVIDIA, G1Sim

        nvidia = G1Sim(source=SOURCE_NVIDIA, gains=GAINS_NATIVE, free_base=False)
        names = [
            mujoco.mj_id2name(nvidia.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(nvidia.model.njnt)
        ]
        self.assertEqual(names, list(sp.joint_names()))

    def test_anchored_plant_tracks_a_step_target(self):
        from g1demo.sim import G1Sim

        sim = G1Sim(free_base=False)
        target = sp.default_angles().copy()
        target[0] += 0.05
        sim.set_targets(target)
        for _ in range(20):
            sim.advance()
        self.assertLess(abs(sim.joint_pos[0] - target[0]), 0.05)

    def test_native_gains_stand_and_sonic_gains_do_not(self):
        """The core sim2sim finding, asserted so it cannot regress silently.

        Menagerie ships kp=500 and the G1 holds the default pose unaided. SONIC's
        deployment gains are 14-99, far too soft for this plant, and the robot
        topples. That is why the closed loop needs plant calibration.
        """
        from g1demo.sim import GAINS_NATIVE, GAINS_SONIC, G1Sim

        native = G1Sim(gains=GAINS_NATIVE)
        for _ in range(250):
            native.advance()
        self.assertFalse(native.is_fallen(), "native gains should hold the pose")

        sonic = G1Sim(gains=GAINS_SONIC)
        for _ in range(250):
            sonic.advance()
        self.assertTrue(sonic.is_fallen(), "sonic gains are expected to topple")

    def test_sonic_gains_are_applied_to_the_model(self):
        from g1demo import sonic_params as sp
        from g1demo.sim import G1Sim

        sim = G1Sim(gains="sonic")
        for joint_index, joint_name in enumerate(sp.joint_names()):
            actuator = sim._actuator_for[joint_name]
            self.assertAlmostEqual(
                sim.model.actuator_gainprm[actuator, 0], sp.kp()[joint_index]
            )
            self.assertAlmostEqual(
                sim.model.actuator_biasprm[actuator, 2], -sp.kd()[joint_index]
            )

    @unittest.expectedFailure
    def test_free_base_default_pose_is_passively_stable(self):
        """KNOWN LIMITATION - expected to fail. See docs/sim2sim_gap.md.

        Holding the default pose with the SONIC PD gains, and no policy, the
        MuJoCo G1 topples within about a second: the ankle pitch sags under the
        ground-reaction moment, the body pitches forward, and the equilibrium is
        statically unstable. Stabilising it takes roughly 8x the shipped gains.
        The SONIC policy is what is meant to close this loop, which is why the
        closed-loop result depends on plant fidelity rather than on conventions.
        """
        for _ in range(100):
            self.sim.advance()
        self.assertFalse(self.sim.is_fallen())


class TestClosedLoop(unittest.TestCase):
    @unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
    def test_hand_commands_reach_the_callback_each_tick(self):
        from g1demo.loop import WholeBodyController
        from g1demo.policy import ScriptedReference
        from g1demo.sim import G1Sim

        class WithHands(ScriptedReference):
            def actions(self, sim=None, instruction=None):
                chunk = super().actions(sim, instruction)
                chunk[:, contract()["left_end_effector"].slice] = 0.25
                chunk[:, contract()["right_end_effector"].slice] = 0.75
                return chunk

        received = []
        WholeBodyController().run(
            G1Sim(free_base=False), WithHands(horizon=60), ticks=3,
            on_hand_command=lambda left, right: received.append((left, right)),
        )
        self.assertEqual(len(received), 3)
        for left, right in received:
            np.testing.assert_allclose(left, 0.25)
            np.testing.assert_allclose(right, 0.75)

    @unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
    def test_short_run_produces_a_sane_summary(self):
        from g1demo.loop import WholeBodyController
        from g1demo.policy import ScriptedReference
        from g1demo.sim import G1Sim

        sim = G1Sim()
        controller = WholeBodyController()
        summary, records = controller.run(
            sim, ScriptedReference(motion="standing", horizon=60), ticks=15
        )
        self.assertEqual(summary.ticks, len(records))
        self.assertEqual(summary.ticks, 15)
        self.assertTrue(all(np.isfinite(r.token).all() for r in records))
        self.assertTrue(all(np.isfinite(r.q_target).all() for r in records))
        self.assertEqual(records[0].token.shape, (64,))
        self.assertEqual(records[0].q_target.shape, (29,))
        self.assertEqual(
            summary.max_abs_raw_action,
            max(float(np.abs(r.raw_action).max()) for r in records),
        )

    @unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
    def test_action_gain_scales_the_target(self):
        from g1demo.loop import WholeBodyController
        from g1demo.policy import ScriptedReference
        from g1demo.sim import G1Sim

        reference = ScriptedReference(motion="standing", horizon=60)
        results = {}
        for gain in (1.0, 0.5):
            sim = G1Sim()
            controller = WholeBodyController(action_gain=gain)
            summary, records = controller.run(sim, reference, ticks=1)
            results[gain] = records[0].q_target

        delta_full = results[1.0] - sp.default_angles()
        delta_half = results[0.5] - sp.default_angles()
        np.testing.assert_allclose(delta_half, 0.5 * delta_full, rtol=1e-9, atol=1e-12)

    @unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
    def test_replan_every_controls_how_often_the_policy_is_queried(self):
        """Regression: replan_every used to be accepted and then ignored."""
        from g1demo.loop import WholeBodyController
        from g1demo.policy import ScriptedReference
        from g1demo.sim import G1Sim

        class Counting(ScriptedReference):
            calls = 0

            def actions(self, sim=None, instruction=None):
                Counting.calls += 1
                return super().actions(sim, instruction)

        # A 100-frame chunk supports 55 ticks of lookahead, so with no explicit
        # interval one chunk covers 25 ticks and the policy is asked exactly once.
        for every, expected in ((None, 1), (5, 5), (10, 3)):
            Counting.calls = 0
            WholeBodyController().run(
                G1Sim(free_base=False),
                Counting(motion="standing", horizon=100),
                ticks=25,
                replan_every=every,
            )
            self.assertEqual(Counting.calls, expected, f"replan_every={every}")

    @unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
    def test_chunk_shorter_than_the_lookahead_is_rejected(self):
        from g1demo.loop import WholeBodyController
        from g1demo.policy import ScriptedReference
        from g1demo.sim import G1Sim

        class TooShort(ScriptedReference):
            def actions(self, sim=None, instruction=None):  # noqa: ARG002 (interface)
                return np.zeros((10, contract().action_dim))

        with self.assertRaises(ValueError):
            WholeBodyController().run(G1Sim(free_base=False), TooShort(), ticks=1)

    def test_rejects_non_sonic_rate(self):
        from g1demo.loop import WholeBodyController

        with self.assertRaises(ValueError):
            WholeBodyController(control_hz=30)

    @unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
    def test_rejects_non_positive_ticks(self):
        from g1demo.loop import WholeBodyController

        # The tick-count guard must fire before the simulator is touched, which is
        # why None is an acceptable stand-in for both sim and source here.
        with self.assertRaises(ValueError):
            WholeBodyController(control_hz=50).run(None, None, ticks=0)


class TestScriptedReference(unittest.TestCase):
    def test_every_motion_stays_within_joint_limits(self):
        from g1demo.policy import ScriptedReference

        for motion in ("standing", "arms_up", "squat", "wave"):
            chunk = ScriptedReference(motion=motion, horizon=100).actions()
            self.assertEqual(chunk.shape[1], contract().action_dim)
            joints = chunk[:, contract().flat_indices_for(sp.joint_names())]
            self.assertTrue(
                ((joints >= sp.joint_lower() - 1e-9) & (joints <= sp.joint_upper() + 1e-9)).all(),
                f"{motion} leaves the joint limits",
            )

    def test_targets_the_named_joint_not_a_contract_index(self):
        """Regression: joint lookup must use MuJoCo order, not contract order."""
        from g1demo.policy import ScriptedReference

        c = contract()
        chunk = ScriptedReference(motion="squat", horizon=60).actions()
        start = c.named_joints(chunk[0])
        end = c.named_joints(chunk[-1])
        self.assertGreater(end["left_knee_joint"] - start["left_knee_joint"], 0.3)
        # An arm joint must not move during a squat.
        self.assertAlmostEqual(end["left_elbow_joint"], start["left_elbow_joint"])

    def test_unknown_motion_rejected(self):
        from g1demo.policy import ScriptedReference

        with self.assertRaises(ValueError):
            ScriptedReference(motion="backflip")


class TestFastWAMAdapter(unittest.TestCase):
    def test_observation_and_normalization_reach_infer_action(self):
        from g1demo.policy import FastWAMPolicy

        class FakeModel:
            def infer_action(self, **kwargs):
                self.inputs = kwargs
                return {"action": np.ones((50, contract().action_dim), dtype=np.float32)}

        policy = FastWAMPolicy(action_horizon=50)
        policy.model = FakeModel()
        policy.set_normalizer(np.full(contract().action_dim, 2),
                              np.full(contract().action_dim, 3))
        policy.set_proprio_normalizer(np.ones(29), np.full(29, 2))
        image = np.full((32, 48, 3), 255, dtype=np.uint8)
        actions = policy.predict(image, np.full(29, 3.0), "raise the right hand")
        np.testing.assert_allclose(actions, 5)
        self.assertEqual(policy.model.inputs["prompt"], "raise the right hand")
        np.testing.assert_allclose(policy.model.inputs["proprio"].numpy(), 1)
        np.testing.assert_allclose(policy.model.inputs["input_image"].numpy(), 1)

    def test_missing_checkpoint_and_statistics_fail_closed(self):
        from g1demo.policy import FastWAMPolicy

        policy = FastWAMPolicy()
        with self.assertRaises(FileNotFoundError):
            policy.load()
        with self.assertRaises(RuntimeError):
            policy.predict(np.zeros((32, 32, 3), dtype=np.uint8),
                           np.zeros(29), "stand")


class TestStage2Data(unittest.TestCase):
    @unittest.skipUnless(HAVE_SONIC, SONIC_SKIP)
    def test_real_sonic_encoder_produces_one_stage2_label(self):
        import tempfile

        from g1demo.stage2_data import prepare_episode

        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "raw.npz", Path(folder) / "prepared.npz"
            np.savez_compressed(
                source, rgb=np.zeros((46, 32, 32, 3), dtype=np.uint8),
                joint_pos=np.tile(sp.default_angles(), (46, 1)),
                base_quat_wxyz=np.tile([1, 0, 0, 0], (46, 1)),
                action_ref=default_chunk(46), timestamp_s=np.arange(46) / 50,
                instruction="stand still", task_id="test/standing",
            )
            prepare_episode(source, output)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data["sonic_token"].shape, (1, 64))
                self.assertEqual(data["latent_action"].shape,
                                 (1, 64 + sum(contract().hand_sizes.values())))
                self.assertTrue(np.isfinite(data["latent_action"]).all())

    def test_prepared_rows_align_observation_token_and_hands(self):
        import tempfile

        from g1demo.stage2_data import prepare_episode

        class Encoder:
            def encode(self, actions, _quat, _yaw, frame=0):
                return {
                    "token": np.full(64, frame, dtype=np.float32),
                    "left_hand": actions[frame, contract()["left_end_effector"].slice],
                    "right_hand": actions[frame, contract()["right_end_effector"].slice],
                }

        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "raw.npz", Path(folder) / "prepared.npz"
            actions = default_chunk(48)
            actions[:, contract()["left_end_effector"].slice] = np.arange(48)[:, None]
            actions[:, contract()["right_end_effector"].slice] = 2
            np.savez_compressed(
                source, rgb=np.zeros((48, 32, 32, 3), dtype=np.uint8),
                joint_pos=np.tile(sp.default_angles(), (48, 1)),
                base_quat_wxyz=np.tile([1, 0, 0, 0], (48, 1)),
                action_ref=actions, timestamp_s=np.arange(48) / 50,
                instruction="raise the right hand", task_id="test/raise_hand",
            )
            prepare_episode(source, output, encoder=Encoder())
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data["sonic_token"].shape, (3, 64))
                np.testing.assert_array_equal(data["sonic_token"][:, 0], [0, 1, 2])
                np.testing.assert_array_equal(data["left_hand"][:, 0], [0, 1, 2])
                self.assertEqual(str(data["instruction"]), "raise the right hand")


class TestBridge(unittest.TestCase):
    def test_split_chunk_orders_joints_isaaclab(self):
        from g1demo.bridge import split_chunk

        chunk = default_chunk(10)
        joints, root, hands = split_chunk(chunk)
        self.assertEqual(joints.shape, (10, 29))
        self.assertEqual(root.shape, (10, 3))
        # IsaacLab index 9 is the left knee (0.669 in the default pose).
        self.assertAlmostEqual(joints[0][9], 0.669, places=6)
        self.assertAlmostEqual(joints[0][0], -0.312, places=6)

    def test_split_rejects_out_of_limit(self):
        from g1demo.bridge import split_chunk

        chunk = default_chunk(10)
        chunk[0, 0] = 100.0
        with self.assertRaises(ValueError):
            split_chunk(chunk)

    def test_export_writes_the_expected_files(self):
        import tempfile

        from g1demo.bridge import export

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "clip"
            export(default_chunk(20), out, origin="test")
            for name in ("joint_pos.csv", "joint_vel.csv", "body_pos.csv",
                         "body_quat.csv", "metadata.txt", "hands.json",
                         "bridge_manifest.json"):
                self.assertTrue((out / name).exists(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
