"""A MuJoCo G1 that the SONIC decoder can actually drive.

The official G1 MJCF ships STL meshes that are Git-LFS pointers in a normal
checkout, so it will not load. This module rebuilds the *same* kinematic tree
from it with primitive visuals.

Crucially, the rebuild keeps the model's collision behaviour identical to the
official one. In the upstream MJCF the visual meshes are ``contype=0
conaffinity=0`` and the only colliding geometry is a set of four small spheres
under each foot. Those spheres are primitives, not meshes, so they survive the
rebuild untouched. The robot therefore stands on exactly the contacts NVIDIA's
own scene uses.

The actuators are position servos with the gains the real controller sends::

    torque = kp * (q_target - q) - kd * qdot

with ``kp``/``kd`` from the SONIC deployment header and the torque clamped to
the model's own actuator limits. That is the same PD law as
``motor_command.q_target`` + ``kp``/``kd`` + ``tau_ff = 0`` in the C++
deployment.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..paths import ASSETS, sonic_repo
from ..sonic_params import (
    default_angles,
    effort_limit,
    joint_lower,
    joint_names,
    joint_upper,
    kd,
    kp,
)

#: Control rate of the SONIC policy, in Hz.
CONTROL_HZ = 50
#: Physics rate, in Hz. Four physics steps per control tick.
PHYSICS_HZ = 200
PHYSICS_DT = 1.0 / PHYSICS_HZ
STEPS_PER_CONTROL = PHYSICS_HZ // CONTROL_HZ

SOURCE_RELATIVE = Path("gear_sonic/data/robots/g1/g1_29dof_old.xml")
DEFAULT_OUTPUT = ASSETS / "g1_meshfree.xml"

# Visual radius per body, chosen so the primitive roughly fills the mesh it
# replaces. Meshes are purely cosmetic here, so this only affects legibility.
_RADIUS_BY_KEYWORD = (
    ("pelvis", 0.085),
    ("torso", 0.075),
    ("waist", 0.055),
    ("head", 0.055),
    ("shoulder", 0.045),
    ("elbow", 0.038),
    ("wrist", 0.030),
    ("hip", 0.045),
    ("knee", 0.048),
    ("ankle", 0.035),
)
_DEFAULT_RADIUS = 0.035

#: The foot contact spheres upstream places on each ``*_ankle_roll_link``,
#: as (x, y, z) offsets in the link frame. Used to size the replacement sole.
_SOURCE_FOOT_SPHERES = (
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
)


def _sole_footprint(points) -> dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "center_x": (min(xs) + max(xs)) / 2.0,
        "half_x": (max(xs) - min(xs)) / 2.0,
        "half_y": (max(ys) - min(ys)) / 2.0,
    }


def _radius(body_name: str) -> float:
    for keyword, radius in _RADIUS_BY_KEYWORD:
        if keyword in body_name:
            return radius
    return _DEFAULT_RADIUS


def _effort_limits(source_root: ET.Element) -> dict[str, float]:
    """Torque limits, taken from the official model's own motor ctrlrange."""
    limits: dict[str, float] = {}
    actuator = source_root.find("actuator")
    if actuator is None:
        return limits
    for motor in actuator.findall("motor"):
        joint = motor.get("joint")
        span = motor.get("ctrlrange")
        if joint and span:
            low, high = (float(value) for value in span.split())
            limits[joint] = min(abs(low), abs(high))
    return limits


def build_model(source: Path | None = None, output: Path | None = None,
                free_base: bool = True, integrator: str = "implicitfast",
                policy_effort_limits: bool = False, foot_sole: bool = False) -> Path:
    """Write a mesh-free G1 MJCF and return its path.

    ``free_base=False`` welds the pelvis to the world, which is useful for
    isolating controller bugs from balance failures.

    The two fidelity switches default to *upstream*, because neither measurably
    improved balance and both move the model away from NVIDIA's own scene:

    ``foot_sole``
        ``False`` keeps upstream's four 5 mm contact spheres per foot. ``True``
        replaces them with a thin sole box, which is closer to the real G1 foot
        but settles ~4 cm lower and did not stabilise the balance loop.
    ``policy_effort_limits``
        ``False`` uses the MJCF's own motor ratings. ``True`` uses the policy's
        IsaacLab effort limits, which disagree for several joints because the
        shipped MJCF describes the *older* G1 hardware while the policy header
        describes the newer one ("old is 7520_14 new is 7520_22").
    """
    source = Path(source) if source else sonic_repo() / SOURCE_RELATIVE
    output = Path(output) if output else DEFAULT_OUTPUT
    if not source.exists():
        raise FileNotFoundError(
            f"Missing the official G1 MJCF: {source}\n"
            f"Set SONIC_REPO to your GR00T-WholeBodyControl checkout."
        )

    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "g1_29dof_meshfree_sonic_demo")

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)

    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)

    world = root.find("worldbody")
    pelvis = world.find("body") if world is not None else None
    if pelvis is None:
        raise ValueError(f"Unexpected G1 MJCF: no pelvis body in {source}")

    if not free_base:
        for joint in list(pelvis.findall("joint")):
            if joint.get("type") == "free":
                pelvis.remove(joint)

    # Replace each visual mesh with a primitive, keeping every non-mesh geom
    # (the foot contact spheres) exactly as upstream defined it.
    for body in world.iter("body"):
        name = body.get("name", "")
        radius = _radius(name)
        rgba = "0.30 0.45 0.70 1" if "left" in name else "0.72 0.76 0.82 1"
        for geom in list(body.findall("geom")):
            if geom.get("type") == "mesh":
                body.remove(geom)
                ET.SubElement(
                    body, "geom", type="sphere", size=f"{radius}",
                    pos=geom.get("pos", "0 0 0"), contype="0", conaffinity="0",
                    group="1", rgba=rgba,
                )
        if foot_sole and name.endswith("_ankle_roll_link"):
            # Upstream stands on four 5 mm spheres per foot. That is a usable
            # contact model for a policy trained against it, but it is a much
            # softer, point-like support than the real foot plate, and a balance
            # policy is extremely sensitive to it. Replace the spheres with a
            # thin sole box spanning the same footprint.
            for geom in list(body.findall("geom")):
                if geom.get("type") != "mesh":
                    body.remove(geom)
            points = _sole_footprint(_SOURCE_FOOT_SPHERES)
            ET.SubElement(
                body, "geom", name=f"{name}_sole", type="box",
                size=f"{points['half_x']:.4f} {points['half_y']:.4f} 0.006",
                pos=f"{points['center_x']:.4f} 0 0.002", group="2",
                condim="4", friction="1.0 0.01 0.001", rgba="0.15 0.15 0.18 1",
            )
        # A capsule towards each child makes the limb read as a limb.
        for child in body.findall("body"):
            offset = [float(value) for value in child.get("pos", "0 0 0").split()]
            if sum(value * value for value in offset) < 1e-4:
                continue
            ET.SubElement(
                body, "geom", type="capsule", size=f"{radius * 0.62:.4f}",
                fromto="0 0 0 " + " ".join(f"{value:.6f}" for value in offset),
                contype="0", conaffinity="0", group="1", rgba=rgba,
            )

    # A ground plane to stand on.
    ET.SubElement(
        world, "geom", name="floor", type="plane", size="4 4 0.05",
        rgba="0.22 0.24 0.28 1", condim="3", friction="1.0 0.005 0.0001",
    )

    # `implicitfast` integrates the actuator damping implicitly. With position
    # servos at the SONIC gains (kp up to ~99) the default explicit Euler
    # integrator is far less stable, and the instability shows up as the
    # balance loop diverging rather than as anything obviously numerical.
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("integrator", integrator)
    option.set("timestep", f"{PHYSICS_DT:.6f}")

    _write_position_actuators(root, policy_effort_limits)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=True)
    return output


def _write_position_actuators(root: ET.Element, policy_effort_limits: bool = True) -> None:
    """Swap the model's torque motors for position servos with SONIC gains."""
    model_limits = _effort_limits(root)
    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)

    actuator = ET.SubElement(root, "actuator")
    gains_p, gains_d = kp(), kd()
    lower, upper = joint_lower(), joint_upper()
    policy_limits = effort_limit()
    for index, name in enumerate(joint_names()):
        attributes = {
            "name": name,
            "joint": name,
            "kp": f"{gains_p[index]:.6f}",
            "kv": f"{gains_d[index]:.6f}",
            # ctrl is a joint angle, so let it span the mechanical range ...
            "ctrlrange": f"{lower[index]:.6f} {upper[index]:.6f}",
        }
        # ... while torque is capped at the effort limit. The policy's IsaacLab
        # limit and the MJCF's own motor rating disagree for several joints
        # (hip pitch 139 vs 88, ankle 25 vs 50), so this choice is explicit.
        limit = policy_limits[index] if policy_effort_limits else model_limits.get(
            name, policy_limits[index]
        )
        attributes["forcerange"] = f"{-limit:.6f} {limit:.6f}"
        ET.SubElement(actuator, "position", **attributes)


class G1Sim:
    """A steppable MuJoCo G1 with SONIC-matching actuators."""

    def __init__(self, xml: Path | str | None = None, free_base: bool = True,
                 physics_dt: float = PHYSICS_DT, integrator: str = "implicitfast",
                 policy_effort_limits: bool = False, foot_sole: bool = False) -> None:
        import mujoco

        self._mujoco = mujoco
        path = Path(xml) if xml else build_model(
            free_base=free_base,
            integrator=integrator,
            policy_effort_limits=policy_effort_limits,
            foot_sole=foot_sole,
        )
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        self.free_base = free_base
        self.physics_dt = physics_dt
        self.steps_per_control = max(1, int(round((1.0 / CONTROL_HZ) / physics_dt)))

        if self.model.nu != 29:
            raise RuntimeError(f"Expected 29 actuators, got {self.model.nu}")
        expected = 30 if free_base else 29
        if self.model.njnt != expected:
            raise RuntimeError(
                f"Expected {expected} joints, got {self.model.njnt}"
            )

        self._actuator_joint = np.array(
            [int(self.model.actuator_trnid[i, 0]) for i in range(self.model.nu)]
        )
        self._pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self._pelvis < 0:
            raise RuntimeError("Could not find the pelvis body in the G1 model")
        # Map the model's actuator order onto MuJoCo joint order.
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            for joint in self._actuator_joint
        ]
        self._actuator_for = {
            name: index for index, name in enumerate(names)
        }
        missing = [name for name in joint_names() if name not in self._actuator_for]
        if missing:
            raise RuntimeError(f"Model is missing actuators for: {missing}")

        self.reset()

    # -- state ------------------------------------------------------------
    @property
    def nq_offset(self) -> int:
        return 7 if self.free_base else 0

    @property
    def nv_offset(self) -> int:
        return 6 if self.free_base else 0

    def reset(self, settle: bool = True) -> None:
        """Reset to the default standing pose and let it settle briefly."""
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self.free_base:
            self.data.qpos[0:3] = [0.0, 0.0, 0.793]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.nq_offset:] = default_angles()
        self.set_targets(default_angles())
        self._mujoco.mj_forward(self.model, self.data)
        if self.free_base:
            self._drop_onto_floor()
        if settle:
            # Hold the default pose for a few ticks so the contacts are loaded
            # before the controller takes over.
            for _ in range(5 * self.steps_per_control):
                self._physics_step()

    def _drop_onto_floor(self) -> None:
        """Lower the pelvis until the collision geometry just touches z = 0.

        The shipped MJCF places the pelvis at 0.793 m, which leaves the foot
        contact spheres about 3.6 cm above the ground. Dropping from there lands
        on four 5 mm point contacts, and the resulting impact is enough to topple
        the robot before any controller runs. Starting in contact removes that.
        """
        lowest = np.inf
        for geom in range(self.model.ngeom):
            if self.model.geom_contype[geom] == 0 and self.model.geom_conaffinity[geom] == 0:
                continue  # visual-only geometry cannot collide
            lowest = min(lowest, self.data.geom_xpos[geom][2] - self.model.geom_rbound[geom])
        if np.isfinite(lowest) and lowest > 0.0:
            self.data.qpos[2] -= lowest
            self._mujoco.mj_forward(self.model, self.data)

    def set_targets(self, q_target_mujoco: np.ndarray) -> None:
        """Command joint position targets (MuJoCo order, radians)."""
        target = np.asarray(q_target_mujoco, dtype=np.float64).reshape(-1)
        if target.shape != (29,):
            raise ValueError(f"Expected 29 joint targets, got {target.shape}")
        for index, name in enumerate(joint_names()):
            self.data.ctrl[self._actuator_for[name]] = target[index]

    def _physics_step(self) -> None:
        self._mujoco.mj_step(self.model, self.data)

    def advance(self) -> None:
        """Run one 50 Hz control tick of physics."""
        for _ in range(self.steps_per_control):
            self._physics_step()

    # -- observations -----------------------------------------------------
    @property
    def joint_pos(self) -> np.ndarray:
        """Measured joint positions, MuJoCo order, radians."""
        return self.data.qpos[self.nq_offset:].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        """Measured joint velocities, MuJoCo order, rad/s."""
        return self.data.qvel[self.nv_offset:].copy()

    @property
    def base_quat_wxyz(self) -> np.ndarray:
        if not self.free_base:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return self.data.qpos[3:7].copy()

    @property
    def base_position(self) -> np.ndarray:
        """Pelvis position in world coordinates, valid in both base modes."""
        return self.data.xpos[self._pelvis].copy()

    @property
    def base_ang_vel_body(self) -> np.ndarray:
        """Body-frame angular velocity (rad/s) -- what an IMU gyro reports.

        MuJoCo stores a free joint's rotational velocity in the local body frame,
        which matches SONIC's ``his_base_angular_velocity`` observation and needs
        no conversion.
        """
        if not self.free_base:
            return np.zeros(3)
        return self.data.qvel[3:6].copy()

    def is_fallen(self, threshold: float = 0.45) -> bool:
        """Whether the pelvis has dropped far below the standing height.

        An anchored robot has no free joint and therefore cannot fall.
        """
        if not self.free_base:
            return False
        return bool(self.base_position[2] < threshold)

    def lowest_contact_height(self) -> float:
        """Height of the lowest colliding geometry above the floor plane.

        Should be ~0 after :meth:`reset`: the robot must start *in contact*. The
        shipped MJCF otherwise leaves the feet in mid-air.
        """
        lowest = np.inf
        for geom in range(self.model.ngeom):
            if self.model.geom_contype[geom] == 0 and self.model.geom_conaffinity[geom] == 0:
                continue
            lowest = min(
                lowest,
                self.data.geom_xpos[geom][2] - self.model.geom_rbound[geom],
            )
        return float(lowest)

    # -- rendering --------------------------------------------------------
    def render(self, height: int = 480, width: int = 640,
               camera=None, azimuth: float = 135.0, elevation: float = -12.0,
               distance: float = 2.4) -> np.ndarray:
        """Render one offscreen frame as an ``(H, W, 3)`` uint8 array."""
        import mujoco

        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [0.0, 0.0, 0.7]
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if hasattr(self, "_renderer"):
            self._renderer.close()
            del self._renderer
