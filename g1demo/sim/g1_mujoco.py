"""The G1 models the demo can run, and how the SONIC decoder drives them.

Two sources are supported, both exposing 29 actuators in the same joint order —
which is also the order in ``configs/g1_sonic_params.json``, so neither needs a
permutation.

``menagerie`` (default)
    The official Unitree G1 29-DoF rev 1.0 from ``mujoco_menagerie``: real
    collision meshes, a checkered scene, 33.3 kg. This is a real G1 to look at.

``nvidia``
    A rebuild of the MJCF shipped inside GEAR-SONIC. Its STL files are Git-LFS
    pointers, so a normal clone cannot load it; the rebuild keeps the kinematic
    tree, inertias, joint limits and foot contact spheres, and replaces the
    visual meshes with primitives. It renders as a stick figure, which is why it
    is no longer the default.

Gains
    ``gains="sonic"`` applies the kp/kd from SONIC's C++ deployment header, i.e.
    what the real robot is actually commanded with. ``gains="native"`` keeps the
    model's own servos. This choice matters a great deal: Menagerie ships
    ``kp=500`` and stands on its own, while SONIC's deployment gains are 14-99
    and the same plant topples. See ``docs/sim2sim_gap.md``.

    Either way the torque law is the same as the deployment's::

        torque = kp * (q_target - q) - kd * qdot
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..paths import ASSETS, menagerie_g1_scene, sonic_repo
from ..sonic_params import (
    default_angles,
    effort_limit,
    joint_lower,
    joint_names,
    joint_upper,
    kd,
    kp,
)

SOURCE_MENAGERIE = "menagerie"
SOURCE_NVIDIA = "nvidia"
GAINS_SONIC = "sonic"
GAINS_NATIVE = "native"

#: Control rate of the SONIC policy, in Hz.
CONTROL_HZ = 50
#: Physics rate, in Hz. Four physics steps per control tick.
PHYSICS_HZ = 200
PHYSICS_DT = 1.0 / PHYSICS_HZ

NVIDIA_SOURCE_RELATIVE = Path("gear_sonic/data/robots/g1/g1_29dof_old.xml")
MENAGERIE_G1_XML = "g1.xml"
DEFAULT_OUTPUT = ASSETS / "g1_meshfree.xml"
ANCHORED_OUTPUT = ASSETS / "g1_anchored.xml"

# Visual radius per body keyword, for the mesh-free rebuild only. Purely
# cosmetic: meshes there are contype=0 and never collide.
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


def _radius(body_name: str) -> float:
    for keyword, radius in _RADIUS_BY_KEYWORD:
        if keyword in body_name:
            return radius
    return _DEFAULT_RADIUS


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
def _add_visuals_and_floor(root: ET.Element) -> None:
    """Add the lighting and ground an ``<include>``-style MJCF expects."""
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7",
                  ambient="0.35 0.35 0.35", specular="0.1 0.1 0.1")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")

    world = root.find("worldbody")
    world.insert(0, ET.Element("light", pos="0 0 2.5", dir="0 0 -1",
                               directional="true", castshadow="false"))
    ET.SubElement(world, "geom", name="floor", type="plane", size="4 4 0.05",
                  rgba="0.32 0.34 0.38 1", condim="3",
                  friction="1.0 0.005 0.0001")


def _set_integrator(root: ET.Element, integrator: str) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("integrator", integrator)
    option.set("timestep", f"{PHYSICS_DT:.6f}")


def _remove_free_joint(root: ET.Element) -> None:
    """Weld the root to the world by deleting its free joint.

    MuJoCo has two spellings: a ``<freejoint/>`` element, which Menagerie uses,
    and ``<joint type="free"/>``, which NVIDIA's MJCF uses. Both must be handled
    or the "anchored" model silently keeps 30 joints and can still fall over.
    """
    world = root.find("worldbody")
    pelvis = world.find("body") if world is not None else None
    if pelvis is None:
        raise ValueError("Unexpected MJCF: no root body in worldbody")
    removed = 0
    for child in list(pelvis):
        if child.tag == "freejoint" or (
            child.tag == "joint" and child.get("type") == "free"
        ):
            pelvis.remove(child)
            removed += 1
    if removed != 1:
        raise ValueError(f"Expected exactly one free joint on the root, found {removed}")

    # Keyframes encode the free-base qpos width (36 = 7 + 29), so they become
    # invalid once the root is welded. Nothing here uses them: the pose is set
    # from SONIC's default_angles at reset.
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)


def _write(tree: ET.ElementTree, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="unicode", xml_declaration=True)
    return output


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def menagerie_model(free_base: bool = True, integrator: str = "implicitfast",
                    output: Path | None = None) -> Path:
    """Return the Menagerie G1 scene, or an anchored variant of it.

    The free-base case uses the shipped ``scene.xml`` unchanged. Anchored mode
    has to strip the root free joint, which lives in the included ``g1.xml``, so
    that variant is generated from ``g1.xml`` instead and given its own lighting
    and floor.
    """
    scene = menagerie_g1_scene()
    if free_base:
        return scene

    output = Path(output) if output else ANCHORED_OUTPUT
    tree = ET.parse(scene.parent / MENAGERIE_G1_XML)
    root = tree.getroot()
    root.set("model", "g1_29dof_menagerie_anchored")
    # The generated file lives outside the Menagerie tree, so point at its assets.
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((scene.parent / "assets").resolve()))
    _remove_free_joint(root)
    _add_visuals_and_floor(root)
    _set_integrator(root, integrator)
    return _write(tree, output)


def nvidia_meshfree_model(free_base: bool = True, integrator: str = "implicitfast",
                          output: Path | None = None) -> Path:
    """Rebuild NVIDIA's G1 MJCF without its (LFS-pointer) meshes.

    The rebuild keeps the kinematic tree, inertias, joint limits and the four
    foot contact spheres that are the only colliding geometry upstream defines.
    """
    source = sonic_repo() / NVIDIA_SOURCE_RELATIVE
    if not source.exists():
        raise FileNotFoundError(
            f"Missing the SONIC G1 MJCF: {source}\n"
            f"Set SONIC_REPO to your GR00T-WholeBodyControl checkout, or use "
            f"source='{SOURCE_MENAGERIE}'."
        )
    output = Path(output) if output else DEFAULT_OUTPUT

    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "g1_29dof_nvidia_meshfree")
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)

    world = root.find("worldbody")
    if not free_base:
        _remove_free_joint(root)

    for body in world.iter("body"):
        name = body.get("name", "")
        radius = _radius(name)
        rgba = "0.30 0.45 0.70 1" if "left" in name else "0.72 0.76 0.82 1"
        for geom in list(body.findall("geom")):
            if geom.get("type") == "mesh":
                body.remove(geom)
                ET.SubElement(body, "geom", type="sphere", size=f"{radius}",
                              pos=geom.get("pos", "0 0 0"), contype="0",
                              conaffinity="0", group="1", rgba=rgba)
        for child in body.findall("body"):
            offset = [float(v) for v in child.get("pos", "0 0 0").split()]
            if sum(v * v for v in offset) < 1e-4:
                continue
            ET.SubElement(body, "geom", type="capsule", size=f"{radius * 0.62:.4f}",
                          fromto="0 0 0 " + " ".join(f"{v:.6f}" for v in offset),
                          contype="0", conaffinity="0", group="1", rgba=rgba)

    _add_visuals_and_floor(root)
    _set_integrator(root, integrator)
    _write_position_actuators(root)
    return _write(tree, output)


def _write_position_actuators(root: ET.Element) -> None:
    """Replace the NVIDIA MJCF's torque motors with SONIC position servos.

    Menagerie already ships position servos, so this is only needed for its
    rebuild. Gains are applied to the compiled model instead.
    """
    existing = root.find("actuator")
    if existing is not None:
        root.remove(existing)
    actuator = ET.SubElement(root, "actuator")
    lower, upper = joint_lower(), joint_upper()
    limits = effort_limit()
    for index, name in enumerate(joint_names()):
        ET.SubElement(actuator, "position",
                      name=name, joint=name,
                      kp=f"{kp()[index]:.6f}", kv=f"{kd()[index]:.6f}",
                      ctrlrange=f"{lower[index]:.6f} {upper[index]:.6f}",
                      forcerange=f"{-limits[index]:.6f} {limits[index]:.6f}")


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class G1Sim:
    """A steppable MuJoCo G1 with the SONIC gains and joint order."""

    def __init__(self, xml: Path | str | None = None,
                 source: str = SOURCE_MENAGERIE, gains: str = GAINS_SONIC,
                 free_base: bool = True, integrator: str = "implicitfast") -> None:
        import mujoco

        if source not in (SOURCE_MENAGERIE, SOURCE_NVIDIA):
            raise ValueError(
                f"Unknown source {source!r}; expected {SOURCE_MENAGERIE!r} or "
                f"{SOURCE_NVIDIA!r}"
            )
        if gains not in (GAINS_SONIC, GAINS_NATIVE):
            raise ValueError(
                f"Unknown gains {gains!r}; expected {GAINS_SONIC!r} or {GAINS_NATIVE!r}"
            )

        self._mujoco = mujoco
        if xml is not None:
            path = Path(xml)
        elif source == SOURCE_MENAGERIE:
            path = menagerie_model(free_base=free_base, integrator=integrator)
        else:
            path = nvidia_meshfree_model(free_base=free_base, integrator=integrator)

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.source = source
        self.gains = gains
        self.free_base = free_base
        # Derive the substep count from the model's own timestep rather than an
        # assumed one. Getting this wrong silently changes the control rate,
        # which invalidates every timing in SONIC's 50 Hz reference window.
        self.physics_dt = float(self.model.opt.timestep)
        self.steps_per_control = max(1, int(round((1.0 / CONTROL_HZ) / self.physics_dt)))
        achieved = self.steps_per_control * self.physics_dt
        if abs(achieved * CONTROL_HZ - 1.0) > 0.02:
            raise ValueError(
                f"Model timestep {self.physics_dt} cannot realise {CONTROL_HZ} Hz: "
                f"{self.steps_per_control} substeps give {achieved:.5f} s per control tick"
            )

        if self.model.nu != 29:
            raise RuntimeError(f"Expected 29 actuators, got {self.model.nu}")
        expected_joints = 30 if free_base else 29
        if self.model.njnt != expected_joints:
            raise RuntimeError(f"Expected {expected_joints} joints, got {self.model.njnt}")

        self._pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self._pelvis < 0:
            raise RuntimeError("Could not find the pelvis body")

        # Map each joint name onto the actuator that drives it. Actuator order is
        # not guaranteed to match joint order, so look it up rather than assume.
        self._actuator_for = {
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, int(self.model.actuator_trnid[i, 0])
            ): i
            for i in range(self.model.nu)
        }
        missing = [name for name in joint_names() if name not in self._actuator_for]
        if missing:
            raise RuntimeError(f"Model is missing actuators for: {missing}")
        # Model actuator order expressed in SONIC's joint order.
        self._order = np.array(
            [joint_names().index(name) for name in self._actuator_for]
        )

        if gains == GAINS_SONIC:
            self._apply_sonic_gains()

        self.reset()

    def _apply_sonic_gains(self) -> None:
        """Overwrite the model's servos with SONIC's deployment kp/kd.

        This is what the real robot is commanded with, so it is the honest
        choice for testing the policy; it is also much softer than Menagerie's
        own servos, which is the substance of the sim2sim gap.
        """
        for actuator, joint_index in enumerate(self._order):
            self.model.actuator_gainprm[actuator, 0] = kp()[joint_index]
            self.model.actuator_biasprm[actuator, 1] = -kp()[joint_index]
            self.model.actuator_biasprm[actuator, 2] = -kd()[joint_index]
            limit = effort_limit()[joint_index]
            self.model.actuator_forcerange[actuator] = [-limit, limit]

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
        if settle:
            # Hold the default pose briefly so the contacts load before the
            # controller takes over.
            for _ in range(5 * self.steps_per_control):
                self._physics_step()

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
        which matches SONIC's ``his_base_angular_velocity`` and needs no change.
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

    def joint_torques(self) -> np.ndarray:
        """Actuator torque per joint, MuJoCo order (N.m)."""
        return np.array([self.data.actuator_force[i] for i in range(self.model.nu)])

    # -- rendering --------------------------------------------------------
    def render(self, height: int = 480, width: int = 640,
               camera=None, azimuth: float = 125.0, elevation: float = -6.0,
               distance: float = 1.9, lookat=(0.0, 0.0, 0.72)) -> np.ndarray:
        """Render one offscreen frame as an ``(H, W, 3)`` uint8 array.

        Defaults frame the standing robot. Set ``MUJOCO_GL`` to ``glfw`` where a
        display exists, or ``egl`` headless; without either, MuJoCo cannot create
        a GL context and this raises.
        """
        import mujoco

        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if hasattr(self, "_renderer"):
            self._renderer.close()
            del self._renderer
