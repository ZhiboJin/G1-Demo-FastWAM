"""Build a mesh-free, fixed-torso G1 task scene from the local NVIDIA MJCF.

Only for the first manipulation demo. It is deliberately not a SONIC dynamics model.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from project_paths import ROOT, sonic_repo

SOURCE = sonic_repo() / "gear_sonic/data/robots/g1/g1_29dof_old.xml"
OUTPUT = ROOT / "assets/g1_meshfree.xml"


def build():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing local G1 source model: {SOURCE}")
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "g1_29dof_meshfree_reaching_demo")
    compiler = root.find("compiler")
    compiler.attrib.pop("meshdir", None)
    assets = root.find("asset")
    for mesh in list(assets.findall("mesh")):
        assets.remove(mesh)
    world = root.find("worldbody")
    pelvis = world.find("body")
    for joint in list(pelvis.findall("joint")):
        if joint.get("type") == "free":
            pelvis.remove(joint)
    for body in world.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("type") == "mesh":
                body.remove(geom)
        name = body.get("name", "")
        radius = 0.085 if name == "pelvis" else (0.055 if "torso" in name else 0.035)
        rgba = "0.25 0.45 0.75 1" if "left" in name else "0.7 0.75 0.8 1"
        ET.SubElement(body, "geom", type="sphere", size=str(radius), pos="0 0 0",
                      contype="0", conaffinity="0", rgba=rgba)
        for child in body.findall("body"):
            xyz = [float(x) for x in child.get("pos", "0 0 0").split()]
            if sum(x*x for x in xyz) > 0.0025:
                ET.SubElement(body, "geom", type="capsule", size="0.022",
                              fromto="0 0 0 " + " ".join(str(x) for x in xyz),
                              contype="0", conaffinity="0", rgba=rgba)
        if name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            ET.SubElement(body, "site", name=name.split("_wrist")[0] + "_ee",
                          pos="0.11 0 0", size="0.018", rgba="0 1 0 1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05",
                  rgba="0.15 0.18 0.22 1")
    for side, xyz in (("left", "0.12 0.25 0.85"), ("right", "0.12 -0.25 0.85")):
        ET.SubElement(world, "site", name=side + "_target", pos=xyz,
                      size="0.035", rgba="1 0.3 0.2 1")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="unicode", xml_declaration=True)
    return OUTPUT


if __name__ == "__main__":
    print(build())
