"""Extract the authoritative G1 constants from the official SONIC C++ header.

Why this exists
---------------
The SONIC deployment binary turns a decoder action into a motor command with::

    q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i]

``default_angles``, ``g1_action_scale``, the joint permutations and the PD gains
are *hard-coded* in one header inside NVlabs/GR00T-WholeBodyControl. Rather than
transcribing 100+ magic numbers by hand (and risking a silent typo that would
quietly corrupt every joint target), this tool parses that header and emits a
JSON file the demo loads at runtime.

The parser evaluates the header's *own* C++ constant expressions in a restricted
namespace, so ``0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22`` is computed
exactly as the compiler would, with no duplicated arithmetic.

Usage
-----
    python tools/extract_sonic_params.py --sonic-repo /path/to/GR00T-WholeBodyControl

Writes ``configs/g1_sonic_params.json``. Re-run it if the SONIC checkout is
updated; the emitted file records the source header's SHA-256 so drift is
detectable.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADER_RELATIVE = Path(
    "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
)
MJCF_RELATIVE = Path("gear_sonic/data/robots/g1/g1_29dof_old.xml")
URDF_RELATIVE = Path("gear_sonic/data/robots/g1/g1_29dof.urdf")
OUTPUT = ROOT / "configs/g1_sonic_params.json"


# --------------------------------------------------------------------------
# A tiny, safe evaluator for the arithmetic used in the header.
# --------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST, names: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"Unknown identifier {node.id!r} in header expression")
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, names), _eval(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand, names))
    raise ValueError(f"Unsupported expression node {ast.dump(node)}")


def evaluate(expression: str, names: dict[str, float]) -> float:
    return _eval(ast.parse(expression, mode="eval"), names)


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------
def scalar_constants(text: str) -> dict[str, float]:
    """Collect ``const double NAME = <expr>;`` definitions, in dependency order."""
    names: dict[str, float] = {}
    pattern = re.compile(
        r"^\s*const\s+double\s+(\w+)\s*=\s*([^;]+);", re.MULTILINE
    )
    # Two passes so a constant may reference one declared later in the file.
    pending = [(m.group(1), m.group(2).strip()) for m in pattern.finditer(text)]
    for _ in range(len(pending) + 1):
        remaining = []
        progress = False
        for name, expression in pending:
            try:
                names[name] = evaluate(expression, names)
                progress = True
            except ValueError:
                remaining.append((name, expression))
        pending = remaining
        if not pending or not progress:
            break
    if pending:
        unresolved = ", ".join(name for name, _ in pending)
        raise ValueError(f"Could not resolve header constants: {unresolved}")
    return names


def array_body(text: str, declaration: str) -> str:
    """Return the ``{...}`` body of ``const std::array<...> declaration = {...};``.

    Anchoring on the full declaration (rather than a bare name search) matters:
    several of these names also appear in the header's prose comments, and
    naively matching the first occurrence picks up the wrong initializer.
    """
    pattern = re.compile(
        r"const\s+std::array\s*<[^>]*>\s*" + re.escape(declaration) + r"\s*=\s*\{"
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Declaration not found: {declaration}")
    brace = match.end() - 1
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    raise ValueError(f"Unterminated initializer for {declaration}")


def expressions(body: str) -> list[str]:
    """Split an initializer body into one expression per array element.

    Comments are removed first and the remainder is split on commas, because the
    header formats some arrays one-element-per-line (with a trailing ``//``
    comment) and others wrapped across lines. A line-based split would silently
    merge wrapped elements into one bogus token.
    """
    without_comments = re.sub(r"//[^\n]*", "", body)
    return [token.strip() for token in without_comments.split(",") if token.strip()]


def line_comments(body: str) -> list[str]:
    """Trailing ``//`` comments, one per source line (used for joint names)."""
    comments = []
    for raw in body.splitlines():
        _, marker, comment = raw.partition("//")
        if marker:
            comments.append(comment.strip())
    return comments


def evaluate_array(text: str, declaration: str, names: dict[str, float]) -> list[float]:
    return [evaluate(expr, names) for expr in expressions(array_body(text, declaration))]


def int_array(text: str, declaration: str) -> list[int]:
    return [int(expr) for expr in expressions(array_body(text, declaration))]


def effort_limits(text: str, names: dict[str, float]) -> list[float]:
    """Per-joint effort limit (MuJoCo order), read from each action-scale entry.

    Every ``g1_action_scale`` entry names the motor type it was derived from, so
    the effort limit can be recovered from the expression itself instead of
    duplicating a 29-entry table.
    """
    limits: list[float] = []
    for expr in expressions(array_body(text, "g1_action_scale")):
        match = re.search(r"EFFORT_LIMIT_(\w+)", expr)
        if match is None:
            raise SystemExit(f"No EFFORT_LIMIT_* in action-scale entry: {expr!r}")
        key = f"EFFORT_LIMIT_{match.group(1)}"
        if key not in names:
            raise SystemExit(f"Unknown motor constant {key}")
        limits.append(names[key])
    return limits


def joint_names_of(text: str) -> list[str]:
    """Joint names in MuJoCo order, taken from the ``default_angles`` comments."""
    names: list[str] = []
    for comment in line_comments(array_body(text, "default_angles")):
        name = re.sub(r"\s*[（(].*$", "", comment).strip()
        names.append(name or f"joint_{len(names)}")
    return names


# --------------------------------------------------------------------------
# G1 joint limits, so the encoder path needs no SONIC checkout at all.
# --------------------------------------------------------------------------
def _limits_from_mjcf(path: Path) -> tuple[list[str], list[float], list[float]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    names, lower, upper = [], [], []
    for joint in root.find("worldbody").iter("joint"):
        if joint.get("type") == "free":
            continue
        names.append(joint.get("name"))
        lo, hi = (float(value) for value in joint.get("range").split())
        lower.append(lo)
        upper.append(hi)
    return names, lower, upper


def _limits_from_urdf(path: Path) -> dict[str, tuple[float, float]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is not None:
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def joint_limits(sonic_repo: Path, expected_order: list[str]) -> dict[str, list[float]]:
    """Joint limits in MuJoCo order, cross-checked against the URDF."""
    mjcf = sonic_repo / MJCF_RELATIVE
    urdf = sonic_repo / URDF_RELATIVE
    if not mjcf.exists():
        raise SystemExit(f"G1 MJCF not found: {mjcf}")

    names, lower, upper = _limits_from_mjcf(mjcf)
    if len(names) != 29 or set(names) != set(expected_order):
        raise SystemExit("G1 MJCF joint list does not match the header's joint names")

    # The MJCF is the simulator's own model, but confirm it agrees with the URDF
    # the deployment validates against; a mismatch means one of them drifted.
    if urdf.exists():
        urdf_limits = _limits_from_urdf(urdf)
        for name, lo, hi in zip(names, lower, upper, strict=True):
            if name not in urdf_limits:
                raise SystemExit(f"{name} present in MJCF but missing from the URDF")
            ulo, uhi = urdf_limits[name]
            if abs(ulo - lo) > 1e-3 or abs(uhi - hi) > 1e-3:
                raise SystemExit(
                    f"MJCF and URDF limits differ for {name}: "
                    f"MJCF [{lo}, {hi}] vs URDF [{ulo}, {uhi}]"
                )

    order = {name: index for index, name in enumerate(expected_order)}
    lower_out = [0.0] * 29
    upper_out = [0.0] * 29
    for name, lo, hi in zip(names, lower, upper, strict=True):
        lower_out[order[name]] = lo
        upper_out[order[name]] = hi
    return {"lower": lower_out, "upper": upper_out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sonic-repo",
        type=Path,
        default=None,
        help="Path to the GR00T-WholeBodyControl checkout. Defaults to $SONIC_REPO.",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.sonic_repo is not None:
        sonic_repo = args.sonic_repo.expanduser().resolve()
    else:
        import sys

        sys.path.insert(0, str(ROOT / "g1demo"))
        from paths import sonic_repo as resolve_sonic_repo  # type: ignore

        sonic_repo = resolve_sonic_repo()

    header = sonic_repo / HEADER_RELATIVE
    if not header.exists():
        raise SystemExit(f"SONIC header not found: {header}")

    text = header.read_text(encoding="utf-8")
    names = scalar_constants(text)
    joint_names = joint_names_of(text)
    limits = joint_limits(sonic_repo, joint_names)

    payload = {
        "_provenance": {
            "source_repository": "https://github.com/NVlabs/GR00T-WholeBodyControl",
            "source_file": str(HEADER_RELATIVE).replace("\\", "/"),
            "source_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
            "limits_source_file": str(MJCF_RELATIVE).replace("\\", "/"),
            "limits_source_sha256": hashlib.sha256(
                (sonic_repo / MJCF_RELATIVE).read_bytes()
            ).hexdigest(),
            "urdf_cross_check_file": str(URDF_RELATIVE).replace("\\", "/"),
            "generator": "tools/extract_sonic_params.py",
            "note": (
                "Joint orders, default standing pose, action scales, PD gains and joint "
                "limits for the Unitree G1 29-DoF SONIC policy. Values are evaluated from "
                "the header's own C++ expressions and the shipped MJCF; do not hand-edit."
            ),
        },
        "num_joints": 29,
        # Index i is a MuJoCo(URDF) joint index; the value is its IsaacLab index.
        "isaaclab_to_mujoco": int_array(text, "isaaclab_to_mujoco"),
        # Index i is an IsaacLab joint index; the value is its MuJoCo(URDF) index.
        # (The header calls this `mujoco_to_isaaclab`.)
        "mujoco_to_isaaclab": int_array(text, "mujoco_to_isaaclab"),
        "default_angles": evaluate_array(text, "default_angles", names),
        "action_scale": evaluate_array(text, "g1_action_scale", names),
        "kp": evaluate_array(text, "kps", names),
        "kd": evaluate_array(text, "kds", names),
        "effort_limit": effort_limits(text, names),
        "joint_names": joint_names,
        "joint_lower": limits["lower"],
        "joint_upper": limits["upper"],
    }

    # Cross-checks: the two permutations must be exact inverses of each other.
    forward = payload["isaaclab_to_mujoco"]
    inverse = payload["mujoco_to_isaaclab"]
    if sorted(forward) != list(range(29)) or sorted(inverse) != list(range(29)):
        raise SystemExit("Joint permutations are not permutations of 0..28")
    for mujoco_index, isaac_index in enumerate(forward):
        if inverse[isaac_index] != mujoco_index:
            raise SystemExit(
                f"Permutations disagree at mujoco index {mujoco_index}"
            )
    for key in ("default_angles", "action_scale", "kp", "kd", "effort_limit",
                "joint_lower", "joint_upper"):
        if len(payload[key]) != 29:
            raise SystemExit(f"{key} has {len(payload[key])} entries, expected 29")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"  source sha256: {payload['_provenance']['source_sha256']}")
    print(f"  action_scale : min {min(payload['action_scale']):.4f} "
          f"max {max(payload['action_scale']):.4f}")
    print(f"  kp range     : {min(payload['kp']):.2f} .. {max(payload['kp']):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
