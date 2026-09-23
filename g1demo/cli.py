"""Command line entry points for the G1 FastWAM/SONIC demo.

    python -m g1demo.cli download-sonic            # fetch the SONIC ONNX graphs
    python -m g1demo.cli extract-params            # regenerate constants from SONIC
    python -m g1demo.cli verify                    # checks that need no physics
    python -m g1demo.cli demo --motion standing    # closed-loop simulation
    python -m g1demo.cli gain-sweep                # plant gain calibration
    python -m g1demo.cli bridge-export             # clip for NVIDIA's C++ simulator
    python -m g1demo.cli prepare-stage2             # encode one recorded episode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .contract import contract
from .paths import ARTIFACTS, ROOT
from .policy import ScriptedReference
from .sim.g1_mujoco import (
    GAINS_NATIVE,
    GAINS_SONIC,
    SOURCE_MENAGERIE,
    SOURCE_NVIDIA,
    G1Sim,
)
from .sonic.decoder import INPUT_DIM as DECODER_INPUT_DIM
from .sonic.decoder import NUM_FRAMES, SonicDecoder
from .sonic.encoder import INPUT_DIM as ENCODER_INPUT_DIM
from .sonic.encoder import TOKEN_DIM, SonicEncoder, pack_observation
from .sonic_params import (
    action_scale,
    default_angles,
    joint_lower,
    joint_names,
    joint_upper,
    kd,
    kp,
    measured_to_isaaclab_relative,
    q_target_from_action,
    validate_against_contract,
)

MOTIONS = ("standing", "arms_up", "squat", "wave")


# ---------------------------------------------------------------------------
# download-sonic
# ---------------------------------------------------------------------------
def cmd_download_sonic(args) -> int:
    from .download import download_sonic

    return download_sonic(force=args.force, planner_out=args.planner_out)


# ---------------------------------------------------------------------------
# extract-params
# ---------------------------------------------------------------------------
def cmd_extract_params(args) -> int:
    """Run tools/extract_sonic_params.py, forwarding the arguments it accepts."""
    sys.path.insert(0, str(ROOT / "tools"))
    import extract_sonic_params as extractor  # type: ignore[import-not-found]

    argv: list[str] = []
    if args.sonic_repo:
        argv += ["--sonic-repo", str(args.sonic_repo)]
    if args.out:
        argv += ["--out", str(args.out)]

    original = sys.argv
    sys.argv = ["extract_sonic_params.py", *argv]
    try:
        return extractor.main()
    finally:
        sys.argv = original


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def cmd_verify(args) -> int:
    """Check the contract, the SONIC chain and the action convention.

    Needs neither a GPU, nor upstream FastWAM weights, nor physics. This is the
    cheapest way to confirm a fresh checkout is wired up correctly.
    """
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))

    c = contract()

    # -- contract ---------------------------------------------------------
    expected_width = 29 + sum(c.hand_sizes.values()) + c.root_size
    check("contract width is 29 joints + hands + root",
          c.action_dim == expected_width, f"action_dim={c.action_dim}")

    check("contract joint names match the SONIC model",
          set(c.joint_names) == set(joint_names()), f"{c.num_joints} joints")
    validate_against_contract(c.joint_names)

    in_range = (default_angles() > joint_lower()) & (default_angles() < joint_upper())
    check("joint limits bracket the default pose",
          bool(in_range.all()), f"{int(in_range.sum())}/29")

    gains_ok = (
        np.isfinite(kp()).all()
        and np.isfinite(kd()).all()
        and (kp() > 0).all()
        and (action_scale() > 0).all()
    )
    detail = (
        f"kp {kp().min():.1f}..{kp().max():.1f}, "
        f"scale {action_scale().min():.3f}..{action_scale().max():.3f}"
    )
    check("PD gains and action scales are finite and positive", bool(gains_ok), detail)

    # -- encoder ----------------------------------------------------------
    flat = c.flat_indices_for(joint_names())
    chunk = np.zeros((60, c.action_dim))
    chunk[:, flat] = default_angles()

    obs, _ = pack_observation(chunk, [1, 0, 0, 0], 0.0, frame=0)
    check("encoder observation width",
          obs.shape == (1, ENCODER_INPUT_DIM), f"{obs.shape} vs [1,{ENCODER_INPUT_DIM}]")
    check("encoder input is finite", bool(np.isfinite(obs).all()))
    check("encoder mode slot encodes mode 0 as zeros",
          bool(np.allclose(obs[0, :4], 0.0)), f"{obs[0, :4]}")

    identity_6d = [1, 0, 0, 1, 0, 0]
    check("orientation block is identity for a level robot holding world yaw",
          bool(np.allclose(obs[0, 584:590], identity_6d, atol=1e-6)),
          f"{np.round(obs[0, 584:590], 4)}")

    encoder = SonicEncoder()
    packet = encoder.encode(chunk, [1, 0, 0, 0], 0.0, frame=0)
    token = packet["token"]
    check("token shape", token.shape == (TOKEN_DIM,), f"{token.shape}")
    check("token is FSQ-quantized",
          bool(np.allclose(token * 16, np.round(token * 16), atol=1e-6)))

    # The hands must not reach the graph.
    altered = chunk.copy()
    altered[:, c["left_end_effector"].slice] = 0.73
    altered[:, c["right_end_effector"].slice] = -0.42
    second = encoder.encode(altered, [1, 0, 0, 0], 0.0, frame=0)
    hands_bypass = (
        np.array_equal(token, second["token"])
        and np.allclose(second["left_hand"], 0.73)
        and np.allclose(second["right_hand"], -0.42)
    )
    check("hand commands bypass the encoder", bool(hands_bypass))

    # A short chunk must be rejected, never silently padded.
    try:
        pack_observation(chunk[:45], [1, 0, 0, 0], 0.0, frame=0)
        check("short reference is rejected", False, "no error raised")
    except ValueError as error:
        check("short reference is rejected", True, f"{str(error)[:48]}...")

    # -- decoder ----------------------------------------------------------
    decoder = SonicDecoder()
    history = measured_to_isaaclab_relative(np.tile(default_angles(), (NUM_FRAMES, 1)))
    empty = np.zeros((NUM_FRAMES, 29))
    raw = decoder.decode(
        token=token,
        joint_pos_history=history,
        joint_vel_history=empty,
        last_action_history=empty,
        base_ang_vel_history=np.zeros((NUM_FRAMES, 3)),
        base_quat_wxyz=[1, 0, 0, 0],
    )
    check("decoder output shape", raw.shape == (29,), f"{raw.shape}")
    check("decoder output is finite", bool(np.isfinite(raw).all()),
          f"range {raw.min():.3f}..{raw.max():.3f}")

    target = q_target_from_action(raw)
    saturated = int(((target < joint_lower()) | (target > joint_upper())).sum())
    drift = float(np.abs(target - default_angles()).max())
    check("default-pose reference stays inside joint limits", saturated == 0,
          f"{saturated}/29 saturated, max |dq|={drift:.3f} rad")

    # The action -> target convention must reproduce the C++ expression exactly.
    stored = json.loads((ROOT / "configs/g1_sonic_params.json").read_text())
    permutation = np.asarray(stored["isaaclab_to_mujoco"])
    manual = default_angles() + raw[permutation] * action_scale()
    check("q_target matches the C++ convention elementwise",
          bool(np.allclose(target, manual)))

    # -- report -----------------------------------------------------------
    passed = sum(1 for _, ok, _ in checks if ok)
    width = max(len(name) for name, _, _ in checks)
    print(f"contract + SONIC chain verification  ({passed}/{len(checks)} passed)\n")
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        suffix = f"   [{detail}]" if detail else ""
        print(f"  {mark}  {name:<{width}}{suffix}")

    report = {
        "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in checks],
        "passed": passed,
        "total": len(checks),
        "action_dim": c.action_dim,
        "encoder_input_dim": ENCODER_INPUT_DIM,
        "decoder_input_dim": DECODER_INPUT_DIM,
        "token_dim": TOKEN_DIM,
        "note": (
            "No learned G1 FastWAM checkpoint exists upstream; this verifies the "
            "contract, the pretrained SONIC encoder/decoder and the action-to-target "
            "convention."
        ),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0 if passed == len(checks) else 1


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> str:
    """Write frames as mp4 when possible, otherwise as a GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path.with_suffix(".mp4"), frames, fps=fps, quality=8)
        return str(path.with_suffix(".mp4"))
    except Exception:
        from PIL import Image

        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(path.with_suffix(".gif"), save_all=True,
                       append_images=images[1:], duration=int(1000 / fps), loop=0)
        return str(path.with_suffix(".gif"))


def _write_plots(records, summary, path: Path, control_hz: int) -> str:
    """Plot what the controller commanded and how the robot responded.

    MuJoCo offscreen rendering needs a GL context, which a headless machine may
    not have. These plots need none, so they are written even when ``--video``
    cannot be.
    """
    # Matplotlib insists on a writable config dir, and the sandbox may deny
    # ~/.config, so keep it beside the run outputs.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.arange(len(records)) / control_hz
    heights = np.array([record.base_height for record in records])
    action_max = np.array([np.abs(record.raw_action).max() for record in records])
    saturated = np.array([record.saturated.sum() for record in records])
    tokens = np.array([record.token for record in records])
    commanded = np.stack([record.q_target for record in records])
    measured = np.stack([record.q_measured for record in records])
    tracking = np.abs(measured - np.clip(commanded, joint_lower(), joint_upper()))

    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    figure.suptitle(
        f"G1 whole-body control: SONIC v1.1 encoder -> 64-D token -> decoder "
        f"({summary.ticks} ticks, gain {summary.action_gain})",
        fontsize=11,
    )

    ax = axes[0][0]
    ax.plot(times, heights, color="tab:blue")
    ax.axhline(0.79, ls="--", lw=0.8, color="grey", label="nominal standing height")
    ax.axhline(0.45, ls=":", lw=0.8, color="red", label="fall threshold")
    ax.set_ylabel("pelvis height [m]")
    ax.set_title("Balance")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot(times, action_max, color="tab:orange", label="max |decoder action|")
    ax.set_ylabel("|action| [normalized]")
    ax.set_title("Decoder output magnitude")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(times, saturated, color="tab:red", lw=0.9, alpha=0.7,
             label="saturated joints")
    ax2.set_ylabel("joints outside limits", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax = axes[1][0]
    ax.plot(times, tracking.mean(axis=1), color="tab:green", label="mean")
    ax.plot(times, tracking.max(axis=1), color="tab:purple", lw=0.8, label="max")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|tracking error| [rad]")
    ax.set_title("Joint tracking error (clamped target vs measured)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    rest = default_angles()
    for index, name in enumerate(joint_names()):
        if np.abs(commanded[:, index] - rest[index]).max() > 0.05:
            ax.plot(times, commanded[:, index], lw=1.0, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("commanded target [rad]")
    ax.set_title("Commanded targets that moved > 0.05 rad")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)

    # A second figure for the token, which is the SONIC latent.
    figure, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(times, np.linalg.norm(tokens, axis=1), color="tab:blue")
    axes[0].set_ylabel("||token||")
    axes[0].set_title("64-D whole-body latent produced by the SONIC encoder")
    axes[0].grid(alpha=0.3)
    axes[1].imshow(tokens.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=(times[0], times[-1], 0, tokens.shape[1]))
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("token dim")
    axes[1].set_title("FSQ token values (32 levels per dimension)")
    figure.tight_layout()
    token_path = path.with_name(path.stem + "_token.png")
    figure.savefig(token_path, dpi=110)
    plt.close(figure)
    return f"{path} and {token_path}"


def cmd_demo(args) -> int:
    from .loop import WholeBodyController, joint_error_table

    free_base = args.mode == "free"
    sim = G1Sim(source=args.source, gains=args.model_gains, free_base=free_base)
    controller = WholeBodyController(action_gain=args.action_gain)
    source = ScriptedReference(
        motion=args.motion,
        horizon=max(60, args.replan_horizon),
        amplitude=args.amplitude,
    )

    frames: list[np.ndarray] = []
    render_every = max(1, int(round(controller.control_hz / (args.video_fps or 25.0))))
    render_error: str | None = None

    def on_step(record, sim_) -> None:
        nonlocal render_error
        if render_error is not None or record.tick % render_every:
            return
        try:
            frames.append(sim_.render(height=args.height, width=args.width))
        except Exception as error:  # no GL context available
            render_error = f"{type(error).__name__}: {error}"

    summary, records = controller.run(
        sim,
        source,
        ticks=args.ticks,
        replan_every=args.replan_every,
        stop_on_fall=free_base and not args.allow_fall,
        on_step=on_step if args.video else None,
    )

    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace,
            tick=np.asarray([r.tick for r in records], dtype=np.int64),
            frame=np.asarray([r.frame for r in records], dtype=np.int64),
            sonic_token=np.stack([r.token for r in records]),
            raw_action=np.stack([r.raw_action for r in records]),
            q_target=np.stack([r.q_target for r in records]),
            q_measured=np.stack([r.q_measured for r in records]),
            base_quat_wxyz=np.stack([r.base_quat_wxyz for r in records]),
            left_hand=np.stack([r.left_hand for r in records]),
            right_hand=np.stack([r.right_hand for r in records]),
        )

    payload = {
        "mode": "free_base" if free_base else "anchored",
        "model_source": args.source,
        "model_gains": args.model_gains,
        "motion": args.motion,
        "amplitude": args.amplitude,
        "controller": "SONIC v1.1 encoder -> 64-D token -> SONIC v1.1 decoder -> PD targets",
        "policy_source": source.name,
        "fastwam_inference": False,
        "fastwam_note": (
            "No G1 FastWAM checkpoint exists upstream; a scripted reference drives "
            "the identical [T, 34] interface."
        ),
        "hands_routed": "bypass encoder and decoder",
        "summary": summary.as_dict(),
        "worst_joints_by_rms_error": [
            {"joint": name, "rms_error_rad": round(error, 4)}
            for name, error in joint_error_table(summary)[:5]
        ],
    }

    if args.plot:
        payload["plot"] = _write_plots(records, summary, Path(args.plot),
                                       controller.control_hz)

    if args.video:
        if frames:
            payload["video"] = _write_video(frames, Path(args.video),
                                            int(args.video_fps or 25.0))
            montage = np.concatenate(
                [frames[0], frames[len(frames) // 2], frames[-1]], axis=1
            )
            from PIL import Image

            snapshot = Path(args.video).with_name(Path(args.video).stem + "_montage.png")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(montage).save(snapshot)
            payload["snapshot"] = str(snapshot)
        else:
            payload["video_error"] = (
                f"offscreen rendering unavailable ({render_error}). MuJoCo needs a GL "
                f"context; try MUJOCO_GL=osmesa on a machine with OSMesa, or a working GPU."
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"mode            {payload['mode']}")
    print(f"model           {args.source}  (gains: {args.model_gains})")
    print(f"motion          {args.motion}  (source: {source.name})")
    print(f"action gain     {summary.action_gain}")
    print(f"ticks           {summary.ticks}  "
          f"({summary.seconds:.2f} s at {controller.control_hz} Hz)")
    print(f"fell            {summary.fell}")
    print(f"base height     min {summary.min_base_height:.3f} m "
          f"-> final {summary.final_base_height:.3f} m")
    print(f"tracking error  mean {summary.mean_joint_tracking_error:.4f} rad, "
          f"max {summary.max_joint_tracking_error:.4f} rad")
    print(f"saturation      {100 * summary.saturation_fraction:.2f}% of joint-ticks")
    print(f"max |action|    {summary.max_abs_raw_action:.3f}")
    if payload["worst_joints_by_rms_error"]:
        worst = ", ".join(
            f"{row['joint']}={row['rms_error_rad']:.3f}"
            for row in payload["worst_joints_by_rms_error"]
        )
        print(f"worst joints    {worst}")
    for key in ("plot", "video", "snapshot", "video_error"):
        if key in payload:
            print(f"{key:<15} {payload[key]}")
    print(f"report          {out}")
    return 0


# ---------------------------------------------------------------------------
# gain-sweep
# ---------------------------------------------------------------------------
def cmd_gain_sweep(args) -> int:
    """Calibrate the effective loop gain of the MuJoCo plant.

    NVIDIA's convention is gain 1.0. This sweep exists because the MuJoCo position
    servos are stiffer than the IsaacLab actuators the policy was trained against,
    so 1.0 over-drives the balance loop. See docs/sim2sim_gap.md.
    """
    from .loop import WholeBodyController

    rows = []
    for gain in args.action_gains:
        sim = G1Sim(source=args.source, gains=args.model_gains,
                    free_base=args.mode == "free")
        controller = WholeBodyController(action_gain=gain)
        summary, _ = controller.run(
            sim, ScriptedReference(motion=args.motion, horizon=300), ticks=args.ticks
        )
        rows.append({"action_gain": gain, **summary.as_dict()})
        print(f"  gain {gain:<5} ticks={summary.ticks:>4} fell={str(summary.fell):<5} "
              f"h_final={summary.final_base_height:.3f} "
              f"err_mean={summary.mean_joint_tracking_error:.4f} "
              f"sat={100 * summary.saturation_fraction:.1f}%")

    best = max(rows, key=lambda row: (not row["fell"], row["ticks"],
                                      -row["mean_joint_tracking_error_rad"]))
    print(f"\n  most stable gain: {best['action_gain']}")
    out = ARTIFACTS / "gain_sweep.json"
    out.write_text(json.dumps(
        {"motion": args.motion, "mode": args.mode, "rows": rows}, indent=2) + "\n")
    print(f"  wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# bridge-export
# ---------------------------------------------------------------------------
def cmd_bridge(args) -> int:
    """Export an action chunk as a SONIC reference clip for NVIDIA's C++ simulator."""
    from .bridge import export

    if args.actions:
        actions = np.load(args.actions)
        origin = args.origin or Path(args.actions).stem
    else:
        actions = ScriptedReference(motion=args.motion, horizon=args.horizon).actions()
        origin = args.origin or f"scripted:{args.motion}"
        # Keep the exact chunk that produced the clip, so the export is auditable.
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / "source_actions.npy", actions)

    out = export(actions, args.out, args.root_height, origin)
    print(f"wrote SONIC reference clip to {out}")
    print(f"  source: {origin}  shape={list(np.shape(actions))}  fps=50")
    print("  next: run NVIDIA's deployment against this clip:")
    print("        cd $SONIC_REPO && bash install_scripts/install_mujoco_sim.sh")
    print("        python gear_sonic/scripts/run_sim_loop.py   # terminal 1")
    print("        cd gear_sonic_deploy && bash deploy.sh sim  # terminal 2")
    return 0


def cmd_prepare_stage2(args) -> int:
    """Turn one synchronized episode into SONIC-latent supervision."""
    from .stage2_data import prepare_episode

    prepared = prepare_episode(args.episode, args.out)
    with np.load(prepared, allow_pickle=False) as data:
        print(f"wrote {prepared} ({len(data['sonic_token'])} labeled frames)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g1demo",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-sonic", help="fetch the SONIC v1.1 ONNX graphs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--planner-out", type=Path, default=None)
    p.set_defaults(func=cmd_download_sonic)

    p = sub.add_parser("extract-params",
                       help="regenerate configs/g1_sonic_params.json")
    p.add_argument("--sonic-repo", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_extract_params)

    p = sub.add_parser("verify", help="verify the contract and the SONIC chain")
    p.add_argument("--out", type=Path, default=ARTIFACTS / "verify.json")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("demo", help="run the closed-loop MuJoCo demo")
    p.add_argument("--motion", choices=MOTIONS, default="standing")
    p.add_argument("--mode", choices=("free", "anchored"), default="free",
                   help="free base (whole-body balance) or pelvis welded to the world")
    p.add_argument("--source", choices=(SOURCE_MENAGERIE, SOURCE_NVIDIA),
                   default=SOURCE_MENAGERIE,
                   help="menagerie: the real G1 with meshes; nvidia: mesh-free rebuild")
    p.add_argument("--model-gains", choices=(GAINS_SONIC, GAINS_NATIVE),
                   default=GAINS_SONIC,
                   help="sonic: the deployment's kp/kd; native: the model's own servos")
    p.add_argument("--ticks", type=int, default=250)
    p.add_argument("--action-gain", type=float, default=1.0,
                   help="1.0 is NVIDIA's convention; see docs/sim2sim_gap.md")
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--replan-horizon", type=int, default=100,
                   help="frames per requested action chunk")
    p.add_argument("--replan-every", type=int, default=None,
                   help="re-query the policy every N ticks (default: when a chunk runs out)")
    p.add_argument("--allow-fall", action="store_true", help="keep running after a fall")
    p.add_argument("--plot", type=Path, default=ARTIFACTS / "run.png",
                   help="write trajectory plots here")
    p.add_argument("--video", type=Path, default=None,
                   help="write mp4/gif here (needs a working GL context)")
    p.add_argument("--video-fps", type=float, default=25.0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out", type=Path, default=ARTIFACTS / "run.json")
    p.add_argument("--trace", type=Path, default=None,
                   help="write per-tick token, actions, targets and state to .npz")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("gain-sweep", help="calibrate the effective loop gain")
    p.add_argument("--motion", choices=MOTIONS, default="standing")
    p.add_argument("--mode", choices=("free", "anchored"), default="free")
    p.add_argument("--ticks", type=int, default=250)
    p.add_argument("--action-gains", type=float, nargs="+",
                   default=[1.0, 0.75, 0.5, 0.35, 0.25],
                   help="decoder-output gains to sweep")
    p.add_argument("--source", choices=(SOURCE_MENAGERIE, SOURCE_NVIDIA),
                   default=SOURCE_MENAGERIE)
    p.add_argument("--model-gains", choices=(GAINS_SONIC, GAINS_NATIVE),
                   default=GAINS_SONIC)
    p.set_defaults(func=cmd_gain_sweep)

    p = sub.add_parser("bridge-export",
                       help="export a SONIC reference clip for NVIDIA's C++ simulator")
    p.add_argument("--actions", type=Path, default=None, help="[T, 34] .npy chunk")
    p.add_argument("--motion", choices=MOTIONS, default="standing",
                   help="used when --actions is omitted")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--root-height", type=float, default=0.793)
    p.add_argument("--origin", default=None, help="provenance label")
    p.add_argument("--out", type=Path, default=ARTIFACTS / "sonic_reference")
    p.set_defaults(func=cmd_bridge)

    p = sub.add_parser("prepare-stage2",
                       help="validate an episode and encode SONIC token targets")
    p.add_argument("--episode", type=Path, required=True, help="synchronized .npz input")
    p.add_argument("--out", type=Path, required=True, help="prepared .npz output")
    p.set_defaults(func=cmd_prepare_stage2)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
