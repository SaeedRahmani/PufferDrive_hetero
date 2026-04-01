"""
Policy rollout visualization — runs a trained policy on a REAL Waymo scenario
and saves an animated GIF.

Shows: full road map, ground-truth trajectories (dashed), policy rollout
       trajectories (solid), agent bounding boxes at current timestep.

Architecture is auto-detected from the checkpoint; obs is sliced to match.

Usage (from guided_original/):
    .venv/bin/python rollout_video.py
    .venv/bin/python rollout_video.py \\
        --map-dir /data/lynx/saeani/pufferdrive_guided_full/pufferlib/resources/drive/binaries/training \\
        --checkpoint pufferlib/resources/drive/pufferdrive_weights.pt \\
        --output rollout_video.gif --fps 10
"""

import os
import sys
import argparse
import copy
import types
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
import pufferlib.ocean.torch as ocean_torch
import pufferlib.models


# ──────────────────────────────────────────────────────────────────────────────
# Defaults  (override via CLI args)
# ──────────────────────────────────────────────────────────────────────────────
TRAINING_MAP_DIR = (
    "/data/lynx/saeani/pufferdrive_guided_full"
    "/pufferlib/resources/drive/binaries/training"
)
DEFAULT_CHECKPOINT = "pufferlib/resources/drive/pufferdrive_weights.pt"
DEFAULT_OUTPUT     = "rollout_video.gif"
EPISODE_STEPS      = 91
FPS                = 10
NUM_AGENTS         = 128
CANVAS_PX          = 800       # canvas width & height in pixels
DPI                = 100


# ──────────────────────────────────────────────────────────────────────────────
# Architecture auto-detection
# ──────────────────────────────────────────────────────────────────────────────
def infer_arch(sd):
    ego_features      = sd["policy.ego_encoder.0.weight"].shape[1]
    input_size        = sd["policy.ego_encoder.0.weight"].shape[0]
    hidden_policy     = sd["policy.shared_embedding.1.weight"].shape[0]
    partner_features  = sd["policy.partner_encoder.0.weight"].shape[1]
    road_after_onehot = sd["policy.road_encoder.0.weight"].shape[1]
    lstm_hidden       = sd["lstm.weight_hh_l0"].shape[0] // 4
    lstm_input        = sd["lstm.weight_ih_l0"].shape[1]
    return dict(
        ego_features=ego_features, input_size=input_size,
        hidden_policy=hidden_policy, partner_features=partner_features,
        road_features=road_after_onehot - 6,
        lstm_hidden=lstm_hidden, lstm_input=lstm_input,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Env + policy construction
# ──────────────────────────────────────────────────────────────────────────────
def build_env(map_dir):
    return Drive(
        map_dir=map_dir,
        num_maps=1,
        num_agents=NUM_AGENTS,
        episode_length=EPISODE_STEPS,
        init_steps=0,
        resample_frequency=EPISODE_STEPS,
        use_guided_autonomy=1,
        use_guidance_observations=0,
        guidance_speed_weight=0.5,
        guidance_heading_weight=0.5,
        prep_human_data=False,
        save_data_to_disk=False,
        reward_vehicle_collision=-0.5,
        reward_offroad_collision=-0.5,
    )


def build_policy(env, checkpoint, device):
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    arch = infer_arch(sd)
    print(f"  Arch: ego={arch['ego_features']} input_size={arch['input_size']} "
          f"hidden={arch['hidden_policy']} lstm={arch['lstm_hidden']}")

    proxy = types.SimpleNamespace(
        single_observation_space=env.single_observation_space,
        single_action_space=env.single_action_space,
        max_partner_objects=env.max_partner_objects,
        partner_features=arch["partner_features"],
        max_road_objects=env.max_road_objects,
        road_features=arch["road_features"],
        ego_features=arch["ego_features"],
    )
    base   = ocean_torch.Drive(proxy, input_size=arch["input_size"],
                               hidden_size=arch["hidden_policy"])
    policy = pufferlib.models.LSTMWrapper(proxy, base,
                                          input_size=arch["lstm_input"],
                                          hidden_size=arch["lstm_hidden"])
    policy = policy.to(device)
    policy.load_state_dict(sd)
    policy.eval()
    policy._ckpt_ego = arch["ego_features"]
    policy._env_ego  = binding.EGO_FEATURES_CLASSIC
    return policy


def adapt_obs(obs_np, policy):
    ce, ee = policy._ckpt_ego, policy._env_ego
    if ce == ee:
        return obs_np
    return np.concatenate([obs_np[:, :ce], obs_np[:, ee:]], axis=-1)


def init_lstm(n, h, device):
    return {"lstm_h": torch.zeros(n, h, device=device),
            "lstm_c": torch.zeros(n, h, device=device)}


def sample_actions(logits, env):
    if env.single_action_space.__class__.__name__ == "Box":
        return logits[0].detach().cpu().numpy()
    if isinstance(logits, (list, tuple)):
        return torch.cat([lg.argmax(-1).unsqueeze(-1) for lg in logits], -1).cpu().numpy()
    return logits.argmax(-1).unsqueeze(-1).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Rollout — collect (obs, state) at every step
# ──────────────────────────────────────────────────────────────────────────────
def rollout(env, policy, device):
    obs_np, _ = env.reset()
    lstm_state = init_lstm(env.num_agents, policy.hidden_size, device)
    frame_states = [env.get_global_agent_state()]
    for _ in range(EPISODE_STEPS - 1):
        obs_t = torch.tensor(adapt_obs(obs_np, policy), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_t, lstm_state)
        obs_np, _, _, _, _ = env.step(sample_actions(logits, env))
        frame_states.append(env.get_global_agent_state())
    return frame_states


# ──────────────────────────────────────────────────────────────────────────────
# Map helpers
# ──────────────────────────────────────────────────────────────────────────────
def build_road_lc(road, cx, cy):
    """LineCollection for all road edge segments."""
    segs = []
    pt_idx = 0
    for seg_len in road["lengths"]:
        xs = road["x"][pt_idx: pt_idx + seg_len] - cx
        ys = road["y"][pt_idx: pt_idx + seg_len] - cy
        if seg_len >= 2:
            segs.append(np.column_stack([xs, ys]))
        pt_idx += seg_len
    return LineCollection(segs, colors="#e8e8e8", linewidths=0.6, alpha=0.9, zorder=1)


def box_corners(x, y, length, width, heading):
    """Return (4,2) array of bounding-box corners for a vehicle."""
    c, s = np.cos(heading), np.sin(heading)
    hw, hl = width / 2, length / 2
    corners = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ])
    rot = np.array([[c, -s], [s, c]])
    return (corners @ rot.T) + np.array([x, y])


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame renderer
# ──────────────────────────────────────────────────────────────────────────────
def render_frame(ax, road_lc, cx, cy, xlim, ylim,
                 gt_x, gt_y, gt_valid,        # (N, T)
                 rollout_x, rollout_y,         # (N, step+1)
                 agent_state, step, total, n_agents):

    ax.cla()
    ax.set_facecolor("#1c1c2e")

    # Road
    ax.add_collection(copy.copy(road_lc))

    # Ground-truth trajectories (full, dashed, grey)
    gt_segs = []
    for i in range(n_agents):
        mask = gt_valid[i] > 0
        if mask.sum() < 2:
            continue
        xs = gt_x[i, mask] - cx
        ys = gt_y[i, mask] - cy
        gt_segs.append(np.column_stack([xs, ys]))
    if gt_segs:
        lc_gt = LineCollection(gt_segs, colors="#888888", linewidths=0.7,
                               linestyles="dashed", alpha=0.55, zorder=2)
        ax.add_collection(lc_gt)

    # Policy rollout trails — skip teleport/respawn jumps
    # At 30 m/s and 0.1 s dt an agent moves at most ~3 m/step; 15 m is a safe
    # threshold to detect goal-respawn teleports without clipping normal motion.
    TELEPORT_THRESHOLD = 15.0
    cmap = plt.cm.get_cmap("tab20")
    trail_segs, trail_colors = [], []
    for i in range(n_agents):
        xs = rollout_x[i] - cx
        ys = rollout_y[i] - cy
        if len(xs) < 2:
            continue
        pts = np.column_stack([xs, ys])
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        valid_segs = dists < TELEPORT_THRESHOLD
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        T = segs.shape[0]
        color = cmap(i % 20)
        alphas = np.linspace(0.2, 0.9, T)
        for seg, a, ok in zip(segs, alphas, valid_segs):
            if ok:
                trail_segs.append(seg)
                trail_colors.append((*color[:3], a))
    if trail_segs:
        ax.add_collection(LineCollection(trail_segs, colors=trail_colors,
                                         linewidths=1.0, zorder=3))

    # Bounding boxes at current step
    ax_xs = agent_state["x"] - cx
    ax_ys = agent_state["y"] - cy
    headings = agent_state["heading"]
    lengths  = agent_state["length"]
    widths   = agent_state["width"]

    boxes = []
    box_colors = []
    for i in range(n_agents):
        corners = box_corners(ax_xs[i], ax_ys[i],
                              lengths[i], widths[i], headings[i])
        boxes.append(Polygon(corners))
        c = cmap(i % 20)
        box_colors.append(c)

    pc = PatchCollection(boxes, facecolors=box_colors, edgecolors="white",
                         linewidths=0.4, alpha=0.85, zorder=4)
    ax.add_collection(pc)

    # Heading arrows (short)
    arrow_len = max((xlim[1] - xlim[0]) * 0.015, 0.5)
    u = np.cos(headings) * arrow_len
    v = np.sin(headings) * arrow_len
    ax.quiver(ax_xs, ax_ys, u, v,
              color=np.array([cmap(i % 20) for i in range(n_agents)]),
              angles="xy", scale_units="xy", scale=1,
              width=0.002, headwidth=4, headlength=5, zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#444444")

    # Legend
    legend_elems = [
        mpatches.Patch(facecolor="#888888", linestyle="--", label="Ground truth"),
        mpatches.Patch(facecolor="#4fc3f7", label="Policy rollout"),
        mpatches.Patch(facecolor="#e8e8e8", label="Road edges"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7,
              facecolor="#2a2a3e", edgecolor="#555555",
              labelcolor="white", markerscale=0.8)

    ax.set_title(f"Step {step+1}/{total}  ·  {n_agents} agents",
                 color="white", fontsize=10, pad=5)


# ──────────────────────────────────────────────────────────────────────────────
# Full render pipeline
# ──────────────────────────────────────────────────────────────────────────────
def make_gif(frame_states, road, gt_traj, output_path, fps=FPS):
    from PIL import Image

    n_agents = len(frame_states[0]["x"])
    total    = len(frame_states)

    # Map bounds from road edges
    rx, ry = road["x"], road["y"]
    cx, cy = rx.mean(), ry.mean()
    pad = max(rx.ptp(), ry.ptp()) * 0.05 + 5
    xlim = (rx.min() - cx - pad, rx.max() - cx + pad)
    ylim = (ry.min() - cy - pad, ry.max() - cy + pad)

    # Pre-built road LineCollection
    road_lc = build_road_lc(road, cx, cy)

    # Ground truth arrays
    gt_x     = gt_traj["x"].squeeze(1)      # (N, T)
    gt_y     = gt_traj["y"].squeeze(1)
    gt_valid = gt_traj["valid"].squeeze(1)

    # Rollout position arrays (accumulated per step)
    rollout_x = np.array([s["x"] for s in frame_states]).T  # (N, total)
    rollout_y = np.array([s["y"] for s in frame_states]).T

    # Figure sized to be square, matching map aspect
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    max_span = max(x_span, y_span)
    fw = CANVAS_PX / DPI * (x_span / max_span)
    fh = CANVAS_PX / DPI * (y_span / max_span)
    fw, fh = max(fw, 4.0), max(fh, 4.0)

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#1c1c2e",
                           dpi=DPI)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.06)

    pil_frames = []
    for step in range(total):
        render_frame(
            ax, road_lc, cx, cy, xlim, ylim,
            gt_x, gt_y, gt_valid,
            rollout_x[:, :step+1], rollout_y[:, :step+1],
            frame_states[step], step, total, n_agents,
        )
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        pil_frames.append(Image.fromarray(rgba[:, :, :3]))
        if (step + 1) % 10 == 0:
            print(f"  {step+1}/{total} frames", flush=True)

    plt.close(fig)

    # Save
    if output_path.lower().endswith(".mp4"):
        output_path = output_path[:-4] + ".gif"

    duration_ms = int(1000 / fps)
    pil_frames[0].save(output_path, save_all=True,
                       append_images=pil_frames[1:],
                       duration=duration_ms, loop=0, optimize=False)
    print(f"\nSaved → {output_path}  ({len(pil_frames)} frames, {fps} fps)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--map-dir",    default=TRAINING_MAP_DIR)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--fps",        type=int,  default=FPS)
    parser.add_argument("--cpu",        action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device : {device}")
    print(f"Map dir: {args.map_dir}")
    print(f"Ckpt   : {args.checkpoint}")

    env = build_env(args.map_dir)

    print("Loading policy...")
    policy = build_policy(env, args.checkpoint, device)
    print(f"  num_agents={env.num_agents}")

    # Static map data
    env.reset()
    road   = env.get_road_edge_polylines()
    gt     = env.get_ground_truth_trajectories()
    rx, ry = road["x"], road["y"]
    print(f"Road span: {rx.ptp():.1f} m × {ry.ptp():.1f} m  "
          f"({len(road['lengths'])} segments)")
    valid_agents = (gt["valid"].squeeze(1).sum(axis=1) > 0).sum()
    print(f"Agents with valid GT trajectories: {valid_agents}/{env.num_agents}")

    # Rollout
    print("Rolling out policy...")
    env.reset()
    frame_states = rollout(env, policy, device)
    print(f"Collected {len(frame_states)} frames")

    # Render
    print("Rendering GIF...")
    make_gif(frame_states, road, gt, args.output, fps=args.fps)

    env.close()


if __name__ == "__main__":
    main()


