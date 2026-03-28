"""
Rollout for pretty-totem-2 (wandb run eodvv1ay, project hetero-style).
Checkpoint: /data/lynx/saeani/pufferdrive_hetero_v2/experiments/puffer_drive_eodvv1ay.pt

Run from pufferdrive_hetero_v2/:
    .venv/bin/python rollout_pretty_totem_2.py
"""

import os
import sys
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

# Run from the hetero_v2 directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
import pufferlib.ocean.torch as ocean_torch
import pufferlib.models

CHECKPOINT = "/data/lynx/saeani/pufferdrive_hetero_v2/experiments/puffer_drive_eodvv1ay.pt"
MAP_DIR    = "pufferlib/resources/drive/binaries/training"
OUTPUT     = "/users/saeani/src/guided_original/rollout_pretty_totem_2.gif"
# This map has 4 valid controlled agents per world.
# Setting num_agents=4 gives exactly 1 parallel world so get_global_agent_state()
# returns a clean single-world rollout. Higher values create N duplicate worlds
# all on the same map, stacked on top of each other in the visualization.
NUM_AGENTS  = 4
EPISODE_STEPS = 91
FPS = 10
CANVAS_PX = 800
DPI = 100


def build_env():
    return Drive(
        map_dir=MAP_DIR,
        num_maps=1,
        num_agents=NUM_AGENTS,
        episode_length=EPISODE_STEPS,
        init_steps=0,
        resample_frequency=99999,   # no respawning during rollout
        use_guided_autonomy=1,
        use_guidance_observations=1,
        guidance_dropout_mode="max",
        guidance_dropout_prob=0.95,
        guidance_speed_weight=0.5,
        guidance_heading_weight=0.5,
        prep_human_data=False,
        save_data_to_disk=False,
        reward_vehicle_collision=-0.5,
        reward_offroad_collision=-0.5,
    )


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


def build_policy(env, device):
    sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
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
    return policy


def rollout(env, policy, device):
    """Returns (frame_states, done_step) sliced to world-0 agents only.
    All parallel worlds run the same map at the same positions; we only
    visualize the first world to avoid stacked duplicates."""
    w0_start = int(env.agent_offsets[0])   # always 0
    w0_end   = int(env.agent_offsets[1])   # agents in world 0
    n_total  = env.num_agents
    h        = policy.hidden_size
    obs_np, _ = env.reset()
    state = {"lstm_h": torch.zeros(n_total, h, device=device),
             "lstm_c": torch.zeros(n_total, h, device=device)}

    def w0_state():
        st = env.get_global_agent_state()
        return {k: v[w0_start:w0_end] for k, v in st.items()}

    frame_states = [w0_state()]
    n_w0 = w0_end - w0_start
    done_step  = np.full(n_w0, EPISODE_STEPS, dtype=int)
    ever_done  = np.zeros(n_w0, dtype=bool)

    for step in range(EPISODE_STEPS - 1):
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_t, state)
        if isinstance(logits, (list, tuple)):
            actions = torch.cat([lg.argmax(-1).unsqueeze(-1) for lg in logits], -1).cpu().numpy()
        else:
            actions = logits.argmax(-1).unsqueeze(-1).cpu().numpy()
        obs_np, _, terminated, truncated, _ = env.step(actions)
        # Only track termination for world-0 agents
        newly_done = (terminated[w0_start:w0_end] | truncated[w0_start:w0_end]) & ~ever_done
        done_step[newly_done] = step + 1
        ever_done |= newly_done
        frame_states.append(w0_state())
    return frame_states, done_step


def box_corners(x, y, length, width, heading):
    c, s = np.cos(heading), np.sin(heading)
    hw, hl = width / 2, length / 2
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return (corners @ rot.T) + np.array([x, y])


def render_frame(ax, road_lc, cx, cy, xlim, ylim,
                 gt_x, gt_y, gt_valid,
                 rollout_x, rollout_y,
                 agent_state, step, total, n_agents, done_step):
    ax.cla()
    ax.set_facecolor("#1c1c2e")
    ax.add_collection(copy.copy(road_lc))

    # active[i] = True if agent i is still alive at this step
    active = step < done_step

    # GT trajectories (dashed grey)
    gt_segs = []
    for i in range(n_agents):
        mask = gt_valid[i] > 0
        if mask.sum() < 2:
            continue
        gt_segs.append(np.column_stack([gt_x[i, mask] - cx, gt_y[i, mask] - cy]))
    if gt_segs:
        ax.add_collection(LineCollection(gt_segs, colors="#888888", linewidths=0.7,
                                         linestyles="dashed", alpha=0.55, zorder=2))

    # Policy trails — only up to current step AND agent's done_step
    cmap = plt.cm.get_cmap("tab20")
    trail_segs, trail_colors = [], []
    for i in range(n_agents):
        end = min(done_step[i], step + 1)   # whichever comes first: termination or now
        xs = rollout_x[i, :end] - cx
        ys = rollout_y[i, :end] - cy
        if len(xs) < 2:
            continue
        pts = np.column_stack([xs, ys])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        color = cmap(i % 20)
        alphas = np.linspace(0.2, 0.9, len(segs))
        for seg, a in zip(segs, alphas):
            trail_segs.append(seg)
            trail_colors.append((*color[:3], a))
    if trail_segs:
        ax.add_collection(LineCollection(trail_segs, colors=trail_colors,
                                         linewidths=1.0, zorder=3))

    # Bounding boxes — only for agents still active
    ax_xs = agent_state["x"] - cx
    ax_ys = agent_state["y"] - cy
    headings = agent_state["heading"]
    lengths  = agent_state["length"]
    widths   = agent_state["width"]
    boxes, box_colors = [], []
    for i in range(n_agents):
        if not active[i]:
            continue
        boxes.append(Polygon(box_corners(ax_xs[i], ax_ys[i], lengths[i], widths[i], headings[i])))
        box_colors.append(cmap(i % 20))
    if boxes:
        ax.add_collection(PatchCollection(boxes, facecolors=box_colors,
                                          edgecolors="white", linewidths=0.4,
                                          alpha=0.85, zorder=4))

    # Arrows — only active agents
    if active.any():
        arrow_len = max((xlim[1] - xlim[0]) * 0.015, 0.5)
        ax.quiver(ax_xs[active], ax_ys[active],
                  np.cos(headings[active]) * arrow_len,
                  np.sin(headings[active]) * arrow_len,
                  color=[cmap(i % 20) for i in np.where(active)[0]],
                  angles="xy", scale_units="xy", scale=1,
                  width=0.002, headwidth=4, headlength=5, zorder=5)

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#444444")

    legend_elems = [
        mpatches.Patch(facecolor="#888888", linestyle="--", label="Ground truth"),
        mpatches.Patch(facecolor="#4fc3f7", label="Policy rollout"),
        mpatches.Patch(facecolor="#e8e8e8", label="Road edges"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7,
              facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
    ax.set_title(f"pretty-totem-2  ·  step {step+1}/{total}  ·  {n_agents} agents",
                 color="white", fontsize=10, pad=5)


def make_gif(frame_states, done_step, road, gt_traj, output_path):
    from PIL import Image

    n_agents = len(frame_states[0]["x"])
    total    = len(frame_states)

    rx, ry = road["x"], road["y"]
    cx, cy = rx.mean(), ry.mean()
    pad = max(rx.ptp(), ry.ptp()) * 0.05 + 5
    xlim = (rx.min() - cx - pad, rx.max() - cx + pad)
    ylim = (ry.min() - cy - pad, ry.max() - cy + pad)

    # Road segments
    segs, pt_idx = [], 0
    for seg_len in road["lengths"]:
        xs = road["x"][pt_idx:pt_idx+seg_len] - cx
        ys = road["y"][pt_idx:pt_idx+seg_len] - cy
        if seg_len >= 2:
            segs.append(np.column_stack([xs, ys]))
        pt_idx += seg_len
    road_lc = LineCollection(segs, colors="#e8e8e8", linewidths=0.6, alpha=0.9, zorder=1)

    gt_x     = gt_traj["x"].squeeze(1)
    gt_y     = gt_traj["y"].squeeze(1)
    gt_valid = gt_traj["valid"].squeeze(1)

    rollout_x = np.array([s["x"] for s in frame_states]).T
    rollout_y = np.array([s["y"] for s in frame_states]).T

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    max_span = max(x_span, y_span)
    fw = CANVAS_PX / DPI * (x_span / max_span)
    fh = CANVAS_PX / DPI * (y_span / max_span)
    fw, fh = max(fw, 4.0), max(fh, 4.0)

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#1c1c2e", dpi=DPI)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.06)

    pil_frames = []
    for step in range(total):
        render_frame(ax, road_lc, cx, cy, xlim, ylim,
                     gt_x, gt_y, gt_valid,
                     rollout_x, rollout_y,
                     frame_states[step], step, total, n_agents, done_step)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        pil_frames.append(Image.fromarray(rgba[:, :, :3]))
        if (step + 1) % 10 == 0:
            print(f"  {step+1}/{total} frames", flush=True)

    plt.close(fig)
    pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:],
                       duration=int(1000/FPS), loop=0, optimize=False)
    print(f"\nSaved → {output_path}  ({len(pil_frames)} frames, {FPS} fps)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Ckpt    : {CHECKPOINT}")
    print(f"Map dir : {MAP_DIR}")

    env = build_env()
    print("Loading policy...")
    policy = build_policy(env, device)
    print(f"  num_agents={env.num_agents}")

    env.reset()
    road = env.get_road_edge_polylines()
    gt   = env.get_ground_truth_trajectories()
    rx, ry = road["x"], road["y"]
    print(f"Road span: {rx.ptp():.1f} m × {ry.ptp():.1f} m  ({len(road['lengths'])} segments)")
    print(f"num_envs={env.num_envs}, agents per world={env.agent_offsets[1]}")
    n_world_agents = int(env.agent_offsets[1])
    if env.num_envs > 1:
        print(f"WARNING: {env.num_envs} parallel worlds detected — only world-0 "
              f"({n_world_agents} agents) will be visualized.")
    print(f"Valid GT agents: {(gt['valid'].squeeze(1).sum(1)>0).sum()}/{env.num_agents}")

    print("Rolling out...")
    env.reset()
    frame_states, done_step = rollout(env, policy, device)
    n_w0 = len(frame_states[0]["x"])
    n_finished = (done_step < EPISODE_STEPS).sum()
    print(f"Collected {len(frame_states)} frames  |  world-0: {n_w0} agents  |  {n_finished}/{n_w0} reached goal")

    print("Rendering GIF...")
    make_gif(frame_states, done_step, road, gt, OUTPUT)
    env.close()


if __name__ == "__main__":
    main()
