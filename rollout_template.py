"""
Generic rollout script — run from each project's directory using its own venv.

Usage (from project root, e.g. /data/lynx/saeani/pufferdrive_hetero_v2/):
    .venv/bin/python /users/saeani/src/guided_original/rollout_template.py \
        --checkpoint experiments/puffer_drive_eodvv1ay.pt \
        --run_name pretty-totem-2 \
        --output_dir /users/saeani/src/guided_original/rollouts \
        --train_scenarios 0,1000,2000,3000,4000,5000,6000,7000,8000,8900 \
        --val_scenarios 0,100,200,300,400,500,600,700,800,900

The script auto-detects:
  - Architecture type from the checkpoint state dict
  - Required env kwargs from the config/ocean/drive.ini file
"""

import argparse
import copy
import os
import sys
import tempfile
import types
import configparser

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from PIL import Image

# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--run_name",         required=True,
                   help="Short label used in GIF title and filename")
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--train_scenarios",  default="0,1000,2000,3000,4000,5000,6000,7000,8000,8900",
                   help="Comma-separated 0-based indices into training map dir")
    p.add_argument("--val_scenarios",    default="0,100,200,300,400,500,600,700,800,900",
                   help="Comma-separated 0-based indices into validation map dir")
    p.add_argument("--episode_steps",    type=int, default=91)
    p.add_argument("--fps",              type=int, default=10)
    p.add_argument("--canvas_px",        type=int, default=700)
    p.add_argument("--dpi",              type=int, default=100)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Architecture inference from checkpoint
# ────────────────────────────────────────────────────────────────────────────
def infer_arch(sd):
    ego_feat = sd["policy.ego_encoder.0.weight"].shape[1]
    has_style_gru = "policy.style_encoder.gru.weight_ih_l0" in sd

    if has_style_gru:
        arch_type = "causal"
        style_z_dim = sd["policy.style_encoder.fc_mu.weight"].shape[0]
        style_enc_hidden = sd["policy.style_encoder.gru.weight_ih_l0"].shape[0] // 3
    elif ego_feat > 191:
        arch_type = "vae"
        style_z_dim = ego_feat - 190
        style_enc_hidden = 0
    elif ego_feat == 191:
        arch_type = "hetero"
        style_z_dim = 0
        style_enc_hidden = 0
    else:
        arch_type = "standard"
        style_z_dim = 0
        style_enc_hidden = 0

    input_size   = sd["policy.ego_encoder.0.weight"].shape[0]
    hidden_size  = sd["policy.shared_embedding.1.weight"].shape[0]
    partner_feat = sd["policy.partner_encoder.0.weight"].shape[1]
    road_feat    = sd["policy.road_encoder.0.weight"].shape[1] - 6
    lstm_hidden  = sd["lstm.weight_hh_l0"].shape[0] // 4
    lstm_input   = sd["lstm.weight_ih_l0"].shape[1]

    return dict(
        arch_type=arch_type,
        ego_features=ego_feat,
        input_size=input_size,
        hidden_size=hidden_size,
        partner_features=partner_feat,
        road_features=road_feat,
        lstm_hidden=lstm_hidden,
        lstm_input=lstm_input,
        style_z_dim=style_z_dim,
        style_enc_hidden=style_enc_hidden,
    )


# ────────────────────────────────────────────────────────────────────────────
# Read relevant env params from config/ocean/drive.ini
# ────────────────────────────────────────────────────────────────────────────
def read_drive_config():
    cfg = configparser.ConfigParser(allow_no_value=True, strict=False,
                                    inline_comment_prefixes=(';', '#'))
    cfg.read("config/ocean/drive.ini")
    env_sec = cfg["env"] if "env" in cfg else {}

    def getf(key, default):
        v = env_sec.get(key, None)
        if v is None:
            return default
        # Strip inline comments and quotes
        v = v.split(";")[0].split("#")[0].strip().strip('"').strip("'")
        try:
            return float(v)
        except ValueError:
            return v

    def getb(key, default):
        v = env_sec.get(key, None)
        if v is None:
            return default
        v = v.split(";")[0].split("#")[0].strip().lower()
        return v in ("1", "true", "yes")

    return dict(
        guidance_dropout_prob  = getf("guidance_dropout_prob",   None),
        guidance_dropout_mode  = getf("guidance_dropout_mode",   None),
        style_rand_prob        = getf("style_rand_prob",         None),
        style_z_dim            = getf("style_z_dim",             None),
        guidance_speed_weight  = getf("guidance_speed_weight",   0.5),
        guidance_heading_weight= getf("guidance_heading_weight", 0.5),
        use_guided_autonomy    = getb("use_guided_autonomy",     True),
        use_guidance_observations = getb("use_guidance_observations", True),
        reward_vehicle_collision = getf("reward_vehicle_collision", -0.5),
        reward_offroad_collision = getf("reward_offroad_collision", -0.5),
    )


# ────────────────────────────────────────────────────────────────────────────
# Build env for a single scenario (temp symlink dir)
# ────────────────────────────────────────────────────────────────────────────
def build_env_for_scenario(Drive, map_file_path, arch, cfg, episode_steps):
    tmpdir = tempfile.mkdtemp(prefix="rollout_map_")
    link = os.path.join(tmpdir, "map_000.bin")
    os.symlink(os.path.abspath(map_file_path), link)

    kwargs = dict(
        map_dir             = tmpdir,
        num_maps            = 1,
        num_agents          = 128,
        episode_length      = episode_steps,
        init_steps          = 0,
        resample_frequency  = 99999,
        use_guided_autonomy = int(cfg["use_guided_autonomy"]),
        use_guidance_observations = int(cfg["use_guidance_observations"]),
        guidance_speed_weight  = cfg["guidance_speed_weight"],
        guidance_heading_weight= cfg["guidance_heading_weight"],
        prep_human_data     = False,
        save_data_to_disk   = False,
        reward_vehicle_collision = cfg["reward_vehicle_collision"],
        reward_offroad_collision = cfg["reward_offroad_collision"],
    )

    # Optional params — only pass if supported by this env version
    if cfg["guidance_dropout_prob"] is not None:
        kwargs["guidance_dropout_prob"] = cfg["guidance_dropout_prob"]
        kwargs["guidance_dropout_mode"] = cfg["guidance_dropout_mode"] or "max"

    if cfg["style_rand_prob"] is not None:
        kwargs["style_rand_prob"] = 0.0   # no randomization at eval

    if arch["style_z_dim"] > 0:
        kwargs["style_z_dim"] = arch["style_z_dim"]

    env = Drive(**kwargs)
    return env, tmpdir


# ────────────────────────────────────────────────────────────────────────────
# Build policy
# ────────────────────────────────────────────────────────────────────────────
def build_policy(env, arch, sd, device, ocean_torch, pufferlib):
    proxy = types.SimpleNamespace(
        single_observation_space  = env.single_observation_space,
        single_action_space       = env.single_action_space,
        max_partner_objects       = env.max_partner_objects,
        partner_features          = arch["partner_features"],
        max_road_objects          = env.max_road_objects,
        road_features             = arch["road_features"],
        ego_features              = arch["ego_features"],
        # ego_features_base is used by causal encoder to size the style GRU
        # (= ego_features minus z_dim slots)
        ego_features_base         = arch["ego_features"] - arch["style_z_dim"],
    )

    kwargs = dict(input_size=arch["input_size"], hidden_size=arch["hidden_size"])
    if arch["arch_type"] == "causal":
        kwargs["style_z_dim"] = arch["style_z_dim"]
        kwargs["style_encoder_hidden_size"] = arch["style_enc_hidden"]
        kwargs["style_stop_grad_rl"] = True

    base   = ocean_torch.Drive(proxy, **kwargs)
    policy = pufferlib.models.LSTMWrapper(
        proxy, base,
        input_size=arch["lstm_input"],
        hidden_size=arch["lstm_hidden"],
    )
    policy = policy.to(device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy


# ────────────────────────────────────────────────────────────────────────────
# Rollout loop
# ────────────────────────────────────────────────────────────────────────────
def run_rollout(env, policy, device, episode_steps):
    w0_start = int(env.agent_offsets[0])
    w0_end   = int(env.agent_offsets[1])
    n_total  = env.num_agents
    h        = policy.hidden_size

    obs_np, _ = env.reset()
    state = {
        "lstm_h": torch.zeros(n_total, h, device=device),
        "lstm_c": torch.zeros(n_total, h, device=device),
    }

    def w0_state():
        st = env.get_global_agent_state()
        return {k: v[w0_start:w0_end] for k, v in st.items()}

    frame_states = [w0_state()]
    n_w0 = w0_end - w0_start
    done_step = np.full(n_w0, episode_steps, dtype=int)
    ever_done  = np.zeros(n_w0, dtype=bool)

    for step in range(episode_steps - 1):
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_t, state)
        if isinstance(logits, (list, tuple)):
            actions = torch.cat(
                [lg.argmax(-1).unsqueeze(-1) for lg in logits], -1
            ).cpu().numpy()
        else:
            actions = logits.argmax(-1).unsqueeze(-1).cpu().numpy()

        obs_np, _, terminated, truncated, _ = env.step(actions)
        newly = (terminated[w0_start:w0_end] | truncated[w0_start:w0_end]) & ~ever_done
        done_step[newly] = step + 1
        ever_done |= newly
        frame_states.append(w0_state())

    return frame_states, done_step


# ────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ────────────────────────────────────────────────────────────────────────────
CMAP = plt.cm.get_cmap("tab20")

def box_corners(x, y, length, width, heading):
    c, s = np.cos(heading), np.sin(heading)
    hw, hl = width / 2, length / 2
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return (corners @ rot.T) + np.array([x, y])


def render_frame(ax, road_lc, cx, cy, xlim, ylim,
                 gt_x, gt_y, gt_valid,
                 rollout_x, rollout_y,
                 agent_state, step, total, n_agents, done_step,
                 title_str):
    ax.cla()
    ax.set_facecolor("#1c1c2e")
    ax.add_collection(copy.copy(road_lc))

    active = step < done_step

    # GT trajectories
    gt_segs = []
    for i in range(n_agents):
        mask = gt_valid[i] > 0
        if mask.sum() < 2:
            continue
        gt_segs.append(np.column_stack([gt_x[i, mask] - cx, gt_y[i, mask] - cy]))
    if gt_segs:
        ax.add_collection(LineCollection(gt_segs, colors="#888888", linewidths=0.7,
                                         linestyles="dashed", alpha=0.55, zorder=2))

    # Policy trails
    trail_segs, trail_colors = [], []
    for i in range(n_agents):
        end = min(done_step[i], step + 1)
        xs = rollout_x[i, :end] - cx
        ys = rollout_y[i, :end] - cy
        if len(xs) < 2:
            continue
        pts = np.column_stack([xs, ys])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        color = CMAP(i % 20)
        alphas = np.linspace(0.2, 0.9, len(segs))
        for seg, a in zip(segs, alphas):
            trail_segs.append(seg)
            trail_colors.append((*color[:3], a))
    if trail_segs:
        ax.add_collection(LineCollection(trail_segs, colors=trail_colors,
                                         linewidths=1.0, zorder=3))

    # Bounding boxes
    ax_xs    = agent_state["x"] - cx
    ax_ys    = agent_state["y"] - cy
    headings = agent_state["heading"]
    lengths  = agent_state["length"]
    widths   = agent_state["width"]
    boxes, box_colors = [], []
    for i in range(n_agents):
        if not active[i]:
            continue
        boxes.append(Polygon(box_corners(ax_xs[i], ax_ys[i],
                                          lengths[i], widths[i], headings[i])))
        box_colors.append(CMAP(i % 20))
    if boxes:
        ax.add_collection(PatchCollection(boxes, facecolors=box_colors,
                                           edgecolors="white", linewidths=0.4,
                                           alpha=0.85, zorder=4))

    if active.any():
        arrow_len = max((xlim[1] - xlim[0]) * 0.015, 0.5)
        ax.quiver(ax_xs[active], ax_ys[active],
                  np.cos(headings[active]) * arrow_len,
                  np.sin(headings[active]) * arrow_len,
                  color=[CMAP(i % 20) for i in np.where(active)[0]],
                  angles="xy", scale_units="xy", scale=1,
                  width=0.002, headwidth=4, headlength=5, zorder=5)

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#aaaaaa", fontsize=7)
    ax.set_ylabel("Y (m)", color="#aaaaaa", fontsize=7)
    ax.tick_params(colors="#888888", labelsize=6)
    for sp in ax.spines.values():
        sp.set_color("#444444")

    legend_elems = [
        mpatches.Patch(facecolor="#888888", linestyle="--", label="Ground truth"),
        mpatches.Patch(facecolor="#4fc3f7", label="Policy rollout"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=6,
              facecolor="#2a2a3e", edgecolor="#555555", labelcolor="white")
    ax.set_title(f"{title_str}  ·  step {step+1}/{total}",
                 color="white", fontsize=9, pad=4)


def save_gif(frame_states, done_step, road, gt_traj, output_path,
             run_name, scenario_label, episode_steps, fps, canvas_px, dpi):
    n_agents = len(frame_states[0]["x"])
    total    = len(frame_states)

    rx, ry = road["x"], road["y"]
    cx, cy = rx.mean(), ry.mean()
    pad    = max(rx.ptp(), ry.ptp()) * 0.05 + 5
    xlim   = (rx.min() - cx - pad, rx.max() - cx + pad)
    ylim   = (ry.min() - cy - pad, ry.max() - cy + pad)

    segs, pt_idx = [], 0
    for seg_len in road["lengths"]:
        xs = road["x"][pt_idx:pt_idx+seg_len] - cx
        ys = road["y"][pt_idx:pt_idx+seg_len] - cy
        if seg_len >= 2:
            segs.append(np.column_stack([xs, ys]))
        pt_idx += seg_len
    road_lc = LineCollection(segs, colors="#e8e8e8", linewidths=0.6,
                              alpha=0.9, zorder=1)

    gt_x     = gt_traj["x"].squeeze(1)
    gt_y     = gt_traj["y"].squeeze(1)
    gt_valid = gt_traj["valid"].squeeze(1)
    rollout_x = np.array([s["x"] for s in frame_states]).T
    rollout_y = np.array([s["y"] for s in frame_states]).T

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    max_span = max(x_span, y_span)
    fw = canvas_px / dpi * (x_span / max_span)
    fh = canvas_px / dpi * (y_span / max_span)
    fw, fh = max(fw, 4.0), max(fh, 4.0)

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#1c1c2e", dpi=dpi)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07)

    title_str = f"{run_name}  |  {scenario_label}"
    pil_frames = []
    for step in range(total):
        render_frame(ax, road_lc, cx, cy, xlim, ylim,
                     gt_x, gt_y, gt_valid,
                     rollout_x, rollout_y,
                     frame_states[step], step, total, n_agents, done_step,
                     title_str)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        pil_frames.append(Image.fromarray(rgba[:, :, :3]))

    plt.close(fig)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_frames[0].save(output_path, save_all=True,
                       append_images=pil_frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=False)
    print(f"  Saved → {output_path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # cd into project dir = cwd when invoked via shell runner
    project_dir = os.getcwd()
    sys.path.insert(0, project_dir)

    from pufferlib.ocean.drive.drive import Drive
    import pufferlib.ocean.torch as ocean_torch
    import pufferlib.models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Run name  : {args.run_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device    : {device}")

    # Load checkpoint and detect architecture
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    arch = infer_arch(sd)
    print(f"Arch type   : {arch['arch_type']}  (ego={arch['ego_features']}, "
          f"lstm={arch['lstm_hidden']}, style_z={arch['style_z_dim']})")

    # Read drive config
    cfg = read_drive_config()
    print(f"Drive cfg   : dropout_prob={cfg['guidance_dropout_prob']}, "
          f"style_rand={cfg['style_rand_prob']}, style_z_dim={cfg['style_z_dim']}")

    # Data directories
    map_subdirs = {
        "train": "pufferlib/resources/drive/binaries/training",
        "val":   "pufferlib/resources/drive/binaries/validation",
    }

    scenario_map = {
        "train": [int(x) for x in args.train_scenarios.split(",")],
        "val":   [int(x) for x in args.val_scenarios.split(",")],
    }

    # File name for the run (replace spaces/slashes)
    run_safe = args.run_name.replace("/", "_").replace(" ", "_")

    for split, indices in scenario_map.items():
        print(f"\n── {split.upper()} scenarios: {indices}")
        map_dir = map_subdirs[split]
        # Build sorted map file list (same as Drive does internally)
        all_maps = sorted(
            [f for f in os.listdir(map_dir) if f.endswith(".bin")],
            key=lambda f: int(f.replace("map_", "").replace(".bin", ""))
        )

        for idx in indices:
            map_file = os.path.join(map_dir, all_maps[idx])
            scenario_label = f"{split}_{all_maps[idx].replace('.bin','')}"
            out_name = f"{run_safe}.gif"
            out_path = os.path.join(args.output_dir, scenario_label, out_name)

            if os.path.exists(out_path):
                print(f"  [SKIP] {scenario_label} — already exists")
                continue

            print(f"  Scenario {scenario_label}  ({map_file})")

            env, tmpdir = build_env_for_scenario(
                Drive, map_file, arch, cfg, args.episode_steps
            )
            try:
                # Build policy once per scenario (to reset states cleanly)
                policy = build_policy(env, arch, sd, device, ocean_torch, pufferlib)

                env.reset()
                road = env.get_road_edge_polylines()
                gt   = env.get_ground_truth_trajectories()
                n_w0 = int(env.agent_offsets[1]) - int(env.agent_offsets[0])
                print(f"    agents in world-0: {n_w0}  |  "
                      f"road span: {road['x'].ptp():.0f}×{road['y'].ptp():.0f} m")

                env.reset()
                frame_states, done_step = run_rollout(
                    env, policy, device, args.episode_steps
                )

                save_gif(
                    frame_states, done_step, road, gt, out_path,
                    args.run_name, scenario_label,
                    args.episode_steps, args.fps, args.canvas_px, args.dpi
                )
            finally:
                env.close()
                # Clean up temp symlink dir
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nDone — {args.run_name}")


if __name__ == "__main__":
    main()
