"""
Z-Sweep Style Visualization
============================
For style-conditioned projects (VAE, Causal, Hetero), runs the SAME scenario
3 times with different style z values (z=-0.95, z=0.0, z=+0.95) and renders:

  - zsweep_comparison.mp4      (3-panel side-by-side animated video)
  - trajectory_snapshot.png    (static 3-panel trajectory overview)
  - z0_00.mp4 / z0_05.mp4 ...  (individual per-z MP4s, with --individual_videos)
  - summary.csv                (per-scenario metrics: goal rate, speed, ADE)

Three evaluation modes (run all by default, or select a subset):

  init_goal  — Start from timestep 0, guidance waypoints fully dropped out
               (guidance_dropout_prob=1.0), so the agent only sees its initial
               state + goal position. z latent fully controls driving style.
               Uses goal_radius=4.0m (relaxed, same as other evals).

  10step     — Start from timestep 10 (10 steps of expert replay), then
               guidance waypoints fully dropped. Agent drives from step 10
               onward using only start state + goal.

  sdc_only   — Only the SDC (agent 0) is controlled by the policy; all other
               agents replay their logged trajectories. This isolates the effect
               of z on a single agent in realistic traffic.

Architecture support:
  - VAE  (style_z_dim > 0, no GRU encoder):
      z is patched into the last style_z_dim dims of every agent's obs each step.
  - Causal (style_z_dim > 0, has GRU encoder):
      Uses policy.policy.set_style_override(z_tensor) to bypass the encoder.
  - Hetero (style_z_dim == 1, 1D style score):
      z is patched as a scalar style score (0=conservative, 1=aggressive).
  - Standard (style_z_dim == 0):
      Detected automatically; skipped with a warning.
"""

import argparse
import configparser
import csv
import os
import shutil
import sys
import tempfile
import types

import math
import imageio
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection, PatchCollection
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Reused from analyze_behavior.py
# ─────────────────────────────────────────────────────────────────────────────

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
        style_z_dim = 1           # hetero injects 1D z into obs
        style_enc_hidden = 0
    else:
        arch_type = "standard"
        style_z_dim = 0
        style_enc_hidden = 0

    return dict(
        arch_type        = arch_type,
        ego_features     = ego_feat,
        input_size       = sd["policy.ego_encoder.0.weight"].shape[0],
        hidden_size      = sd["policy.shared_embedding.1.weight"].shape[0],
        partner_features = sd["policy.partner_encoder.0.weight"].shape[1],
        road_features    = sd["policy.road_encoder.0.weight"].shape[1] - 6,
        lstm_hidden      = sd["lstm.weight_hh_l0"].shape[0] // 4,
        lstm_input       = sd["lstm.weight_ih_l0"].shape[1],
        style_z_dim      = style_z_dim,
        style_enc_hidden = style_enc_hidden,
    )


def read_drive_config():
    cfg = configparser.ConfigParser(allow_no_value=True, strict=False,
                                    inline_comment_prefixes=(";", "#"))
    cfg.read("config/ocean/drive.ini")
    env_sec = cfg["env"] if "env" in cfg else {}

    def getf(key, default):
        v = env_sec.get(key, None)
        if v is None: return default
        v = v.split(";")[0].split("#")[0].strip().strip('"').strip("'")
        try: return float(v)
        except ValueError: return v

    def getb(key, default):
        v = env_sec.get(key, None)
        if v is None: return default
        return v.split(";")[0].split("#")[0].strip().lower() in ("1", "true", "yes")

    return dict(
        guidance_dropout_prob     = getf("guidance_dropout_prob",    None),
        guidance_dropout_mode     = getf("guidance_dropout_mode",    None),
        style_rand_prob           = getf("style_rand_prob",          None),
        style_z_dim               = getf("style_z_dim",              None),
        guidance_speed_weight     = getf("guidance_speed_weight",    0.5),
        guidance_heading_weight   = getf("guidance_heading_weight",  0.5),
        use_guided_autonomy       = getb("use_guided_autonomy",      True),
        use_guidance_observations = getb("use_guidance_observations", True),
        reward_vehicle_collision  = getf("reward_vehicle_collision", -0.5),
        reward_offroad_collision  = getf("reward_offroad_collision", -0.5),
        reward_goal               = getf("reward_goal",              1.0),
    )



def get_z_configs(z_str, sweep_mode, style_z_dim):
    base_vals = [float(z) for z in z_str.split(",")]
    if style_z_dim <= 1:
        # hetero or standard
        return [{"label": f"{z:.2f}", "val": z, "scalar": z} for z in base_vals]
    
    configs = []
    if sweep_mode == "uniform":
        for z in base_vals:
            arr = np.full((style_z_dim,), z, dtype=np.float32)
            configs.append({"label": f"{z:.2f}", "val": arr, "scalar": z})
    elif sweep_mode == "pca":
        # First principal component extracted from Waymo expert VAE latents
        pc1 = np.array([-0.2689, -0.3497, -0.6714, -0.5955], dtype=np.float32)
        if style_z_dim != len(pc1):
            raise ValueError(f"PCA sweep requires style_z_dim={len(pc1)}, got {style_z_dim}")
        for z in base_vals:
            # We scale the PC1 vector by z
            arr = pc1 * z
            configs.append({"label": f"PC1({z:.2f})", "val": arr, "scalar": z})
    elif sweep_mode == "axis":
        # Generate neutral first if present
        if 0.0 in base_vals:
            arr = np.zeros((style_z_dim,), dtype=np.float32)
            configs.append({"label": "z0.00", "val": arr, "scalar": 0.0})
        for d in range(style_z_dim):
            for z in base_vals:
                if z == 0.0: continue
                arr = np.zeros((style_z_dim,), dtype=np.float32)
                arr[d] = z
                label_val = f"+{z}" if z > 0 else f"{z}"
                configs.append({"label": f"D{d}={label_val}", "val": arr, "scalar": z})
    else:
        raise ValueError(f"Unknown sweep_mode {sweep_mode}")
    return configs


def build_env_for_scenario(Drive, map_file_path, arch, cfg, episode_steps,
                           init_steps=0, goal_radius=4.0,
                           control_mode="control_wosac"):
    tmpdir = tempfile.mkdtemp(prefix="zsweep_map_")
    os.symlink(os.path.abspath(map_file_path),
               os.path.join(tmpdir, "map_000.bin"))

    kwargs = dict(
        map_dir                   = tmpdir,
        num_maps                  = 1,
        num_agents                = 128,
        episode_length            = episode_steps,
        init_steps                = init_steps,
        resample_frequency        = 99999,
        # -- Eval-mode settings --
        init_mode                 = "create_all_valid",
        control_mode              = control_mode,
        goal_behavior             = 3,          # stop + truncate
        goal_radius               = goal_radius,
        # -- Drop ALL guidance waypoints so z controls behavior --
        guidance_dropout_prob     = 1.0,
        guidance_dropout_mode     = "remove_all",
        # -- Keep guided autonomy observations in obs layout --
        use_guided_autonomy       = int(cfg["use_guided_autonomy"]),
        use_guidance_observations = int(cfg["use_guidance_observations"]),
        guidance_speed_weight     = cfg["guidance_speed_weight"],
        guidance_heading_weight   = cfg["guidance_heading_weight"],
        prep_human_data           = False,
        save_data_to_disk         = False,
        reward_vehicle_collision  = cfg["reward_vehicle_collision"],
        reward_offroad_collision  = cfg["reward_offroad_collision"],
    )
    if cfg["style_rand_prob"] is not None:
        kwargs["style_rand_prob"] = 0.0          # disable random z during eval
    if arch["style_z_dim"] > 0:
        kwargs["style_z_dim"] = arch["style_z_dim"]

    # Some project builds don't accept style_z_dim / style_rand_prob as
    # constructor args (they read from config instead). Try gracefully.
    try:
        env = Drive(**kwargs)
    except TypeError:
        for key in ("style_z_dim", "style_rand_prob"):
            kwargs.pop(key, None)
        env = Drive(**kwargs)

    return env, tmpdir


def build_policy(env, arch, sd, device, ocean_torch, pufferlib):
    # For causal arch, z is computed internally (not in obs), so
    # ego_features for obs slicing must be the base (without z).
    if arch["arch_type"] == "causal" and arch["style_z_dim"] > 0:
        ego_feat_for_obs = arch["ego_features"] - arch["style_z_dim"]
    else:
        ego_feat_for_obs = arch["ego_features"]
    proxy = types.SimpleNamespace(
        single_observation_space = env.single_observation_space,
        single_action_space      = env.single_action_space,
        max_partner_objects      = env.max_partner_objects,
        partner_features         = arch["partner_features"],
        max_road_objects         = env.max_road_objects,
        road_features            = arch["road_features"],
        ego_features             = ego_feat_for_obs,
        ego_features_base        = ego_feat_for_obs - arch["style_z_dim"] if arch["arch_type"] != "causal" else ego_feat_for_obs,
    )
    kwargs = dict(input_size=arch["input_size"], hidden_size=arch["hidden_size"])
    if arch["arch_type"] == "causal":
        kwargs["style_z_dim"]               = arch["style_z_dim"]
        kwargs["style_encoder_hidden_size"] = arch["style_enc_hidden"]
        kwargs["style_stop_grad_rl"]        = True

    base   = ocean_torch.Drive(proxy, **kwargs)
    policy = pufferlib.models.LSTMWrapper(
        proxy, base,
        input_size  = arch["lstm_input"],
        hidden_size = arch["lstm_hidden"],
    )
    policy = policy.to(device)
    policy.load_state_dict(sd)
    policy.eval()
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Z-fixed rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_rollout_z(env, policy, device, z_value, arch, z_obs_slice,
                  is_sdc_only=False):
    """
    Run one episode with a fixed z value.
    Episode length is determined by env (episode_length - init_steps).
    z_obs_slice: (start, end) indices into observation for patching z.
    If is_sdc_only=True, only the SDC (agent 0 per world) gets z_value;
    all other agents get z=0 (neutral style).
    Returns:
        frame_states : list of dicts with x/y/heading/length/width (w0 agents)
        rewards_trace: list of (n_w0,) arrays
        n_w0         : number of world-0 agents
        gt           : ground-truth trajectory dict (world-0 agents only)
    """
    arch_type   = arch["arch_type"]
    style_z_dim = arch["style_z_dim"]
    z_start, z_end = z_obs_slice

    w0_start = int(env.agent_offsets[0])
    w0_end   = int(env.agent_offsets[1])
    n_total  = env.num_agents
    h        = policy.hidden_size
    sim_steps = env.episode_length - env.init_steps

    # For causal: set fixed z via the policy's override mechanism
    if arch_type == "causal" and style_z_dim > 0:
        z_tensor = torch.zeros((n_total, style_z_dim), dtype=torch.float32, device=device)
        if isinstance(z_value, (np.ndarray, list)):
            z_val_t = torch.tensor(z_value, dtype=torch.float32, device=device)
        else:
            z_val_t = torch.tensor([z_value]*style_z_dim, dtype=torch.float32, device=device)
            
        if is_sdc_only:
            z_tensor[w0_start] = z_val_t
        else:
            z_tensor[:] = z_val_t
        policy.policy.set_style_override(z_tensor)

    obs_np, _ = env.reset()

    # Collect GT on first reset
    gt_full = env.get_ground_truth_trajectories()
    gt = {k: v[w0_start:w0_end] for k, v in gt_full.items()
          if isinstance(v, np.ndarray) and v.shape[0] >= w0_end}

    state = {
        "lstm_h": torch.zeros(n_total, h, device=device),
        "lstm_c": torch.zeros(n_total, h, device=device),
    }

    def snapshot_w0():
        st = env.get_global_agent_state()
        return {k: v[w0_start:w0_end].copy() for k, v in st.items()}

    frame_states  = [snapshot_w0()]
    rewards_trace = []

    n_w0 = w0_end - w0_start

    for _ in range(sim_steps - 1):
        # Patch z in the observation at the correct indices
        # (skip for causal — z is handled via set_style_override)
        if style_z_dim > 0 and z_start < z_end:
            if is_sdc_only:
                obs_np[:, z_start:z_end] = 0.0                # neutral z for all
                obs_np[w0_start, z_start:z_end] = z_value      # SDC gets test z
            else:
                obs_np[:, z_start:z_end] = z_value

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_t, state)

        if isinstance(logits, (list, tuple)):
            actions = torch.cat(
                [lg.argmax(-1).unsqueeze(-1) for lg in logits], -1
            ).cpu().numpy()
        else:
            actions = logits.argmax(-1).unsqueeze(-1).cpu().numpy()

        obs_np, rew_np, _, _, _ = env.step(actions)
        rewards_trace.append(rew_np[w0_start:w0_end].copy())
        frame_states.append(snapshot_w0())

    # Clear override after rollout
    if arch_type == "causal" and hasattr(policy, "policy") and \
            hasattr(policy.policy, "set_style_override"):
        policy.policy.set_style_override(None)

    return frame_states, rewards_trace, n_w0, gt


# ─────────────────────────────────────────────────────────────────────────────
# Road geometry
# ─────────────────────────────────────────────────────────────────────────────

def try_get_road_edges(env):
    """Try to get road edge polylines from the env (best-effort)."""
    for method in ("get_road_edge_polylines", "get_road_geometry", "road_edges"):
        if callable(getattr(env, method, None)):
            try:
                return getattr(env, method)()
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Rendering utilities
# ─────────────────────────────────────────────────────────────────────────────

# Color palette: distinct colors for up to 16 agents
AGENT_COLORS = plt.get_cmap("tab20").colors   # 20 distinct colors


def _box_corners(x, y, length, width, heading):
    """Return (4, 2) array of rotated bounding box corners."""
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    hw, hl = width / 2.0, length / 2.0
    local = np.array([[ hl,  hw], [ hl, -hw],
                      [-hl, -hw], [-hl,  hw]])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return local @ rot.T + np.array([x, y])


def _compute_bounds(gt, n_w0, padding=20.0):
    """Compute axis bounds from GT trajectories."""
    xs, ys = [], []
    gx = gt.get("x", None)
    gy = gt.get("y", None)
    gv = gt.get("valid", None)
    if gx is not None:
        gx_w0 = gx[:n_w0].squeeze(1) if gx.ndim == 3 else gx[:n_w0]
        gy_w0 = gy[:n_w0].squeeze(1) if gy.ndim == 3 else gy[:n_w0]
        if gv is not None:
            gv_w0 = gv[:n_w0].squeeze(1).astype(bool) if gv.ndim == 3 else gv[:n_w0].astype(bool)
            mask  = gv_w0.flatten() > 0
        else:
            mask = np.ones(gx_w0.size, dtype=bool)
        xs.append(gx_w0.flatten()[mask])
        ys.append(gy_w0.flatten()[mask])

    if xs:
        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        return (all_x.min() - padding, all_x.max() + padding,
                all_y.min() - padding, all_y.max() + padding)
    return (-50, 50, -50, 50)


def _draw_road_edges(ax, road_edges):
    """Draw road edge polylines onto ax using LineCollection (fast)."""
    if road_edges is None:
        return
    lengths = road_edges.get("lengths", [])
    x_arr   = road_edges.get("x", np.array([]))
    y_arr   = road_edges.get("y", np.array([]))
    segments = []
    pt_idx = 0
    for length in lengths:
        length = int(length)
        px = x_arr[pt_idx: pt_idx + length]
        py = y_arr[pt_idx: pt_idx + length]
        if len(px) >= 2:
            segments.append(np.column_stack([px, py]))
        pt_idx += length
    if segments:
        ax.add_collection(LineCollection(segments, colors='#888888',
                                         linewidths=0.8, alpha=0.35, zorder=1))


def _draw_gt_paths(ax, gt, n_w0):
    """Draw ground truth agent paths as thin dashed gray lines."""
    gx = gt.get("x", None)
    gy = gt.get("y", None)
    gv = gt.get("valid", None)
    if gx is None:
        return
    gx_w0 = gx[:n_w0].squeeze(1) if gx.ndim == 3 else gx[:n_w0]
    gy_w0 = gy[:n_w0].squeeze(1) if gy.ndim == 3 else gy[:n_w0]
    for i in range(min(n_w0, 16)):
        if gv is not None:
            valid_i = gv[i].squeeze() if gv.ndim == 3 else gv[i]
            mask    = valid_i.astype(bool)
        else:
            mask = np.ones(gx_w0.shape[1], dtype=bool)
        ax.plot(gx_w0[i][mask], gy_w0[i][mask],
                color="lightgray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=2)


def _draw_agents_at_t(ax, frame_states, t, n_w0, trail_len=10, is_sdc_only=False):
    """Draw all agents at timestep t with trail using batched draw calls."""
    INVALID_POS = -10000.0
    trail_start = max(0, t - trail_len)

    # In sdc_only mode: draw replay agents (muted gray) then SDC (bold red)
    if is_sdc_only:
        groups = [
            ("replay", range(1, min(n_w0, 20)), "#aaaaaa", 0.6, 0.25, 0.4, 0.2, 3, 4),
            ("sdc",    [0],                     "#d32f2f", 2.5, 0.7,  0.95, 0.5, 5, 6),
        ]
    else:
        groups = [("all", range(min(n_w0, 20)), None, 1.2, 0.4, 0.9, 0.4, 3, 4)]

    for (gname, agents, fixed_color, lw, alpha_trail, alpha_box,
         edge_lw, z_trail, z_box) in groups:
        trail_segments, trail_colors = [], []
        patches, patch_facecolors = [], []

        for i in agents:
            color = fixed_color if fixed_color else AGENT_COLORS[i % len(AGENT_COLORS)]
            tx = [float(frame_states[ti]["x"][i]) for ti in range(trail_start, t + 1)
                  if float(frame_states[ti]["x"][i]) > INVALID_POS]
            ty = [float(frame_states[ti]["y"][i]) for ti in range(trail_start, t + 1)
                  if float(frame_states[ti]["x"][i]) > INVALID_POS]
            if len(tx) >= 2:
                trail_segments.append(np.column_stack([tx, ty]))
                trail_colors.append(color)
            s = frame_states[t]
            if float(s["x"][i]) <= INVALID_POS:
                continue
            try:
                corners = _box_corners(
                    float(s["x"][i]), float(s["y"][i]),
                    float(s["length"][i]), float(s["width"][i]),
                    float(s["heading"][i]),
                )
                patches.append(MplPolygon(corners, closed=True))
                patch_facecolors.append(color)
            except (KeyError, IndexError, ValueError):
                pass

        if trail_segments:
            ax.add_collection(LineCollection(trail_segments, colors=trail_colors,
                                             linewidths=lw, alpha=alpha_trail, zorder=z_trail))
        if patches:
            pc = PatchCollection(patches, facecolors=patch_facecolors,
                                 edgecolors='black', linewidths=edge_lw,
                                 alpha=alpha_box, zorder=z_box)
            ax.add_collection(pc)


def _setup_ax(ax, bounds, title, t=None, T=None):
    """Configure axis limits, aspect, title, etc."""
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    full_title = title
    if t is not None:
        full_title += f"\nt={t}/{T-1}  ({t * 0.1:.1f}s)"
    ax.set_title(full_title, fontsize=9, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory snapshot PNG (static overview of full episode trajectories)
# ─────────────────────────────────────────────────────────────────────────────

def render_trajectory_snapshot(
    all_frame_states, gt, n_w0, bounds, road_edges,
    z_values, run_name, scenario_label, mode_label, out_dir,
    is_sdc_only=False,
):
    """
    Grid static figure: each panel shows full episode trajectory per z value.
    Max 3 columns, wraps into new rows. Agents colored by ID, time by alpha.
    Start = triangle, End = circle.
    In sdc_only mode: SDC (agent 0) in bold red, others in muted gray.
    """
    n_panels = len(z_values)
    MAX_COLS = 3
    ncols = min(n_panels, MAX_COLS)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes_flat = np.array(axes).flatten()
    # Hide any unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for panel_idx, (z_label, frames) in enumerate(zip(z_values, all_frame_states)):
        ax = axes_flat[panel_idx]
        _draw_road_edges(ax, road_edges)
        _draw_gt_paths(ax, gt, n_w0)

        T = len(frames)
        # In sdc_only mode: draw replay agents first (muted), then SDC on top
        agent_order = list(range(1, min(n_w0, 20))) + [0] if is_sdc_only \
            else list(range(min(n_w0, 20)))
        for i in agent_order:
            if is_sdc_only:
                color = "#d32f2f" if i == 0 else "#bbbbbb"
                lw = 2.5 if i == 0 else 0.7
                base_alpha = 0.5 if i == 0 else 0.15
                ms = 7 if i == 0 else 3
            else:
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                lw = 1.5
                base_alpha = 0.35
                ms = 5
            xs_i = [float(frames[t]["x"][i]) for t in range(T)]
            ys_i = [float(frames[t]["y"][i]) for t in range(T)]
            # Main path with time-gradient alpha
            for t in range(T - 1):
                alpha = base_alpha + (0.9 - base_alpha) * t / max(T - 1, 1)
                ax.plot(xs_i[t:t+2], ys_i[t:t+2],
                        color=color, linewidth=lw, alpha=alpha, zorder=3 if not is_sdc_only or i != 0 else 5)
            # Start marker
            ax.plot(xs_i[0], ys_i[0], "^", color=color,
                    markersize=ms, alpha=0.8, zorder=5 if not is_sdc_only or i != 0 else 7,
                    markeredgecolor="black", markeredgewidth=0.3)
            # End marker
            ax.plot(xs_i[-1], ys_i[-1], "o", color=color,
                    markersize=ms, alpha=0.9, zorder=5 if not is_sdc_only or i != 0 else 7,
                    markeredgecolor="black", markeredgewidth=0.3)

        _setup_ax(ax, bounds, f"{z_label}")

    fig.suptitle(
        f"{run_name}  |  {scenario_label}  |  {mode_label}\n"
        f"\u25b3 = start, \u25cf = end, gray dashed = GT",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "trajectory_snapshot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: trajectory_snapshot.png")


# ─────────────────────────────────────────────────────────────────────────────
# MP4 video rendering (imageio)
# ─────────────────────────────────────────────────────────────────────────────

def _render_panel(ax, frame_states, t, n_w0, road_edges, gt, bounds, z_label, T,
                  is_sdc_only=False):
    """Render one panel (one z value) at timestep t onto ax."""
    ax.clear()
    ax.set_facecolor("#f8f8f8")
    _draw_road_edges(ax, road_edges)
    _draw_gt_paths(ax, gt, n_w0)
    _draw_agents_at_t(ax, frame_states, t, n_w0, trail_len=10,
                      is_sdc_only=is_sdc_only)
    _setup_ax(ax, bounds, f"{z_label}", t=t, T=T)


def render_mp4_comparison(
    all_frame_states, gt, n_w0, bounds, road_edges,
    z_values, run_name, scenario_label, mode_label, fps, out_dir,
    render_every=2, is_sdc_only=False,
):
    """
    Render a grid MP4 video showing all z values simultaneously.
    Max 3 columns, wraps into new rows. Output: zsweep_comparison.mp4
    """
    T = len(all_frame_states[0])
    n_panels = len(z_values)
    MAX_COLS = 3
    ncols = min(n_panels, MAX_COLS)
    nrows = math.ceil(n_panels / ncols)
    out_path = os.path.join(out_dir, "zsweep_comparison.mp4")

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), dpi=80)
    axes_flat = np.array(axes).flatten()
    # Hide any unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.patch.set_facecolor("#f0f0f0")

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=8)
    n_frames = 0
    for t in range(0, T, render_every):
        for panel_idx, (z_val, frame_states) in enumerate(zip(z_values, all_frame_states)):
            _render_panel(axes_flat[panel_idx], frame_states, t, n_w0,
                          road_edges, gt, bounds, z_val, T,
                          is_sdc_only=is_sdc_only)
        fig.suptitle(
            f"{run_name}  |  {scenario_label}  |  {mode_label}  ({n_w0} agents)",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()
        writer.append_data(frame)
        n_frames += 1
    writer.close()
    plt.close(fig)
    print(f"    Saved: zsweep_comparison.mp4  ({n_frames} frames @ {fps}fps)")


def render_individual_mp4(
    frame_states, gt, n_w0, bounds, road_edges,
    z_label, run_name, scenario_label, mode_label, fps, out_dir,
    render_every=2, is_sdc_only=False,
):
    """Render a single-panel MP4 for one z value."""
    T = len(frame_states)
    z_str = z_label.replace(".", "_").replace("(", "").replace(")", "").replace("+", "p").replace("-", "m").replace("=", "_")
    out_path = os.path.join(out_dir, f"z{z_str}.mp4")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    fig.patch.set_facecolor("#f0f0f0")

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=8)
    n_frames = 0
    for t in range(0, T, render_every):
        _render_panel(ax, frame_states, t, n_w0, road_edges, gt, bounds, z_label, T,
                      is_sdc_only=is_sdc_only)
        ax.set_title(f"{run_name}  |  {z_label}   t={t * 0.1:.1f}s",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()
        writer.append_data(frame)
        n_frames += 1
    writer.close()
    plt.close(fig)
    print(f"    Saved: z{z_str}.mp4  ({n_frames} frames @ {fps}fps)")


# ─────────────────────────────────────────────────────────────────────────────
# Quick metrics for summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def compute_quick_metrics(frame_states, rewards_trace, gt, n_w0, dt=0.1,
                           goal_dist_m=2.0, incident_thresh=-0.45,
                           sdc_only_index=None):
    """Compute goal rate, incident rate, mean speed, ADE for one rollout.
    If sdc_only_index is set, compute metrics only for that agent."""
    T  = len(frame_states)
    xs = np.stack([s["x"][:n_w0] for s in frame_states], axis=1)
    ys = np.stack([s["y"][:n_w0] for s in frame_states], axis=1)

    # If sdc_only, restrict to just the SDC
    if sdc_only_index is not None:
        xs = xs[sdc_only_index:sdc_only_index+1]
        ys = ys[sdc_only_index:sdc_only_index+1]
        n_w0 = 1

    # Speed
    dx = np.diff(xs, axis=1)
    dy = np.diff(ys, axis=1)
    speed = np.sqrt(dx**2 + dy**2) / dt
    mean_speed = float(np.nanmean(speed[speed < 80]))

    # Incident rate (reward-based)
    if rewards_trace:
        rew_arr = np.stack(rewards_trace, axis=1)
        if sdc_only_index is not None:
            rew_arr = rew_arr[sdc_only_index:sdc_only_index+1]
        incident_per_agent = (rew_arr < incident_thresh).any(axis=1)
        incident_rate = float(incident_per_agent.mean())
    else:
        incident_rate = float("nan")

    # ADE and goal rate
    gt_x = gt.get("x", None)
    gt_y = gt.get("y", None)
    gt_v = gt.get("valid", None)
    ades, goals = [], []
    if gt_x is not None:
        if sdc_only_index is not None:
            gx = gt_x[sdc_only_index:sdc_only_index+1]
            gy = gt_y[sdc_only_index:sdc_only_index+1]
            gv_raw = gt_v[sdc_only_index:sdc_only_index+1] if gt_v is not None else None
        else:
            gx = gt_x[:n_w0]
            gy = gt_y[:n_w0]
            gv_raw = gt_v[:n_w0] if gt_v is not None else None
        gx = gx.squeeze(1) if gx.ndim == 3 else gx
        gy = gy.squeeze(1) if gy.ndim == 3 else gy
        gv = gv_raw.squeeze(1).astype(bool) if gv_raw is not None else \
             np.ones_like(gx, dtype=bool)
        for i in range(n_w0):
            valid_idx = np.where(gv[i])[0]
            vi = valid_idx[valid_idx < T]
            if len(vi) == 0:
                continue
            dist = np.sqrt((xs[i, vi] - gx[i, vi])**2 + (ys[i, vi] - gy[i, vi])**2)
            ades.append(float(dist.mean()))
            # Goal: closest approach to GT endpoint
            last_gt = int(valid_idx[-1])
            active_xs = xs[i, gv[i, :T]]
            active_ys = ys[i, gv[i, :T]]
            if len(active_xs) > 0:
                min_d = float(np.sqrt(
                    (active_xs - gx[i, last_gt])**2 + (active_ys - gy[i, last_gt])**2
                ).min())
                goals.append(min_d < goal_dist_m)

    ade         = float(np.mean(ades))       if ades  else float("nan")
    goal_rate   = float(np.mean(goals))      if goals else float("nan")

    return {
        "mean_speed":    mean_speed,
        "incident_rate": incident_rate,
        "goal_rate":     goal_rate,
        "ade":           ade,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Z-sweep visualization for style-conditioned PufferDrive policies"
    )
    p.add_argument("--checkpoint",       required=True,
                   help="Path to .pt checkpoint file (relative to project dir)")
    p.add_argument("--run_name",         required=True,
                   help="Short project name, used in filenames")
    p.add_argument("--output_dir",       required=True,
                   help="Root output directory")
    p.add_argument("--split",            default="val",
                   choices=["train", "val"],
                   help="Which map split to use (default: val)")
    p.add_argument("--scenarios",        default="0,100,200,300,400,500,600,700,800,900",
                   help="Comma-separated map indices to visualize")
    p.add_argument("--sweep_mode", default="uniform", choices=["uniform", "pca", "axis"], help="Sweep mode for multi-dim z space")
    p.add_argument("--z_values",         default="-0.95,0.0,0.95",
                   help="Comma-separated z scalar values to sweep")
    p.add_argument("--modes",            default="init_goal,10step,sdc_only",
                   help="Comma-separated eval modes to run (default: init_goal,10step,sdc_only)")
    p.add_argument("--fps",               type=int, default=10,
                   help="Video FPS (default: 10 = real-time at dt=0.1)")
    p.add_argument("--episode_steps",     type=int, default=91,
                   help="Episode length in steps (default: 91)")
    p.add_argument("--dt",                type=float, default=0.1,
                   help="Simulation timestep in seconds (default: 0.1)")
    p.add_argument("--goal_radius",       type=float, default=4.0,
                   help="Goal achievement distance threshold in meters (default: 4.0)")
    p.add_argument("--individual_videos", action="store_true",
                   help="Also save individual per-z MP4s")
    p.add_argument("--render_every",      type=int, default=2,
                   help="Render every N-th step for videos (default: 2)")
    p.add_argument("--skip_videos",       action="store_true",
                   help="Skip MP4 video generation (only save PNG + CSV)")
    p.add_argument("--control_mode",      default=None,
                   choices=["control_vehicles", "control_wosac", "control_agents"],
                   help="Override control mode for all eval modes (default: use MODE_DEFS)")
    return p.parse_args()


# Mode definitions: (init_steps, control_mode, label_short, label_long)
MODE_DEFS = {
    "init_goal":  (0,  "control_wosac",   "init+goal",   "Init+Goal only (no waypoints)"),
    "10step":     (10, "control_wosac",    "10-step",     "10-step past then free drive"),
    "sdc_only":   (0,  "control_wosac",    "sdc-only",    "SDC-only z override (others z=0)"),
}


def run_mode_for_scenario(
    env_cls, map_file, arch, cfg, sd, device, ocean_torch, pufferlib,
    z_configs, mode_name, init_steps, control_mode, mode_label, mode_label_long,
    episode_steps, goal_radius, run_name, sc_label, out_dir, args,
):
    """Run all z values for one scenario in one mode. Returns summary rows."""
    env, tmpdir = build_env_for_scenario(
        env_cls, map_file, arch, cfg, episode_steps,
        init_steps=init_steps, goal_radius=goal_radius,
        control_mode=control_mode,
    )
    is_sdc_only = (mode_name == "sdc_only")
    summary_rows = []

    # Compute observation indices for z patching
    GUIDANCE_OBS_SIZE = 182   # 91 waypoints × 2 (x, y)
    if arch["style_z_dim"] > 0:
        if arch["arch_type"] == "causal":
            # Causal: z is computed internally by the GRU, not in observations.
            # Override is done via policy.set_style_override().
            z_obs_slice = (0, 0)
            print(f"      z injection: via set_style_override (not in obs)")
        elif arch["arch_type"] == "hetero":
            # Hetero: style score is in the ego_classic section, BEFORE guidance
            guidance_dim = GUIDANCE_OBS_SIZE if cfg["use_guidance_observations"] else 0
            z_start = arch["ego_features"] - guidance_dim - arch["style_z_dim"]
            z_obs_slice = (z_start, z_start + arch["style_z_dim"])
            print(f"      z_obs_indices: obs[:, {z_start}:{z_start + arch['style_z_dim']}]")
        else:
            # VAE: z is appended AFTER guidance, at end of ego section
            z_start = arch["ego_features"] - arch["style_z_dim"]
            z_obs_slice = (z_start, z_start + arch["style_z_dim"])
            print(f"      z_obs_indices: obs[:, {z_start}:{z_start + arch['style_z_dim']}]")
    else:
        z_obs_slice = (0, 0)

    try:
        policy = build_policy(env, arch, sd, device, ocean_torch, pufferlib)
        road_edges = try_get_road_edges(env)

        all_frame_states = []
        all_rewards      = []
        gt_ref           = None

        for z_cfg in z_configs:
            z_val = z_cfg["val"]
            z_label = z_cfg["label"]
            print(f"      z={z_label} ... ", end="", flush=True)
            frames, rewards, n_w0, gt = run_rollout_z(
                env, policy, device,
                z_val, arch, z_obs_slice,
                is_sdc_only=is_sdc_only,
            )
            if gt_ref is None:
                gt_ref = gt
            all_frame_states.append(frames)
            all_rewards.append(rewards)
            print(f"done ({n_w0} agents, {len(frames)} steps)")

        bounds = _compute_bounds(gt_ref if gt_ref else {}, n_w0, padding=20.0)

        # ── Trajectory snapshot PNG ──────────────────────────────────────
        render_trajectory_snapshot(
            all_frame_states, gt_ref, n_w0, bounds, road_edges,
            [c["label"] for c in z_configs], run_name, sc_label, mode_label_long, out_dir,
            is_sdc_only=is_sdc_only,
        )

        # ── MP4 videos ───────────────────────────────────────────
        if not args.skip_videos:
            render_mp4_comparison(
                all_frame_states, gt_ref, n_w0, bounds, road_edges,
                [c["label"] for c in z_configs], run_name, sc_label, mode_label_long,
                args.fps, out_dir, render_every=args.render_every,
                is_sdc_only=is_sdc_only,
            )
            if args.individual_videos:
                for z_cfg, frames in zip(z_configs, all_frame_states):
                    render_individual_mp4(
                        frames, gt_ref, n_w0, bounds, road_edges,
                        z_cfg["label"], run_name, sc_label, mode_label_long,
                        args.fps, out_dir, render_every=args.render_every,
                        is_sdc_only=is_sdc_only,
                    )

        # ── Quick metrics ────────────────────────────────────────────────
        for z_cfg, frames, rewards in zip(z_configs, all_frame_states, all_rewards):
            m = compute_quick_metrics(
                frames, rewards, gt_ref, n_w0, dt=args.dt,
                goal_dist_m=goal_radius,
                sdc_only_index=0 if is_sdc_only else None,
            )
            summary_rows.append([
                sc_label, mode_name, z_cfg["label"],
                f"{m['goal_rate']:.3f}", f"{m['incident_rate']:.3f}",
                f"{m['mean_speed']:.2f}", f"{m['ade']:.2f}",
            ])

    finally:
        env.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return summary_rows


def main():
    args = parse_args()

    project_dir = os.getcwd()
    sys.path.insert(0, project_dir)

    from pufferlib.ocean.drive.drive import Drive
    import pufferlib.ocean.torch as ocean_torch
    import pufferlib.models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint & infer arch ──────────────────────────────────────────
    sd   = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    arch = infer_arch(sd)
    cfg  = read_drive_config()

    # Override style_z_dim from config if available (handles vae_1d_ll0
    # which infer_arch misclassifies as hetero with z=0)
    if cfg["style_z_dim"] is not None and int(cfg["style_z_dim"]) != arch["style_z_dim"]:
        arch["style_z_dim"] = int(cfg["style_z_dim"])

    # Override arch_type: if infer_arch says "hetero" but config has style_z_dim > 0,
    # it's actually a 1D VAE (z appended after guidance at end of ego section),
    # NOT a hetero model (z at position 8 before guidance). Hetero projects have
    # no style_z_dim in their config.
    if arch["arch_type"] == "hetero" and cfg["style_z_dim"] is not None and int(cfg["style_z_dim"]) > 0:
        print(f"  [OVERRIDE] arch_type 'hetero' → 'vae' (config has style_z_dim={int(cfg['style_z_dim'])})")
        arch["arch_type"] = "vae"

    modes = [m.strip() for m in args.modes.split(",")]
    sweep_mode = getattr(args, "sweep_mode", "uniform")
    z_configs = get_z_configs(args.z_values, sweep_mode, arch["style_z_dim"])

    print(f"\n{'='*70}")
    print(f"Z-Sweep Visualization")
    print(f"  Run         : {args.run_name}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Device      : {device}")
    print(f"  Arch        : {arch['arch_type']}  (style_z_dim={arch['style_z_dim']})")
    print(f"  Modes       : {modes}")
    labels = [c["label"] for c in z_configs]
    print(f"  Z configs   : {labels}")
    print(f"  Goal radius : {args.goal_radius}m")
    print(f"{'='*70}")

    if arch["style_z_dim"] == 0:
        print(f"\n[SKIP] {args.run_name} has style_z_dim=0 — no z override possible.")
        return

    # ── Map files ─────────────────────────────────────────────────────────────
    map_subdir = {
        "train": "pufferlib/resources/drive/binaries/training",
        "val":   "pufferlib/resources/drive/binaries/validation",
    }[args.split]

    all_maps = sorted(
        [f for f in os.listdir(map_subdir) if f.endswith(".bin")],
        key=lambda f: int(f.replace("map_", "").replace(".bin", ""))
    )
    scenario_indices = [int(x) for x in args.scenarios.split(",")]

    # ── Output directory ──────────────────────────────────────────────────────
    run_out = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_out, exist_ok=True)
    print(f"  Output      : {run_out}")
    print(f"  Scenarios   : {scenario_indices}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_rows  = []
    csv_header    = ["scenario", "mode", "z_value", "goal_rate",
                     "incident_rate", "mean_speed", "ade"]

    # ── Per-scenario × per-mode loop ──────────────────────────────────────────
    for sc_idx, map_idx in enumerate(scenario_indices):
        if map_idx >= len(all_maps):
            print(f"  [SKIP] map_idx={map_idx} out of range ({len(all_maps)} maps)")
            continue

        map_file = os.path.join(map_subdir, all_maps[map_idx])
        sc_label = f"{args.split}_{all_maps[map_idx].replace('.bin','')}"

        print(f"\n{'─'*60}")
        print(f"  Scenario {sc_idx+1}/{len(scenario_indices)}: {sc_label}")

        for mode_name in modes:
            if mode_name not in MODE_DEFS:
                print(f"    [WARN] Unknown mode '{mode_name}', skipping")
                continue

            init_steps, cm, mode_label, mode_label_long = MODE_DEFS[mode_name]
            control_mode = args.control_mode if args.control_mode else cm
            mode_out_dir = os.path.join(run_out, f"scenario_{sc_label}", mode_name)
            os.makedirs(mode_out_dir, exist_ok=True)

            print(f"    [{mode_label}] init_steps={init_steps}, "
                  f"control={control_mode}, goal_radius={args.goal_radius}m")

            rows = run_mode_for_scenario(
                Drive, map_file, arch, cfg, sd, device, ocean_torch, pufferlib,
                z_configs, mode_name, init_steps, control_mode, mode_label, mode_label_long,
                args.episode_steps, args.goal_radius,
                args.run_name, sc_label, mode_out_dir, args,
            )
            summary_rows.extend(rows)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(run_out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)
        w.writerows(summary_rows)
    print(f"\nSaved summary → {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    if summary_rows:
        print(f"\n{'='*90}")
        print(f"Z-SWEEP SUMMARY: {args.run_name}  (goal_radius={args.goal_radius}m)")
        print(f"{'='*90}")
        hdr = (f"{'Scenario':<20}  {'Mode':<10}  {'z':>5}  "
               f"{'Goal':>6}  {'Inc':>6}  {'Speed':>7}  {'ADE':>7}")
        print(hdr)
        print("─" * len(hdr))
        for row in summary_rows:
            print(f"{row[0]:<20}  {row[1]:<10}  {row[2]:>5}  "
                  f"{row[3]:>6}  {row[4]:>6}  {row[5]:>7}  {row[6]:>7}")
        print("=" * 90)

    n_files = sum(
        len(os.listdir(os.path.join(dp, dn)))
        for dp, dns, _ in os.walk(run_out)
        for dn in dns
        if os.path.isdir(os.path.join(dp, dn))
    )
    print(f"\nAll done! {n_files} output files → {run_out}")


if __name__ == "__main__":
    main()
